################################
# Assumptions:
#   1. sql is correct
#   2. only table name has alias
#   3. only one intersect/union/except
#
# val: number(float)/string(str)/sql(dict)/list(column)
# col_unit: (agg_id, col_id, isDistinct(bool))
# val_unit: (unit_op, col_unit1, col_unit2)
# table_unit: (table_type, col_unit/sql)
# cond_unit: (agg_id, op_id, val_unit, val1, val2)
# condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
# sql {
#   'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
#   'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
#   'where': condition
#   'groupBy': [col_unit1, col_unit2, ...]
#   'orderBy': ('asc'/'desc', [(agg_id, val_unit1), （agg_id, val_unit2), ...])
#   'having': condition
#   'limit': None/limit value
#   'intersect': None/sql
#   'except': None/sql
#   'union': None/sql
# }
################################

import json, sys, os, re
from nltk import word_tokenize

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'not in', 'not like')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')


class SchemaID():
    """ Schema of one database which maps table&column to a unique identifier (int)
    """
    def __init__(self, table: dict):
        self._table = {"table_names_original": table["table_names_original"], "column_names_original": table["column_names_original"]}
        self._schema = self._table2columns(self._table)
        self._idMap = self._schema2id(self._table)


    @property
    def schema(self):
        return self._schema


    @property
    def idMap(self):
        return self._idMap


    @property
    def table(self):
        return self._table['table_names_original']


    def _table2columns(self, table):
        column_names_original = table['column_names_original']
        table_names_original = table['table_names_original']

        schema = {}
        for i, tabn in enumerate(table_names_original):
            schema[str(tabn.lower())] = [str(col.lower()) for td, col in column_names_original if td == i]
        return schema


    def _schema2id(self, table):
        column_names_original = table['column_names_original']
        table_names_original = table['table_names_original']

        idMap = {'*': 0} # the first column should be wildcard *
        for i, (tab_id, col) in enumerate(column_names_original):
            if i == 0: continue
            if tab_id < 0 and col.lower() == 'time_now':
                idMap['time_now'] = i
                continue
            idMap[table_names_original[tab_id].lower() + "." + col.lower()] = i

        for i, tab in enumerate(table_names_original):
            idMap[tab.lower()] = i

        return idMap


def tokenize(string):
    string = str(string)
    string = string.replace("\'", "\"")  # ensures all string values wrapped by "" problem??
    quote_idxs = [idx for idx, char in enumerate(string) if char == '"']
    assert len(quote_idxs) % 2 == 0, "Unexpected quote"

    # keep string value as token
    vals = {}
    for i in range(len(quote_idxs)-1, -1, -2):
        qidx1 = quote_idxs[i-1]
        qidx2 = quote_idxs[i]
        val = string[qidx1: qidx2+1]
        key = "__val_{}_{}__".format(qidx1, qidx2)
        string = string[:qidx1] + key + string[qidx2+1:]
        vals[key] = val

    toks = [word.lower() for word in word_tokenize(string)]
    # replace with string value token
    for i in range(len(toks)):
        if toks[i] in vals:
            toks[i] = vals[toks[i]]

    # find if there exists !=, >=, <=
    eq_idxs = [idx for idx, tok in enumerate(toks) if tok == "="]
    eq_idxs.reverse()
    prefix = ('!', '>', '<')
    for eq_idx in eq_idxs:
        pre_tok = toks[eq_idx-1]
        if pre_tok in prefix:
            toks = toks[:eq_idx-1] + [pre_tok + "="] + toks[eq_idx+1: ]

    return toks


def scan_table_alias(toks):
    """Scan the index of 'as' and build the map for all alias"""
    as_idxs = [idx for idx, tok in enumerate(toks) if tok == 'as']
    alias = {}
    for idx in as_idxs:
        alias[toks[idx+1]] = toks[idx-1] # overwritten and mismatch problem in nested sqls
    return alias


def exist_table_alias_contradiction(toks):
    as_idxs = [idx for idx, tok in enumerate(toks) if tok == 'as']
    alias = {}
    for idx in as_idxs:
        a = toks[idx+1]
        if a in alias and alias[a] != toks[idx-1]:
            return True
        alias[a] = toks[idx-1]
    return False


def toks2nested(toks):
    """
        Determine the scope for each sub-sql
        mapping [select, count, (, c1, ), from, (, select c1, c2, from, t, ), ... ] into
        [select, count, (, c1, ), from, [select, c1, c2, from, t], ... ]
    """
    def detect_sql(idx):
        count, sql_list = 0, []
        while idx < len(toks):
            if toks[idx] == '(':
                count += 1
                if toks[idx + 1] == 'select':
                    sub_sql_list, idx = detect_sql(idx + 1)
                    count -= 1
                    sql_list.append(sub_sql_list)
                else:
                    sql_list.append('(')
            elif toks[idx] == ')':
                count -= 1
                if count < 0:
                    return sql_list, idx
                else:
                    sql_list.append(')')
            else:
                sql_list.append(toks[idx])
            idx += 1
        return sql_list, idx

    def intersect_union_except(tok_list):
        for idx, tok in enumerate(tok_list):
            if type(tok) == list:
                new_tok = intersect_union_except(tok)
                tok_list[idx] = new_tok
        for op in ['intersect', 'union', 'except']:
            if op in tok_list:
                idx = tok_list.index(op)
                tok_list = [tok_list[:idx]] + [op] + [tok_list[idx+1:]]
                break
        return tok_list

    try:
        nested_toks, _ = detect_sql(0)
        # not all sqls are wrapped with (), e.g. sql1 intersect sql2
        # add wrapper for each sql on the left and right handside of intersect/union/except
        nested_toks = intersect_union_except(nested_toks)
        return nested_toks
    except:
        print('Something unknown happened when transforming %s' % (' '.join(toks)))
        return None


def reassign_table_alias(nested_toks, index):
    current_map = {} # map old alias in the current sql to new alias in global map
    as_idxs = [idx for idx, tok in enumerate(nested_toks) if tok == 'as']
    for idx in as_idxs:
        index += 1 # add 1 to global index for table alias before assignment
        assert nested_toks[idx+1] not in current_map
        current_map[nested_toks[idx+1]] = 't' + str(index)
        nested_toks[idx+1] = 't' + str(index)
    for j, tok in enumerate(nested_toks):
        if type(tok) == list:
            new_tok, index = reassign_table_alias(tok, index)
            nested_toks[j] = new_tok
        elif '.' in tok:
            for alias in current_map.keys():
                if tok.startswith(alias + '.'):
                    nested_toks[j] = current_map[alias] + '.' + tok[tok.index('.')+1:]
                    break
    return nested_toks, index


def normalize_table_alias(toks):
    """ Make sure that different table alias are assigned to different tables """
    if toks.count('select') == 1:
        return toks # no nested sql, don't worry
    elif toks.count('select') > 1:
        if exist_table_alias_contradiction(toks): # avoid unnecessary normalization process
            nested_toks = toks2nested(toks)
            index = 0 # global index for table alias
            nested_toks, _ = reassign_table_alias(nested_toks, index)
            def flatten(x):
                if type(x) == list and ('intersect' in x or 'union' in x or 'except' in x):
                    assert len(x) == 3 # sql1 union sql2
                    return ['('] + flatten(x[0])[1:-1] + [x[1]] + flatten(x[2])[1:-1] + [')']
                elif type(x) == list:
                    return ['('] + [y for l in x for y in flatten(l)] + [')']
                else:
                    return [x]
            toks = flatten(nested_toks)[1:-1]
        return toks
    else:
        raise ValueError('Something wrong in sql because no select clause is found!')


def get_tables_with_alias(schema, toks):
    toks = normalize_table_alias(toks)
    tables = scan_table_alias(toks)
    for key in schema:
        assert key not in tables, "Alias {} has the same name in table".format(key)
        tables[key] = key
    return tables, toks


def parse_col(toks, start_idx, tables_with_alias, schema, default_tables=None):
    """
        :returns next idx, column id
    """
        

    tok = toks[start_idx]
    if tok in ["*", "time_now"]:
        return start_idx + 1, schema.idMap[tok]

    if start_idx + 1 < len(toks) and toks[start_idx + 1] == '(': # special case in DuSQL, column 'GDP总计(亿)'
        assert toks[start_idx + 3] == ')', 'Error col: {}'.format(tok)
        tok += f'({toks[start_idx + 2]})'
        start_idx += 3

    if '.' in tok:  # if token is a composite
        alias, col = tok.split('.')
        key = tables_with_alias[alias] + "." + col
        return start_idx + 1, schema.idMap[key]

    assert default_tables is not None and len(default_tables) > 0, "Default tables should not be None or empty"

    for alias in default_tables:
        table = tables_with_alias[alias]
        if tok in schema.schema[table]:
            key = table + "." + tok
            return start_idx + 1, schema.idMap[key]

    assert False, "Error col: {}".format(tok)


def parse_col_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):
    """
        :returns next idx, (agg_op id, col_id)
    """
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    isDistinct = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    if toks[idx] in AGG_OPS:
        agg_id = AGG_OPS.index(toks[idx])
        idx += 1
        assert idx < len_ and toks[idx] == '('
        idx += 1
        if toks[idx] == "distinct":
            idx += 1
            isDistinct = True
        idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)
        assert idx < len_ and toks[idx] == ')'
        idx += 1
        return idx, (agg_id, col_id, isDistinct)

    if toks[idx] == "distinct":
        idx += 1
        isDistinct = True
    agg_id = AGG_OPS.index("none")
    idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)

    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'

    return idx, (agg_id, col_id, isDistinct)


def parse_val_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    col_unit1 = None
    col_unit2 = None
    unit_op = UNIT_OPS.index('none')

    idx, col_unit1 = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
    if idx < len_ and toks[idx] in UNIT_OPS:
        unit_op = UNIT_OPS.index(toks[idx])
        idx += 1
        idx, col_unit2 = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)

    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'

    return idx, (unit_op, col_unit1, col_unit2)


def parse_table_unit(toks, start_idx, tables_with_alias, schema):
    """
        :returns next idx, table id, table name
    """
    idx = start_idx
    len_ = len(toks)
    key = tables_with_alias[toks[idx]]

    if idx + 1 < len_ and toks[idx+1] == "as":
        idx += 3
    else:
        idx += 1

    return idx, schema.idMap[key], key


def parse_value(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)

    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    if toks[idx] == 'select':
        idx, val = parse_sql(toks, idx, tables_with_alias, schema)
    elif "\"" in toks[idx]: # token is a string value
        val = toks[idx]
        idx += 1
    elif re.search(r'^[\d:\-~]+$', toks[idx].strip()): # date string
        val = '"' + toks[idx].strip() + '"'
        idx += 1
    else:
        try:
            float(toks[idx])
            val = toks[idx]
            idx += 1
        except:
            end_idx = idx
            while end_idx < len_ and toks[end_idx] != ',' and toks[end_idx] != ')'\
                and toks[end_idx] != 'and' and toks[end_idx] not in CLAUSE_KEYWORDS and toks[end_idx] not in JOIN_KEYWORDS:
                    end_idx += 1

            idx, val = parse_col_unit(toks[start_idx: end_idx], 0, tables_with_alias, schema, default_tables)
            idx = end_idx

    if isBlock:
        assert toks[idx] == ')'
        idx += 1

    return idx, val


def parse_condition(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)
    conds = []
    is_block = 0

    while idx < len_:
        while idx < len_ and toks[idx] == '(':
            is_block += 1
            idx += 1

        agg_id = AGG_OPS.index('none')
        if idx < len_ and toks[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(toks[idx])
            idx += 1

        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        not_op = False
        if toks[idx] == 'not':
            not_op = True
            idx += 1

        assert idx < len_ and toks[idx] in WHERE_OPS, "Error condition: idx: {}, tok: {}".format(idx, toks[idx])
        cmp_op = 'not ' + toks[idx] if not_op else toks[idx]
        op_id = WHERE_OPS.index(cmp_op)
        idx += 1
        val1 = val2 = None
        if 'between' in cmp_op:  # between..and... special case: dual values
            idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
            assert toks[idx] == 'and'
            idx += 1
            idx, val2 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
        else:  # normal case: single value
            idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
            val2 = None

        conds.append((agg_id, op_id, val_unit, val1, val2))

        while is_block > 0 and idx < len_ and toks[idx] == ")":
            is_block -= 1
            idx += 1

        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";") or toks[idx] in JOIN_KEYWORDS):
            break

        if idx < len_ and toks[idx] in COND_OPS:
            conds.append(toks[idx])
            idx += 1  # skip and/or

    return idx, conds


def parse_select(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)

    assert toks[idx] == 'select', "'select' not found"
    idx += 1
    isDistinct = False
    if idx < len_ and toks[idx] == 'distinct':
        idx += 1
        isDistinct = True
    val_units = []

    while idx < len_ and toks[idx] not in CLAUSE_KEYWORDS:
        agg_id = AGG_OPS.index("none")
        if toks[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(toks[idx])
            idx += 1
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        val_units.append((agg_id, val_unit))
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','

    return idx, (isDistinct, val_units)


def parse_from(toks, start_idx, tables_with_alias, schema):
    """
    Assume in the from clause, all table units are combined with join
    """
    assert 'from' in toks[start_idx:], "'from' not found"

    len_ = len(toks)
    idx = toks.index('from', start_idx) + 1
    default_tables = []
    table_units = []
    conds = []
    last_table = None

    while idx < len_:
        isBlock = False
        if toks[idx] == '(':
            isBlock = True
            idx += 1

        if toks[idx] == 'select':
            idx, sql = parse_sql(toks, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE['sql'], sql))
            last_table = schema.table[sql['from']['table_units'][0][1]].lower()
        else:
            if idx < len_ and toks[idx] == 'join':
                idx += 1  # skip join
            idx, table_unit, table_name = parse_table_unit(toks, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE['table_unit'],table_unit))
            default_tables.append(table_name)
        if idx < len_ and toks[idx] == "on":
            idx += 1  # skip on
            idx, this_conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
            if len(conds) > 0:
                conds.append('and')
            conds.extend(this_conds)

        if isBlock:
            assert toks[idx] == ')'
            idx += 1

        if idx < len_ and toks[idx] == 'a':
            assert last_table is not None and last_table in schema.idMap, 'last_table should be a table name string'
            tables_with_alias['a'] = last_table
            idx += 2
        elif idx < len_ and toks[idx] == 'b':
            assert last_table is not None and last_table in schema.idMap, 'last_table should be a table name string'
            tables_with_alias['b'] = last_table
            idx += 1

        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
            break

    return idx, table_units, conds, default_tables


def parse_where(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'where':
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
    return idx, conds


def parse_group_by(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    col_units = []

    if idx >= len_ or toks[idx] != 'group':
        return idx, col_units

    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, col_unit = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
        col_units.append(col_unit)
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, col_units


def parse_order_by(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    val_units = []
    order_type = 'asc' # default type is 'asc'

    if idx >= len_ or toks[idx] != 'order':
        return idx, val_units

    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        agg_id = AGG_OPS.index("none")
        if toks[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(toks[idx])
            idx += 1

        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        val_units.append((agg_id, val_unit))
        if idx < len_ and toks[idx] in ORDER_OPS:
            order_type = toks[idx]
            idx += 1
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, (order_type, val_units)


def parse_having(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'having':
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
    return idx, conds


def parse_limit(toks, start_idx):
    idx = start_idx
    len_ = len(toks)

    if idx < len_ and toks[idx] == 'limit':
        idx += 2
        return idx, int(toks[idx-1])

    return idx, None


def parse_sql(toks, start_idx, tables_with_alias, schema):
    isBlock = False # indicate whether this is a block of sql/sub-sql
    len_ = len(toks)
    idx = start_idx

    sql = {}
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    # parse from clause in order to get default tables
    from_end_idx, table_units, conds, default_tables = parse_from(toks, start_idx, tables_with_alias, schema)
    sql['from'] = {'table_units': table_units, 'conds': conds}
    # select clause
    _, select_col_units = parse_select(toks, idx, tables_with_alias, schema, default_tables)
    idx = from_end_idx
    sql['select'] = select_col_units
    # where clause
    idx, where_conds = parse_where(toks, idx, tables_with_alias, schema, default_tables)
    sql['where'] = where_conds
    # group by clause
    idx, group_col_units = parse_group_by(toks, idx, tables_with_alias, schema, default_tables)
    sql['groupBy'] = group_col_units
    # having clause
    idx, having_conds = parse_having(toks, idx, tables_with_alias, schema, default_tables)
    sql['having'] = having_conds
    # order by clause
    idx, order_col_units = parse_order_by(toks, idx, tables_with_alias, schema, default_tables)
    sql['orderBy'] = order_col_units
    # limit clause
    idx, limit_val = parse_limit(toks, idx)
    sql['limit'] = limit_val

    idx = skip_semicolon(toks, idx)
    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'
    idx = skip_semicolon(toks, idx)

    # intersect/union/except clause
    for op in SQL_OPS:  # initialize IUE
        sql[op] = None
    if idx < len_ and toks[idx] in SQL_OPS:
        sql_op = toks[idx]
        idx += 1
        idx, IUE_sql = parse_sql(toks, idx, tables_with_alias, schema)
        sql[sql_op] = IUE_sql
    return idx, sql


def get_sql(query, table):
    toks = tokenize(query)
    schema = SchemaID(table)
    tables_with_alias, toks = get_tables_with_alias(schema.schema, toks)
    _, sql = parse_sql(toks, 0, tables_with_alias, schema)
    return sql


def skip_semicolon(toks, start_idx):
    idx = start_idx
    while idx < len(toks) and toks[idx] == ";":
        idx += 1
    return idx


class JsonParser():

    def parse(self, sql: str, table: dict):
        return get_sql(sql, table)


def parse_dataset(input_path, table_path):
    with open(input_path, 'r') as f:
        dataset = json.load(f)
    with open(table_path, 'r') as f:
        tables = {db['db_id']: db for db in json.load(f)}

    parser = JsonParser()
    for ex in dataset:
        if 'interaction' in ex: # conversational text-to-SQL

            # error fixing during training
            if 'WHERE WHERE' in ex['final']['query']:
                ex['final']['query'] = ex['final']['query'].replace('WHERE WHERE', 'WHERE')
            elif 'AND (T1.allergy  =  "Milk" OR T1.allergy  =  "Eggs")' in ex['final']['query']:
                ex['final']['query'] = ex['final']['query'].replace('AND (T1.allergy  =  "Milk" OR T1.allergy  =  "Eggs")', 'AND T1.allergy  =  "Milk" OR T1.allergy  =  "Eggs"')
            elif 'S: What IS the largest college?' in ex['final']['query']:
                ex['final']['query'] = ex['final']['query'].replace('S: What IS the largest college?', '')
            elif 'from 国家 AS T1 JOIN 大洲 AS T2 ON T1.所属洲id = T2.洲id where 洲名 = "北美洲" limit' in ex['final']['query']:
                ex['final']['query'] = ex['final']['query'].replace('limit 1', '')
            elif 'and 诗词名 = "声声慢·寻寻觅觅"' in ex['final']['query']:
                ex['final']['query'] = ex['final']['query'].replace('and 诗词名 =', 'where 诗词名 =')
            elif 'ON T2.公司id = T3.公司id and t3.名称 = "京东"' in ex['final']['query']:
                ex['final']['query'] = ex['final']['query'].replace('and t3.名称 =', 'where t3.名称 =')

            ex['final']['sql'] = parser.parse(ex['final']['query'], tables[ex['database_id']])

            for turn in ex['interaction']:

                # error fixing during training
                if re.search(r'SELECT\s+FROM', turn['query']):
                    turn['query'] = re.sub(r'SELECT\s+FROM', 'SELECT * FROM', turn['query'])
                elif 'LIMIT 2,1' in turn['query']:
                    turn['query'] = 'SELECT name  FROM manufacturers ORDER BY revenue asc LIMIT 3'
                elif 'SELECT breed_code, count(*) FROM Dogs GROUP BY breed_code limit' in turn['query']:
                    turn['query'] = turn['query'].replace('GROUP BY breed_code', 'GROUP BY breed_code ORDER BY count(*) DESC')
                elif 'HAVING count ( * )   <  200 limit 3' in turn['query']:
                    turn['query'] = turn['query'].replace('limit 3', 'ORDER BY count(*) limit 3')
                elif 'countryid = 3 )' in turn['query'] and turn['query'].count('(') == 0:
                    turn['query'] = turn['query'].replace('countryid = 3 )', 'countryid = 3')
                elif '=  ""' in turn['query']:
                    turn['query'] = turn['query'].replace('=  ""', '=  "null"')
                elif 'from 国家 AS T1 JOIN 大洲 AS T2 ON T1.所属洲id = T2.洲id where 洲名 = "北美洲" limit' in turn['query']:
                    turn['query'] = turn['query'].replace('limit 1', '')
                elif 'select 品种代码 , count(*) from 狗 group by 品种代码 limit' in turn['query']:
                    turn['query'] = turn['query'].replace('limit', 'order by count(*) desc limit')
                elif 'and 诗词名 = "声声慢·寻寻觅觅"' in turn['query']:
                    turn['query'] = turn['query'].replace('and 诗词名 =', 'where 诗词名 =')
                elif 'ON T2.公司id = T3.公司id and t3.名称 = "京东"' in turn['query']:
                    turn['query'] = turn['query'].replace('and t3.名称 =', 'where t3.名称 =')
                turn['sql'] = parser.parse(turn['query'], tables[ex['database_id']])
        else:

            # error fixing during training
            if 'Ref_Company_Types' in ex['query']:
                ex['question'] = 'What is the type of the company who concluded its contracts most recently ?'
                ex['question_toks'] = ex['question'].split(' ')
                ex['query'] = 'SELECT T1.company_type FROM Third_Party_Companies AS T1 JOIN Maintenance_Contracts AS T2 ON T1.company_id  =  T2.maintenance_contract_company_id ORDER BY T2.contract_end_date DESC LIMIT 1'
            elif 'WHERE T2.maxOccupancy  =  T1.Adults + T1.Kids' in ex['query']:
                ex['query'] = ex['query'].replace('T2.maxOccupancy  =  T1.Adults + T1.Kids', 'T1.Adults + T1.Kids = T2.maxOccupancy')
            elif 'T1.融资轮次 from 企业融资 as T1 join 企业 as T2 join 企业 as T3 on 企业融资.企业id = 企业.词条id and 投资公司.企业id = 企业.词条id order by T1.融资总额' in ex['query']:
                ex['query'] = ex['query'].replace('企业 as T3 on', '投资公司 as T3 on').replace('T3.中文名', 'T2.中文名')
            elif 'T2.actid  =  T2.actid' in ex['query']:
                ex['query'] = ex['query'].replace('T2.actid  =  T2.actid', 'T2.actid  =  T3.actid') if 'AS T3' in ex['query'] else ex['query'].replace('T2.actid  =  T2.actid', 'T1.actid  =  T2.actid')
            elif 'SELECT T2.dormid FROM dorm AS T3' in ex['query']:
                ex['query'] = ex['query'].replace('SELECT T2.dormid FROM dorm AS T3', 'SELECT T3.dormid FROM dorm AS T3')

            ex['sql'] = parser.parse(ex['query'], tables[ex['db_id']])

    with open(input_path, 'w') as of:
        json.dump(dataset, of, ensure_ascii=False, sort_keys=True, indent=4, separators=(',', ': '))
        print(f'Parsed dataset is serialized into file: {input_path}')


if __name__ == '__main__':

    import argparse
    from transition_system import CONFIG_PATHS

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='dataset', default='spider', choices=['spider', 'sparc', 'cosql', 'dusql', 'chase'], help='filepath to the dataset')
    parser.add_argument('-s', dest='data_split', default='train', choices=['train', 'dev', 'all', 'dev_ext'], help='dataset split')
    args = parser.parse_args(sys.argv[1:])

    table_path = CONFIG_PATHS[args.dataset]['tables']
    data_split = ['train', 'dev'] if args.data_split == 'all' else [args.data_split] if args.data_split != 'dev_ext' else ['dev_syn', 'dev_dk', 'dev_realistic']
    for split in data_split:
        input_path = CONFIG_PATHS[args.dataset][split]
        if os.path.exists(input_path):
            parse_dataset(input_path, table_path)
        else:
            print('[WARNING]: The dataset file does not exist: {}'.format(input_path))
