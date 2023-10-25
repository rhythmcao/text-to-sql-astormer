#coding=utf8
import json, sys, os, re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nsts.parse_sql_to_json import AGG_OPS, UNIT_OPS, WHERE_OPS
from nsts.value_processor import is_number, is_int


class JsonUnparser():

    def __init__(self, keyword_uppercase: bool = False, ignore_table_number: bool = False) -> None:
        super(JsonUnparser, self).__init__()
        self.keyword_uppercase = keyword_uppercase
        self.ignore_table_number = ignore_table_number


    def unparse(self, sql: dict, db: dict):
        for conj in ['intersect', 'union', 'except']:
            nested_sql = sql[conj]
            if nested_sql:
                conj = ' ' + (conj.upper() if self.keyword_uppercase else conj) + ' '
                query = self.unparse_sql(sql, db) + conj + self.unparse(nested_sql, db)
                break
        else: query = self.unparse_sql(sql, db)
        return re.sub(r'\s+', ' ', query.strip())


    def unparse_sql(self, sql: dict, db: dict):
        from_str, from_tables = self.unparse_from(sql['from'], db)
        from_str = ('FROM ' if self.keyword_uppercase else 'from ') + from_str
        select_str = ('SELECT ' if self.keyword_uppercase else 'select ') + self.unparse_select(sql['select'], db, from_tables)
        where_str = ('WHERE ' if self.keyword_uppercase else 'where ') + self.unparse_conds(sql['where'], db, 'where') if sql['where'] else ''
        groupby_str = ('GROUP BY ' if self.keyword_uppercase else 'group by ') + self.unparse_groupby(sql['groupBy'], db) if sql['groupBy'] else ''
        having_str = ('HAVING ' if self.keyword_uppercase else 'having ') + self.unparse_conds(sql['having'], db, 'having') if sql['having'] else ''
        orderby_str = ('ORDER BY ' if self.keyword_uppercase else 'order by ') + self.unparse_orderby(sql['orderBy'], db) if sql['orderBy'] else ''
        limit_str = ('LIMIT ' if self.keyword_uppercase else 'limit ') + str(sql['limit']) if sql['limit'] else ''
        return ' '.join([select_str, from_str, where_str, groupby_str, having_str, orderby_str, limit_str])


    def unparse_from(self, from_clause, db):
        table_units = from_clause['table_units']
        from_tables = []
        if table_units[0][0] == 'sql':
            if len(table_units) == 1:
                return '( ' + self.unparse(table_units[0][1], db) + ' )', from_tables
            else:
                return '( ' + self.unparse(table_units[0][1], db) + ' ) a , ( ' + self.unparse(table_units[1][1], db) + ' ) b', ['a', 'b']
        else:
            if type(table_units[0][1]) != int and self.ignore_table_number:
                return 'TABLE_OR_TABLEJOINS', from_tables
            table_names = db['table_names_original']
            table_list = [table_unit[1] if type(table_unit[1]) != int else table_names[int(table_unit[1])] for table_unit in table_units]
            if len(table_list) > 1:
                cond_str = self.unparse_conds(from_clause['conds'], db, 'from')
                cond_str = (' ON ' if self.keyword_uppercase else ' on ') + cond_str if cond_str.strip() else ''
                conj = ' JOIN ' if self.keyword_uppercase else ' join '
                return conj.join(table_list) + cond_str, from_tables
            else: return table_list[0], from_tables


    def unparse_distinct(self, distinct_flag):
        if type(distinct_flag) == bool:
            if distinct_flag:
                distinct_str = 'DISTINCT ' if self.keyword_uppercase else 'distinct '
            else: distinct_str = ''
        else: distinct_str = distinct_flag + ' '
        return distinct_str


    def unparse_select(self, select, db, from_tables=[]):
        if len(from_tables) > 0: # dusql from_left + from_right
            assert len(select[1]) == 1 and len(from_tables) == 2
        select_str = [self.unparse_col_unit(self.convert_to_column_unit((agg, val_unit)), db, from_tables) for agg, val_unit in select[1]]
        distinct_str = self.unparse_distinct(select[0])
        return distinct_str + ' , '.join(select_str)


    def retrieve_column_name(self, cid, db, tab_name = None):
        if type(cid) != int:
            col_name = cid
        else:
            tab_name = '' if db['column_names_original'][cid][0] < 0 else tab_name + '.' if tab_name is not None else db['table_names_original'][db['column_names_original'][cid][0]] + '.'
            col_name = tab_name + db['column_names_original'][cid][1]
        return col_name


    def unparse_groupby(self, groupby, db):
        return ' , '.join([self.retrieve_column_name(col_unit[1], db) for col_unit in groupby])


    def unparse_orderby(self, orderby, db):
        orderby_cols = ' , '.join([self.unparse_col_unit(self.convert_to_column_unit((agg, val_unit)), db) for agg, val_unit in orderby[1]])
        suffix = (orderby[0].upper() if self.keyword_uppercase else orderby[0].lower()) if orderby[0].lower() in ['asc', 'desc'] else orderby[0]
        return orderby_cols + ' ' + suffix


    def unparse_conds(self, conds: list, db: dict, clause: str = 'where'):
        if not conds: return ''
        cond_str = [self.unparse_cond(cond, db, clause) if cond not in ['and', 'or'] else 
                    cond.upper() if self.keyword_uppercase else cond.lower() for cond in conds]
        return ' '.join(cond_str)


    def unparse_cond(self, cond: list, db: dict, clause: str = 'where'):
        agg_id, cmp_id, val_unit, val1, val2 = cond
        val_str = self.unparse_col_unit(self.convert_to_column_unit((agg_id, val_unit)), db, clause=clause)
        val1_str = self.unparse_val(val1, db, clause)
        if type(cmp_id) == int:
            cmp_op = WHERE_OPS[cmp_id].upper() if self.keyword_uppercase else WHERE_OPS[cmp_id].lower()
        else: cmp_op = cmp_id
        if val2 is not None:
            val2_str = self.unparse_val(val2, db, clause)
            return val_str + f' {cmp_op} ' + val1_str + (' AND ' if self.keyword_uppercase else ' and ') + val2_str
        return val_str + ' ' + cmp_op + ' ' + val1_str


    def unparse_val(self, val, db, clause='where'):
        if type(val) in [str, int, float]:
            val = str(val).strip('"\'')
            if is_int(val):
                if (not val.startswith('0')) or val.startswith('0.') or val == '0':
                    val = int(float(val))
            elif is_number(val):
                val = float(val)
            val_str = '"' + str(val) + '"' if type(val) == str else str(val)
        elif type(val) in [list, tuple]:
            val_str = self.retrieve_column_name(val[1], db)
        else:
            assert type(val) == dict
            val_str = '( ' + self.unparse(val, db) + ' )'
        return val_str


    def convert_to_column_unit(self, agg_val_unit):
        if len(agg_val_unit) == 2: # agg, val_unit
            agg, val_unit = agg_val_unit
            if val_unit[0] == 0: # unary op: agg, unit, distinct, col
                column_unit = [agg, 0, val_unit[1][2], val_unit[1][1]]
            else: # binary op: agg, unit, col1, col2
                column_unit = [agg, val_unit[0], val_unit[1][1], val_unit[2][1]]
        else: # val_unit
            unit, col_unit1, col_unit2 = agg_val_unit
            if unit == 0:
                column_unit = [col_unit1[0], 0, col_unit1[2], col_unit1[1]]
            else:
                column_unit = [0, unit, col_unit1[1], col_unit2[1]]
        return column_unit


    def unparse_col_unit(self, col_unit, db, from_tables: list = [], clause: str = 'unknown'):
        agg_op = '' if col_unit[0] == 0 or clause in ['from', 'where'] else (AGG_OPS[int(col_unit[0])].upper() if self.keyword_uppercase else AGG_OPS[int(col_unit[0])].lower()) if type(col_unit[0]) == int else col_unit[0]
        if col_unit[1] == 0: # unary op
            distinct_str = self.unparse_distinct(col_unit[2]) if clause not in ['from', 'where'] else ''
            col_name = self.retrieve_column_name(col_unit[3], db)
            col_str = f'{distinct_str}{col_name}' if agg_op == '' else f'{agg_op} ( {distinct_str}{col_name} )'
        else: # binary op
            tname1, tname2 = (None, None) if len(from_tables) == 0 else (from_tables[0], from_tables[1])
            cname1 = self.retrieve_column_name(col_unit[2], db, tname1)
            cname2 = self.retrieve_column_name(col_unit[3], db, tname2)
            unit_op = UNIT_OPS[col_unit[1]] if type(col_unit[1]) == int else col_unit[1]
            col_name = cname1 + ' ' + unit_op + ' ' + cname2
            col_str = col_name if agg_op == '' else f'{agg_op} ( {col_name} )'
        return col_str


def recover_dataset(dataset_path, gold_path, pred_path, table_path):
    unparser = JsonUnparser()
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    with open(table_path, 'r') as f:
        tables = {db['db_id']: db for db in json.load(f)}

    gold_sqls, output_sqls = [], []
    for ex in dataset:
        if 'interaction' in ex: # contextual text-to-SQL
            gold_turn_sqls, turn_sqls = [], []
            for turn in ex['interaction']:
                query = ' '.join(' '.join(turn['query'].split('\t')).split('\n'))
                s = unparser.unparse(turn['sql'], tables[ex['database_id']])
                turn_sqls.append(s)
                gold_turn_sqls.append(query + '\t' + ex['database_id'])
            output_sqls.append(turn_sqls)
            gold_sqls.append(gold_turn_sqls)
        else:
            s = unparser.unparse(ex['sql'], tables[ex['db_id']])
            query = ' '.join(' '.join(ex['query'].split('\t')).split('\n'))
            if 'question_id' in ex and type(ex['question_id']) == str: # special care about DuSQL
                output_sqls.append(ex['question_id'] + '\t' + s)
                gold_sqls.append(ex['question_id'] + '\t' + query + '\t' + ex['db_id'])
            else:
                output_sqls.append(s)
                gold_sqls.append(query + '\t' + ex['db_id'])

    with open(gold_path, 'w') as of:
        for l in gold_sqls:
            if type(l) == list:
                for s in l: of.write(s + '\n')
                of.write('\n')
            else:
                of.write(l + '\n')
        print(f'Write golden SQLs into file: {gold_path}')

    with open(pred_path, 'w') as of:
        for l in output_sqls:
            if type(l) == list:
                for s in l: of.write(s + '\n')
                of.write('\n')
            else:
                of.write(l + '\n')
        print(f'Write recovered SQLs into file: {pred_path}')
    return


if __name__ == '__main__':

    import argparse
    from nsts.transition_system import CONFIG_PATHS

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='dataset', default='spider', choices=['spider', 'sparc', 'cosql', 'dusql', 'chase'], help='dataset name')
    parser.add_argument('-s', dest='data_split', default='train', choices=['train', 'dev'], help='dataset split')
    args = parser.parse_args(sys.argv[1:])

    table_path = CONFIG_PATHS[args.dataset]['tables']
    dataset_path = CONFIG_PATHS[args.dataset][args.data_split]
    pred_path = os.path.join(os.path.dirname(dataset_path), args.data_split + '_json.sql')
    gold_path = os.path.join(os.path.dirname(dataset_path), args.data_split + '_gold.sql')
    recover_dataset(dataset_path, gold_path, pred_path, table_path)

    if args.dataset == 'dusql':
        from eval.evaluation_dusql import evaluate_dusql, build_foreign_key_map_from_json
        kmaps = build_foreign_key_map_from_json(CONFIG_PATHS[args.dataset]['tables'])
        evaluate_dusql(gold_path, pred_path, table_path, kmaps)
    elif args.dataset == 'chase':
        from eval.evaluation_chase import evaluate_chase, build_foreign_key_map_from_json
        kmaps = build_foreign_key_map_from_json(CONFIG_PATHS[args.dataset]['tables'])
        evaluate_chase(gold_path, pred_path, table_path, kmaps)
    else:
        from eval.evaluation import evaluate, build_foreign_key_map_from_json
        kmaps = build_foreign_key_map_from_json(CONFIG_PATHS[args.dataset]['tables'])
        evaluate(gold_path, pred_path, CONFIG_PATHS[args.dataset]['db_dir'], 'all', kmaps, False, False, False)