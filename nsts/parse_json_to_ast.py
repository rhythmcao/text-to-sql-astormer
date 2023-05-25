#coding=utf8
from functools import wraps
from nsts.parse_sql_to_json import WHERE_OPS
from nsts.asdl import ASDLConstructor, ASDLGrammar
from nsts.asdl_ast import AbstractSyntaxTree
from nsts.value_processor import ValueProcessor


PARSER_DEBUG = False # if DEBUG, allow ERROR raised to traceback


AGG_OPS_NAME = ('None', 'Max', 'Min', 'Count', 'Sum', 'Avg')
UNIT_OPS_NAME = ('', 'Minus', 'Plus', 'Times', 'Divide')
WHERE_OPS_NAME = {
    '=': 'Equal', '>': 'GreaterThan', '<': 'LessThan', '>=': 'GreaterEqual', '<=': 'LessEqual',
    '!=': 'NotEqual', 'in': 'In', 'not in': 'NotIn', 'like': 'Like', 'not like': 'NotLike', 'is': 'Is'
}


def ignore_error(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if PARSER_DEBUG: # allow error to be raised
            return func(self, *args, **kwargs)
        else: # prevent runtime error
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                print('Something Error happened while parsing:', e)
                # if fail to parse, just return SELECT * FROM table(id=0)
                error_sql = {
                    "select": [False, [(0, [0, [0, 0, False], None])]],
                    "from": {'table_units': [('table_unit', 0)], 'conds': []},
                    "where": [], "groupBy": [], "orderBy": [], "having": [], "limit": None,
                    "intersect": [], "union": [], "except": []
                }
                root_node = self.parse_sql(error_sql)
                return root_node
    return wrapper


class ASTParser():
    """ Parse a SQL json object into AbstractSyntaxTree object according to pre-defined grammar rules.
    The structure of the json dict exactly follows the convention of benchmark Spider.
    """
    def __init__(self, grammar: ASDLGrammar, value_processor: ValueProcessor):
        super(ASTParser, self).__init__()
        self.grammar, self.value_processor = grammar, value_processor


    @ignore_error
    def parse(self, sql: dict):
        """ Entrance function.
        @params:
            sql: the 'sql' field of each data sample
        @return:
            ast_node: root node of the SQL AbstractSyntaxTree
        """
        root_node = self.parse_sql(sql)
        return root_node


    def parse_sql(self, sql: dict):
        """ Determine whether sql has intersect/union/except nested sql, and parse different sub-clauses.
        """
        for choice in ['intersect', 'union', 'except']:
            if sql[choice]:
                ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_name(choice.title()))
                nested_sql = sql[choice]
                sql_field1 = ast_node[self.grammar.get_field_by_name('sql left_sql')][0]
                sql_field1.add_value(self.parse_sql_unit(sql))
                sql_field2 = ast_node[self.grammar.get_field_by_name('sql right_sql')][0]
                sql_field2.add_value(self.parse_sql(nested_sql))
                return ast_node
        return self.parse_sql_unit(sql)


    def parse_sql_unit(self, sql: dict):
        # only a single SQL
        ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_name('SQL'))
        from_field = ast_node[self.grammar.get_field_by_name('from from')][0]
        from_field.add_value(self.parse_from(sql['from']))
        select_field = ast_node[self.grammar.get_field_by_name('select select')][0]
        select_field.add_value(self.parse_select(sql['select']))
        where_field = ast_node[self.grammar.get_field_by_name('condition where')][0]
        where_field.add_value(self.parse_condition(sql['where']))
        groupby_field = ast_node[self.grammar.get_field_by_name('groupby groupby')][0]
        groupby_field.add_value(self.parse_groupby(sql['groupBy'], sql['having']))
        orderby_field = ast_node[self.grammar.get_field_by_name('orderby orderby')][0]
        orderby_field.add_value(self.parse_orderby(sql['orderBy'], sql['limit']))
        return ast_node


    def convert_val_unit(self, val_unit):
        if val_unit[0] == 0:
            return [val_unit[1][0], val_unit[0], val_unit[1][2], val_unit[1][1]]
        return [val_unit[1][0], val_unit[0], val_unit[1][1], val_unit[2][1]]


    def convert_agg_val_unit(self, agg, val_unit):
        if val_unit[0] != 0: # (agg_op, unit_op, col_id1, col_id2)
            return [agg, val_unit[0], val_unit[1][1], val_unit[2][1]]
        else: # (agg_op, unit_op, distinct, col_id)
            return [agg, val_unit[0], val_unit[1][2], val_unit[1][1]]


    def parse_col_unit(self, col_unit):
        if col_unit[1] != 0: # unit_op is not none, BinaryColumnUnit
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_name('BinaryColumnUnit'))
            agg_op_node = AbstractSyntaxTree(self.grammar.get_prod_by_name(AGG_OPS_NAME[col_unit[0]]))
            ast_node[self.grammar.get_field_by_name('agg_op agg_op')][0].add_value(agg_op_node)
            unit_op_node = AbstractSyntaxTree(self.grammar.get_prod_by_name(UNIT_OPS_NAME[col_unit[1]]))
            ast_node[self.grammar.get_field_by_name('unit_op unit_op')][0].add_value(unit_op_node)
            ast_node[self.grammar.get_field_by_name('col_id left_col_id')][0].add_value(int(col_unit[2]))
            ast_node[self.grammar.get_field_by_name('col_id right_col_id')][0].add_value(int(col_unit[3]))
        else:
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_name('UnaryColumnUnit'))
            agg_op_node = AbstractSyntaxTree(self.grammar.get_prod_by_name(AGG_OPS_NAME[col_unit[0]]))
            ast_node[self.grammar.get_field_by_name('agg_op agg_op')][0].add_value(agg_op_node)
            distinct_node = AbstractSyntaxTree(self.grammar.get_prod_by_name(str(col_unit[2])))
            ast_node[self.grammar.get_field_by_name('distinct distinct')][0].add_value(distinct_node)
            ast_node[self.grammar.get_field_by_name('col_id col_id')][0].add_value(int(col_unit[3]))
        return ast_node


    def parse_from(self, from_clause: dict):
        """ Ignore from conditions, since it is not evaluated in evaluation script
        """
        table_units, from_conds = from_clause['table_units'], from_clause['conds']
        t = table_units[0][0]
        if t == 'table_unit':
            ctr_name = 'FromTable' + ASDLConstructor.number2word[len(table_units)]
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_name(ctr_name))
            table_fields = ast_node[self.grammar.get_field_by_name('tab_id tab_id')]
            for idx, (_, tab_id) in enumerate(table_units):
                table_fields[idx].add_value(int(tab_id))
            cond_field = ast_node[self.grammar.get_field_by_name('condition from')][0]
            cond_field.add_value(self.parse_condition(from_conds))
        else:
            assert t == 'sql'
            if len(table_units) == 1:
                from_sql = table_units[0][1]
                ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_name('FromSQL'))
                ast_node[self.grammar.get_field_by_name('sql from_sql')][0].add_value(self.parse_sql(from_sql))
            else: # in DuSQL, two SQLs in from clause
                left_sql, right_sql = table_units[0][1], table_units[1][1]
                ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_name('FromSQLTwo'))
                ast_node[self.grammar.get_field_by_name('sql from_sql')][0].add_value(self.parse_sql(left_sql))
                ast_node[self.grammar.get_field_by_name('sql from_sql')][1].add_value(self.parse_sql(right_sql))
        return ast_node


    def parse_select(self, select_clause: list):
        distinct_flag, select_columns = select_clause[0], select_clause[1]
        ctr_name = 'SelectColumn' + ASDLConstructor.number2word[len(select_columns)]
        ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_name(ctr_name))
        distinct_field = ast_node[self.grammar.get_field_by_name('distinct distinct')][0]
        distinct_field.add_value(AbstractSyntaxTree(self.grammar.get_prod_by_name(str(distinct_flag))))
        column_fields = ast_node[self.grammar.get_field_by_name('col_unit col_unit')]
        for idx, (agg, val_unit) in enumerate(select_columns):
            col_unit = self.convert_agg_val_unit(agg, val_unit)
            col_node = self.parse_col_unit(col_unit)
            column_fields[idx].add_value(col_node)
        return ast_node


    def parse_groupby(self, groupby_clause: list, having_clause: list):
        if groupby_clause:
            ctr_name = 'GroupByColumn' + ASDLConstructor.number2word[len(groupby_clause)]
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_name(ctr_name))
            groupby_fields = ast_node[self.grammar.get_field_by_name('col_id col_id')]
            for idx, col_unit in enumerate(groupby_clause):
                groupby_fields[idx].add_value(int(col_unit[1]))
            having_field = ast_node[self.grammar.get_field_by_name('condition having')][0]
            having_field.add_value(self.parse_condition(having_clause))
        else: ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_name('NoGroupBy'))
        return ast_node


    def parse_orderby(self, orderby_clause: list, limit: int):
        if orderby_clause:
            ctr_name = 'OrderByLimitColumn' + ASDLConstructor.number2word[len(orderby_clause[1])] if limit else \
                'OrderByColumn' + ASDLConstructor.number2word[len(orderby_clause[1])]
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_name(ctr_name))
            col_fields = ast_node[self.grammar.get_field_by_name('col_unit col_unit')]
            for idx, (agg, val_unit) in enumerate(orderby_clause[1]):
                col_unit = self.convert_agg_val_unit(agg, val_unit)
                col_node = self.parse_col_unit(col_unit)
                col_fields[idx].add_value(col_node)
            order_node = AbstractSyntaxTree(self.grammar.get_prod_by_name(orderby_clause[0].title()))
            ast_node[self.grammar.get_field_by_name('order order')][0].add_value(order_node)
            if limit:
                ids = self.value_processor.preprocess(str(limit))
                ast_node[self.grammar.get_field_by_name('val_id limit')][0].add_value(ids)
        else: ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_name('NoOrderBy'))
        return ast_node


    def parse_condition(self, condition: list):
        if len(condition) == 0:
            return AbstractSyntaxTree(self.grammar.get_prod_by_name('NoCondition'))

        def parse_condition_list(condition_list, conjunction='and'):
            # try parse the condition list recursively
            max_condition_num = 4
            if len(condition_list) > max_condition_num:
                ctr_name = conjunction.title() + 'Condition' + ASDLConstructor.number2word[max_condition_num]
                ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_name(ctr_name))
                fields = ast_node[self.grammar.get_field_by_name('condition condition')]
                # directly parse the first max_condition_num - 1 conditions
                for idx, cond_unit in enumerate(condition_list[:max_condition_num - 1]):
                    fields[idx].add_value(self.parse_cond_unit(cond_unit))
                 # parse the other conditions recursively
                fields[max_condition_num - 1].add_value(parse_condition_list(condition_list[max_condition_num - 1:], conjunction=conjunction))
            elif len(condition_list) == 1: # directly parse the single condition
                ast_node = self.parse_cond_unit(condition_list[0])
            else: # directly parse each cond_unit and assign the parsed node to each individual field
                ctr_name = conjunction.title() + 'Condition' + ASDLConstructor.number2word[len(condition_list)]
                ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_name(ctr_name))
                fields = ast_node[self.grammar.get_field_by_name('condition condition')]
                for idx, cond_unit in enumerate(condition_list):
                    fields[idx].add_value(self.parse_cond_unit(cond_unit))
            return ast_node

        conjs = set([cond for cond in condition if cond in ['and', 'or']])
        if len(conjs) == 2: # contain both AND and OR, create an extra AndConditionTwo node on top of them
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_name('AndConditionTwo'))
            condition_fields = ast_node[self.grammar.get_field_by_name('condition condition')]
            and_condition_list, or_condition_list = [], []
            conj, cond_unit = 'and', condition[0]
            for cur_cond in condition[1:]:
                if cur_cond in ['and', 'or']:
                    conj = cur_cond
                    if conj == 'and': and_condition_list.append(cond_unit)
                    else: or_condition_list.append(cond_unit)
                else: cond_unit = cur_cond
            if conj == 'and': and_condition_list.append(cond_unit)
            else: or_condition_list.append(cond_unit)
            assert len(or_condition_list) != 1
            condition_fields[0].add_value(parse_condition_list(and_condition_list, conjunction='and'))
            condition_fields[1].add_value(parse_condition_list(or_condition_list, conjunction='or'))
        else: # AND condition / OR condition
            condition_list = [cond for cond in condition if cond not in ['and', 'or']]
            conjunction = conjs.pop() if len(conjs) == 1 else 'and'
            ast_node = parse_condition_list(condition_list, conjunction=conjunction)
        return ast_node


    def parse_cond_unit(self, cond_unit: list):
        agg_id, cmp_op, val_unit = cond_unit[0], WHERE_OPS[cond_unit[1]], cond_unit[2]
        ctr_name = 'CmpCondition' if cmp_op != 'between' else 'BetweenCondition'
        ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_name(ctr_name))
        col_node = self.parse_col_unit(self.convert_agg_val_unit(agg_id, val_unit))
        ast_node[self.grammar.get_field_by_name('col_unit col_unit')][0].add_value(col_node)
        val1, val2 = cond_unit[3], cond_unit[4]
        if cmp_op == 'between':
            left_value_node = self.parse_value(val1)
            ast_node[self.grammar.get_field_by_name('value left_value')][0].add_value(left_value_node)
            right_value_node = self.parse_value(val2)
            ast_node[self.grammar.get_field_by_name('value right_value')][0].add_value(right_value_node)
        else:
            ctr_name = WHERE_OPS_NAME[cmp_op]
            cmp_node = AbstractSyntaxTree(self.grammar.get_prod_by_name(ctr_name))
            ast_node[self.grammar.get_field_by_name('cmp_op cmp_op')][0].add_value(cmp_node)
            value_node = self.parse_value(val1)
            ast_node[self.grammar.get_field_by_name('value value')][0].add_value(value_node)
        return ast_node


    def parse_value(self, val):
        if type(val) == dict: # nested sql
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_name('SQLValue'))
            ast_node[self.grammar.get_field_by_name('sql value_sql')][0].add_value(self.parse_sql(val))
        elif type(val) in [list, tuple]: # column
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_name('ColumnValue'))
            ast_node[self.grammar.get_field_by_name('col_id col_id')][0].add_value(int(val[1]))
        else: # literal value
            ast_node = AbstractSyntaxTree(self.grammar.get_prod_by_name('LiteralValue'))
            value_field = ast_node[self.grammar.get_field_by_name('val_id val_id')][0]
            ids = self.value_processor.preprocess(val)
            value_field.add_value(ids)
        return ast_node