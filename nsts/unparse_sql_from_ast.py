#coding=utf8
from typing import List
from nsts.asdl import ASDLGrammar
from nsts.asdl_ast import AbstractSyntaxTree
from nsts.value_processor import ValueProcessor, State
from functools import wraps


UNPARSER_DEBUG = True


WHERE_OPS_MAPPING = {
    'Equal': '=', 'NotEqual': '!=', 'GreaterThan': '>', 'GreaterEqual': '>=', 'LessThan': '<', 'LessEqual': '<=',
    'Like': 'like', 'NotLike': 'not like', 'In': 'in', 'NotIn': 'not in', 'Is': 'is'
}


UNIT_OPS_MAPPING = {
    'Minus': '-', 'Plus': '+', 'Times': '*', 'Divide': '/'
}


def ignore_error(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if UNPARSER_DEBUG: # allow error to be raised
            return func(*args, **kwargs)
        else: # prevent runtime error
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print('Something Error happened while unparsing:', e)
                table = args[2]['table_names_original'][0]
                return f'select * from {table}', False
    return wrapper


class ASTUnParser():

    def __init__(self, grammar: ASDLGrammar, value_processor: ValueProcessor):
        super(ASTUnParser, self).__init__()
        self.grammar, self.value_processor = grammar, value_processor


    @ignore_error
    def unparse(self, sql_ast: AbstractSyntaxTree, db: dict, entry: dict):
        if sql_ast is None:
            raise ValueError('root node is None')
        sql_query = self.unparse_sql(sql_ast, db, entry)
        return sql_query, True


    def unparse_sql(self, sql_ast: AbstractSyntaxTree, db: dict, entry: dict, **kargs):
        prod_name = sql_ast.production.constructor.name
        if prod_name != 'SQL':
            left_sql = self.unparse_sql(sql_ast[self.grammar.get_field_by_name('sql left_sql')][0].value, db, entry, **kargs)
            right_sql = self.unparse_sql(sql_ast[self.grammar.get_field_by_name('sql right_sql')][0].value, db, entry, **kargs)
            return '%s %s %s' % (left_sql, prod_name.lower(), right_sql)
        else:
            from_field = sql_ast[self.grammar.get_field_by_name('from from')][0]
            from_str, table_names = self.unparse_from(from_field.value, db, entry, **kargs)
            select_field = sql_ast[self.grammar.get_field_by_name('select select')][0]
            select_str = self.unparse_select(select_field.value, db, table_names=table_names, **kargs)
            where_field = sql_ast[self.grammar.get_field_by_name('condition where')][0]
            where_str = self.unparse_condition(where_field.value, db, entry, clause='where', add_prefix=True, **kargs)
            groupby_field = sql_ast[self.grammar.get_field_by_name('groupby groupby')][0]
            groupby_str = self.unparse_groupby(groupby_field.value, db, entry, **kargs)
            orderby_field = sql_ast[self.grammar.get_field_by_name('orderby orderby')][0]
            orderby_str = self.unparse_orderby(orderby_field.value, db, entry, **kargs)
            return ' '.join([select_str, from_str, where_str, groupby_str, orderby_str]).strip()


    def retrieve_column_name(self, col_id: int, db: dict, tab_name: str = None):
        if db['column_names_original'][col_id][0] < 0:
            return db['column_names_original'][col_id][1]
        tab_id, col_name = db['column_names_original'][col_id]
        if tab_name is None:
            col_name = db['table_names_original'][tab_id] + '.' + col_name
        else: col_name = tab_name + '.' + col_name # a.col_name, b.col_name
        return col_name


    def unparse_col_unit(self, col_unit_ast: AbstractSyntaxTree, db: dict, table_names: List[str] = [], **kargs):
        agg_op = col_unit_ast[self.grammar.get_field_by_name('agg_op agg_op')][0].value.production.constructor.name.lower()
        ctr_name = col_unit_ast.production.constructor.name
        if 'Binary' in ctr_name:
            col_id1 = col_unit_ast[self.grammar.get_field_by_name('col_id left_col_id')][0].value
            tab_name1 = table_names[0] if len(table_names) > 0 else None
            col_name1 = self.retrieve_column_name(col_id1, db, tab_name1)
            col_id2 = col_unit_ast[self.grammar.get_field_by_name('col_id right_col_id')][0].value
            tab_name2 = table_names[1] if len(table_names) > 1 else tab_name1
            col_name2 = self.retrieve_column_name(col_id2, db, tab_name2)
            unit_op = col_unit_ast[self.grammar.get_field_by_name('unit_op unit_op')][0].value.production.constructor.name
            unit_op = UNIT_OPS_MAPPING[unit_op]
            col_name = col_name1 + ' ' + unit_op + ' ' + col_name2
        else:
            unit_op = 'none'
            distinct = col_unit_ast[self.grammar.get_field_by_name('distinct distinct')][0].value.production.constructor.name
            distinct = 'distinct ' if distinct == 'True' else ''
            col_id1 = col_unit_ast[self.grammar.get_field_by_name('col_id col_id')][0].value
            tab_name = table_names[0] if len(table_names) > 0 else None
            col_name = distinct + self.retrieve_column_name(col_id1, db, tab_name)
        if agg_op == 'none': return col_name, (agg_op, unit_op, col_id1)
        else: return agg_op.lower() + ' ( ' + col_name + ' ) ', (agg_op, unit_op, col_id1)


    def unparse_select(self, select_ast: AbstractSyntaxTree, db: dict, table_names: List[str] = [], **kargs):
        distinct = select_ast[self.grammar.get_field_by_name('distinct distinct')][0].value.production.constructor.name
        distinct = 'distinct ' if distinct == 'True' else ''
        select_fields = select_ast[self.grammar.get_field_by_name('col_unit col_unit')]
        select_items = []
        for col_unit_field in select_fields:
            col_unit_str = self.unparse_col_unit(col_unit_field.value, db, table_names, **kargs)[0]
            select_items.append(col_unit_str)
        return 'select ' + distinct + ' , '.join(select_items)


    def unparse_from(self, from_ast: AbstractSyntaxTree, db: dict, entry: dict, **kargs):
        ctr_name = from_ast.production.constructor.name
        if 'Table' in ctr_name:
            table_names = []
            table_fields = from_ast[self.grammar.get_field_by_name('tab_id tab_id')]
            for tab_field in table_fields:
                table_name = db['table_names_original'][int(tab_field.value)]
                table_names.append(table_name)
            cond_field = from_ast[self.grammar.get_field_by_name('condition from')][0]
            cond_str = self.unparse_condition(cond_field.value, db, entry, clause='from', add_prefix=True, **kargs)
            return 'from ' + ' join '.join(table_names) + cond_str, []
        elif 'Two' not in ctr_name:
            return 'FROM ( ' + self.unparse_sql(from_ast[self.grammar.get_field_by_name('sql from_sql')][0].value, db, entry, **kargs) + ' )', []
        else: # two SQLs in from clause (benchmark DuSQL)
            return 'FROM ( ' + self.unparse_sql(from_ast[self.grammar.get_field_by_name('sql from_sql')][0].value, db, entry, **kargs) + ' ) a , ( ' + \
                self.unparse_sql(from_ast[self.grammar.get_field_by_name('sql from_sql')][1].value, db, entry, **kargs) + ' ) b', ['a', 'b']


    def unparse_groupby(self, groupby_ast: AbstractSyntaxTree, db: dict, entry: dict, **kargs):
        ctr_name = groupby_ast.production.constructor.name
        if ctr_name == 'NoGroupBy': return ''
        groupby_str = []
        groupby_fields = groupby_ast[self.grammar.get_field_by_name('col_id col_id')]
        for col_field in groupby_fields:
            col_name = self.retrieve_column_name(col_field.value, db)
            groupby_str.append(col_name)
        groupby_str = 'GROUP BY ' + ' , '.join(groupby_str)
        having_field = groupby_ast[self.grammar.get_field_by_name('condition having')][0]
        having_str = self.unparse_condition(having_field.value, db, entry, clause='having', add_prefix=True, **kargs)
        return groupby_str + having_str


    def unparse_orderby(self, orderby_ast: AbstractSyntaxTree, db: dict, entry: dict, **kargs):
        ctr_name = orderby_ast.production.constructor.name
        if ctr_name == 'NoOrderBy': return ''
        col_names = []
        col_fields = orderby_ast[self.grammar.get_field_by_name('col_unit col_unit')]
        for col_field in col_fields:
            col_name = self.unparse_col_unit(col_field.value, db, **kargs)[0]
            col_names.append(col_name)
        orderby_str = ' , '.join(col_names)
        order = orderby_ast[self.grammar.get_field_by_name('order order')][0].value.production.constructor.name.lower()

        limit_str = ''
        if 'Limit' in ctr_name:
            limit_field = orderby_ast[self.grammar.get_field_by_name('val_id limit')][0]
            limit_val = self.value_processor.postprocess(limit_field.value, State('limit', 'none', '=', 'none', 0), db, entry)
            limit_str = ' limit ' + limit_val
        return 'order by ' + orderby_str + ' ' + order + limit_str


    def unparse_condition(self, condition_ast: AbstractSyntaxTree, db: dict, entry: dict, clause: str = '', add_prefix=True, **kargs):
        ctr_name = condition_ast.production.constructor.name
        if ctr_name == 'NoCondition': return ''
        elif 'AndCondition' in ctr_name or 'OrCondition' in ctr_name:
            cond_fields = condition_ast[self.grammar.get_field_by_name('condition condition')]
            conds = []
            for cond_field in cond_fields:
                cond = self.unparse_condition(cond_field.value, db, entry, clause=clause, add_prefix=False, **kargs)
                conds.append(cond)
            conj = ' and ' if 'AndCondition' in ctr_name else ' or '
            cond_str =  conj.join(conds)
        else: cond_str = self.unparse_cond_unit(condition_ast, db, entry, clause=clause, **kargs)
        # take care of the prefix and whitespace
        if add_prefix and clause == 'from': return ' on ' + cond_str
        elif add_prefix and clause == 'where': return 'where ' + cond_str
        elif add_prefix and clause == 'having': return ' having ' + cond_str
        else: return cond_str


    def unparse_cond_unit(self, cond_ast: AbstractSyntaxTree, db: dict, entry: dict, clause: str = 'where', **kargs):
        col_field = cond_ast[self.grammar.get_field_by_name('col_unit col_unit')][0]
        col_name, (agg_op, unit_op, col_id) = self.unparse_col_unit(col_field.value, db, **kargs)
        ctr_name = cond_ast.production.constructor.name
        if 'Between' in ctr_name:
            cmp_op = 'between'
            state = State(clause, agg_op, cmp_op, unit_op, col_id)
            val_field = cond_ast[self.grammar.get_field_by_name('value left_value')][0]
            val_str1 = self.unparse_value(val_field.value, state, db, entry, **kargs)
            val_field = cond_ast[self.grammar.get_field_by_name('value right_value')][0]
            val_str2 = self.unparse_value(val_field.value, state, db, entry, **kargs)
            val_str = val_str1 + ' and ' + val_str2
        else:
            cmp_op = cond_ast[self.grammar.get_field_by_name('cmp_op cmp_op')][0].value.production.constructor.name
            cmp_op = WHERE_OPS_MAPPING[cmp_op]
            state = State(clause, agg_op, cmp_op, unit_op, col_id)
            val_field = cond_ast[self.grammar.get_field_by_name('value value')][0]
            val_str = self.unparse_value(val_field.value, state, db, entry, **kargs)
        return ' '.join([col_name, cmp_op, val_str])


    def unparse_value(self, val_ast: AbstractSyntaxTree, state: State, db: dict, entry: dict, **kargs):
        ctr_name = val_ast.production.constructor.name
        if ctr_name == 'LiteralValue':
            val_field = val_ast[self.grammar.get_field_by_name('val_id val_id')][0]
            val_str = self.value_processor.postprocess(val_field.value, state, db, entry)
        elif ctr_name == 'SQLValue':
            val_field = val_ast[self.grammar.get_field_by_name('sql value_sql')][0]
            val_str = '( ' + self.unparse_sql(val_field.value, db, entry, **kargs) + ' )'
        else: # ColumnValue
            col_id = int(val_ast[self.grammar.get_field_by_name('col_id col_id')][0].value)
            val_str = self.retrieve_column_name(col_id, db)
        return val_str