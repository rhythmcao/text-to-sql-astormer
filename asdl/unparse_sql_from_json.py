#coding=utf8
import json, sys, os, re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asdl.parse_sql_to_json import AGG_OPS, UNIT_OPS, WHERE_OPS
from asdl.value_processor import is_number


class JsonUnparser():

    def unparse(self, sql: dict, db: dict):
        if sql['intersect']:
            query = self.unparse_sql(sql, db) + ' intersect ' + self.unparse(sql['intersect'], db)
        elif sql['union']:
            query = self.unparse_sql(sql, db) + ' union ' + self.unparse(sql['union'], db)
        elif sql['except']:
            query = self.unparse_sql(sql, db) + ' except ' + self.unparse(sql['except'], db)
        else: query = self.unparse_sql(sql, db)
        return re.sub(r'\s+', ' ', query).strip()


    def unparse_sql(self, sql: dict, db: dict):
        from_str = 'from ' + self.unparse_from(sql['from'], db)
        select_str = 'select ' + self.unparse_select(sql['select'], db)
        where_str = 'where ' + self.unparse_conds(sql['where'], db) if sql['where'] else ''
        groupby_str = 'group by ' + self.unparse_groupby(sql['groupBy'], db) if sql['groupBy'] else ''
        having_str = 'having ' + self.unparse_conds(sql['having'], db) if sql['having'] else ''
        orderby_str = 'order by ' + self.unparse_orderby(sql['orderBy'], db) if sql['orderBy'] else ''
        limit_str = 'limit ' + str(sql['limit']) if sql['limit'] else ''
        return ' '.join([select_str, from_str, where_str, groupby_str, having_str, orderby_str, limit_str])


    def unparse_select(self, select, db):
        select_str = []
        for agg_id, val_unit in select[1]:
            col_name = self.unparse_val_unit(val_unit, db)
            if agg_id == 0: select_str.append(col_name)
            else: select_str.append(AGG_OPS[agg_id].lower() + f' ( {col_name} )')
        dis = 'distinct ' if select[0] else ''
        return dis + ' , '.join(select_str)


    def unparse_from(self, from_clause, db):
        table_units = from_clause['table_units']
        if table_units[0][0] == 'sql':
            return '( ' + self.unparse(table_units[0][1], db) + ' )'
        else:
            table_names = db['table_names_original']
            table_list = [table_names[int(table_unit[1])] for table_unit in table_units]
            if len(table_list) > 1:
                cond_str = self.unparse_conds(from_clause['conds'], db)
                cond_str = ' on ' + cond_str if cond_str.strip() else ''
                return ' join '.join(table_list) + cond_str
            else: return table_list[0]


    def unparse_groupby(self, groupby, db):
        return ' , '.join([self.unparse_col_unit(col_unit, db) for col_unit in groupby])


    def unparse_orderby(self, orderby, db):
        return ' , '.join([self.unparse_val_unit(val_unit, db) for val_unit in orderby[1]]) + ' ' + orderby[0].lower()


    def unparse_conds(self, conds: list, db: dict):
        if not conds: return ''
        cond_str = [self.unparse_cond(cond, db) if cond not in ['and', 'or'] else cond.lower() for cond in conds]
        return ' '.join(cond_str)


    def unparse_cond(self, cond: list, db: dict):
        not_op, cmp_op, val_unit, val1, val2 = cond
        val_str = self.unparse_val_unit(val_unit, db)
        val1_str = self.unparse_val(val1, db)
        if not_op:
            assert cmp_op in [8, 9]
            not_str = 'not '
        else: not_str = ''
        if cmp_op == 1:
            val2_str = self.unparse_val(val2, db)
            return val_str + ' between ' + val1_str + ' and ' + val2_str
        cmp_str = WHERE_OPS[cmp_op].lower()
        return val_str + ' ' + not_str + cmp_str + ' ' + val1_str


    def unparse_val(self, val, db):
        if type(val) in [str, int, float]:
            if is_number(val):
                val_str = str(int(val)) if float(val) % 1 == 0 else str(float(val))
            else:
                val = val.strip('"') # remove quotes
                val_str = str(val)
            val_str = '"' + val_str + '"' if type(val) == str else val_str
        elif type(val) in [list, tuple]:
            val_str = self.unparse_col_unit(val, db)
        else:
            assert type(val) == dict
            val_str = '( ' + self.unparse(val, db) + ' )'
        return val_str


    def unparse_val_unit(self, val_unit: list, db: dict):
        unit_op, col_unit1, col_unit2 = val_unit
        if unit_op == 0:
            return self.unparse_col_unit(col_unit1, db)
        else:
            unit_join = ' ' + UNIT_OPS[unit_op].lower() + ' '
            return unit_join.join([self.unparse_col_unit(col_unit1, db), self.unparse_col_unit(col_unit2, db)])


    def unparse_col_unit(self, col_unit: list, db: dict):
        agg_id, col_id, dis = col_unit
        tab_name = '' if col_id == 0 else db['table_names_original'][db['column_names_original'][col_id][0]] + '.'
        col_name = tab_name + db['column_names_original'][col_id][1]
        col_name = 'distinct ' + col_name if dis else col_name
        if agg_id == 0:
            return col_name
        return AGG_OPS[agg_id].lower() + f' ( {col_name} )'


def recover_dataset(dataset_path, output_path, table_path):
    unparser = JsonUnparser()
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    with open(table_path, 'r') as f:
        tables = {db['db_id']: db for db in json.load(f)}

    output_sqls = []
    for ex in dataset:
        if 'interaction' in ex: # contextual text-to-SQL
            turn_sqls = []
            for turn in ex:
                s = unparser.unparse(turn['sql'], tables[ex['database_id']])
                turn_sqls.append(s)
            output_sqls.append(turn_sqls)
        else:
            s = unparser.unparse(ex['sql'], tables[ex['db_id']])
            output_sqls.append(s)

    with open(output_path, 'w') as of:
        for l in output_sqls:
            if type(l) == list:
                for s in l: of.write(s + '\n')
                of.write('\n')
            else: of.write(l + '\n')
        print(f'Write recovered SQLs into file: {output_path}')
    return


if __name__ == '__main__':

    import argparse
    from asdl.transition_system import CONFIG_PATHS
    from eval.evaluation import evaluate, build_foreign_key_map_from_json

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='dataset', default='spider', choices=['spider', 'sparc', 'cosql'], help='dataset name')
    parser.add_argument('-s', dest='data_split', default='train', choices=['train', 'dev'], help='dataset split')
    parser.add_argument('-o', dest='output_path', help='filepath to the outputpath, if not specified, re-use dataset_path')
    args = parser.parse_args(sys.argv[1:])

    table_path = CONFIG_PATHS[args.dataset]['tables']
    dataset_path = CONFIG_PATHS[args.dataset][args.data_split]
    pred_path = os.path.join(os.path.dirname(dataset_path), args.data_split + '_ast.sql') if args.output_path is None else args.output_path
    recover_dataset(dataset_path, pred_path, table_path)

    gold_path = os.path.join(os.path.dirname(dataset_path), args.data_split + '_gold.sql')
    kmaps = build_foreign_key_map_from_json(CONFIG_PATHS[args.dataset]['tables'])
    evaluate(gold_path, pred_path, CONFIG_PATHS[args.dataset]['db_dir'], 'all', kmaps, False, False, False)