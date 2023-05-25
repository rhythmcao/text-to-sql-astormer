#coding=utf8
import json, sys, os, re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nsts.parse_sql_to_json import AGG_OPS, UNIT_OPS, WHERE_OPS
from nsts.value_processor import is_number, is_int


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
        from_str, from_tables = self.unparse_from(sql['from'], db)
        from_str = 'from ' + from_str
        select_str = 'select ' + self.unparse_select(sql['select'], db, from_tables)
        where_str = 'where ' + self.unparse_conds(sql['where'], db, 'where') if sql['where'] else ''
        groupby_str = 'group by ' + self.unparse_groupby(sql['groupBy'], db) if sql['groupBy'] else ''
        having_str = 'having ' + self.unparse_conds(sql['having'], db, 'having') if sql['having'] else ''
        orderby_str = 'order by ' + self.unparse_orderby(sql['orderBy'], db) if sql['orderBy'] else ''
        limit_str = 'limit ' + str(sql['limit']) if sql['limit'] else ''
        return ' '.join([select_str, from_str, where_str, groupby_str, having_str, orderby_str, limit_str])


    def unparse_select(self, select, db, from_tables=[]):
        select_str = []
        for agg_id, val_unit in select[1]:
            col_name = self.unparse_val_unit(val_unit, db, from_tables)
            if agg_id == 0: select_str.append(col_name)
            else: select_str.append(AGG_OPS[agg_id].lower() + f' ( {col_name} )')
        dis = 'distinct ' if select[0] else ''
        return dis + ' , '.join(select_str)


    def unparse_from(self, from_clause, db):
        table_units = from_clause['table_units']
        from_tables = []
        if table_units[0][0] == 'sql':
            if len(table_units) == 1:
                return '( ' + self.unparse(table_units[0][1], db) + ' )', from_tables
            else:
                return '( ' + self.unparse(table_units[0][1], db) + ' ) a , ( ' + self.unparse(table_units[1][1], db) + ' ) b', ['a', 'b']
        else:
            table_names = db['table_names_original']
            table_list = [table_names[int(table_unit[1])] for table_unit in table_units]
            if len(table_list) > 1:
                cond_str = self.unparse_conds(from_clause['conds'], db, 'from')
                cond_str = ' on ' + cond_str if cond_str.strip() else ''
                return ' join '.join(table_list) + cond_str, from_tables
            else: return table_list[0], from_tables


    def unparse_groupby(self, groupby, db):
        return ' , '.join([self.unparse_col_unit(col_unit, db) for col_unit in groupby])


    def unparse_orderby(self, orderby, db):
        return ' , '.join([self.unparse_val_unit(val_unit, db) if agg == 0 else AGG_OPS[agg].lower() + f' ( {self.unparse_val_unit(val_unit, db)} ) ' for agg, val_unit in orderby[1]]) + ' ' + orderby[0].lower()


    def unparse_conds(self, conds: list, db: dict, clause: str = 'where'):
        if not conds: return ''
        cond_str = [self.unparse_cond(cond, db, clause) if cond not in ['and', 'or'] else cond.lower() for cond in conds]
        return ' '.join(cond_str)


    def unparse_cond(self, cond: list, db: dict, clause: str = 'where'):
        agg_id, cmp_id, val_unit, val1, val2 = cond
        agg_op = AGG_OPS[agg_id].lower()
        if agg_op == 'none':
            val_str = self.unparse_val_unit(val_unit, db)
        else:
            val_str = agg_op + f'( {self.unparse_val_unit(val_unit, db)} )'
        val1_str = self.unparse_val(val1, db, clause)
        cmp_op = WHERE_OPS[cmp_id].lower()
        if 'between' in cmp_op:
            val2_str = self.unparse_val(val2, db, clause)
            return val_str + ' between ' + val1_str + ' and ' + val2_str
        return val_str + ' ' + cmp_op + ' ' + val1_str


    def unparse_val(self, val, db, clause='where'):
        if type(val) in [str, int, float]:
            val = str(val).strip('"\'')
            if is_int(val):
                if not val.startswith('0') or val.startswith('0.'):
                    val = int(float(val))
            elif is_number(val):
                val = float(val)
            val_str = '"' + str(val) + '"' if type(val) == str else str(val)
        elif type(val) in [list, tuple]:
            val_str = self.unparse_col_unit(val, db)
        else:
            assert type(val) == dict
            val_str = '( ' + self.unparse(val, db) + ' )'
        return val_str


    def unparse_val_unit(self, val_unit: list, db: dict, from_tables: list = []):
        unit_op, col_unit1, col_unit2 = val_unit
        if unit_op == 0:
            return self.unparse_col_unit(col_unit1, db)
        else:
            unit_join = ' ' + UNIT_OPS[unit_op].lower() + ' '
            tab_name1, tab_name2 = None, None
            if len(from_tables) > 0:
                tab_name1 = from_tables[0]
            if len(from_tables) > 0:
                tab_name2 = from_tables[-1]
            return unit_join.join([self.unparse_col_unit(col_unit1, db, tab_name1), self.unparse_col_unit(col_unit2, db, tab_name2)])


    def unparse_col_unit(self, col_unit: list, db: dict, tab_name = None):
        agg_id, col_id, dis = col_unit
        tab_name = '' if db['column_names_original'][col_id][0] < 0 else tab_name + '.' if tab_name is not None else db['table_names_original'][db['column_names_original'][col_id][0]] + '.'
        col_name = tab_name + db['column_names_original'][col_id][1]
        col_name = 'distinct ' + col_name if dis else col_name
        if agg_id == 0:
            return col_name
        return AGG_OPS[agg_id].lower() + f' ( {col_name} )'


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