#coding=utf8
import json, sys, tempfile, os, asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.evaluation import build_foreign_key_map_from_json, evaluate
from eval.evaluation import Evaluator as Engine
from eval.exec_eval import postprocess, exec_on_db
from nsts.transition_system import CONFIG_PATHS, TransitionSystem
from nsts.parse_sql_to_json import get_sql


class SurfaceChecker():

    def validity_check(self, sql: str, db: dict) -> bool:
        """ Check whether the given sql query is valid, including:
        1. only use columns in tables mentioned in FROM clause
        2. table JOIN conditions t1.col1=t2.col2, t1.col1 and t2.col2 belongs to different tables
        3. comparison operator or MAX/MIN/SUM/AVG only applied to columns of type number/time
        @args:
            sql(str): SQL query
            db(dict): database dict
        @return:
            flag(boolean)
        """
        try:
            sql = get_sql(sql, db)
            return self.sql_check(sql, db)
        except Exception as e:
            return False


    def sql_check(self, sql: dict, db: dict):
        if sql['intersect']:
            return self.sqlunit_check(sql, db) & self.sqlunit_check(sql['intersect'], db)
        if sql['union']:
            return self.sqlunit_check(sql, db) & self.sqlunit_check(sql['union'], db)
        if sql['except']:
            return self.sqlunit_check(sql, db) & self.sqlunit_check(sql['except'], db)
        return self.sqlunit_check(sql, db)


    def sqlunit_check(self, sql: dict, db: dict):
        if sql['from']['table_units'][0][0] == 'sql':
            if not self.sql_check(sql['from']['table_units'][0][1], db): return False
            table_ids = []
        else:
            table_ids = list(map(lambda table_unit: table_unit[1], sql['from']['table_units']))
            if len(sql['from']['conds']) > 0: # predict FROM conditions
                if not self.from_condition_check(sql['from']['conds'], table_ids, db): return False
        return self.select_check(sql['select'], table_ids, db) & \
            self.cond_check(sql['where'], table_ids, db) & \
            self.groupby_check(sql['groupBy'], table_ids, db) & \
            self.cond_check(sql['having'], table_ids, db) & \
            self.orderby_check(sql['orderBy'], table_ids, db)


    def from_condition_check(self, conds: list, table_ids: list, db: dict):
        count = {tid: table_ids.count(tid) for tid in table_ids} # number of occurrences for each table
        for cond in conds:
            if cond in ['and', 'or']: continue
            _, _, val_unit, val1, _ = cond
            col_id1, col_id2 = val_unit[1][1], val1[1]
            tid1, tid2 = db['column_names'][col_id1][0], db['column_names'][col_id2][0]
            if tid1 not in table_ids or tid2 not in table_ids: return False
            if tid1 == tid2 and count[tid1] == 1: return False # if JOIN the same table, table must appear multiple times in FROM
        return True


    def select_check(self, select, table_ids: list, db: dict):
        select = select[1]
        for agg_id, val_unit in select:
            if not self.valunit_check(val_unit, table_ids, db): return False
        return True


    def cond_check(self, cond, table_ids: list, db: dict):
        if len(cond) == 0: return True
        for idx in range(0, len(cond), 2):
            cond_unit = cond[idx]
            _, cmp_op, val_unit, val1, val2 = cond_unit
            flag = self.valunit_check(val_unit, table_ids, db)
            if type(val1) == dict:
                flag &= self.sql_check(val1, db)
            if type(val2) == dict:
                flag &= self.sql_check(val2, db)
            if not flag: return False
        return True


    def groupby_check(self, groupby, table_ids: list, db: dict):
        if not groupby: return True
        for col_unit in groupby:
            if not self.colunit_check(col_unit, table_ids, db): return False
        return True


    def orderby_check(self, orderby, table_ids: list, db: dict):
        if not orderby: return True
        orderby = orderby[1]
        for val_unit in orderby:
            if not self.valunit_check(val_unit, table_ids, db): return False
        return True


    def colunit_check(self, col_unit: list, table_ids: list, db: dict):
        """ Check from the following aspects:
        1. column belongs to the tables in FROM clause
        2. column type is valid for AGG_OP
        """
        agg_id, col_id, _ = col_unit
        if col_id == 0: return True
        tab_id = db['column_names'][col_id][0]
        if tab_id not in table_ids: return False
        col_type = db['column_types'][col_id]
        if agg_id in [1, 2, 4, 5]: # MAX, MIN, SUM, AVG
            return (col_type in ['time', 'number'])
        return True


    def valunit_check(self, val_unit: list, table_ids: list, db: dict):
        unit_op, col_unit1, col_unit2 = val_unit
        if unit_op == 0: return self.colunit_check(col_unit1, table_ids, db)
        if not (self.colunit_check(col_unit1, table_ids, db) and self.colunit_check(col_unit2, table_ids, db)): return False
        # COUNT/SUM/AVG -> number
        agg_id1, col_id1, _ = col_unit1
        agg_id2, col_id2, _ = col_unit2
        t1 = 'number' if agg_id1 > 2 else db['column_types'][col_id1]
        t2 = 'number' if agg_id2 > 2 else db['column_types'][col_id2]
        if (t1 not in ['number', 'time']) or (t2 not in ['number', 'time']) or t1 != t2: return False
        return True


class ExecutionChecker():

    def __init__(self, table_path: str, db_dir: str) -> None:
        super(ExecutionChecker, self).__init__()
        self.table_path, self.db_dir = table_path, db_dir
        self.surface_checker = SurfaceChecker()


    def validity_check(self, sql: str, db: dict) -> bool:
        db_id = db['db_id']
        db_path = os.path.join(self.db_dir, db_id, db_id + ".sqlite")
        sql = postprocess(sql)
        # if not os.path.exists(db_path):
        return self.surface_checker.validity_check(sql, db)
        flag, _ = asyncio.run(exec_on_db(db_path, sql))
        if flag == 'exception':
            return False
        return True


class Evaluator():

    def __init__(self, dataset: str, tranx: TransitionSystem, table_path: str = None, db_dir: str = None):
        super(Evaluator, self).__init__()
        self.dataset, self.tranx = dataset, tranx
        self.table_path = CONFIG_PATHS[dataset]['tables'] if table_path is None else table_path
        self.db_dir = CONFIG_PATHS[dataset]['db_dir'] if db_dir is None else db_dir
        self.kmaps = build_foreign_key_map_from_json(self.table_path)
        self.engine, self.exec_checker = Engine(), ExecutionChecker(self.table_path, self.db_dir)


    def change_database(self, db_dir):
        """ Obtaining testsuite execution accuracy is too expensive if evaluated repeatedly during training,
        thus just evaluate on testsuite database after training is finished
        """
        self.db_dir = db_dir
        self.exec_checker.db_dir = db_dir


    def accuracy(self, pred_sqls, dataset, output_path=None, etype='all'):
        assert etype in ['match', 'exec', 'all']
        result = self.evaluate_with_official_interface(pred_sqls, dataset, output_path, etype)

        if etype == 'match':
            return float(result['exact'])
        elif etype == 'exec':
            return float(result['exec'])
        else: return (float(result['exact']), float(result['exec']))


    def postprocess(self, batch_hyps, dataset, decode_method='ast', execution_guided=True):
        """ For each sample, choose the top of beam hypothesis in the final beam without execution error.
        """
        pred_sqls = []
        for eid, beam_hyps in enumerate(batch_hyps):
            cur_preds = []
            for hyp in beam_hyps:
                # different decoding methods
                if decode_method == 'ast':
                    pred, flag = self.tranx.unparse_sql_from_ast(hyp.tree, dataset[eid].db, dataset[eid].ex)
                else:
                    pred, flag = self.tranx.unparse_sql_from_seq(hyp.action, dataset[eid].db, dataset[eid].ex)

                if not execution_guided: # not execution guided, directly return top of beam prediction
                    pred_sqls.append(pred)
                    break

                if flag and self.exec_checker.validity_check(pred, dataset[eid].db):
                    pred_sqls.append(pred)
                    break
                cur_preds.append(pred)
            else: # if all hyps failed on execution check, directly return the top of beam
                pred_sqls.append(cur_preds[0])
        return pred_sqls


    def evaluate_with_official_interface(self, pred_sqls, dataset, output_path: str = None, etype: str = 'all'):
        with tempfile.NamedTemporaryFile('w+t', encoding='utf8', suffix='.sql') as tmp_pred, \
            tempfile.NamedTemporaryFile('w+t', encoding='utf8', suffix='.sql') as tmp_ref:
            assert len(pred_sqls) == len(dataset)
            # write pred sqls
            prev_id = 0
            for s, ex in zip(pred_sqls, dataset):
                prefix = '\n' if ex.id != prev_id else ''
                tmp_pred.write(prefix + s + '\n')
                prev_id = ex.id
            tmp_pred.flush()

            # write gold sqls
            prev_id = 0
            for ex in dataset:
                prefix = '\n' if ex.id != prev_id else ''
                tmp_ref.write(prefix + ex.query + '\t' + ex.db['db_id'] + '\n')
                prev_id = ex.id
            tmp_ref.flush()

            of = open(output_path, 'w', encoding='utf8') if output_path is not None else tempfile.TemporaryFile('w+t')
            old_print, sys.stdout = sys.stdout, of
            results = evaluate(tmp_ref.name, tmp_pred.name, self.db_dir, etype, self.kmaps, False, False, False)
            sys.stdout = old_print
            of.close()
        return results


if __name__ == '__main__':

    import argparse
    from tqdm import tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='dataset', default='spider', choices=['spider', 'sparc', 'cosql'], help='dataset name')
    parser.add_argument('-s', dest='data_split', default='train', choices=['train', 'dev'])
    args = parser.parse_args(sys.argv[1:])

    error_count = 0
    tables = {db['db_id']: db for db in json.load(open(CONFIG_PATHS[args.dataset]['tables'], 'r'))}
    checker = ExecutionChecker(CONFIG_PATHS[args.dataset]['tables'], CONFIG_PATHS[args.dataset]['db_dir'])
    dataset = json.load(open(CONFIG_PATHS[args.dataset][args.data_split], 'r'))

    for idx, ex in tqdm(enumerate(dataset), desc='Execution check', total=len(dataset)):
        if 'interaction' in ex:
            db_id = ex['database_id']
            for turn in ex['interaction']:
                query = turn['query']
                flag = checker.validity_check(query, tables[db_id])
                if not flag:
                    print(query + '\t' + db_id)
                    error_count += 1
        else:
            query, db_id = ex['query'], ex['db_id']
            flag = checker.validity_check(query, tables[db_id])
            if not flag:
                print(query + '\t' + db_id)
                error_count += 1
    print('Total number of invalid SQLs is %d .' % (error_count))
