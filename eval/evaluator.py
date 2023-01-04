#coding=utf8
import json, sys, tempfile, os, asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.evaluation import build_foreign_key_map_from_json, evaluate
from eval.evaluation import Evaluator as Engine
from eval.exec_eval import postprocess, exec_on_db
from eval.process_sql import get_schema, Schema
from nsts.transition_system import CONFIG_PATHS, TransitionSystem


class ExecutionChecker():

    def __init__(self, db_dir: str) -> None:
        super(ExecutionChecker, self).__init__()
        self.db_dir = db_dir


    def validity_check(self, sql: str, db: dict) -> bool:
        db_id = db['db_id']
        db_path = os.path.join(self.db_dir, db_id, db_id + ".sqlite")
        sql = postprocess(sql)
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
        self.schemas = self._load_schemas(self.table_path)
        self.engine, self.exec_checker = Engine(), ExecutionChecker(self.db_dir)


    def _load_schemas(self, table_path):
        schemas = {}
        tables = json.load(open(table_path, 'r'))
        for db in tables:
            db = db['db_id']
            db_path = os.path.join(self.db_dir, db, db + ".sqlite")
            schemas[db] = Schema(get_schema(db_path))
        return schemas


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
    checker = ExecutionChecker(CONFIG_PATHS[args.dataset]['db_dir'])
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