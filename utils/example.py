#coding=utf8
import json, random, torch
from torch.utils.data import Dataset
from itertools import chain
from nsts.transition_system import TransitionSystem, CONFIG_PATHS
from preprocess.data_preprocess import PreProcessor
from eval.evaluator import Evaluator


class SQLDataset(Dataset):

    def __init__(self, examples) -> None:
        super(SQLDataset, self).__init__()
        self.examples = examples


    def __len__(self) -> int:
        return len(self.examples)


    def __getitem__(self, index: int):
        return self.examples[index]


class Example():

    @classmethod
    def configuration(cls, dataset, swv=False, plm=None, encode_method='rgatsql', decode_method='ast',
            table_path=None, db_dir=None):
        cls.dataset, cls.swv, cls.plm = dataset, swv, plm
        cls.encode_method, cls.decode_method = encode_method, decode_method
        table_path = CONFIG_PATHS[cls.dataset]['tables'] if table_path is None else table_path
        db_dir = CONFIG_PATHS[cls.dataset]['db_dir'] if db_dir is None else db_dir

        cls.tranx = TransitionSystem(cls.dataset, cls.plm, db_dir)
        cls.evaluator = Evaluator(cls.dataset, cls.tranx, table_path, db_dir)
        cls.grammar, cls.tokenizer = cls.tranx.grammar, cls.tranx.tokenizer
        cls.processor = PreProcessor(cls.dataset, cls.tokenizer, db_dir, cls.encode_method)
        table_list = json.load(open(table_path, 'r'))
        cls.tables = {db['db_id']: db if 'table_token_ids' in db else cls.processor.preprocess_database(db) for db in table_list}
        if cls.encode_method == 'rgatsql':
            for db_id in cls.tables:
                cls.tables[db_id]['relation'] = torch.tensor(cls.tables[db_id]['relation'], dtype=torch.long)


    @classmethod
    def load_dataset(cls, choice='train', dataset_path=None, DEBUG=True):
        if dataset_path is None:
            assert choice in ['train', 'dev']
            choice = 'train' if DEBUG else choice
            dataset = json.load(open(CONFIG_PATHS[cls.dataset][choice], 'r'))
        else:
            choice = 'test'
            dataset = json.load(open(dataset_path, 'r'))

        examples = []
        for idx, ex in enumerate(dataset):
            if 'interaction' in ex: # multi-turn dataset
                db = cls.tables[ex['database_id']]
                if choice == 'train' and len(db['column_names']) > 100: continue # skip large dataset
                if len(ex['interaction']) > 0 and 'question_ids' not in ex['interaction'][0]:
                    ex = cls.processor.pipeline(ex, db)
                for turn in ex['interaction']:
                    turn['db_id'] = db['db_id']
                    cur_ex = cls(turn, db, turn_id=idx)
                    if choice == 'train' and len(cur_ex.input_id) > 430: continue
                    examples.append(cur_ex)
                if DEBUG and len(examples) >= 100: break
            else: # single-turn dataset
                db = cls.tables[ex['db_id']]
                if choice == 'train' and len(db['column_names']) > 100: continue # skip large dataset
                if 'question_ids' not in ex:
                    ex = cls.processor.pipeline(ex, db)
                idx = ex['question_id'] if cls.dataset == 'dusql' else 0
                examples.append(cls(ex, db, id=idx))
                if DEBUG and len(examples) >= 100: break

        return SQLDataset(examples)


    @classmethod
    def use_database_testsuite(cls, db_dir=None):
        testsuite = CONFIG_PATHS[cls.dataset]['testsuite'] if db_dir is None else db_dir
        cls.evaluator.change_database(testsuite)
        cls.tranx.change_database(testsuite)


    def __init__(self, ex: dict, db: dict, id: str = 0, turn_id: int = 0):
        super(Example, self).__init__()
        self.ex, self.db, self.id, self.turn_id = ex, db, id, turn_id
        t = Example.tokenizer

        self.question_id = ex['question_ids']
        self.question_len = len(self.question_id)
        self.separator_pos = ex.get('separator_pos', [self.question_len])
        self.table_token_len = db['table_token_lens']
        column_toks = [
            token_ids + ex['value_id'][str(cid)] if str(cid) in ex['value_id'] else token_ids
            for cid, token_ids in enumerate(db['column_token_ids'])
        ]
        self.column_token_len = [len(toks) for toks in column_toks]
        self.schema_id = sum(db['table_token_ids'], []) + sum(column_toks, [])
        self.schema_len = len(db['table_names']) + len(db['column_names'])

        # input_id: [CLS] question_toks [SEP] table_toks column_toks~(contain BRIDGE cell values) [SEP]
        self.input_id = [t.cls_token_id] + self.question_id + [t.sep_token_id] + self.schema_id + [t.sep_token_id]
        self.segment_id = [0] * (self.question_len + 2) + [1] * (len(self.schema_id) + 1) \
            if Example.plm != 'grappa_large_jnt' and 'roberta' not in Example.plm else [0] * len(self.input_id)
        self.plm_question_mask = [0] + [1] * self.question_len + [0] * (len(self.schema_id) + 2)
        self.plm_schema_mask = [0] * (self.question_len + 2) + [1] * len(self.schema_id) + [0]

        if Example.encode_method == 'rgatsql':
            self.select_copy_mask = [1] * self.question_len + [0] * self.schema_len
            self.select_schema_mask = [0] * self.question_len + [1] * self.schema_len
            self.encoder_relation = (torch.tensor(ex['schema_linking'][0], dtype=torch.long), torch.tensor(ex['schema_linking'][1], dtype=torch.long))
        else: # directly use the outputs from PLM
            self.select_copy_mask, self.select_schema_mask = self.plm_question_mask, self.plm_schema_mask
        self.copy_id = self.question_id

        # labeled outputs
        self.query, self.ast, self.action, self.decoder_relation = '', None, [], []
        if 'query' in ex and ex['query'].strip():
            self.query = ' '.join(' '.join(ex['query'].split('\n')).split('\t'))
            if Example.decode_method == 'ast':
                self.ast = Example.tranx.parse_sql_to_ast(ex['sql'], db)
                self.action_info, self.decoder_relation, _ = Example.tranx.get_action_info_and_relation_from_ast(self.ast)
                self.action, self.decoder_relation = Example.tranx.get_outputs_from_ast(action_infos=self.action_info, relations=self.decoder_relation, order='dfs+l2r')
            else: self.action = Example.tranx.parse_sql_to_seq(ex['sql'], db)


def get_position_ids(ex: Example, shuffle: bool = True):
    # cluster columns with their corresponding table and randomly shuffle tables and columns
    # [CLS] q1 q2 ... [SEP] * t1 c1 {optional cells for c1} c2 {..} t2 c4 {optioinal cells for c4} ... [SEP]
    db, table_token_len, column_token_len = ex.db, ex.table_token_len, ex.column_token_len
    table_num, column_num = len(db['table_names']), len(db['column_names'])
    question_position_id = list(range(ex.question_len + 2))
    table_position_id, column_position_id = [None] * table_num, [None] * column_num

    start = len(question_position_id)
    for cid, (tid, _) in enumerate(db['column_names']):
        if tid != -1: break
        column_position_id[cid] = list(range(start, start + column_token_len[cid]))
        start += column_token_len[cid]

    table_idxs = list(range(table_num))
    if shuffle:
        random.shuffle(table_idxs)
    for idx in table_idxs:
        col_idxs = db['table2columns'][idx]
        table_position_id[idx] = list(range(start, start + table_token_len[idx]))
        start += table_token_len[idx]
        if shuffle:
            random.shuffle(col_idxs)
        for col_id in col_idxs:
            column_position_id[col_id] = list(range(start, start + column_token_len[col_id]))
            start += column_token_len[col_id]
    position_id = question_position_id + list(chain.from_iterable(table_position_id)) + \
        list(chain.from_iterable(column_position_id)) + [start]
    assert len(position_id) == len(ex.input_id)
    return position_id
