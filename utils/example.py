#coding=utf8
import json, random
import numpy as np
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
    def configuration(cls, dataset, plm=None, encode_method='none', decode_method='ast',
            table_path=None, db_dir=None):
        cls.dataset, cls.plm = dataset, plm
        cls.encode_method, cls.decode_method = encode_method, decode_method
        table_path = CONFIG_PATHS[cls.dataset]['tables'] if table_path is None else table_path
        db_dir = CONFIG_PATHS[cls.dataset]['db_dir'] if db_dir is None else db_dir

        cls.tranx = TransitionSystem(cls.dataset, cls.plm, db_dir)
        cls.evaluator = Evaluator(cls.dataset, cls.tranx, table_path, db_dir)
        cls.grammar, cls.tokenizer = cls.tranx.grammar, cls.tranx.tokenizer
        cls.processor = PreProcessor(cls.tokenizer, db_dir, cls.encode_method)
        table_list = json.load(open(table_path, 'r'))
        cls.tables = {db['db_id']: db if 'table_id' in db else cls.processor.preprocess_database(db) for db in table_list}


    @classmethod
    def load_dataset(cls, choice='train', dataset_path=None, DEBUG=False):
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
                if len(ex['interaction']) > 0 and 'question_id' not in ex['interaction'][0]:
                    ex = cls.processor.pipeline(ex, db)
                for turn in ex['interaction']:
                    turn['db_id'] = db['db_id']
                    examples.append(cls(turn, db, idx))
                if DEBUG and len(examples) >= 100: break
            else: # single-turn dataset
                db = cls.tables[ex['db_id']]
                if choice == 'train' and len(db['column_names']) > 100: continue # skip large dataset
                if 'question_id' not in ex: ex = cls.processor.pipeline(ex, db)
                examples.append(cls(ex, db))
                if DEBUG and len(examples) >= 100: break

        return SQLDataset(examples)


    @classmethod
    def use_database_testsuite(cls, db_dir=None):
        testsuite = CONFIG_PATHS[cls.dataset]['testsuite'] if db_dir is None else db_dir
        cls.evaluator.change_database(testsuite)
        cls.tranx.change_database(testsuite)


    def __init__(self, ex: dict, db: dict, id: str = 0):
        super(Example, self).__init__()
        self.ex, self.db, self.id = ex, db, id
        t = Example.tokenizer

        self.question_id = ex['question_id']
        self.question_len = len(self.question_id)
        self.table_token_len = db['table_token_len']
        column_toks = [
            token_ids + ex['value_id'][str(cid)] if str(cid) in ex['value_id'] else token_ids
            for cid, token_ids in enumerate(db['column_token_id'])
        ]
        self.column_token_len = [len(toks) for toks in column_toks]
        self.schema_id = db['table_id'] + sum(column_toks, [])
        self.schema_len = len(db['table_names']) + len(db['column_names'])

        # input_id: [CLS] question_toks [SEP] table_toks column_toks~(contain BRIDGE cell values) [SEP]
        self.input_id = [t.cls_token_id] + self.question_id + [t.sep_token_id] + self.schema_id + [t.sep_token_id]
        self.segment_id = [0] * (self.question_len + 2) + [1] * (len(self.schema_id) + 1) \
            if Example.plm != 'grappa_large_jnt' and 'roberta' not in Example.plm else [0] * len(self.input_id)
        self.plm_question_mask = [0] + [1] * self.question_len + [0] * (len(self.schema_id) + 2)
        self.plm_schema_mask = [0] * (self.question_len + 2) + [1] * len(self.schema_id) + [0]

        if Example.encode_method == 'none': # directly use the outputs from PLM, copy_id include BRIDGE cell values
            self.select_copy_mask = [0] + [1] * self.question_len + [0] * (len(db['table_id']) + 1) + \
                sum([[0] * (len(token_ids) + 1) + [1] * (len(ex['value_id'][str(cid)]) - 1) if str(cid) in ex['value_id'] else [0] * len(token_ids) for cid, token_ids in enumerate(db['column_token_id'])], []) + [0]
            self.select_schema_mask = self.plm_schema_mask
            self.copy_id = self.question_id + sum([ex['value_id'][str(cid)][1:] if str(cid) in ex['value_id'] else [] for cid in range(len(db['column_names']))], [])
        else:
            self.select_copy_mask = [1] * self.question_len + [0] * self.schema_len
            self.select_schema_mask = [0] * self.question_len + [1] * self.schema_len
            self.copy_id = self.question_id
            self.encoder_relation = [] # TODO

        # labeled outputs
        self.query, self.ast, self.action, self.decoder_relation = '', None, [], []
        if ex['query'].strip():
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
    column_position_id[0] = list(range(start, start + column_token_len[0]))
    start += column_token_len[0]

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
