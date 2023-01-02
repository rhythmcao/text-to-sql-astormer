#coding=utf8
import os, sqlite3, string
import numpy as np
import stanza
from nltk.corpus import stopwords
from itertools import product, combinations
from utils.constants import MAX_RELATIVE_DIST
from preprocess.graph_utils import GraphProcessor
from preprocess.process_utils import is_number, quote_normalization
from preprocess.bridge_content_encoder import get_database_matches


class InputProcessor():

    def __init__(self, encode_method='lgesql', db_dir='data/spider/database', db_content=True, bridge=True, **kargs):
        super(InputProcessor, self).__init__()
        self.db_dir = db_dir
        self.db_content, self.bridge = db_content, bridge
        self.nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma')#, use_gpu=False)
        self.stopwords = set(stopwords.words("english")) - {'no'}
        self.graph_processor = GraphProcessor(encode_method)
        self.table_pmatch, self.table_ematch = 0, 0
        self.column_pmatch, self.column_ematch, self.column_vmatch = 0, 0, 0

    def pipeline(self, entry: dict, db: dict, verbose: bool = False):
        """ db should be preprocessed """
        entry = self.preprocess_question(entry, verbose=verbose)
        entry = self.schema_linking(entry, db, verbose=verbose)
        entry = self.graph_processor.process_graph_utils(entry, db)
        return entry

    def preprocess_database(self, db: dict, verbose: bool = False):
        """ Tokenize, lemmatize, lowercase table and column names for each database """
        table_toks, processed_table_toks, processed_table_names = [], [], []
        for tab in db['table_names']:
            doc = self.nlp(tab)
            tab = [w.text.lower() for s in doc.sentences for w in s.words]
            ptab = [w.lemma.lower() for s in doc.sentences for w in s.words]
            table_toks.append(tab)
            processed_table_toks.append(ptab)
            processed_table_names.append(" ".join(ptab))
        db['table_toks'] = table_toks
        db['processed_table_toks'] = processed_table_toks
        db['processed_table_names'] = processed_table_names

        column_toks, processed_column_toks, processed_column_names = [], [], []
        for _, c in db['column_names']:
            doc = self.nlp(c)
            c = [w.text.lower() for s in doc.sentences for w in s.words]
            pc = [w.lemma.lower() for s in doc.sentences for w in s.words]
            column_toks.append(c)
            processed_column_toks.append(pc)
            processed_column_names.append(" ".join(pc))
        db['column_toks'] = column_toks
        db['processed_column_toks'] = processed_column_toks
        db['processed_column_names'] =  processed_column_names

        column2table = list(map(lambda x: x[0], db['column_names'])) # from column id to table id
        table2columns = [[] for _ in range(len(db['table_names']))] # from table id to column ids list
        for col_id, col in enumerate(db['column_names']):
            if col_id == 0: continue
            table2columns[col[0]].append(col_id)
        db['column2table'], db['table2columns'] = column2table, table2columns

        t_num, c_num, dtype = len(db['table_names']), len(db['column_names']), '<U100'

        # relations in tables, tab_num * tab_num
        tab_mat = np.array([['table-table-generic'] * t_num for _ in range(t_num)], dtype=dtype)
        table_fks = set(map(lambda pair: (column2table[pair[0]], column2table[pair[1]]), db['foreign_keys']))
        for (tab1, tab2) in table_fks:
            if (tab2, tab1) in table_fks:
                tab_mat[tab1, tab2], tab_mat[tab2, tab1] = 'table-table-fkb', 'table-table-fkb'
            else:
                tab_mat[tab1, tab2], tab_mat[tab2, tab1] = 'table-table-fk', 'table-table-fkr'
        tab_mat[list(range(t_num)), list(range(t_num))] = 'table-table-identity'

        # relations in columns, c_num * c_num
        col_mat = np.array([['column-column-generic'] * c_num for _ in range(c_num)], dtype=dtype)
        for i in range(t_num):
            col_ids = [idx for idx, t in enumerate(column2table) if t == i]
            col1, col2 = list(zip(*list(product(col_ids, col_ids))))
            col_mat[col1, col2] = 'column-column-sametable'
        col_mat[list(range(c_num)), list(range(c_num))] = 'column-column-identity'
        if len(db['foreign_keys']) > 0:
            col1, col2 = list(zip(*db['foreign_keys']))
            col_mat[col1, col2], col_mat[col2, col1] = 'column-column-fk', 'column-column-fkr'
        col_mat[0, list(range(c_num))] = 'column-column-generic'
        col_mat[list(range(c_num)), 0] = 'column-column-generic'
        col_mat[0, 0] = 'column-column-identity'

        # relations between tables and columns, t_num*c_num and c_num*t_num
        tab_col_mat = np.array([['table-column-generic'] * c_num for _ in range(t_num)], dtype=dtype)
        col_tab_mat = np.array([['column-table-generic'] * t_num for _ in range(c_num)], dtype=dtype)
        cols, tabs = list(zip(*list(map(lambda x: (x, column2table[x]), range(1, c_num))))) # ignore *
        col_tab_mat[cols, tabs], tab_col_mat[tabs, cols] = 'column-table-has', 'table-column-has'
        if len(db['primary_keys']) > 0:
            cols, tabs = list(zip(*list(map(lambda x: (x, column2table[x]), db['primary_keys']))))
            col_tab_mat[cols, tabs], tab_col_mat[tabs, cols] = 'column-table-pk', 'table-column-pk'
        col_tab_mat[0, list(range(t_num))] = 'column-table-has' # column-table-generic
        tab_col_mat[list(range(t_num)), 0] = 'table-column-has' # table-column-generic

        relations = np.concatenate([
            np.concatenate([tab_mat, tab_col_mat], axis=1),
            np.concatenate([col_tab_mat, col_mat], axis=1)
        ], axis=0)
        db['relations'] = relations.tolist()

        if verbose:
            print('Tables:', ', '.join(db['table_names']))
            print('Lemmatized:', ', '.join(processed_table_names))
            print('Columns:', ', '.join(list(map(lambda x: x[1], db['column_names']))))
            print('Lemmatized:', ', '.join(processed_column_names), '\n')
        return db

    def preprocess_question(self, entry: dict, verbose: bool = False):
        """ Tokenize, lemmatize, lowercase question"""
        # stanza tokenize, lemmatize and POS tag
        question = quote_normalization(entry['question_toks'])
        doc = self.nlp(question)
        cased_toks = [w.text for s in doc.sentences for w in s.words]
        uncased_toks = [w.text.lower() for s in doc.sentences for w in s.words]
        processed_toks = [w.lemma.lower() for s in doc.sentences for w in s.words]
        # pos_tags = [w.xpos for s in doc.sentences for w in s.words]
        entry['cased_question_toks'] = cased_toks
        entry['uncased_question_toks'] = uncased_toks
        entry['processed_question_toks'] = processed_toks
        # entry['pos_tags'] = pos_tags

        # relations in questions, q_num * q_num
        q_num, dtype = len(processed_toks), '<U100'
        if q_num <= MAX_RELATIVE_DIST + 1:
            dist_vec = ['question-question-dist' + str(i) if i != 0 else 'question-question-identity'
                for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1, 1)]
            starting = MAX_RELATIVE_DIST
        else:
            dist_vec = ['question-question-generic'] * (q_num - MAX_RELATIVE_DIST - 1) + \
                ['question-question-dist' + str(i) if i != 0 else 'question-question-identity' \
                    for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1, 1)] + \
                    ['question-question-generic'] * (q_num - MAX_RELATIVE_DIST - 1)
            starting = q_num - 1
        q_mat = np.array([dist_vec[starting - i: starting - i + q_num] for i in range(q_num)], dtype=dtype)
        entry['relations'] = q_mat.tolist()

        if verbose:
            print('Question:', entry['question'])
            print('Tokenized:', ' '.join(entry['uncased_question_toks']))
            print('Lemmatized:', ' '.join(entry['processed_question_toks']))
            # print('Pos tags:', ' '.join(entry['pos_tags']), '\n')
        return entry

    def schema_linking(self, entry: dict, db: dict, verbose: bool = False):
        """ Perform schema linking: both question and database need to be preprocessed """
        uncased_question_toks, question_toks = entry['uncased_question_toks'], entry['processed_question_toks']
        table_toks, column_toks = db['processed_table_toks'], db['processed_column_toks']
        table_names, column_names = db['processed_table_names'], db['processed_column_names']
        q_num, t_num, c_num, dtype = len(question_toks), len(table_toks), len(column_toks), '<U100'

        def question_schema_matching_method(schema_toks, schema_names, category):
            assert category in ['table', 'column']
            s_num, matched_pairs = len(schema_names), {'partial': [], 'exact': []}
            q_s_mat = np.array([[f'question-{category}-nomatch'] * s_num for _ in range(q_num)], dtype=dtype)
            s_q_mat = np.array([[f'{category}-question-nomatch'] * q_num for _ in range(s_num)], dtype=dtype)
            max_len = max([len(t) for t in schema_toks])
            index_pairs = list(filter(lambda x: x[1] - x[0] <= max_len, combinations(range(q_num + 1), 2)))
            index_pairs = sorted(index_pairs, key=lambda x: x[1] - x[0])
            for i, j in index_pairs:
                phrase = ' '.join(question_toks[i: j])
                if phrase in self.stopwords: continue
                for idx, name in enumerate(schema_names):
                    if category == 'column' and idx == 0: continue
                    if phrase == name: # fully match will overwrite partial match due to sort
                        q_s_mat[range(i, j), idx] = f'question-{category}-exactmatch'
                        s_q_mat[idx, range(i, j)] = f'{category}-question-exactmatch'
                        if verbose:
                            matched_pairs['exact'].append(str((name, idx, phrase, i, j)))
                    elif (j - i == 1 and phrase in name.split()) or (j - i > 1 and phrase in name):
                        q_s_mat[range(i, j), idx] = f'question-{category}-partialmatch'
                        s_q_mat[idx, range(i, j)] = f'{category}-question-partialmatch'
                        if verbose:
                            matched_pairs['partial'].append(str((name, idx, phrase, i, j)))
            return q_s_mat, s_q_mat, matched_pairs

        q_tab_mat, tab_q_mat, table_matched_pairs = question_schema_matching_method(table_toks, table_names, 'table')
        self.table_pmatch += np.sum(q_tab_mat == 'question-table-partialmatch')
        self.table_ematch += np.sum(q_tab_mat == 'question-table-exactmatch')
        q_col_mat, col_q_mat, column_matched_pairs = question_schema_matching_method(column_toks, column_names, 'column')
        self.column_pmatch += np.sum(q_col_mat == 'question-column-partialmatch')
        self.column_ematch += np.sum(q_col_mat == 'question-column-exactmatch')

        if self.db_content:
            column_matched_pairs['value'] = []
            db_file = os.path.join(self.db_dir, db['db_id'], db['db_id'] + '.sqlite')
            try:
                conn = sqlite3.connect(db_file)
                conn.text_factory = lambda b: b.decode(errors='ignore')
                conn.execute('pragma foreign_keys=ON')
                for i, (tab_id, col_name) in enumerate(db['column_names_original']):
                    if i == 0 or 'id' in column_toks[i]: # ignore * and special token 'id'
                        continue
                    tab_name = db['table_names_original'][tab_id]
                    try:
                        cursor = conn.execute("SELECT DISTINCT \"%s\" FROM \"%s\";" % (col_name, tab_name))
                        cell_values = cursor.fetchall()
                        cell_values = [str(each[0]) for each in cell_values]
                        cell_values = [[str(float(each))] if is_number(each) else each.lower().split() for each in cell_values]
                    except: cell_values = []
                    for j, word in enumerate(uncased_question_toks):
                        word = str(float(word)) if is_number(word) else word
                        for c in cell_values:
                            if word in c and 'nomatch' in q_col_mat[j, i] and word not in self.stopwords and word not in string.punctuation:
                                q_col_mat[j, i] = 'question-column-valuematch'
                                col_q_mat[i, j] = 'column-question-valuematch'
                                if verbose:
                                    column_matched_pairs['value'].append(str((column_names[i], i, word, j, j + 1)))
                                break
                conn.close()
            except:
                print('Error while connecting database %s in path %s' % (db['db_id'], db_file))
            self.column_vmatch += np.sum(q_col_mat == 'question-column-valuematch')

        if self.bridge:
            question = entry['question']
            cells, processed_cells = [[]], [[]]
            db_file = os.path.join(self.db_dir, db['db_id'], db['db_id'] + '.sqlite')
            for tab_id, col_name in db['column_names_original']:
                if tab_id < 0: continue
                tab_name = db['table_names_original'][tab_id]
                candidates = get_database_matches(question, tab_name, col_name, db_file)
                processed_candidates = []
                if candidates:
                    candidates = [self.nlp(c) for c in candidates]
                    processed_candidates = ['='] + sum([[w.lemma.lower() for s in c.sentences for w in s.words] + [','] for c in candidates], [])[:-1]
                    candidates = ['='] + sum([[w.text.lower() for s in c.sentences for w in s.words] + [','] for c in candidates], [])[:-1]
                cells.append(candidates)
                processed_cells.append(processed_candidates)
        else: cells = processed_cells = [[] for _ in range(len(db['column_names_original']))]
        entry['cells'], entry['processed_cells'] = cells, processed_cells

        # two symmetric schema linking matrix: q_num x (t_num + c_num), (t_num + c_num) x q_num
        q_col_mat[:, 0] = 'question-column-nomatch'
        col_q_mat[0] = 'column-question-nomatch'
        q_schema = np.concatenate([q_tab_mat, q_col_mat], axis=1)
        schema_q = np.concatenate([tab_q_mat, col_q_mat], axis=0)
        entry['schema_linking'] = (q_schema.tolist(), schema_q.tolist())

        if verbose:
            print('Question:', ' '.join(question_toks))
            print('Table matched: (table name, column id, question span, start id, end id)')
            print('Exact match:', ', '.join(table_matched_pairs['exact']) if table_matched_pairs['exact'] else 'empty')
            print('Partial match:', ', '.join(table_matched_pairs['partial']) if table_matched_pairs['partial'] else 'empty')
            print('Column matched: (column name, column id, question span, start id, end id)')
            print('Exact match:', ', '.join(column_matched_pairs['exact']) if column_matched_pairs['exact'] else 'empty')
            print('Partial match:', ', '.join(column_matched_pairs['partial']) if column_matched_pairs['partial'] else 'empty')
            if self.db_content:
                print('Value match:', ', '.join(column_matched_pairs['value']) if column_matched_pairs['value'] else 'empty')
            print('\n')
        return entry
