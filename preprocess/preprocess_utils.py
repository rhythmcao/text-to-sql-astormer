#coding=utf8
import numpy as np
import os, re, sqlite3, string, stanza, torch
from typing import List, Tuple
from nltk.corpus import stopwords
from itertools import product, combinations
from nsts.relation_utils import ENCODER_RELATIONS, MAX_RELATIVE_DIST
from preprocess.bridge_content_encoder import get_database_matches


def get_question_relation(separator_pos: List[int] = []) -> torch.LongTensor:
    """ Return relations among question nodes, q_num * q_num.
    @args:
        separator_pos represents the indexes of separators for different utterances.
    """
    start_pos, dtype = 0, np.int64
    relations = np.zeros(shape=(0, 0), dtype=dtype)
    qqp, qqa = ENCODER_RELATIONS.index(f'question-question-previous'), ENCODER_RELATIONS.index(f'question-question-after')
    for end_pos in separator_pos:
        q_num = end_pos - start_pos
        if q_num <= MAX_RELATIVE_DIST + 1:
            dist_vec = [ENCODER_RELATIONS.index(f'question-question-dist{i:d}') for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1, 1)]
            starting = MAX_RELATIVE_DIST
        else:
            dist_vec = [ENCODER_RELATIONS.index('question-question-generic')] * (q_num - MAX_RELATIVE_DIST - 1) + \
                [ENCODER_RELATIONS.index(f'question-question-dist{i:d}') for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1, 1)] + \
                    [ENCODER_RELATIONS.index('question-question-generic')] * (q_num - MAX_RELATIVE_DIST - 1)
            starting = q_num - 1
        q_mat = np.array([dist_vec[starting - i: starting - i + q_num] for i in range(q_num)], dtype=dtype)
        relations = np.concatenate([
            np.concatenate([relations, np.full((start_pos, q_num), qqp, dtype=dtype)], axis=1),
            np.concatenate([np.full((q_num, start_pos), qqa, dtype=dtype), q_mat], axis=1)
        ], axis=0)
        start_pos = end_pos
    return torch.from_numpy(relations)


class PreProcessor(object):

    utterance_separator = ' | '
    value_separator = ' , '
    column_value_separator = ': '

    def __init__(self, tokenizer, db_dir, encode_method, db_content=True):
        super(PreProcessor, self).__init__()
        self.tokenizer, self.db_dir, self.encode_method = tokenizer, db_dir, encode_method
        self.stopwords = set(stopwords.words("english")) - {'no'}
        use_gpu = torch.cuda.is_available()
        self.nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma', use_gpu=False)
        self.matches = {'table': {'partial': 0, 'exact': 0}, 'column': {'partial': 0, 'exact': 0, 'value': 0}}
        self.db_content, self.bridge_value = db_content, 0


    def clear_statistics(self):
        self.matches = {'table': {'partial': 0, 'exact': 0}, 'column': {'partial': 0, 'exact': 0, 'value': 0}}
        self.bridge_value = 0


    def accumulate_matchings(self, qs_mat):
        self.matches['table']['exact'] += np.sum(qs_mat == ENCODER_RELATIONS.index('question-table-exactmatch'))
        self.matches['table']['partial'] += np.sum(qs_mat == ENCODER_RELATIONS.index('question-table-partialmatch'))
        self.matches['column']['exact'] += np.sum(qs_mat == ENCODER_RELATIONS.index('question-column-exactmatch'))
        self.matches['column']['partial'] += np.sum(qs_mat == ENCODER_RELATIONS.index('question-column-partialmatch'))
        self.matches['column']['value'] += np.sum(qs_mat == ENCODER_RELATIONS.index('question-column-valuematch'))


    def pipeline(self, entry: dict, db: dict, tools=['tok', 'vl', 'sl'], verbose: bool = False):
        """ db should be preprocessed """
        if 'tok' in tools:
            entry = self.preprocess_question(entry, verbose=verbose)
        if 'vl' in tools:
            entry = self.value_linking(entry, db, verbose=verbose)
        if 'sl' in tools and self.encode_method != 'none':
            entry = self.schema_linking(entry, db, verbose=verbose)
        return entry


    def preprocess_database(self, db: dict):
        table_prefix = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('table'))
        table_ids = [
            table_prefix + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(table_name))
            for table_name in db['table_names']
        ]
        db['table_token_len'] = [len(toks) for toks in table_ids]
        db['table_id'] = sum(table_ids, []) # flatten all content
        db['table_toks'] = [[w.lemma.lower() for s in self.nlp(name).sentences for w in s.words] for name in db['table_names']]

        column_ids = [
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(db['column_types'][cid])) +
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(db['column_names'][cid][1]))
            for cid in range(len(db['column_names']))
        ]
        db['column_token_id'] = column_ids # need to add instance-specific BRIDGE cells
        db['column_toks'] = [[w.lemma.lower() for s in self.nlp(name).sentences for w in s.words] for _, name in db['column_names']]

        column2table = list(map(lambda x: x[0], db['column_names'])) # from column id to table id
        table2columns = [[] for _ in range(len(db['table_names']))] # from table id to column ids list
        for col_id, col in enumerate(db['column_names']):
            if col_id == 0: continue
            table2columns[col[0]].append(col_id)
        db['column2table'], db['table2columns'] = column2table, table2columns

        if self.encode_method != 'none':
            # construct relations intra database schema items
            t_num, c_num, dtype = len(db['table_names']), len(db['column_names']), np.int64
            # relations in tables, tab_num * tab_num
            ttg, tti = ENCODER_RELATIONS.index('table-table-generic'), ENCODER_RELATIONS.index('table-table-identity')
            ttf, ttfr, ttfb = ENCODER_RELATIONS.index('table-table-fk'), ENCODER_RELATIONS.index('table-table-fkr'), ENCODER_RELATIONS.index('table-table-fkb')
            tab_mat = np.array([[ttg] * t_num for _ in range(t_num)], dtype=dtype)
            table_fks = set(map(lambda pair: (column2table[pair[0]], column2table[pair[1]]), db['foreign_keys']))
            for (tab1, tab2) in table_fks:
                if (tab2, tab1) in table_fks: tab_mat[tab1, tab2], tab_mat[tab2, tab1] = ttfb, ttfb
                else: tab_mat[tab1, tab2], tab_mat[tab2, tab1] = ttf, ttfr
            tab_mat[list(range(t_num)), list(range(t_num))] = tti

            # relations in columns, c_num * c_num
            ccg, ccs, cci = ENCODER_RELATIONS.index('column-column-generic'), ENCODER_RELATIONS.index('column-column-sametable'), ENCODER_RELATIONS.index('column-column-identity')
            ccf, ccfr = ENCODER_RELATIONS.index('column-column-fk'), ENCODER_RELATIONS.index('column-column-fkr')
            col_mat = np.array([[ccg] * c_num for _ in range(c_num)], dtype=dtype)
            for i in range(t_num):
                col_ids = [idx for idx, t in enumerate(column2table) if t == i]
                col1, col2 = list(zip(*list(product(col_ids, col_ids))))
                col_mat[col1, col2] = ccs
            col_mat[list(range(c_num)), list(range(c_num))] = cci
            if len(db['foreign_keys']) > 0:
                col1, col2 = list(zip(*db['foreign_keys']))
                col_mat[col1, col2], col_mat[col2, col1] = ccf, ccfr
            col_mat[0, list(range(c_num))] = ccg
            col_mat[list(range(c_num)), 0] = ccg
            col_mat[0, 0] = cci

            # relations between tables and columns, t_num*c_num and c_num*t_num
            tcg, ctg = ENCODER_RELATIONS.index('table-column-generic'), ENCODER_RELATIONS.index('column-table-generic')
            cth, tch = ENCODER_RELATIONS.index('column-table-has'), ENCODER_RELATIONS.index('table-column-has')
            ctp, tcp = ENCODER_RELATIONS.index('column-table-pk'), ENCODER_RELATIONS.index('table-column-pk')
            tab_col_mat = np.array([[tcg] * c_num for _ in range(t_num)], dtype=dtype)
            col_tab_mat = np.array([[ctg] * t_num for _ in range(c_num)], dtype=dtype)
            cols, tabs = list(zip(*list(map(lambda x: (x, column2table[x]), range(1, c_num))))) # ignore *
            col_tab_mat[cols, tabs], tab_col_mat[tabs, cols] = cth, tch
            if len(db['primary_keys']) > 0:
                cols, tabs = list(zip(*list(map(lambda x: (x, column2table[x]), db['primary_keys']))))
                col_tab_mat[cols, tabs], tab_col_mat[tabs, cols] = ctp, tcp
            col_tab_mat[0, list(range(t_num))] = cth # column-table-generic
            tab_col_mat[list(range(t_num)), 0] = tch # table-column-generic

            relation = np.concatenate([
                np.concatenate([tab_mat, tab_col_mat], axis=1),
                np.concatenate([col_tab_mat, col_mat], axis=1)
            ], axis=0)
            db['relation'] = relation.tolist()

        return db


    def preprocess_question(self, entry: dict, verbose: bool = False):
        """ Tokenize the question sentence. For contextual text-to-SQL, append history questions to the current utterance.
        """
        if 'interaction' in entry:
            prev_questions, prev_question_ids, prev_sep_pos, sep = [], [], [], PreProcessor.utterance_separator
            sep_id = self.tokenizer.convert_tokens_to_ids(sep.strip())
            for turn in entry['interaction']:
                question = re.sub(r'\s+', ' ', turn['utterance']).strip()
                turn['utterance'] = question
                turn['utterance_toks'] = self.tokenizer.tokenize(question)
                # add previous input user question in reverse order
                prev_questions = [question] + prev_questions
                turn['question'] = sep.join(prev_questions) # used in value-linking and postprocess for cased SQL value retrieval
                turn['question_id'] = self.tokenizer.convert_tokens_to_ids(turn['utterance_toks']) + prev_question_ids
                prev_question_ids = [sep_id] + turn['question_id']
                # record the index of separators for different utterances (to construct question relations)
                sep_pos = len(turn['utterance_toks'])
                prev_sep_pos = turn['separator_pos'] = [sep_pos] + [idx + sep_pos + 1 for idx in prev_sep_pos]
        else:
            entry['question'] = re.sub(r'\s+', ' ', entry['question']).strip() # remove redundant whitespaces
            entry['question_toks'] = self.tokenizer.tokenize(entry['question'])
            entry['question_id'] = self.tokenizer.convert_tokens_to_ids(entry['question_toks'])
        return entry


    def value_linking(self, entry: dict, db: dict, verbose: bool = False):
        # extract database content according the question in each turn
        db_file = os.path.join(self.db_dir, db['db_id'], db['db_id'] + '.sqlite')

        def extract_candidate_values(question):
            matched_values, matched_value_ids = {}, {}
            for cid, (tab_id, col_name) in enumerate(db['column_names_original']):
                if tab_id < 0: continue
                tab_name = db['table_names_original'][tab_id]
                values = get_database_matches(question, tab_name, col_name, db_file)
                self.bridge_value += len(values)
                if len(values) > 0:
                    matched_values[str(cid)] = values
                    value_span = PreProcessor.column_value_separator + PreProcessor.value_separator.join(values)
                    matched_value_ids[str(cid)] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value_span))
            return (matched_values, matched_value_ids)

        if 'interaction' in entry:
            for turn in entry['interaction']:
                if not self.db_content:
                    turn['value'], turn['value_id'] = {}, {}
                    continue
                turn['value'], turn['value_id'] = extract_candidate_values(turn['question'])
        else:
            entry['value'], entry['value_id'] = ({}, {}) if not self.db_content else extract_candidate_values(entry['question'])
        return entry


    def schema_linking(self, entry: dict, db: dict, verbose: bool = False):
        """ Perform schema linking: the question need to be preprocessed and value linking needs to be performed in advance.
        """
        dtype = np.int64
        table_names, column_names = [t.lower() for t in db['table_names']], [c.lower() for _, c in db['column_names']]
        table_toks, column_toks = db['table_toks'], db['column_toks']

        def normalize_toks(toks):
            return self.tokenizer.convert_tokens_to_string(toks).lower().strip()


        def extract_index_phrase_pairs(question_toks) -> List[Tuple[Tuple[int, int], str]]:
            """ For tokenized questions, extract all (start, end) index pairs and the corresponding lower-cased question span pairs.
            """
            filter_func = lambda x: 1 <= x[1] - x[0] <= 15 and question_toks[x[0]] not in ['Ġ', '▁'] and question_toks[x[1] - 1] not in ['Ġ', '▁'] \
                and not question_toks[x[0]].startswith('#')
            indexes = sorted(filter(filter_func, combinations(range(len(question_toks) + 1), 2)), key=lambda x: x[1] - x[0])
            index_phrase_pairs = map(lambda pair: (pair, normalize_toks(question_toks[pair[0]: pair[1]])), indexes)
            return list(filter(lambda x: x[1] not in self.stopwords and x[1] not in string.punctuation, index_phrase_pairs))


        def question_schema_matching(index_phrase_pairs, schema_names, schema_toks, category, matching_infos=None):
            assert category in ['table', 'column']
            if len(matching_infos) == 3: qs_mat, sq_mat, matched_pairs = matching_infos
            else:
                q_num, s_num = matching_infos
                qsn, sqn = ENCODER_RELATIONS.index(f'question-{category}-nomatch'), ENCODER_RELATIONS.index(f'{category}-question-nomatch')
                qs_mat = np.array([[qsn] * s_num for _ in range(q_num)], dtype=dtype)
                sq_mat = np.array([[sqn] * q_num for _ in range(s_num)], dtype=dtype)
                matched_pairs = {'partial': [], 'exact': [], 'value': []}

            qsp, sqp = ENCODER_RELATIONS.index(f'question-{category}-partialmatch'), ENCODER_RELATIONS.index(f'{category}-question-partialmatch')
            qse, sqe = ENCODER_RELATIONS.index(f'question-{category}-exactmatch'), ENCODER_RELATIONS.index(f'{category}-question-exactmatch')
            for sid, sname in enumerate(schema_names):
                if category == 'column' and sid == 0: continue
                max_len = len(self.tokenizer.tokenize(sname))
                sname_toks = schema_toks[sid]
                for (start, end), phrase in index_phrase_pairs:
                    if end - start > max_len: break
                    if phrase == sname or (sname in phrase and len(sname) >= 2 * len(phrase) / 3): # over-write partial match due to sort according to length
                        qs_mat[range(start, end), sid], sq_mat[sid, range(start, end)] = qse, sqe
                        if verbose: matched_pairs['exact'].append(str((sname, sid, phrase, start, end)))
                    elif phrase in sname.split(' ') or phrase in sname_toks:
                        qs_mat[range(start, end), sid], sq_mat[sid, range(start, end)] = qsp, sqp
                        if verbose: matched_pairs['partial'].append(str((sname, sid, phrase, start, end)))
            return qs_mat, sq_mat, matched_pairs


        def extract_nontext_cell_values(db: dict):
            if not self.db_content: return [[] for _ in range(len(db['column_names']))]
            cell_values = []
            db_file = os.path.join(self.db_dir, db['db_id'], db['db_id'] + '.sqlite')
            try:
                conn = sqlite3.connect(db_file)
                conn.text_factory = lambda b: b.decode(errors='ignore')
                conn.execute('pragma foreign_keys=ON')
                for cid, (tid, col_name) in enumerate(db['column_names_original']):
                    if cid == 0 or db['column_types'][cid] == 'text' or 'id' in db['column_names'][cid][1].lower().split(' '):
                        cell_values.append([])
                        continue
                    try:
                        tab_name = db['table_names'][tid]
                        cursor = conn.execute(f"SELECT DISTINCT \"{col_name}\" FROM \"{tab_name}\";")
                        cells = cursor.fetchall()
                        cells = [each[0] for each in cell_values]
                        cells = [str(c) for c in cells] if len(cells) > 0 and type(cells[0]) != str else []
                    except: cells = []
                    cell_values.append(cells)
                conn.close()
            except:
                print('Error while connecting database %s in path %s' % (db['db_id'], db_file))
                cell_values = [[] for _ in range(len(db['column_names']))]
            return cell_values

        def question_cell_matching(index_span_pairs, question, cell_values, bridge_values, matching_infos=None):
            if len(matching_infos) == 3: qc_mat, cq_mat, matched_pairs = matching_infos
            else:
                q_num, c_num = matching_infos
                qcn, cqn = ENCODER_RELATIONS.index('question-column-nomatch'), ENCODER_RELATIONS.index('column-question-nomatch')
                qc_mat = np.array([[qcn] * c_num for _ in range(q_num)], dtype=dtype)
                cq_mat = np.array([[cqn] * q_num for _ in range(c_num)], dtype=dtype)
                matched_pairs = {'partial': [], 'exact': [], 'value': []}
            if not self.db_content: return qc_mat, cq_mat, matched_pairs

            qcv, cqv = ENCODER_RELATIONS.index('question-column-valuematch'), ENCODER_RELATIONS.index('column-question-valuematch')
            for cid, cells in enumerate(cell_values):
                # re-use value-linking/BRIDGE information to save time
                is_text, cells = (True, bridge_values[str(cid)]) if str(cid) in bridge_values else (False, cells)
                for c in cells:
                    if is_text or (not is_text and c in question):
                        toks = self.tokenizer.tokenize(c)
                        l, c = len(toks), normalize_toks(toks)
                        for (start, end), phrase in filter(lambda x: l - 2 <= x[0][1] - x[0][0] <= l + 2, index_span_pairs):
                            if phrase in c:
                                qc_mat[range(start, end), cid], cq_mat[cid, range(start, end)] = qcv, cqv
                                if verbose: matched_pairs['value'].append(str((column_names[cid], cid, phrase, start, end)))
            return qc_mat, cq_mat, matched_pairs


        def print_matching_infos(question_toks, table_matched_pairs, column_matched_pairs):
            print('Question:', question_toks)
            print('Table matched: (table name, table id, question span, start id, end id)')
            print('Exact match:', ', '.join(table_matched_pairs['exact']) if table_matched_pairs['exact'] else 'empty')
            print('Partial match:', ', '.join(table_matched_pairs['partial']) if table_matched_pairs['partial'] else 'empty')
            print('Column matched: (column name, column id, question span, start id, end id)')
            print('Exact match:', ', '.join(column_matched_pairs['exact']) if column_matched_pairs['exact'] else 'empty')
            print('Partial match:', ', '.join(column_matched_pairs['partial']) if column_matched_pairs['partial'] else 'empty')
            print('Value match:', ', '.join(column_matched_pairs['value']) if column_matched_pairs['value'] else 'empty')
            print('\n')
            return


        if 'interaction' in entry:
            cell_values = extract_nontext_cell_values(db)
            t_num, c_num = len(table_names), len(column_names)
            prev_qs, prev_sq = np.zeros((0, t_num + c_num), dtype=dtype), np.zeros((t_num + c_num, 0), dtype=dtype)
            qtn, qcn = ENCODER_RELATIONS.index('question-table-nomatch'), ENCODER_RELATIONS.index('question-column-nomatch')
            tqn, cqn = ENCODER_RELATIONS.index('table-question-nomatch'), ENCODER_RELATIONS.index('column-question-nomatch')
            qs_separator, sq_separator = np.array([[qtn] * t_num + [qcn] * c_num], dtype=dtype), np.array([[tqn] * t_num + [cqn] * c_num], dtype=dtype).T
            for turn in entry['interaction']:
                question, question_toks, q_num = turn['utterance'].lower(), turn['utterance_toks'], len(turn['utterance_toks'])
                index_span_pairs = extract_index_phrase_pairs(question_toks)
                q_tab_mat, tab_q_mat, table_matched_pairs = question_schema_matching(index_span_pairs, table_names, table_toks, 'table',
                    matching_infos=(q_num, t_num))
                q_col_mat, col_q_mat, column_matched_pairs = question_cell_matching(index_span_pairs, question, cell_values, turn['value'],
                    matching_infos=(q_num, c_num))
                q_col_mat, col_q_mat, column_matched_pairs = question_schema_matching(index_span_pairs, column_names, column_toks, 'column',
                    matching_infos=(q_col_mat, col_q_mat, column_matched_pairs))
                q_schema = np.concatenate([q_tab_mat, q_col_mat], axis=1)
                schema_q = np.concatenate([tab_q_mat, col_q_mat], axis=0)
                self.accumulate_matchings(q_schema)
                # avoid redundant work on previous utterances
                qs = np.concatenate([q_schema, prev_qs], axis=0)
                sq = np.concatenate([schema_q, prev_sq], axis=1)
                turn['schema_linking'] = (qs.tolist(), sq.tolist())
                prev_qs, prev_sq = np.concatenate([qs_separator, qs], axis=0), np.concatenate([sq_separator, sq], axis=1)

                if verbose: print_matching_infos(question_toks, table_matched_pairs, column_matched_pairs)
        else:
            question, question_toks, q_num = entry['question'].lower(), entry['question_toks'], len(entry['question_toks'])
            index_span_pairs = extract_index_phrase_pairs(question_toks)
            q_tab_mat, tab_q_mat, table_matched_pairs = question_schema_matching(index_span_pairs, table_names, table_toks, 'table',
                matching_infos=(q_num, len(table_names)))
            cell_values = extract_nontext_cell_values(db)
            q_col_mat, col_q_mat, column_matched_pairs = question_cell_matching(index_span_pairs, question, cell_values, entry['value'],
                matching_infos=(q_num, len(column_names)))
            q_col_mat, col_q_mat, column_matched_pairs = question_schema_matching(index_span_pairs, column_names, column_toks, 'column',
                matching_infos=(q_col_mat, col_q_mat, column_matched_pairs))
            q_schema = np.concatenate([q_tab_mat, q_col_mat], axis=1)
            schema_q = np.concatenate([tab_q_mat, col_q_mat], axis=0)
            self.accumulate_matchings(q_schema)
            entry['schema_linking'] = (q_schema.tolist(), schema_q.tolist())

            if verbose: print_matching_infos(question_toks, table_matched_pairs, column_matched_pairs)
        return entry
