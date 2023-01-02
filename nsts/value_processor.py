import os, sqlite3, re
from typing import List
from fuzzywuzzy import process
from collections import namedtuple, Counter


State = namedtuple('State', ['clause', 'agg_op', 'cmp_op', 'unit_op', 'col_id'])


BOOL_TRUE = ['Y', 'y', 'T', 't', '1', 1, 'yes', 'Yes', 'true', 'True', 'YES', 'TRUE']
BOOL_FALSE = ['N', 'n', 'F', 'f', '0', 0, 'no', 'No', 'false', 'False', 'NO', 'FALSE']


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_int(s):
    if is_number(s) and float(s) % 1 == 0:
        return True
    return False


def extract_raw_question_span(s: str, q: str):
    """ During postprocessing, if the SQL value happen to appear in the raw question,
    instead of the tokenized version which may be wrong due to tokenization error (e.g. `bob @ example . org` ).
    Notice that q should be cased version, and if ignore whitespaces, s should occur in q.lower()
    """
    q = re.sub(r'\s+', ' ', q)
    if s in q.lower(): # preserve upper/lower case
        start_id = q.lower().index(s)
        return q[start_id: start_id + len(s)]
    s_, q_ = s.replace(' ', ''), q.replace(' ', '')
    index_mapping = [idx for idx, c in enumerate(q) if c != ' ']
    try:
        start_id = q_.lower().index(s_)
        start, end = index_mapping[start_id], index_mapping[start_id + len(s_) - 1]
        return q[start: end + 1]
    except: return s


def try_fuzzy_matching(raw_value: str, cell_values: List[str], question: str, score: int = 85):
    if len(cell_values) == 0:
        value = extract_raw_question_span(raw_value, question)
    else:
        cell_values = [str(v) for v in cell_values]
        # fuzzy match, choose the most similar cell value above the threshold from the DB
        matched_value, matched_score = process.extractOne(raw_value, cell_values)
        if matched_score >= score: value = matched_value
        else: value = extract_raw_question_span(raw_value, question)
    return value.strip()


class ValueProcessor():
    
    def __init__(self, tokenizer, db_dir: str = None, eov_token: str = '[SEP]') -> None:
        super(ValueProcessor, self).__init__()
        self.tokenizer, self.db_dir = tokenizer, db_dir
        self.eov_id = self.tokenizer.convert_tokens_to_ids(eov_token)


    def retrieve_column_value_set(self, col_id: int, db: dict):
        if self.db_dir is None or col_id == 0: return []
        db_file = os.path.join(self.db_dir, db['db_id'], db['db_id'] + '.sqlite')
        if not os.path.exists(db_file):
            print('Cannot find DB file:', db_file)
            return []
        conn = sqlite3.connect(db_file)
        conn.text_factory = lambda b: b.decode(errors='ignore')
        table_id, column_name = db['column_names_original'][col_id]
        table_name = db['table_names_original'][table_id]
        cursor = conn.execute("SELECT DISTINCT \"%s\" FROM \"%s\";" % (column_name, table_name))
        cell_values = cursor.fetchall()
        cell_values = [each[0] for each in cell_values if str(each[0]).strip().lower() not in ['', 'null', 'none']]
        conn.close()
        return cell_values


    def preprocess(self, sql_value: str, **kwargs):
        sql_value = str(sql_value).strip().strip('"') # remove whitespaces and quotes
        if is_int(sql_value) and (not sql_value.startswith('0') or sql_value.startswith('0.')):
            sql_value = str(int(float(sql_value))) # take care of int values
        ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sql_value)) + [self.eov_id]
        return ids


    def postprocess(self, token_ids: List[int], state: State, db: dict, entry: dict, **kwargs):
        if token_ids[-1] == self.eov_id: token_ids = token_ids[:-1]
        raw_toks = self.tokenizer.convert_ids_to_tokens(token_ids)
        raw_val = self.tokenizer.convert_tokens_to_string(raw_toks).strip()

        clause, col_id = state.clause, state.col_id

        if clause in ['limit', 'having']:
            raw_val = re.sub(r'[,\s"\']+', '', raw_val)
            if is_int(raw_val): value = str(int(float(raw_val)))
            else: value = "1" if clause == 'limit' else str(raw_val) if is_number(raw_val) else '"' + str(raw_val) + '"'
        else: # where clause, take care of whitespaces and upper/lower cases
            raw_question = entry['question'] if 'question' in entry else entry['utterance']
            col_type = db['column_types'][col_id]
            agg_op, cmp_op = state.agg_op.lower(), state.cmp_op.lower()
            like_op = 'like' in cmp_op

            do_lower_case = self.tokenizer.do_lower_case if hasattr(self.tokenizer, 'do_lower_case') else False

            # for most generative PLM which use BPE tokenizer and is case-sensitive
            if not do_lower_case:
                return '"' + raw_val.strip() + '"'

            # for lowercased BERT/ELECTRA series autoencoder PLM which use WordPiece tokenzier
            # need to recover the case information and pay attention to whitespaces according to the DB or input question
            if not like_op and (col_type == 'number' or agg_op in ['count', 'sum', 'avg']):
                raw_val = re.sub(r'[,\s"\']+', '', raw_val) # remove redundant whitespaces
                value = str(int(float(raw_val))) if is_int(raw_val) and not raw_val.startswith('0') else str(raw_val)
            elif not like_op and col_type == 'time': # remove whitespaces before and after hyphen - , : or /
                value = re.sub(r'\s*(-|:|/)\s*', lambda match: match.group(1), raw_val).strip()
            elif col_type == 'boolean':
                evidence = []
                column_values = self.retrieve_column_value_set(col_id, db)
                for cv in column_values:
                    for idx, (t, f) in enumerate(zip(BOOL_TRUE, BOOL_FALSE)):
                        if cv == t or cv == f:
                            evidence.append(idx)
                            break
                if len(evidence) > 0 and raw_val in BOOL_TRUE + BOOL_FALSE:
                    bool_idx = Counter(evidence).most_common(1)[0][0]
                    value = BOOL_TRUE[bool_idx] if raw_val in BOOL_TRUE else BOOL_FALSE[bool_idx]
                else: value = str(int(float(raw_val))) if is_number(raw_val) else str(raw_val).upper() if len(raw_val) == 1 else raw_val
            else: # text type
                if like_op:
                    raw_val = re.sub(r'([^a-zA-Z])\s+([^a-zA-Z])', lambda match_obj: match_obj.group(1) + match_obj.group(2), raw_val)
                    ls = '%' if raw_val.startswith('%') else ''
                    rs = '%' if raw_val.endswith('%') else ''
                    if len(ls + rs) == 0: ls, rs = '%', '%'
                    raw_val = raw_val.strip('%').strip()
                    value = ls + extract_raw_question_span(raw_val, raw_question) + rs
                elif re.search(r'\d+', raw_val): value = extract_raw_question_span(raw_val, raw_question)
                else:
                    column_values = self.retrieve_column_value_set(col_id, db)
                    value = try_fuzzy_matching(raw_val, column_values, raw_question)
            value = '"' + str(value).strip() + '"'

        return value