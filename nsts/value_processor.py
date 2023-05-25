#coding=utf8
import os, sqlite3, re
import datetime
import jionlp as jio
from typing import List
from fuzzywuzzy import process
from collections import namedtuple, Counter

PLACEHOLDER = '|'

State = namedtuple('State', ['clause', 'agg_op', 'cmp_op', 'unit_op', 'col_id'])

ZH_NUMBER_1 = '零一二三四五六七八九'
ZH_NUMBER_2 = '〇壹贰叁肆伍陆柒捌玖'
ZH_TWO_ALIAS = '两俩'
ZH_NUMBER = ZH_NUMBER_1 + ZH_NUMBER_2 + ZH_TWO_ALIAS
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
    instead of the tokenized lowercased version which may be wrong due to tokenization error (e.g. `bob @ example . org` ).
    Notice that q should be cased version, and if ignore whitespaces, s should occur in q.lower()
    """
    if s in q: return s
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
    return value


def compare_and_extract_date(t1, t2):
    y1, m1, d1 = t1.split('-')
    y2, m2, d2 = t2.split('-')
    if y1 == y2 and m1 == m2:
        return y1 + '-' + m1
    if m1 == m2 and d1 == d2:
        return m1 + '.' + d1 # return m1 + '-' + d1
    if y1 == y2: return y1
    if m1 == m2: return m1
    if d1 == d2: return d1
    return None


def compare_and_extract_time(t1, t2, full=False):
    h1, m1, s1 = t1.split(':')
    h2, m2, s2 = t2.split(':')
    if h1 == h2 and m1 == m2: # ignore second is better
        return h1.lstrip('0') + ':' + m1 if not full else h1 + ':' + m1 + ':00'
    if h1 == h2: # no difference whether ignore second
        return h1.lstrip('0') + ':00' if not full else h1 + ':00:00'
    if m1 == m2 and s1 == s2:
        return m1.lstrip('0') + ':' + s1
    if m1 == m2:
        return m1.lstrip('0') + ':00'
    return None


def extract_time_or_date(result, val):
    # extract date or time string from jio parsed result
    tt, obj = result['type'], result['time']
    if tt in ['time_span', 'time_point']: # compare list
        t1, t2 = obj[0], obj[1]
        t1_date, t1_time = t1.split(' ')
        t2_date, t2_time = t2.split(' ')
        today = datetime.datetime.today()
        if t1_time == '00:00:00' and t2_time == '23:59:59': # ignore time
            if t1_date == t2_date:
                if str(today.year) == t1_date[:4]: # ignore current year prepended
                    return t1_date[5:].replace('-', '.') # return t1_date[5:]
                return t1_date
            else: # find the common part
                return compare_and_extract_date(t1_date, t2_date)
        elif t1_date == t2_date == today.strftime("%Y-%m-%d"): # ignore date
            if '到' not in val: return compare_and_extract_time(t1_time, t2_time)
            else: return '-'.join(t1_time, t2_time)
        else: # preserve both date and time
            date_part = t1_date
            time_part = compare_and_extract_time(t1_time, t2_time, full=True)
            return date_part + '-' + time_part
    elif tt == 'time_delta':
        v = [obj[k] for k in ['year', 'month', 'day', 'hour'] if k in obj]
        if len(v) > 0: # only use one value is enough in DuSQL
            return v[0]
    return None


class ValueProcessor():

    def __init__(self, dataset, tokenizer, db_dir: str = None, eov_token: str = '[SEP]') -> None:
        super(ValueProcessor, self).__init__()
        self.dataset = dataset.lower()
        self.tokenizer, self.db_dir = tokenizer, db_dir
        self.eov_id = self.tokenizer.convert_tokens_to_ids(eov_token)


    def retrieve_column_value_set(self, col_id: int, db: dict):
        if self.db_dir is None or col_id == 0: return []
        db_file = os.path.join(self.db_dir, db['db_id'], db['db_id'] + '.sqlite')
        if not os.path.exists(db_file):
            # print('Cannot find DB file:', db_file)
            return []
        try:
            conn = sqlite3.connect(db_file)
            conn.text_factory = lambda b: b.decode(errors='ignore')
            table_id, column_name = db['column_names_original'][col_id]
            table_name = db['table_names_original'][table_id]
            cursor = conn.execute("SELECT DISTINCT \"%s\" FROM \"%s\";" % (column_name, table_name))
            cell_values = cursor.fetchall()
            cell_values = [each[0] for each in cell_values if str(each[0]).strip() != '']
        except:
            cell_values = []
        finally:
            conn.close()
        return cell_values


    def preprocess(self, sql_value: str, **kwargs):
        sql_value = str(sql_value).strip().strip('"') # remove whitespaces and quotes
        if is_int(sql_value) and (not sql_value.startswith('0') or sql_value.startswith('0.')):
            sql_value = str(int(float(sql_value))) # take care of int values
        ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sql_value)) + [self.eov_id]
        return ids


    def postprocess(self, *args, **kwargs):
        if self.dataset == 'dusql':
            return self.dusql_postprocess(*args, **kwargs)
        elif self.dataset == 'chase':
            return self.chase_postprocess(*args, **kwargs)
        else: return self.spider_postprocess(*args, **kwargs)


    def spider_postprocess(self, token_ids: List[int], state: State, db: dict, entry: dict, **kwargs):
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
                raw_val = re.sub(r'[,\s"\']+', '', raw_val) # deal with numbers
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


    def dusql_postprocess(self, token_ids: List[int], state: State, db: dict, entry: dict, **kwargs):
        if token_ids[-1] == self.eov_id: token_ids = token_ids[:-1]
        raw_toks = self.tokenizer.convert_ids_to_tokens(token_ids)
        raw_val = self.tokenizer.convert_tokens_to_string(raw_toks).strip()
        # tackle the whitespace problems in Chinese PLMs
        raw_val= re.sub(r'([a-zA-Z0-9])\s+([a-zA-Z0-9])', lambda match_obj: match_obj.group(1) + PLACEHOLDER + match_obj.group(2), raw_val)
        raw_val = re.sub(r'\s+', '', raw_val).replace(PLACEHOLDER, ' ')

        clause, col_id = state.clause, state.col_id
        col_type = db['column_types'][col_id]
        if clause == 'limit': # value should be integers
            if is_number(raw_val):
                value = str(int(float(raw_val)))
            else: value = '1' # for all other cases, use 1
        elif clause == 'having': # value should be numbers
            if is_number(raw_val):
                value = str(int(float(raw_val))) if is_int(raw_val) else str(float(raw_val))
            else: value = '1'
        else: # where value
            def word_to_time_or_date(val):
                try:
                    if re.search(r'[\d%s]{1,4}年[\d%s十]{1,2}月[\d%s十]{1,3}$' % (ZH_NUMBER, ZH_NUMBER, ZH_NUMBER), val): val += '日'
                    match_obj = re.search(r'(\d{1,4}[-/]\d{1,2}[-/]\d{1,4})( \d{1,2}:\d{1,2}:\d{1,2})?', val)
                    if match_obj: val = match_obj.group(0)
                    result = jio.parse_time(val)
                    result = extract_time_or_date(result, val)
                    if result is not None:
                        nonlocal value
                        value = result
                        return True
                except: pass
                return False
            
            like_op = 'like' in state.cmp_op
            if col_type == 'number':
                value = str(raw_val).strip('"\'')
            elif col_type == 'time':
                if is_number(raw_val):
                    value = str(raw_val)
                elif word_to_time_or_date(raw_val): pass
                else: value = str(raw_val)
            else: # text values, fuzzy match
                cell_values = self.retrieve_column_value_set(col_id, db)
                if len(cell_values) > 0 and not like_op:
                    most_similar = process.extractOne(raw_val, cell_values)
                    value = most_similar[0] if most_similar[1] > 50 else raw_val
                    # additional attention: the retrieved value must have the same number within
                    if re.sub(r'[^\d]', '', str(value)) != re.sub(r'[^\d]', '', raw_val): value = raw_val
                else: value = str(raw_val)
            # value = str(raw_val)
        return str(value) if is_number(value) else "\"" + value + "\""


    def chase_postprocess(self, token_ids: List[int], state: State, db: dict, entry: dict, **kwargs):
        if token_ids[-1] == self.eov_id: token_ids = token_ids[:-1]
        raw_toks = self.tokenizer.convert_ids_to_tokens(token_ids)
        raw_val = self.tokenizer.convert_tokens_to_string(raw_toks).strip()
        # tackle the whitespace problems in Chinese PLMs
        raw_val= re.sub(r'([a-zA-Z0-9])\s+([a-zA-Z0-9])', lambda match_obj: match_obj.group(1) + PLACEHOLDER + match_obj.group(2), raw_val)
        raw_val = re.sub(r'\s+', '', raw_val).replace(PLACEHOLDER, ' ')

        clause, col_id = state.clause, state.col_id
        col_type = db['column_types'][col_id]
        if clause == 'limit': # value should be integers
            if is_number(raw_val):
                value = str(int(float(raw_val)))
            else: value = '1' # for all other cases, use 1
        elif clause == 'having': # value should be numbers
            if is_number(raw_val):
                value = str(int(float(raw_val))) if is_int(raw_val) else str(float(raw_val))
            else: value = '1'
        else: # WHERE value, TODO: use fuzzymatch
            like_op = 'like' in state.cmp_op
            if col_type == 'number':
                value = str(raw_val).strip('"\'')
            else: # text values, fuzzy match
                cell_values = self.retrieve_column_value_set(col_id, db)
                if len(cell_values) > 0 and not like_op:
                    most_similar = process.extractOne(raw_val, cell_values)
                    value = most_similar[0] if most_similar[1] > 50 else raw_val
                    # additional attention: the retrieved value must have the same number within
                    if re.sub(r'[^\d]', '', str(value)) != re.sub(r'[^\d]', '', raw_val): value = raw_val
                else: value = str(raw_val)
        return str(value) if is_number(value) else "\"" + value + "\""