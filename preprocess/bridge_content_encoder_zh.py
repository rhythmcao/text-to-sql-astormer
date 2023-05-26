#coding=utf8
import os, sys, math, re, sqlite3
import functools
from rouge import Rouge

MAX_CELL_NUM = 4

REPLACEMENT = dict(zip('０１２３４５６７８９％：．～幺（）：％‘’\'`—', '0123456789%:.~一():%“”""-'))

NORM = lambda s: re.sub(r'[０１２３４５６７８９％：．～幺（）：％‘’\'`—]', lambda c: REPLACEMENT[c.group(0)], s)

STOPWORDS = set(['哦', '哪位', '它', '后', '呗', '多大', '它们', '叫', '按', '其他', '低', '各', '吧', '这个', '不足', '不到', '前', '及', '比',
        '都是', '需要', '只', '之后', '给', '各个', '麻烦', '几个', '数', '中', '这边', '他', '有', '了', '跟', '那本书', '那么', '她们', 
        '一共', '这本书', '得', '她', '怎么', '知道', '人', '诶', '能不能', '还有', '并', '问一下', '是', '没有', '算算', '这种', '且', 
        '高于', '以下', '说说', '种', '不止', '这本', '现在', '查询', '该', '少于', '分别', '哪家', '包含', '多于', '的话', '起来', '上', 
        '目前有', '多少年', '对应', '呃', '你', '有没有', '长', '少', '在', '做', '如何', '之', '来自', '每个', '额', '才能', '多多少', 
        '超过', '可以', '拥有', '什么', '以及', '可', '个', '为', '被', '那些', '几家', '呀', '时候', '到', '以上', '他们', '哪一个', 
        '并且', '每一个', '还是', '你好', '多少', '和', '要', '值', '目前', '从', '于', '进行', '里', '谁', '想问问', '这', '但', '能', 
        '而且', '那个', '找', '至少', '了解到', '与', '所', '高', '查', '正好', '请', '或者', '哪些', '就是', '看', '呢', '时', '想知道', 
        '等于', '总共', '哪个', '用', '或', '去', '谢谢', '怎样', '由于', '找出', '了解', '小于', '几本', '但是', '这样', '其', '那里', '超', 
        '帮', '要求', '那', '看一下', '就', '一下', '查查', '问', '又', '吗', '话', '这次', '由', '占', '加', '想', '啥', '当', '也', '给出', 
        '含', '是什么', '过', '属于', '多少次', '按照', '看看', '及其', '告知', '多', '这些', '已经', '找到', '什么时候', '不在', '告诉我', '都', 
        '怎么样', '下', '低于', '哪', '还', '所在', '不是', '达到', '不', '同时', '看到', '请问', '哪几个', '啊', '如果', '所有', '这里', 
        '哪里', '这部', '想要', '的', '我', '大于', '不少', '没'])


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def float_equal(val1, val2, multiplier=1):
    val1, val2 = float(val1), float(val2)
    if math.fabs(val1 - val2) < 1e-5: return True
    elif math.fabs(val1 * multiplier - val2) < 1e-5 or math.fabs(val1 - val2 * multiplier) < 1e-6: return True
    return False


@functools.lru_cache(maxsize=2000, typed=False)
def get_column_picklist(table_name: str, column_name: str, db_path: str) -> list:
    fetch_sql = "SELECT DISTINCT `{}` FROM `{}`".format(column_name, table_name)
    picklist = set()
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode(errors='ignore')
        cursor = conn.cursor()
        cell_values = cursor.fetchall()
        picklist = [each[0] for each in cell_values if str(each[0]).strip() != '']
        conn.close()
    except:
        # print('Error while connecting database in path %s' % (db_path))
        pass
    return list(picklist)


ROUGE = Rouge(metrics=["rouge-1", "rouge-l"])
ROUGE_SCORE = lambda pred, ref: ROUGE.get_scores(' '.join(list(pred)), ' '.join(list(ref)))[0]



def get_database_matches(self, question: str, col_id: int, db: dict):
    # extract candidate cell values for each column given the current question
    tid, cname = db['column_names_original'][col_id][1]
    tname = db['table_names_original'][tid]
    cells = self.retrieve_cell_values(cname, tname, db['db_id'])
    col_types = db['column_types']

    question_toks = entry['uncased_question_toks']
    numbers = extract_numbers_in_question(''.join(question_toks))
    question = ''.join(filter(lambda s: s not in self.stopwords, question_toks))

    def number_score(c, numbers):
        if (not is_number(c)) or len(numbers) == 0: return 0.
        return max([1. if float_equal(c, r) else 0.5 if float_equal(c, r, 100) or float_equal(c, r, 1e3) or float_equal(c, r, 1e4) or float_equal(c, r, 1e8) else 0. for r in numbers])

    candidates = [[]] # map column_id to candidate values relevant to the question
    for col_id, col_cells in enumerate(cells):
        if col_id == 0: continue
        tmp_candidates = []
        for c in col_cells:
            if c.startswith('item_') and c in entry['item_mapping']:
                tmp_candidates.append(entry['item_mapping'][c])
        if len(tmp_candidates) > 0:
            candidates.append(tmp_candidates)
            continue
        if col_types[col_id] == 'binary': candidates.append([])
        elif col_types[col_id] == 'time':
            candidates.append([c for c in col_cells if c in question][:MAX_CELL_NUM])
        elif col_types[col_id] == 'number':
            scores = sorted(filter(lambda x: x[1] > 0, [(cid, number_score(c, numbers)) for cid, c in enumerate(col_cells)]), key=lambda x: - x[1])[:MAX_CELL_NUM]
            if len(scores) > 1:
                scores = scores[:1] + list(filter(lambda x: x[1] >= 0.6, scores[1:]))
            candidates.append([normalize_cell_value(raw_cells[col_id][cid]) for cid, _ in scores])
        else: # by default, text
            scores = [(c, self.rouge_score(c, question)) for c in col_cells if 0 < len(c) < 50 and c != '.']
            scores = sorted(filter(lambda x: x[1]['rouge-l']['f'] > 0, scores), key=lambda x: (- x[1]['rouge-l']['f'], - x[1]['rouge-1']['p']))[:MAX_CELL_NUM]
            if len(scores) > 1: # at most two cells but the second one must have high rouge-1 precision
                scores = scores[:1] + list(filter(lambda x: x[1]['rouge-1']['p'] >= 0.6, scores[1:]))
            candidates.append([c for c, _ in scores])
    return candidates