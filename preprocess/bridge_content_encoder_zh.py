#coding=utf8
import sqlite3
import functools
from rouge import Rouge

MAX_CELL_NUM = 4


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


@functools.lru_cache(maxsize=2000, typed=False)
def get_column_picklist(table_name: str, column_name: str, db_path: str) -> list:
    fetch_sql = "SELECT DISTINCT `{}` FROM `{}`".format(column_name, table_name)
    picklist = set()
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode(errors='ignore')
        cursor = conn.cursor()
        cursor.execute(fetch_sql)
        cell_values = cursor.fetchall()
        picklist = [str(each[0]).strip() for each in cell_values if str(each[0]).strip() != '']
        conn.close()
    except:
        # print('Error while connecting database in path %s' % (db_path))
        pass
    return list(picklist)


ROUGE = Rouge(metrics=["rouge-1", "rouge-l"])
ROUGE_SCORE = lambda pred, ref: ROUGE.get_scores(' '.join(list(pred)), ' '.join(list(ref)))[0]


def get_database_matches_zh(question: str, tab_name: str, col_name: str, db_file: str, col_type='text'):
    cells = get_column_picklist(tab_name, col_name, db_file)
    if col_type == 'binary': candidates = []
    elif col_type in ['time', 'number']:
        candidates = [c for c in cells if c.lower() in question.lower()][:MAX_CELL_NUM]
    else: # by default, text
        scores = [(c, ROUGE_SCORE(c, question)) for c in cells if 0 < len(c) < 50 and c != '.']
        scores = sorted(filter(lambda x: x[1]['rouge-l']['f'] >= 0.1, scores), key=lambda x: (- x[1]['rouge-l']['f'], - x[1]['rouge-1']['p']))[:MAX_CELL_NUM]
        if len(scores) > 1: # at most two cells but the following one must have high rouge-1 precision
            scores = scores[:1] + list(filter(lambda x: x[1]['rouge-1']['p'] >= 0.6, scores[1:]))
        candidates = [c for c, _ in scores]
    return candidates