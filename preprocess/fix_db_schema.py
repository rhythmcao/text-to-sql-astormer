#coding=utf8
from typing import Dict, List, Tuple
import sys, json, os, sqlite3, re, collections, itertools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nsts.transition_system import CONFIG_PATHS


DUSQL_COLUMN_TYPES: Dict[str, List[Tuple[int, str]]] = {
    '中国传统节日': [(5, 'time')],
    '中国戏剧': [(6, 'time')],
    '智能手机全球占比': [(-3, 'text')],
    '互联网企业': [(7, 'text')],
    '世界湖泊': [(8, 'text'), (12, 'text')],
    '酒店预订': [(-3, 'text')],
    '社交软件': [(10, 'number')],
    '空调': [(24, 'number')],
    '智能音箱': [(14, 'number')],
    '中国文学奖': [(11, 'text'), (14, 'number')],
    '中国菜系': [(7, 'binary'), (8, 'binary')],
    '教材辅助参考书': [(11, 'number')],
    '澳网公开赛': [(10, 'time')],
    '植物经济价值': [(18, 'text')],
    '笔记本电脑': [(26, 'text')],
    '诺贝尔奖项': [(19, 'text')],
    '中国高校': [(11, 'text')],
    '大洲与国家': [(33, 'text'), (34, 'text')],
    '各城市院士情况': [(3, 'text')],
    '城市拥堵': [(14, 'text')],
    '医院': [(21, 'text')],
    '地震': [(25, 'text'), (26, 'time')],
    '中国演员和电影': [(15, 'text')],
    '中国城市潜力': [(19, 'text')],
    '中国宜居城市': [(5, 'text'), (6, 'text'), (10, 'text'), (11, 'text'), (15, 'text'), (16, 'text')]
}

SPIDER_COLUMN_TYPES: Dict[str, List[Tuple[int, str]]] = {
    'coffee_shop': [(3, 'number'), (4, 'time')],
    'apartment_rentals': [(14, 'number')],
    'customers_and_invoices': [(-5, 'number'), (-11, 'number')],
    'formula_1': [(59, 'number'), (-8, 'number'), (16, 'time')],
    'news_report': [(9, 'number')],
    'music_1': [(11, 'number')],
    'department_store': [(-7, 'number'), (13, 'number')],
    'customers_campaigns_ecommerce': [(-1, 'number')],
    'cre_Drama_Workshop_Groups': [(75, 'number'), (81, 'number')],
    'customers_and_addresses': [(-1, 'number')],
    'customers_and_products_contacts': [(-1, 'number')],
    'shop_membership': [(8, 'time'), (11, 'number'), (14, 'time'), (17, 'time')],
    'tracking_share_transactions': [(13, 'number')],
    'roller_coaster': [(7, 'number')],
    'ship_1': [(4, 'number')],
    'e_government': [(27, 'number')],
    'driving_school': [(-2, 'time')],
    'geo': [(14, 'number'), (17, 'number')],
    'car_1': [(-7, 'number'), (-4, 'number')],
    'dog_kennels': [(26, 'number')],
    'orchestra': [(-6, 'number')],
    'product_catalog': [(-8, 'number'), (-7, 'number'), (-6, 'number'), (-5, 'number')],
    'wrestler': [(4, 'number')],
    'party_people': [(9, 'time'), (10, 'time')],
    'party_host': [(-4, 'number')],
    'products_gen_characteristics': [(-6, 'number'), (-7, 'number')],
    'concert_singer': [(-3, 'time')],
    'museum_visit': [(4, 'time')],
    'cre_Docs_and_Epenses': [(-1, 'number')]
}


def fix_dusql_column_types(tables_list):
    for db in tables_list:
        if db['db_id'] in DUSQL_COLUMN_TYPES:
            for (cid, tp) in DUSQL_COLUMN_TYPES[db['db_id']]:
                db['column_types'][cid] = tp
    return tables_list


def fix_dusql_primary_and_foreign_keys(tables_list):
    pass
    return tables_list


def fix_spider_column_types(tables_list: List[dict], db_dir: str = 'data/spider/database'):
    """ Search and replace boolean types.
    """
    for db in tables_list:
        db_id, tables = db['db_id'], db['table_names_original']
        columns, column_types = db['column_names_original'], db['column_types']
        for cid, (tid, cname) in enumerate(columns):
            if column_types[cid] in ['boolean', 'number', 'time'] or cname == '*' or 'gender' in cname.lower() or 'sex' in cname.lower(): continue
            # according to column names
            if re.search(r'^([Ii][Ff])[_ A-Z]', cname) or re.search(r'^([Ii][Ss])[_ A-Z]', cname) \
                or re.search(r'[_ ]([yY][nN])$', cname) or re.search(r'[_ ]([tT][fF])$', cname):
                db['column_types'][cid] = 'boolean'
                continue
            # according to cell values
            db_path = os.path.join(db_dir, db_id, db_id + '.sqlite')
            try:
                tname = tables[tid]
                conn = sqlite3.connect(db_path)
                conn.text_factory = lambda b: b.decode(errors='ignore')
                cursor = conn.execute("PRAGMA table_info('{}') ".format(tname))
                column_infos = list(cursor.fetchall())
                ctype = [infos[2].lower() for infos in column_infos if infos[1].lower() == cname.lower()]
                ctype = 'text' if len(ctype) == 0 else ctype[0]
                cursor = conn.execute('SELECT DISTINCT \"%s\" FROM \"%s\" ;' % (cname, tname))
                values = list(cursor.fetchall())
                values = set([str(v[0]).lower() for v in values if str(v[0]).strip() != ''])
                conn.close()
            except:
                ctype, values = 'text', set()
            evidence_set = [{'y', 'n'}, {'yes', 'no'}, {'t', 'f'}, {'true', 'false'}]
            if ctype == 'bool': db['column_types'][cid] = 'boolean'
            elif len(values) > 0 and any(len(values - s) == 0 for s in evidence_set): db['column_types'][cid] = 'boolean'
            elif len(values) > 0 and len(values - {'0', '1'}) == 0 and all(v not in cname.lower() for v in ['value', 'code', 'id']): db['column_types'][cid] = 'boolean'

    for db in tables_list:
        if db['db_id'] in SPIDER_COLUMN_TYPES:
            for (cid, tp) in SPIDER_COLUMN_TYPES[db['db_id']]:
                db['column_types'][cid] = tp
    return tables_list


def fix_spider_primary_and_foreign_keys(tables_list):
    for table in tables_list:
        if len(table['column_names']) > 100: continue

        # add primary keys
        pks, fks = table['primary_keys'], table['foreign_keys']
        pk_tables = set([table['column_names'][c][0] for c in pks])
        candidates = set([pair[1] for pair in fks if pair[1] not in pks and table['column_names'][pair[1]][0] not in pk_tables])
        table['primary_keys'] = sorted(pks + list(candidates))

        # add foreign keys
        column_sets = collections.defaultdict(list)
        for cid, (tid, cname) in enumerate(table['column_names_original']):
            column_sets[cname.lower()].append((tid, cid))
        primary_keys = table['primary_keys']
        foreign_keys = set([tuple(x) for x in table['foreign_keys']])
        for cname in column_sets:
            if cname in ['*', 'name', 'id', 'code']: continue
            if len(column_sets[cname]) > 1: # existing columns that have the same name
                for (p_t, p), (q_t, q) in itertools.combinations(column_sets[cname], 2):
                    if p_t == q_t: continue # columns from the same table, skip
                    if p in primary_keys and q not in primary_keys:
                        if ((p, q) in foreign_keys) and ((q, p) not in foreign_keys):
                            foreign_keys.remove((p, q))
                            foreign_keys.add((q, p))
                        elif (q, p) not in foreign_keys:
                            connect_tables = set(map(lambda k: table['column_names'][k[1]][0] if k[0] == p else table['column_names'][k[0]][0], filter(lambda k: k[0] == p or k[1] == p, foreign_keys)))
                            if connect_tables and q_t not in connect_tables:
                                foreign_keys.add((q, p))
                    elif q in primary_keys and p not in primary_keys:
                        if ((q, p) in foreign_keys) and ((p, q) not in foreign_keys):
                            foreign_keys.remove((q, p))
                            foreign_keys.add((p, q))
                        elif (p, q) not in foreign_keys:
                            connect_tables = set(map(lambda k: table['column_names'][k[1]][0] if k[0] == q else table['column_names'][k[0]][0], filter(lambda k: k[0] == q or k[1] == q, foreign_keys)))
                            if connect_tables and p_t not in connect_tables:
                                foreign_keys.add((p, q))
        foreign_keys = sorted(foreign_keys, key=lambda x: (x[0], x[1]))
        table['foreign_keys'] = foreign_keys
    return tables_list


if __name__ == '__main__':

    tables = json.load(open(CONFIG_PATHS['dusql']['tables'], 'r'))
    tables = fix_dusql_column_types(tables)
    tables = fix_dusql_primary_and_foreign_keys(tables)
    json.dump(tables, open(CONFIG_PATHS['dusql']['tables'], 'w'), indent=4, ensure_ascii=False)

    tables = json.load(open(CONFIG_PATHS['spider']['tables'], 'r'))
    tables = fix_spider_column_types(tables, db_dir=CONFIG_PATHS['spider']['db_dir'])
    tables = fix_spider_primary_and_foreign_keys(tables)
    json.dump(tables, open(CONFIG_PATHS['spider']['tables'], 'w'), indent=4, ensure_ascii=False)
