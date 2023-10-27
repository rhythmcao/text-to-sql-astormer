#coding=utf8
""" Convert data format of DuSQL/Chase, including:
1. Tables: db_schema.json -> tables.json
    fix two errors: invalid db_id 世博会/园博会 and duplicate column_names 中国交通->火车站->投用日期
    add special column TIME_NOW after *, which is frequently used in samples
2. Database: db_content.json -> database/*.sqlite
    change db_content.json to separate .sqlite files under database directory
3. Dataset:
    db_id fix about 世博会/园博会
    cmp operator == -> =
"""
import os, sys, json, shutil, sqlite3
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nsts.transition_system import CONFIG_PATHS


def convert_dusql_tables(path='data/dusql/db_schema.json'):
    with open(path, 'r') as inf:
        table_list = json.load(inf)
        table_names, tables = set(), []
        for table in table_list:
            if table['db_id'] in table_names: continue
            tables.append(table)
            table_names.add(table['db_id'])

    for db in tables:
        if 'table_names_original' not in db:
            db['table_names_original'] = db['table_names']
        if 'column_names_original' not in db:
            db['column_names_original'] = db['column_names']
        if db['db_id'] == '世博会/园博会': db['db_id'] = '世博会和园博会'
        if db['db_id'] == '中国交通':
            db['column_names'][18] = [1, '投用日期备份']
            db['column_names_original'][18] = [1, '投用日期备份']

        # add special column TIME_NOW after *
        db['column_names'] = [[-1, '*'], [-1, 'TIME_NOW']] + db['column_names'][1:]
        db['column_names_original'] = [[-1, '*'], [-1, 'TIME_NOW']] + db['column_names_original'][1:]
        db['column_types'] = ['text', 'time'] + db['column_types'][1:]
        db['primary_keys'] = [k + 1 for k in db['primary_keys']]
        db['foreign_keys'] = [[x + 1, y + 1] for x, y in db['foreign_keys']]

    with open(CONFIG_PATHS['dusql']['tables'], 'w') as of:
        json.dump(tables, of, ensure_ascii=False, indent=4)
    return tables


def convert_dusql_database(path='data/dusql/db_content.json'):
    with open(path, 'r') as inf:
        dbs = json.load(inf)
    output_dir = CONFIG_PATHS['dusql']['db_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for db in dbs:
        db_id = db['db_id']
        if db_id == '世博会/园博会': db_id = '世博会和园博会'
        if db_id == '中国交通':
            db['tables']['火车站']['header'][-1] = '投用日期备份'

        output_path = os.path.join(output_dir, db_id)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        db_path = os.path.join(output_path, db_id + '.sqlite')
        if os.path.exists(db_path): # delete already existing .sqlite file
            os.remove(db_path)

        con = sqlite3.connect(db_path)
        cur = con.cursor()
        for table_name in db['tables']:
            table = db['tables'][table_name]
            # create tables command (ignore primary/foreign keys and invalid column types such as number)
            schema = ', '.join(['"' + col + '" ' + tp for col, tp in zip(table['header'], table['type'])])
            command = f'CREATE TABLE "{table_name}"({schema}) ;'
            cur.execute(command)
            # insert db content commands
            placeholder = ', '.join(['?'] * len(table['header']))
            command = f'INSERT INTO "{table_name}" VALUES ({placeholder}) ;'
            cur.executemany(command, table['cell'])
        con.commit()
        con.close()
    return


def convert_dusql_dataset():
    for split in ['train', 'dev']:
        path = CONFIG_PATHS['dusql'][split]
        with open(path, 'r') as inf:
            data = json.load(inf)

        for ex in data:
            ex['query'] = ex['query'].replace('==', '=')
            if ex['db_id'] == '世博会/园博会':
                ex['db_id'] = '世博会和园博会'
        with open(path, 'w') as of:
            json.dump(data, of, indent=4, ensure_ascii=False)
    return


def convert_chase_database(path='data/chase/database'):
    files = os.listdir(path)
    for f in files:
        if not f.endswith('.sqlite'): continue
        prefix = os.path.splitext(f)[0]
        output_path = os.path.join(path, prefix)
        if not os.path.exists(output_path) or not os.path.isdir(output_path):
            os.makedirs(output_path)

        original_file = os.path.join(path, f)
        output_file = os.path.join(output_path, f)
        if os.path.exists(output_file):
            os.remove(output_file)
        shutil.move(original_file, output_file)
    return


if __name__ == '__main__':

    convert_dusql_tables()

    convert_dusql_database()

    convert_dusql_dataset()

    convert_chase_database()
