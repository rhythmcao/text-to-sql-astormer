#coding=utf8
import os, json, argparse, sys, time
from typing import List
from transformers import AutoTokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nsts.transition_system import CONFIG_PATHS
from preprocess.preprocess_utils import PreProcessor


def process_tables(processor: PreProcessor, tables_list: List[dict], output_path: str = None, verbose: bool = False):
    tables = []
    for each in tables_list:
        print('*************** Processing database %s **************' % (each['db_id']))
        tables.append(processor.preprocess_database(each))
    print('In total, process %d databases .' % (len(tables)))
    if output_path is not None:
        json.dump(tables, open(output_path, 'w'), ensure_ascii=False, indent=4)
    return tables


def process_dataset_input(processor: PreProcessor, dataset: List[dict], tables: List[dict], output_path: str = None, skip_large: bool = False, verbose: bool = False):
    processed_dataset = []
    tables = {db['db_id']: db for db in tables}
    processor.clear_statistics()
    for idx, entry in enumerate(dataset):
        db_id = entry['db_id'] if 'db_id' in entry else entry['database_id']
        if skip_large and len(tables[db_id]['column_names']) > 100: continue
        if (idx + 1) % 500 == 0:
            print('*************** Processing inputs of the %d-th sample **************' % (idx + 1))

        # tokenize question, perform schema linking and value linking
        entry = processor.pipeline(entry, tables[db_id], verbose=verbose)
        processed_dataset.append(entry)

    print('Table: partial match %d ; exact match %d' % (processor.matches['table']['partial'], processor.matches['table']['exact']))
    print('Column: partial match %d ; exact match %d ; value match %d' % (processor.matches['column']['partial'], processor.matches['column']['exact'], processor.matches['column']['value']))
    print('Bridge count: %d' % (processor.bridge_value))
    print('In total, process %d samples , skip %d extremely large databases.' % (len(processed_dataset), len(dataset) - len(processed_dataset)))

    if output_path is not None: # serialize preprocessed dataset
        json.dump(processed_dataset, open(output_path, 'w'), ensure_ascii=False, indent=4)
    return processed_dataset


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-d', dest='dataset', type=str, required=True, choices=['spider', 'sparc', 'cosql', 'dusql', 'chase'])
    arg_parser.add_argument('-t', dest='tokenizer', type=str, default='grappa_large_jnt', help='PLM name used for the tokenizer and vocabulary')
    arg_parser.add_argument('-s', dest='data_split', type=str, default='all', choices=['train', 'dev', 'all'], help='dataset path')
    arg_parser.add_argument('-e', dest='encode_method', type=str, default='none', choices=['none', 'rgatsql'], help='encode method')
    args = arg_parser.parse_args()

    db_dir = CONFIG_PATHS[args.dataset]['db_dir']
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(CONFIG_PATHS['plm_dir'], args.tokenizer), add_prefix_space=True)
    processor = PreProcessor(args.dataset, tokenizer, db_dir=db_dir, encode_method=args.encode_method)

    table_path = CONFIG_PATHS[args.dataset]['tables']
    tables = json.load(open(table_path, 'r'))
    tables = process_tables(processor, tables, output_path=table_path)

    data_split = ['train', 'dev'] if args.data_split == 'all' else [args.data_split]
    for split in data_split:
        start_time = time.time()
        dataset_path = CONFIG_PATHS[args.dataset][split]
        dataset = json.load(open(dataset_path, 'r'))
        dataset = process_dataset_input(processor, dataset, tables, output_path=dataset_path, skip_large=(split=='train'))
        print('%s dataset preprocessing costs %.4fs .' % (split, time.time() - start_time))