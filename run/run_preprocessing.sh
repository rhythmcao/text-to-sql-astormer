#!/bin/bash

plm=electra-small-discriminator
encode_method=rgatsql

python3 -c "import json; json.dump(json.load(open('data/spider/train_spider.json', 'r')) + json.load(open('data/spider/train_others.json', 'r')), open('data/spider/train.json', 'w'), ensure_ascii=False, indent=4)"
python3 -u asdl/parse_sql_to_json.py -d spider -s all
python3 -u asdl/parse_sql_to_json.py -d sparc -s all
python3 -u asdl/parse_sql_to_json.py -d cosql -s all
python3 -u preprocess/data_preprocess.py -d spider -t $plm -e $encode_method -s all
python3 -u preprocess/data_preprocess.py -d sparc -t $plm -e $encode_method -s all
python3 -u preprocess/data_preprocess.py -d cosql -t $plm -e $encode_method -s all
