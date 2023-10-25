#!/bin/bash

plm=electra-small-discriminator
zh_plm=chinese-electra-180g-small-discriminator
encode_method=rgatsql

python3 -c "import json; json.dump(json.load(open('data/spider/train_spider.json', 'r')) + json.load(open('data/spider/train_others.json', 'r')), open('data/spider/train.json', 'w'), ensure_ascii=False, indent=4)"
python3 -u nsts/parse_sql_to_json.py -d spider -s all
python3 -u nsts/parse_sql_to_json.py -d spider -s dev_ext
python3 -u nsts/parse_sql_to_json.py -d sparc -s all
python3 -u nsts/parse_sql_to_json.py -d cosql -s all

python3 -u preprocess/convert_data_format.py
python3 -u nsts/parse_sql_to_json.py -d dusql -s all
python3 -u nsts/parse_sql_to_json.py -d chase -s all

python3 -u preprocess/data_preprocess.py -d spider -t $plm -e $encode_method -s all
python3 -u preprocess/data_preprocess.py -d spider -t $plm -e $encode_method -s dev_ext
python3 -u preprocess/data_preprocess.py -d sparc -t $plm -e $encode_method -s all
python3 -u preprocess/data_preprocess.py -d cosql -t $plm -e $encode_method -s all

python3 -u preprocess/data_preprocess.py -d dusql -t $zh_plm -e $encode_method -s all
python3 -u preprocess/data_preprocess.py -d chase -t $zh_plm -e $encode_method -s all
