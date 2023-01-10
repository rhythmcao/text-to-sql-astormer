#!/bin/bash

plm=roberta-base
encode_method=none

python3 -u nsts/parse_sql_to_json.py -d spider -s all
python3 -u nsts/parse_sql_to_json.py -d sparc -s all
python3 -u nsts/parse_sql_to_json.py -d cosql -s all
python3 -u preprocess/data_preprocess.py -d spider -t $plm -e $encode_method -s all
python3 -u preprocess/data_preprocess.py -d sparc -t $plm -e $encode_method -s all
python3 -u preprocess/data_preprocess.py -d cosql -t $plm -e $encode_method -s all
