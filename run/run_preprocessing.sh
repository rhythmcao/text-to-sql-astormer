#!/bin/bash

dataset=spider
plm=grappa_large_jnt
encode_method=none

python3 -u nsts/parse_sql_to_json.py -d $dataset -s all
python3 -u preprocess/data_preprocess.py -d $dataset -t $plm -e $encode_method -s all