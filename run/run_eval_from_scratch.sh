#!/bin/bash

read_model_path= # directory which stores the model.bin and params.json, e.g., exp/task_astormer/electra-transformer/
#db_dir= # directory path to the database, e.g., data/spider/database-testsuite
#table_path= # file path to tables.json, e.g., data/spider/tables.json
#dataset_path= # file path to dev.json or test.json, e.g., data/spider/dev.json
output_file=predicted_sql.txt # output file path to save the predicted SQLs, e.g., predicted_sql.txt
batch_size=20
beam_size=5
n_best=5
device=0

python3 -u scripts/eval_from_scratch.py --read_model_path $read_model_path --output_file $output_file --batch_size $batch_size --beam_size $beam_size --n_best $n_best --device $device #--dataset_path $dataset_path --db_dir $db_dir --table_path $table_path
