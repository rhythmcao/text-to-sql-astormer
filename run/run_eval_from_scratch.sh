#!/bin/bash

read_model_path=
db_dir=
table_path=
dataset_path=
output_path=
batch_size=20
beam_size=5
n_best=1
device=0

python3 -u scripts/eval_from_scratch.py --read_model_path $read_model_path --db_dir $db_dir --table_path $table_path --dataset_path $dataset_path \
    --output_path $output_path --batch_size $batch_size --beam_size $beam_size --n_best $n_best --device $device