#!/bin/bash

read_model_path=
ddp=''
batch_size=20
grad_accumulate=1
test_batch_size=50
beam_size=5
n_best=1
device=0

python3 -u scripts/train_and_eval.py --load_optimizer --read_model_path $read_model_path --device $device $ddp \
    --batch_size $batch_size --grad_accumulate $grad_accumulate --test_batch_size $test_batch_size --beam_size $beam_size --n_best $n_best