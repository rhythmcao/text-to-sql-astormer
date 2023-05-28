#!/bin/bash

read_model_path=exp/task_inference-time__dataset_${1}__encoder_rgatsql__decoder_${2}/electra-small-discriminator__enc_256_x_2__hd_8__order_dfs+l2r__dec_256_x_2__cell_${3}__bs_20__dp_0.2__lr_0.0004_ld_0.8__l2_0.1__mi_100__mn_5.0__bm_5__nb_5__seed_999/
test_batch_size=50
beam_size=5
n_best=5
device=0

python -u scripts/train_and_eval.py --read_model_path $read_model_path --test_batch_size $test_batch_size --device $device --testing --beam_size $beam_size --n_best $n_best
