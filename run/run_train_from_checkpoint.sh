#!/bin/bash

read_model_path=$1
batch_size=20
grad_accumulate=1
test_batch_size=50
beam_size=5
n_best=5

params="----load_optimizer --read_model_path $read_model_path --batch_size $batch_size --grad_accumulate $grad_accumulate --test_batch_size --beam_size $beam_size --n_best $n_best"

GPU_PER_NODE=${GPU_PER_NODE:-1}
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
if [ "$GPU_PER_NODE" -gt 1 ] || [ "$WORLD_SIZE" -gt 1 ] ; then
    python3 -um torch.distributed.launch --nproc_per_node $GPU_PER_NODE --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT scripts/train_and_eval.py --ddp $params
else
    python3 -u scripts/train_and_eval.py $params
fi
