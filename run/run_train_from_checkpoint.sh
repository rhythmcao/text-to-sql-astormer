#!/bin/bash

read_model_path=$1
batch_size=20
grad_accumulate=1
test_batch_size=50
beam_size=5
n_best=5

params="--load_optimizer --read_model_path $read_model_path --batch_size $batch_size --grad_accumulate $grad_accumulate --test_batch_size --beam_size $beam_size --n_best $n_best"

GPU_PER_NODE=${GPU_PER_NODE:-1}
NUM_NODES=${NUM_NODES:-1}
if [ "$GPU_PER_NODE" -gt 1 ] || [ "$NUM_NODES" -gt 1 ] ; then
    NODE_RANK=${NODE_RANK:-0}
    MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
    MASTER_PORT=${MASTER_PORT:-23456}
    python3 -um torch.distributed.launch --nproc_per_node $GPU_PER_NODE --nnodes $NUM_NODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT scripts/train_and_eval.py --ddp $params
else
    python3 -u scripts/train_and_eval.py $params
fi
