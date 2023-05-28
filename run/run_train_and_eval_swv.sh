task=astormer-swv
dataset=$1
seed=999
device=0
ddp='' # --ddp

plm=$2 #electra-small-discriminator
encode_method=rgatsql
encoder_hidden_size=256
encoder_num_layers=2
num_heads=8

decode_method=ast
decode_order=dfs+l2r
decoder_cell=transformer
decoder_hidden_size=256
decoder_num_layers=2

dropout=0.2
batch_size=20
test_batch_size=50
grad_accumulate=1
lr=5e-4
l2=1e-4
layerwise_decay=0.8
warmup_ratio=0.1
lr_schedule=linear
eval_after_iter=60
max_iter=100
max_norm=5
beam_size=5
n_best=5

python3 -u scripts/train_and_eval.py --task $task --dataset $dataset --seed $seed --device $device $ddp \
    --encode_method $encode_method --swv $swv --plm $plm --encoder_hidden_size $encoder_hidden_size --encoder_num_layers $encoder_num_layers --num_heads $num_heads \
    --decode_method $decode_method --decode_order $decode_order --decoder_cell $decoder_cell --decoder_hidden_size $decoder_hidden_size --decoder_num_layers $decoder_num_layers \
    --dropout $dropout --batch_size $batch_size --test_batch_size $test_batch_size --grad_accumulate $grad_accumulate --eval_after_iter $eval_after_iter --max_iter $max_iter \
    --lr $lr --l2 $l2 --layerwise_decay $layerwise_decay --warmup_ratio $warmup_ratio --lr_schedule $lr_schedule --max_norm $max_norm --beam_size $beam_size --n_best $n_best
