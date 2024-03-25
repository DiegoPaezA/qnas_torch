#!/bin/bash
# Define variables for organamnist and exp8
dataset="cifar10"
exp="exp8"

echo "Starting $exp F10 repeat 1"
exp_path="experiments_${dataset}/${exp}_repeat_7"

CUDA_VISIBLE_DEVICES=0 python retrain_model.py \
    --experiment_path "$exp_path" \
    --data_path "${dataset}_data" \
    --dataset "$dataset" \
    --retrain_folder retrain \
    --config_code F10 \
    --log_level INFO \
    --max_epochs 100 \
    --batch_size 128 \
    --eval_batch_size 128 \
    --device cuda:0 \
    --num_repetitions 3 \
    --lr_scheduler "multistep" \
    --data_augmentation \
    --optimizer "AdamW" &

pid_exp_1=$!  # capture the process ID 

echo "Starting $exp F10 repeat 2"

exp_path="experiments_${dataset}/${exp}_repeat_8"

CUDA_VISIBLE_DEVICES=2 python retrain_model.py \
    --experiment_path "$exp_path" \
    --data_path "${dataset}_data" \
    --dataset "$dataset" \
    --retrain_folder retrain \
    --config_code F10 \
    --log_level INFO \
    --max_epochs 100 \
    --batch_size 128 \
    --eval_batch_size 128 \
    --device cuda:0 \
    --num_repetitions 3 \
    --lr_scheduler "multistep" \
    --data_augmentation \
    --optimizer "AdamW" &

pid_exp_2=$!  # capture the process ID

echo ""

wait $pid_exp_1
wait $pid_exp_2

echo "$exp repeat 1 and $exp repeat 2 done"
echo ""