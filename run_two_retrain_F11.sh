#!/bin/bash
# Define variables for organamnist, exp8
dataset="organamnist"
exp="exp8"

echo "Starting $exp F11 repeat 1"
exp_path="experiments_${dataset}_mob/${exp}_repeat_1"

CUDA_VISIBLE_DEVICES=2 python retrain_model.py \
    --experiment_path "$exp_path" \
    --data_path "${dataset}_data" \
    --dataset "$dataset" \
    --retrain_folder retrain \
    --config_code F11 \
    --log_level INFO \
    --max_epochs 200 \
    --batch_size 128 \
    --eval_batch_size 128 \
    --device cuda:0 \
    --num_repetitions 3 \
    --lr_scheduler "multistep" \
    --data_augmentation \
    --optimizer "AdamW" &

pid_exp_F11_1=$!  # capture the process ID 

echo "Starting $exp F11 repeat 2"

exp_path="experiments_${dataset}_mob/${exp}_repeat_2"

CUDA_VISIBLE_DEVICES=2 python retrain_model.py \
    --experiment_path "$exp_path" \
    --data_path "${dataset}_data" \
    --dataset "$dataset" \
    --retrain_folder retrain \
    --config_code F11 \
    --log_level INFO \
    --max_epochs 200 \
    --batch_size 128 \
    --eval_batch_size 128 \
    --device cuda:0 \
    --num_repetitions 3 \
    --lr_scheduler "multistep" \
    --data_augmentation \
    --optimizer "AdamW" &

pid_exp_F11_2=$!  # capture the process ID

echo "Starting $exp F11 repeat 3"

exp_path="experiments_${dataset}_mob/${exp}_repeat_3"

CUDA_VISIBLE_DEVICES=2 python retrain_model.py \
    --experiment_path "$exp_path" \
    --data_path "${dataset}_data" \
    --dataset "$dataset" \
    --retrain_folder retrain \
    --config_code F11 \
    --log_level INFO \
    --max_epochs 200 \
    --batch_size 128 \
    --eval_batch_size 128 \
    --device cuda:0 \
    --num_repetitions 3 \
    --lr_scheduler "multistep" \
    --data_augmentation \
    --optimizer "AdamW" &

pid_exp_F11_3=$!  # capture the process ID

echo ""

wait $pid_exp_F11_1
wait $pid_exp_F11_2
wait $pid_exp_F11_3

echo "$exp repeat 1 and $exp repeat 2 and $exp repeat 3 completed"
echo ""