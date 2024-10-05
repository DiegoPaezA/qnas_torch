#!/bin/bash

# Define variables for retrain experiment
dataset="atleta_coronal"
exp="exp5"
repeat="1"
echo "Starting $exp F13 repeat $repeat"
exp_path="experiments_${dataset}_v3/${exp}_repeat_${repeat}"

# Retrain model
CUDA_VISIBLE_DEVICES=2 python retrain_model.py \
    --experiment_path "$exp_path" \
    --data_path "${dataset}_data" \
    --dataset "$dataset" \
    --retrain_folder retrain \
    --config_code F13 \
    --log_level INFO \
    --max_epochs 300 \
    --batch_size 32 \
    --eval_batch_size 16 \
    --device cuda:0 \
    --num_repetitions 3 \
    --lr_scheduler "multistep" \
    --data_augmentation \
    --optimizer "AdamW"

# Check if the previous command was successful
if [ $? -ne 0 ]; then
    echo "Error: Retrain model script failed."
    exit 1
fi

echo "Retrain model script completed successfully."
