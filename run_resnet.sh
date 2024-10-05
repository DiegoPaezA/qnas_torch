#!/bin/bash

echo "Starting ResNet 18"
exp_path="experiments_atleta_axial_v3/resnet_18"

CUDA_VISIBLE_DEVICES=0 python train_resnet.py \
    --experiment_path "$exp_path" \
    --data_path atleta_axial_data \
    --dataset atleta_axial \
    --model_flag resnet18 \
    --retrain_folder retrain \
    --config_code F10 \
    --log_level INFO \
    --max_epochs 300 \
    --batch_size 32 \
    --eval_batch_size 16 \
    --device cuda:0 \
    --num_repetitions 3 \
    --lr_scheduler "multistep" \
    --data_augmentation \
    --optimizer "AdamW" &

pid_resnet18=$!  # capture the process ID of the ResNet 18 command

echo "Starting ResNet 50"
exp_path="experiments_atleta_axial_v3/resnet_50"

CUDA_VISIBLE_DEVICES=0 python train_resnet.py \
    --experiment_path "$exp_path" \
    --data_path atleta_axial_data \
    --dataset atleta_axial \
    --model_flag resnet50 \
    --retrain_folder retrain \
    --config_code F10 \
    --log_level INFO \
    --max_epochs 300 \
    --batch_size 32 \
    --eval_batch_size 16 \
    --device cuda:0 \
    --num_repetitions 3 \
    --lr_scheduler "multistep" \
    --data_augmentation \
    --optimizer "AdamW"  &

pid_resnet50=$!  # capture the process ID of the ResNet 50 command

# Wait for both processes to finish
wait $pid_resnet18
wait $pid_resnet50

echo "ResNet 18 and ResNet 50 done"
echo ""