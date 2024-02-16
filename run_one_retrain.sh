#!/bin/bash
exp_path="exp9_adamw_repeat_5"

CUDA_VISIBLE_DEVICES=0 python run_retrain.py \
    --experiment_path "$exp_path" \
    --data_path cifar10_data \
    --retrain_folder retrain_1 \
    --log_level INFO \
    --max_epochs 300 \
    --batch_size 256 \
    --eval_batch_size 1000 \
    --device cuda:0 \
    --num_repetitions 5