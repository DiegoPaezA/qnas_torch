#!/bin/bash

# Run the evolution experiment multiple times
# echo "Running evolution experiment with config1.txt"

# exp_path="exp_test_x"

# # CUDA_VISIBLE_DEVICES="0,1" python run_evolution.py \    
# CUDA_VISIBLE_DEVICES=1 python run_evolution.py \
#         --experiment_path "$exp_path" \
#         --config_file 'config_files/config1.txt' \
#         --data_path cifar10_data \
#         --log_level INFO

echo "Running evolution experiment with config10.txt"

for ((i=1; i<=2; i++)); do
    exp_path="exp15_adamw_repeat_$i"
    
    CUDA_VISIBLE_DEVICES=0 python run_evolution.py \
        --experiment_path "$exp_path" \
        --config_file 'config_files/config10.txt' \
        --data_path cifar10_data \
        --log_level INFO
done

