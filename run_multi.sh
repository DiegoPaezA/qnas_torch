#!/bin/bash

# Run the evolution experiment multiple times
# echo "Running evolution experiment with config10.txt"

# for ((i=1; i<=5; i++)); do
#     exp_path="exp12_adamw_repeat_$i"
    
#     python run_evolution.py \
#         --experiment_path "$exp_path" \
#         --config_file 'config_files/config10.txt' \
#         --data_path cifar10_data \
#         --log_level INFO
# done

# echo "Running evolution experiment with config11.txt"

# for ((i=1; i<=5; i++)); do
#     exp_path="exp13_adamw_repeat_$i"
    
#     python run_evolution.py \
#         --experiment_path "$exp_path" \
#         --config_file 'config_files/config11.txt' \
#         --data_path cifar10_data \
#         --log_level INFO
# done

echo "Running evolution experiment with config13.txt"

for ((i=1; i<=5; i++)); do
    exp_path="exp15_adamw_repeat_$i"
    
    CUDA_VISIBLE_DEVICES='0,1' python run_evolution.py \
        --experiment_path "$exp_path" \
        --config_file 'config_files/config13.txt' \
        --data_path cifar10_data \
        --log_level INFO
done
