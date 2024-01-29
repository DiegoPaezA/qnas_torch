#!/bin/bash

# Run the evolution experiment multiple times
echo "Running evolution experiment with config2.txt"

for ((i=1; i<=5; i++)); do
    exp_path="exp10_adamw_repeat_$i"
    
    python run_evolution.py \
        --experiment_path "$exp_path" \
        --config_file 'config_files/config2.txt' \
        --data_path cifar10_data \
        --log_level INFO
done

# Run the evolution experiment multiple times with different config files rmsprop
# echo "Running evolution experiment with config7.txt"

# for ((i=1; i<=5; i++)); do
#     exp_path="exp9_rms_repeat$i"
    
#     python run_evolution.py \
#         --experiment_path "$exp_path" \
#         --config_file 'config_files/config7.txt' \
#         --data_path cifar10_data \
#         --log_level INFO
# done