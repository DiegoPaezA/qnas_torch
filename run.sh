#!/bin/bash

# Run the evolution experiment
echo "Running evolution experiment with config2.txt"

for ((i=1; i<=5; i++)); do
    exp_path="new_50_exp_adamw_$i"
    
    python run_evolution.py \
        --experiment_path "$exp_path" \
        --config_file 'config_files/config2.txt' \
        --data_path cifar10_data \
        --log_level INFO
done