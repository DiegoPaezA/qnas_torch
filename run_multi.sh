#!/bin/bash

# Run the evolution experiment multiple times
echo "Running evolution experiment with config2.txt"

for ((i=1; i<=5; i++)); do
    exp_path="multi_50_exp_$i"
    
    python run_evolution.py \
        --experiment_path "$exp_path" \
        --config_file 'config_files/config2.txt' \
        --data_path cifar10_data \
        --log_level INFO
done

# Run the evolution experiment multiple times with different config files
echo "Running evolution experiment with config3.txt"

for ((i=1; i<=5; i++)); do
    exp_path="multi_100_exp_$i"
    
    python run_evolution.py \
        --experiment_path "$exp_path" \
        --config_file 'config_files/config3.txt' \
        --data_path cifar10_data \
        --log_level INFO
done

# Run the evolution experiment multiple times with different config files rmsprop
echo "Running evolution experiment with config4.txt"

for ((i=1; i<=5; i++)); do
    exp_path="multi_50_exp_rms_$i"
    
    python run_evolution.py \
        --experiment_path "$exp_path" \
        --config_file 'config_files/config4.txt' \
        --data_path cifar10_data \
        --log_level INFO
done
