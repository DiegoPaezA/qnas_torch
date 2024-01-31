#!/bin/bash

# Run the evolution experiment multiple times
echo "Running evolution experiment with config10.txt"

for ((i=1; i<=5; i++)); do
    exp_path="exp12_adamw_repeat_$i"
    
    python run_evolution.py \
        --experiment_path "$exp_path" \
        --config_file 'config_files/config10.txt' \
        --data_path cifar10_data \
        --log_level INFO
done

echo "Running evolution experiment with config11.txt"

for ((i=1; i<=5; i++)); do
    exp_path="exp13_adamw_repeat_$i"
    
    python run_evolution.py \
        --experiment_path "$exp_path" \
        --config_file 'config_files/config11.txt' \
        --data_path cifar10_data \
        --log_level INFO
done

echo "Running evolution experiment with config12.txt"

for ((i=1; i<=5; i++)); do
    exp_path="exp14_adamw_repeat_$i"
    
    python run_evolution.py \
        --experiment_path "$exp_path" \
        --config_file 'config_files/config12.txt' \
        --data_path cifar10_data \
        --log_level INFO
done
