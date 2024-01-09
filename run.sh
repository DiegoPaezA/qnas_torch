#!/bin/bash

# Run the evolution experiment multiple times
echo "Running evolution experiment with config2.txt"

exp_path="exp_test_x"
    
python run_evolution.py \
        --experiment_path "$exp_path" \
        --config_file 'config_files/config2.txt' \
        --data_path cifar10_data \
        --log_level INFO