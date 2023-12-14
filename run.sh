#!/bin/bash

# Run the evolution experiment
echo "Running evolution experiment"

python run_evolution.py \
    --experiment_path my_exp_4 \
    --config_file 'config_files/config2.txt' \
    --data_path cifar10_data \
    --log_level INFO