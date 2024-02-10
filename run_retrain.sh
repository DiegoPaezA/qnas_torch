#!/bin/bash

# Run the retrains for the experiment

# Directory containing the experiment folders
experiments_directory="retrain_experiments"

# Array to store exp_path values
exp_paths=()

# Function to run the retrain script for a given exp_path
run_retrain() {
    local exp_path="$1"
    python run_retrain.py \
        --experiment_path "$exp_path" \
        --data_path cifar10_data \
        --retrain_folder retrain_1 \
        --log_level INFO \
        --max_epochs 300 \
        --batch_size 256 \
        --eval_batch_size 1000 \
        --limit_data False \
        --device cuda:0 \
        --num_repetitions 5
}

# Populate exp_paths with the folders in experiments_directory
for exp_folder in "$experiments_directory"/*; do
    if [ -d "$exp_folder" ]; then
        exp_paths+=("$exp_folder")
    fi
done

# Number of concurrent processes
max_processes=2
current_processes=0

# Iterate through exp_paths
for exp_path in "${exp_paths[@]}"; do
    # Wait until a slot is available
    while [ "$current_processes" -ge "$max_processes" ]; do
        sleep 1
    done

    # Run the retrain function in the background
    run_retrain "$exp_path" &
    ((current_processes++))
done

# Wait for all background jobs to finish
wait