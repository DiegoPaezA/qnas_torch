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
        --device cuda:0 \
        --num_repetitions 5
}

# Print the contents of exp_paths
echo "Contents of exp_paths:"
for exp_folder in "$experiments_directory"/*; do
    if [ -d "$exp_folder" ]; then
        run_retrain "$exp_folder" &
        echo "Started process for $exp_folder"
    fi

    # Limit to two processes in parallel
    if [ $(jobs -p | wc -l) -ge 2 ]; then
        wait -n || { echo "Waiting for background jobs..."; wait -n; }
    fi
done

# Wait for all remaining background jobs to finish
wait