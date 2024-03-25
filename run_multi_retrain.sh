#!/bin/bash

# Run the retrains for the experiment

# Directory containing the experiment folders
experiments_directory="retrain_cifar10_base_1"

# Array to store exp_path values
exp_paths=()

# Function to run the retrain script for a given exp_path
run_retrain() {
    local exp_path="$1"
    CUDA_VISIBLE_DEVICES=0 python retrain_model.py \
        --experiment_path "$exp_path" \
        --data_path cifar10_data \
        --dataset cifar10 \
        --retrain_folder retrain \
        --config_code F10 \
        --log_level INFO \
        --max_epochs 100 \
        --batch_size 128 \
        --eval_batch_size 128 \
        --device cuda:0 \
        --num_repetitions 3 \
        --lr_scheduler "multistep" \
        --data_augmentation \
        --optimizer "AdamW"
}

# Print the contents of exp_paths
num_parallel_processes=4
echo "Contents of exp_paths:"
for exp_folder in "$experiments_directory"/*; do
    if [ -d "$exp_folder" ]; then
        run_retrain "$exp_folder" &
        echo "Started process for $exp_folder"
    fi

    # Limit to num_parallel_processes
    if [ $(jobs -p | wc -l) -ge $num_parallel_processes ]; then
        wait -n || { echo "Waiting for background jobs..."; wait -n; }
    fi
done

# Wait for all remaining background jobs to finish
wait