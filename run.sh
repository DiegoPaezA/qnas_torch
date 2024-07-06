#!/bin/bash

# Define common variables for the evolution experiments
dataset="cifar100"
exp_path_base="experiments_${dataset}_v2"
config_file="config_files_cifar"
fitness_metric="best_accuracy"
data_path="${dataset}_data"
log_level="INFO"

configs=("config3.txt")
exps=("exp2")
cuda_devices=("0,1")

# Loop over the length of the configs array
for ((j=0; j<${#configs[@]}; j++)); do
    config="${configs[$j]}"
    exp="${exps[$j]}"
    cuda_device="${cuda_devices[$j]}"
    
    echo "Running evolution experiment with $config"

    for ((i=2; i<=3; i++)); do # Change the range to the number of repeats
        exp_path="${exp_path_base}/${exp}_repeat_$i"
        
        CUDA_VISIBLE_DEVICES="$cuda_device" python run_evolution.py \
            --experiment_path "$exp_path" \
            --config_file "${config_file}/${config}" \
            --data_path "$data_path" \
            --dataset "$dataset" \
            --fitness_metric "$fitness_metric" \
            --log_level "$log_level"
    done
done
