#!/bin/bash

echo "Running evolution experiment with config2.txt"

# Define variables for evolution experiment
dataset="cifar10"
exp_path_base="experiments_${dataset}"
config_file="config_files_cifar"
fitness_metric="median_accuracy"

for ((i=1; i<=3; i++)); do
    exp_path="${exp_path_base}/exp2_repeat_$i"
    
    CUDA_VISIBLE_DEVICES="0,2" python run_evolution.py \
        --experiment_path "$exp_path" \
        --config_file "${config_file}/config2.txt" \
        --data_path ${dataset}_data \
        --dataset ${dataset} \
        --fitness_metric ${fitness_metric} \
        --log_level INFO
done

echo "Running evolution experiment with config4.txt"

for ((i=1; i<=3; i++)); do
    exp_path="${exp_path_base}/exp4_repeat_$i"
    
    CUDA_VISIBLE_DEVICES="0,2" python run_evolution.py \
        --experiment_path "$exp_path" \
        --config_file "${config_file}/config4.txt" \
        --data_path ${dataset}_data \
        --dataset ${dataset} \
        --fitness_metric ${fitness_metric} \
        --log_level INFO
done

echo "Running evolution experiment with config7.txt"

for ((i=1; i<=3; i++)); do
    exp_path="${exp_path_base}/exp7_repeat_$i"
    
    CUDA_VISIBLE_DEVICES="0,2" python run_evolution.py \
        --experiment_path "$exp_path" \
        --config_file "${config_file}/config7.txt" \
        --data_path ${dataset}_data \
        --dataset ${dataset} \
        --fitness_metric ${fitness_metric} \
        --log_level INFO
done

echo "Running evolution experiment with config8.txt"

for ((i=1; i<=3; i++)); do
    exp_path="${exp_path_base}/exp8_repeat_$i"
    
    CUDA_VISIBLE_DEVICES="0,2" python run_evolution.py \
        --experiment_path "$exp_path" \
        --config_file "${config_file}/config8.txt" \
        --data_path ${dataset}_data \
        --dataset ${dataset} \
        --fitness_metric ${fitness_metric} \
        --log_level INFO
done

