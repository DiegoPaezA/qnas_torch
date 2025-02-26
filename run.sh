#!/bin/bash

# Define common variables for the evolution experiments
dataset="cifar10"
exp_path_base="experiment_${dataset}_acc_17"
config_file="config_files_cifar"
fitness_metric="best_accuracy"
data_path="${dataset}_data"
log_level="INFO"
network_config="default"
dataset_sample_size=10000

configs=("config15.txt")
exps=("exp1")
cuda_devices=("0")

# Loop over the length of the configs array
for ((j=0; j<${#configs[@]}; j++)); do
    config="${configs[$j]}"
    exp="${exps[$j]}"
    cuda_device="${cuda_devices[$j]}"

    echo "Running evolution experiment with $config"

    for ((i=1; i<=3; i++)); do # Change the range to the number of repeats
        exp_path="${exp_path_base}/${exp}_repeat_$i"

        CUDA_VISIBLE_DEVICES="$cuda_device" python run_evolution.py \
            --experiment_path "$exp_path" \
            --config_file "${config_file}/${config}" \
            --data_path "$data_path" \
            --dataset "$dataset" \
            --limit_data_value "$dataset_sample_size" \
            --fitness_metric "$fitness_metric" \
            --early_stopping \
            --network_config "$network_config" \
            --en_pop_crossover \
            --log_level "$log_level"
    done
done


# dataset="atleta_axial"
# exp_path_base="experiments_${dataset}_mo"
# config_file="config_files_atleta"
# fitness_metric="scalar_multi_objective"
# data_path="${dataset}_data"
# log_level="DEBUG" # INFO

# # configs=("config3.txt" "config4.txt" "config6.txt" "config7.txt" "config10.txt")
# # exps=("exp1" "exp2" "exp3" "exp4" "exp5")
# # cuda_devices=("0,1" "0,1" "0,1" "0,1" "0,1")

# configs=("config3.txt" "config10.txt")
# exps=("exp1" "exp5")
# cuda_devices=("0" "0")

# for ((j=0; j<${#configs[@]}; j++)); do
#     config="${configs[$j]}"
#     exp="${exps[$j]}"
#     cuda_device="${cuda_devices[$j]}"
    
#     echo "Running evolution experiment with $config"

#     for ((i=1; i<=3; i++)); do # Change the range to the number of repeats
#         exp_path="${exp_path_base}/${exp}_repeat_$i"
        
#         CUDA_VISIBLE_DEVICES="$cuda_device" python run_evolution.py \
#             --experiment_path "$exp_path" \
#             --config_file "${config_file}/${config}" \
#             --data_path "$data_path" \
#             --dataset "$dataset" \
#             --fitness_metric "$fitness_metric" \
#             --early_stopping \
#             --log_level "$log_level"
#     done
# done


# dataset="atleta_coronal"
# exp_path_base="experiments_${dataset}_mo"
# config_file="config_files_atleta"
# fitness_metric="scalar_multi_objective"
# data_path="${dataset}_data"
# log_level="INFO"

# configs=("config4.txt" "config10.txt")
# exps=("exp2" "exp5")
# cuda_devices=("0" "0")

# # Loop over the length of the configs array
# for ((j=0; j<${#configs[@]}; j++)); do
#     config="${configs[$j]}"
#     exp="${exps[$j]}"
#     cuda_device="${cuda_devices[$j]}"
    
#     echo "Running evolution experiment with $config"

#     for ((i=1; i<=3; i++)); do # Change the range to the number of repeats
#         exp_path="${exp_path_base}/${exp}_repeat_$i"
        
#         CUDA_VISIBLE_DEVICES="$cuda_device" python run_evolution.py \
#             --experiment_path "$exp_path" \
#             --config_file "${config_file}/${config}" \
#             --data_path "$data_path" \
#             --dataset "$dataset" \
#             --fitness_metric "$fitness_metric" \
#             --data_augmentation \
#             --log_level "$log_level"
#     done
# done
