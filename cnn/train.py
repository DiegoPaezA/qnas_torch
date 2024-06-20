""" Copyright (c) 2023, Diego Páez
* Licensed under the MIT license

- Compute the fitness of a model_net using the evolved networks.

Documentation:

    - Automatic mixed precision training (AMP): 
        - https://pytorch.org/docs/stable/amp.html, 
        - https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#all-together-automatic-mixed-precision
    - Profiling: Inference time
        - https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-profiler-to-analyze-execution-time
    
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Union, Any
from cnn import model, input
from util import create_info_file, init_log, load_yaml
from torch.cuda.amp import GradScaler
from torch.profiler import profile, record_function, ProfilerActivity



TRAIN_TIMEOUT = 5400
MAX_PARAMS = 11170884 # Maximum number of parameters for a model - ResNet18 has 11,170,884 parameters
MAX_INFER_TIME = 2586 # Maximum inference time in microseconds

current_directory = os.path.dirname(os.path.dirname(__file__))
log_directory = os.path.join(current_directory, 'logs')
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
    
log_file = os.path.join(log_directory, 'train.log')
LOGGER = init_log("INFO", name=__name__, file_path=log_file)

def mofitness(acc, params, inference_time, T_p, T_t) -> float:
    """
    Scalarized multi-objective fitness function.

    :param acc: Classification accuracy of the architecture
    :param params: Number of parameters of the architecture
    :param inference_time: Inference time of the architecture
    :param T_p: Maximum allowable number of parameters
    :param T_t: Maximum allowable inference time
    :return: Fitness value
    """
    acc_in_range = 0 <= acc <= 1
    acc = acc if acc_in_range else acc / 100.0
    # Determine weights based on parameters and inference time
    if params <= T_p:
        w_p = -0.01
    else:
        w_p = -1

    if inference_time <= T_t:
        w_t = -0.01
    else:
        w_t = -1

    # Calculate the fitness function
    fitness_value = (acc) * (params / T_p) ** w_p * (inference_time / T_t) ** w_t
    fitness_value = fitness_value if acc_in_range else fitness_value * 100.0

    return fitness_value


def train_epoch(model, criterion, optimizer, data_loader, params, scaler):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    device = params['device']
    amp_device = device.split(':')[0] if device != 'cpu' else 'cpu'
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with torch.autocast(device_type=amp_device, dtype=torch.float16, enabled=params['mixed_precision']):
            y_logits = model(inputs)
            if params['task'] == 'multi-class':
                labels = labels.squeeze().long()
            loss = criterion(y_logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
        _, predicted = y_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    accuracy = 100 * correct / total
    train_loss /= len(data_loader)
    return train_loss, accuracy

def evaluate(model, criterion, data_loader, params):
    model.eval()
    validation_loss = 0.0
    correct = 0
    total = 0
    device = params['device']
    amp_device = device.split(':')[0] if device != 'cpu' else 'cpu'

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.autocast(device_type=amp_device, dtype=torch.float16, enabled=params['mixed_precision']):
                y_logits = model(inputs)
                if params['task'] == 'multi-class':
                    labels = labels.squeeze().long() # medmnist
                loss = criterion(y_logits, labels)
            validation_loss += loss.item()
            _, predicted = y_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    validation_loss /= len(data_loader)

    return validation_loss, accuracy

def train(model:torch.nn.Module, criterion:torch.nn.Module, optimizer:torch.optim.Optimizer, 
          train_loader:torch.utils.data.DataLoader, val_loader:torch.utils.data.DataLoader, 
          params:Dict, debug=False) -> Dict:
    """
    Train a neural network model.

    Args:
        model: Model to be trained.
        criterion: Loss function.
        optimizer: Optimization algorithm.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        params: Dictionary with parameters necessary for training
            - max_epochs: Number of epochs to train.
            - epochs_to_eval: Number of epochs before starting validation.
            - t0: Time when the training started.
        device: Device to run the training on (CPU or GPU).

    Returns:
        training_results: Dictionary with the training results.
        
            -training_losses: List of training losses for each epoch.
            -validation_losses: List of validation losses for each epoch.
            -best_accuracy: Best validation accuracy achieved.
    """
    model.train()
    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []
    best_accuracy = 0.0
    best_validation_loss = float('inf')
    mean_eval_accuracy = 0.0
    median_eval_accuracy = 0.0
    
    training_results = {}
    max_epochs = params['max_epochs']
    epochs_to_eval = params['epochs_to_eval']
    start_eval = max_epochs - epochs_to_eval
    
    # Automatic mixed precision training (AMP)
    scaler = GradScaler(enabled=params['mixed_precision']) 

    for epoch in range(1, max_epochs + 1):
        train_loss, train_accuracy = train_epoch(model, criterion, optimizer, train_loader, params,scaler)
        training_losses.append(train_loss)
        training_accuracies.append(train_accuracy)

        if epoch < start_eval and (time.time() - params['t0']) > TRAIN_TIMEOUT:
            print("Timeout reached")
            raise TimeoutError()
        
        if epoch > start_eval:
            validation_loss, accuracy = evaluate(model, criterion, val_loader, params)
            validation_losses.append(validation_loss)
            validation_accuracies.append(accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                create_info_file(params['model_path'], {'best_accuracy': best_accuracy}, 'best_accuracy.txt')
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                create_info_file(params['model_path'], {'best_validation_loss': best_validation_loss}, 'best_validation_loss.txt')
            if debug:
                if epoch % 1 == 0:
                    print(f"Epoch [{epoch}/{max_epochs}] - Training loss: {train_loss} - Validation loss: {validation_loss} - Validation accuracy: {accuracy}%")
        if debug:    
            if epoch % 5 == 0 and epoch < start_eval:
                print(f"Epoch [{epoch}/{max_epochs}] - Training loss: {train_loss}")
    
    mean_eval_accuracy = float(np.mean(validation_accuracies))
    median_eval_accuracy = float(np.median(validation_accuracies))
            
    params['t1'] = time.time()
    params['training_time'] = params['t1'] - params['t0']
    
    # Measure inference time
    inference_images = next(iter(val_loader))[0][:10].to(params['device'])
    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],profile_memory=True, record_shapes=True) as prof:
        with record_function("model_inference"):
            model(inference_images)
    
    model_memory_usage = sum(event.cuda_memory_usage for event in prof.key_averages()) / (1024 ** 2)
    cpu_inference_time = prof.key_averages()[0].cpu_time
    cuda_inference_time = prof.key_averages()[0].cuda_time
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    
    # Scalarized multi-objective function
    scalar_multi_objective = mofitness(best_accuracy, total_trainable_params, cuda_inference_time, params['max_params'], params['max_inference_time'])
    fitness_val_loss = 1 - best_validation_loss
    
    params['total_params'] = total_params
    params['total_trainable_params'] = total_trainable_params
    params['cuda_inference_time'] = cuda_inference_time
    params['cpu_inference_time'] = cpu_inference_time
    params['model_memory_usage'] = model_memory_usage
    params['best_accuracy'] = best_accuracy
    params['mean_eval_accuracy'] = mean_eval_accuracy
    params['median_eval_accuracy'] = median_eval_accuracy
    params['scalar_multi_objective'] = scalar_multi_objective

    
    LOGGER.info(f"Cuda Inference time: {cuda_inference_time} microseconds")
    LOGGER.info(f"Model memory usage: {model_memory_usage} MB")
    
    create_info_file(params['model_path'], params, 'training_params.txt')
    
    training_results['training_losses'] = training_losses
    training_results['training_accuracies'] = training_accuracies
    training_results['validation_losses'] = validation_losses
    training_results['validation_accuracies'] = validation_accuracies
    training_results['cuda_inference_time'] = cuda_inference_time # in microseconds
    training_results['model_memory_usage'] = model_memory_usage # in MB
    training_results['total_trainable_params'] = total_trainable_params
    training_results['best_accuracy'] = best_accuracy
    training_results['mean_eval_accuracy'] = mean_eval_accuracy
    training_results['median_eval_accuracy'] = median_eval_accuracy
    training_results['scalar_multi_objective'] = scalar_multi_objective        
    return training_results


def fitness_calculation(id_num:str, params:Dict[str, Any], 
                        fn_dict:Dict[str, Any], net_list:List[str],
                        train_loader:torch.utils.data.DataLoader, val_loader:torch.utils.data.DataLoader,
                        return_val,debug:bool=False) -> Dict[str, Union[List[float], float]]:
    """Train and evaluate a model using evolved hyperparameters.

    This function trains and evaluates a convolutional neural network model using the specified
    configuration and evolved hyperparameters.

    Args:
        id_num (str): A string identifying the generation number and the individual number.
        params (Dict[str, Any]): A dictionary with parameters necessary for training, including
            the evolved hyperparameters.
        fn_dict (Dict[str, Any]): A dictionary with definitions of the possible layers, including
            their names and parameters.
        net_list (List[str]): A list with names of layers defining the network, in the order they appear.

    Returns:
        Dict[str, Union[List[float], float]]: A dictionary containing the training results.

        - 'training_losses' (List[float]): List of training losses for each epoch.
        - 'validation_losses' (List[float]): List of validation losses for each epoch.
        - 'best_accuracy' (float): Best validation accuracy achieved.

    Raises:
        TimeoutError: If the training process takes too long to complete.
    """
   
    device = params['device']
    params['net_list'] = net_list
    model_path = os.path.join(params['experiment_path'], id_num)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    params['model_path'] = model_path
    params['generation'] = id_num.split('_')[0]
    params['individual'] = id_num.split('_')[1]
     
    LOGGER.info(f"Training model {id_num} on device {device} ...")
    # Load data info
    if hasattr(input.available_datasets, params['dataset'].lower()):
        dataset_info = input.available_datasets[params['dataset'].lower()]
    else:
        dataset_info = load_yaml(os.path.join(params['data_path'], 'data_info.txt'))
    
    params['num_classes'] = dataset_info["num_classes"]
    params['task'] = dataset_info["task"]
        
    # check if cbam is a key in the fn_dict
    has_cbam_key = any(key.startswith("cbam") for key in fn_dict)
    
    # Create the model
    model_net = model.NetworkGraph(num_classes=dataset_info["num_classes"])    
    filtered_dict = {key: item for key, item in fn_dict.items() if key in net_list}
    model_net.create_functions(fn_dict=filtered_dict, net_list=net_list, cbam=has_cbam_key)
    
    # Add the fully connected layer to the model
    input_shape =  [params['batch_size']] + dataset_info['shape']
    inputs = torch.randn(input_shape)
    _ = model_net(inputs)
    model_net.to(device)
    
    params['input_shape'] = input_shape
    
    criterion = nn.CrossEntropyLoss()
    
    if params['optimizer'] == 'RMSProp':
        optimizer = torch.optim.RMSprop(model_net.parameters())
    elif params['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model_net.parameters())
    elif params['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model_net.parameters())
    else:
        optimizer = torch.optim.SGD(model_net.parameters(), lr=params['learning_rate'])

    # Training time start counting here.
    params['t0'] = time.time()
    
    # Train the model in fitness scheme
    try:
        results_dict = train(model_net, criterion, optimizer, train_loader, val_loader,params,debug)
        if debug:
            result = results_dict
            return result
        else:
            #return_val.value = results_dict['best_accuracy']
            if params['fitness_metric'] == 'best_accuracy':
                return_val[0] = results_dict['best_accuracy']
                return_val[1] = results_dict['model_memory_usage']
                return_val[2] = results_dict['cuda_inference_time']
            elif params['fitness_metric'] == 'mean_accuracy':
                return_val[0] = results_dict['mean_eval_accuracy']
                return_val[1] = results_dict['model_memory_usage']
                return_val[2] = results_dict['cuda_inference_time']
            elif params['fitness_metric'] == 'median_accuracy':
                return_val[0] = results_dict['median_eval_accuracy']
                return_val[1] = results_dict['model_memory_usage']
                return_val[2] = results_dict['cuda_inference_time']
            elif params['fitness_metric'] == 'scalar_multi_objective':
                return_val[0] = results_dict['scalar_multi_objective']
                return_val[1] = results_dict['model_memory_usage']
                return_val[2] = results_dict['cuda_inference_time']
            else:
                raise ValueError(f"Invalid fitness metric: {params['fitness_metric']}")
        LOGGER.info(f"Training of model {id_num} finished, best {params['fitness_metric']}: {round(return_val[0], 2)}")
        
    except TimeoutError:
        LOGGER.error("Training timed out. Penalizing the model with accuracy 0.0.")
        #return_val.value = 0.0
        return_val[0] = 0.0
        return_val[1] = 0.0
        return_val[2] = 0.0
    except MemoryError:
        LOGGER.error(f"CUDA out of memory exception, error: {e}")
        return_val[0] = 0.0
        return_val[1] = 0.0
        return_val[2] = 0.0
    except Exception as e:
        if "out of memory" in str(e):
            LOGGER.error(f"CUDA out of memory exception, error: {e}")
            return_val[0] = 0.0
            return_val[1] = 0.0
            return_val[2] = 0.0
        else:
            LOGGER.error(f"Exception: {e}")
            return_val[0] = 0.0
            return_val[1] = 0.0
            return_val[2] = 0.0
        raise e