""" Copyright (c) 2023, Diego Páez
* Licensed under the MIT license

- Compute the fitness of a model_net using the evolved networks.


"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from cnn.metrics import *
from typing import Dict, List, Union, Any
from cnn import model, input
from util import create_info_file


TRAIN_TIMEOUT = 5400

def train_epoch(model, criterion, optimizer, data_loader, device):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        y_logits = model(inputs)
        loss = criterion(y_logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = y_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    accuracy = 100 * correct / total
    train_loss /= len(data_loader)
    return train_loss, accuracy

def evaluate(model, criterion, data_loader, device):
    model.eval()
    validation_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            y_logits = model(inputs)
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
          params:Dict, device:torch.device) -> Dict:
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
    training_results = {}
    max_epochs = params['max_epochs']
    epochs_to_eval = params['epochs_to_eval']
    start_eval = max_epochs - epochs_to_eval
    best_model_path = os.path.join(params['model_path'], 'best_model.pth')

    for epoch in tqdm(range(1, max_epochs + 1), desc="Training Fitness Scheme"):
        train_loss, train_accuracy = train_epoch(model, criterion, optimizer, train_loader, device)
        training_losses.append(train_loss)
        training_accuracies.append(train_accuracy)

        if epoch < start_eval and (time.time() - params['t0']) > TRAIN_TIMEOUT:
            print("Timeout reached")
            raise TimeoutError()
        
        if epoch > start_eval:
            validation_loss, accuracy = evaluate(model, criterion, val_loader, device)
            validation_losses.append(validation_loss)
            validation_accuracies.append(accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), best_model_path)
                create_info_file(params['model_path'], {'best_accuracy': best_accuracy}, 'best_accuracy.txt')

            if epoch % 1 == 0:
                print(f"Epoch [{epoch}/{max_epochs}] - Training loss: {train_loss} - Validation loss: {validation_loss} - Validation accuracy: {accuracy}%")

        if epoch % 5 == 0 and epoch < start_eval:
            print(f"Epoch [{epoch}/{max_epochs}] - Training loss: {train_loss}")
            
    params['t1'] = time.time()
    
    create_info_file(params['model_path'], params, 'training_params.txt')
    
    training_results['training_losses'] = training_losses
    training_results['training_accuracies'] = training_accuracies
    training_results['validation_losses'] = validation_losses
    training_results['validation_accuracies'] = validation_accuracies
    training_results['best_accuracy'] = best_accuracy        
    return training_results


def fitness_calculation(id_num: str,
                        params: Dict[str, Any], 
                        fn_dict: Dict[str, Any], 
                        net_list: List[str],
                        debug:bool=False) -> Dict[str, Union[List[float], float]]:
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


    model_path = os.path.join(params['experiment_path'], id_num)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    params['model_path'] = model_path
    
    if params['limit_data']:
        limit_data = params['limit_data_value']
    else:
        limit_data = None

    # Load data
    if params['dataset'] == 'Cifar10':
        
        data_info = input.cifar10_info
        data_path = 'cifar10_data'
        
        train_loader, val_loader = input.CIFAR10_loader(data_path, limit_data=limit_data,
                                                        for_train=True, 
                                                        data_aug=params['data_augmentation'],
                                                        batch_size=params['batch_size'],
                                                        eval_batch_size=params['eval_batch_size'])
        
    elif params['dataset'] == 'Cifar100':
        data_info = input.cifar100_info

    model_net = model.NetworkGraph(num_classes=data_info["num_classes"], mu=0.99)
    
    filtered_dict = {key: item for key, item in fn_dict.items() if key in net_list}
    
    model_net.create_functions(fn_dict=filtered_dict, net_list=net_list)

    params['model_net'] = model_net
    params['net_list'] = net_list


    
    device = params['device']
    
    # Load the model_net to the GPU
    inputs, _ = next(iter(train_loader))
    _ = model_net(inputs)
    model_net.to(device)
    
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
        results_dict = train(model_net, criterion, optimizer, train_loader, val_loader,params,device)
        if debug:
            result = results_dict
        else:
            result = results_dict['best_accuracy']
        
    except TimeoutError:
        result = 0.0 # Penalize the model if it takes too long to train.
        return result
    
    return result
