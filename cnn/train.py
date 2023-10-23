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
from typing import Tuple, Dict, List
from cnn import model, input

TRAIN_TIMEOUT = 5400

# Define a function for training
def train_model(model:torch.nn.Module, criterion:torch.nn.Module, optimizer:torch.optim.Optimizer,
                train_loader:torch.utils.data.DataLoader, device:torch.device):
    """
    Args:
        model: model to be trained
        criterion: loss function
        optimizer: optimization algorithm
        train_loader: data loader for training set
        device: device to run the training on (CPU or GPU)
        
    Returns:
        average loss over the training set
    """
    model.train()
    total_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        y_logits = model(inputs)
        loss = criterion(y_logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

# Define a function for validation
def validate_model(model:torch.nn.Module, criterion:torch.nn.Module, 
                   val_loader:torch.utils.data.DataLoader, device:torch.device):
    """
    Args:
        model: model to be evaluated
        criterion: loss function
        val_loader: data loader for validation set
        device: device to run the evaluation on (CPU or GPU)
        
    Returns:
        average loss and accuracy over the validation set
    """
    model.eval()
    validation_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            y_logits = model(inputs)
            loss = criterion(y_logits, labels)
            validation_loss += loss.item()
            _, predicted = y_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    return validation_loss / len(val_loader), accuracy



def fitness_calculation(id_num, data_info, params, fn_dict, net_list):
    """ Train and evaluate a model_net using evolved parameters.

    Args:
        id_num: string identifying the generation number and the individual number.
        data_info: dictionary with information about the dataset (number of classes, etc.).
        params: dictionary with parameters necessary for training, including the evolved
            hyperparameters.
        fn_dict: dict with definitions of the possible layers (name and parameters).
        net_list: list with names of layers defining the network, in the order they appear.

    Returns:
        accuracy of the model_net for the validation set.
    """


    model_path = os.path.join(params['experiment_path'], id_num)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    best_model_path = os.path.join(model_path, 'best_model.pth')

    # Load data
    if params['dataset'] == 'Cifar10':
        
        data_info = input.cifar10_info
        data_path = 'cifar10_data'
        
        train_loader, val_loader = input.CIFAR10_loader(data_path, limit_data=params['limit_data'],
                                                        for_train=True, 
                                                        data_aug=params['data_augmentation'],
                                                        batch_size=params['batch_size'])
        
    elif params['dataset'] == 'Cifar100':
        data_info = input.cifar100_info

    model_net = model.NetworkGraph(num_classes=data_info["num_classes"], mu=0.99)
    
    filtered_dict = {key: item for key, item in fn_dict.items() if key in net_list}
    
    model_net.create_functions(fn_dict=filtered_dict, net_list=net_list)

    params['model_net'] = model_net
    params['net_list'] = net_list

    # Training time start counting here.
    params['t0'] = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model_net to the GPU
    inputs, labels = next(iter(train_loader))
    _ = model_net(inputs)
    model_net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    if params['optimizer'] == 'RMSProp':
        optimizer = torch.optim.RMSprop(model_net.parameters(), lr=params['learning_rate'], 
                                    momentum=params['momentum'], weight_decay=params['weight_decay'],
                                    alpha=params['decay'])
    else:
        optimizer = torch.optim.SGD(model_net.parameters(), lr=params['learning_rate'], 
                                    momentum=params['momentum'],weight_decay=params['weight_decay'])

    total_epochs = params['max_epochs']
    train_epochs_without_validation = total_epochs - params['epochs_to_eval']
    best_accuracy = 0.0
    
    # Train the model for 45 epochs without validation
    for epoch in tqdm(range(1, train_epochs_without_validation + 1), desc="Training without validation"):
        train_loss = train_model(model_net, criterion, optimizer, train_loader, device)
        if epoch % 5 == 0:
            print(f"Epoch [{epoch}/{total_epochs}] - Training loss: {train_loss}")
            if time.time() - params['t0'] > TRAIN_TIMEOUT:
                print("Training time exceeded. Returning low accuracy.")
                return 0.0
            
    # Train the model for 5 epochs with validation
    for epoch in tqdm(range(train_epochs_without_validation, total_epochs), desc="Training with validation"):
        train_loss = train_model(model_net, criterion, optimizer, train_loader, device)
        
        if epoch >= train_epochs_without_validation:
            val_loss, accuracy = validate_model(model_net, criterion, val_loader, device)
            print(f"Epoch [{epoch+1}/{total_epochs}] - Training loss: {train_loss} - Validation loss: {val_loss} - Validation accuracy: {accuracy}%")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model_net.state_dict(), best_model_path)
    
    params['t1'] = time.time()
    print(f"Best Validation Accuracy: {best_accuracy}%")
    
    # print time spent in training in minutes
    training_time = round((params['t1'] - params['t0'])/60, 3)
    print(f"Training time: {training_time} minutes")
    
    try:
        accuracy = best_accuracy
    except torch.nn.modules.module.ModuleAttributeError:
        # If the model_net is not valid, it will raise an exception.
        # We return a very low accuracy, so that this individual is not selected.
        accuracy = 0.01
    except RuntimeError:
        # If the model_net is not valid, it will raise an exception.
        # We return a very low accuracy, so that this individual is not selected.
        accuracy = 0.01
    except ValueError:
        # If the model_net is not valid, it will raise an exception.
        # We return a very low accuracy, so that this individual is not selected.
        accuracy = 0.01

    return accuracy
