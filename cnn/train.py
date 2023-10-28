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
from util import create_info_file

TRAIN_TIMEOUT = 5400

def train_epoch(model, criterion, optimizer, data_loader, device):
    model.train()
    train_loss = 0.0

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        y_logits = model(inputs)
        loss = criterion(y_logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(data_loader)

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

def train(model, criterion, optimizer, train_loader, val_loader, params, device):
    """
    Train a neural network model.

    Args:
        model: Model to be trained.
        criterion: Loss function.
        optimizer: Optimization algorithm.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        max_epochs: Number of epochs to train.
        epochs_to_eval: Number of epochs before starting validation.
        device: Device to run the training on (CPU or GPU).

    Returns:
        training_losses: List of training losses for each epoch.
        validation_losses: List of validation losses for each epoch.
        best_accuracy: Best validation accuracy achieved.
    """
    model.train()
    training_losses = []
    validation_losses = []
    best_accuracy = 0.0
    max_epochs = params['max_epochs']
    epochs_to_eval = params['epochs_to_eval']
    start_eval = max_epochs - epochs_to_eval

    for epoch in tqdm(range(1, max_epochs + 1), desc="Training Fitness Scheme"):
        train_loss = train_epoch(model, criterion, optimizer, train_loader, device)
        training_losses.append(train_loss)

        if epoch < start_eval and (time.time() - params['t0']) > TRAIN_TIMEOUT:
            print("Timeout reached")
            return training_losses, validation_losses, 0.0
        
        if epoch >= start_eval:
            validation_loss, accuracy = evaluate(model, criterion, val_loader, device)
            validation_losses.append(validation_loss)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                # Save the model and the accuracy value

            if epoch % 1 == 0:
                print(f"Epoch [{epoch}/{max_epochs}] - Training loss: {train_loss} - Validation loss: {validation_loss} - Validation accuracy: {accuracy}%")

        if epoch % 1 == 0 and epoch < start_eval:
            print(f"Epoch [{epoch}/{max_epochs}] - Training loss: {train_loss}")
            
    return training_losses, validation_losses, best_accuracy


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
                                                        batch_size=params['batch_size'])
        
    elif params['dataset'] == 'Cifar100':
        data_info = input.cifar100_info

    model_net = model.NetworkGraph(num_classes=data_info["num_classes"], mu=0.99)
    
    filtered_dict = {key: item for key, item in fn_dict.items() if key in net_list}
    
    model_net.create_functions(fn_dict=filtered_dict, net_list=net_list)

    params['model_net'] = model_net
    params['net_list'] = net_list


    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model_net to the GPU
    inputs, labels = next(iter(train_loader))
    _ = model_net(inputs)
    model_net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    if params['optimizer'] == 'RMSProp':
        #optimizer = torch.optim.RMSprop(model_net.parameters(), lr=params['learning_rate'], 
        #                             momentum=params['momentum'], weight_decay=params['weight_decay'],
        #                             alpha=params['decay'])
        optimizer = torch.optim.RMSprop(model_net.parameters(), lr=params['learning_rate'])
    else:
        optimizer = torch.optim.SGD(model_net.parameters(), lr=params['learning_rate'], 
                                    momentum=params['momentum'],weight_decay=params['weight_decay'])

    # Training time start counting here.
    params['t0'] = time.time()
    
    # Train the model in fitness scheme
    
    _, _, best_accuracy = train(model_net, criterion, optimizer, train_loader, val_loader,params,device)

    params['t1'] = time.time()
    
    return best_accuracy
