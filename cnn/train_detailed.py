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
from sklearn.metrics import confusion_matrix
from cnn import model, input
from util import create_info_file


TRAIN_TIMEOUT = 5400

def realese_gpu_memory(gpu_name='cuda:0'):
    """
    Release GPU memory.
    """
    # Set the device to GPU named "cuda:1"
    torch.cuda.set_device(gpu_name)
    torch.cuda.empty_cache()

    # Print memory statistics
    #print(f"Allocated GPU memory: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
    #print(f"Reserved GPU memory: {torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB")

def compute_confusion_matrix(model, data_loader, device):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            y_logits = model(inputs)
            _, predicted = y_logits.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    conf_matrix = confusion_matrix(all_labels, all_predictions)
    return conf_matrix

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

def evaluate(model, criterion, data_loader, device, test=False):
    model.eval()
    eval_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            y_logits = model(inputs)
            loss = criterion(y_logits, labels)
            eval_loss += loss.item()
            _, predicted = y_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    eval_loss /= len(data_loader)
    
    if test:
        confusion_matrix = compute_confusion_matrix(model, data_loader, device)
        return eval_loss, accuracy, confusion_matrix

    return eval_loss, accuracy

def train(model: torch.nn.Module,
          criterion: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader,
          test_loader: torch.utils.data.DataLoader,
          params: Dict[str, Union[int, float, str]],
          device: torch.device) -> Dict[str, Union[List[float], float]]:
    """
    Retrain a convolutional neural network model.

    Args:
        model (Module): Model to be trained.
        criterion (Module): Loss function.
        optimizer (Optimizer): Optimization algorithm.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        test_loader (DataLoader): DataLoader for the test set.
        params (Dict[str, Union[int, float, str]]): Dictionary with parameters necessary for training.
            - 'max_epochs' (int): Number of epochs to train.
            - 'model_path' (str): Path to save the trained model.
        device (torch.device): Device to run the training on (CPU or GPU).

    Returns:
        Dict[str, Union[List[float], float]]: Dictionary with the training results.
        
        - 'training_losses' (List[float]): List of training losses for each epoch.
        - 'training_accuracies' (List[float]): List of training accuracies for each epoch.
        - 'validation_losses' (List[float]): List of validation losses for each epoch.
        - 'validation_accuracies' (List[float]): List of validation accuracies for each epoch.
        - 'best_accuracy' (float): Best validation accuracy achieved.
        - 'test_loss' (float): Loss on the test set.
        - 'test_accuracy' (float): Accuracy on the test set.
        - 'confusion_matrix' (numpy.ndarray): Confusion matrix on the test set.
    """
    model.train()
    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []
    best_accuracy = 0.0
    training_results = {}
    max_epochs = params['max_epochs']

    best_model_path = os.path.join(params['model_path'], 'best_model.pth')

    for epoch in tqdm(range(1, max_epochs + 1), desc="Retrain Scheme"):
        train_loss, train_accuracy = train_epoch(model, criterion, optimizer, train_loader, device)
        training_losses.append(train_loss)
        training_accuracies.append(train_accuracy)
        
        validation_loss, accuracy = evaluate(model, criterion, val_loader, device)
        validation_losses.append(validation_loss)
        validation_accuracies.append(accuracy)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)
            create_info_file(params['model_path'], {'best_accuracy': best_accuracy}, 'best_accuracy.txt')

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{max_epochs}] - Training loss: {train_loss} - Validation loss: {validation_loss} - Validation accuracy: {accuracy}%")

    test_loss, test_accuracy, confusion_matrix = evaluate(model, criterion, test_loader, device, test=True)
    
    print(f"Test loss: {test_loss} - Test accuracy: {test_accuracy}%")
            
    params['t1'] = time.time()
    
    create_info_file(params['model_path'], params, 'retraining_params.txt')
    
    training_results['training_losses'] = training_losses
    training_results['training_accuracies'] = training_accuracies
    training_results['validation_losses'] = validation_losses
    training_results['validation_accuracies'] = validation_accuracies
    training_results['best_accuracy'] = best_accuracy
    training_results['test_loss'] = test_loss
    training_results['test_accuracy'] = test_accuracy
    training_results['confusion_matrix'] = confusion_matrix
            
    return training_results


def train_and_eval(params: Dict[str, Any], 
                    fn_dict: Dict[str, Any], 
                    net_list: List[str]) -> Dict[str, Union[List[float], float]]:
    """
    This function retrains and evaluates a convolutional neural network model using the specified
    configuration.

    Args:
        params (Dict[str, Any]): A dictionary with parameters necessary for training, including.
        fn_dict (Dict[str, Any]): A dictionary with definitions of the possible layers, including
            their names and parameters.
        net_list (List[str]): A list with names of layers defining the network, in the order they appear.

    Returns:
        Dict[str, Union[List[float], float]]: Dictionary with the training results.
        
        - 'training_losses' (List[float]): List of training losses for each epoch.
        - 'training_accuracies' (List[float]): List of training accuracies for each epoch.
        - 'validation_losses' (List[float]): List of validation losses for each epoch.
        - 'validation_accuracies' (List[float]): List of validation accuracies for each epoch.
        - 'best_accuracy' (float): Best validation accuracy achieved.
        - 'test_loss' (float): Loss on the test set.
        - 'test_accuracy' (float): Accuracy on the test set.
        - 'confusion_matrix' (numpy.ndarray): Confusion matrix on the test set.
    """
    
    
    model_path = os.path.join(params['experiment_path'], params['retrain_folder'])
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    params['model_path'] = model_path
    # Load data
    if params['dataset'] == 'Cifar10':
        
        data_info = input.cifar10_info
        data_path = params['data_path']
        
        train_loader, val_loader = input.CIFAR10_loader(data_path,
                                                        for_train=True, 
                                                        data_aug=params['data_augmentation'],
                                                        batch_size=params['batch_size'],
                                                        eval_batch_size=params['eval_batch_size'])
        test_loader = input.CIFAR10_loader(data_path,
                                             for_train=False,
                                             eval_batch_size=params['test_batch_size'])
        
    elif params['dataset'] == 'Cifar100':
        data_info = input.cifar100_info

    model_net = model.NetworkGraph(num_classes=data_info["num_classes"], mu=0.99)
    
    filtered_dict = {key: item for key, item in fn_dict.items() if key in net_list}
    
    model_net.create_functions(fn_dict=filtered_dict, net_list=net_list)

    params['model_net'] = model_net
    params['net_list'] = net_list
    params['num_classes'] = data_info["num_classes"]

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
        optimizer = torch.optim.SGD(model_net.parameters(), lr=params['learning_rate'], momentum=0.9)

    # Training time start counting here.
    params['t0'] = time.time()
    
    # Train the model in retreining mode
    results_dict = train(model_net, criterion, optimizer, train_loader, val_loader, test_loader, params, device)
    
    realese_gpu_memory(gpu_name=params['device'])
    
    return results_dict
