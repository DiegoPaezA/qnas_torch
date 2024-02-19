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
from typing import Dict, List, Union, Any
from sklearn.metrics import confusion_matrix
from cnn import model, input
from util import create_info_file, init_log
from torch.profiler import profile, record_function, ProfilerActivity
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, CosineAnnealingLR

current_directory = os.path.dirname(os.path.dirname(__file__))
log_directory = os.path.join(current_directory, 'logs')
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
    
log_file = os.path.join(log_directory, 'retrain.log')
LOGGER = init_log("INFO", name=__name__, file_path=log_file)

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

def reset_and_load_best_model(params, best_model_path, device):
    # Reinitialize the original model
    
    best_model = model.NetworkGraph(num_classes=params["num_classes"], mu=0.99)
    filtered_dict = {key: item for key, item in params['fn_dict'].items() if key in params['net_list']}
    best_model.create_functions(fn_dict=filtered_dict, net_list=params['net_list'])

    input_random = torch.randn(params['input_shape'])
    _ = best_model(input_random)
    # Load the state dictionary of the best model into the new model
    best_model.load_state_dict(torch.load(best_model_path))
    best_model.to(device)

    return best_model

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
    
    if params['lr_scheduler'] == 'exponential':
        lr_scheduler = ExponentialLR(optimizer, gamma=0.9)
    elif params['lr_scheduler'] == 'reduce_on_plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.1)
    elif params['lr_scheduler'] == 'cosine':
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=0, last_epoch=-1)
    else:
        lr_scheduler = None
    #for epoch in tqdm(range(1, max_epochs + 1), desc="Retrain Scheme"):
    for epoch in range(1, max_epochs + 1):
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

        if epoch % 50 == 0:
            LOGGER.info(f"Experiment: {params['experiment_path']} - Epoch [{epoch}/{max_epochs}] - Training loss: {train_loss:.2f} - Validation loss: {validation_loss:.2f} - Validation accuracy: {accuracy:.2f}%")
            #print(f"Epoch [{epoch}/{max_epochs}] - Training loss: {train_loss} - Validation loss: {validation_loss} - Validation accuracy: {accuracy}%")

        if lr_scheduler is not None:
            if params['lr_scheduler'] == 'reduce_on_plateau':
                lr_scheduler.step(accuracy)                
            else:
                lr_scheduler.step()
        
    best_model_loaded = reset_and_load_best_model(params, best_model_path, device)
    test_loss, test_accuracy, confusion_matrix = evaluate(best_model_loaded, criterion, test_loader, device, test=True)
    
    LOGGER.info(f"Experiment: {params['experiment_path']} - Test loss: {test_loss:.2f} - Test accuracy: {test_accuracy:.2f}%")
    #print(f"Test loss: {test_loss} - Test accuracy: {test_accuracy}%")
            
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
                    fn_dict: Dict[str, Any],net_list:List[str],
                    train_loader:torch.utils.data.DataLoader, 
                    val_loader:torch.utils.data.DataLoader,
                    test_loader:torch.utils.data.DataLoader) -> Dict[str, Union[List[float], float]]:
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
    
    device = params['device']
    model_path = os.path.join(params['experiment_path'])
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    params['model_path'] = model_path
    
    LOGGER.info(f"Start retraining of the experiment: {params['experiment_path']}")
    # Load data information
    dataset_info = input.available_datasets[params['dataset'].lower()]
    

    model_net = model.NetworkGraph(num_classes=dataset_info["num_classes"], mu=0.99)
    filtered_dict = {key: item for key, item in fn_dict.items() if key in net_list}
    model_net.create_functions(fn_dict=filtered_dict, net_list=net_list)

    params['model_net'] = model_net
    params['net_list'] = net_list
    params['fn_dict'] = fn_dict
    params['num_classes'] = dataset_info["num_classes"]

    device = params['device']
    
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
    
    try:
        results_dict = train(model_net, criterion, optimizer, train_loader, val_loader, test_loader, params, device)
    except Exception as e:
        if "out of memory" in str(e):
            LOGGER.error(f"Out of memory error: {e}")
            results_dict = None
        else:
            LOGGER.error(f"An error occurred during training: {e}")
            results_dict = None
    
    realese_gpu_memory(gpu_name=params['device'])
    
    return results_dict
