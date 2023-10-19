"""
based on: https://github.com/mdrs-thiago/PUC_Redes_Neurais/blob/main/pos_grad/lista%201/model_utils.py
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

def train(model: torch.nn.Module , train_loader: torch.utils.data.DataLoader, 
         val_set: Tuple[torch.Tensor, torch.Tensor], epochs: int, device: torch.device, 
         lr: float, binary: bool = False, optimizer_name:str="RMSProp", skip: int = 1) -> Tuple[Dict[str, List[float]], torch.Tensor]:
    """
    Trains a Pytorch model on a given training data.

    Parameters:
    model (torch.nn.Module): The model to be trained
    train_loader (DataLoader): The training data in the form of a Pytorch DataLoader
    val_set (tuple): A tuple containing the validation data and labels
    epochs (int): The number of times the training data should be passed through the model
    device (str or torch.device): The device on which to perform the computations (e.g. 'cpu' or 'cuda')
    lr (float): The learning rate for the optimizer
    binary (bool): Boolean indicating whether the task is binary classification or not. Default: True
    optimizer_name (str): The name of the optimizer to be used. Default: 'RMSProp'
    skip (int): The number of epochs after which the training and validation results will be printed. Default: 1

    Returns:
    tuple: A tuple containing the training history and the predicted labels after training
    """


    if binary:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    if optimizer_name == 'RMSProp':
        #optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=decay, momentum=momentum)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    history = {'acc_train' : [], 'loss_train': [], 'acc_val': [], 'loss_val': []}

    for e in tqdm(range(1, epochs+1)):

        y_hat = np.array([])

        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X, y = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            
            y_pred = model(X)
            
            loss = criterion(y_pred, y)
            if binary:
                acc = binary_acc(y_pred,y)
            else:
                acc = accuracy(y_pred, y)
            
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
            train_epoch_acc += acc.item()
            y_p = torch.argmax(y_pred, dim=1)
            y_hat = np.concatenate((y_hat, y_p))


        model.eval()
        _, val_loss, val_acc = evaluate(model, val_set, criterion, binary=binary)

        history['acc_train'].append(train_epoch_acc/len(train_loader))
        history['loss_train'].append(train_epoch_loss/len(train_loader))
        history['acc_val'].append(val_acc)
        history['loss_val'].append(val_loss)

        if e%skip == 0:
            print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.3f} | Val Loss: {val_loss:.4f} | Train Acc: {train_epoch_acc/len(train_loader):.4f}| Val Acc: {val_acc:.4f}')
    return history, y_hat


def evaluate(model: torch.nn.Module, val_set: Tuple[torch.tensor, torch.tensor], 
            criterion: torch.nn.Module, binary:bool =True) -> Tuple[torch.tensor, float, float]:
    """
    Evaluates a Pytorch model on a given dataset.
    Parameters:
    model (torch.nn.Module): The model to be evaluated
    data (tuple): A tuple containing the data and labels
    criterion (torch.nn.Module): The loss function to be used
    binary (bool): Boolean indicating whether the task is binary classification or not. Default: True
    Returns:
    tuple: A tuple containing predicted labels, loss, and accuracy
    """
    
    X = val_set.X_data
    y = val_set.y_data
    
    with torch.no_grad():
        y_pred = model(X)
    loss = criterion(y_pred, y)
    if binary:
        acc = binary_acc(y_pred, y)
    else:
        acc = accuracy(y_pred, y)
    y_pred = torch.argmax(y_pred, dim=1)

    return y_pred, loss.item(), acc.item()

def fitness_calculation(id_num, data_info, params, fn_dict, net_list):
    """ Train and evaluate a model using evolved parameters.

    Args:
        id_num: string identifying the generation number and the individual number.
        data_info: dictionary with information about the dataset (number of classes, etc.).
        params: dictionary with parameters necessary for training, including the evolved
            hyperparameters.
        fn_dict: dict with definitions of the possible layers (name and parameters).
        net_list: list with names of layers defining the network, in the order they appear.

    Returns:
        accuracy of the model for the validation set.
    """


    model_path = os.path.join(params['experiment_path'], id_num)

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

    net = model.NetworkGraph(num_classes=data_info["num_classes"], mu=0.99)
    filtered_dict = {key: item for key, item in fn_dict.items() if key in net_list}
    net.create_functions(fn_dict=filtered_dict, net_list=net_list)

    params['net'] = net
    params['net_list'] = net_list

    # Training time start counting here. It needs to be defined outside model_fn(), to make it
    # valid in the multiple calls to classifier.train(). Otherwise, it would be restarted.
    params['t0'] = time.time()
    


    try:
        # accuracy = train_and_eval(params=hparams, run_config=config,
        #                           train_input_fn=train_input_fn,
        #                           eval_input_fn=eval_input_fn)
        accuracy = 0
    except torch.nn.modules.module.ModuleAttributeError:
        # If the model is not valid, it will raise an exception.
        # We return a very low accuracy, so that this individual is not selected.
        accuracy = 0.01
    except RuntimeError:
        # If the model is not valid, it will raise an exception.
        # We return a very low accuracy, so that this individual is not selected.
        accuracy = 0.01
    except ValueError:
        # If the model is not valid, it will raise an exception.
        # We return a very low accuracy, so that this individual is not selected.
        accuracy = 0.01

    return accuracy
