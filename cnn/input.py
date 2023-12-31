""" Copyright (c) 2023, Diego Páez
* Licensed under the MIT license

- Input module - Generic data loader for PyTorch, supporting various datasets and data augmentation.

"""
import torch
import random
import util
import os
from time import time
import numpy as np
from collections import defaultdict
import torchvision.datasets
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToTensor, Resize, Compose, RandomCrop, Normalize, TrivialAugmentWide

cifar10_info = {
  'dataset': 'CIFAR10',
  'mean': [0.491400808095932, 0.48215898871421814, 0.44653093814849854],
  'std': [0.24703224003314972, 0.24348513782024384, 0.26158785820007324],
  'shape': [3, 32, 32], 
  'num_classes': 10
}

cifar100_info = {
  'dataset': 'CIFAR100',
  'mean': [0.5070757865905762, 0.48655030131340027, 0.4409191310405731],
  'std': [0.2673342823982239, 0.2564384639263153, 0.2761504650115967],
  'shape': [3, 32, 32],
  'num_classes': 100
}

available_datasets = {
  'cifar10': cifar10_info,
  'cifar100': cifar100_info
}

class GenericDataLoader:
  """A generic data loader for PyTorch, supporting various datasets and data augmentation."""
  def __init__(self, params: dict, train_split=0.9, seed=None, info: dict = None):
    """
    Initialize the GenericDataLoader.
    
    Parameters:
      params (dict): Dictionary containing parameters.
      train_split (float): Split ratio for training data.
      seed (int): Seed for randomization.
      info (dict): Additional information about the dataset.
      
    Returns:
      None
    """
    self.params = params
    self.train_split = train_split
    error_msg = "[!] train_split should be in the range [0, 1]."
    assert 0 <= self.train_split <= 1, error_msg    
    if seed is None:
        seed = int(time())
        random.seed(seed)
        torch.manual_seed(seed)
    self.info_dict = {'dataset': f'{self.params["dataset"]}'}
    self.info_dict['seed'] = seed
    self.download_status = not os.path.exists(self.params['data_path'])
    if info is None:
        # Check if the dataset is available in the available_datasets dict
        if self.params['dataset'].lower() in available_datasets.keys():
          dataset_info = getattr(self, f"{self.params['dataset'].lower()}_info", None)
          self.mean = dataset_info['mean']
          self.std = dataset_info['std']
          self.channels, self.height, self.width = dataset_info['shape']
          self.num_classes = dataset_info['num_classes']
        # check if the dataset is in the torchvision datasets and compute the parameters
        elif hasattr(torchvision.datasets, self.params['dataset'].upper()):
          dataset_class = getattr(torchvision.datasets, self.params['dataset'].upper())
          dataset_ = dataset_class(self.params['data_path'], download=True, transform=ToTensor())
          loader = DataLoader(dataset_, batch_size=len(dataset_), num_workers=0, shuffle=False)
          data = next(iter(loader))
          self.mean = data[0].mean(dim=(0, 2, 3)).tolist()
          self.std = data[0].std(dim=(0, 2, 3)).tolist()
          self.channels, self.height, self.width = dataset_[0][0].shape
          self.num_classes = len(dataset_.classes)
        else:
          raise ValueError(f"Dataset class {self.params['dataset']} not found in torchvision.datasets or available_datasets.")
    else:
      raise NotImplementedError('Custom dataset is not implemented yet.')             
    self.info_dict['shape'] = [self.channels, self.height, self.width]  
    self.info_dict['mean'] = self.mean
    self.info_dict['std'] = self.std
    # Transformations
    transform = Compose([
        ToTensor(),
        Normalize(mean=self.mean, std=self.std)
    ])
    if self.params['data_aug']:
        pad = 4
        train_transform = Compose([
            Resize((self.height + pad, self.width + pad)),
            RandomCrop((self.height, self.width)),
            TrivialAugmentWide(num_magnitude_bins=31),
            ToTensor(),
            Normalize(mean=self.mean, std=self.std)
        ])
    else:
        train_transform = Compose([
            ToTensor(),
            Normalize(mean=self.mean, std=self.std)
        ])     
    # create the dataset
    if hasattr(torchvision.datasets, self.params['dataset'].upper()):
      dataset_class = getattr(torchvision.datasets, self.params['dataset'].upper())
      self.train_dataset = dataset_class(self.params['data_path'], train=True, download=self.download_status,transform=train_transform)
      self.valid_dataset = dataset_class(self.params['data_path'], train=True, download=self.download_status,transform=transform)
      self.test_dataset = dataset_class(self.params['data_path'], train=False, download=self.download_status,transform=transform)
    else:
      raise NotImplementedError('Custom dataset is not implemented yet.')
        
  # Function to balance the dataset by class
  def _balance_dataset(self,dataset, indices, max_samples_per_class):
    """
      Balance the dataset by class.
      
      Parameters:
          dataset: The dataset object.
          indices (list): List of indices.
          max_samples_per_class (int): Maximum samples allowed per class.
      
      Returns:
          list: Balanced indices.
    """
    class_samples = defaultdict(int)
    balanced_indices = []
    random.shuffle(indices)
    for idx in indices:
        _, target = dataset[idx]
        if class_samples[target] < max_samples_per_class:
            balanced_indices.append(idx)
            class_samples[target] += 1
            if len(balanced_indices) == max_samples_per_class * len(set(class_samples.values())):
                break
    return balanced_indices
      
  def get_loader(self, for_train=True):
    """
    Get data loader for training or validation/testing.

    Parameters:
      for_train (bool): If True, returns the training loader; otherwise, returns the validation/testing loader.

    Returns:
      DataLoader: PyTorch DataLoader.
    """
        
    if not for_train:
      test_loader = DataLoader(
        self.test_dataset,
        batch_size=self.params['eval_batch_size'],
        num_workers=0,
        shuffle=False,
        pin_memory=True)
      return test_loader
    
    val_split = 1 - self.train_split
    num_train = len(self.train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_split * num_train)) + 1
    
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    if self.params['limit_data'] and self.params['limit_data_value'] < num_train:
      train_samples = int(self.train_split * self.params['limit_data_value'])
      val_samples = int(self.params['limit_data_value'] - train_samples)
      
      max_samples_per_class_train = train_samples/self.num_classes
      max_samples_per_class_val = val_samples/self.num_classes
      train_indices_limited = self._balance_dataset(self.train_dataset, train_indices, max_samples_per_class_train)
      val_indices_limited = self._balance_dataset(self.valid_dataset, val_indices, max_samples_per_class_val)
      self.train_dataset = Subset(self.train_dataset, train_indices_limited)
      self.valid_dataset = Subset(self.valid_dataset, val_indices_limited)
    else:
      self.train_dataset = Subset(self.train_dataset, train_indices)
      self.valid_dataset = Subset(self.valid_dataset, val_indices)
      
    train_loader = DataLoader(
      self.train_dataset,
      batch_size=self.params['batch_size'],
      num_workers=0,
      shuffle=True,
      pin_memory=True)

    val_loader = DataLoader(
      self.valid_dataset,
      batch_size=self.params['eval_batch_size'],
      num_workers=0,
      shuffle=False,
      pin_memory=True)
  
    self.info_dict['train_records'] = len(self.train_dataset)
    self.info_dict['valid_records'] = len(self.valid_dataset)
    self.info_dict['test_records'] = len(self.test_dataset)
            
    util.create_info_file(out_path=self.params['data_path'], info_dict=self.info_dict)
    
    return train_loader, val_loader