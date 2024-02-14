""" Copyright (c) 2023, Diego Páez
* Licensed under the MIT license

- Input module - Generic data loader for PyTorch, supporting various datasets and data augmentation.

"""
import torch
import random
import util
import os
from time import time
import torchvision.datasets
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.transforms import ToTensor, Resize, Compose, RandomCrop, Normalize, TrivialAugmentWide, ToPILImage
from sklearn.model_selection import StratifiedShuffleSplit

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

class MyDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
          x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)
        
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
          dataset_info = available_datasets[self.params['dataset'].lower()]
          mean = dataset_info['mean']
          std = dataset_info['std']
          channels, height, width = dataset_info['shape']
          self.num_classes = dataset_info['num_classes']
          
        # check if the dataset is in the torchvision datasets and compute the parameters
        elif hasattr(torchvision.datasets, self.params['dataset'].upper()):
          dataset_class = getattr(torchvision.datasets, self.params['dataset'].upper())
          dataset_ = dataset_class(self.params['data_path'], download=True, transform=ToTensor())
          loader = DataLoader(dataset_, batch_size=len(dataset_), num_workers=0, shuffle=False)
          data = next(iter(loader))
          mean = data[0].mean(dim=(0, 2, 3)).tolist()
          std = data[0].std(dim=(0, 2, 3)).tolist()
          channels, height, width = dataset_[0][0].shape
          self.num_classes = len(dataset_.classes)
        else:
          raise ValueError(f"Dataset class {self.params['dataset']} not found in torchvision.datasets or available_datasets.")
    else:
      raise NotImplementedError('Custom dataset is not implemented yet.')             
    
    self.info_dict['shape'] = [channels, height, width]  
    self.info_dict['mean'] = mean
    self.info_dict['std'] = std
    
    # Transformations
    self.transform = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    
    if self.params['data_augmentation']:
        pad = 4
        self.train_transform = Compose([
            Resize((height + pad, width + pad)),
            RandomCrop((height, width)),
            TrivialAugmentWide(num_magnitude_bins=31),
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])
    else:
        self.train_transform = Compose([
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])
      
  def get_loader(self, for_train=True, pin_memory_device="cuda"):
    """
    Get data loader for training or validation/testing.

    Parameters:
      for_train (bool): If True, returns the training and val loader; otherwise, returns the testing loader.

    Returns:
      DataLoader: PyTorch DataLoader.
    """
    # create the dataset
    if hasattr(torchvision.datasets, self.params['dataset'].upper()):
      dataset_class = getattr(torchvision.datasets, self.params['dataset'].upper())
      full_dataset = dataset_class(self.params['data_path'], train=True, download=self.download_status)
      test_dataset = dataset_class(self.params['data_path'], train=False, download=self.download_status,transform=self.transform)
      self.download_status = not os.path.exists(self.params['data_path'])
    else:
      raise NotImplementedError('Custom dataset is not implemented yet.')
        
    if not for_train:
      test_loader = DataLoader(
        test_dataset,
        batch_size=self.params['eval_batch_size'],
        num_workers=self.params['num_workers'],
        shuffle=False,
        pin_memory=True,
        pin_memory_device=pin_memory_device)
      return test_loader
    
    val_split = 1 - self.train_split
    # Get the labels and create StratifiedShuffleSplit
    labels = full_dataset.targets
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=val_split)

    # Get the training and validation indices
    train_idx, val_idx = next(stratified_split.split(labels, labels))
    
    num_train = len(full_dataset)
    
    # split the dataset into train and validation
    if self.params['limit_data'] and self.params['limit_data_value'] < num_train:
      train_samples = (int(self.train_split * self.params['limit_data_value'])) 
      val_samples = (int(self.params['limit_data_value'] - train_samples))
      
      train_count_per_class = train_samples // self.num_classes
      val_count_per_class = val_samples // self.num_classes
      
      train_indices = []
      val_indices = []
      for label in set(labels):
        label_indices = [i for i in train_idx if labels[i] == label]
        train_indices.extend(label_indices[:train_count_per_class])

        label_indices = [i for i in val_idx if labels[i] == label]
        val_indices.extend(label_indices[:val_count_per_class])
      
      train_subset = Subset(full_dataset, train_indices)
      valid_subset = Subset(full_dataset, val_indices)

    else:
      train_subset = Subset(full_dataset, train_idx)
      valid_subset = Subset(full_dataset, val_idx)
    
    # Apply transformations to the datasets
    train_dataset = MyDataset(train_subset, transform=self.train_transform)
    valid_dataset = MyDataset(valid_subset, transform=self.transform)
    
    train_loader = DataLoader(
      train_dataset,
      batch_size=self.params['batch_size'],
      num_workers=self.params['num_workers'],
      shuffle=True,
      pin_memory=True, 
      pin_memory_device=pin_memory_device)

    val_loader = DataLoader(
      valid_dataset,
      batch_size=self.params['eval_batch_size'],
      num_workers=self.params['num_workers'],
      shuffle=False,
      pin_memory=True, 
      pin_memory_device=pin_memory_device)
  
    self.info_dict['train_records'] = len(train_dataset)
    self.info_dict['valid_records'] = len(valid_dataset)
    self.info_dict['test_records'] = len(test_dataset)
            
    util.create_info_file(out_path=self.params['data_path'], info_dict=self.info_dict)
    
    return train_loader, val_loader