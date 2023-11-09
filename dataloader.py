import torch
import random
import util
import os
from time import time
import numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToTensor, Resize, Compose, RandomCrop, RandomHorizontalFlip, Normalize

cifar10_info = {
  'dataset': 'CIFAR10',
  'mean': [0.491400808095932, 0.48215898871421814, 0.44653093814849854],
  'std': [0.24703224003314972, 0.24348513782024384, 0.26158785820007324],
  'shape': [3, 32, 32]
}

cifar100_info = {
  'dataset': 'CIFAR100',
  'mean': [0.5070757865905762, 0.48655030131340027, 0.4409191310405731],
  'std': [0.2673342823982239, 0.2564384639263153, 0.2761504650115967],
  'shape': [3, 32, 32]
}

# Function to balance the dataset by class
def balance_dataset(dataset, indices, max_samples_per_class):
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

def CIFAR10_loader(data_path:str, train_split=0.9,batch_size=24,limit_data=None,seed=None,info:dict=None,data_aug=True,for_train=True, num_workers=2):
  """
  This function creates a dataloader for the CIFAR10 dataset.
  
  Args:
    data_path: Path to the data folder
    train_split: float (0,1 - default 0.9)
      Percentage of the data to be used for training
    batch_size: int (default 24)
    limit_data: int (default None)
      Limit the number of samples to be used for training and validation
    seed: int (default None)
      Seed for the random number generator
    info: dict (default None)
      Dictionary containing the mean, std and shape of the dataset
    data_aug: bool (default True)
      If True, data augmentation and normalization is applied to the dataset
    for_train: bool (default True)
      If True, returns train and validation dataloaders, otherwise returns test dataloader
    num_workers: int (default 2)
      Number of workers for the dataloader
  Returns:
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
  """
  error_msg = "[!] train_split should be in the range [0, 1]."
  assert ((train_split >= 0) and (train_split <= 1)), error_msg
  
  if seed is None:
    seed = int(time())
    random.seed(seed)
    torch.manual_seed(seed)
    
  file_path = os.path.join(data_path, 'data_info.txt')
  file_exists = util.check_file_exists(file_path)
  
  if info is None and file_exists:
    #print("Loading info locally")
    mean = cifar10_info['mean']
    std = cifar10_info['std']
    channels,height,width  = cifar10_info['shape']
  else:
    # Calculate mean and standard deviation from the entire dataset
    dataset = CIFAR10(data_path, download=True,train=True, transform=ToTensor())
    loader = DataLoader(dataset, batch_size=len(dataset), num_workers=num_workers, shuffle=False)

    data = next(iter(loader))
    mean = data[0].mean(dim=(0, 2, 3)).tolist()    # Calculate mean for each channel
    std = data[0].std(dim=(0, 2, 3)).tolist()      # Calculate std for each channel

    channels,height,width = dataset[0][0].shape
      
  transform = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
      ])
  
  if data_aug:
    pad = 4
    train_transform = Compose([
      Resize((height + pad, width + pad)),
      RandomCrop((height, width)),
      RandomHorizontalFlip(),
      ToTensor(),
      Normalize(mean=mean, std=std)
    ])
  else:
    train_transform = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
      ])
  
  
  # Load CIFAR-10 dataset
  train_dataset = CIFAR10(data_path, train=True, download=True, transform=train_transform)
  valid_dataset = CIFAR10(data_path, train=True, download=True, transform=transform)
  test_dataset = CIFAR10(data_path, train=False, download=True, transform=transform)
  
  if not for_train:
    test_loader = DataLoader(
      test_dataset,
      batch_size=batch_size,
      num_workers=num_workers,
      shuffle=False,
      pin_memory=True)
    print("All set for testing!")
    return test_loader
  
  # Split train_dataset into train and validation
  val_split = 1 - train_split
  num_train = len(train_dataset)
  indices = list(range(num_train))
  split = int(np.floor(val_split * num_train)) + 1
  
  np.random.shuffle(indices)
  train_indices, val_indices = indices[split:], indices[:split]
  
  if limit_data is not None and limit_data < num_train:
    train_samples = int(train_split * limit_data)
    val_samples = int(limit_data - train_samples)
    
    max_samples_per_class_train = train_samples/10
    max_samples_per_class_val = val_samples/10

    train_indices_limited = balance_dataset(train_dataset, train_indices, max_samples_per_class_train)
    val_indices_limited = balance_dataset(valid_dataset, val_indices, max_samples_per_class_val)

    train_dataset = Subset(train_dataset, train_indices_limited)
    valid_dataset = Subset(valid_dataset, val_indices_limited)

  else:
    train_dataset = Subset(train_dataset, train_indices)
    valid_dataset = Subset(valid_dataset, val_indices)
      
  train_loader = DataLoader(
      train_dataset,
      batch_size=batch_size,
      num_workers=num_workers,
      shuffle=True,
      pin_memory=True)
  
  val_loader = DataLoader(
      valid_dataset,
      batch_size=batch_size,
      num_workers=num_workers,
      shuffle=False,
      pin_memory=True)
  
  
  #print(f"Limiting training samples to {len(train_dataset)} and validation samples to {len(val_dataset)}")  
  info_dict = {'dataset': f'CIFAR{10}'}
  info_dict['seed'] = seed
  
  info_dict['train_records'] = len(train_dataset)
  info_dict['valid_records'] = len(valid_dataset)
  info_dict['test_records'] = len(test_dataset)
  
  info_dict['shape'] = [channels, height, width]
    
  info_dict['mean'] = mean
  info_dict['std'] = std
  
  util.create_info_file(out_path=data_path, info_dict=info_dict)
  
  print("All set for training!")
  
  return train_loader, val_loader



def CIFAR100_loader(data_path:str, train_split=0.9,batch_size=24,limit_data=None,seed=None,info:dict=None,data_aug=True,for_train=True, num_workers=2):
  """
  This function creates a dataloader for the CIFAR100 dataset.
  
  Args:
    data_path: Path to the data folder
    train_split: float (0,1 - default 0.9)
      Percentage of the data to be used for training
    batch_size: int (default 24)
    limit_data: int (default None)
      Limit the number of samples to be used for training and validation
    seed: int (default None)
      Seed for the random number generator
    info: dict (default None)
      Dictionary containing the mean, std and shape of the dataset
    data_aug: bool (default True)
      If True, data augmentation and normalization is applied to the dataset
    for_train: bool (default True)
      If True, returns train and validation dataloaders, otherwise returns test dataloader
    num_workers: int (default 2)
      Number of workers for the dataloader
  Returns:
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
  """
  
  if seed is None:
    seed = int(time())
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
  file_path = os.path.join(data_path, 'data_info.txt')
  file_exists = util.check_file_exists(file_path)
  
  if info is None and file_exists:
    #print("Loading info locally")
    mean = cifar100_info['mean']
    std = cifar100_info['std']
    channels,height,width  = cifar100_info['shape']
  else:
    # Calculate mean and standard deviation from the entire dataset
    dataset = CIFAR100(data_path, download=True, transform=ToTensor())
    loader = DataLoader(dataset, batch_size=len(dataset), num_workers=num_workers, shuffle=False)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    
    for images, _ in loader:
        mean += torch.mean(images, dim=(0, 2, 3))
        std += torch.std(images, dim=(0, 2, 3))
    mean /= len(loader)
    std /= len(loader)
    mean = mean.tolist()
    std = std.tolist()
    channels,height,width = dataset[0][0].shape
      
  transform = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
      ])
  
  # Load CIFAR-100 dataset
  train_dataset_raw = CIFAR100(data_path, train=True, download=True, transform=transform)
  test_dataset = CIFAR100(data_path, train=False, download=True, transform=transform)
  
  if not for_train:
    test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True)
    print("All set for testing!")
    return test_loader
  
  # Split train_dataset into train and validation
  val_split = 1 - train_split
  
  # Create train and validation indices using stratified sampling
  labels = np.array(train_dataset_raw.targets)
  splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split)
  train_indices, val_indices = next(splitter.split(labels, labels)) 

  total_train_samples = len(train_indices)
  
  if limit_data is not None and limit_data < total_train_samples:
    train_samples = int(train_split * limit_data)
    val_samples = int(limit_data - train_samples)
    
    max_samples_per_class_train = train_samples/10
    max_samples_per_class_val = val_samples/10

    # Create a dictionary to track the number of samples per class
    train_samples_per_class = defaultdict(int)
    val_samples_per_class = defaultdict(int)
    
    train_indices_limited = []
    val_indices_limited = []

    random.shuffle(train_indices)
    random.shuffle(val_indices)
    
    # Iterate through the training indices to limit the number of samples per class
    for idx in train_indices:
      _, target = train_dataset_raw[idx]
      if not train_samples_per_class[target] >= max_samples_per_class_train:
          train_indices_limited.append(idx)
          train_samples_per_class[target] += 1

    # Iterate through the validation indices to limit the number of samples per class
    for idx in val_indices:
      _, target = train_dataset_raw[idx]
      if not val_samples_per_class[target] >= max_samples_per_class_val:
        val_indices_limited.append(idx)
        val_samples_per_class[target] += 1

    # Create Subset objects using the limited indices
    train_dataset = Subset(train_dataset_raw, train_indices_limited)
    val_dataset = Subset(train_dataset_raw, val_indices_limited)
  else:
    train_dataset = Subset(train_dataset_raw, train_indices)
    val_dataset = Subset(train_dataset_raw, val_indices)
  
  if data_aug:
    pad = 4
    train_transform = Compose([
      Resize((height + pad, width + pad)),
      RandomCrop((height, width)),
      RandomHorizontalFlip()
    ])
    train_subset = [(train_transform(sample), target) for sample, target in train_dataset]
  else:
    train_subset = train_dataset
  
  
  train_loader = DataLoader(
      train_subset,
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=True,
      shuffle=True)
  
  val_loader = DataLoader(
      val_dataset,
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=True)
  
  #print(f"Limiting training samples to {len(train_dataset)} and validation samples to {len(val_dataset)}")  
  info_dict = {'dataset': f'CIFAR{100}'}
  info_dict['seed'] = seed
  
  info_dict['train_records'] = len(train_dataset)
  info_dict['valid_records'] = len(val_dataset)
  info_dict['test_records'] = len(test_dataset)
  
  info_dict['shape'] = [channels, height, width]
    
  info_dict['mean'] = mean
  info_dict['std'] = std
  
  util.create_info_file(out_path=data_path, info_dict=info_dict)
  
  print("All set for training!")
  
  return train_loader, val_loader