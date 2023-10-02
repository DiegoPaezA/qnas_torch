""" Copyright (c) 2023, Diego Páez
* Licensed under the MIT license

- Input function and dataset info classes.

"""
import torch
import random
import util
import os
from time import time
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, random_split, Dataset, Subset
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

class CustomCIFAR(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, target = self.data[index], self.targets[index]

        if self.transform:
            image = self.transform(image)

        return image, target

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
    mean = cifar10_info['mean']
    std = cifar10_info['std']
    channels,height,width  = cifar10_info['shape']
  else:
    # Calculate mean and standard deviation from the entire dataset
    dataset = CIFAR10(data_path, download=True, transform=ToTensor())
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
  
  if data_aug:
    pad = 4
    train_transform = Compose([
      Resize((height + pad, width + pad)),
      RandomCrop((height, width)),
      RandomHorizontalFlip()
    ])
  else:
    train_transform = None

  # Load CIFAR-10 dataset
  train_dataset = CIFAR10(data_path, train=True, download=True, transform=transform)
  test_dataset = CIFAR10(data_path, train=False, download=True, transform=transform)
  
  if not for_train:
    test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True)
    print("All set for testing!")
    return test_loader
  
  # Split train_dataset into train and validation
  train_size = int(train_split * len(train_dataset))
  val_size = len(train_dataset) - train_size
  train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

  # Create tensors for train and validation data
  train_images = torch.stack([image for image, _ in train_dataset])
  train_labels = torch.tensor([label for _, label in train_dataset])
  val_images = torch.stack([image for image, _ in val_dataset])
  val_labels = torch.tensor([label for _, label in val_dataset])
  
  # Create custom dataset objects
  train_custom_dataset = CustomCIFAR(data=train_images, targets=train_labels, transform=train_transform)
  val_custom_dataset = CustomCIFAR(data=val_images, targets=val_labels)
  
  total_train_samples = len(train_custom_dataset)
  total_val_samples = len(val_custom_dataset)
  
  if limit_data is not None and limit_data < total_train_samples:
      train_samples = int(train_split * limit_data)
      val_samples = int(limit_data - train_samples)

      train_indices = random.sample(range(total_train_samples), train_samples)
      val_indices = random.sample(range(total_val_samples), val_samples)
      
      train_dataset = Subset(train_custom_dataset, train_indices)
      val_dataset = Subset(val_custom_dataset, val_indices)
  else:
      train_dataset = train_custom_dataset
      val_dataset = val_custom_dataset
      
  train_loader = DataLoader(
      train_dataset,
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
  info_dict = {'dataset': f'CIFAR{10}'}
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
  
  if data_aug:
    pad = 4
    train_transform = Compose([
      Resize((height + pad, width + pad)),
      RandomCrop((height, width)),
      RandomHorizontalFlip()
    ])
  else:
    train_transform = None

  # Load CIFAR-100 dataset
  train_dataset = CIFAR100(data_path, train=True, download=True, transform=transform)
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
  train_size = int(train_split * len(train_dataset))
  val_size = len(train_dataset) - train_size
  train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

  # Create tensors for train and validation data
  train_images = torch.stack([image for image, _ in train_dataset])
  train_labels = torch.tensor([label for _, label in train_dataset])
  val_images = torch.stack([image for image, _ in val_dataset])
  val_labels = torch.tensor([label for _, label in val_dataset])
  
  # Create custom dataset objects
  train_custom_dataset = CustomCIFAR(data=train_images, targets=train_labels, transform=train_transform)
  val_custom_dataset = CustomCIFAR(data=val_images, targets=val_labels)
  
  total_train_samples = len(train_custom_dataset)
  total_val_samples = len(val_custom_dataset)
  
  if limit_data is not None and limit_data < total_train_samples:
      train_samples = int(train_split * limit_data)
      val_samples = int(limit_data - train_samples)

      train_indices = random.sample(range(total_train_samples), train_samples)
      val_indices = random.sample(range(total_val_samples), val_samples)
      
      train_dataset = Subset(train_custom_dataset, train_indices)
      val_dataset = Subset(val_custom_dataset, val_indices)
  else:
      train_dataset = train_custom_dataset
      val_dataset = val_custom_dataset
      
  train_loader = DataLoader(
      train_dataset,
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