import torch
import random
import util
from time import time
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor, Resize, Compose, RandomCrop, RandomHorizontalFlip, Normalize

def CIFAR10_loader(data_path:str, train_split=0.9,batch_size=24,limit_data=None,seed=None,data_aug=True,for_train=True, num_workers=2):
  """
  This function creates a dataloader for the CIFAR10 dataset.
  
  Args:
    data_path: Path to the data folder
    train_split: float (0,1 - default 0.9)
      Percentage of the data to be used for training
    limit_data: int (default None)
      Limit the number of samples to be used for training and validation
    batch_size: int (default 24)
    seed: int (default None)
      Seed for the random number generator
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
  
  info_dict = {'dataset': f'CIFAR{10}'}
  
  if seed is None:
    seed = int(time())
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
  info_dict['seed'] = seed
  
  # Calculate mean and standard deviation from the entire dataset
  dataset = CIFAR10(data_path, download=True, transform=ToTensor())
  # loader = DataLoader(dataset, batch_size=len(dataset), num_workers=num_workers, shuffle=False)
  # mean = torch.zeros(3)
  # std = torch.zeros(3)
  # for images, _ in loader:
  #     mean += torch.mean(images, dim=(0, 2, 3))
  #     std += torch.std(images, dim=(0, 2, 3))
  # mean /= len(loader)
  # std /= len(loader)

  # info_dict['mean'] = mean.tolist()
  # info_dict['std'] = std.tolist()
 
  mean = [0.4914, 0.4822, 0.4465]
  std = [0.2023, 0.1994, 0.2010]
  if for_train:
    if data_aug:
      height, width = dataset[0][0].shape[1:]
      pad = 4
      train_transform = Compose([
        Resize((height + pad, width + pad)),
        RandomCrop((height, width)),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=mean, std=std)  # Normalize with std=std to keep images in the [0, 1] range
      ])
    else:
      train_transform = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
      ])
  else:
    # Normalize for validation and testing
    validation_transform = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)])
    
    test_dataset = CIFAR10(data_path, download=True, train=False, transform=validation_transform)
    test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True)
    
  train_dataset = CIFAR10(data_path, download=True, train=True, transform=train_transform)
  
  
  total_samples = len(train_dataset)
  
  if limit_data is not None and limit_data < total_samples:
      sample_indices = random.sample(range(total_samples), limit_data)
      train_size = int(train_split * limit_data)
      val_size = limit_data - train_size

      train_indices = sample_indices[:train_size]
      val_indices = sample_indices[train_size:]

      train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
      val_dataset = torch.utils.data.Subset(train_dataset, val_indices)
  else:
      train_size = int(train_split * total_samples)
      val_size = total_samples - train_size

      train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
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
  
  # train_imgs, _ = next(iter(train_loader))
  
  # info_dict['train_records'] = len(train_dataset)
  # info_dict['valid_records'] = len(val_dataset)
  # info_dict['test_records'] = len(test_dataset)
  # info_dict['shape'] = list(train_imgs.shape[1:])
  
  # util.create_info_file(out_path=data_path, info_dict=info_dict)
  
  if for_train:
    print("All set for training!")
    return train_loader, val_loader
  else:
    print("All set for testing!")
    return test_loader


def CIFAR100_loader(data_path:str, train_split=0.9,batch_size=24,limit_data=None,seed=None,data_aug=True,for_train=True, num_workers=2):
  """
  This function creates a dataloader for the CIFAR100 dataset.
  
  Args:
  
    data_path: Path to the data folder
    train_split: float (0,1 - default 0.9)
      Percentage of the data to be used for training
    limit_data: int (default None)
      Limit the number of samples to be used for training and validation
    batch_size: int (default 24)
    seed: int (default None)
      Seed for the random number generator
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
  
  info_dict = {'dataset': f'CIFAR{100}'}
  
  if seed is None:
    seed = int(time())
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
  info_dict['seed'] = seed
  
  # Calculate mean and standard deviation from the entire dataset
  dataset = CIFAR100(data_path, download=True, transform=ToTensor())
  mean = torch.stack([sample.mean(1).mean(1) for sample, _ in dataset]).mean(0)
      
  if data_aug:
    height, width = dataset[0][0].shape[1:]
    pad = 4
    transform = Compose([
      ToTensor(),
      Resize((height + pad, width + pad)),
      RandomCrop((height, width)),
      RandomHorizontalFlip(),
      Normalize(mean=mean, std=[1.0, 1.0, 1.0])  # Normalize with std=[1.0, 1.0, 1.0] to keep images in the [0, 1] range
    ])
  else:
    transform = Compose([
      ToTensor(),
      Normalize(mean=mean, std=[1.0, 1.0, 1.0])
    ])
    
  #val_percent = 1 - train_split
  train_dataset = CIFAR100(data_path, download=True, train=True, transform=transform)
  test_dataset = CIFAR100(data_path, download=True, train=False, transform=transform)
  
  total_samples = len(train_dataset)
  
  if limit_data is not None and limit_data < total_samples:
      train_size = min(int(train_split * limit_data), limit_data)
      val_size = int(limit_data - train_size)
      train_dataset, _ = random_split(train_dataset, [limit_data, total_samples - limit_data])
  else:
      train_size = int(train_split * total_samples)
      val_size = int(total_samples - train_size)
      
  train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
  
  test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True)
  
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
  
  train_imgs, _ = next(iter(train_loader))
  
  info_dict['train_records'] = len(train_dataset)
  info_dict['valid_records'] = len(val_dataset)
  info_dict['test_records'] = len(test_dataset)
  info_dict['shape'] = list(train_imgs.shape[1:])
  
  util.create_info_file(out_path=data_path, info_dict=info_dict)
  
  if for_train:
    return train_loader, val_loader
  else:
    return test_loader