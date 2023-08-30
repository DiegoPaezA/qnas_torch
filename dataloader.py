import torch
import random
import util
from time import time
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from torchvision.transforms import ToTensor, Resize, Compose, RandomCrop, RandomHorizontalFlip, Normalize

class CustomCIFAR10(Dataset):
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
  loader = DataLoader(dataset, batch_size=len(dataset), num_workers=num_workers, shuffle=False)
  mean = torch.zeros(3)
  std = torch.zeros(3)
  
  for images, _ in loader:
      mean += torch.mean(images, dim=(0, 2, 3))
      std += torch.std(images, dim=(0, 2, 3))
  mean /= len(loader)
  std /= len(loader)

  info_dict['mean'] = mean.tolist()
  info_dict['std'] = std.tolist()
  
  transform = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
      ])
  
  if data_aug:
    height, width = dataset[0][0].shape[1:]
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
  train_custom_dataset = CustomCIFAR10(data=train_images, targets=train_labels, transform=train_transform)
  val_custom_dataset = CustomCIFAR10(data=val_images, targets=val_labels)
  
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
  
  test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True)
  
  print(f"Limiting training samples to {len(train_dataset)} and validation samples to {len(val_dataset)}")  
  info_dict['train_records'] = len(train_dataset)
  info_dict['valid_records'] = len(val_dataset)
  info_dict['test_records'] = len(test_dataset)
  info_dict['shape'] = list(train_dataset[0][0].shape)
  util.create_info_file(out_path=data_path, info_dict=info_dict)
  
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