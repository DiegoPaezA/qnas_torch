import torch
import random
import util
from time import time
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor, Resize, Compose, RandomCrop, RandomHorizontalFlip

class Preprocessor:
    def __init__(self, info):
        self.info = info
        self.transform = Compose([
            Resize((self.info.height + self.info.pad, self.info.width + self.info.pad)),
            RandomCrop((self.info.height, self.info.width)),
            RandomHorizontalFlip(),
        ])

    def preprocess(self, image):
        """ Pad, resize and randomly flip a single image with shape = [H, W, C].

        Args:
            image: raw image (torch.tensor with dtype=torch.float32, values in [0, 1],
                   and shape = [num_channels, height, width]).

        Returns:
            preprocessed image, with the same shape.
        """

        image = self.transform(image)
        return image



def CIFAR10_loader(data_path, train_split=0.9, limit_data=None, batch_size=24, num_workers=2, seed=None, ):
    
    info_dict = {'dataset': f'CIFAR{10}'}
    
    if seed is None:
      seed = int(time())
      random.seed(seed)
      torch.manual_seed(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
      
    info_dict['seed'] = seed
    
    val_percent = 1 - train_split
    train_dataset = CIFAR10(data_path, download=True, train=True, transform=ToTensor())
    test_dataset = CIFAR10(data_path, download=True, train=False, transform=ToTensor())
    
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
      pin_memory=True,
      #collate_fn=my_collate
      )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        # collate_fn=my_collate
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        # collate_fn=my_collate
      )
    
    train_imgs, train_labels = next(iter(train_loader))
    
    info_dict['train_records'] = len(train_dataset)
    info_dict['valid_records'] = len(val_dataset)
    info_dict['test_records'] = len(test_dataset)
    info_dict['shape'] = list(train_imgs.shape[1:])
    
    util.create_info_file(out_path=data_path, info_dict=info_dict)
    
    return train_loader, val_loader, test_loader


def CIFAR100_loader(data_path, train_split=0.9, limit_data=None, batch_size=24, num_workers=2, seed=None):
    
    info_dict = {'dataset': f'CIFAR{100}'}
    
    if seed is None:
      seed = int(time())
      random.seed(seed)
      torch.manual_seed(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
      
    info_dict['seed'] = seed
    
    val_percent = 1 - train_split
    train_dataset = CIFAR100(data_path, download=True, train=True, transform=ToTensor())
    test_dataset = CIFAR100(data_path, download=True, train=False, transform=ToTensor())
    
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
      pin_memory=True,
      #collate_fn=my_collate
      )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        # collate_fn=my_collate
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        # collate_fn=my_collate
      )
    
    train_imgs, train_labels = next(iter(train_loader))
    
    info_dict['train_records'] = len(train_dataset)
    info_dict['valid_records'] = len(val_dataset)
    info_dict['test_records'] = len(test_dataset)
    info_dict['shape'] = list(train_imgs.shape[1:])
    
    util.create_info_file(out_path=data_path, info_dict=info_dict)
    
    return train_loader, val_loader, test_loader

Cifar10Info = {
  'dataset': 'CIFAR10',
  'data_path': 'cifar10',
  'num_classes': 10,
  'height': 32,
  'width': 32,
  'channels': 3,
  'pad': 4
}
Cifar100Info = {
  'dataset': 'CIFAR100',
  'data_path': 'cifar100',
  'num_classes': 100,
  'height': 32,
  'width': 32,
  'channels': 3,
  'pad': 4
}

def create_loader(data_path, dataset='cifar10'):
  if dataset == 'cifar10':
    return CIFAR10_loader(data_path)
  elif dataset == 'cifar100':
    return CIFAR100_loader(data_path)
  else:
    raise Exception('Dataset not supported')
