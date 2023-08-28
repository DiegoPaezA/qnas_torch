from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor, Normalize, Resize, Compose, AutoAugment, AutoAugmentPolicy
import random
import torch
from torchvision.datasets import CIFAR10, CIFAR100
from time import time
import util

class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.tensor(transposed_data[1])

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return {'pixel_values': self.inp, 'labels': self.tgt}


def my_collate(batch):
    return SimpleCustomBatch(batch)


def CIFAR10_loader(data_path, train_split=0.9, limit_data=None, batch_size=24, num_workers=2, seed=None):
    
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






def create_loader(data_path, dataset, data_augmentation = False):
  if dataset == 'cifar10':
    return CIFAR10_loader(data_path)
  elif dataset == 'cifar100':
    return CIFAR100_loader(data_path, data_augmentation = data_augmentation)
  else:
    raise Exception('Dataset not supported')
