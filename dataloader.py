from transformers import AutoFeatureExtractor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Resize, Compose, AutoAugment, AutoAugmentPolicy

import torch
from torchvision.datasets import CIFAR10, CIFAR100

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



def create_CIFAR10_loader(data_path, batch_size = 24, num_workers = 2, ID=False):
  val_loader = DataLoader( 
      CIFAR10('./', download=True, train=False, transform=ToTensor()),
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=True,
      #collate_fn=my_collate
  )

  if ID:
    train_loader = DataLoader( 
      CIFAR10('./', download=True, transform=ToTensor()),
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=True,
      #collate_fn=my_collate
    )
    return train_loader, val_loader
  
  return val_loader

def create_CIFAR100_loader(data_path, batch_size = 24, num_workers = 2, ID=False, data_augmentation = False):
  val_loader = DataLoader( 
      CIFAR100('./', download=True, train=False, transform=ToTensor()),
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=True,
      collate_fn=my_collate
  )

  if ID:
    train_loader = DataLoader( 
      CIFAR100('./', download=True, transform=ToTensor()),
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=True,
      collate_fn=my_collate
    )
    return train_loader, val_loader
  
  return val_loader


def create_loader(data_path, dataset, ID=False, data_augmentation = False):
  if dataset == 'cifar10':
    return create_CIFAR10_loader(data_path, ID=ID)
  elif dataset == 'cifar100':
    return create_CIFAR100_loader(data_path, ID=ID, data_augmentation = data_augmentation)
  else:
    raise Exception('Dataset not supported')
