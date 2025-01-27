"""
Copyright (c) 2024, Diego R. Páez Ardila
Licensed under The MIT License [see LICENSE for details]

This script shows how to fine-tune a previously trained model (best_model.pth)
with a new or extended dataset.
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Union, Any
import time

from cnn import model, input, metrics
from util import init_log, load_yaml, create_info_file
from cnn.train_detailed import train_epoch, evaluate, compute_metrics, release_gpu_memory

# Global cache dictionary
dataset_info_cache = {}

# Function to get dataset info with caching
def get_dataset_info(dataset_name, data_path):
    if dataset_name in dataset_info_cache:
        #print(f"Using cached dataset info for {dataset_name}")
        return dataset_info_cache[dataset_name]

    dataset_info_path = os.path.join(data_path, 'data_info.txt')
    dataset_info = load_yaml(dataset_info_path)

    if dataset_info is not None:
        dataset_info_cache[dataset_name] = dataset_info
    return dataset_info

def load_trained_model(decoded_net, train_params):
    """
    Build the model based on the decoded network architecture and training parameters.

    Parameters:
    - decoded_net: The network architecture definition.
    - train_params: Dictionary containing training parameters.

    Returns:
    - model_instance: The constructed model ready for training.
    """
    # Load data info
    dataset_name = train_params['dataset'].lower()
    
    if dataset_name in input.available_datasets:
        dataset_info = input.available_datasets[dataset_name]
    else:
        dataset_info = get_dataset_info(dataset_name, train_params['data_path'])
    if dataset_info is None:
        raise ValueError(f"Failed to load dataset information for {dataset_name}. Check if the dataset is available or if 'data_info.txt' exists and is correctly formatted.")

    # Update train_params with dataset info
    train_params['num_classes'] = dataset_info['num_classes']
    train_params['task'] = dataset_info['task']
    train_params['input_shape'] = [train_params['batch_size']] + dataset_info['shape']

    # Check if 'cbam' is a key in the fn_dict
    has_cbam_key = any(key.startswith('cbam') for key in train_params['fn_dict'])

    # Filter fn_dict to include only keys present in decoded_net
    filtered_fn_dict = {
        key: item for key, item in train_params['fn_dict'].items() if key in decoded_net
    }

    # Create the model
    model_instance = model.NetworkGraph(num_classes=dataset_info['num_classes'])
    model_instance.create_functions(
        fn_dict=filtered_fn_dict, net_list=decoded_net, cbam=has_cbam_key
    )

    input_random = torch.randn(train_params['input_shape'])
    with torch.no_grad():
        _ = model_instance(input_random)

    model_instance.load_state_dict(torch.load(train_params['best_model_path']))
    model_instance.to(train_params['device'])
    return model_instance

def freeze_layers(
    model: nn.Module, 
    freeze_pattern: str = None
) -> None:
    """
    Freeze certain layers of the model for fine-tuning. For example, you can:
    - freeze all but the last layer
    - freeze only the first few convolution layers
    - or do not freeze at all (freeze_pattern=None)

    Args:
        model (nn.Module): The model to partially freeze.
        freeze_pattern (str): Some pattern to decide what to freeze.
            E.g., "all_but_last" or "first_conv" or None for no freezing.
    """
    if freeze_pattern == "all_but_last":
        # Example: freeze everything except the last linear layer
        # You need to adapt this logic depending on how your model is structured.
        for name, param in model.named_parameters():
            if 'fc' not in name:  # Suppose your last layer is named something with "fc"
                param.requires_grad = False

    elif freeze_pattern == "first_conv":
        # Example logic: freeze only the first conv layer
        for name, param in model.named_parameters():
            if "conv1" in name:
                param.requires_grad = False

    # If freeze_pattern is None or unrecognized, do nothing (i.e., fine-tune entire network).

##############################################################################
# 3. Training loop for fine-tuning (reuse from your train_detailed.py)
##############################################################################
def fine_tune_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    params: Dict[str, Any],
    max_epochs: int = 10
):
    """
    Similar to your 'train(...)' function, but simplified for fine-tuning. 
    Adjust as needed.

    Args:
        model (nn.Module): The loaded model (some layers may be frozen).
        train_loader (DataLoader): DataLoader for the fine-tuning train dataset.
        val_loader (DataLoader): DataLoader for the fine-tuning validation dataset.
        params (Dict[str, Any]): Dictionary with training hyperparameters.
        max_epochs (int): Number of epochs to run for fine-tuning.

    Returns:
        None or a metrics dictionary
    """
    # Use an existing training function or replicate your training logic:
    from cnn.train_detailed import train_epoch, evaluate
    
    # Example: define a new optimizer with a smaller learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_val_accuracy = 0.0
    best_model_path = os.path.join(params['fine_tune_path'], 'best_finetuned_model.pth')
    
    for epoch in range(1, max_epochs + 1):
        # -- Train Phase
        train_loss, train_acc = train_epoch(
            model, criterion, optimizer, train_loader, params
        )
        
        # -- Validation Phase
        val_loss, val_acc = evaluate(model, criterion, val_loader, params)
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), best_model_path)
        
        print(f"[Epoch {epoch}/{max_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

##############################################################################
# 4. Main script entry-point
##############################################################################
def main(args):
    """
    Main function that loads the trained model, optionally freezes some layers, 
    and fine-tunes on a new dataset.
    """
    # 1) Create the folder to save the fine-tuned model if it doesn't exist
    fine_tune_path = os.path.join(args['fine_tune_path'])
    if not os.path.exists(fine_tune_path):
        os.makedirs(fine_tune_path)
    
    # 2) Construct params dictionary
    params = {
        'device': args['device'],
        'fn_dict': {},         # Load or define your layer dictionary
        'net_list': [],        # The layer list in the same order used originally
        'input_shape': [],     # e.g., [1, 3, 32, 32] for a single CIFAR image
        'num_classes': args['num_classes'], 
        'best_model_path': args['best_model_path'],
        'fine_tune_path': fine_tune_path,
    }
    # (You might load these from a config file, just like before.)
    
    # 3) Load your new dataset
    #    Option A: Use your existing data loader code
    #    Option B: Just create your own DataLoader from a standard dataset
    #
    # Example: (replace with your own data code)
    # train_dataset = ...
    # val_dataset   = ...
    # train_loader  = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    # val_loader    = DataLoader(val_dataset, batch_size=args['eval_batch_size'], shuffle=False)
    #
    # For demonstration, we’ll assume you already have train_loader, val_loader:
    train_loader = None  # placeholder
    val_loader   = None  # placeholder
    
    # 4) Load the previously trained model
    model = load_trained_model(params)
    
    # 5) (Optional) Freeze some or all layers
    freeze_layers(model, freeze_pattern=args['freeze_pattern'])
    
    # 6) Fine-tune
    fine_tune_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        params=params,
        max_epochs=args['fine_tune_epochs']
    )
    
    # 7) Evaluate on a test set if desired
    # test_loader = ...
    # test_loss, test_accuracy = evaluate(model, criterion, test_loader, params)
    # print("Test Accuracy after fine-tuning:", test_accuracy)

    # 8) Clean up, if needed
    print("Fine-tuning completed.")

##############################################################################
# 5. Command-line interface
##############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--best_model_path', type=str, required=True,
                        help='Path to the best_model.pth from your previous training.')
    parser.add_argument('--fine_tune_path', type=str, default='fine_tuned',
                        help='Directory where the fine-tuned model will be saved.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for fine-tuning (e.g., cuda or cpu).')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes in the new (or same) dataset.')
    parser.add_argument('--freeze_pattern', type=str, default=None,
                        help='Pattern for freezing layers. e.g., "all_but_last"')
    parser.add_argument('--fine_tune_epochs', type=int, default=10,
                        help='Number of epochs for fine tuning.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=256)
    
    args = parser.parse_args()
    main(vars(args))
