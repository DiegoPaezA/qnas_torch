import yaml
import pickle as pkl
import os
import re
import matplotlib.pyplot as plt



def natural_key(string):
    """ Key to use with sort() in order to sort string lists in natural order.
        Example: [1_1, 1_2, 1_5, 1_10, 1_13].
    """

    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string)]

def load_yaml(file_path):
    """ Wrapper to load a yaml file.

    Args:
        file_path: (str) path to the file to load.

    Returns:
        dict with loaded parameters.
    """

    with open(file_path, 'r') as f:
        file = yaml.safe_load(f)

    return file

def load_pkl(file_path):
    """ Load a pickle file.

    Args:
        file_path: (str) path to the file to load.

    Returns:
        loaded data.
    """

    with open(file_path, 'rb') as f:
        file = pkl.load(f)

    return file

def create_info_file(out_path, info_dict):
    """ Saves info in *info_dict* in a txt file.

    Args:
        out_path: (str) path to the directory where to save info file.
        info_dict: dict with all relevant info the user wants to save in the info file.
    """

    with open(os.path.join(out_path, 'data_info.txt'), 'w') as f:
        yaml.dump(info_dict, f)
        
def check_file_exists(file_path):
    """ Check if a file exists.
    
    Args:
        file_path: (str) path to the file to check.
        
    Returns:
        True if the file exists, False otherwise.
    """
    if os.path.exists(file_path):
        return True
    else:
        return False
    
def plot_training_history(results_dict:dict, params:dict):
    """ Plot the training history of a model.
    
    Args:
        results_dict: (dict) dictionary with the training history.
    """
        
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    total_epochs = len(results_dict["training_losses"])
    epochs = range(1, total_epochs + 1)
    eval_starts = params["max_epochs"] - params["epochs_to_eval"]
    epochs_val = range(eval_starts+1, total_epochs+1)
    
    ax[0].plot(epochs, results_dict["training_losses"], 'b', label='Training loss')
    ax[0].plot(epochs_val, results_dict["validation_losses"], 'r', label='Validation loss')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].grid(True)
    

    ax[1].plot(epochs, results_dict["training_accuracies"], 'b', label='Training Acc')
    ax[1].plot(epochs_val, results_dict["validation_accuracies"], 'r', label='Validation Acc')
    max_acc, index = max(results_dict["validation_accuracies"]), results_dict["validation_accuracies"].index(max(results_dict["validation_accuracies"]))
    ax[1].plot(index+1, max_acc, 'go', label='Max Acc')
    ax[1].text(index+1, max_acc+0.1, f'{max_acc:.2f}', fontsize=12)
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    ax[1].grid(True)
    
    plt.show()