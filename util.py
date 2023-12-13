import logging
import yaml
import pickle as pkl
import os
import re
import matplotlib.pyplot as plt
from shutil import rmtree




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

def create_info_file(out_path, info_dict, file_name='data_info.txt'):
    """ Saves info in *info_dict* in a txt file.

    Args:
        out_path: (str) path to the directory where to save info file.
        info_dict: dict with all relevant info the user wants to save in the info file.
    """
    
    with open(os.path.join(out_path, file_name), 'w') as f:
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
    
def plot_training_history(results_dict:dict, params:dict, retrain:bool=False):
    """ Plot the training history of a model.
    
    Args:
        results_dict: (dict) dictionary with the training history.
    """
        
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    total_epochs = len(results_dict["training_losses"])
    epochs = range(1, total_epochs + 1)
    if retrain:
        ax[0].plot(epochs, results_dict["training_losses"], 'b', label='Training loss')
        ax[0].plot(epochs, results_dict["validation_losses"], 'r', label='Validation loss')
        ax[0].set_title('Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()
        ax[0].grid(True)
        

        ax[1].plot(epochs, results_dict["training_accuracies"], 'b', label='Training Acc')
        ax[1].plot(epochs, results_dict["validation_accuracies"], 'r', label='Validation Acc')
        max_acc, index = max(results_dict["validation_accuracies"]), results_dict["validation_accuracies"].index(max(results_dict["validation_accuracies"]))
        
        ax[1].axhline(y=results_dict["test_accuracy"], color='r', linestyle='--', label='Test Acc')
        ax[1].text(1, results_dict["test_accuracy"]+0.1, f'Test Acc: {results_dict["test_accuracy"]:.2f}', fontsize=12)

        ax[1].plot(index+1, max_acc, 'go', label='Max Val Acc')
        ax[1].text(index+1, max_acc+0.1, f'{max_acc:.2f}', fontsize=12)
        ax[1].set_title('Accuracy')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()
        ax[1].grid(True)
    else:
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

def delete_old_dirs(path, keep_best=False, best_id=''):
    """ Delete directories with old training files (models, checkpoints...). Assumes the
        directories' names start with digits.

    Args:
        path: (str) path to the experiment folder.
        keep_best: (bool) True if user wants to keep files from the best individual.
        best_id: (str) id of the best individual.
    """

    folders = [os.path.join(path, d) for d in os.listdir(path)
               if os.path.isdir(os.path.join(path, d)) and d[0].isdigit()]
    folders.sort(key=natural_key)

    if keep_best and best_id:
        folders = [d for d in folders if os.path.basename(d) != best_id]

    for f in folders:
        rmtree(f)

def check_files(exp_path):
    """ Check if exp_path exists and if it does, check if log_file is valid.

    Args:
        exp_path: (str) path to the experiment folder.
    """

    if not os.path.exists(exp_path):
        raise OSError('User must provide a valid \"--experiment_path\" to continue '
                      'evolution or to retrain a model.')

    file_path = os.path.join(exp_path, 'data_QNAS.pkl')

    if os.path.exists(file_path):
        if os.stat(file_path).st_size == 0:
            raise OSError('User must provide an \"--experiment_path\" with a valid data file to '
                          'continue evolution or to retrain a model.')
    else:
        raise OSError('log_file not found!')

    file_path = os.path.join(exp_path, 'log_params_evolution.txt')

    if os.path.exists(file_path):
        if os.stat(file_path).st_size == 0:
            raise OSError('User must provide an \"--experiment_path\" with a valid config_file '
                          'to continue evolution or to retrain a model.')
    else:
        raise OSError('log_params_evolution.txt not found!')
    
def init_log(log_level, name, file_path=None):
    """ Initialize a logging.Logger with level *log_level* and name *name*.

    Args:
        log_level: (str) one of 'NONE', 'INFO' or 'DEBUG'.
        name: (str) name of the module initiating the logger (will be the logger name).
        file_path: (str) path to the log file. If None, stdout is used.

    Returns:
        logging.Logger object.
    """

    logger = logging.getLogger(name)

    if file_path is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(file_path)

    formatter = logging.Formatter('%(levelname)s: %(module)s: %(asctime)s.%(msecs)03d '
                                  '- %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if log_level == 'INFO':
        logger.setLevel(logging.INFO)
    elif log_level == 'DEBUG':
        logger.setLevel(logging.DEBUG)

    return logger