import argparse
import os

import qnas
import qnas_config as cfg
import evaluation
from util import check_files, init_log, download_dataset

def main(**args):
    
    logger = init_log(args['log_level'], name=__name__)

    if not os.path.exists(args['experiment_path']):
        logger.info(f"Creating {args['experiment_path']} ...")
        os.makedirs(args['experiment_path'])

    # Evolution or continue previous evolution
    if not args['continue_path']:
        phase = 'evolution'
    else:
        phase = 'continue_evolution'
        logger.info(f"Continue evolution from: {args['continue_path']}. Checking files ...")
        check_files(args['continue_path'])

    logger.info(f"Getting parameters from {args['config_file']} ...")
    config = cfg.ConfigParameters(args, phase=phase)
    config.get_parameters()
    logger.info(f"Saving parameters for {config.phase} phase ...")
    config.save_params_logfile()
    
    if config.train_spec['mixed_precision']:
        logger.info(f"Using mixed precision training ...")
        
    # Download dataset
    download_dataset(params=config.train_spec)
    
    eval_pop = evaluation.EvalPopulation(params=config.train_spec,
                                                fn_dict=config.fn_dict,
                                                log_level=config.train_spec['log_level'])
    
    qnas_cnn = qnas.QNAS(eval_pop, config.train_spec['experiment_path'],
                         log_file=config.files_spec['log_file'],
                         log_level=config.train_spec['log_level'],
                         data_file=config.files_spec['data_file'])

    qnas_cnn.initialize_qnas(**config.QNAS_spec)
    
    # Start evolution
    logger.info(f"Starting evolution ...")
    qnas_cnn.evolve()
    logger.info(f"Evolution finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, required=True,
                        help='Directory where to write logs and model files.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to input data.')
    parser.add_argument('--dataset', type=str, required=True,  help='Dataset name.', 
                        choices=['cifar10', 'cifar100', 'pathmnist', 'octmnist', 'tissuemnist', 'organamnist'])
    parser.add_argument('--config_file', type=str, required=True,
                        help='Configuration file name.')
    parser.add_argument('--continue_path', type=str, default='',
                        help='If the user wants to continue a previous evolution, point to '
                             'the corresponding experiment path. Evolution parameters will be '
                             'loaded from this folder.')
    parser.add_argument('--log_level', choices=['NONE', 'INFO', 'DEBUG'], default='NONE',
                        help='Logging information level.')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['RMSProp', 'Adam', 'AdamW', 'SGD'],
                        help='Optimizer to be used during training. Default = AdamW.')
    parser.add_argument('--fitness_metric', type=str, default='best_accuracy', 
                        choices=['best_accuracy', 'mean_accuracy', 'median_accuracy', 'scalar_multi_objective'],
                        help='Fitness metric to be used during evolution. Default = accuracy.')
    parser.add_argument('--data_augmentation', action='store_true',
                    help='Disable data augmentation during training. Default = False.')
    parser.add_argument('--save_checkpoints_epochs', type=int, default=5,
                        help='Number of epochs to save the model. Default = 5.')

    arguments = parser.parse_args()

    main(**vars(arguments))