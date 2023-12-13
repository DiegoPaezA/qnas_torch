""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Distribute population eval using MPI.
"""

import dask
from dask.distributed import Client, LocalCluster
from dask_cuda import LocalCUDACluster
import time
import numpy as np
from cnn import train
from util import init_log
import random
import torch

class EvalPopulationDask(object):
    """
    Evaluate a population using Dask.

    This class is designed to distribute the evaluation of a population of models
    using Dask, a parallel computing library. It allows parallel training and
    evaluation of multiple models on a Dask cluster with GPU support.
    
    Parameters
    ----------
    params : dict
        A dictionary containing parameters for the evaluation process.
    fn_dict : dict
        A dictionary containing definitions of the functions.
    log_level : str, optional
        The logging level for the internal logger (default is 'INFO').

    Attributes
    ----------
    train_params : dict
        Parameters for the training and evaluation process.
    fn_dict : dict
        Definitions of the functions used in the evaluation.
    timeout : int
        Timeout value for the Dask operations.
    logger : logger
        Internal logger for logging messages.
    gpus : list
        List of GPU devices available for evaluation.
    client : Client
        Dask client for managing the distributed computation.

    Methods
    -------
    __call__(decoded_params, decoded_nets, generation)
        Perform the evaluation of the population.
    
    """
    def __init__(self, params: dict, fn_dict: dict, log_level: str = 'INFO'):
        """
        Initialize the EvalPopulationDask object.

        Arguments:
        params : dict
            A dictionary containing parameters for the evaluation process.
        fn_dict : dict
            A dictionary containing definitions of the functions.
        log_level : str, optional
            The logging level for the internal logger (default is 'INFO').
        """
        
        self.train_params = params
        self.fn_dict = fn_dict
        self.timeout = 9000
        self.logger = init_log(log_level, name=__name__)
        self.gpus = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        self.threads_per_worker = 1
        
        # Set up Dask client
        gpu_num = len(self.gpus)
        cluster = LocalCUDACluster(n_workers=gpu_num)
        self.client = Client(cluster)

    def __call__(self, decoded_params: list, decoded_nets: list, generation: int):
        """
        Evaluate the population using Dask.

        Parameters
        ----------
        decoded_params : list
            List of dictionaries containing the parameters for each model.
        decoded_nets : list
            List of lists containing the network architectures for each model.
        generation : int
            The generation number for tracking purposes.

        Returns
        -------
        np.ndarray
            An array containing the evaluations for each model.

        Raises
        ------
        TimeoutError
            If the Dask operations exceed the specified timeout.
        """
        pop_size = len(decoded_nets)
        #num_threads = self.threads_per_worker*len(self.client.ncores())

        #assert pop_size == num_threads
        
        futures = []

        try:
            for i in range(pop_size):
                id_num = f'{generation}_{i}'
                
                self.train_params['device'] = self.gpus[i%len(self.gpus)]
                print(f"Training model {id_num} on device {self.train_params['device']} ...")
                args = {'id_num': id_num,
                        'params': {**self.train_params},
                        'fn_dict': self.fn_dict,
                        'net_list': decoded_nets[i]}

                future = self.client.submit(train.fitness_calculation, **args)
                futures.append(future)
            print(f"Waiting for evaluations ...")
            evaluations = self.client.gather(futures)
            # save evaluations history
            # self.logger.info(f"Saving evaluations history ...")
            
        except TimeoutError:
            self.client.shutdown()
            raise

        return np.array(evaluations)
