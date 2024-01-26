""" Copyright (c) 2023, Diego Páez
    * Licensed under The MIT License [see LICENSE for details]

    - Distribute and Evaluate the population using multiple processes.
"""

import torch.multiprocessing as mp
from typing import Dict, Any, List
import numpy as np
from cnn import train
from util import init_log
import torch
from cnn import input
import time

class EvalPopulation(object):
    """
    Evaluate a population using multiple processes.

    This class is designed to distribute the evaluation of a population of models
    using multiple processes.
    
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
        Initialize the EvalPopulation object.

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
        self.logger = init_log(log_level, name=__name__)
        desired_gpus = params["available_gpus"]
        self.gpus = [f'cuda:{i}' for i in desired_gpus if i < torch.cuda.device_count()]
        self.loader = input.GenericDataLoader(params=self.train_params)
        #mp.set_start_method('fork')  # fork for linux, spawn for windows
        self.logger.info(f"Evaluation process initialized with {len(self.gpus)} GPUs")
        print("\n")
        
        
    def __call__(self, decoded_params: list, decoded_nets: list, generation: int):
        """
        Evaluate the population.

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
        evaluations = []
        results_queue = mp.Queue()

        # create a list of tuples with the individual, the selected thread, the decoded net, the decoded params
        # and the return value
        individual_per_thread = [(idx, idx % pop_size, decoded_nets[idx], decoded_params[idx])
                                for idx in range(pop_size)]

        train_loader, val_loader = self.loader.get_loader(pin_memory_device=self.gpus[0])
        processes = []

        self.logger.info(f"Starting the Generation {generation} with {pop_size} individuals")
        evol_time_start = time.perf_counter()

        for idx in range(pop_size):
            individuals_selected_thread = list(filter(lambda x: x[1] == idx, individual_per_thread))
            gpu_device = self.gpus[idx % len(self.gpus)]
            process = mp.Process(target=self.run_individuals, args=(generation,
                                                                self.train_params,
                                                                self.fn_dict,
                                                                train_loader,
                                                                val_loader,
                                                                individuals_selected_thread,
                                                                gpu_device,
                                                                results_queue))
            process.start()
            processes.append(process)

        for p in processes:
            p.join()

        while not results_queue.empty():
            result = results_queue.get()
            evaluations.append(result['accuracy'])

        evol_end = time.perf_counter()
        time_elapsed_min = (evol_end - evol_time_start) / 60
        time_elapsed_sec = (evol_end - evol_time_start) % 60
        self.logger.info(f"Time elapsed for {pop_size} individuals: {time_elapsed_min:.0f}m {time_elapsed_sec:.0f}s")
        
        return np.array(evaluations)
            
            
    def run_individuals(self, generation, train_params, fn_dict, train_loader, val_loader,
                        individuals_selected_thread, gpu_device, results_queue):
        for individual, selected_thread, decoded_net, decoded_params in individuals_selected_thread:
            self.train_params['device'] = gpu_device
            accuracy, inference_time, memory_consumption = train.fitness_calculation(
                f"{generation}_{individual}",
                {**train_params},
                fn_dict,
                decoded_net,
                train_loader,
                val_loader,
            )
            result = {
                'individual': individual,
                'selected_thread': selected_thread,
                'accuracy': accuracy,
                'inference_time': inference_time,
                'memory_consumption': memory_consumption,
            }
            results_queue.put(result)
            self.logger.info(f"Calculated fitness of individual {individual} on thread {selected_thread} with "
                             f"Accuracy: {round(accuracy, 4)}, Inference Time: {round(inference_time, 4)} uS, "
                             f"Memory Consumption: {round(memory_consumption, 4)} MB")

