""" Copyright (c) 2023, Diego Páez
    * Licensed under The MIT License [see LICENSE for details]

    - Distribute and Evaluate the population using multiple processes.
"""
import time
import torch
import GPUtil
import numpy as np
from cnn import train, input
import torch.multiprocessing as mp
from typing import Dict, Any, List



from util import init_log, estimate_total_gpu_memory


class EvalPopulation(object):
    def __init__(self, params: dict, fn_dict: dict, log_level: str = 'INFO'):
        self.train_params = params
        self.fn_dict = fn_dict
        self.logger = init_log(log_level, name=__name__)
        self.gpus = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        self.loader = input.GenericDataLoader(params=self.train_params)
        self.gputil = GPUtil.getGPUs()
        self.logger.info(f"Evaluation process initialized with {len(self.gpus)} GPUs")
        
    def calculate_gpu_fractions(self, decoded_nets: List[Any], generation: int) -> List[float]:
        """
        Estimate memory usage and calculate GPU fractions for each model in the population.

        Parameters:
        - decoded_nets (list): List of network architectures for each individual in the population.
        - generation (int): Identifier for the current generation of models, used for logging.

        Returns:
        - gpu_fractions (list of floats): A list of GPU fractions required for each model.
        """
        gpu_fractions = []
        
        for idx, decoded_net in enumerate(decoded_nets):
            model_id = f"{generation}_{idx}"

            # Estimate memory usage in MB
            estimated_memory = estimate_total_gpu_memory(decoded_net, self.train_params, self.fn_dict)
            gpu_fraction_ = estimated_memory / self.total_gpu_memory if estimated_memory else 1.0
            gpu_fraction = round(min(max(gpu_fraction_ * 1.20, 0.01), 1.0), 2)

            # Log the memory estimation and GPU fraction
            self.logger.info(f"{model_id} Estimated memory: {estimated_memory} MB, GPU fraction: {gpu_fraction}")

            # Store the GPU fraction for use in evaluations
            gpu_fractions.append(gpu_fraction)

        return gpu_fractions

    def __call__(self, decoded_params: list, decoded_nets: list, generation: int):
        pop_size = len(decoded_nets)
        evaluations = np.empty(shape=(pop_size, ))
        variables = [mp.Array('f', 3) for _ in range(pop_size)]
        
        processes = []
        wait_queue = []
        
        # Get the available GPU memory for each GPU
        available_gpu_memory = [self.gputil[i].memoryFree for i in range(len(self.gputil))] 

        # Calculate the total GPU memory available
        self.total_gpu_memory = min(available_gpu_memory)
        
        # Step 1: Calculate GPU fractions for each model
        self.gpu_fractions = self.calculate_gpu_fractions(decoded_nets, generation)
        
        available_gpus = {gpu.id: gpu.memoryFree for gpu in self.gputil}

        self.logger.info(f"Starting Generation {generation} with {pop_size} individuals")
        evol_time_start = time.perf_counter()

        # Start by trying to allocate each model to an available GPU
        for idx in range(pop_size):
            model_id = f"{generation}_{idx}"
            decoded_net = decoded_nets[idx]
            gpu_fraction = self.gpu_fractions[idx]
            required_memory = gpu_fraction * min(gpu.memoryTotal for gpu in self.gputil)  # Estimate memory in MB

            allocated = False
            for gpu_id, free_memory in available_gpus.items():
                if free_memory >= required_memory:
                    gpu_device = f'cuda:{gpu_id}'
                    available_gpus[gpu_id] -= required_memory  # Update available memory
                    train_loader, val_loader = self.loader.get_loader(pin_memory_device=gpu_device)

                    process = mp.Process(
                        target=self.run_individuals,
                        args=(generation, self.train_params, self.fn_dict, train_loader, val_loader, 
                            [(idx, decoded_nets[idx], decoded_params[idx], variables[idx])], gpu_device)
                    )
                    process.start()
                    processes.append(process)
                    self.logger.info(f"Started process for individual {idx} on GPU {gpu_id} with GPU fraction {gpu_fraction}")
                    allocated = True
                    break

            if not allocated:
                # Add to wait_queue if no GPU could accommodate
                wait_queue.append((idx, decoded_nets[idx], decoded_params[idx], variables[idx]))
                self.logger.info(f"Added individual {idx} to wait queue due to insufficient GPU memory")

        # Wait for all current processes to finish, then process the wait queue
        for p in processes:
            p.join()

        # Now process the wait_queue after clearing all other processes
        for idx, decoded_net, decoded_param, variable in wait_queue:
            gpu_fraction = self.gpu_fractions[idx]
            required_memory = gpu_fraction * min(gpu.memoryTotal for gpu in self.gputil)

            for gpu_id, free_memory in available_gpus.items():
                if free_memory >= required_memory:
                    gpu_device = f'cuda:{gpu_id}'
                    available_gpus[gpu_id] -= required_memory  # Update available memory
                    train_loader, val_loader = self.loader.get_loader(pin_memory_device=gpu_device)

                    process = mp.Process(
                        target=self.run_individuals,
                        args=(generation, self.train_params, self.fn_dict, train_loader, val_loader, 
                            [(idx, decoded_net, decoded_param, variable)], gpu_device)
                    )
                    process.start()
                    process.join()  # Ensure each process finishes sequentially in the wait queue
                    self.logger.info(f"Started process for waitlisted individual {idx} on GPU {gpu_id} with GPU fraction {gpu_fraction}")
                    break

        # Collect results
        for idx, val in enumerate(variables):
            evaluations[idx] = val[0]  # Accuracy or chosen metric
        
        evol_end = time.perf_counter()
        time_elapsed_min = (evol_end - evol_time_start) / 60
        time_elapsed_sec = (evol_end - evol_time_start) % 60
        self.logger.info(f"Time elapsed for {pop_size} individuals: {time_elapsed_min:.0f}m {time_elapsed_sec:.0f}s")
        return evaluations
            
    def run_individuals(self, generation, train_params, fn_dict, train_loader, val_loader, individual_list, gpu_device):
        for idx, decoded_net, decoded_params, return_val in individual_list:
            self.train_params['device'] = gpu_device
            train.fitness_calculation(
                f"{generation}_{idx}", {**train_params}, fn_dict, decoded_net, train_loader, val_loader, return_val
            )
            self.logger.info(
                f"Calculated fitness of individual {idx} on GPU {gpu_device} with "
                f"Best Metric: {round(return_val[0], 3)}, Params: {round(return_val[1], 2)}M, "
                f"Inference Time: {round(return_val[2], 3)} uS"
            )
