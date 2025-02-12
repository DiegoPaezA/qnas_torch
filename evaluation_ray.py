""" Copyright (c) 2024, Diego Páez
    * Licensed under The MIT License [see LICENSE for details]

    - Distribute and Evaluate the population using Ray.
"""
import time
import ray
import torch
import GPUtil
import numpy as np
from typing import Dict, Any, List
from cnn import train, input
from util import init_log, estimate_total_gpu_memory


@ray.remote
def run_individual(model_id: str, train_params: Dict[str, Any], fn_dict: Dict[str, Any], decoded_net: Any, decoded_params: Any) -> Dict[str, Any]:
    """
    Evaluate an individual model remotely using Ray.

    Parameters:
    - model_id (str): Identifier for the model being evaluated.
    - train_params (dict): Dictionary containing training parameters (e.g., batch size, device settings).
    - fn_dict (dict): Function definitions used for specific tasks (e.g., loss functions, custom layers).
    - decoded_net (Any): The architecture of the neural network being evaluated.
    - decoded_params (Any): Specific parameters for the decoded network.

    Returns:
    - result (dict): A dictionary containing the evaluation results, including:
        - model_id (str): Identifier for the evaluated model.
        - fitness (float): The computed fitness value of the model.
        - params (float): The number of parameters of the model (in millions).
        - inference_time (float): Inference time of the model (in microseconds).
        - status (str): 'success' if evaluation was successful, otherwise 'error'.
        - error_msg (str, optional): Error message if evaluation failed.
    """

    # Set the device
    device = "cuda"
    train_params['device'] = device

    # Instantiate data loaders
    loader = input.GenericDataLoader(params=train_params)
    train_loader, val_loader = loader.get_loader(pin_memory_device = device)

    # Prepare return value container
    return_val = [0.0, 0.0, 0.0]  # [accuracy, params, inference_time]

    try:
        train.fitness_calculation(
            model_id,
            train_params,
            fn_dict,
            decoded_net,
            train_loader,
            val_loader,
            return_val
        )

        result = {
            'model_id': model_id,
            'fitness': return_val[0],
            'params': return_val[1],
            'inference_time': return_val[2],
            'status': 'success' if all(value != 0.0 for value in return_val) else 'error'
        }
    except Exception as e:
        result = {
            'model_id': model_id,
            'fitness': 0.0,
            'params': 0.0,
            'inference_time': 0.0,
            'status': 'error',
            'error_msg': str(e)
        }
    finally:
        del train_loader, val_loader
        torch.cuda.synchronize()  # Wait for all pending operations to complete
        if torch.cuda.memory_allocated() > 0 or torch.cuda.memory_reserved() > 0:
            torch.cuda.empty_cache()

    return result

class EvalPopulation(object):
    def __init__(self, params: Dict[str, Any], fn_dict: Dict[str, Any], log_level: str = 'INFO') -> None:
        """
        Initialize the evaluation environment for the population of models.

        Parameters:
        - params (dict): Training parameters that include information about dataset, batch size, etc.
        - fn_dict (dict): Function definitions used for specific operations.
        - log_level (str, optional): Logging level (default is 'INFO').
        """
        self.train_params = params
        self.fn_dict = fn_dict
        self.logger = init_log(log_level, name=__name__)
        self.gpus = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        self.logger.info(f"Evaluation process initialized with {len(self.gpus)} GPUs")
        self.temporal_loader = input.GenericDataLoader(params=self.train_params) # Initialize data loader to download dataset
        #del temporal_loader  # Delete the data loader to free up resources
        self.gputil = GPUtil.getGPUs()
        # Initialize Ray once
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=True)  # Use ignore_reinit_error to prevent issues in notebooks or during testing


    def __call__(self, decoded_params: List[Any], decoded_nets: List[Any], generation: int) -> np.ndarray:
        """
        Execute the evaluation of a given population of models for a specific generation.

        Parameters:
        - decoded_params (list): List of parameters for each model in the population.
        - decoded_nets (list): List of network architectures for each individual in the population.
        - generation (int): Identifier for the current generation of models.

        Returns:
        - evaluations (np.ndarray): An array containing the fitness value for each individual in the population.
        """
        pop_size = len(decoded_nets)
        evaluations = np.empty(shape=(pop_size,))
        retry_queue = []  # Queue to store models that need re-evaluation

        self.logger.info(f"Starting Generation {generation} with {pop_size} individuals")
        evol_time_start = time.perf_counter()

        # Get the available GPU memory for each GPU
        available_gpu_memory = [self.gputil[i].memoryFree for i in range(len(self.gputil))] 

        # Calculate the total GPU memory available
        self.total_gpu_memory = min(available_gpu_memory)
        
        #self.logger.info(f"Available GPU memory: {self.total_gpu_memory}")
        
        # Step 1: Calculate GPU fractions for each model
        self.gpu_fractions = self.calculate_gpu_fractions(decoded_nets, generation)
        
        # Step 2: Run models with precomputed GPU fractions
        result_refs = []
        for idx in range(pop_size):
            model_id = f"{generation}_{idx}"
            decoded_net = decoded_nets[idx]
            decoded_param = decoded_params[idx]
            gpu_fraction = self.gpu_fractions[idx]

            result_ref = run_individual.options(num_gpus=gpu_fraction,max_retries = 3).remote(
                model_id,
                self.train_params,
                self.fn_dict,
                decoded_net,
                decoded_param
            )
            result_refs.append(result_ref)

        # Collect results
        results = ray.get(result_refs)

        # Process results and queue errors
        for idx, result in enumerate(results):
            try:
                if result is not None and result['status'] == 'success':
                    evaluations[idx] = result['fitness']
                    self.logger.info(
                        f"Calculated fitness of individual {idx} with "
                        f"Fitness: {round(result['fitness'], 3)}, Params: {round(result['params'], 2)}M, "
                        f"Inference Time: {round(result['inference_time'], 3)} μs"
                    )
                else:
                    evaluations[idx] = 0.0
                    retry_queue.append((idx, decoded_nets[idx], decoded_params[idx]))  # Add to retry queue
                    error_message = result.get('error_msg', 'Unknown error') if result else 'Result is None'
                    self.logger.error(
                        f"Failed to evaluate individual {idx}: {error_message}"
                    )
            except Exception as e:
                # Handle any other unforeseen errors
                evaluations[idx] = 0.0
                retry_queue.append((idx, decoded_nets[idx], decoded_params[idx]))  # Add to retry queue
                self.logger.error(f"Unexpected error for individual {idx}: {str(e)}")

        # Retry failed models with a maximum retry limit
        max_retries = 3  # Define the maximum number of retries
        if retry_queue:
            self.retry_failed_models(retry_queue, generation, evaluations, max_retries)

        evol_end = time.perf_counter()
        time_elapsed_min = (evol_end - evol_time_start) / 60
        time_elapsed_sec = (evol_end - evol_time_start) % 60
        self.logger.info(f"Time elapsed for {pop_size} individuals: {time_elapsed_min:.0f}m {time_elapsed_sec:.0f}s")

        return evaluations

    def calculate_gpu_fractions(self, decoded_nets: List[Any], generation: int) -> List[float]:
        gpu_fractions = []
        # Get a temporary dataloader to feed into estimation
        train_loader, _ = self.temporal_loader.get_loader(pin_memory_device=self.gpus[0]) 
        for idx, decoded_net in enumerate(decoded_nets):
            model_id = f"{generation}_{idx}"
            
            # Estimate memory using the revised function
            estimated_memory = estimate_total_gpu_memory(decoded_net, self.train_params, self.fn_dict, train_loader)

            if estimated_memory == 0.0:
                gpu_fraction = 1.0
            else:
                gpu_fraction_ = estimated_memory / self.total_gpu_memory
                # Add safety margin
                gpu_fraction = round(min(max(gpu_fraction_ * 1, 0.01), 1.0), 2)

            self.logger.info(f"{model_id} Estimated memory: {estimated_memory} MB, GPU fraction: {gpu_fraction}")
            gpu_fractions.append(gpu_fraction)

        return gpu_fractions
    
    def retry_failed_models(self, retry_queue: List[tuple], generation: int, evaluations: np.ndarray, max_retries: int) -> None:
        """
        Re-evaluate models that failed in the initial evaluation up to a maximum retry limit.

        Parameters:
        - retry_queue (list): List of tuples containing (index, decoded_net, decoded_param) for failed models.
        - generation (int): Generation identifier for logging.
        - evaluations (np.ndarray): Array to store the fitness value for each individual in the population.
        - max_retries (int): Maximum number of retries for failed models.
        """
        retries = 0
        while retry_queue and retries < max_retries:
            self.logger.info(f"Retry attempt {retries + 1} for {len(retry_queue)} failed evaluations")
            retry_results = []
            next_retry_queue = []

            for idx, decoded_net, decoded_param in retry_queue:
                model_id = f"{generation}_{idx}"                
                gpu_fraction = self.gpu_fractions[idx]
                result_ref = run_individual.options(num_gpus=gpu_fraction).remote(
                    model_id,
                    self.train_params,
                    self.fn_dict,
                    decoded_net,
                    decoded_param
                )
                retry_results.append((idx, result_ref))

            # Collect retry results
            retry_results_fetched = ray.get([ref[1] for ref in retry_results])

            for (idx, result) in zip([ref[0] for ref in retry_results], retry_results_fetched):
                if result is not None and result['status'] == 'success':
                    evaluations[idx] = result['fitness']
                    self.logger.info(
                        f"Retry successful for individual {idx} with "
                        f"Fitness: {round(result['fitness'], 3)}, Params: {round(result['params'], 2)}M, "
                        f"Inference Time: {round(result['inference_time'], 3)} μs"
                    )
                else:
                    next_retry_queue.append((idx, decoded_net, decoded_param))
                    self.logger.error(
                        f"Retry failed for individual {idx}: {result.get('error_msg', 'Unknown error')}"
                    )

            # Update retry queue for the next attempt
            retry_queue = next_retry_queue
            retries += 1

        # Log any models that failed after all retries
        if retry_queue:
            for idx, _, _ in retry_queue:
                self.logger.error(f"Model {idx} failed after {max_retries} retries")
    
    def __del__(self) -> None:
        """
        Destructor to shut down Ray when the EvalPopulation object is destroyed.
        Ensures that Ray is properly shut down to free up resources.
        """
        if ray.is_initialized():
            ray.shutdown()
