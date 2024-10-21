""" Copyright (c) 2023, Diego Páez
    * Licensed under The MIT License [see LICENSE for details]

    - Distribute and Evaluate the population using Ray.
"""
import time
import ray
import torch
import numpy as np
import torch.multiprocessing as mp
from typing import Dict, Any, List
from cnn import train, input
from util import init_log, estimate_model_memory


@ray.remote
def run_individual(model_id, train_params, fn_dict, decoded_net, decoded_params):

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
            'status': 'success'
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
        memory_allocated_before = torch.cuda.memory_allocated()
        memory_reserved_before = torch.cuda.memory_reserved()
        if memory_allocated_before > 0 or memory_reserved_before > 0:
            torch.cuda.empty_cache()

    return result

class EvalPopulation(object):
    def __init__(self, params: dict, fn_dict: dict, log_level: str = 'INFO'):
        self.train_params = params
        self.fn_dict = fn_dict
        self.logger = init_log(log_level, name=__name__)
        self.gpus = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        self.loader = input.GenericDataLoader(params=self.train_params)
        self.logger.info(f"Evaluation process initialized with {len(self.gpus)} GPUs")

        # Initialize Ray
        ray.init()

    def __call__(self, decoded_params: list, decoded_nets: list, generation: int):
        pop_size = len(decoded_nets)
        evaluations = np.empty(shape=(pop_size, ))

        self.logger.info(f"Starting Generation {generation} with {pop_size} individuals")
        evol_time_start = time.perf_counter()

        result_refs = []
        
        # Obtain GPU memory capacities
        gpu_memory_list = [torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())]
        total_gpu_memory = min(gpu_memory_list)
        

        for idx in range(pop_size):
            model_id = f"{generation}_{idx}"
            decoded_net = decoded_nets[idx]
            decoded_param = decoded_params[idx]

            # Estimate model memory
            estimated_memory = estimate_model_memory(decoded_net, self.train_params, self.fn_dict)

            if estimated_memory is None:
                gpu_fraction = 1.0
                self.logger.warning(f"Memory estimation failed for model {model_id}, assigning full GPU.")
            else:
                gpu_fraction = estimated_memory / total_gpu_memory
                gpu_fraction = min(max(gpu_fraction, 0.1), 1.0)

            result_ref = run_individual.options(num_gpus=gpu_fraction).remote(
                model_id,
                self.train_params,
                self.fn_dict,
                decoded_net,
                decoded_param
            )
            result_refs.append(result_ref)

        results = ray.get(result_refs)
        
        for idx, result in enumerate(results):
            if result['status'] == 'success':
                evaluations[idx] = result['fitness']
                self.logger.info(
                    f"Calculated fitness of individual {idx} with "
                    f"Fitness: {round(result['fitness'], 3)}, Params: {round(result['params'], 2)}M, "
                    f"Inference Time: {round(result['inference_time'], 3)} μs"
                )
            else:
                evaluations[idx] = 0.0
                self.logger.error(
                    f"Failed to evaluate individual {idx}: {result.get('error_msg', 'Unknown error')}"
                )

        evol_end = time.perf_counter()
        time_elapsed_min = (evol_end - evol_time_start) / 60
        time_elapsed_sec = (evol_end - evol_time_start) % 60
        self.logger.info(f"Time elapsed for {pop_size} individuals: {time_elapsed_min:.0f}m {time_elapsed_sec:.0f}s")

        # Shutdown Ray
        ray.shutdown()

        return evaluations
