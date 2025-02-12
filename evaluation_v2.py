import time
import torch
import GPUtil
import numpy as np
from cnn import train, input
import torch.multiprocessing as mp
from typing import Dict, Any, List
from util import init_log

# Helper function to get the per-process GPU memory usage in MB using NVML (via GPUtil)
def get_process_gpu_memory(pid: int, gpu_id: int) -> float:
    """Return the GPU memory usage in MB for a given PID on a specified GPU."""
    gpus = GPUtil.getGPUs()
    # Filter GPU by ID
    selected_gpu = [gpu for gpu in gpus if gpu.id == gpu_id]
    if not selected_gpu:
        return 0.0
    selected_gpu = selected_gpu[0]

    # Unfortunately, GPUtil doesn't directly give per-process memory usage.
    # We need to use pynvml for per-process usage:
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    mem_usage = 0.0
    for p in procs:
        if p.pid == pid:
            mem_usage = p.usedGpuMemory / (1024**2)  # bytes to MB
            break
    pynvml.nvmlShutdown()
    return mem_usage

class EvalPopulation(object):
    def __init__(self, params: dict, fn_dict: dict, log_level: str = 'INFO'):
        self.train_params = params
        self.fn_dict = fn_dict
        self.logger = init_log(log_level, name=__name__)
        self.gpus = [f'cuda:{i}' for i in range(torch.cuda.device_count())]

        # Initialize data loader
        self.loader = input.GenericDataLoader(params=self.train_params)
        self.gputil = GPUtil.getGPUs()

        # Use the minimum total memory from all GPUs as a baseline, if needed
        self.total_gpu_memory = min([gpu.memoryTotal for gpu in self.gputil])
        self.logger.info(f"Evaluation process initialized with {len(self.gpus)} GPUs")

    def __call__(self, decoded_params: list, decoded_nets: list, generation: int):
        pop_size = len(decoded_nets)
        evaluations = np.empty(shape=(pop_size,))
        variables = [mp.Array('f', 3) for _ in range(pop_size)]

        wait_queue = []
        processes_info = []  # To store (process, idx, gpu_id, required_memory) for cleanup

        # Initially assume full free memory on each GPU
        available_gpus = {gpu.id: gpu.memoryFree for gpu in self.gputil}

        self.logger.info(f"Starting Generation {generation} with {pop_size} individuals")
        evol_time_start = time.perf_counter()

        # Try to allocate each model on some GPU
        for idx, (net, param) in enumerate(zip(decoded_nets, decoded_params)):
            allocated = False
            for gpu_id, free_memory in available_gpus.items():
                # Start the model process
                gpu_device = f'cuda:{gpu_id}'
                train_loader, val_loader = self.loader.get_loader(pin_memory_device=gpu_device)

                process = mp.Process(
                    target=self.run_individuals,
                    args=(generation, self.train_params, self.fn_dict, train_loader, val_loader,
                          [(idx, net, param, variables[idx])], gpu_device)
                )
                process.start()
                
                # Wait for the model to load and stabilize memory usage
                time.sleep(10)  # 5 seconds to allow the model to load
                
                # Check memory usage of this process
                mem_usage = get_process_gpu_memory(process.pid, gpu_id)
                if mem_usage == 0.0:
                    # If we can't detect usage, assume full GPU (or kill process)
                    self.logger.error(f"Cannot detect memory usage for model {idx}, sending to wait queue.")
                    process.terminate()
                    process.join()
                    wait_queue.append((idx, net, param, 0.0))  # 0.0 means we don't know memory yet
                else:
                    # If enough memory is actually available
                    if free_memory >= mem_usage:
                        # Deduct the used memory
                        available_gpus[gpu_id] -= mem_usage
                        processes_info.append((process, idx, gpu_id, mem_usage))
                        self.logger.info(f"Process for individual {idx} started on GPU {gpu_id}. Used memory: {mem_usage} MB")
                        allocated = True
                    else:
                        # Not enough memory, kill process and add to wait queue
                        self.logger.info(f"GPU {gpu_id} not enough memory for model {idx} (needs {mem_usage}MB, has {free_memory}MB).")
                        process.terminate()
                        process.join()
                        wait_queue.append((idx, net, param, mem_usage))
                
                if allocated:
                    break

            if not allocated and not any(wq[0] == idx for wq in wait_queue):
                # No GPU found for this model
                wait_queue.append((idx, net, param, 0.0))
                self.logger.info(f"No GPU could accommodate model {idx}, added to wait queue.")

        # Wait for all current processes to finish
        for process, idx, gpu_id, mem_usage in processes_info:
            process.join()
            # Once done, free the memory reservation
            available_gpus[gpu_id] += mem_usage

        processes_info.clear()

        # Now try the wait queue again after freeing memory
        # If models in wait queue have known memory usage, try to allocate them again
        # If unknown, just try again with the same approach
        for idx, net, param, known_mem in wait_queue:
            allocated = False
            for gpu_id, free_memory in available_gpus.items():
                # Start process again
                gpu_device = f'cuda:{gpu_id}'
                train_loader, val_loader = self.loader.get_loader(pin_memory_device=gpu_device)

                process = mp.Process(
                    target=self.run_individuals,
                    args=(generation, self.train_params, self.fn_dict, train_loader, val_loader,
                          [(idx, net, param, variables[idx])], gpu_device)
                )
                process.start()

                time.sleep(5)  # Wait again for loading
                mem_usage = get_process_gpu_memory(process.pid, gpu_id)
                if mem_usage == 0.0 and known_mem > 0.0:
                    mem_usage = known_mem  # Fall back to known memory if new measurement fails

                if mem_usage > 0.0 and free_memory >= mem_usage:
                    # Allocate
                    available_gpus[gpu_id] -= mem_usage
                    process.join()
                    available_gpus[gpu_id] += mem_usage
                    self.logger.info(f"Waitlisted model {idx} successfully evaluated on GPU {gpu_id} using {mem_usage}MB")
                    allocated = True
                    break
                else:
                    self.logger.warn(f"Still not enough memory for waitlisted model {idx} on GPU {gpu_id}. Killing process.")
                    process.terminate()
                    process.join()

            if not allocated:
                self.logger.error(f"Model {idx} could not be allocated even after waiting.")

        # Collect results
        for idx, val in enumerate(variables):
            evaluations[idx] = val[0]  # first slot is the fitness metric

        evol_end = time.perf_counter()
        time_elapsed_min = (evol_end - evol_time_start) / 60
        time_elapsed_sec = (evol_end - evol_time_start) % 60
        self.logger.info(f"Time elapsed for {pop_size} individuals: {time_elapsed_min:.0f}m {time_elapsed_sec:.0f}s")
        return evaluations

    def run_individuals(self, generation, train_params, fn_dict, train_loader, val_loader, individual_list, gpu_device):
        """
        This function runs in a separate process for each allocated model. It trains/evaluates the model,
        and stores the results in return_val.
        """
        for idx, decoded_net, decoded_params, return_val in individual_list:
            train_params['device'] = gpu_device
            train.fitness_calculation(
                f"{generation}_{idx}", {**train_params}, fn_dict, decoded_net, train_loader, val_loader, return_val
            )
            self.logger.info(
                f"Calculated fitness of individual {idx} with "
                f"Best Metric: {round(return_val[0], 3)}, Params: {round(return_val[1], 2)}M, "
                f"Inference Time: {round(return_val[2], 3)} uS"
            )
