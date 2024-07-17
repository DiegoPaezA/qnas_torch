import torch
import time

def measure_inference_time(model, input_data, warmup_runs=10, measure_runs=10):
    """
    Measure the inference time of a PyTorch model. 
    The function assumes that the model and the input data are on the same device (GPU)

    Parameters:
    - model: PyTorch model
    - input_data: Input data for the model - tensor or list of tensors
    - warmup_runs: Number of warmup runs before measuring
    - measure_runs: Number of runs to measure for averaging

    Returns:
    - average_inference_time: Average inference time in microseconds
    """

    model.eval()

    # Warm up the model
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_data)

    # Measure inference time
    inference_times = []
    with torch.no_grad():
        for _ in range(measure_runs):
            
            torch.cuda.synchronize()

            start_time = time.time()
            _ = model(input_data)
            
            torch.cuda.synchronize()

            end_time = time.time()
            inference_times.append(end_time - start_time)

    average_inference_time = (sum(inference_times) / len(inference_times)) * 1e6  # Convert seconds to microseconds
    return average_inference_time
