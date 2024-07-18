import torch
import time
import pynvml

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

class ModelMetrics:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

    def _to_device(self, data):
        if isinstance(data, list) or isinstance(data, tuple):
            return [self._to_device(d) for d in data]
        return data.to(self.device)

    def _check_device(self, data):
        if isinstance(data, list) or isinstance(data, tuple):
            return all(d.device == torch.device(self.device) for d in data)
        return data.device == torch.device(self.device)

    def measure_inference_time(self, input_data, warmup_runs=10, measure_runs=10):
        """
        Measure the inference time of a PyTorch model. 
        The function assumes that the model and the input data are on the same device (GPU).

        Parameters:
        - input_data: Input data for the model - tensor or list of tensors
        - warmup_runs: Number of warmup runs before measuring
        - measure_runs: Number of runs to measure for averaging

        Returns:
        - average_inference_time: Average inference time in microseconds
        """
        self.model.eval()

        if not self._check_device(input_data):
            input_data = self._to_device(input_data)

        if next(self.model.parameters()).device != torch.device(self.device):
            self.model.to(self.device)

        # Warm up the model
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(input_data)

        # Measure inference time
        inference_times = []
        with torch.no_grad():
            for _ in range(measure_runs):
                torch.cuda.synchronize()
                start_time = time.time()
                _ = self.model(input_data)
                torch.cuda.synchronize()
                end_time = time.time()
                inference_times.append(end_time - start_time)

        average_inference_time = (sum(inference_times) / len(inference_times)) * 1e6  # Convert seconds to microseconds
        return average_inference_time
    
    def measure_madd(self, input_shape):
        """
        Measure the Multiply-Add operations of a PyTorch model.

        Parameters:
        - input_shape: Shape of the input tensor

        Returns:
        - total_madd: Total Multiply-Add operations
        """
        self.model.eval()

        # Generate random input data
        input_data = torch.randn(input_shape).to(self.device)

        if not self._check_device(input_data):
            input_data = self._to_device(input_data)

        if next(self.model.parameters()).device != torch.device(self.device):
            self.model.to(self.device)

        # Count the Multiply-Add operations
        total_madd = 0
        with torch.no_grad():
            for module in self.model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    out_h = int((input_data.shape[2] + 2 * module.padding[0] - module.dilation[0] * (module.kernel_size[0] - 1) - 1) / module.stride[0] + 1)
                    out_w = int((input_data.shape[3] + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1) / module.stride[1] + 1)
                    madd = module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1] * out_h * out_w
                    total_madd += madd
                elif isinstance(module, torch.nn.Linear):
                    madd = module.in_features * module.out_features
                    total_madd += madd

        return total_madd
    
    def measure_memory(self, input_shape):
        """
        Measure the memory usage of a PyTorch model.

        Parameters:
        - input_shape: Shape of the input tensor

        Returns:
        - memory: Memory usage in bytes
        """
        self.model.eval()

        # Generate random input data
        input_data = torch.randn(input_shape).to(self.device)

        if not self._check_device(input_data):
            input_data = self._to_device(input_data)

        if next(self.model.parameters()).device != torch.device(self.device):
            self.model.to(self.device)

        # Measure memory usage
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = self.model(input_data)
        memory = torch.cuda.max_memory_allocated(self.device)

        return memory
    
    def measure_flops(self, input_shape):
        """
        Measure the Floating Point Operations of a PyTorch model.

        Parameters:
        - input_shape: Shape of the input tensor

        Returns:
        - total_flops: Total Floating Point Operations
        """
        self.model.eval()

        # Generate random input data
        input_data = torch.randn(input_shape).to(self.device)

        if not self._check_device(input_data):
            input_data = self._to_device(input_data)

        if next(self.model.parameters()).device != torch.device(self.device):
            self.model.to(self.device)

        # Function to count the Floating Point Operations
        def count_flops(module, input, output):
            flops = 0
            if isinstance(module, torch.nn.Conv2d):
                output_dims = output.shape[2:]
                kernel_dims = module.kernel_size
                in_channels = module.in_channels
                out_channels = module.out_channels
                flops = in_channels * out_channels * kernel_dims[0] * kernel_dims[1] * output_dims[0] * output_dims[1] * 2
            elif isinstance(module, torch.nn.Linear):
                flops = module.in_features * module.out_features * 2
            return flops

        total_flops = 0

        # Register hooks to count FLOPs
        def hook(module, input, output):
            nonlocal total_flops
            total_flops += count_flops(module, input, output)

        hooks = []
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                hooks.append(module.register_forward_hook(hook))

        # Perform a forward pass to trigger the hooks
        with torch.no_grad():
            _ = self.model(input_data)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return total_flops
    
    def measure_parameters(self):
        """
        Measure the number of parameters in a PyTorch model.

        Returns:
        - total_params: Total number of parameters
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        return total_params

    def measure_energy_consumption(self, input_data, warmup_runs=10, measure_runs=10):
        """
        Measure the energy consumption of a PyTorch model during inference.

        Parameters:
        - input_data: Input data for the model - tensor or list of tensors
        - warmup_runs: Number of warmup runs before measuring
        - measure_runs: Number of runs to measure for averaging

        Returns:
        - average_power: Average power consumption in watts
        - total_energy: Total energy consumption in joules
        """
        self.model.eval()

        if not self._check_device(input_data):
            input_data = self._to_device(input_data)

        if next(self.model.parameters()).device != torch.device(self.device):
            self.model.to(self.device)

        # Warm up the model
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(input_data)

        # Measure energy consumption
        power_measurements = []
        with torch.no_grad():
            for _ in range(measure_runs):
                torch.cuda.synchronize()
                
                start_time = time.time()
                power_start = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000  # Convert milliwatts to watts
                _ = self.model(input_data)
                torch.cuda.synchronize()
                end_time = time.time()
                power_end = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000  # Convert milliwatts to watts

                elapsed_time = end_time - start_time
                power_measurements.append((power_start + power_end) / 2 * elapsed_time)  # Average power * time

        total_energy = sum(power_measurements)  # in joules
        average_power = total_energy / (measure_runs * (end_time - start_time))

        return average_power, total_energy