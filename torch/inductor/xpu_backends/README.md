# Intel XPU Backend for PyTorch Inductor

This module provides specialized optimizations for Intel GPU hardware in PyTorch's Inductor compiler framework. It aims to improve performance for deep learning models running on Intel GPUs by providing custom kernel implementations and optimization strategies.

## Features

- Optimized matrix multiplication operations for Intel XPU devices
- Specialized convolution kernels with efficient memory access patterns
- Activation function optimizations (GELU, ReLU, etc.)
- Reduction operations with efficient parallel implementations
- Performance benchmarking utilities
- Configurable optimization settings

## Requirements

- PyTorch 2.0 or higher
- Intel OneAPI Base Toolkit
- Intel XPU-compatible GPU hardware

## Usage

### Automatic Initialization

The XPU backend is automatically initialized if Intel GPU hardware is detected. You can control this behavior with environment variables:

```python
# Disable automatic initialization
os.environ["PYTORCH_XPU_INDUCTOR_AUTO_INIT"] = "0"

# Manually initialize
from torch.inductor.xpu_backends.integration import initialize_xpu_backend
initialize_xpu_backend()
```

### Configuration Options

The backend can be configured using environment variables:

```python
# Enable fast math approximations
os.environ["PYTORCH_XPU_FAST_MATH"] = "1"

# Set tile sizes for matrix multiplication
os.environ["PYTORCH_XPU_TILE_SIZE_M"] = "64"
os.environ["PYTORCH_XPU_TILE_SIZE_N"] = "64"
os.environ["PYTORCH_XPU_TILE_SIZE_K"] = "16"
```

### Example

```python
import torch
import torch.nn as nn
from torch.inductor.xpu_backends import benchmark

# Create a model
model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.ReLU()
)

# Move to XPU device if available
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    model = model.to('xpu')
    
    # Use torch.compile with Inductor backend
    model = torch.compile(model, backend="inductor")
    
    # Run inference
    input_tensor = torch.randn(1, 3, 224, 224, device='xpu')
    output = model(input_tensor)
```

## Benchmarking

The module includes benchmarking utilities to measure and compare performance:

```python
from torch.inductor.xpu_backends import benchmark

# Create benchmarking object
bench = benchmark.XPUBenchmark(warm_up_iterations=5, test_iterations=10)

# Benchmark matrix multiplication
a_cpu = torch.randn(1024, 1024)
b_cpu = torch.randn(1024, 1024)

# Run on CPU
a_cpu = torch.randn(1024, 1024)
b_cpu = torch.randn(1024, 1024)

# Run on XPU if available
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    a_xpu = a_cpu.to('xpu')
    b_xpu = b_cpu.to('xpu')
    
    results = bench.benchmark_operation(
        torch.matmul,
        args_cpu=(a_cpu, b_cpu),
        args_xpu=(a_xpu, b_xpu)
    )
    
    print(f"CPU time: {results['cpu']:.4f} ms")
    print(f"XPU time: {results['xpu']:.4f} ms")
```

## Extending

To add new optimized kernels for Intel XPU devices, follow these steps:

1. Create a new module in the `torch/inductor/xpu_backends` directory
2. Implement your optimized operations
3. Add your kernels to the registration process in `integration.py`

## Testing

Run tests with:

```bash
python -m unittest torch.inductor.xpu_backends.test_xpu_backend
```
