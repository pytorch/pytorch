"""
Example usage of Intel XPU backend for PyTorch Inductor.

This module demonstrates how to use the Intel XPU backend optimizations
in PyTorch Inductor with example models and operations.
"""

import os
import time
import argparse
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Import our XPU backend modules
from torch.inductor.xpu_backends import matmul, kernels, integration, utils, benchmark
from torch.inductor.xpu_backends.config import config as xpu_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_model() -> nn.Module:
    """
    Create a sample neural network model for benchmarking.
    
    Returns:
        PyTorch model
    """
    class SampleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2)
            self.fc1 = nn.Linear(128 * 8 * 8, 512)
            self.relu3 = nn.ReLU()
            self.fc2 = nn.Linear(512, 10)
            
        def forward(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = x.view(-1, 128 * 8 * 8)
            x = self.relu3(self.fc1(x))
            x = self.fc2(x)
            return x
            
    return SampleModel()


def benchmark_matmul_operations(
    bench: benchmark.XPUBenchmark,
    sizes: List[Tuple[int, int, int]] = [(128, 128, 128), (512, 512, 512), (1024, 1024, 1024)],
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark matrix multiplication operations of different sizes.
    
    Args:
        bench: Benchmark instance
        sizes: List of matrix sizes (M, N, K)
        
    Returns:
        Dictionary of benchmark results
    """
    results = {}
    
    for m, n, k in sizes:
        op_name = f"matmul_{m}x{k}_{k}x{n}"
        logger.info(f"Benchmarking {op_name}...")
        
        # Create test tensors on different devices
        a_cpu = torch.randn(m, k)
        b_cpu = torch.randn(k, n)
        
        args_cpu = (a_cpu, b_cpu)
        args_cuda = None
        args_xpu = None
        
        if torch.cuda.is_available():
            a_cuda = a_cpu.cuda()
            b_cuda = b_cpu.cuda()
            args_cuda = (a_cuda, b_cuda)
            
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            a_xpu = a_cpu.to("xpu")
            b_xpu = b_cpu.to("xpu")
            args_xpu = (a_xpu, b_xpu)
            
        # Run benchmarks
        results[op_name] = bench.benchmark_operation(torch.matmul, args_cpu, args_cuda, args_xpu)
        
    return results


def benchmark_conv_operations(
    bench: benchmark.XPUBenchmark,
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark convolution operations with different parameters.
    
    Args:
        bench: Benchmark instance
        
    Returns:
        Dictionary of benchmark results
    """
    results = {}
    
    # Define convolution configurations
    configs = [
        # (batch_size, in_channels, out_channels, input_size, kernel_size)
        (8, 3, 64, 32, 3),
        (8, 64, 128, 16, 3),
        (8, 128, 256, 8, 3),
    ]
    
    for batch_size, in_channels, out_channels, size, kernel_size in configs:
        op_name = f"conv2d_b{batch_size}_i{in_channels}_o{out_channels}_s{size}_k{kernel_size}"
        logger.info(f"Benchmarking {op_name}...")
        
        # Create test tensors on different devices
        input_cpu = torch.randn(batch_size, in_channels, size, size)
        weight_cpu = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        bias_cpu = torch.randn(out_channels)
        
        args_cpu = (input_cpu, weight_cpu, bias_cpu)
        args_cuda = None
        args_xpu = None
        
        if torch.cuda.is_available():
            input_cuda = input_cpu.cuda()
            weight_cuda = weight_cpu.cuda()
            bias_cuda = bias_cpu.cuda()
            args_cuda = (input_cuda, weight_cuda, bias_cuda)
            
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            input_xpu = input_cpu.to("xpu")
            weight_xpu = weight_cpu.to("xpu")
            bias_xpu = bias_cpu.to("xpu")
            args_xpu = (input_xpu, weight_xpu, bias_xpu)
            
        # Run benchmarks
        results[op_name] = bench.benchmark_operation(
            nn.functional.conv2d, 
            args_cpu, 
            args_cuda, 
            args_xpu,
            kwargs={"stride": 1, "padding": 1}
        )
        
    return results


def benchmark_model(
    bench: benchmark.XPUBenchmark,
    model: nn.Module,
    batch_size: int = 32,
    compile_mode: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark a PyTorch model on different devices.
    
    Args:
        bench: Benchmark instance
        model: PyTorch model
        batch_size: Batch size
        compile_mode: Model compilation mode (None, "default", "reduce-overhead", "max-autotune")
        
    Returns:
        Dictionary of benchmark results
    """
    results = {}
    
    # Create sample input
    input_cpu = torch.randn(batch_size, 3, 32, 32)
    input_cuda = None
    input_xpu = None
    
    if torch.cuda.is_available():
        input_cuda = input_cpu.cuda()
        
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        input_xpu = input_cpu.to("xpu")
        
    # Benchmark with different compilation modes
    modes = [None]
    if compile_mode is not None:
        modes.append(compile_mode)
        
    for mode in modes:
        if mode is None:
            model_name = "model_eager"
            benchmark_model = model
        else:
            model_name = f"model_compiled_{mode}"
            try:
                benchmark_model = torch.compile(model, backend="inductor", mode=mode)
            except Exception as e:
                logger.error(f"Failed to compile model with mode {mode}: {e}")
                continue
                
        logger.info(f"Benchmarking {model_name}...")
        results[model_name] = bench.benchmark_model(benchmark_model, input_cpu, input_cuda, input_xpu)
        
    return results


def main():
    """Main function to run benchmarks and demonstrate XPU backend."""
    parser = argparse.ArgumentParser(description="PyTorch Intel XPU Backend Example")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--device-info", action="store_true", help="Show XPU device information")
    parser.add_argument("--config", action="store_true", help="Show XPU configuration")
    args = parser.parse_args()
    
    # Initialize XPU backend
    logger.info("Initializing Intel XPU backend for PyTorch Inductor...")
    
    # Show XPU device information if requested
    if args.device_info:
        device_info = utils.get_xpu_device_info()
        if device_info["available"]:
            logger.info(f"Found {device_info['count']} Intel XPU device(s):")
            for i, device in enumerate(device_info["devices"]):
                logger.info(f"  Device {i}: {device['name']}")
                logger.info(f"    Memory: {device['total_memory'] / (1024**3):.2f} GB")
                if device["eu_count"] is not None:
                    logger.info(f"    Execution Units: {device['eu_count']}")
        else:
            logger.info("No Intel XPU devices found.")
            
    # Show XPU configuration if requested
    if args.config:
        xpu_config.log_config()
        
    # Run benchmarks if requested
    if args.benchmark:
        logger.info("Running XPU backend benchmarks...")
        bench = benchmark.XPUBenchmark(warm_up_iterations=3, test_iterations=10)
        
        # Benchmark matrix multiplication
        matmul_results = benchmark_matmul_operations(bench)
        
        # Benchmark convolution
        conv_results = benchmark_conv_operations(bench)
        
        # Benchmark model
        model = create_sample_model()
        model_results = benchmark_model(bench, model, batch_size=32, compile_mode="max-autotune")
        
        # Combine results
        all_results = {**matmul_results, **conv_results, **model_results}
        
        # Generate report
        report = benchmark.generate_benchmark_report(all_results, "Intel XPU Backend Benchmark Report")
        
        # Print and save report
        print("\n" + report)
        with open("xpu_benchmark_report.md", "w") as f:
            f.write(report)
        logger.info("Benchmark report saved to 'xpu_benchmark_report.md'")
        
    logger.info("Done.")


if __name__ == "__main__":
    main()
