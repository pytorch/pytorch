"""
Benchmarking utilities for Intel XPU backend operations.

This module provides tools for measuring and comparing the performance
of Intel GPU operations against CPU and CUDA implementations.
"""

import time
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import numpy as np

logger = logging.getLogger(__name__)


class XPUBenchmark:
    """
    Benchmark class for measuring performance of operations on different devices.
    
    This class provides methods to measure and compare execution time
    of tensor operations on CPU, CUDA, and XPU devices.
    """
    
    def __init__(self, warm_up_iterations: int = 5, test_iterations: int = 10):
        """
        Initialize the benchmark.
        
        Args:
            warm_up_iterations: Number of warm-up iterations before measuring
            test_iterations: Number of iterations for measurement
        """
        self.warm_up_iterations = warm_up_iterations
        self.test_iterations = test_iterations
        
    def _time_function(self, func: Callable, *args, **kwargs) -> float:
        """
        Measure execution time of a function.
        
        Args:
            func: Function to measure
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Average execution time in milliseconds
        """
        # Warm-up runs
        for _ in range(self.warm_up_iterations):
            result = func(*args, **kwargs)
            if torch.is_tensor(result):
                # Ensure the operation is completed
                result.cpu()
                
        # Synchronize devices before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # For XPU devices, we would add a similar synchronization
        # if torch.xpu.is_available():
        #     torch.xpu.synchronize()
            
        # Measure execution time
        start_time = time.time()
        
        for _ in range(self.test_iterations):
            result = func(*args, **kwargs)
            if torch.is_tensor(result):
                # Ensure the operation is completed
                result.cpu()
                
        # Synchronize again before stopping timer
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # For XPU devices, we would add a similar synchronization
        # if torch.xpu.is_available():
        #     torch.xpu.synchronize()
                
        end_time = time.time()
        
        # Calculate average time in milliseconds
        avg_time_ms = ((end_time - start_time) / self.test_iterations) * 1000
        return avg_time_ms
        
    def benchmark_operation(
        self, 
        operation: Callable, 
        args_cpu: Tuple[Any, ...],
        args_cuda: Optional[Tuple[Any, ...]] = None,
        args_xpu: Optional[Tuple[Any, ...]] = None,
        kwargs: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        """
        Benchmark an operation on different devices.
        
        Args:
            operation: Operation to benchmark
            args_cpu: Arguments for CPU version
            args_cuda: Arguments for CUDA version (optional)
            args_xpu: Arguments for XPU version (optional)
            kwargs: Keyword arguments for all versions
            
        Returns:
            Dictionary with execution times in milliseconds for each device
        """
        if kwargs is None:
            kwargs = {}
            
        results = {}
        
        # Benchmark on CPU
        try:
            cpu_time = self._time_function(operation, *args_cpu, **kwargs)
            results["cpu"] = cpu_time
            logger.info(f"CPU execution time: {cpu_time:.4f} ms")
        except Exception as e:
            logger.error(f"Error benchmarking on CPU: {e}")
            results["cpu"] = float("nan")
            
        # Benchmark on CUDA if available
        if torch.cuda.is_available() and args_cuda is not None:
            try:
                cuda_time = self._time_function(operation, *args_cuda, **kwargs)
                results["cuda"] = cuda_time
                logger.info(f"CUDA execution time: {cuda_time:.4f} ms")
                
                if "cpu" in results and results["cpu"] > 0:
                    speedup = results["cpu"] / cuda_time
                    logger.info(f"CUDA speedup over CPU: {speedup:.2f}x")
            except Exception as e:
                logger.error(f"Error benchmarking on CUDA: {e}")
                results["cuda"] = float("nan")
        else:
            logger.info("CUDA benchmarking skipped (device not available or args not provided)")
            
        # Benchmark on XPU if available
        xpu_available = hasattr(torch, 'xpu') and torch.xpu.is_available()
        if xpu_available and args_xpu is not None:
            try:
                xpu_time = self._time_function(operation, *args_xpu, **kwargs)
                results["xpu"] = xpu_time
                logger.info(f"XPU execution time: {xpu_time:.4f} ms")
                
                if "cpu" in results and results["cpu"] > 0:
                    speedup = results["cpu"] / xpu_time
                    logger.info(f"XPU speedup over CPU: {speedup:.2f}x")
                    
                if "cuda" in results and results["cuda"] > 0:
                    speedup = results["cuda"] / xpu_time
                    if speedup > 1:
                        logger.info(f"XPU is {speedup:.2f}x faster than CUDA")
                    else:
                        logger.info(f"CUDA is {1/speedup:.2f}x faster than XPU")
            except Exception as e:
                logger.error(f"Error benchmarking on XPU: {e}")
                results["xpu"] = float("nan")
        else:
            logger.info("XPU benchmarking skipped (device not available or args not provided)")
            
        return results
        
    def benchmark_model(
        self,
        model: torch.nn.Module,
        input_cpu: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        input_cuda: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
        input_xpu: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
    ) -> Dict[str, float]:
        """
        Benchmark a PyTorch model on different devices.
        
        Args:
            model: PyTorch model to benchmark
            input_cpu: Input tensor(s) for CPU
            input_cuda: Input tensor(s) for CUDA (optional)
            input_xpu: Input tensor(s) for XPU (optional)
            
        Returns:
            Dictionary with execution times in milliseconds for each device
        """
        results = {}
        
        # Copy model for each device to avoid issues
        model_cpu = type(model)()
        model_cpu.load_state_dict(model.state_dict())
        model_cpu.eval()
        
        # Benchmark on CPU
        try:
            # Convert input to tuple if it's a single tensor
            if torch.is_tensor(input_cpu):
                input_cpu = (input_cpu,)
                
            cpu_time = self._time_function(model_cpu, *input_cpu)
            results["cpu"] = cpu_time
            logger.info(f"CPU model execution time: {cpu_time:.4f} ms")
        except Exception as e:
            logger.error(f"Error benchmarking model on CPU: {e}")
            results["cpu"] = float("nan")
            
        # Benchmark on CUDA if available
        if torch.cuda.is_available() and input_cuda is not None:
            try:
                model_cuda = type(model)()
                model_cuda.load_state_dict(model.state_dict())
                model_cuda = model_cuda.cuda()
                model_cuda.eval()
                
                # Convert input to tuple if it's a single tensor
                if torch.is_tensor(input_cuda):
                    input_cuda = (input_cuda,)
                    
                cuda_time = self._time_function(model_cuda, *input_cuda)
                results["cuda"] = cuda_time
                logger.info(f"CUDA model execution time: {cuda_time:.4f} ms")
                
                if "cpu" in results and results["cpu"] > 0:
                    speedup = results["cpu"] / cuda_time
                    logger.info(f"CUDA model speedup over CPU: {speedup:.2f}x")
            except Exception as e:
                logger.error(f"Error benchmarking model on CUDA: {e}")
                results["cuda"] = float("nan")
        else:
            logger.info("CUDA model benchmarking skipped (device not available or input not provided)")
            
        # Benchmark on XPU if available
        xpu_available = hasattr(torch, 'xpu') and torch.xpu.is_available()
        if xpu_available and input_xpu is not None:
            try:
                model_xpu = type(model)()
                model_xpu.load_state_dict(model.state_dict())
                model_xpu = model_xpu.to("xpu")
                model_xpu.eval()
                
                # Convert input to tuple if it's a single tensor
                if torch.is_tensor(input_xpu):
                    input_xpu = (input_xpu,)
                    
                xpu_time = self._time_function(model_xpu, *input_xpu)
                results["xpu"] = xpu_time
                logger.info(f"XPU model execution time: {xpu_time:.4f} ms")
                
                if "cpu" in results and results["cpu"] > 0:
                    speedup = results["cpu"] / xpu_time
                    logger.info(f"XPU model speedup over CPU: {speedup:.2f}x")
                    
                if "cuda" in results and results["cuda"] > 0:
                    speedup = results["cuda"] / xpu_time
                    if speedup > 1:
                        logger.info(f"XPU model is {speedup:.2f}x faster than CUDA")
                    else:
                        logger.info(f"CUDA model is {1/speedup:.2f}x faster than XPU")
            except Exception as e:
                logger.error(f"Error benchmarking model on XPU: {e}")
                results["xpu"] = float("nan")
        else:
            logger.info("XPU model benchmarking skipped (device not available or input not provided)")
            
        return results


def generate_benchmark_report(
    benchmark_results: Dict[str, Dict[str, float]],
    title: str = "Performance Benchmark Report",
) -> str:
    """
    Generate a formatted benchmark report.
    
    Args:
        benchmark_results: Dictionary mapping operation names to result dictionaries
        title: Report title
        
    Returns:
        Formatted report string
    """
    report = [f"# {title}", ""]
    
    # Calculate summary statistics
    devices = set()
    for op_results in benchmark_results.values():
        devices.update(op_results.keys())
        
    devices = sorted(list(devices))
    
    # Generate table header
    header = ["Operation"] + [device.upper() for device in devices]
    if len(devices) > 1:
        for i in range(len(devices)):
            for j in range(i+1, len(devices)):
                header.append(f"{devices[i].upper()}/{devices[j].upper()} Ratio")
                
    # Generate table rows
    rows = []
    for op_name, op_results in benchmark_results.items():
        row = [op_name]
        
        # Add timing for each device
        for device in devices:
            if device in op_results:
                row.append(f"{op_results[device]:.4f} ms")
            else:
                row.append("N/A")
                
        # Add speedup ratios
        if len(devices) > 1:
            for i in range(len(devices)):
                for j in range(i+1, len(devices)):
                    dev_i, dev_j = devices[i], devices[j]
                    if dev_i in op_results and dev_j in op_results and op_results[dev_j] > 0:
                        ratio = op_results[dev_i] / op_results[dev_j]
                        if ratio > 1:
                            row.append(f"{ratio:.2f}x slower")
                        else:
                            row.append(f"{1/ratio:.2f}x faster")
                    else:
                        row.append("N/A")
        
        rows.append(row)
        
    # Format as table
    col_widths = [max(len(row[i]) for row in [header] + rows) for i in range(len(header))]
    
    report.append(" | ".join(f"{header[i]:{col_widths[i]}}" for i in range(len(header))))
    report.append("-|-".join("-" * width for width in col_widths))
    
    for row in rows:
        report.append(" | ".join(f"{row[i]:{col_widths[i]}}" for i in range(len(row))))
    
    return "\n".join(report)
