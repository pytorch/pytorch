"""
Utility functions for Intel XPU backend operations.

This module provides helper functions and utilities for working with
Intel GPUs in the PyTorch Inductor framework.
"""

import os
import logging
import subprocess
from typing import Dict, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)


def get_xpu_device_info() -> Dict[str, any]:
    """
    Get information about available Intel XPU devices.
    
    Returns:
        Dictionary containing device information
    """
    info = {
        "available": False,
        "count": 0,
        "devices": []
    }
    
    try:
        if not hasattr(torch, 'xpu') or not torch.xpu.is_available():
            return info
            
        info["available"] = True
        info["count"] = torch.xpu.device_count()
        
        for i in range(info["count"]):
            device_props = torch.xpu.get_device_properties(i)
            info["devices"].append({
                "id": i,
                "name": device_props.name,
                "total_memory": device_props.total_memory,
                "eu_count": getattr(device_props, "eu_count", None),
                "max_shared_memory_per_block": getattr(device_props, "max_shared_memory_per_block", None),
                "max_threads_per_block": getattr(device_props, "max_threads_per_block", None),
                "clock_rate": getattr(device_props, "clock_rate", None),
            })
            
    except Exception as e:
        logger.warning(f"Error getting XPU device information: {e}")
        
    return info


def optimize_memory_layout(tensor: torch.Tensor) -> torch.Tensor:
    """
    Optimize memory layout of a tensor for XPU operations.
    
    This function rearranges the memory layout of a tensor to maximize
    performance on Intel GPU architectures.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Tensor with optimized memory layout
    """
    if not tensor.is_xpu:
        return tensor
        
    # For contiguous tensors, we don't need to do anything
    if tensor.is_contiguous():
        return tensor
        
    # Make a contiguous copy with the optimal memory layout for XPU
    return tensor.contiguous()


def estimate_kernel_performance(
    op_type: str, 
    input_shapes: List[List[int]],
    output_shape: List[int],
    dtype: torch.dtype = torch.float32
) -> Dict[str, float]:
    """
    Estimate the performance of an operation on XPU hardware.
    
    This function provides theoretical performance estimates for common
    operations on Intel GPU hardware based on operation type and tensor shapes.
    
    Args:
        op_type: Type of operation (e.g., "matmul", "conv2d")
        input_shapes: Shapes of input tensors
        output_shape: Shape of output tensor
        dtype: Data type of the tensors
        
    Returns:
        Dictionary with performance estimates (FLOPS, memory bandwidth, etc.)
    """
    result = {
        "estimated_flops": 0,
        "estimated_memory_bytes": 0,
        "estimated_execution_time_ms": 0,
    }
    
    # Calculate element sizes based on dtype
    element_size = {
        torch.float16: 2,
        torch.float32: 4,
        torch.float64: 8,
        torch.int8: 1,
        torch.int16: 2,
        torch.int32: 4,
        torch.int64: 8,
    }.get(dtype, 4)  # Default to float32 size
    
    if op_type == "matmul":
        if len(input_shapes) >= 2:
            # Get dimensions from input shapes
            m, k = input_shapes[0][-2], input_shapes[0][-1]
            k_b, n = input_shapes[1][-2], input_shapes[1][-1]
            
            # Matrix multiplication requires 2*m*n*k FLOPs
            # (m*k multiplications + m*k additions per output element)
            result["estimated_flops"] = 2 * m * n * k
            
            # Memory access: read m*k + k*n elements, write m*n elements
            result["estimated_memory_bytes"] = (m*k + k*n + m*n) * element_size
            
            # Very rough estimate - depends heavily on hardware
            estimated_flops_per_second = 1e12  # 1 TFLOPS (very approximate)
            result["estimated_execution_time_ms"] = (result["estimated_flops"] / estimated_flops_per_second) * 1000
            
    elif op_type == "conv2d":
        if len(input_shapes) >= 2:
            # Input shape: (N, C_in, H, W)
            # Weight shape: (C_out, C_in, kH, kW)
            N, C_in, H, W = input_shapes[0]
            C_out, C_in_w, kH, kW = input_shapes[1]
            
            # Output dimensions
            Out_H, Out_W = output_shape[-2], output_shape[-1]
            
            # Convolution FLOPs: 2 * N * C_out * Out_H * Out_W * C_in * kH * kW
            result["estimated_flops"] = 2 * N * C_out * Out_H * Out_W * C_in * kH * kW
            
            # Memory: input + weights + output
            result["estimated_memory_bytes"] = (N*C_in*H*W + C_out*C_in*kH*kW + N*C_out*Out_H*Out_W) * element_size
            
            # Very rough estimate
            estimated_flops_per_second = 1e12  # 1 TFLOPS (very approximate)
            result["estimated_execution_time_ms"] = (result["estimated_flops"] / estimated_flops_per_second) * 1000
    
    return result


def get_optimal_launch_config(
    op_type: str,
    tensor_shapes: List[List[int]],
    block_size: Optional[int] = None
) -> Dict[str, int]:
    """
    Get optimal launch configuration for XPU kernels.
    
    This function determines the optimal thread block and grid sizes
    for launching kernels on Intel GPUs based on the operation type
    and input tensor shapes.
    
    Args:
        op_type: Type of operation (e.g., "matmul", "conv2d")
        tensor_shapes: Shapes of input tensors
        block_size: Optional block size constraint
        
    Returns:
        Dictionary with launch configuration parameters
    """
    config = {
        "block_size_x": 16,
        "block_size_y": 16,
        "block_size_z": 1,
        "grid_size_x": 1,
        "grid_size_y": 1, 
        "grid_size_z": 1,
        "shared_memory_bytes": 0,
    }
    
    # Default block sizes for different operations on Intel GPUs
    default_block_sizes = {
        "matmul": (16, 16, 1),
        "conv2d": (8, 8, 8),
        "element_wise": (256, 1, 1),
        "reduction": (256, 1, 1),
    }
    
    if op_type in default_block_sizes:
        config["block_size_x"], config["block_size_y"], config["block_size_z"] = default_block_sizes[op_type]
    
    # Override with user-provided block size if specified
    if block_size is not None:
        config["block_size_x"] = block_size
        
    # Calculate grid dimensions based on operation type and tensor shapes
    if op_type == "matmul" and len(tensor_shapes) >= 2:
        m, n = tensor_shapes[0][-2], tensor_shapes[1][-1]
        config["grid_size_x"] = (n + config["block_size_x"] - 1) // config["block_size_x"]
        config["grid_size_y"] = (m + config["block_size_y"] - 1) // config["block_size_y"]
        
        # For matrix multiply, allocate shared memory for tiles
        tile_size = min(config["block_size_x"], config["block_size_y"])
        config["shared_memory_bytes"] = 2 * tile_size * tile_size * 4  # 2 tiles * size * sizeof(float)
        
    elif op_type == "element_wise":
        # For element-wise operations, simply divide the total number of elements
        total_elements = 1
        for dim in tensor_shapes[0]:
            total_elements *= dim
            
        config["grid_size_x"] = (total_elements + config["block_size_x"] - 1) // config["block_size_x"]
    
    return config
