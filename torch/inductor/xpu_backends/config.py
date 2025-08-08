"""
Configuration settings for Intel XPU backend in PyTorch Inductor.

This module provides configuration options and settings specific to
Intel GPU optimizations in the PyTorch Inductor framework.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


@dataclass
class XPUInductorConfig:
    """Configuration options for Intel XPU backend in PyTorch Inductor."""
    
    # Enable/disable XPU optimizations
    enabled: bool = True
    
    # Use SYCL caching for kernels
    use_kernel_caching: bool = True
    
    # Enable profiling and performance measurements
    enable_profiling: bool = False
    
    # Maximum workspace size (in MB) for XPU operations
    max_workspace_size: int = 1024
    
    # Use fast approximate math functions when possible
    fast_math: bool = True
    
    # Memory allocation mode (0=default, 1=auto, 2=manual)
    memory_allocation_mode: int = 0
    
    # Enable/disable specific optimizations
    enable_matmul_fusion: bool = True
    enable_conv_fusion: bool = True
    enable_activation_fusion: bool = True
    enable_pointwise_fusion: bool = True
    
    # Minimum sizes for which to use XPU optimized implementations
    min_matmul_size: int = 64  # Minimum matrix size for XPU matmul
    min_conv_channels: int = 16  # Minimum channels for XPU convolution
    
    # Tuning parameters
    tile_size_m: int = 64
    tile_size_n: int = 64
    tile_size_k: int = 16
    
    # List of operations to force on CPU regardless of device
    force_cpu_ops: Set[str] = field(default_factory=set)
    
    # The maximal number of threads per thread group
    max_threads_per_group: int = 512
    
    # Codegen options
    xpu_codegen_format: str = "sycl"  # Options: "sycl", "opencl"
    
    # Debug options
    debug_level: int = 0  # 0=none, 1=basic, 2=verbose
    debug_dump_ir: bool = False
    debug_dump_kernels: bool = False
    
    @classmethod
    def from_env(cls) -> "XPUInductorConfig":
        """
        Create configuration from environment variables.
        
        Returns:
            XPUInductorConfig initialized from environment variables
        """
        config = cls()
        
        # Parse boolean options
        bool_options = {
            "PYTORCH_XPU_INDUCTOR_ENABLED": "enabled",
            "PYTORCH_XPU_KERNEL_CACHING": "use_kernel_caching",
            "PYTORCH_XPU_PROFILING": "enable_profiling",
            "PYTORCH_XPU_FAST_MATH": "fast_math",
            "PYTORCH_XPU_MATMUL_FUSION": "enable_matmul_fusion",
            "PYTORCH_XPU_CONV_FUSION": "enable_conv_fusion",
            "PYTORCH_XPU_ACTIVATION_FUSION": "enable_activation_fusion",
            "PYTORCH_XPU_POINTWISE_FUSION": "enable_pointwise_fusion",
            "PYTORCH_XPU_DEBUG_DUMP_IR": "debug_dump_ir",
            "PYTORCH_XPU_DEBUG_DUMP_KERNELS": "debug_dump_kernels",
        }
        
        for env_var, attr_name in bool_options.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                setattr(config, attr_name, env_value.lower() not in ("0", "false", "off", "no"))
                
        # Parse integer options
        int_options = {
            "PYTORCH_XPU_MAX_WORKSPACE_SIZE": "max_workspace_size",
            "PYTORCH_XPU_MEMORY_ALLOCATION_MODE": "memory_allocation_mode",
            "PYTORCH_XPU_MIN_MATMUL_SIZE": "min_matmul_size",
            "PYTORCH_XPU_MIN_CONV_CHANNELS": "min_conv_channels",
            "PYTORCH_XPU_TILE_SIZE_M": "tile_size_m",
            "PYTORCH_XPU_TILE_SIZE_N": "tile_size_n",
            "PYTORCH_XPU_TILE_SIZE_K": "tile_size_k",
            "PYTORCH_XPU_MAX_THREADS_PER_GROUP": "max_threads_per_group",
            "PYTORCH_XPU_DEBUG_LEVEL": "debug_level",
        }
        
        for env_var, attr_name in int_options.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                try:
                    setattr(config, attr_name, int(env_value))
                except ValueError:
                    logger.warning(f"Invalid value for {env_var}: {env_value}")
                    
        # Parse string options
        if "PYTORCH_XPU_CODEGEN_FORMAT" in os.environ:
            config.xpu_codegen_format = os.environ["PYTORCH_XPU_CODEGEN_FORMAT"]
            
        # Parse list/set options
        if "PYTORCH_XPU_FORCE_CPU_OPS" in os.environ:
            ops = os.environ["PYTORCH_XPU_FORCE_CPU_OPS"].split(",")
            config.force_cpu_ops = set(op.strip() for op in ops if op.strip())
            
        return config
    
    def log_config(self, level=logging.INFO) -> None:
        """
        Log the current configuration settings.
        
        Args:
            level: Logging level
        """
        logger.log(level, "XPU Inductor Configuration:")
        for key, value in self.__dict__.items():
            logger.log(level, f"  {key}: {value}")


# Global singleton configuration
config = XPUInductorConfig.from_env()
