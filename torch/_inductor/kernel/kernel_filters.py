from __future__ import annotations

import dataclasses
import threading
from typing import Any, Callable, Dict, List, Optional, TypeVar
from functools import wraps
import logging

import torch

from torch.utils._ordered_set import OrderedSet
from torch._inductor.kernel.kernel_lut import get_lookup_table

# Type definitions for better type checking
T = TypeVar('T')
ConfigType = TypeVar('ConfigType')
ConfigFilterFunc = Callable[[List[ConfigType]], List[ConfigType]]

logger = logging.getLogger(__name__)

# Thread-local storage for filter parameters
class FilterContext:
    """Thread-local storage for filter parameters."""
    
    def __init__(self):
        self.scale_params: Dict[str, Any] = {
            "m": 128,
            "n": 128,
            "k": 128,
            "scale": 1.0,
            "has_int8_tensor": False,
            "exclude": lambda m, n, k: False,
        }

# Create a thread-local instance
_filter_context = threading.local()

def get_filter_context() -> FilterContext:
    """Get the thread-local filter context."""
    if not hasattr(_filter_context, "instance"):
        _filter_context.instance = FilterContext()
    return _filter_context.instance

def set_scale_params(
    m: int, 
    n: int, 
    k: int, 
    scale: float = 1.0, 
    has_int8_tensor: bool = False,
    exclude: Callable[[int, int, int], bool] = lambda m, n, k: False
) -> None:
    """
    Set parameters for the scale_mm_configs filter.
    
    Args:
        m: First matrix dimension
        n: Second matrix dimension
        k: Third matrix dimension
        scale: Scaling factor for block sizes
        has_int8_tensor: Whether int8 tensors are involved
        exclude: Function to filter out configs
    """
    ctx = get_filter_context()
    ctx.scale_params = {
        "m": m,
        "n": n,
        "k": k,
        "scale": scale,
        "has_int8_tensor": has_int8_tensor,
        "exclude": exclude,
    }

class GemmConfigFilterRegistry:
    """
    Registry for Gemm config filter functions.
    
    This class provides a middleware system for managing Gemm configurations.
    Users can register functions that take in a list of triton gemm configs
    and return a new list of triton gemm configs (either filtered or changed).
    
    The registered functions are called in the order they were registered.
    """
    
    def __init__(self):
        self.filters: List[ConfigFilterFunc] = []
        self.filter_names: List[str] = []
    
    def register(self, func: Optional[ConfigFilterFunc] = None, *, name: Optional[str] = None) -> Any:
        """
        Register a filter function to the registry.
        
        Can be used as a decorator:
            @registry.register
            def my_filter(configs):
                return [c for c in configs if some_condition(c)]
                
        Or called directly:
            registry.register(my_filter)
            
        Args:
            func: The filter function to register
            name: Optional name for the filter function
            
        Returns:
            The original function if used as a decorator, or None if called directly
        """
        def decorator(filter_func: ConfigFilterFunc) -> ConfigFilterFunc:
            filter_name = name or filter_func.__name__
            
            @wraps(filter_func)
            def wrapped_filter(configs: List[ConfigType]) -> List[ConfigType]:
                result = filter_func(configs)
                logger.debug(f"Filter '{filter_name}' processed {len(configs)} configs -> {len(result)} configs")
                return result
            
            self.filters.append(wrapped_filter)
            self.filter_names.append(filter_name)
            logger.info(f"Registered Gemm config filter: {filter_name}")
            return filter_func
            
        if func is not None:
            return decorator(func)
        return decorator
    
    def apply_filters(self, configs: List[ConfigType]) -> List[ConfigType]:
        """
        Apply all registered filters to the configs in order.
        
        Args:
            configs: The initial list of configs
            
        Returns:
            The filtered/modified list of configs
        """
        result = configs
        for filter_func in self.filters:
            result = filter_func(result)
        return result
    
    def clear(self) -> None:
        """Clear all registered filters."""
        self.filters = []
        self.filter_names = []
    
    def get_registered_filters(self) -> List[str]:
        """Get the names of all registered filters."""
        return self.filter_names.copy()


# Create a global instance of the registry
gemm_config_registry = GemmConfigFilterRegistry()

# Example of how to register a custom filter:
#
# @gemm_config_registry.register
# def my_custom_filter(configs):
#     """
#     Filter out configs with block_m < 64
#     """
#     return [c for c in configs if c.block_m >= 64]
#
# # Or register with a specific name:
# @gemm_config_registry.register(name="filter_by_block_size")
# def filter_large_blocks(configs):
#     """
#     Filter out configs with very large block sizes
#     """
#     return [c for c in configs if c.block_m * c.block_n <= 16384]

@gemm_config_registry.register(name="kernel_lut_filter")
def kernel_lut_filter(configs):
    """
    Filter configs based on a lookup table file.
    
    This filter uses a lookup table file to find the optimal configuration
    for a given problem size. If a matching entry is found in the lookup table,
    only that configuration is returned. Otherwise, all configs are returned.
    
    Args:
        configs: List of triton gemm configs to filter
        
    Returns:
        Filtered list of configs based on the lookup table
    """
    if not configs:
        return configs
    
    try:
        # Get the lookup table
        lut = get_lookup_table()
        if lut is None:
            logger.info("No lookup table found, returning all configs")
            return configs
        
        # Get the matrix dimensions from the filter context
        ctx = get_filter_context()
        m = ctx.scale_params["m"]
        n = ctx.scale_params["n"]
        k = ctx.scale_params["k"]
        
        # Filter configs based on the lookup table
        filtered_configs = lut.filter_configs(configs, m, n, k)
        
        logger.info(f"Kernel LUT filter: {len(configs)} configs -> {len(filtered_configs)} configs")
        return filtered_configs
    except Exception as e:
        # If there's any error in the lookup table filtering, fall back to returning all configs
        logger.warning(f"Error in kernel_lut_filter: {e}. Returning all configs.")
        return configs


@gemm_config_registry.register(name="model_predicted_top_k")
def model_predicted_top_k(configs, top_k=None):
    """
    Filter configs to keep only the top k configs as predicted by the neural network model.
    
    This filter uses the pre-trained neural network model from mymodel.py to predict
    the performance of each config and returns the top k configs with the best
    predicted performance.
    
    Args:
        configs: List of triton gemm configs to filter
        top_k: Number of top configs to keep (default: determined by matmul_gemm_autotune_benchmarking_space)
        
    Returns:
        List of the top k configs with best predicted performance
    """
    if not configs:
        return configs
    
    try:
        import time
        from torch._inductor.mymodel import wrappedmodel
        from torch._inductor.virtualized import V
        from torch._inductor import config as inductor_config
        from torch._inductor.template_heuristics import CUDAConfigHeuristic
        
        # Determine the top_k value based on the matmul_gemm_autotune_benchmarking_space config
        if top_k is None:
            benchmarking_space = inductor_config.matmul_gemm_autotune_benchmarking_space
            if isinstance(benchmarking_space, int):
                top_k = benchmarking_space
            elif benchmarking_space == "EXHAUSTIVE":
                # Use the size of the exhaustive configs list
                top_k = len(CUDAConfigHeuristic().exhaustive_configs)
            else:  # "DEFAULT" or any other value
                # Use the size of the default configs list
                top_k = len(CUDAConfigHeuristic().mm_configs)
            
            logger.info(f"Using top_k={top_k} based on matmul_gemm_autotune_benchmarking_space={benchmarking_space}")
        
        # If top_k is larger than the number of configs, just return all configs
        if top_k >= len(configs):
            logger.info(f"top_k ({top_k}) >= number of configs ({len(configs)}), returning all configs")
            return configs
        
        # Get the matrix dimensions from the filter context
        ctx = get_filter_context()
        m = ctx.scale_params["m"]
        n = ctx.scale_params["n"]
        k = ctx.scale_params["k"]
        
        # Get the dtype from the first config's kwargs
        # This is a simplification - in a real scenario, you might want to get this from elsewhere
        first_config = configs[0]
        kwargs = first_config.all_kwargs()
        
        # Default to float16 if we can't determine the dtype
        import torch
        dtype = torch.float16
        
        # Encode the configs for the model
        start_time = time.time()
        encoded_configs = wrappedmodel.encode(m, n, k, dtype, configs)
        encoding_time = time.time() - start_time
        
        # Run inference to get performance predictions
        start_time = time.time()
        predictions = torch.exp(wrappedmodel.inference(encoded_configs))
        inference_time = time.time() - start_time
        
        # Convert to list and pair with configs
        prediction_values = predictions.flatten().tolist()
        config_pairs = list(zip(prediction_values, configs))
        
        # Sort by predicted performance (lower is better)
        config_pairs.sort(key=lambda x: x[0])
        
        # Take the top k configs
        top_configs = [config for _, config in config_pairs[:top_k]]
        
        # Log some information about the filtering
        logger.info(f"Model predicted top_k filter: {len(configs)} configs -> {len(top_configs)} configs")
        logger.info(f"Encoding time: {encoding_time:.4f}s, Inference time: {inference_time:.4f}s")
        
        return top_configs
    except Exception as e:
        # If there's any error in the model prediction, fall back to returning all configs
        logger.warning(f"Error in model_predicted_top_k filter: {e}. Returning all configs.")
        return configs


# Register default filters
def import_default_filters():
    """
    Register the default filters for Gemm configs.
    """
    from torch._inductor import config
    from torch._inductor.virtualized import V
    from torch._inductor.runtime.runtime_utils import next_power_of_2
    
    # Register finalize_mm_configs as a filter
    @gemm_config_registry.register(name="finalize_mm_configs")
    def finalize_mm_configs(configs):
        """
        Finalizes configs after scaling, applying additional constraints.
        """
        from torch.utils._ordered_set import OrderedSet
        
        used: OrderedSet[tuple[int, ...]] = OrderedSet()
        max_mm_configs = config.test_configs.max_mm_configs

        result_configs = []
        for conf in configs:
            # Each warp computes a 16x16 tile = 256 elements
            num_warps = min(conf.num_warps, conf.block_m * conf.block_n // 256)

            # Construct key for finding duplicate configs
            key: tuple[int, ...] = (
                conf.block_m,
                conf.block_n,
                conf.block_k,
                conf.num_stages,
                num_warps,
            )

            # Check if gemm specific arg exists - add to key if does
            group_m = getattr(conf, "group_m", None)
            if group_m is not None:
                key += (group_m,)

            if key not in used and (
                max_mm_configs is None or len(used) < max_mm_configs
            ):
                used.add(key)
                result_configs.append(conf)
                
        return result_configs
    
    # Register scale_mm_configs as a filter
    @gemm_config_registry.register(name="scale_mm_configs")
    def scale_mm_configs(configs):
        """
        Scales and filters matrix multiplication configs based on input size.
        
        Uses parameters from the thread-local filter context.
        """
        from torch._inductor.runtime.runtime_utils import next_power_of_2
        
        # Get parameters from thread-local context
        ctx = get_filter_context()
        m = ctx.scale_params["m"]
        n = ctx.scale_params["n"]
        k = ctx.scale_params["k"]
        scale = ctx.scale_params["scale"]
        has_int8_tensor = ctx.scale_params["has_int8_tensor"]
        exclude = ctx.scale_params["exclude"]
        
        min_block_size = 16
        min_block_size_k = 32 if has_int8_tensor else 16

        # Scale matrix dimensions to power of 2
        m = max(
            next_power_of_2(
                V.graph.sizevars.size_hint(
                    m,
                    fallback=config.unbacked_symint_fallback,  # type: ignore[arg-type]
                )
            ),
            min_block_size,
        )
        n = max(
            next_power_of_2(
                V.graph.sizevars.size_hint(
                    n,
                    fallback=config.unbacked_symint_fallback,  # type: ignore[arg-type]
                )
            ),
            min_block_size,
        )
        k = max(
            next_power_of_2(
                V.graph.sizevars.size_hint(
                    k,
                    fallback=config.unbacked_symint_fallback,  # type: ignore[arg-type]
                )
            ),
            min_block_size_k,
        )

        # Scale each config
        scaled_configs = []
        for c in configs:
            scaled_config = dataclasses.replace(
                c,
                block_m=max(min(int(c.block_m * scale), m), min_block_size),
                block_n=max(min(int(c.block_n * scale), n), min_block_size),
                block_k=max(min(int(c.block_k * scale), k), min_block_size_k),
            )

            if not exclude(
                scaled_config.block_m, scaled_config.block_n, scaled_config.block_k
            ):
                scaled_configs.append(scaled_config)

        return scaled_configs


# Initialize the registry with default filters
import_default_filters()
