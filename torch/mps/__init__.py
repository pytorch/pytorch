# mypy: allow-untyped-defs
r"""
This package enables an interface for accessing MPS (Metal Performance Shaders) backend in Python.
Metal is Apple's API for programming metal GPU (graphics processor unit). Using MPS means that increased
performance can be achieved, by running work on the metal GPU(s).
See https://developer.apple.com/documentation/metalperformanceshaders for more details.
"""

from typing import Union

import torch
from torch import Tensor


_is_in_bad_fork = getattr(torch._C, "_mps_is_in_bad_fork", lambda: False)
_default_mps_generator: torch._C.Generator = None  # type: ignore[assignment]


# local helper function (not public or exported)
def _get_default_mps_generator() -> torch._C.Generator:
    global _default_mps_generator
    if _default_mps_generator is None:
        _default_mps_generator = torch._C._mps_get_default_generator()
    return _default_mps_generator


def device_count() -> int:
    r"""Returns the number of available MPS devices."""
    return int(torch._C._has_mps and torch._C._mps_is_available())


def synchronize() -> None:
    r"""Waits for all kernels in all streams on a MPS device to complete."""
    return torch._C._mps_deviceSynchronize()


def get_rng_state(device: Union[int, str, torch.device] = "mps") -> Tensor:
    r"""Returns the random number generator state as a ByteTensor.

    Args:
        device (torch.device or int, optional): The device to return the RNG state of.
            Default: ``'mps'`` (i.e., ``torch.device('mps')``, the current MPS device).
    """
    return _get_default_mps_generator().get_state()


def set_rng_state(
    new_state: Tensor, device: Union[int, str, torch.device] = "mps"
) -> None:
    r"""Sets the random number generator state.

    Args:
        new_state (torch.ByteTensor): The desired state
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'mps'`` (i.e., ``torch.device('mps')``, the current MPS device).
    """
    new_state_copy = new_state.clone(memory_format=torch.contiguous_format)
    _get_default_mps_generator().set_state(new_state_copy)


def manual_seed(seed: int) -> None:
    r"""Sets the seed for generating random numbers.

    Args:
        seed (int): The desired seed.
    """
    # the torch.mps.manual_seed() can be called from the global
    # torch.manual_seed() in torch/random.py. So we need to make
    # sure mps is available (otherwise we just return without
    # erroring out)
    if not torch._C._has_mps:
        return
    seed = int(seed)
    _get_default_mps_generator().manual_seed(seed)


def seed() -> None:
    r"""Sets the seed for generating random numbers to a random number."""
    _get_default_mps_generator().seed()


def empty_cache() -> None:
    r"""Releases all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other GPU applications and clears
    the MPS graph cache.

    .. note::
       This function performs comprehensive cleanup including:
       - Synchronizes all command buffers (waits for GPU to finish)
       - Frees buffers pending completion (buffers_pending_free)
       - Clears the buffer allocator cache
       - Clears the MPSEvent pool
       - Clears graph and kernel caches (if called)

       This is the recommended way to prevent memory leaks between epochs.

    .. warning::
       Calling this frequently during training can impact performance as graphs
       will need to be recompiled. It's recommended to call this between epochs
       or when memory pressure is high.

    Example::
        >>> # At the end of each training epoch
        >>> for epoch in range(num_epochs):
        ...     # training loop
        ...     for batch in dataloader:
        ...         # forward, backward, optimizer step
        ...         pass
        ...     # Clear all caches at end of epoch
        ...     torch.mps.empty_cache()

    or at the end of a batch of training::
    >>> # At the end of each training batch
        >>> for epoch in range(num_epochs):
        ...     # training loop
        ...     for batch in dataloader:
        ...         # forward, backward, optimizer step
        ...         torch.mps.empty_cache()
        ...         pass
        ...     # Clear all caches at end of epoch
    Ideal placement should be at the end of the call for `optimizer.step()`
    """
    # Synchronization, buffer freeing, and event pool clearing
    # now handled internally in the C++ implementation
    torch._C._mps_emptyCache()

    # Clear graph and kernel caches
    if hasattr(torch._C, '_mps_emptyGraphCache'):
        torch._C._mps_emptyGraphCache()


def empty_graph_cache() -> None:
    r"""Clears the cached MPSGraph and MPSKernel objects.

    This function clears the internal graph and kernel caches used by the MPS backend.
    These caches can accumulate during training, especially when using varying tensor
    shapes, leading to memory leaks between epochs.

    .. warning::
       This is an advanced function. Clearing the graph cache will cause graph
       recompilation on the next operation, which may impact performance temporarily.
       For most use cases, :func:`empty_cache` is sufficient as it includes graph
       cache clearing.

    .. note::
       This function only clears the graph/kernel caches, not the buffer allocator
       cache. To clear both, use :func:`empty_cache` instead.

    Example::
        >>> # Clear only graph caches (advanced usage)
        >>> torch.mps.synchronize()
        >>> torch.mps.empty_graph_cache()
        >>>
        >>> # Clear both graph and buffer caches (recommended)
        >>> torch.mps.empty_cache()  # Preferred approach
    """
    if hasattr(torch._C, '_mps_emptyGraphCache'):
        torch._C._mps_emptyGraphCache()
    else:
        # Fallback for older PyTorch versions
        import warnings
        warnings.warn(
            "empty_graph_cache() is not available in this PyTorch build. "
            "Falling back to empty_cache().",
            RuntimeWarning
        )
        empty_cache()


def set_per_process_memory_fraction(fraction) -> None:
    r"""Set memory fraction for limiting process's memory allocation on MPS device.
    The allowed value equals the fraction multiplied by recommended maximum device memory
    (obtained from Metal API device.recommendedMaxWorkingSetSize).
    If trying to allocate more than the allowed value in a process, it will raise an out of
    memory error in allocator.

    Args:
        fraction(float): Range: 0~2. Allowed memory equals total_memory * fraction.

    .. note::
       Passing 0 to fraction means unlimited allocations
       (may cause system failure if out of memory).
       Passing fraction greater than 1.0 allows limits beyond the value
       returned from device.recommendedMaxWorkingSetSize.
    """

    if not isinstance(fraction, float):
        raise TypeError("Invalid type for fraction argument, must be `float`")
    if fraction < 0 or fraction > 2:
        raise ValueError(f"Invalid fraction value: {fraction}. Allowed range: 0~2")

    torch._C._mps_setMemoryFraction(fraction)


def current_allocated_memory() -> int:
    r"""Returns the current GPU memory occupied by tensors in bytes.

    .. note::
       The returned size does not include cached allocations in
       memory pools of MPSAllocator.
    """
    return torch._C._mps_currentAllocatedMemory()


def driver_allocated_memory() -> int:
    r"""Returns total GPU memory allocated by Metal driver for the process in bytes.

    .. note::
       The returned size includes cached allocations in MPSAllocator pools
       as well as allocations from MPS/MPSGraph frameworks.
    """
    return torch._C._mps_driverAllocatedMemory()


def recommended_max_memory() -> int:
    r"""Returns recommended max Working set size for GPU memory in bytes.

    .. note::
       Recommended max working set size for Metal.
       returned from device.recommendedMaxWorkingSetSize.
    """
    return torch._C._mps_recommendedMaxMemory()


def compile_shader(source: str):
    r"""Compiles compute shader from source and allows one to invoke kernels
    defined there from the comfort of Python runtime
    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_MPS)
        >>> lib = torch.mps.compile_shader(
        ... "kernel void full(device float* out, constant float& val, uint idx [[thread_position_in_grid]]) { out[idx] = val; }"
        ...  )
        >>> x = torch.zeros(16, device="mps")
        >>> lib.full(x, 3.14)
    """
    from pathlib import Path

    from torch.utils._cpp_embed_headers import _embed_headers

    if not hasattr(torch._C, "_mps_compileShader"):
        raise RuntimeError("MPS is not available")
    source = _embed_headers(
        [l + "\n" for l in source.split("\n")],
        [Path(__file__).parent.parent / "include"],
        set(),
    )
    return torch._C._mps_compileShader(source)


def is_available() -> bool:
    return device_count() > 0


def set_command_buffer_flush_threshold(threshold: int) -> None:
    r"""Sets the command buffer flush threshold for MPS operations.

    This controls how many operations are accumulated in a command buffer before it's
    automatically flushed to prevent unbounded memory growth. The MPS backend maintains
    a persistent LRU (Least Recently Used) cache that tracks operation signatures across
    command buffer flushes, providing performance benefits by reusing compiled graphs.

    Args:
        threshold (int): Number of operations before flushing (default: 100).
            - Lower values (e.g., 25-50): More frequent flushes, lower memory usage
            - Higher values (e.g., 150-200): Less frequent flushes, higher performance

    .. note::
       The default threshold of 100 provides an excellent balance between memory safety
       and performance. The command buffer is flushed every 100 operations to prevent
       unbounded accumulation of completion handlers and resources.

    .. note::
       The LRU cache persists across command buffer flushes for the entire training
       session. This allows compiled graphs to be reused across epochs and batches,
       providing significant performance benefits for repetitive workloads.

       - Cache tracks operations by their signatures (e.g., MPSGraph pointers)
       - Cached operations can reuse compiled graphs (faster execution)
       - Cache is NOT cleared when buffer is flushed (persistent across training)

    .. warning::
       Setting this too high (e.g., >500) may cause memory accumulation during long
       training runs as completion handlers and resources build up in the command buffer.
       Setting this too low (e.g., <25) may impact performance due to frequent flushes.

    Example::
        >>> # Standard setting for most workloads (default)
        >>> torch.mps.set_command_buffer_flush_threshold(100)
        >>>
        >>> # For very long training runs or limited memory
        >>> torch.mps.set_command_buffer_flush_threshold(50)
        >>>
        >>> # For maximum performance with adequate memory
        >>> torch.mps.set_command_buffer_flush_threshold(200)
    """
    if not isinstance(threshold, int):
        raise TypeError("threshold must be an integer")
    if threshold < 1:
        raise ValueError(f"threshold must be at least 1, got {threshold}")

    torch._C._mps_setCommandBufferFlushThreshold(threshold)


def get_command_buffer_flush_threshold() -> int:
    r"""Returns the current command buffer flush threshold.

    Returns:
        int: The current number of operations before automatic flushing occurs.

    Example::
        >>> current_threshold = torch.mps.get_command_buffer_flush_threshold()
        >>> print(f"Current threshold: {current_threshold}")
    """
    return torch._C._mps_getCommandBufferFlushThreshold()


def set_max_operation_cache_size(size: int) -> None:
    r"""Sets the maximum size of the persistent LRU operation cache.

    This controls how many unique operation signatures are cached across command
    buffer flushes. The LRU cache persists throughout training and enables compiled
    graph reuse for performance. When the cache is full, least recently used
    operations are evicted.

    Args:
        size (int): Maximum number of operations in the cache (default: 100).
            - Lower values (e.g., 50): Less memory for cache, more evictions
            - Higher values (e.g., 200): More memory for cache, fewer evictions

    .. note::
       The default size of 100 matches the flush threshold and works well for most
       models. Increase this for very large models with many unique operations.

    .. note::
       This cache is separate from the operation count threshold. The cache persists
       across flushes to enable graph reuse, while the operation count resets on flush.

    .. warning::
       Setting this too low may cause frequent evictions and reduce the performance
       benefit of graph caching. Setting it too high uses more memory for tracking.

    Example::
        >>> # Standard setting (default)
        >>> torch.mps.set_max_operation_cache_size(100)
        >>>
        >>> # For very large models with diverse operations
        >>> torch.mps.set_max_operation_cache_size(200)
        >>>
        >>> # For memory-constrained environments
        >>> torch.mps.set_max_operation_cache_size(50)
    """
    if not isinstance(size, int):
        raise TypeError("size must be an integer")
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}")

    torch._C._mps_setMaxOperationCacheSize(size)


def get_max_operation_cache_size() -> int:
    r"""Returns the current maximum operation cache size.

    Returns:
        int: The maximum number of operations that can be cached.

    Example::
        >>> current_size = torch.mps.get_max_operation_cache_size()
        >>> print(f"Current cache size: {current_size}")
    """
    return torch._C._mps_getMaxOperationCacheSize()


from . import profiler
from .event import Event


__all__ = [
    "compile_shader",
    "device_count",
    "get_rng_state",
    "manual_seed",
    "seed",
    "set_rng_state",
    "synchronize",
    "empty_cache",
    "empty_graph_cache",
    "set_per_process_memory_fraction",
    "current_allocated_memory",
    "driver_allocated_memory",
    "Event",
    "profiler",
    "recommended_max_memory",
    "is_available",
    "set_command_buffer_flush_threshold",
    "get_command_buffer_flush_threshold",
    "set_max_operation_cache_size",
    "get_max_operation_cache_size",
]
