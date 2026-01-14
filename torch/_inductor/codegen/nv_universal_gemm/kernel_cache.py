# mypy: allow-untyped-defs
"""
Global kernel cache for NVIDIA Universal GEMM.

This module provides a lazy-initialized cache for cutlass_api kernels,
avoiding expensive manifest scans on every kernel lookup.

The first call to get_kernel_by_name() loads all kernels from cutlass_api
(~10 seconds) and builds a name->kernel dict. Subsequent calls use the
dict for O(1) lookup (~0.1 μs).
"""

import logging
from typing import Any, Optional
from collections.abc import Callable


log = logging.getLogger(__name__)

# Global cache: kernel_name -> kernel object
_kernel_by_name_cache: Optional[dict[str, Any]] = None


def _build_kernel_cache() -> dict[str, Any]:
    """Build the kernel name -> kernel object cache."""
    import cutlass_api

    log.debug("Building NVGEMM kernel cache (this may take a few seconds)...")
    all_kernels = cutlass_api.get_kernels()
    cache = {k.metadata.kernel_name: k for k in all_kernels}
    log.debug("NVGEMM kernel cache built: %d kernels", len(cache))
    return cache


def get_compatible_kernels(
    args: Any,
    cc: int,
    metadata_filter: Optional[Callable[[Any], bool]] = None,
) -> list[Any]:
    """Get kernels compatible with the given arguments from the cache.

    Filters the cached kernels by:
    1. Compute capability
    2. Optional metadata filter
    3. Argument compatibility

    Args:
        args: cutlass_api.arguments.GemmArguments
        cc: CUDA compute capability (e.g., 90 for SM90)
        metadata_filter: Optional filter function on kernel metadata

    Returns:
        List of compatible kernels.

    This reuses the global kernel cache instead of calling get_kernels()
    again, avoiding redundant manifest scans.
    """
    global _kernel_by_name_cache

    if _kernel_by_name_cache is None:
        _kernel_by_name_cache = _build_kernel_cache()

    compatible = []
    for kernel in _kernel_by_name_cache.values():
        # Check compute capability
        if kernel.metadata.min_cc > cc:
            continue
        # Check metadata filter
        if metadata_filter is not None and not metadata_filter(kernel.metadata):
            continue
        # Check if kernel supports the arguments
        # Status.error is None for supported kernels
        status = kernel.supports(args)
        if status.error is not None:
            continue
        compatible.append(kernel)

    log.debug(
        "Found %d compatible kernels from cache of %d total",
        len(compatible),
        len(_kernel_by_name_cache),
    )
    return compatible


def get_kernel_by_name(kernel_name: str) -> Any:
    """Get a cutlass_api kernel by name using the global cache.

    Args:
        kernel_name: The full kernel name (e.g., "cutedsl.PersistentDenseGemmKernel_...")

    Returns:
        The kernel object, or None if not found.
    """
    global _kernel_by_name_cache

    if _kernel_by_name_cache is None:
        _kernel_by_name_cache = _build_kernel_cache()

    return _kernel_by_name_cache.get(kernel_name)


def ensure_cache_initialized() -> None:
    """Ensure the kernel cache is initialized.

    Call this during compilation to front-load the cache building cost,
    rather than paying it on the first runtime kernel call.
    """
    global _kernel_by_name_cache

    if _kernel_by_name_cache is None:
        _kernel_by_name_cache = _build_kernel_cache()


def clear_cache() -> None:
    """Clear the kernel cache."""
    global _kernel_by_name_cache
    _kernel_by_name_cache = None
