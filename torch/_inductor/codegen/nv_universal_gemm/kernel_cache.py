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
import threading
from collections.abc import Callable
from typing import Any

import torch


log = logging.getLogger(__name__)


def _epilogue_args_signature(epilogue_args: Any) -> tuple:
    """Extract a hashable signature of epilogue args for cache keying.

    Two callers with the same `(efc_kernel_name, epilogue_source)` but
    different aux-tensor specs (dtype, shape, stride) would otherwise share
    a kernel object whose internal JIT state is mutated by each call to
    `kernel.compile(args)` — a silent miscompile, since the compiled
    artifact's launch closure reads the kernel's CURRENT JIT at launch time
    rather than the one from when the artifact was built.
    """
    if epilogue_args is None:
        return ()
    tensors = getattr(epilogue_args, "tensors", None)
    if not tensors:
        return ()
    sig: list[tuple] = []
    for name, val in tensors.items():
        if torch.is_tensor(val):
            sig.append(
                (name, "tensor", val.dtype, tuple(val.shape), tuple(val.stride()))
            )
        else:
            sig.append((name, type(val).__name__))
    return tuple(sig)


_cache_lock = threading.Lock()

# Global cache: kernel_name -> kernel object
_kernel_by_name_cache: dict[str, Any] | None = None


def _build_kernel_cache() -> dict[str, Any]:
    """Build the kernel name -> kernel object cache."""
    import cutlass_api

    log.debug("Building NVGEMM kernel cache (this may take a few seconds)...")

    try:
        from torch._inductor.kernel.vendored_templates.cutedsl import (  # noqa: F401
            wrappers,
        )
    except ImportError:
        log.debug("Vendored kernel wrappers not available")

    all_kernels = cutlass_api.get_kernels()
    cache = {k.metadata.kernel_name: k for k in all_kernels}
    log.debug("NVGEMM kernel cache built: %d kernels", len(cache))
    return cache


def _get_kernel_cache() -> dict[str, Any]:
    """Return the kernel cache, initializing lazily if needed.

    Snapshot to local frame: a concurrent clear_cache() rebinding the global to
    None cannot turn the caller's subsequent read into AttributeError.
    """
    global _kernel_by_name_cache
    if _kernel_by_name_cache is None:
        with _cache_lock:
            if _kernel_by_name_cache is None:
                _kernel_by_name_cache = _build_kernel_cache()
    return _kernel_by_name_cache


def get_compatible_kernels(
    args: Any,
    cc: int,
    metadata_filter: Callable[[Any], bool] | None = None,
) -> list[Any]:
    """Get kernels compatible with the given arguments from the cache."""
    cache = _get_kernel_cache()
    compatible = []
    for kernel in cache.values():
        if kernel.metadata.min_cc > cc:
            continue

        if metadata_filter is not None and not metadata_filter(kernel.metadata):
            continue

        status = kernel.supports(args)
        if status.error is not None:
            continue
        compatible.append(kernel)

    log.debug(
        "Found %d compatible kernels from cache of %d total",
        len(compatible),
        len(cache),
    )
    return compatible


def partition_compatible_kernels(
    args: Any,
    cc: int,
    classifier: Callable[[Any], int],
    num_buckets: int,
) -> list[list[Any]]:
    """Partition compatible kernels into N buckets in a single pass.

    `classifier(metadata)` returns a bucket index in [0, num_buckets-1] or
    -1 to drop the kernel. This avoids iterating the full kernel cache
    (~390K entries, each with a non-trivial `supports()` call) once per
    bucket.
    """
    cache = _get_kernel_cache()
    buckets: list[list[Any]] = [[] for _ in range(num_buckets)]
    for kernel in cache.values():
        if kernel.metadata.min_cc > cc:
            continue
        bucket = classifier(kernel.metadata)
        if bucket < 0:
            continue
        status = kernel.supports(args)
        if status.error is not None:
            continue
        buckets[bucket].append(kernel)
    log.debug(
        "Partitioned %s compatible kernels from cache of %d total",
        [len(b) for b in buckets],
        len(cache),
    )
    return buckets


def get_kernel_by_name(kernel_name: str) -> Any:
    """Get a cutlass_api kernel by name using the global cache."""
    return _get_kernel_cache().get(kernel_name)


def ensure_cache_initialized() -> None:
    """Ensure the kernel cache is initialized."""
    _get_kernel_cache()


_efc_epilogue_cache: dict[tuple[str, str, tuple], Any] = {}


def clear_cache() -> None:
    """Clear all kernel caches."""
    global _kernel_by_name_cache, _efc_epilogue_cache
    with _cache_lock:
        _kernel_by_name_cache = None
        _efc_epilogue_cache = {}


class _NVGEMMCacheWrapper:
    def cache_clear(self) -> None:
        clear_cache()


from torch._inductor.utils import clear_on_fresh_cache


clear_on_fresh_cache(_NVGEMMCacheWrapper())


def get_efc_kernel_with_epilogue(
    efc_kernel_name: str,
    epilogue_args: Any,
    epilogue_source: str = "",
) -> Any:
    """Get (or create and cache) an EFC kernel bound to a specific epilogue.

    epilogue_source is preferred over inspect.getsource — generated functions
    produce unstable source strings that can't be hashed reliably.
    """
    if not epilogue_source:
        epilogue_source = str(epilogue_args) if epilogue_args is not None else ""

    cache_key = (
        efc_kernel_name,
        epilogue_source,
        _epilogue_args_signature(epilogue_args),
    )

    base_cache = _get_kernel_cache()

    with _cache_lock:
        if cache_key in _efc_epilogue_cache:
            log.debug("EFC kernel with epilogue found in cache: %s", efc_kernel_name)
            return _efc_epilogue_cache[cache_key]

        base_kernel = base_cache.get(efc_kernel_name)
        if base_kernel is None:
            log.debug("Base EFC kernel not found: %s", efc_kernel_name)
            return None

        from cutlass_api.metadata import EpilogueMetadata, KernelMetadata

        epilogue_metadata = EpilogueMetadata.from_args(epilogue_args)

        base_metadata = base_kernel.metadata
        new_metadata = KernelMetadata(
            operands=base_metadata.operands,
            design=base_metadata.design,
            kernel_name=base_metadata.kernel_name,
            kernel_class=base_metadata.kernel_class,
            min_cc=base_metadata.min_cc,
            epilogue=epilogue_metadata,
        )

        kernel_class = base_metadata.kernel_class
        new_kernel = kernel_class(new_metadata)

        _efc_epilogue_cache[cache_key] = new_kernel
        log.debug("Created and cached EFC kernel with epilogue: %s", efc_kernel_name)

        return new_kernel
