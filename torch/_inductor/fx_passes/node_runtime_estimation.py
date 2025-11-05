"""
Collective runtime estimation using CUDA events and power-of-2 rounding.
Cache structure: LocalCache["collective_benchmarking"]["{op}: ({bytes} bytes)"] = runtime_ms
"""

from __future__ import annotations

import functools
from functools import lru_cache
from typing import Any, Optional

import torch
from torch._dynamo.utils import dynamo_timed
from torch._logging import getArtifactLogger, trace_structured


# Setup logger for artifact logging
log = getArtifactLogger(__name__, "node_runtime_estimation")


# ============================================================================
# Cache (following overlap_scheduling.py)
# ============================================================================

@functools.cache
def get_estimation_cache() -> Any:
    from torch._inductor.codecache import LocalCache
    return LocalCache()


@lru_cache(maxsize=1)
def clear_collective_cache_once() -> None:
    """Clear all collective benchmarks once per process (lru_cache ensures one-time)."""
    from torch._inductor import config

    if not config.aten_distributed_optimizations.benchmark_collectives:
        return

    get_estimation_cache().set_value("collective_benchmarking", value=None)


def get_cached_runtime(key: str) -> Optional[float]:
    """Get cached runtime from persistent cache under 'collective_benchmarking' namespace."""
    cache = get_estimation_cache().lookup("collective_benchmarking")
    if cache is None:
        return None
    return cache.get(key)  # type: ignore[return-value]


def set_cached_runtime(key: str, value: float) -> None:
    """Set cached runtime in persistent cache under 'collective_benchmarking' namespace."""
    cache = get_estimation_cache().lookup("collective_benchmarking") or {}
    cache[key] = value
    get_estimation_cache().set_value("collective_benchmarking", value=cache)


# ============================================================================
# Utilities
# ============================================================================

def get_hint(x: int | torch.SymInt) -> Optional[int]:
    if isinstance(x, int):
        return x
    assert isinstance(x, torch.SymInt)
    return x.node.hint if x.node.has_hint() else None


def can_benchmark_collective() -> bool:
    """Check if we can benchmark collectives (not fake process group)."""
    import torch.distributed as c10d

    if not c10d.is_initialized():
        return False
    
    pg = c10d.distributed_c10d._get_default_group()
    if torch.distributed.distributed_c10d.get_backend(pg) == "fake":
        return False

    return True


# ============================================================================
# Collective Benchmarking
# ============================================================================

def benchmark_collective_with_cuda_events(
    n: torch.fx.Node,
    nruns: int = 2,
) -> tuple[float | None, str]:
    """
    Benchmark collective with CUDA events. Returns (runtime_ms, cache_key) or (None, "") on failure.

    Uses power-of-2 rounding for bandwidth-bound ops and byte-based caching
    (dtype-agnostic: fp32 512 == fp16 1024 both = 2048 bytes).
    """
    import torch.distributed as c10d
    from torch._inductor import fx_utils
    from torch._inductor.runtime.runtime_utils import next_power_of_2
    from torch.utils._python_dispatch import _disable_current_modes

    # Early check: can we actually run collectives?
    if not can_benchmark_collective():
        return None, ""

    success, args, kwargs = fx_utils.get_fake_args_kwargs(n)
    if not success:
        return None, ""

    # Extract actual input size in BYTES (first tensor argument)
    actual_bytes = None
    actual_dtype = None
    actual_device = None

    def extract_tensor_info(t: torch.Tensor) -> torch.Tensor:
        nonlocal actual_bytes, actual_dtype, actual_device
        if actual_bytes is None:
            shape = [get_hint(dim) for dim in t.shape]
            if any(s is None for s in shape):
                return t

            total_elems = 1
            for dim in shape:
                total_elems *= dim

            actual_bytes = total_elems * t.element_size()
            actual_dtype = t.dtype
            actual_device = t.device
        return t

    torch.utils._pytree.tree_map_only(torch.Tensor, extract_tensor_info, (args, kwargs))

    if actual_bytes is None:
        return None, ""

    # Find power-of-2 BYTE bounds
    lower_pow2_bytes = next_power_of_2(actual_bytes) // 2
    upper_pow2_bytes = next_power_of_2(actual_bytes)

    # Helper to benchmark a specific power-of-2 byte size
    def benchmark_bytes(bytes_pow2: int, dtype: torch.dtype) -> tuple[float, str]:
        # Cache key by BYTES (dtype-agnostic)
        key = f"{n.target}: ({bytes_pow2} bytes)"

        # Check persistent cache
        if (cached := get_cached_runtime(key)) is not None:
            return cached, key

        # Not in cache, need to benchmark
        # Calculate number of elements for this byte size
        element_size = torch.tensor([], dtype=dtype).element_size()
        num_elements = bytes_pow2 // element_size

        # Create empty tensor for benchmarking
        benchmark_tensor = torch.empty(num_elements, dtype=dtype, device=actual_device)

        # Replace tensors in args/kwargs using tree_map_only pattern
        def to_benchmark_tensor(t: torch.Tensor) -> torch.Tensor:
            return benchmark_tensor

        with _disable_current_modes():
            bench_args, bench_kwargs = torch.utils._pytree.tree_map_only(
                torch.Tensor,
                to_benchmark_tensor,
                (args, kwargs),
            )

        # Warmup: call collective once
        torch.cuda.synchronize()
        try:
            n.target(*bench_args, **bench_kwargs)  # type: ignore[operator]
        except Exception:
            # If direct invocation fails, return None
            return None, key

        # Benchmark with CUDA events
        comm_time = 0.0
        for _ in range(nruns):
            c10d.barrier()
            torch.cuda.synchronize()

            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)

            start_evt.record()
            n.target(*bench_args, **bench_kwargs)  # type: ignore[operator]
            end_evt.record()
            end_evt.synchronize()

            comm_time += start_evt.elapsed_time(end_evt)

        runtime_ms = comm_time / nruns
        return runtime_ms, key

    # If exact power-of-2 bytes, just benchmark it
    if actual_bytes == lower_pow2_bytes or actual_bytes == upper_pow2_bytes:
        return benchmark_bytes(actual_bytes, actual_dtype)

    # Otherwise, benchmark bounds and interpolate
    lower_runtime, lower_key = benchmark_bytes(lower_pow2_bytes, actual_dtype)
    upper_runtime, upper_key = benchmark_bytes(upper_pow2_bytes, actual_dtype)

    # Linear interpolation
    ratio = (actual_bytes - lower_pow2_bytes) / (upper_pow2_bytes - lower_pow2_bytes)
    interpolated = lower_runtime + ratio * (upper_runtime - lower_runtime)

    # Return key showing actual bytes (for tracking)
    actual_key = f"{n.target}: ({actual_bytes} bytes)"

    return interpolated, actual_key
