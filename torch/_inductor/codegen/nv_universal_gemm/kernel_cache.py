# mypy: allow-untyped-defs
"""
Global kernel cache for NVIDIA Universal GEMM.

This module provides a lazy-initialized cache for cutlass.operators kernels,
avoiding expensive manifest scans on every kernel lookup.

The first call to get_kernel_by_name() loads all kernels from cutlass.operators
(~10 seconds) and builds a name->kernel dict. Subsequent calls use the
dict for O(1) lookup (~0.1 μs).
"""

import functools
import logging
import threading
from collections.abc import Callable
from typing import Any

import torch


log = logging.getLogger(__name__)


@functools.cache
def _device_target(cc: int) -> Any:
    """The CUDA arch target for this device (e.g. cc=100 -> TargetSm '100a').

    Used to reject kernels whose `supported_targets` don't cover this device.
    `designed_for_min_cc` alone is insufficient: an arch-conditional sm90 kernel
    reports min_cc=90 (which is <= a 100 device) yet only lists a cc=90 target
    and fails to compile on sm100 ("expects arch sm_90a, but got sm_100a").
    """
    from cutlass.operators.arch import TargetSm

    return TargetSm.ensure(f"{cc}a")


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


def _is_efc_kernel(kernel: Any) -> bool:
    return "EFC" in kernel.metadata.operator_class.__name__


def _operand_dtype_str(operand: Any) -> str | None:
    """Best-effort cutlass dtype name for a kernel operand or args operand."""
    dtype = getattr(operand, "dtype", None)
    if dtype is None:
        tensor = getattr(operand, "tensor", None)
        dtype = getattr(tensor, "dtype", None)
    return None if dtype is None else str(dtype)


def _build_kernel_cache() -> dict[str, Any]:
    """Build the kernel name -> kernel cache."""
    import cutlass.operators

    log.debug("Building NVGEMM kernel cache (this may take a few seconds)...")

    try:
        from torch._inductor.kernel.vendored_templates.cutedsl import (  # noqa: F401
            wrappers,
        )
    except ImportError:
        log.debug("Vendored kernel wrappers not available")

    all_kernels = cutlass.operators.get_operators()
    cache = {k.metadata.operator_name: k for k in all_kernels}
    log.debug("NVGEMM kernel cache built: %d kernels", len(cache))
    return cache


def _ensure_caches() -> None:
    global _kernel_by_name_cache
    if _kernel_by_name_cache is None:
        with _cache_lock:
            if _kernel_by_name_cache is None:
                _kernel_by_name_cache = _build_kernel_cache()


def _get_kernel_cache() -> dict[str, Any]:
    """Return the kernel cache, initializing lazily if needed.

    Snapshot to local frame: a concurrent clear_cache() rebinding the global to
    None cannot turn the caller's subsequent read into AttributeError.
    """
    _ensure_caches()
    cache = _kernel_by_name_cache
    assert cache is not None  # noqa: S101
    return cache


def _operand_sig(operand: Any) -> tuple | None:
    """Hashable (dtype, shape, stride, scale-sig, mode, swizzle) signature."""
    if operand is None:
        return None
    w = getattr(operand, "tensor", operand)
    dtype = _operand_dtype_str(operand)
    shape = getattr(w, "shape", None)
    stride = getattr(w, "stride", None)
    if dtype is None or shape is None or stride is None:
        return None
    scale = getattr(operand, "scale", None)
    scale_sig = _operand_sig(scale) if scale is not None else None
    # ScaledOperand.mode/.swizzle are distinct from the scale tensor's layout
    # and are not folded into it, yet supports(args) depends on them -- two
    # scaled GEMMs with identical operand/scale layouts but different scale or
    # swizzle modes must not collide. Read from __dict__ (only ScaledOperand
    # sets these) so a dense operand's torch.Tensor.mode method isn't captured.
    d = getattr(operand, "__dict__", {})
    mode, swizzle = d.get("mode"), d.get("swizzle")
    return (dtype, tuple(shape), tuple(stride), scale_sig, mode, swizzle)


def _partition_sig(args: Any) -> tuple | None:
    """Signature capturing everything `supports(args)` depends on, or None if
    it can't be fully determined (falls back to a non-memoized scan)."""
    a = _operand_sig(getattr(args, "A", None))
    b = _operand_sig(getattr(args, "B", None))
    o = _operand_sig(getattr(args, "out", None))
    if a is None or b is None or o is None:
        return None
    return (a, b, o, getattr(args, "accumulator_type", None))


# Memoizes partition results. `partition_compatible_kernels` is called once per
# GEMM node during choice generation, but FLUX has hundreds of nodes sharing a
# handful of shapes (e.g. 114 nodes are all 4608x3072x3072). Re-scanning the
# kernel cache per node dominated FLUX compile time (~169s over 618 calls);
# memoizing collapses that to one scan per unique shape.
_partition_cache: dict[tuple, list[list[Any]]] = {}


def partition_compatible_kernels(
    args: Any,
    cc: int,
    classifier: Callable[[Any], int],
    num_buckets: int,
    efc_only: bool = False,
) -> list[list[Any]]:
    """Partition compatible kernels into N buckets in a single pass.

    `classifier(metadata)` returns a bucket index in [0, num_buckets-1] or
    -1 to drop the kernel. `efc_only` restricts the scan to epilogue-fusion-
    capable kernels (the addmm/bias path). Results are memoized per
    (shape, cc, efc_only) -- the single call site uses a pure metadata-only
    classifier, so identical shapes reuse the scan.
    """
    sig = _partition_sig(args)
    cache_key = (sig, cc, num_buckets, efc_only) if sig is not None else None
    if cache_key is not None:
        cached = _partition_cache.get(cache_key)
        if cached is not None:
            return [list(b) for b in cached]

    candidates = list(_get_kernel_cache().values())
    if efc_only:
        candidates = [k for k in candidates if _is_efc_kernel(k)]
    device_target = _device_target(cc)
    buckets: list[list[Any]] = [[] for _ in range(num_buckets)]
    for kernel in candidates:
        if kernel.designed_for_min_cc > cc:
            continue
        if not device_target.supports_operators_from(kernel.metadata.supported_targets):
            continue
        bucket = classifier(kernel.metadata)
        if bucket < 0:
            continue
        status = kernel.supports(args)
        if status.error is not None:
            continue
        buckets[bucket].append(kernel)
    log.debug(
        "Partitioned %s compatible kernels from %d candidates",
        [len(b) for b in buckets],
        len(candidates),
    )
    if cache_key is not None:
        _partition_cache[cache_key] = [list(b) for b in buckets]
    return buckets


def get_kernel_by_name(kernel_name: str) -> Any:
    """Get a cutlass.operators kernel by name using the global cache."""
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
        _partition_cache.clear()


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

        from cutlass.operators.metadata import EpilogueMetadata, OperatorMetadata

        epilogue_metadata = EpilogueMetadata.from_args(epilogue_args)

        base_metadata = base_kernel.metadata
        new_metadata = OperatorMetadata(
            operands=base_metadata.operands,
            design=base_metadata.design,
            operator_name=base_metadata.operator_name,
            operator_class=base_metadata.operator_class,
            supported_targets=base_metadata.supported_targets,
            epilogue=epilogue_metadata,
        )

        kernel_class = base_metadata.operator_class
        new_kernel = kernel_class(new_metadata)

        _efc_epilogue_cache[cache_key] = new_kernel
        log.debug("Created and cached EFC kernel with epilogue: %s", efc_kernel_name)

        return new_kernel
