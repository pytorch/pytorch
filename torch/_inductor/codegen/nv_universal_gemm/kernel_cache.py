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
from typing import Any, Literal

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


# Reentrant: get_efc_kernel_with_epilogue holds this lock and may call
# get_kernel_by_name -> _ensure_caches, which re-acquires it on the same thread.
_cache_lock = threading.RLock()

# Full kernel manifest: kernel_name -> kernel object. Built lazily (~14s, ~290K
# operators) only as a LAST-RESORT fallback -- the common paths (choice
# generation and subprocess precompile) resolve operators via cheap
# args-filtered queries and cache them in _ops_by_name.
_kernel_by_name_cache: dict[str, Any] | None = None


def _operand_dtype_str(operand: Any) -> str | None:
    """Best-effort cutlass dtype name for a kernel operand or args operand."""
    dtype = getattr(operand, "dtype", None)
    if dtype is None:
        tensor = getattr(operand, "tensor", None)
        dtype = getattr(tensor, "dtype", None)
    return None if dtype is None else str(dtype)


def _build_kernel_cache() -> dict[str, Any]:
    """Build the full kernel name -> kernel manifest (fallback path only)."""
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
    log.debug("NVGEMM kernel manifest built: %d kernels", len(cache))
    return cache


def _ensure_caches() -> None:
    global _kernel_by_name_cache
    if _kernel_by_name_cache is None:
        with _cache_lock:
            if _kernel_by_name_cache is None:
                _kernel_by_name_cache = _build_kernel_cache()


def _get_kernel_cache() -> dict[str, Any]:
    """Return the fallback manifest, building it lazily if needed.

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
    # mode/swizzle are cutlass.operators enums (ScaleMode/ScaleSwizzleMode) that
    # are unhashable, so key on their .name to keep the sig usable as a dict key.
    mode = getattr(mode, "name", mode)
    swizzle = getattr(swizzle, "name", swizzle)
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


# Per-process cache of instantiated operators, keyed by operator_name. Populated
# incrementally by the args-filtered queries below (choice generation and the
# subprocess precompile fast path). One args query returns the whole set of
# operators compatible with a shape (hundreds); we index all of them, so any
# later lookup -- another config of that shape, another shape sharing those
# templates, or get_kernel_by_name for a chosen
# kernel -- resolves in O(1) without rebuilding the ~14s manifest. Operator
# names embed the arch (e.g. "..._sm100_..."), so the name alone is unambiguous
# within a process. Reusing an operator across different args is safe for the
# same reason the manifest always has been: an operator is a config template
# (its args-specific state lives in the artifact returned by compile(args)), and
# every (name, args) pairing routed here was validated by supports(args).
_ops_by_name: dict[str, Any] = {}

# Memoizes partition results. `partition_compatible_kernels` is called once per
# GEMM node during choice generation, but FLUX has hundreds of nodes sharing a
# handful of shapes (e.g. 114 nodes are all 4608x3072x3072); memoizing collapses
# that to one args query per unique shape.
_partition_cache: dict[tuple, list[list[Any]]] = {}


def _args_query_candidates(args: Any, cc: int, efc_only: bool) -> list[Any]:
    """Compatible operators via an args-filtered get_operators query (~0.05s).

    cutlass prunes by supports(args) and the device target internally. Only
    valid for the dense GEMM path: for that path this returns the identical set
    as the full-manifest scan (verified), avoiding the ~14s manifest build.
    """
    import cutlass.operators

    metadata_filter = (
        (lambda md: "EFC" in md.operator_class.__name__) if efc_only else None
    )
    return cutlass.operators.get_operators(
        args=args, target_sm=f"{cc}a", metadata_filter=metadata_filter
    )


def _filter_supported(kernels: Any, args: Any, cc: int) -> list[Any]:
    """Keep the kernels that support `args` on this device.

    Prunes by (A, B) operand dtype first (a cheap string compare) so the
    expensive supports() runs only on the ~hundreds that could possibly match --
    the ~290K manifest spans all dtypes, and a GEMM only matches its own.
    """
    device_target = _device_target(cc)
    dtype_a = _operand_dtype_str(getattr(args, "A", None))
    dtype_b = _operand_dtype_str(getattr(args, "B", None))
    out = []
    for kernel in kernels:
        if kernel.designed_for_min_cc > cc:
            continue
        operands = kernel.metadata.operands
        if (
            _operand_dtype_str(getattr(operands, "A", None)) != dtype_a
            or _operand_dtype_str(getattr(operands, "B", None)) != dtype_b
        ):
            continue
        if not device_target.supports_operators_from(kernel.metadata.supported_targets):
            continue
        if kernel.supports(args).error is not None:
            continue
        out.append(kernel)
    return out


def _manifest_candidates(args: Any, cc: int, efc_only: bool) -> list[Any]:
    """Compatible operators via a full-manifest scan (last-resort fallback)."""
    kernels = _get_kernel_cache().values()
    if efc_only:
        kernels = (
            kernel
            for kernel in kernels
            if "EFC" in kernel.metadata.operator_class.__name__
        )
    return _filter_supported(kernels, args, cc)


def _blockscaled_provider_classes() -> list[Any]:
    """The operator classes that produce block-scaled (fp4/fp8) kernels.

    Both the cutlass built-in and the vendored provider matter: for NVFP4 the
    vendored kernels supply most of the working configs (60 of 96), so missing
    either under-generates and breaks scaled autotuning.
    """
    classes: list[Any] = []
    try:
        from cutlass.operators.providers.cutedsl.gemm.sm100_dense_blockscaled_static_persistent import (
            PersistentDenseBlockScaledGemmOperator,
        )

        classes.append(PersistentDenseBlockScaledGemmOperator)
    except ImportError:
        pass
    try:
        from torch._inductor.kernel.vendored_templates.cutedsl.wrappers.dense_blockscaled_gemm_kernel import (
            VendoredDenseBlockScaledGemmKernel,
        )

        classes.append(VendoredDenseBlockScaledGemmKernel)
    except ImportError:
        pass
    return classes


@functools.cache
def _blockscaled_operators() -> tuple:
    """Generate the architecture-neutral block-scaled operator pool."""
    ops: list[Any] = []
    for cls in _blockscaled_provider_classes():
        ops.extend(cls.generate_operators(lambda md: True, args=None))
    return tuple(ops)


def _scaled_operand_type_signature(args: Any) -> tuple:
    def operand_signature(operand: Any) -> tuple:
        scale = getattr(operand, "scale", None)
        return (
            _operand_dtype_str(operand),
            _operand_dtype_str(scale),
            str(getattr(operand, "mode", None)),
            str(getattr(operand, "swizzle", None)),
        )

    return (
        operand_signature(args.A),
        operand_signature(args.B),
        _operand_dtype_str(args.out),
        str(getattr(args, "accumulator_type", None)),
    )


def _scaled_metadata_type_signature(metadata: Any) -> tuple:
    operands = metadata.operands

    def operand_signature(operand: Any) -> tuple:
        scale = getattr(operand, "scale", None)
        return (
            _operand_dtype_str(operand),
            _operand_dtype_str(scale),
            str(getattr(operand, "mode", None)),
            str(getattr(operand, "swizzle", None)),
        )

    return (
        operand_signature(operands.A),
        operand_signature(operands.B),
        _operand_dtype_str(operands.out),
        str(getattr(operands, "accumulator_type", None)),
    )


@functools.cache
def _blockscaled_manifest(cc: int, type_signature: tuple):
    """Build a block-scaled manifest for one operand type recipe.

    The provider set is generated without a concrete target to preserve its
    complete design space. The manifest applies shape and target filtering when
    candidates are requested.
    """
    import cutlass.operators

    manifest = cutlass.operators.Manifest()
    manifest.add_operators(
        [
            op
            for op in _blockscaled_operators()
            if _scaled_metadata_type_signature(op.metadata) == type_signature
        ]
    )
    return manifest


def _scaled_candidates(args: Any, cc: int, efc_only: bool) -> list[Any]:
    """Compatible operators for a scaled GEMM via direct block-scaled enumeration.

    get_operators(args=...) derives operand configs from the args and
    under-generates for scaled (e.g. ~36 of 96 valid fp4 kernels), so we scan the
    block-scaled sub-provider's full design space instead -- complete, and ~50x
    cheaper than the manifest. Falls back to the manifest if the provider is
    unavailable or nothing matches (e.g. a future non-block-scaled scaled dtype).
    """
    manifest = _blockscaled_manifest(cc, _scaled_operand_type_signature(args))
    if manifest.operators:
        metadata_filter = (
            (lambda md: "EFC" in md.operator_class.__name__) if efc_only else None
        )
        out = manifest.filter_operators(
            args=args,
            metadata_filter=metadata_filter,
            target_sm=f"{cc}a",
        )
        if out:
            return out
    return _manifest_candidates(args, cc, efc_only)


def partition_compatible_kernels(
    args: Any,
    cc: int,
    classifier: Callable[[Any], int],
    num_buckets: int,
    efc_only: bool = False,
    candidate_source: Literal["args", "scaled", "manifest"] = "manifest",
    classifier_key: str | None = None,
) -> list[list[Any]]:
    """Partition the operators compatible with `args` into N buckets.

    `classifier(metadata)` returns a bucket index in [0, num_buckets-1] or -1 to
    drop the kernel. `candidate_source` selects how the compatible set is found,
    all of which avoid the ~14s full manifest except "manifest":
      - "args"     dense GEMM: cheap args-filtered get_operators query
      - "scaled"   scaled GEMM: direct block-scaled sub-provider enumeration
      - "manifest" fallback (e.g. grouped GEMM): full-manifest scan
    `efc_only` restricts to epilogue-fusion-capable kernels (the addmm/bias
    path). Results are memoized per (shape, cc, num_buckets, efc_only, source).
    """
    sig = _partition_sig(args)
    cache_key = (
        (sig, cc, num_buckets, efc_only, candidate_source, classifier_key)
        if sig is not None and classifier_key is not None
        else None
    )
    if cache_key is not None:
        cached = _partition_cache.get(cache_key)
        if cached is not None:
            return [list(b) for b in cached]

    if candidate_source == "args":
        candidates = _args_query_candidates(args, cc, efc_only)
    elif candidate_source == "scaled":
        candidates = _scaled_candidates(args, cc, efc_only)
    elif candidate_source == "manifest":
        candidates = _manifest_candidates(args, cc, efc_only)
    else:
        raise ValueError(f"Unknown NVGEMM candidate source: {candidate_source}")

    buckets: list[list[Any]] = [[] for _ in range(num_buckets)]
    for kernel in candidates:
        # Share every compatible operator so a later get_kernel_by_name (e.g. the
        # scheduling max_active_clusters lookup for the chosen kernel) hits
        # without building the manifest.
        _ops_by_name.setdefault(kernel.metadata.operator_name, kernel)
        bucket = classifier(kernel.metadata)
        if bucket < 0:
            continue
        buckets[bucket].append(kernel)
    log.debug(
        "Partitioned %d compatible operators into buckets %s",
        len(candidates),
        [len(b) for b in buckets],
    )
    if cache_key is not None:
        _partition_cache[cache_key] = [list(b) for b in buckets]
    return buckets


def get_kernel_by_name(kernel_name: str) -> Any:
    """Get a kernel by name.

    Checks the incrementally-populated operator cache first (filled by the
    args-filtered queries during choice generation / precompile), falling back
    to the full manifest only if the name was never resolved via args.
    """
    kernel = _ops_by_name.get(kernel_name)
    if kernel is not None:
        return kernel
    return _get_kernel_cache().get(kernel_name)


def get_kernel_by_name_via_args(kernel_name: str, args: Any, cc: int) -> Any:
    """Fast single-kernel lookup via an args-filtered get_operators query.

    Passing concrete `args` (a cutlass RuntimeArguments) plus the device target
    prunes the operator space to the ~hundreds compatible with this exact GEMM
    in ~0.05s, versus ~14s to enumerate/construct the full ~294K-kernel manifest
    (`get_kernel_by_name`). Used by subprocess precompile workers, which need
    only the single named kernel and would otherwise each rebuild the whole
    manifest.

    A miss runs one args query and indexes every operator it returns into
    _ops_by_name, so the ~50ms cost is paid once per distinct operator (not once
    per lookup) and is amortized across all configs and shapes a persistent
    worker handles. Returns None if no operator matches the name (caller then
    falls back to the full manifest).
    """
    import cutlass.operators

    cached = _ops_by_name.get(kernel_name)
    if cached is not None:
        return cached

    ops = cutlass.operators.get_operators(args=args, target_sm=f"{cc}a")
    for op in ops:
        # setdefault: keep an already-cached (possibly already-compiled) object
        # rather than replacing it with a fresh instance from this query.
        _ops_by_name.setdefault(op.metadata.operator_name, op)
    return _ops_by_name.get(kernel_name)


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
        _ops_by_name.clear()
        _blockscaled_operators.cache_clear()
        _blockscaled_manifest.cache_clear()


class _NVGEMMCacheWrapper:
    def cache_clear(self) -> None:
        clear_cache()


from torch._inductor.utils import clear_on_fresh_cache


clear_on_fresh_cache(_NVGEMMCacheWrapper())


def get_efc_kernel_with_epilogue(
    efc_kernel_name: str,
    epilogue_args: Any,
    epilogue_source: str = "",
    base_kernel: Any | None = None,
) -> Any:
    """Get (or create and cache) an EFC kernel bound to a specific epilogue.

    epilogue_source is preferred over inspect.getsource — generated functions
    produce unstable source strings that can't be hashed reliably.

    base_kernel: pre-resolved base EFC kernel (e.g. from the args-filtered fast
    lookup in a subprocess worker). When None, falls back to the full manifest.
    """
    if not epilogue_source:
        epilogue_source = str(epilogue_args) if epilogue_args is not None else ""

    cache_key = (
        efc_kernel_name,
        epilogue_source,
        _epilogue_args_signature(epilogue_args),
    )

    with _cache_lock:
        if cache_key in _efc_epilogue_cache:
            log.debug("EFC kernel with epilogue found in cache: %s", efc_kernel_name)
            return _efc_epilogue_cache[cache_key]

        if base_kernel is None:
            # Prefer the args-resolved operator cache (populated by choice
            # generation); only build the full manifest if the name was never
            # seen via args.
            base_kernel = get_kernel_by_name(efc_kernel_name)
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
