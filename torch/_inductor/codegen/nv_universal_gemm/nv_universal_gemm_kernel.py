# mypy: allow-untyped-defs
"""
NVIDIA Universal GEMM kernel code generation.

This module generates Python code that calls cutlass.operators to execute GEMM operations.
The runtime helpers (_nvgemm_run, _nvgemm_precompile, etc.) are imported by the
generated wrapper at runtime, keeping the generated code thin.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib
import logging
import re
import threading
from collections import OrderedDict
from typing import Any, cast, TYPE_CHECKING

from torch._inductor.codegen.common import (
    IndentedBuffer,
    Kernel,
    WorkspaceArg,
    WorkspaceZeroMode,
)
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import CuteDSLOpOverrides
from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_utils import (
    to_cutlass_scale_mode,
)
from torch._inductor.ir import (
    BaseView,
    Buffer,
    ExternKernel,
    MutableBox,
    ReinterpretView,
)
from torch._inductor.kernel.gemm_epilogue import (
    GEMM_ACCUMULATOR_ARG_NAME,
    GemmEpiloguePlan,
    GemmReductionArguments,
)
from torch._inductor.kernel.gemm_epilogue_codegen import (
    CuTeDSLEpilogueSchema,
    CuTeDSLEpilogueSource,
)
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm import GemmVariant
    from torch._inductor.kernel.gemm_epilogue import GemmReductionPlan


log = logging.getLogger(__name__)

_NVGEMM_BIAS_ADD_EPILOGUE_SOURCE = (
    "def _epilogue_fn(accum, bias):\n    D = accum + bias\n    return D"
)


def _local_reduce_source_constant(field: str) -> str:
    return f"_LOCAL_REDUCE_{field.removesuffix('_fn').upper()}_FN_SRC"


def _normalize_epilogue_input_tensor(value: Any, permute=None) -> Any:
    import torch

    if not isinstance(value, torch.Tensor):
        return value
    is_scalar = value.ndim == 0
    if is_scalar:
        value = value.view(1, 1)
    if permute is not None:
        value = value.permute(permute)
    if is_scalar:
        value = value.expand(8, 1).contiguous()
    elif not value.is_contiguous():
        value = value.contiguous()
    return value


@functools.cache
def _cutedsl_epilogue_io(
    epilogue_fn: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    from cutlass.operators.fusion import trace_in_out

    inputs, outputs = trace_in_out(epilogue_fn)
    if GEMM_ACCUMULATOR_ARG_NAME not in inputs:
        inputs = [GEMM_ACCUMULATOR_ARG_NAME, *inputs]
    return tuple(inputs), tuple(outputs)


class CuTeDSLEpilogueArguments:
    """Epilogue arguments for direct CuTeDSL functions that bypass EVT tracing."""

    def __init__(self, epilogue_fn: str, **kwargs: Any) -> None:
        import torch

        inputs, outputs = _cutedsl_epilogue_io(epilogue_fn)
        scalar_broadcast_names = frozenset(
            name
            for name, value in kwargs.items()
            if isinstance(value, torch.Tensor) and value.ndim == 0
        )
        schema = CuTeDSLEpilogueSchema(inputs, outputs, scalar_broadcast_names)
        names = schema.parameter_names
        unexpected = kwargs.keys() - OrderedSet(names)
        missing = OrderedSet(names) - kwargs.keys()
        if missing or unexpected:
            raise ValueError(
                f"CuTeDSL epilogue argument mismatch: missing={missing}, "
                f"unexpected={unexpected}"
            )
        self.epilogue_fn = CuTeDSLEpilogueSource(epilogue_fn, schema)
        self.tensors = OrderedDict((name, kwargs[name]) for name in names)
        self.traced_epilogue = None

    def with_tensors(self, tensors: dict[str, Any]) -> CuTeDSLEpilogueArguments:
        result = object.__new__(type(self))
        result.epilogue_fn = self.epilogue_fn
        result.tensors = OrderedDict(
            (name, tensors[name]) for name in self.epilogue_fn.schema.parameter_names
        )
        result.traced_epilogue = None
        return result

    @property
    def parameters(self) -> list[Any]:
        return list(self.tensors.values())

    @property
    def parameter_names(self) -> list[str]:
        return list(self.tensors)

    def trace(self, accumulator_shape, accumulator_type) -> None:
        return None

    def to_tensor_wrappers(self, permute: list[int] | None = None) -> None:
        from cutlass.operators.utils.tensor import TensorWrapper

        import torch

        input_names = self.epilogue_fn.schema.inputs
        for name, value in self.tensors.items():
            if name in input_names:
                value = _normalize_epilogue_input_tensor(value, permute)
            elif isinstance(value, torch.Tensor) and permute is not None:
                value = value.permute(permute)
            if isinstance(value, torch.Tensor):
                self.tensors[name] = TensorWrapper(value)


@functools.cache
def _nvgemm_source_fingerprint() -> str:
    from torch._inductor.codecache import torch_key

    return torch_key().hex()


def _current_target_sm(dev_idx: int):
    """TargetSm for the current CUDA device, used to stamp disk-cached artifacts."""
    from cutlass.operators.arch import TargetSm

    import torch

    cc = torch.cuda.get_device_capability(dev_idx)
    return TargetSm(cc=cc[0] * 10 + cc[1])


def _make_disk_config_key(
    kernel_name: str,
    variant_name: str,
    accumulator_type: Any,
    scale_type_a: Any | None = None,
    scale_type_b: Any | None = None,
    swizzle_type_a: Any | None = None,
    swizzle_type_b: Any | None = None,
) -> tuple:
    return (
        kernel_name,
        variant_name,
        str(accumulator_type),
        str(scale_type_a),
        str(scale_type_b),
        str(swizzle_type_a),
        str(swizzle_type_b),
        _nvgemm_source_fingerprint(),
    )


def _compile_nvgemm(
    variant_name,
    input_tensors,
    out,
    accumulator_type,
    *,
    kernel_obj=None,
    kernel_name: str | None = None,
    args_kwargs=None,
    epilogue_args=None,
    epilogue_source="",
    fallback_fn=None,
    cc: int | None = None,
    base_kernel=None,
):
    """Compile an NVGEMM artifact, trying a fallback (disk cache) first.

    Thread safety is handled at the dispatch layer: autotuning precompile
    runs in subprocess workers (process-isolated), runtime is
    single-threaded per graph execution.

    kernel_obj: pre-resolved kernel (skips _lookup_gemm_kernel).
    kernel_name: kernel name for _lookup_gemm_kernel.
    base_kernel: pre-reconstructed operator (from metadata passed by the main
        process); lets _lookup_gemm_kernel skip operator discovery entirely.
    args_kwargs: extra kwargs forwarded to _create_gemm_arguments.
    fallback_fn: callable(kernel) -> artifact | None, called before
        compiling (for disk cache lookup).

    Returns (artifact, args, kernel, was_compiled).
    """
    was_compiled = False

    args = _create_gemm_arguments(
        variant_name,
        input_tensors,
        out,
        accumulator_type,
        epilogue=epilogue_args,
        **(args_kwargs or {}),
    )
    if kernel_obj is not None:
        kernel = kernel_obj
    else:
        kernel = _lookup_gemm_kernel(
            kernel_name,  # pyrefly: ignore[bad-argument-type]
            epilogue_args=epilogue_args,
            epilogue_source=epilogue_source,
            args=args if cc is not None else None,
            cc=cc,
            base_kernel=base_kernel,
            epilogue_specialization=_local_reduce_specialization(args_kwargs),
        )

    artifact = None
    if fallback_fn is not None:
        artifact = fallback_fn(kernel)
    if artifact is None:
        artifact = kernel.compile(args)
        was_compiled = True

    return artifact, args, kernel, was_compiled


class CUDAContextMetadata:
    """CUDA device metadata that subprocess workers can't query on their own.

    Subprocess compile workers don't have CUDA initialized (and can't due to
    fork restrictions). This bundles the values the main process must provide.
    """

    __slots__ = ("max_active_clusters", "device_capability")

    def __init__(
        self,
        max_active_clusters: int | None,
        device_capability: tuple[int, int],
    ):
        self.max_active_clusters = max_active_clusters
        self.device_capability = device_capability

    @staticmethod
    def from_kernel(kernel, device) -> CUDAContextMetadata:
        """Build from a cutlass.operators Kernel and torch device in the main process."""
        import torch

        mac = None
        try:
            cluster_shape = getattr(
                getattr(kernel, "impl", None), "cluster_shape_mn", None
            )
            if cluster_shape is not None:
                from cutlass.operators.providers.cutedsl.integration_utils.mma import (
                    get_max_active_clusters,
                )

                mac = get_max_active_clusters(cluster_shape)
        except Exception:
            log.warning("Failed to query max_active_clusters", exc_info=True)
        cc = torch.cuda.get_device_capability(device)
        return CUDAContextMetadata(mac, cc)


def _patch_max_active_clusters(max_active_clusters):
    """Patch get_max_active_clusters in all CuTeDSL modules.

    Returns a list of (module, original_fn) pairs for restoration.
    Subprocess workers don't have CUDA initialized, so they can't query
    the driver; this patches in the pre-computed value from the main process.
    """
    patched = []

    def mac_fn(*a, **kw):
        return max_active_clusters

    for mod_name in _MAX_ACTIVE_CLUSTERS_MODULES:
        try:
            m = importlib.import_module(mod_name)
        except ImportError:
            continue
        if hasattr(m, "get_max_active_clusters"):
            patched.append((m, m.get_max_active_clusters))
            m.get_max_active_clusters = mac_fn  # pyrefly: ignore [missing-attribute]
    return patched


def _restore_max_active_clusters(patched):
    """Undo _patch_max_active_clusters."""
    for m, orig in patched:
        m.get_max_active_clusters = orig  # pyrefly: ignore [missing-attribute]


def _worker_nvgemm_autotuning_precompile(
    kernel_name,
    variant_name,
    accumulator_type,
    input_tensor_meta,
    output_tensor_meta,
    extra_env,
    cuda_ctx,
    scale_type_a=None,
    scale_type_b=None,
    swizzle_type_a=None,
    swizzle_type_b=None,
    has_bias_epilogue=False,
    swap_ab=False,
    metadata=None,
):
    """Subprocess worker: compile one NVGEMM kernel and save to disk cache.

    Called from AsyncCompile.nvgemm_precompile() in a subprocess worker
    so CuTeDSL compilation is process-isolated (no thread-safety issues).
    Returns (None, elapsed_us) for compatibility with the precompile callback.
    """
    import time

    # Share the persistent-worker env/cache handling used by PyCodeCache workers.
    from torch._inductor.runtime.compile_tasks import (
        _apply_subprocess_env_and_clear_caches,
    )

    _apply_subprocess_env_and_clear_caches(extra_env)

    start_ns = time.time_ns()

    import torch
    from torch._inductor.runtime.cutedsl_cache import disk_cache_set
    from torch._inductor.utils import _ensure_fp4_dtype_registered
    from torch._subclasses.fake_tensor import FakeTensorMode

    _ensure_fp4_dtype_registered()

    with FakeTensorMode():
        input_tensors = tuple(
            torch.empty_strided(m.sizes, m.strides, device=m.device, dtype=m.dtype)
            for m in input_tensor_meta
        )
        out = torch.empty_strided(
            output_tensor_meta.sizes,
            output_tensor_meta.strides,
            device=output_tensor_meta.device,
            dtype=output_tensor_meta.dtype,
        )

    # swap_ab: the kernel was selected for the transposed (N, M) problem, so the
    # worker must swap operands here too -- both to resolve the kernel via the
    # args-filtered fast lookup and to key the disk cache the same way the
    # benchmark's make_run_fn does (it swaps before compiling). Mirrors the
    # runtime swap in _nvgemm_run. swap_ab and a bias epilogue never co-occur.
    if swap_ab and len(input_tensors) >= 2:
        a, b = input_tensors[0], input_tensors[1]
        if len(input_tensors) >= 4:
            sa, sb = input_tensors[2], input_tensors[3]
            input_tensors = (b.t(), a.t(), sb, sa) + input_tensors[4:]
        else:
            input_tensors = (b.t(), a.t()) + input_tensors[2:]
        out = out.t()

    helper_kwargs: dict[str, Any] = {}
    if variant_name == "SCALED_GEMM":
        scale_mode_a, swizzle_mode_a, scale_mode_b, swizzle_mode_b = (
            _get_scaled_gemm_modes(
                scale_type_a, swizzle_type_a, scale_type_b, swizzle_type_b
            )
        )
        helper_kwargs = {
            "scale_mode_a": scale_mode_a,
            "swizzle_mode_a": swizzle_mode_a,
            "scale_mode_b": scale_mode_b,
            "swizzle_mode_b": swizzle_mode_b,
        }

    # For an addmm bias choice the last input is the bias, consumed by a
    # bias-add epilogue; the rest are the GEMM operands. Building the epilogue
    # (and keying the cache with it) here lets the parallel precompile hand off
    # to the benchmark, which reads the same key.
    epilogue_args = None
    epilogue_source = ""
    aux_tensors: tuple = ()
    if has_bias_epilogue:
        *gemm_list, bias = input_tensors
        input_tensors = tuple(gemm_list)
        epilogue_args = CuTeDSLEpilogueArguments(
            _NVGEMM_BIAS_ADD_EPILOGUE_SOURCE, bias=bias, D=out
        )
        epilogue_source = "nvgemm_addmm_bias_v2"
        aux_tensors = (bias,)

    cache_key = _create_gemm_cache_key(
        input_tensors, out, has_epilogue=has_bias_epilogue, aux_tensors=aux_tensors
    )
    dev_idx = input_tensors[0].device.index or 0
    disk_config_key = _make_disk_config_key(
        kernel_name,
        variant_name,
        accumulator_type,
        scale_type_a,
        scale_type_b,
        swizzle_type_a,
        swizzle_type_b,
    )
    disk_fn_cache: dict = {}

    # cc for the args-filtered fast kernel lookup: workers can't query the CUDA
    # driver, so use the bundled device capability instead of rebuilding the
    # full ~294K-kernel manifest just to resolve one kernel by name.
    major, minor = cuda_ctx.device_capability
    worker_cc = major * 10 + minor

    patched = _patch_max_active_clusters(cuda_ctx.max_active_clusters)
    try:
        # Best path: the main process passed the chosen operator's metadata, so
        # reconstruct exactly that one operator (~us) instead of enumerating the
        # operator space. Done inside the patched region since construction may
        # query max_active_clusters, which the worker can't get from the driver.
        base_kernel = None
        if metadata is not None:
            try:
                base_kernel = metadata.operator_class(metadata)
            except Exception:
                log.warning(
                    "Failed to reconstruct NVGEMM operator %s from metadata; "
                    "falling back to argument-based lookup",
                    kernel_name,
                    exc_info=True,
                )

        artifact, _, _, was_compiled = _compile_nvgemm(
            variant_name,
            input_tensors,
            out,
            accumulator_type,
            kernel_name=kernel_name,
            args_kwargs=helper_kwargs,
            epilogue_args=epilogue_args,
            epilogue_source=epilogue_source,
            cc=worker_cc,
            base_kernel=base_kernel,
        )

        if was_compiled:
            obj = _unwrap_efc_compiled_obj(artifact.compiled_obj)
            disk_cache_set(
                disk_fn_cache,
                kernel_name,
                disk_config_key,
                cache_key,
                obj,
                dev_idx,
                device_capability=cuda_ctx.device_capability,
            )
    finally:
        _restore_max_active_clusters(patched)

    elapsed_us = (time.time_ns() - start_ns) // 1000
    return None, elapsed_us


# ── Runtime helpers (imported by generated wrapper code at runtime) ───────────


def _get_scaled_gemm_modes(
    scale_type_a: Any | None,
    swizzle_type_a: Any | None,
    scale_type_b: Any | None,
    swizzle_type_b: Any | None,
) -> tuple[Any, Any, Any, Any]:
    scale_mode_a, swizzle_mode_a = to_cutlass_scale_mode(scale_type_a, swizzle_type_a)
    scale_mode_b, swizzle_mode_b = to_cutlass_scale_mode(scale_type_b, swizzle_type_b)
    if any(
        mode is None
        for mode in (scale_mode_a, swizzle_mode_a, scale_mode_b, swizzle_mode_b)
    ):
        raise NotImplementedError("Unsupported scale/swizzle mode for scaled GEMM")
    return scale_mode_a, swizzle_mode_a, scale_mode_b, swizzle_mode_b


def _create_gemm_arguments(
    variant_name: str,
    input_tensors,
    out,
    accumulator_type: Any,
    *,
    scale_mode_a: Any | None = None,
    swizzle_mode_a: Any | None = None,
    scale_mode_b: Any | None = None,
    swizzle_mode_b: Any | None = None,
    epilogue: Any | None = None,
    local_reduce: GemmReductionArguments | None = None,
):
    import cutlass.operators

    local_reduce = local_reduce or GemmReductionArguments()
    has_local_reduce = local_reduce.enabled
    if has_local_reduce and variant_name not in ("GEMM", "SCALED_GEMM"):
        raise NotImplementedError("NVGEMM local reductions require dense GEMM")
    if has_local_reduce and (
        local_reduce.axis not in (0, 1) or local_reduce.group <= 1
    ):
        raise NotImplementedError(
            "NVGEMM local reductions require an M- or N-axis group greater than 1"
        )

    def with_local_reduce(args):
        from cutlass.operators.utils.tensor import TensorWrapper

        def wrap(output):
            return TensorWrapper(output, alignment_bytes=4)

        args.local_reduce = local_reduce.map_tensors(wrap)
        return args

    if epilogue is not None and variant_name == "GROUPED_GEMM":
        raise NotImplementedError(
            "Epilogue fusion is not yet supported for grouped GEMM"
        )

    match variant_name:
        case "GROUPED_GEMM":
            a, b, offsets = input_tensors
            return cutlass.operators.arguments.GroupedGemmArguments(
                a,
                b,
                out,
                accumulator_type=accumulator_type,
                offsets=offsets,
            )

        case "SCALED_GEMM":
            from cutlass.operators.arguments import ScaledOperand

            a, b, scale_a, scale_b = input_tensors
            scaled_a = ScaledOperand(a, scale_a, scale_mode_a, swizzle_mode_a)
            scaled_b = ScaledOperand(b, scale_b, scale_mode_b, swizzle_mode_b)
            kwargs: dict[str, Any] = {"accumulator_type": accumulator_type}
            if epilogue is not None:
                kwargs["epilogue"] = epilogue
            return with_local_reduce(
                cutlass.operators.arguments.GemmArguments(
                    scaled_a,
                    scaled_b,
                    out,
                    **kwargs,
                )
            )

        case "GEMM":
            a, b = input_tensors
            kwargs: dict[str, Any] = {"accumulator_type": accumulator_type}
            if epilogue is not None:
                kwargs["epilogue"] = epilogue
            return with_local_reduce(
                cutlass.operators.arguments.GemmArguments(a, b, out, **kwargs)
            )

        case _:
            raise NotImplementedError(f"Unsupported NVGEMM variant: {variant_name}")


def _lookup_gemm_kernel(
    kernel_name: str,
    *,
    epilogue_args: Any | None = None,
    epilogue_source: str = "",
    args: Any | None = None,
    cc: int | None = None,
    base_kernel: Any | None = None,
    epilogue_specialization: tuple = (),
):
    from torch._inductor.codegen.nv_universal_gemm.kernel_cache import (
        get_efc_kernel_with_epilogue,
        get_kernel_by_name,
        get_kernel_by_name_via_args,
    )

    # Best path (subprocess precompile): the caller reconstructed the exact
    # operator from metadata passed by the main process (operator_class(metadata),
    # ~us), so we skip operator discovery entirely -- no args-filtered query and
    # no full manifest.
    # Fast path (base_kernel absent): resolve the single named kernel via an
    # args-filtered query (~0.05s) instead of building the full manifest (~14s).
    fast = args is not None and cc is not None

    if epilogue_args is None:
        kernel = base_kernel
        # Fast path can miss when the named kernel doesn't support these exact
        # args (e.g. a swap_ab kernel selected for the transposed problem while
        # the worker holds the original operands); fall back to the manifest.
        if kernel is None and fast:
            kernel = get_kernel_by_name_via_args(kernel_name, args, cc)
        if kernel is None:
            kernel = get_kernel_by_name(kernel_name)
        if kernel is None:
            raise RuntimeError(f"Could not find kernel: {kernel_name}")
        return kernel

    if base_kernel is None and fast:
        base_kernel = get_kernel_by_name_via_args(kernel_name, args, cc)
    kernel = get_efc_kernel_with_epilogue(
        kernel_name,
        epilogue_args,
        epilogue_source=epilogue_source,
        base_kernel=base_kernel,
        specialization=epilogue_specialization,
    )
    if kernel is None:
        raise RuntimeError(f"Could not find EFC kernel: {kernel_name}")
    return kernel


def _tensor_sig(t):
    return (t.shape, t.stride(), t.dtype)


def _local_reduce_specialization(variant_kwargs: dict | None) -> tuple:
    if variant_kwargs is None or not (reduction := variant_kwargs.get("local_reduce")):
        return ()
    specialization = tuple(reduction.specialization_items())
    tensor_specialization = tuple(
        (
            field,
            None if tensor is None else _tensor_sig(tensor),
        )
        for field, tensor in reduction.tensor_items()
    )
    return specialization + tensor_specialization


def _create_gemm_cache_key(
    input_tensors,
    out,
    *,
    has_epilogue: bool = False,
    aux_tensors: tuple = (),
    epilogue_specialization: tuple = (),
):
    cache_key = tuple(s for t in input_tensors for s in _tensor_sig(t))
    cache_key = (*cache_key, *_tensor_sig(out))

    if has_epilogue:
        aux_sig = tuple(_tensor_sig(t) for t in aux_tensors)
        return (*cache_key, "epilogue", aux_sig, epilogue_specialization)
    return cache_key


def _unwrap_efc_compiled_obj(compiled_obj):
    """Extract the inner JIT function from an EFC closure for disk serialization."""
    if hasattr(compiled_obj, "__closure__") and compiled_obj.__closure__:
        inner = compiled_obj.__closure__[0].cell_contents
        if hasattr(inner, "export_to_c"):
            return inner
    return compiled_obj


def _init_efc_jit(kernel, epilogue_args):
    """Lightweight EFC JIT init from epilogue args, without a full kernel.compile().

    EFC.compile() calls analyze_epilogue_with_arguments() then creates the JIT
    object. We replicate just those two steps so that disk-cached artifacts can
    be rewrapped without recompiling the CuTe DSL kernel.
    """
    from cutlass.operators.providers.cutedsl.evt.common_efc import EFC
    from cutlass.operators.utils.tensor import TensorWrapper

    efc = kernel.impl.efc
    params = [
        e.compile_time_tensor if isinstance(e, TensorWrapper) else e
        for e in epilogue_args.parameters
    ]
    efc.analyze_epilogue_with_arguments(params)
    efc.jit = EFC.JIT(efc, efc.epilogue_parameter_names)


def _rewrap_efc_compiled_obj(compiled_fn, kernel, epilogue_args=None):
    """Reconstruct the EFC wrapped_launch closure from a loaded JIT function.

    When epilogue_args is provided and efc.jit is missing, performs a
    lightweight initialization (analyze + JIT creation) so that
    disk-cached EFC artifacts can be reused without a full recompile.
    Falls back to returning None (triggering recompile) only when
    epilogue_args is not available.
    """
    if not hasattr(kernel, "impl") or not hasattr(kernel.impl, "efc"):
        return compiled_fn
    if not hasattr(kernel.impl.efc, "jit"):
        if epilogue_args is not None:
            _init_efc_jit(kernel, epilogue_args)
        else:
            return None

    from cutlass.operators.providers.cutedsl.gemm.sm100_static_persistent_efc import (
        Operand as KernelOperand,
        TensorWrapper,
    )

    def wrapped_launch(a_tensor, b_tensor, stream, *supplemental_args):
        runtime_args = [
            e.runtime_tensor
            if isinstance(e, TensorWrapper)
            else (e.tensor.runtime_tensor if isinstance(e, KernelOperand) else e)
            for e in supplemental_args
        ]
        # A disk-reloaded artifact is the raw kernel module func: it takes the
        # epilogue tensors unpacked (a, b, stream, aux..., out), unlike the
        # freshly-JIT-compiled closure which expects pack_arguments(...). All
        # callers of _rewrap_efc_compiled_obj load from disk, so pass unpacked.
        return compiled_fn(a_tensor, b_tensor, stream, *runtime_args)

    return wrapped_launch


def _update_reuse_args_tensors(
    variant_name, args, input_tensors, out, epilogue_args, args_kwargs=None
) -> bool:
    """Redirect the cached args at this call's runtime tensors.

    A cache hit on ``mem_key`` guarantees identical shapes/strides/dtypes, so
    only the data pointers change; mutating ``TensorWrapper._runtime_tensor`` in
    place is sufficient and lets us skip the (expensive) _create_gemm_arguments
    rebuild. Returns False for GROUPED_GEMM (unsupported), signalling the caller
    to use the rebuild path.
    """
    if variant_name == "GROUPED_GEMM":
        return False
    if variant_name == "SCALED_GEMM":
        a, b, scale_a, scale_b = input_tensors
        args.A.quantized.tensor._runtime_tensor = a
        args.A.scale.tensor._runtime_tensor = scale_a
        args.B.quantized.tensor._runtime_tensor = b
        args.B.scale.tensor._runtime_tensor = scale_b
        args.out.tensor._runtime_tensor = out
    else:
        # GEMM: dense mm and the EFC bias/epilogue addmm path.
        a, b = input_tensors
        args.A.tensor._runtime_tensor = a
        args.B.tensor._runtime_tensor = b
        args.out.tensor._runtime_tensor = out
    epilogue = getattr(args, "epilogue", None)
    if epilogue is not None:
        source = epilogue.epilogue_fn
        input_names = (
            source.schema.inputs if isinstance(source, CuTeDSLEpilogueSource) else ()
        )
        for name, wrapper in epilogue.tensors.items():
            val = epilogue_args.tensors[name]
            runtime_tensor = getattr(val, "runtime_tensor", val)
            if name in input_names:
                runtime_tensor = _normalize_epilogue_input_tensor(runtime_tensor)
            wrapper._runtime_tensor = runtime_tensor
    runtime_reduce = args_kwargs.get("local_reduce") if args_kwargs else None
    for field, output in args.local_reduce.tensor_items():
        if output is not None:
            if runtime_reduce is None:
                raise AssertionError("expected runtime local-reduction arguments")
            output._runtime_tensor = getattr(runtime_reduce, field)
    return True


def _clear_reuse_args_tensors(variant_name, args, epilogue_args):
    """Drop the runtime tensors held by cached args after a launch.

    kernel.run enqueues the launch synchronously and reads the pointers during
    the call, and the caller keeps the tensors alive, so clearing our cached
    references is safe. Required for cudagraph capture, which fails if a cached
    object retains a tensor from the graph pool between iterations.
    """
    if variant_name == "SCALED_GEMM":
        args.A.quantized.tensor._runtime_tensor = None
        args.A.scale.tensor._runtime_tensor = None
        args.B.quantized.tensor._runtime_tensor = None
        args.B.scale.tensor._runtime_tensor = None
        args.out.tensor._runtime_tensor = None
    else:
        args.A.tensor._runtime_tensor = None
        args.B.tensor._runtime_tensor = None
        args.out.tensor._runtime_tensor = None
    epilogue = getattr(args, "epilogue", None)
    if epilogue is not None:
        for wrapper in epilogue.tensors.values():
            wrapper._runtime_tensor = None
        # kernel.run traces the epilogue on first launch, and the trace retains
        # the real output/aux tensors in example_inputs (used only for tracing,
        # not the launch). Replace them with meta tensors so the cached args
        # hold no real storage -- otherwise an intermediate output stays live
        # and CUDA graph capture fails. Mirrors _meta_epilogue_metadata.
        import torch

        te = getattr(epilogue, "traced_epilogue", None)
        example_inputs = getattr(te, "example_inputs", None) if te else None
        if example_inputs:
            for name, val in example_inputs.items():
                if isinstance(val, torch.Tensor) and val.device.type != "meta":
                    example_inputs[name] = torch.empty_strided(
                        val.shape, val.stride(), dtype=val.dtype, device="meta"
                    )
    for _, output in args.local_reduce.tensor_items():
        if output is not None:
            output._runtime_tensor = None


def _nvgemm_run(
    variant_name: str,
    kernel_name: str,
    input_tensors: tuple,
    out,
    accumulator_type,
    compiled_cache: dict,
    disk_fn_cache: dict,
    module_path: str,
    disk_config_key: tuple,
    *,
    stream=None,
    workspace=None,
    variant_kwargs: dict | None = None,
    epilogue_args=None,
    epilogue_source: str = "",
    has_epilogue: bool = False,
    aux_tensors: tuple = (),
    swap_ab: bool = False,
):
    if swap_ab and len(input_tensors) >= 2:
        import torch

        reduction = variant_kwargs.get("local_reduce") if variant_kwargs else None
        if isinstance(reduction, GemmReductionArguments) and reduction.enabled:
            raise NotImplementedError(
                "NVGEMM swap_ab does not support fused local reductions"
            )
        a, b = input_tensors[0], input_tensors[1]
        if len(input_tensors) >= 4:
            sa, sb = input_tensors[2], input_tensors[3]
            input_tensors = (b.t(), a.t(), sb, sa) + input_tensors[4:]
        else:
            input_tensors = (b.t(), a.t()) + input_tensors[2:]
        # Swapped GEMM computes (N, M) = out.t(); write it zero-copy into a
        # transposed (column-major) view of the real (M, N) output -- no temp
        # buffer / copy (the kernel handles column-major C).
        out = out.t()
        if epilogue_args is not None:
            for name, value in epilogue_args.tensors.items():
                if isinstance(value, torch.Tensor) and value.ndim == 2:
                    transposed = value.t()
                    if transposed.shape[0] == 1:
                        stride = transposed.stride(1)
                        transposed = (
                            transposed.contiguous()
                            if stride != 1
                            else transposed.as_strided(
                                transposed.shape, (transposed.shape[1], 1)
                            )
                        )
                    elif transposed.shape[1] == 1:
                        stride = transposed.stride(0)
                        transposed = (
                            transposed.contiguous()
                            if stride != 1
                            else transposed.as_strided(
                                transposed.shape, (1, transposed.shape[0])
                            )
                        )
                    epilogue_args.tensors[name] = transposed
    from cutlass.operators.artifact import CompiledArtifact

    from torch._inductor.runtime.cutedsl_cache import disk_cache_get, disk_cache_set

    epilogue_specialization = _local_reduce_specialization(variant_kwargs)
    cache_key = _create_gemm_cache_key(
        input_tensors,
        out,
        has_epilogue=has_epilogue,
        aux_tensors=aux_tensors,
        epilogue_specialization=epilogue_specialization,
    )
    dev_idx = input_tensors[0].device.index or 0
    mem_key = (cache_key, dev_idx)

    if mem_key in compiled_cache:
        artifact = compiled_cache[mem_key]
        # Fast path: reuse the previously-built args, redirecting only the
        # runtime tensor pointers. Skips _create_gemm_arguments, whose cost
        # (get_type_hints + deepcopy + full epilogue/layout reprocessing)
        # depends solely on shapes/layout -- fixed for a given mem_key.
        reuse = getattr(artifact, "_nvgemm_reuse", None)
        if reuse is not None:
            args, kernel, reuse_lock = reuse
            # The cached args are shared mutable state. Serialize the
            # redirect->launch->clear so concurrent launches on the same cached
            # artifact (e.g. two graph executions sharing this kernel) don't
            # stomp each other's runtime tensor pointers.
            with reuse_lock:
                if _update_reuse_args_tensors(
                    variant_name,
                    args,
                    input_tensors,
                    out,
                    epilogue_args,
                    variant_kwargs,
                ):
                    try:
                        kernel.run(
                            args,
                            artifact,
                            stream=stream,
                            workspace=workspace,
                            assume_supported_args=True,
                        )
                    finally:
                        _clear_reuse_args_tensors(variant_name, args, epilogue_args)
                    return
        # No reusable args (e.g. artifact loaded from disk by a subprocess
        # precompile): build them once, then attach below for future calls.
        args = _create_gemm_arguments(
            variant_name,
            input_tensors,
            out,
            accumulator_type,
            epilogue=epilogue_args,
            **(variant_kwargs or {}),
        )
        kernel = artifact.operator_obj
    else:

        def disk_fallback(kernel):
            compiled_fn = disk_cache_get(
                disk_fn_cache,
                module_path,
                disk_config_key,
                cache_key,
                dev_idx,
            )
            if compiled_fn is not None:
                compiled_fn = _rewrap_efc_compiled_obj(
                    compiled_fn,
                    kernel,
                    epilogue_args=epilogue_args,
                )
            if compiled_fn is not None:
                return CompiledArtifact(
                    compiled_fn, kernel, _current_target_sm(dev_idx)
                )
            return None

        artifact, args, kernel, was_compiled = _compile_nvgemm(
            variant_name,
            input_tensors,
            out,
            accumulator_type,
            kernel_name=kernel_name,
            args_kwargs=variant_kwargs,
            epilogue_args=epilogue_args,
            epilogue_source=epilogue_source,
            fallback_fn=disk_fallback,
            cc=_current_target_sm(dev_idx).cc,
        )

        if was_compiled:
            disk_cache_set(
                disk_fn_cache,
                module_path,
                disk_config_key,
                cache_key,
                _unwrap_efc_compiled_obj(artifact.compiled_obj),
                dev_idx,
            )

        compiled_cache[mem_key] = artifact

    # Attach the built args for reuse on subsequent calls (skipped for
    # GROUPED_GEMM, where _update_reuse_args_tensors returns False). This runs
    # for both the disk-loaded rebuild branch and the fresh-compile branch.
    reuse_supported = _update_reuse_args_tensors(
        variant_name,
        args,
        input_tensors,
        out,
        epilogue_args,
        variant_kwargs,
    )
    if reuse_supported:
        # Publish and launch under the lock so a concurrent reuse-path caller
        # that picks up these args blocks until this launch has cleared them.
        reuse_lock = threading.Lock()
        with reuse_lock:
            artifact._nvgemm_reuse = (args, kernel, reuse_lock)
            try:
                kernel.run(
                    args,
                    artifact,
                    stream=stream,
                    workspace=workspace,
                    assume_supported_args=True,
                )
            finally:
                _clear_reuse_args_tensors(variant_name, args, epilogue_args)
    else:
        kernel.run(
            args,
            artifact,
            stream=stream,
            workspace=workspace,
            assume_supported_args=True,
        )


# Patch both the canonical definition (integration_utils.mma) for callers that
# reference it module-qualified, and each gemm module's namespace for callers
# that imported the bare name. _patch_max_active_clusters skips any without it.
_MAX_ACTIVE_CLUSTERS_MODULES = [
    "cutlass.operators.providers.cutedsl.integration_utils.mma",
    "cutlass.operators.providers.cutedsl.gemm.sm100_persistent",
    "cutlass.operators.providers.cutedsl.gemm.sm100_persistent_preferred_cluster",
    "cutlass.operators.providers.cutedsl.gemm.sm100_static_persistent_efc",
    "cutlass.operators.providers.cutedsl.gemm.sm100_dense_blockscaled_static_persistent",
    "cutlass.operators.providers.cutedsl.gemm.sm100_contiguous_offset_2d3d_dense_gemm",
    "cutlass.operators.providers.cutedsl.gemm.sm100_mixed_input",
    "cutlass.operators.providers.cutedsl.gemm.sm90_static_persistent",
]


def _nvgemm_precompile(
    precompile_shapes: dict,
    precompile_strides: dict,
    precompile_dtypes: dict,
    device_index: int = 0,
    device_capability: tuple | None = None,
    *,
    variant_name: str,
    kernel_name: str,
    accumulator_type,
    compiled_cache: dict,
    disk_fn_cache: dict,
    module_path: str,
    disk_config_key: tuple,
    input_param_names: list[str],
    variant_kwargs: dict | None = None,
    max_active_clusters: int | None = None,
):
    """Precompile an NVGEMM kernel in a subprocess for parallel compilation.

    Precompiles only the base (non-EFC) kernel. EFC kernels produce
    closure-wrapped artifacts that can't be serialized to disk cache.
    """
    import torch
    from torch._inductor.runtime.cutedsl_cache import disk_cache_set
    from torch._subclasses.fake_tensor import FakeTensorMode

    if max_active_clusters is None:
        return

    # cutlass.operators queries device occupancy while compiling. In async-compile
    # workers forked from a CUDA-initialized parent, skip precompile.
    if torch.cuda._is_in_bad_fork():
        return

    device = f"cuda:{device_index}"
    with FakeTensorMode():
        tensors = {}
        for name in [*input_param_names, "output"]:
            tensors[name] = torch.empty_strided(
                tuple(precompile_shapes[name]),
                tuple(precompile_strides[name]),
                device=device,
                dtype=getattr(torch, precompile_dtypes[name]),
            )

    input_tensors = tuple(tensors[n] for n in input_param_names)
    out = tensors["output"]

    patched = _patch_max_active_clusters(max_active_clusters)
    try:
        cache_key = _create_gemm_cache_key(input_tensors, out)
        mem_key = (cache_key, device_index)
        if mem_key not in compiled_cache:
            artifact, _, _, _ = _compile_nvgemm(
                variant_name,
                input_tensors,
                out,
                accumulator_type,
                kernel_name=kernel_name,
                args_kwargs=variant_kwargs,
            )
            disk_cache_set(
                disk_fn_cache,
                module_path,
                disk_config_key,
                cache_key,
                artifact.compiled_obj,
                device_index,
                device_capability=device_capability,
            )
            compiled_cache[mem_key] = artifact
    finally:
        _restore_max_active_clusters(patched)


# ── Kernel wrapper (returned by async_compile.nv_universal_gemm) ─────────────


class NVUniversalGemmKernelWrapper:
    """Wrapper to provide .run() interface for NVIDIA Universal GEMM kernels."""

    def __init__(self, kernel_fn, kernel_path: str | None = None):
        self.kernel_fn = kernel_fn
        self.kernel_path = kernel_path

    def run(self, *args, stream=None, **kwargs):
        """Execute the NVIDIA Universal GEMM kernel."""
        return self.kernel_fn(*args, stream=stream, **kwargs)


# ── Kernel codegen class ─────────────────────────────────────────────────────


def _build_bias_epilogue(bias_name: str, out_name: str) -> GemmEpiloguePlan:
    """Build the epilogue fields for an addmm bias-add (``accum + bias``).

    Uses the bias buffer name as the epilogue-fn parameter, matching
    CutlassEVTCodegen's convention. This deliberately avoids the name ``C``:
    cutlass.operators' LoadSrcImpl claims any epilogue param named ``C`` whose
    shape equals the output (ignoring stride), which shadows the row/column
    broadcast impls and silently mis-reads a broadcast (1D) bias.
    """
    return GemmEpiloguePlan(
        source=(
            f"def _epilogue_fn(accum, {bias_name}):\n"
            f"    D = accum + {bias_name}\n"
            f"    return D"
        ),
        is_cutedsl=True,
        reads=(bias_name,),
        renames={bias_name: bias_name, "D": out_name},
    )


def _compose_bias_into_epilogue(
    epilogue: GemmEpiloguePlan,
    bias_name: str,
) -> GemmEpiloguePlan:
    """Inject an addmm bias-add into a scheduler-fused epilogue.

    The EFC kernel's ``accum`` is the raw ``A @ B``; a fused pointwise epilogue
    written by CutlassEVTCodegen assumes its ``accum`` is the addmm output
    (``A @ B + bias``). We keep ``accum`` as the raw accumulator (required by
    the cutlass epilogue tracer) and rewrite the fused body to read a new
    ``biased = accum + bias`` value instead, adding ``bias`` as an aux input.
    """
    if epilogue.source is None:
        raise AssertionError("expected fused epilogue source")
    lines = epilogue.source.splitlines()
    def_line = lines[0]
    body = lines[1:]
    # Add bias as the 2nd parameter (after the required `accum`), unless the
    # fused epilogue already reads the same buffer -- then it is already a
    # parameter and re-adding it would emit `def fn(accum, b, b)` (SyntaxError).
    if bias_name not in epilogue.reads:
        def_line = def_line.replace("(accum", f"(accum, {bias_name}", 1)
    # Rewrite fused-body references to the accumulator to the biased value.
    rewritten = [re.sub(r"\baccum\b", "biased", bl) for bl in body]
    composed = "\n".join([def_line, f"    biased = accum + {bias_name}", *rewritten])
    reads = list(epilogue.reads)
    if bias_name not in reads:
        reads.append(bias_name)
    renames = dict(epilogue.renames)
    renames[bias_name] = bias_name
    return dataclasses.replace(
        epilogue, source=composed, reads=tuple(reads), renames=renames
    )


class NVUniversalGemmKernel(Kernel):
    """
    Kernel implementation for NVIDIA Universal GEMM.

    Generates a thin Python wrapper that delegates to runtime helpers
    (_nvgemm_run, _nvgemm_precompile) for the actual GEMM execution and
    compilation logic.
    """

    def __init__(
        self,
        kernel_name: str,
        input_nodes: list[Buffer],
        output_node: Buffer,
        kernel_metadata: dict[str, Any],
        accumulator_type: Any,
        variant: GemmVariant,
        workspace_size: int = 0,
        scale_type_a: Any | None = None,
        scale_type_b: Any | None = None,
        swizzle_type_a: Any | None = None,
        swizzle_type_b: Any | None = None,
        epilogue: GemmEpiloguePlan | None = None,
        local_reduce: GemmReductionPlan | None = None,
        swap_ab: bool = False,
        bias_node: Buffer | None = None,
    ) -> None:
        super().__init__()
        self.kernel_name = kernel_name
        self.input_nodes = input_nodes
        self.output_node = output_node
        self.kernel_metadata = kernel_metadata
        self.accumulator_type = accumulator_type
        self.workspace_size = workspace_size
        self.variant = variant
        self.scale_type_a = scale_type_a
        self.scale_type_b = scale_type_b
        self.swizzle_type_a = swizzle_type_a
        self.swizzle_type_b = swizzle_type_b
        self.epilogue = epilogue or GemmEpiloguePlan()
        self.local_reduce = local_reduce
        self.swap_ab = swap_ab

        # An addmm bias baked into the choice becomes a bias-add epilogue. With
        # no scheduler-fused epilogue it's a standalone bias-add; when the
        # scheduler also fuses pointwise ops, compose the bias-add into them so
        # the fused epilogue sees A@B + bias (matches Triton's epilogue fusion).
        if bias_node is not None:
            if not self.epilogue.source:
                self.epilogue = _build_bias_epilogue(
                    bias_node.get_name(), output_node.get_name()
                )
            else:
                self.epilogue = _compose_bias_into_epilogue(
                    self.epilogue, bias_node.get_name()
                )

        self._template_input_args: list[tuple[str, Buffer]] = []

        for i, input_node in enumerate(input_nodes):
            param_name = f"in_ptr{i}"
            self._template_input_args.append((param_name, input_node))

    def render(self) -> str:
        """Render the Python source for the NVGEMM kernel wrapper."""
        kernel_name_str = self.kernel_metadata["kernel_name"]
        acc_dtype_str = CuteDSLOpOverrides.TORCH_TO_CUTE_DTYPE.get(
            self.accumulator_type, "cutlass.Float32"
        )

        input_tensor_names = [f"in_ptr{i}" for i, _ in enumerate(self.input_nodes)]
        output_buffers = self.ordered_output_buffers()
        input_params = list(input_tensor_names)
        input_params.extend(f"out_ptr{i}" for i in range(len(output_buffers)))
        input_params.extend(self.epilogue.reads)
        if self.workspace_size > 0:
            input_params.append("workspace")
        input_params.append("stream=None")
        params_str = ", ".join(input_params)

        if len(input_tensor_names) == 1:
            input_tensors_expr = f"({input_tensor_names[0]},)"
        else:
            input_tensors_expr = f"({', '.join(input_tensor_names)})"

        workspace_arg = "workspace" if self.workspace_size > 0 else "None"
        has_epilogue = bool(self.epilogue.source) or self.local_reduce is not None

        # Build variant_kwargs dict expression for SCALED_GEMM
        variant_kwargs_expr = "None"
        variant_extra_imports = ""
        if self.variant.name == "SCALED_GEMM":
            sma, swza, smb, swzb = _get_scaled_gemm_modes(
                self.scale_type_a,
                self.swizzle_type_a,
                self.scale_type_b,
                self.swizzle_type_b,
            )
            variant_extra_imports = (
                "from cutlass.operators import ScaleMode, ScaleSwizzleMode"
            )
            variant_kwargs_expr = (
                "{"
                f'"scale_mode_a": ScaleMode.{sma.name}, '
                f'"swizzle_mode_a": ScaleSwizzleMode.{swza.name}, '
                f'"scale_mode_b": ScaleMode.{smb.name}, '
                f'"swizzle_mode_b": ScaleSwizzleMode.{swzb.name}'
                "}"
            )

        code = IndentedBuffer()

        # -- Module-level imports and constants --
        code.writeline("import cutlass")
        code.writeline(
            "from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_kernel import ("
        )
        with code.indent():
            code.writeline("_make_disk_config_key,")
            code.writeline("_nvgemm_run,")
            code.writeline("_nvgemm_precompile,")
        code.writeline(")")
        code.writeline("from torch._inductor.utils import _ensure_fp4_dtype_registered")
        code.writeline("_ensure_fp4_dtype_registered()")
        if variant_extra_imports:
            code.writeline(variant_extra_imports)
        if self.local_reduce is not None:
            code.writeline(
                "from torch._inductor.kernel.gemm_epilogue import GemmReductionArguments"
            )
            reduction_args = GemmReductionArguments.from_plan(self.local_reduce)
            for field, value in reduction_args.specialization_items():
                if field.endswith("_fn") and value is not None:
                    code.writeline(
                        f"{_local_reduce_source_constant(field)} = {value!r}"
                    )
        if has_epilogue:
            if self.epilogue.is_cutedsl:
                code.writeline(
                    "from torch._inductor.codegen.nv_universal_gemm."
                    "nv_universal_gemm_kernel import CuTeDSLEpilogueArguments"
                )
            else:
                code.writeline(
                    "from cutlass.operators.arguments import EpilogueArguments"
                )
        code.writeline("")

        # -- Epilogue function definition (must be module-level for cutlass.operators) --
        epilogue_fn_code = self.epilogue.source
        if has_epilogue and epilogue_fn_code is not None:
            epilogue_source_hash = hashlib.sha256(epilogue_fn_code.encode()).hexdigest()
            code.writeline(f'_EPILOGUE_FN_SOURCE = "{epilogue_source_hash}"')
            # Pass the epilogue to cutlass.operators as a SOURCE STRING, not a
            # callable: EpilogueArguments traces a string directly, whereas a
            # callable triggers inspect.getsource() at every kernel launch --
            # which fails ("could not get source code") for generated modules
            # loaded from the subprocess PyCodeCache.
            code.writeline(f"_EPILOGUE_FN_SRC = {epilogue_fn_code!r}")
            code.writeline("")

        # -- Module-level state --
        code.writeline(f"_KERNEL_NAME = {kernel_name_str!r}")
        disk_key_args = ", ".join(
            repr(str(v))
            for v in (
                kernel_name_str,
                self.variant.name,
                self.accumulator_type,
                self.scale_type_a,
                self.scale_type_b,
                self.swizzle_type_a,
                self.swizzle_type_b,
            )
        )
        code.writeline(
            f"_DISK_CACHE_CONFIG_KEY = _make_disk_config_key({disk_key_args})"
        )
        code.writeline("_compiled_cache = {}")
        code.writeline("_disk_fn_cache = {}")
        code.writeline(f"_VARIANT_KWARGS = {variant_kwargs_expr}")
        code.writeline(f"_ACC_TYPE = {acc_dtype_str}")
        input_param_names_repr = repr(input_tensor_names)
        code.writeline(f"_INPUT_PARAM_NAMES = {input_param_names_repr}")
        code.writeline("")

        # -- Main function --
        code.writeline(f"def {self.kernel_name}_main({params_str}):")
        with code.indent():
            feed_main = self.local_reduce is not None and self.local_reduce.feeds_main
            epilogue_d = self.epilogue.renames.get("D")
            view_gemm_out = feed_main or (
                self.epilogue.source
                and output_buffers
                and epilogue_d == output_buffers[0]
            )
            gemm_out = "out_ptr0"
            if view_gemm_out:
                a_name, b_name = input_tensor_names[:2]
                code.writeline(
                    f"gemm_out = out_ptr0.view(*{a_name}.shape[:-2], "
                    f"{a_name}.shape[-2], {b_name}.shape[-1])"
                )
                gemm_out = "gemm_out"
            # Build epilogue args if needed (user-specific variable names)
            epi_args_expr = "None"
            epi_source_expr = '""'
            aux_tensors: list[str] = []
            if self.epilogue.source:
                epilogue_kwargs = self._render_epilogue_kwargs()
                if view_gemm_out:
                    epilogue_kwargs = epilogue_kwargs.replace(
                        "D=out_ptr0", "D=gemm_out"
                    )
                epi_kwargs_str = "epilogue_fn=_EPILOGUE_FN_SRC"
                if epilogue_kwargs:
                    epi_kwargs_str += f", {epilogue_kwargs}"
                epilogue_args_type = (
                    "CuTeDSLEpilogueArguments"
                    if self.epilogue.is_cutedsl
                    else "EpilogueArguments"
                )
                code.writeline(f"epi_args = {epilogue_args_type}({epi_kwargs_str})")
                epi_args_expr = "epi_args"
                epi_source_expr = "_EPILOGUE_FN_SOURCE"
                aux_tensors.extend(self.epilogue.reads)

            run_variant_kwargs = "_VARIANT_KWARGS"
            if self.local_reduce is not None:
                reduction = self.local_reduce
                reduce_name = reduction.reduction_output
                reduce_ptr = (
                    "None"
                    if reduce_name is None
                    else f"out_ptr{output_buffers.index(reduce_name)}"
                )
                if reduce_name is not None:
                    squeeze_dim = -1 if reduction.axis == 1 else -2
                    reduce_ptr = f"{reduce_ptr}.squeeze({squeeze_dim})"

                def feed_output_ptr(output_name: str | None) -> str:
                    if output_name is None:
                        return "None"
                    a_name, b_name = input_tensor_names[:2]
                    output_ptr = f"out_ptr{output_buffers.index(output_name)}"
                    return (
                        f"{output_ptr}.view(*{a_name}.shape[:-2], "
                        f"{a_name}.shape[-2], {b_name}.shape[-1])"
                    )

                feed_ptr = feed_output_ptr(reduction.feed_output)
                secondary_feed_ptr = feed_output_ptr(reduction.secondary_feed_output)
                rendered_reduction = GemmReductionArguments.from_plan(
                    reduction,
                    output=reduce_ptr,
                    feed_output=feed_ptr,
                    secondary_feed_output=secondary_feed_ptr,
                )
                tensor_args = {
                    field: cast(str, value)
                    for field, value in rendered_reduction.tensor_items()
                }
                reduction_args = {
                    field: (
                        _local_reduce_source_constant(field)
                        if field.endswith("_fn") and value is not None
                        else repr(value)
                    )
                    for field, value in rendered_reduction.specialization_items()
                    if value is not None
                }
                reduction_args = tensor_args | reduction_args
                reduction_args_expr = ", ".join(
                    f"{field}={value}" for field, value in reduction_args.items()
                )
                run_variant_kwargs = (
                    "(_VARIANT_KWARGS or {}) | {"
                    f"'local_reduce': GemmReductionArguments({reduction_args_expr})"
                    "}"
                )
                aux_tensors.extend(
                    value for value in tensor_args.values() if value != "None"
                )

            aux_tensors_expr = f"({', '.join(aux_tensors)},)" if aux_tensors else "()"

            code.writeline("_nvgemm_run(")
            with code.indent():
                code.writeline(f'"{self.variant.name}", _KERNEL_NAME,')
                code.writeline(f"{input_tensors_expr}, {gemm_out}, _ACC_TYPE,")
                code.writeline(
                    "_compiled_cache, _disk_fn_cache, __file__, _DISK_CACHE_CONFIG_KEY,"
                )
                code.writeline(f"stream=stream, workspace={workspace_arg},")
                code.writeline(f"variant_kwargs={run_variant_kwargs},")
                code.writeline(f"epilogue_args={epi_args_expr},")
                code.writeline(f"epilogue_source={epi_source_expr},")
                code.writeline(f"has_epilogue={has_epilogue},")
                code.writeline(f"aux_tensors={aux_tensors_expr},")
                if self.swap_ab:
                    code.writeline("swap_ab=True,")
            code.writeline(")")

        # -- Precompile hook --
        code.writeline("")
        code.writeline(
            f"def {self.kernel_name}_precompile("
            "precompile_shapes, precompile_strides, precompile_dtypes, "
            "device_index=0, device_capability=None, **kwargs):"
        )
        with code.indent():
            code.writeline("_nvgemm_precompile(")
            with code.indent():
                code.writeline(
                    "precompile_shapes, precompile_strides, precompile_dtypes,"
                )
                code.writeline("device_index, device_capability,")
                code.writeline(f'variant_name="{self.variant.name}",')
                code.writeline("kernel_name=_KERNEL_NAME,")
                code.writeline("accumulator_type=_ACC_TYPE,")
                code.writeline(
                    "compiled_cache=_compiled_cache, disk_fn_cache=_disk_fn_cache,"
                )
                code.writeline(
                    "module_path=__file__, disk_config_key=_DISK_CACHE_CONFIG_KEY,"
                )
                code.writeline("input_param_names=_INPUT_PARAM_NAMES,")
                code.writeline("variant_kwargs=_VARIANT_KWARGS,")
                code.writeline("max_active_clusters=kwargs.get('max_active_clusters'),")
            code.writeline(")")

        return code.getvalue()

    def ordered_output_buffers(self) -> list[str]:
        """Graph-output buffer names in out_ptr order.

        out_ptr0 is the primary GEMM output (the epilogue's `D` store, passed as
        the kernel's `out`). Additional stores -- a multi-store epilogue where the
        GEMM output feeds more than one graph output -- follow in the epilogue
        plan's write order as out_ptr1, out_ptr2, ...
        """
        if not self.epilogue.writes:
            primary_output = (
                self.local_reduce.primary_output
                if self.local_reduce is not None
                else self.output_node.get_name()
            )
            ordered = [primary_output]
            if self.local_reduce is not None:
                ordered.extend(self.local_reduce.auxiliary_outputs)
            return ordered
        d_buf = self.epilogue.renames.get("D")
        ordered: list[str] = []
        if d_buf is not None:
            ordered.append(d_buf)
        for w in self.epilogue.writes:
            if w != d_buf and w not in ordered:
                ordered.append(w)
        if self.local_reduce is not None:
            ordered.extend(
                output
                for output in self.local_reduce.auxiliary_outputs
                if output not in ordered
            )
        return ordered

    def _render_epilogue_kwargs(self) -> str:
        """Render kwargs for the EpilogueArguments constructor.

        Each epilogue variable maps to either an output pointer -- the `D` store
        and any additional multi-store outputs become out_ptr0, out_ptr1, ... in
        ordered_output_buffers order -- or, for a read, the aux input buffer.
        `accum` is the kernel-supplied accumulator and is not a kwarg.
        """
        out_ptr_of = {
            buf: f"out_ptr{i}" for i, buf in enumerate(self.ordered_output_buffers())
        }
        kwargs_parts = []
        for var_name, buffer_name in self.epilogue.renames.items():
            if var_name == GEMM_ACCUMULATOR_ARG_NAME:
                continue
            if buffer_name in out_ptr_of:
                kwargs_parts.append(f"{var_name}={out_ptr_of[buffer_name]}")
            else:
                kwargs_parts.append(f"{var_name}={buffer_name}")
        return ", ".join(kwargs_parts)

    def _get_reinterpret_view(self, node) -> ReinterpretView | None:
        """Extract or convert to ReinterpretView from a node, handling all views."""
        while isinstance(node, MutableBox):
            node = node.data
        if isinstance(node, BaseView):
            return ExternKernel.convert_to_reinterpret_view(node)
        return None

    def call_kernel(self, name: str, node=None):
        """
        Generate the kernel call in the wrapper code.

        Similar to CuteDSLTemplateKernel.call_kernel but simplified for NVIDIA Universal GEMM.
        Includes epilogue input tensors when epilogue fusion is enabled.
        """
        wrapper = V.graph.wrapper_code

        if self.local_reduce is not None:
            primary_name = self.local_reduce.primary_output
            V.graph.removed_buffers.discard(primary_name)
            primary_output = V.graph.get_buffer(primary_name)
            wrapper.codegen_allocation(cast(Buffer, primary_output or self.output_node))

        call_args: list[str] = []
        arg_types: list[Any] = []
        raw_args: list[Buffer | ReinterpretView | None] = []
        raw_keys: list[str | None] = []

        for param_name, input_node in self._template_input_args:
            reinterpret_view = self._get_reinterpret_view(input_node)
            if reinterpret_view is not None:
                call_args.append(reinterpret_view.codegen_reference())
                # Pass the ReinterpretView as raw_arg so autotune_at_compile_time
                # can use it to generate example tensors
                raw_args.append(reinterpret_view)
            else:
                call_args.append(input_node.get_name())
                raw_args.append(input_node)
            arg_types.append(V.graph.get_dtype(input_node.get_name()))
            raw_keys.append(param_name)

        # The kernel writes the epilogue's output store(s), not the GEMM buffer
        # (which is removed via removed_buffers aliasing). out_ptr0 is the primary
        # (`D`) output; a multi-store epilogue adds out_ptr1, ... in order.
        for i, output_name in enumerate(self.ordered_output_buffers()):
            call_args.append(output_name)
            arg_types.append(V.graph.get_dtype(output_name))
            raw_args.append(None)  # Output buffer is findable by name
            raw_keys.append(f"out_ptr{i}")

        for read_name in self.epilogue.reads:
            call_args.append(read_name)
            arg_types.append(V.graph.get_dtype(read_name))
            buf = V.graph.get_buffer(read_name)
            if buf is None:
                # Epilogue may read model parameters/inputs (e.g. bias) that are
                # graph inputs rather than computed buffers.
                buf = V.graph.graph_inputs.get(read_name)
            # pyrefly: ignore [bad-argument-type]
            raw_args.append(buf)
            raw_keys.append(read_name)

        # Allocate workspace if needed
        ws: WorkspaceArg | None = None
        if self.workspace_size > 0:
            ws = WorkspaceArg(
                count=self.workspace_size,
                device=V.graph.get_current_device_or_throw(),
                zero_mode=WorkspaceZeroMode.UNINITIALIZED,
                outer_name=WorkspaceArg.unique_name(),
            )
            wrapper.generate_workspace_allocation(ws)
            call_args.append(ws.outer_name)
            arg_types.append(ws.dtype)
            raw_args.append(None)
            raw_keys.append(None)

        wrapper.generate_kernel_call(
            name,
            call_args,
            triton=True,
            arg_types=arg_types,
            raw_args=raw_args,
            raw_keys=raw_keys,
        )

        # Deallocate workspace after kernel call
        if ws is not None:
            wrapper.generate_workspace_deallocation(ws)
