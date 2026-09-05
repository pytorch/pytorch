# mypy: allow-untyped-defs
from __future__ import annotations

import contextlib
import dataclasses
import inspect
import math
import os
from typing import Any, TYPE_CHECKING

import torch
from torch._inductor.kernel.flex_gemm.constraints import (
    FlexGemmGroupedMainOutputTransform,
    FlexGemmLocalReduceGeometry,
    INDEXED_OUTPUT_INDICES_ARG_NAME,
    INDEXED_OUTPUT_STORE_ARG_NAME,
    LOCAL_REDUCE_FEED_MAIN_ARG_NAME,
    LOCAL_REDUCE_FRAGMENT_WIDTH,
    LOCAL_REDUCE_RUNTIME_OUT_ERROR,
    LOCAL_REDUCE_STORE_ARG_NAME,
    validate_local_reduce_feed_main_capability,
)
from torch._inductor.kernel.flex_gemm.output_layout import FlexGemmOutputLayout
from torch._inductor.runtime.cache_dir_utils import cache_dir
from torch._prims_common import is_expandable_to


if TYPE_CHECKING:
    from collections.abc import Callable


def inductor_quack_cache_dir() -> str:
    """Return the Inductor-owned QuACK cache root for generated FlexGEMM."""
    return os.path.join(cache_dir(), "quack")


# NOTE [Byte-backed epilogue tensor storage]
# PyTorch bool tensors are byte-addressed while CuTeDSL models cutlass.Boolean as
# a 1-bit logical type; Float4E2M1FN similarly exposes two values per byte. Pass
# both through QuACK as their physical uint8 carrier.
def quack_epilogue_arg(arg: torch.Tensor) -> torch.Tensor:
    """Adapt logical epilogue tensors to QuACK's physical tensor ABI."""
    if arg.dtype in (torch.bool, torch.float4_e2m1fn_x2):
        return arg.view(torch.uint8)
    return arg


def quack_blockscaled_scale_view(
    scale: torch.Tensor,
    mn: int,
    storage_k: int,
    format_name: str,
) -> torch.Tensor:
    """View a public flat SWIZZLE_32_4_4 scale as QuACK's blocked tensor."""
    from torch._vendor.quack.blockscaled import operand as blockscaled

    format = blockscaled.BlockScaledFormat.from_name(format_name)
    logical_k = format.logical_k(storage_k)
    shape = (
        (mn + 127) // 128,
        (logical_k + 4 * format.sf_vec_size - 1) // (4 * format.sf_vec_size),
        32,
        4,
        4,
    )
    expected_numel = math.prod(shape)
    if scale.numel() != expected_numel:
        raise RuntimeError(
            f"{format_name} scale has {scale.numel()} elements, expected "
            f"{expected_numel} for shape {shape}"
        )
    return scale.view(shape)


def normalize_c(
    C: torch.Tensor | None, expected_shape: tuple[int, ...], beta: float
) -> torch.Tensor | None:
    """Return the effective C tensor that QuACK should read for alpha/beta GEMMs."""
    if C is None:
        return None
    if not is_expandable_to(tuple(C.shape), expected_shape):
        raise RuntimeError(
            f"C shape must broadcast to {expected_shape}, got {tuple(C.shape)}"
        )
    if beta == 0:
        return None
    broadcast_C = torch.broadcast_to(C, expected_shape)
    if broadcast_C.ndim not in (2, 3):
        raise NotImplementedError("FlexGEMM currently supports only 2-D or 3-D C")
    if not broadcast_C.is_cuda:
        raise RuntimeError("FlexGEMM requires CUDA C")
    if broadcast_C.stride(-1) != 1 and broadcast_C.stride(-2) != 1:
        raise NotImplementedError("FlexGEMM requires C to be row- or column-major")
    return broadcast_C


@dataclasses.dataclass(frozen=True)
class FlexGemmEpiModIndexedOutputPlan:
    """Runtime buffers for one exact row-indexed auxiliary output."""

    out: torch.Tensor
    indices: torch.Tensor

    def __post_init__(self) -> None:
        if self.out.ndim != 1 or self.out.stride(0) != 1:
            raise RuntimeError("indexed output must be a contiguous vector")
        if self.indices.ndim != 1 or self.indices.stride(0) != 1:
            raise RuntimeError("indexed output indices must be a contiguous vector")
        if self.out.shape != self.indices.shape:
            raise RuntimeError("indexed output and indices must have the same shape")
        if self.indices.dtype not in (torch.int32, torch.int64):
            raise NotImplementedError("indexed output indices must be int32 or int64")

    @property
    def cache_key(self) -> tuple[torch.dtype, torch.dtype]:
        """Return physical output and index dtypes affecting generated code."""
        return quack_epilogue_arg(self.out).dtype, self.indices.dtype


@dataclasses.dataclass(frozen=True)
class FlexGemmEpiModLocalReducePlan:
    """QuACK EpiOp configuration for one analyzed grouped local reduction."""

    geometry: FlexGemmLocalReduceGeometry
    out: torch.Tensor | None = None
    feeds_main: bool = False
    combine: Callable[..., Any] | str | None = None
    finalize: Callable[..., Any] | str | None = None
    reduce_planes: int = 1
    fragment_reduced: bool = False
    store_finalize: Callable[..., Any] | str | None = None
    prepass: Callable[..., Any] | None = None
    prepass_combine: str | None = None
    prepass_finalize: Callable[..., Any] | str | None = None
    output_layout: FlexGemmOutputLayout | None = None

    def __post_init__(self) -> None:
        if self.out is None and not self.feeds_main:
            raise RuntimeError(LOCAL_REDUCE_RUNTIME_OUT_ERROR)
        if self.combine is None:
            raise RuntimeError("FlexGEMM EpiMod local reductions require a combine")
        if self.output_layout is not None and not isinstance(
            self.output_layout, FlexGemmOutputLayout
        ):
            raise TypeError("local-reduce output_layout must be a FlexGemmOutputLayout")
        if (self.prepass is None) != (self.prepass_combine is None):
            raise RuntimeError(
                "FlexGEMM EpiMod prepasses require both a callable and combine"
            )
        if self.prepass_finalize is not None and self.prepass is None:
            raise RuntimeError("FlexGEMM EpiMod prepass finalizers require a prepass")
        if self.feeds_main and not (
            self.axis == 1 and self.group <= LOCAL_REDUCE_FRAGMENT_WIDTH
        ):
            validate_local_reduce_feed_main_capability(self.axis, self.group)

    @property
    def group(self) -> int:
        return self.geometry.group

    @property
    def axis(self) -> int:
        return self.geometry.axis

    @property
    def cache_key(self) -> tuple[Any, ...]:
        return (
            self.geometry,
            self.feeds_main,
            self.combine,
            self.finalize,
            self.reduce_planes,
            self.fragment_reduced,
            self.store_finalize,
            self.prepass,
            self.prepass_combine,
            self.prepass_finalize,
            self.output_layout,
            self.out is not None,
        )


_EPIMOD_CACHE: dict[tuple[Any, ...], Any] = {}


def flex_gemm_epimod(
    epilogue_fn: Any,
    epilogue_args: tuple[torch.Tensor, ...],
    epilogue_arg_kinds: tuple[str, ...],
    aux_output_count: int,
    indexed_output: FlexGemmEpiModIndexedOutputPlan | None,
    local_reduce: FlexGemmEpiModLocalReducePlan | None,
    fragmentwise: bool,
    main_transform: FlexGemmGroupedMainOutputTransform | None,
):
    """Build and cache a QuACK EpiMod from generated FlexGEMM metadata."""
    epilogue_arg_dtypes = tuple(arg.dtype for arg in epilogue_args)
    key = (
        epilogue_fn,
        epilogue_arg_kinds,
        epilogue_arg_dtypes,
        aux_output_count,
        None if indexed_output is None else indexed_output.cache_key,
        None if local_reduce is None else local_reduce.cache_key,
        fragmentwise,
        main_transform,
    )
    epimod = _EPIMOD_CACHE.get(key)
    if epimod is not None:
        return epimod

    from torch._vendor.quack import cute_dsl_utils, epi_math
    from torch._vendor.quack.epilogue import frontend as epilogue_module, ops as epi_ops

    # Generated callbacks reference epi_math without importing QuACK into the
    # generated source. Inject it only into the original function's globals;
    # decorated wrappers may belong to third-party modules.
    inspect.unwrap(epilogue_fn).__globals__["epi_math"] = epi_math
    op_types = {
        "row": epi_ops.RowVecLoad,
        "col": epi_ops.ColVecLoad,
        "tile": epi_ops.TileLoad,
    }
    ops = {}
    for index, (arg, kind) in enumerate(
        zip(epilogue_args, epilogue_arg_kinds, strict=True)
    ):
        name = f"operand{index}"
        dtype = cute_dsl_utils.torch2cute_dtype_map[arg.dtype]
        ops[name] = (
            epi_ops.Scalar(name, dtype=dtype)
            if kind == "scalar"
            else op_types[kind](name, dtype=dtype)
        )
    if main_transform is not None:
        outputs = (
            epi_ops.GroupedMainStore(
                "main",
                main_transform.group,
                paired=not fragmentwise and main_transform.group == 2,
            ),
        )
    else:
        outputs = tuple(f"output{index}" for index in range(aux_output_count))
    sinks = {}
    extra_ops = ()
    if indexed_output is not None:
        physical_out = quack_epilogue_arg(indexed_output.out)
        index_op = epi_ops.ColVecLoad(
            INDEXED_OUTPUT_INDICES_ARG_NAME,
            dtype=cute_dsl_utils.torch2cute_dtype_map[indexed_output.indices.dtype],
        )
        sinks[INDEXED_OUTPUT_STORE_ARG_NAME] = epi_ops.ColVecSelect(
            INDEXED_OUTPUT_STORE_ARG_NAME,
            idx_op=index_op,
            output_dtype=cute_dsl_utils.torch2cute_dtype_map[physical_out.dtype],
        )
        extra_ops = (index_op,)
    prepass = None
    prepass_outs = ()
    if local_reduce is not None:
        from torch._vendor.quack import grouped_reduce

        finalize = local_reduce.finalize
        store_finalize = local_reduce.store_finalize or finalize
        prepass_finalize = local_reduce.prepass_finalize
        output_layout = (
            None
            if local_reduce.output_layout is None
            else local_reduce.output_layout.quack_layout(grouped_reduce)
        )
        if local_reduce.prepass is not None:
            prepass = local_reduce.prepass
            ops[LOCAL_REDUCE_FEED_MAIN_ARG_NAME] = (
                grouped_reduce.GroupedLocalReducePrepass(
                    LOCAL_REDUCE_FEED_MAIN_ARG_NAME,
                    axis=local_reduce.axis,
                    group=local_reduce.group,
                    combine=local_reduce.prepass_combine,
                    finalize=prepass_finalize,
                )
            )
            prepass_outs = (LOCAL_REDUCE_FEED_MAIN_ARG_NAME,)
            if local_reduce.out is not None:
                if (
                    callable(store_finalize)
                    and len(inspect.signature(store_finalize).parameters) == 2
                ):
                    if output_layout is not None:
                        raise RuntimeError(
                            "local-reduce output layouts do not support binary finalizers"
                        )
                    sink = grouped_reduce.GroupedLocalReduceWithFinalizeArg(
                        LOCAL_REDUCE_STORE_ARG_NAME,
                        axis=local_reduce.axis,
                        group=local_reduce.group,
                        combine=local_reduce.combine,
                        finalize=store_finalize,
                    )
                else:
                    sink = grouped_reduce.GroupedLocalReduce(
                        LOCAL_REDUCE_STORE_ARG_NAME,
                        axis=local_reduce.axis,
                        group=local_reduce.group,
                        combine=local_reduce.combine,
                        finalize=store_finalize,
                        output_layout=output_layout,
                        reduce_planes=local_reduce.reduce_planes,
                        fragment_reduced=local_reduce.fragment_reduced,
                    )
                sinks[LOCAL_REDUCE_STORE_ARG_NAME] = sink
        else:
            if local_reduce.feeds_main:
                if output_layout is not None:
                    raise RuntimeError(
                        "feed-main local reductions do not support output layouts"
                    )
                reduce_op = grouped_reduce.GroupedLocalReduceFeed(
                    LOCAL_REDUCE_FEED_MAIN_ARG_NAME,
                    axis=local_reduce.axis,
                    group=local_reduce.group,
                    combine=local_reduce.combine,
                    finalize=finalize,
                )
            else:
                reduce_op = grouped_reduce.GroupedLocalReduce(
                    LOCAL_REDUCE_FEED_MAIN_ARG_NAME,
                    axis=local_reduce.axis,
                    group=local_reduce.group,
                    combine=local_reduce.combine,
                    finalize=finalize,
                    output_layout=output_layout,
                    reduce_planes=local_reduce.reduce_planes,
                    fragment_reduced=local_reduce.fragment_reduced,
                )
            if local_reduce.feeds_main:
                ops[LOCAL_REDUCE_FEED_MAIN_ARG_NAME] = reduce_op
            else:
                sinks[LOCAL_REDUCE_FEED_MAIN_ARG_NAME] = reduce_op
    if fragmentwise:
        epimod = epilogue_module.fragment_epilogue(
            outputs=outputs,
            ops=ops,
            outs=sinks,
            extra_ops=extra_ops,
            prepass=prepass,
            prepass_outs=prepass_outs,
        )(epilogue_fn)
    else:
        epimod = epilogue_module.gemm_epilogue(
            outputs=outputs,
            ops=ops,
            outs=sinks,
            extra_ops=extra_ops,
            mode=(
                "acc_pair"
                if main_transform is not None and main_transform.group == 2
                else None
            ),
            prepass=prepass,
            prepass_outs=prepass_outs,
            vectorize=(
                False
                if main_transform is not None
                and main_transform.group == 2
                and local_reduce is not None
                else None
            ),
        )(epilogue_fn)
    _EPIMOD_CACHE[key] = epimod
    return epimod


def gemm_epimod(
    a: torch.Tensor,
    b: torch.Tensor,
    epilogue_fn,
    *,
    C: torch.Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    SFA: torch.Tensor | None = None,
    SFB: torch.Tensor | None = None,
    bs_format_a: str | None = None,
    bs_format_b: str | None = None,
    out: torch.Tensor,
    aux_outs: tuple[torch.Tensor, ...] = (),
    epilogue_args: tuple[torch.Tensor, ...] = (),
    epilogue_arg_kinds: tuple[str, ...] = (),
    indexed_output: FlexGemmEpiModIndexedOutputPlan | None = None,
    local_reduce: FlexGemmEpiModLocalReducePlan | None = None,
    fragmentwise: bool = False,
    main_transform: FlexGemmGroupedMainOutputTransform | None = None,
    tuned: bool = False,
    config_constraints: tuple[tuple[str, Any], ...] = (),
    stream: int | None = None,
) -> torch.Tensor:
    """Run a dense or block-scaled FlexGEMM call through the vendored QuACK EpiMod."""
    if (SFA is None) != (SFB is None):
        raise RuntimeError("FlexGEMM block-scaled operands require both SFA and SFB")
    if SFA is not None:
        if SFB is None:
            raise AssertionError("paired FlexGEMM scale validation failed")
        if a.ndim != 2 or b.ndim != 2 or bs_format_a is None or bs_format_b is None:
            raise NotImplementedError(
                "FlexGEMM block-scaled main loops currently require 2-D operands and format names"
            )
        SFA = quack_blockscaled_scale_view(SFA, a.shape[-2], a.shape[-1], bs_format_a)
        SFB = quack_blockscaled_scale_view(SFB, b.shape[-1], b.shape[-2], bs_format_b)
    if main_transform is not None and main_transform.chunked and b.stride(-1) == 1:
        raise NotImplementedError(
            "chunked grouped main output requires column-major B storage"
        )
    quack_epilogue_args = tuple(quack_epilogue_arg(arg) for arg in epilogue_args)
    epimod = flex_gemm_epimod(
        epilogue_fn,
        quack_epilogue_args,
        epilogue_arg_kinds,
        len(aux_outs),
        indexed_output,
        local_reduce,
        fragmentwise,
        main_transform,
    )
    effective_C = normalize_c(C, tuple(out.shape), beta)
    operands: dict[str, Any] = {}
    if "alpha" in epimod.operand_names:
        operands["alpha"] = alpha
    if "beta" in epimod.operand_names:
        operands["beta"] = beta
    for index, (arg, kind) in enumerate(
        zip(quack_epilogue_args, epilogue_arg_kinds, strict=True)
    ):
        operands[f"operand{index}"] = (
            arg.squeeze(-1).unsqueeze(0) if kind == "col" else arg
        )
    if indexed_output is not None:
        operands[INDEXED_OUTPUT_INDICES_ARG_NAME] = indexed_output.indices
        operands[INDEXED_OUTPUT_STORE_ARG_NAME] = quack_epilogue_arg(indexed_output.out)
    initialize_local_reduce_out = None
    if local_reduce is not None:
        from torch._vendor.quack import grouped_reduce

        local_reduce_out = local_reduce.out
        if local_reduce_out is not None:
            if local_reduce.output_layout is None:
                grouped_reduce.validate_grouped_reduce_out(
                    LOCAL_REDUCE_FEED_MAIN_ARG_NAME,
                    local_reduce_out,
                    a.shape[-2],
                    b.shape[-1],
                    local_reduce.group,
                    local_reduce.axis,
                )
            else:
                grouped_dim = a.shape[-2] if local_reduce.axis == 0 else b.shape[-1]
                if grouped_dim % local_reduce.group:
                    raise ValueError(
                        f"group {local_reduce.group} must divide the grouped dim "
                        f"{grouped_dim} (axis={local_reduce.axis})"
                    )
                rows, cols = (
                    (a.shape[-2], b.shape[-1] // local_reduce.group)
                    if local_reduce.axis == 1
                    else (a.shape[-2] // local_reduce.group, b.shape[-1])
                )
                if local_reduce_out.numel() != rows * cols:
                    initialize_local_reduce_out = local_reduce_out
                local_reduce_out = local_reduce.output_layout.runtime_view(
                    local_reduce_out, 1, rows, cols
                )
        if local_reduce.prepass is not None:
            operands[LOCAL_REDUCE_FEED_MAIN_ARG_NAME] = None
            if local_reduce.out is not None:
                operands[LOCAL_REDUCE_STORE_ARG_NAME] = local_reduce_out
        else:
            operands[LOCAL_REDUCE_FEED_MAIN_ARG_NAME] = local_reduce_out

    from torch._vendor.quack.cache import cache_dir_override

    output_buffers = (
        {"main": quack_epilogue_arg(out)}
        if main_transform is not None
        else {
            "D": quack_epilogue_arg(out),
            **dict(
                zip(
                    epimod.outputs,
                    (quack_epilogue_arg(aux_out) for aux_out in aux_outs),
                    strict=True,
                )
            ),
        }
    )
    stream_context = (
        torch.cuda.stream(torch.cuda.ExternalStream(stream, device=a.device))
        if stream is not None
        else contextlib.nullcontext()
    )
    with cache_dir_override(inductor_quack_cache_dir()), stream_context:
        # Layout callbacks predicate logical stores but do not own padded bytes.
        if initialize_local_reduce_out is not None:
            initialize_local_reduce_out.zero_()
        blockscaled_kwargs = (
            {}
            if SFA is None
            else {
                "SFA": SFA,
                "SFB": SFB,
                "bs_format_a": bs_format_a,
                "bs_format_b": bs_format_b,
            }
        )
        result = epimod(
            a,
            b,
            C=effective_C,
            out=output_buffers,
            out_dtype=out.dtype,
            store_d=main_transform is None,
            config=None,
            config_constraints=config_constraints,
            tuned=tuned,
            concat_layout=(
                None if main_transform is None else main_transform.concat_layout
            ),
            **blockscaled_kwargs,
            **operands,
        )
    return result["main" if main_transform is not None else "D"]
