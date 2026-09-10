# mypy: allow-untyped-defs
from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import inspect
import logging
import os
import threading
from typing import Any, TYPE_CHECKING

import torch
from torch._inductor.kernel.flex_gemm.constraints import (
    FlexGemmGroupedMainOutputTransform,
    FlexGemmLocalReduceGeometry,
    LOCAL_REDUCE_FEED_MAIN_ARG_NAME,
    LOCAL_REDUCE_RUNTIME_OUT_ERROR,
    LOCAL_REDUCE_STORE_ARG_NAME,
)
from torch._inductor.kernel.flex_gemm.output_layout import FlexGemmOutputLayout
from torch._inductor.runtime.cache_dir_utils import cache_dir
from torch._prims_common import is_expandable_to
from torch._subclasses.fake_tensor import is_fake


if TYPE_CHECKING:
    from collections.abc import Callable


log = logging.getLogger(__name__)


def inductor_quack_cache_dir() -> str:
    """Return the Inductor-owned QuACK cache root for generated FlexGEMM."""
    return os.path.join(cache_dir(), "quack")


_CONFIG_SELECTION: contextvars.ContextVar[list[Any] | None] = contextvars.ContextVar(
    "flex_gemm_config_selection", default=None
)


@contextlib.contextmanager
def select_flex_gemm_configs():
    """Collect QuACK's legal GemmConfigs without launching the generated call."""
    configs: list[Any] = []
    token = _CONFIG_SELECTION.set(configs)
    try:
        yield configs
    finally:
        _CONFIG_SELECTION.reset(token)


_PRECOMPILE_LOCK = threading.Lock()


def precompile_flex_gemm_kernel(run: Callable[[], None]) -> None:
    """Compile the QuACK kernel ``run`` needs now instead of at first call.

    ``run`` invokes the generated kernel on real tensors; the cold ``jit_cache``
    miss compiles in-process. CuTeDSL compilation is serialized because
    Inductor precompiles choices from several threads.
    """
    with _PRECOMPILE_LOCK:
        run()


def flex_gemm_candidate_configs(
    epimod: Any,
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor | None,
    output_buffers: dict[str, torch.Tensor],
    operands: dict[str, Any],
    concat_layout: Any,
) -> list[Any]:
    """Return QuACK's legal configs for this call, its untuned default first.

    Mirrors ``EpiMod.__call__``'s selection: the per-arch default leads when it
    is legal, followed by every other candidate the EpiMod's ops accept.
    """
    from torch._vendor.quack.cute_dsl_utils import get_device_capacity
    from torch._vendor.quack.gemm_config import (
        blockscaled_default_config,
        default_config,
    )
    from torch._vendor.quack.gemm_runtime.autotune import (
        _legal_mod_configs,
        mod_selection_args,
    )

    device = a.device
    capacity = get_device_capacity(device)[0]
    b_kn = capacity >= 9 and not concat_layout
    named_args = mod_selection_args(
        operands,
        {name: output_buffers[name] for name in epimod.outputs},
        A=a,
        B=b if b_kn else b.mT,
        b_kn=b_kn,
        SFA=sfa,
        concat_layout=concat_layout,
    )
    preferred = (
        blockscaled_default_config(a.shape[-2], b.shape[-1], device_capacity=capacity)
        if sfa is not None
        else default_config(device)
    )
    return _legal_mod_configs(epimod, device, named_args, preferred_config=preferred)


# NOTE [Byte-backed epilogue tensor storage]
# PyTorch bool tensors are byte-addressed while CuTeDSL models cutlass.Boolean as
# a 1-bit logical type; Float4E2M1FN similarly exposes two values per byte. Pass
# both through QuACK as their physical uint8 carrier.
def quack_epilogue_arg(arg: torch.Tensor) -> torch.Tensor:
    """Adapt logical epilogue tensors to QuACK's physical tensor ABI."""
    if arg.dtype in (torch.bool, torch.float4_e2m1fn_x2):
        return arg.view(torch.uint8)
    return arg


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
class FlexGemmEpiModLocalReducePlan:
    """QuACK EpiOp configuration for one analyzed grouped local reduction."""

    geometry: FlexGemmLocalReduceGeometry
    out: torch.Tensor | None = None
    feeds_main: bool = False
    combine: str | None = None
    finalize: Callable[..., Any] | str | None = None
    finalize_operands: tuple[str, ...] = ()
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
        if self.finalize_operands and not callable(
            self.store_finalize or self.finalize
        ):
            raise RuntimeError(
                "FlexGEMM EpiMod finalize operands require a generated finalizer"
            )

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
            self.finalize_operands,
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
    local_reduce: FlexGemmEpiModLocalReducePlan | None,
    main_transform: FlexGemmGroupedMainOutputTransform | None,
):
    """Build and cache a QuACK TensorSSA EpiMod from generated FlexGEMM metadata."""
    epilogue_arg_dtypes = tuple(arg.dtype for arg in epilogue_args)
    key = (
        epilogue_fn,
        epilogue_arg_kinds,
        epilogue_arg_dtypes,
        aux_output_count,
        None if local_reduce is None else local_reduce.cache_key,
        main_transform,
    )
    epimod = _EPIMOD_CACHE.get(key)
    if epimod is not None:
        return epimod

    from torch._vendor.quack import cute_dsl_utils
    from torch._vendor.quack.epilogue import frontend as epilogue_module, ops as epi_ops

    op_types = {
        "row": epi_ops.RowVecLoad,
        "col": epi_ops.ColVecLoad,
        "tile": epi_ops.TileLoad,
    }
    ops: dict[str, Any] = {}
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
        from torch._inductor.kernel.flex_gemm.quack_ops.main_store import (
            GroupedMainStore,
        )

        outputs = (GroupedMainStore("main", main_transform.group),)
    else:
        outputs = tuple(f"output{index}" for index in range(aux_output_count))
    sinks: dict[str, Any] = {}
    prepass = None
    prepass_outs = ()
    if local_reduce is not None:
        from torch._inductor.kernel.flex_gemm.quack_ops import grouped_reduce

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
                finalize_arity = (
                    len(inspect.signature(store_finalize).parameters)
                    - len(local_reduce.finalize_operands)
                    if callable(store_finalize)
                    else 1
                )
                if finalize_arity == 2:
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
                        finalize_operands=local_reduce.finalize_operands,
                    )
                else:
                    sink = grouped_reduce.GroupedLocalReduce(
                        LOCAL_REDUCE_STORE_ARG_NAME,
                        axis=local_reduce.axis,
                        group=local_reduce.group,
                        combine=local_reduce.combine,
                        finalize=store_finalize,
                        finalize_operands=local_reduce.finalize_operands,
                        output_layout=output_layout,
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
                    finalize_operands=local_reduce.finalize_operands,
                    output_layout=output_layout,
                )
            if local_reduce.feeds_main:
                ops[LOCAL_REDUCE_FEED_MAIN_ARG_NAME] = reduce_op
            else:
                sinks[LOCAL_REDUCE_FEED_MAIN_ARG_NAME] = reduce_op
    epimod = epilogue_module.fragment_epilogue(
        outputs=outputs,
        ops=ops,
        outs=sinks,
        prepass=prepass,
        prepass_outs=prepass_outs,
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
    out: torch.Tensor,
    aux_outs: tuple[torch.Tensor, ...] = (),
    epilogue_args: tuple[torch.Tensor, ...] = (),
    epilogue_arg_kinds: tuple[str, ...] = (),
    local_reduce: FlexGemmEpiModLocalReducePlan | None = None,
    main_transform: FlexGemmGroupedMainOutputTransform | None = None,
    config: tuple[tuple[str, Any], ...] | None = None,
    stream: int | None = None,
) -> torch.Tensor:
    """Run a dense FlexGEMM call through the vendored QuACK EpiMod.

    ``config`` pins the exact GemmConfig Inductor selected; ``None`` is only
    used by the lowering-time legal-config probe, which returns before launch.
    """
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
        local_reduce,
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
    initialize_local_reduce_out = None
    if local_reduce is not None:
        # QuACK's host_validate checks the compressed buffer against the GEMM
        # problem; only the caller-owned carrier view is built here.
        local_reduce_out = local_reduce.out
        if local_reduce_out is not None and local_reduce.output_layout is not None:
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
    main_name = "main" if main_transform is not None else "D"
    concat_layout = None if main_transform is None else main_transform.concat_layout
    legal_configs = _CONFIG_SELECTION.get()
    if legal_configs is not None:
        if not is_fake(a):
            raise AssertionError("FlexGEMM config probe reached a real GEMM call")
        legal_configs.extend(
            flex_gemm_candidate_configs(
                epimod,
                a,
                b,
                None,
                output_buffers,
                operands,
                concat_layout,
            )
        )
        return output_buffers[main_name]
    if config is not None:
        from torch._vendor.quack.gemm_config import GemmConfig

        quack_config = GemmConfig(**dict(config))
    else:
        quack_config = None
    stream_context = (
        torch.cuda.stream(torch.cuda.ExternalStream(stream, device=a.device))
        if stream is not None
        else contextlib.nullcontext()
    )
    with cache_dir_override(inductor_quack_cache_dir()), stream_context:
        # Layout callbacks predicate logical stores but do not own padded bytes.
        if initialize_local_reduce_out is not None:
            initialize_local_reduce_out.zero_()
        result = epimod(
            a,
            b,
            C=effective_C,
            out=output_buffers,
            out_dtype=out.dtype,
            store_d=main_transform is None,
            config=quack_config,
            tuned=False,
            concat_layout=concat_layout,
            compile_dispatch=False,
            **operands,
        )
    return result[main_name]
