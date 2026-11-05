# Register only TMA-eligible pointwise calls; other layouts keep their ATen kernels.

from __future__ import annotations

from functools import partial
from typing import Any, cast, TYPE_CHECKING

import torch
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND as PromotionKind

from ... import cutedsl_utils as cu
from ...utils import capability as cap
from ...utils.lazy import LazyModule
from .table import (
    POINTWISE_DEF_TABLE,
    POINTWISE_VARIANTS,
    PointwiseDef,
    PointwiseVariant,
    variant_aten_name,
)


if TYPE_CHECKING:
    from ...cutedsl import dtypes as dt
    from . import kernel as K, ops
else:
    dt = LazyModule("torch._native.cutedsl.dtypes")
    K = LazyModule("torch._native.ops.pointwise.kernel")
    ops = LazyModule("torch._native.ops.pointwise.ops")


_SUPPORTED = (torch.float16, torch.bfloat16, torch.float32)
_Inputs = tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor | None]


def _scalars(
    row: PointwiseDef,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    compute: torch.dtype,
) -> tuple[Any, ...]:
    # Return named scalar arguments in row.scalars order. Explicit arguments
    # override scalar_defaults; absent defaults use 1 (the add/sub alpha default).
    # Optional parameters instead use a result-dtype-dependent default callback.
    pos = args[row.nin :]
    opt = row.optional_defaults or {}
    out = []
    for i, name in enumerate(row.scalars):
        if name in kwargs and kwargs[name] is not None:
            out.append(kwargs[name])
            continue
        if i < len(pos) and pos[i] is not None:
            out.append(pos[i])
            continue
        if name in opt:
            out.append(opt[name](compute))
            continue
        out.append(row.scalar_defaults[i] if row.scalar_defaults else 1)
    return tuple(out)


def _mode_of(row: PointwiseDef, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    # The row's mode-kwarg VALUE for this call (gelu's approximate, div's
    # rounding_mode). aten allows it positionally too, after the tensors and any
    # numeric scalars, so fall back to scanning the positional tail for a str/None.
    if row.mode_kwarg in kwargs:
        return kwargs[row.mode_kwarg]
    tail = args[row.nin + len(row.scalars) :]
    return tail[0] if tail else None


def _function(
    row: PointwiseDef, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[str | None, PromotionKind]:
    if row.mode_kwarg is None:
        return row.fn, row.promotion
    mode = _mode_of(row, args, kwargs)
    return (row.mode_fns or {}).get(mode), row.promotion


def _common_inputs(
    row: PointwiseDef,
    variant: PointwiseVariant,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> _Inputs | None:
    # Reject common cases before operator-specific checks or layout selection.
    ins = args[: row.nin]
    if any(not isinstance(t, torch.Tensor) or cap.is_traced(t) for t in ins):
        return None
    if all(t.is_contiguous() for t in ins):
        return None
    ref = ins[0]
    target = kwargs.get("out") if variant.out_from == "out_kw" else None
    if variant.out_from == "out_kw" and (
        not isinstance(target, torch.Tensor) or cap.is_traced(target)
    ):
        return None
    return ins, ref, target


def _layout_cond(
    row: PointwiseDef,
    inputs: _Inputs,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> bool:
    ins, ref, target = inputs
    shape = tuple(ref.shape)
    result_dtype = (
        torch.bool if row.promotion is PromotionKind.ALWAYS_BOOL else ref.dtype
    )
    dtypes = _SUPPORTED if row.dtypes is None else row.dtypes
    if not K.can_run(ins, shape, result_dtype, target, dtypes=dtypes):
        return False
    return all(
        isinstance(s, (bool, int, float))
        for s in _scalars(row, args, kwargs, ref.dtype)
    )


def _cond(
    row: PointwiseDef, variant: PointwiseVariant, *args: Any, **kwargs: Any
) -> bool:
    inputs = _common_inputs(row, variant, args, kwargs)
    return inputs is not None and _layout_cond(row, inputs, args, kwargs)


def _cond_alpha(
    row: PointwiseDef, variant: PointwiseVariant, *args: Any, **kwargs: Any
) -> bool:
    inputs = _common_inputs(row, variant, args, kwargs)
    if inputs is None:
        return False
    _, ref, _ = inputs
    alpha = _scalars(row, args, kwargs, ref.dtype)[0]
    if isinstance(alpha, (bool, complex)):
        return False
    return _layout_cond(row, inputs, args, kwargs)


def _cond_softshrink(
    row: PointwiseDef, variant: PointwiseVariant, *args: Any, **kwargs: Any
) -> bool:
    inputs = _common_inputs(row, variant, args, kwargs)
    if inputs is None:
        return False
    _, ref, _ = inputs
    lambd = _scalars(row, args, kwargs, ref.dtype)[0]
    if not isinstance(lambd, (int, float)) or not ref.is_floating_point():
        return False
    if not 0 <= lambd <= torch.finfo(ref.dtype).max:
        return False
    return _layout_cond(row, inputs, args, kwargs)


def _cond_celu(
    row: PointwiseDef, variant: PointwiseVariant, *args: Any, **kwargs: Any
) -> bool:
    inputs = _common_inputs(row, variant, args, kwargs)
    if inputs is None:
        return False
    _, ref, _ = inputs
    if _scalars(row, args, kwargs, ref.dtype)[0] == 0:
        return False
    return _layout_cond(row, inputs, args, kwargs)


def _cond_clamp(
    row: PointwiseDef, variant: PointwiseVariant, *args: Any, **kwargs: Any
) -> bool:
    inputs = _common_inputs(row, variant, args, kwargs)
    if inputs is None:
        return False
    low = kwargs.get("min", args[1] if len(args) > 1 else None)
    high = kwargs.get("max", args[2] if len(args) > 2 else None)
    if low is None and high is None:
        return False
    return _layout_cond(row, inputs, args, kwargs)


def _cond_mode(
    row: PointwiseDef, variant: PointwiseVariant, *args: Any, **kwargs: Any
) -> bool:
    inputs = _common_inputs(row, variant, args, kwargs)
    if inputs is None:
        return False
    # Decline unknown modes and floor/trunc division: only true division has math.
    if _mode_of(row, args, kwargs) not in (row.mode_fns or {}):
        return False
    return _layout_cond(row, inputs, args, kwargs)


def _cond_heaviside(
    row: PointwiseDef, variant: PointwiseVariant, *args: Any, **kwargs: Any
) -> bool:
    inputs = _common_inputs(row, variant, args, kwargs)
    if inputs is None:
        return False
    ins, _, target = inputs
    if row.precondition is not None and not row.precondition(ins, target):
        return False
    return _layout_cond(row, inputs, args, kwargs)


def _cond_lerp(
    row: PointwiseDef, variant: PointwiseVariant, *args: Any, **kwargs: Any
) -> bool:
    inputs = _common_inputs(row, variant, args, kwargs)
    if inputs is None:
        return False
    ins, _, target = inputs
    if row.precondition is not None:
        if not row.precondition(ins, target):
            return False
    elif ins[0].dtype != ins[1].dtype:
        return False
    return _layout_cond(row, inputs, args, kwargs)


def _cond_bool_out(
    row: PointwiseDef, variant: PointwiseVariant, *args: Any, **kwargs: Any
) -> bool:
    inputs = _common_inputs(row, variant, args, kwargs)
    if inputs is None:
        return False
    _, _, target = inputs
    if target is not None and target.dtype != torch.bool:
        return False
    return _layout_cond(row, inputs, args, kwargs)


_CONDITIONS = {
    "add.Tensor": _cond_alpha,
    "sub.Tensor": _cond_alpha,
    "rsub.Tensor": _cond_alpha,
    "softshrink": _cond_softshrink,
    "celu": _cond_celu,
    "clamp": _cond_clamp,
    "gelu": _cond_mode,
    "div.Tensor_mode": _cond_mode,
    "heaviside": _cond_heaviside,
    "lerp.Scalar": _cond_lerp,
    "lerp.Tensor": _cond_lerp,
    "signbit": _cond_bool_out,
    "isposinf": _cond_bool_out,
    "isneginf": _cond_bool_out,
}


def _impl(
    row: PointwiseDef, variant: PointwiseVariant, *args: Any, **kwargs: Any
) -> torch.Tensor:
    ins = args[: row.nin]
    ref = ins[0]
    scalars = _scalars(row, args, kwargs, ref.dtype)
    fn_name, _ = _function(row, args, kwargs)
    shape = tuple(ref.shape)
    result_dtype = (
        torch.bool if row.promotion is PromotionKind.ALWAYS_BOOL else ref.dtype
    )
    out = (
        kwargs["out"]
        if variant.out_from == "out_kw"
        else torch.empty(shape, device=ref.device, dtype=result_dtype)
    )
    comp_type = dt.torch2cute[torch.float32]
    consts = tuple(comp_type(s) for s in scalars)
    K.run(ops.get_fn(cast(str, fn_name)), row.aten, ins, out, consts, comp_type)
    return out


def register_pointwise_overrides() -> None:
    for row in POINTWISE_DEF_TABLE:
        cond = _CONDITIONS.get(row.aten, _cond)
        for variant in POINTWISE_VARIANTS:
            if row.skip_out_variant and variant.out_from == "out_kw":
                continue
            aten_name = variant_aten_name(row.aten, variant)
            if aten_name is not None:
                cu.register_op_override(
                    "aten",
                    aten_name,
                    "CUDA",
                    cond=partial(cond, row, variant),
                    impl=partial(_impl, row, variant),
                )
