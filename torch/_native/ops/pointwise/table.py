from __future__ import annotations

import math
from typing import Literal, NamedTuple, TYPE_CHECKING

import torch
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND as PromotionKind

from .export_metadata import ARITHMETIC_SPECS
from .signature import Functor


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ._types import DTypes, Mode, Number


class PointwiseDef(NamedTuple):
    aten: str  # ATen symbol including overload.
    nin: int
    fn: Functor
    promotion: PromotionKind = PromotionKind.DEFAULT
    scalars: tuple[str, ...] = ()  # Runtime scalar argument names.
    scalar_defaults: tuple[Number, ...] = ()
    nout: int = 1
    # Dtypes outside these restrictions use ATen.
    dtypes: DTypes | None = None
    int_dtypes: DTypes = ()
    int_via_float: bool = False
    # Skip out overloads incompatible with wrapped scalars.
    skip_out_variant: bool = False
    precondition: (
        Callable[[Sequence[torch.Tensor], torch.Tensor | None], bool] | None
    ) = None
    optional_defaults: dict[str, Callable[[torch.dtype], Number]] | None = None
    mode_kwarg: str | None = None
    mode_fns: dict[Mode, Functor] | None = None
    mode_promotion: dict[Mode, PromotionKind] | None = None
    mode_int_only: tuple[Mode, ...] = ()
    # Allowed CPU scalar input indices; None permits any single input.
    cpu_scalar_inputs: tuple[int, ...] | None = None
    output_dispatch: bool = False


_DEFAULT = PromotionKind.DEFAULT
_INT2FLOAT = PromotionKind.INT_TO_FLOAT
_BOOL = PromotionKind.ALWAYS_BOOL

_INT_DTYPES = (torch.int32, torch.int64)


def _lowest(dt: torch.dtype) -> int | float:
    return -math.inf if dt.is_floating_point else torch.iinfo(dt).min


def _highest(dt: torch.dtype) -> int | float:
    return math.inf if dt.is_floating_point else torch.iinfo(dt).max


def _all_same_dtype(ins: Sequence[torch.Tensor], tgt: torch.Tensor | None) -> bool:
    return all(t.dtype == ins[0].dtype for t in ins) and (
        tgt is None or tgt.dtype == ins[0].dtype
    )


def _boolean_out(ins: Sequence[torch.Tensor], tgt: torch.Tensor | None) -> bool:
    return tgt is None or tgt.dtype is torch.bool


def _lerp_dtypes_ok(ins: Sequence[torch.Tensor], tgt: torch.Tensor | None) -> bool:
    # End must match self; a zero-dimensional weight can promote.
    scalar_weight = len(ins) == 2 or ins[2].dim() == 0
    return (
        ins[0].dtype == ins[1].dtype
        and (scalar_weight or ins[2].dtype == ins[0].dtype)
        and (tgt is None or scalar_weight or tgt.dtype == ins[0].dtype)
    )


PointwiseVariant = Literal["functional", "out", "inplace"]


POINTWISE_VARIANTS: tuple[PointwiseVariant, ...] = (
    "functional",
    "out",
    "inplace",
)

POINTWISE_DEF_TABLE: tuple[PointwiseDef, ...] = (
    # Arithmetic.
    PointwiseDef(
        "neg", 1, Functor("_neg"), int_dtypes=_INT_DTYPES, precondition=_all_same_dtype
    ),
    PointwiseDef(
        "add.Tensor",
        2,
        Functor("_add"),
        scalars=("alpha",),
        int_dtypes=_INT_DTYPES,
    ),
    PointwiseDef(
        "sub.Tensor",
        2,
        Functor("_sub"),
        scalars=("alpha",),
        int_dtypes=_INT_DTYPES,
    ),
    PointwiseDef(
        "mul.Tensor",
        2,
        Functor("_mul"),
        int_dtypes=_INT_DTYPES,
    ),
    # True division promotes integral inputs to floating results.
    PointwiseDef(
        "div.Tensor",
        2,
        Functor("_div"),
        promotion=_INT2FLOAT,
        int_via_float=True,
        cpu_scalar_inputs=(),
    ),
    PointwiseDef("maximum", 2, Functor("_maximum"), int_dtypes=_INT_DTYPES),
    PointwiseDef("minimum", 2, Functor("_minimum"), int_dtypes=_INT_DTYPES),
    PointwiseDef(
        "atan2", 2, Functor("_atan2"), promotion=_INT2FLOAT, int_via_float=True
    ),
    PointwiseDef(
        "rsub.Tensor",
        2,
        Functor("_rsub"),
        scalars=("alpha",),
        int_dtypes=_INT_DTYPES,
    ),
    # fmax/fmin suppress NaNs; maximum/minimum propagate them.
    PointwiseDef("fmax", 2, Functor("_fmax"), int_dtypes=_INT_DTYPES),
    PointwiseDef("fmin", 2, Functor("_fmin"), int_dtypes=_INT_DTYPES),
    PointwiseDef("clamp_min.Tensor", 2, Functor("_maximum"), int_dtypes=_INT_DTYPES),
    PointwiseDef("clamp_max.Tensor", 2, Functor("_minimum"), int_dtypes=_INT_DTYPES),
    PointwiseDef(
        "copysign.Tensor",
        2,
        Functor("_copysign"),
        promotion=_INT2FLOAT,
        int_via_float=True,
        dtypes=(torch.bfloat16, torch.float32, torch.float64),
    ),
    PointwiseDef("hypot", 2, Functor("_hypot")),
    PointwiseDef(
        "logaddexp",
        2,
        Functor("_logaddexp"),
        cpu_scalar_inputs=(),
        output_dispatch=True,
    ),
    PointwiseDef(
        "xlogy.Tensor", 2, Functor("_xlogy"), promotion=_INT2FLOAT, int_via_float=True
    ),
    # Integer power uses ATen.
    PointwiseDef("pow.Tensor_Tensor", 2, Functor("_pow"), skip_out_variant=True),
    PointwiseDef(
        "fmod.Tensor",
        2,
        Functor("_fmod_int"),
        dtypes=_INT_DTYPES,
    ),
    PointwiseDef(
        "remainder.Tensor",
        2,
        Functor("_remainder_int"),
        dtypes=_INT_DTYPES,
    ),
    PointwiseDef(
        "floor_divide",
        2,
        Functor("_floor_divide_int"),
        dtypes=_INT_DTYPES,
    ),
    PointwiseDef("floor", 1, Functor("_floor"), precondition=_all_same_dtype),
    PointwiseDef("ceil", 1, Functor("_ceil"), precondition=_all_same_dtype),
    PointwiseDef("trunc", 1, Functor("_trunc"), precondition=_all_same_dtype),
    PointwiseDef(
        "sign",
        1,
        Functor("_sign"),
        int_dtypes=_INT_DTYPES,
        precondition=_all_same_dtype,
    ),
    PointwiseDef(
        "relu",
        1,
        Functor("_relu"),
        int_dtypes=_INT_DTYPES,
        precondition=_all_same_dtype,
    ),
    PointwiseDef(
        "abs", 1, Functor("_abs"), int_dtypes=_INT_DTYPES, precondition=_all_same_dtype
    ),
    PointwiseDef("round", 1, Functor("_round"), precondition=_all_same_dtype),
    PointwiseDef("frac", 1, Functor("_frac"), precondition=_all_same_dtype),
    *(
        PointwiseDef(
            spec.name,
            1,
            Functor("_" + spec.name),
            promotion=_INT2FLOAT,
            int_via_float=True,
        )
        for spec in ARITHMETIC_SPECS
        if spec.recipe == "int_to_float"
    ),
    # Transcendental math.
    PointwiseDef(
        "logit",
        1,
        Functor("_logit"),
        promotion=_INT2FLOAT,
        scalars=("eps",),
        optional_defaults={"eps": lambda dt: -1.0},
    ),
    # Activations.
    PointwiseDef(
        "silu", 1, Functor("_silu"), promotion=_INT2FLOAT, precondition=_all_same_dtype
    ),
    PointwiseDef(
        "elu",
        1,
        Functor("_elu"),
        scalars=("alpha", "scale", "input_scale"),
        promotion=_INT2FLOAT,
        precondition=_all_same_dtype,
    ),
    PointwiseDef(
        "celu",
        1,
        Functor("_elu"),
        scalars=("alpha",),
        promotion=_INT2FLOAT,
        precondition=_all_same_dtype,
    ),
    PointwiseDef(
        "mish", 1, Functor("_mish"), promotion=_INT2FLOAT, precondition=_all_same_dtype
    ),
    PointwiseDef(
        "softplus",
        1,
        Functor("_softplus"),
        scalars=("beta", "threshold"),
        scalar_defaults=(1, 20),
        promotion=_INT2FLOAT,
        precondition=_all_same_dtype,
    ),
    PointwiseDef(
        "hardtanh",
        1,
        Functor("_hardtanh"),
        scalars=("min_val", "max_val"),
        scalar_defaults=(-1, 1),
        precondition=_all_same_dtype,
    ),
    PointwiseDef(
        "hardsigmoid",
        1,
        Functor("_hardsigmoid"),
        promotion=_INT2FLOAT,
        precondition=_all_same_dtype,
    ),
    PointwiseDef(
        "threshold",
        1,
        Functor("_threshold"),
        scalars=("threshold", "value"),
        precondition=_all_same_dtype,
    ),
    PointwiseDef(
        "hardshrink",
        1,
        Functor("_hardshrink"),
        scalars=("lambd",),
        scalar_defaults=(0.5,),
        precondition=_all_same_dtype,
    ),
    PointwiseDef(
        "softshrink",
        1,
        Functor("_softshrink"),
        scalars=("lambd",),
        scalar_defaults=(0.5,),
        precondition=_all_same_dtype,
    ),
    # Comparisons.
    PointwiseDef(
        "gt.Tensor", 2, Functor("_gt"), promotion=_BOOL, int_dtypes=_INT_DTYPES
    ),
    PointwiseDef(
        "lt.Tensor", 2, Functor("_lt"), promotion=_BOOL, int_dtypes=_INT_DTYPES
    ),
    PointwiseDef(
        "ge.Tensor", 2, Functor("_ge"), promotion=_BOOL, int_dtypes=_INT_DTYPES
    ),
    PointwiseDef(
        "le.Tensor", 2, Functor("_le"), promotion=_BOOL, int_dtypes=_INT_DTYPES
    ),
    PointwiseDef(
        "eq.Tensor", 2, Functor("_eq"), promotion=_BOOL, int_dtypes=_INT_DTYPES
    ),
    PointwiseDef(
        "ne.Tensor", 2, Functor("_ne"), promotion=_BOOL, int_dtypes=_INT_DTYPES
    ),
    PointwiseDef(
        "logical_and",
        2,
        Functor("_logical_and"),
        promotion=_BOOL,
        int_dtypes=_INT_DTYPES,
    ),
    PointwiseDef(
        "logical_or", 2, Functor("_logical_or"), promotion=_BOOL, int_dtypes=_INT_DTYPES
    ),
    PointwiseDef(
        "logical_xor",
        2,
        Functor("_logical_xor"),
        promotion=_BOOL,
        int_dtypes=_INT_DTYPES,
    ),
    PointwiseDef(
        "logical_not",
        1,
        Functor("_logical_not"),
        promotion=_BOOL,
        int_dtypes=_INT_DTYPES,
    ),
    PointwiseDef(
        "signbit",
        1,
        Functor("_signbit"),
        promotion=_BOOL,
        precondition=_boolean_out,
    ),
    PointwiseDef("isnan", 1, Functor("_isnan"), promotion=_BOOL),
    PointwiseDef("isinf", 1, Functor("_isinf"), promotion=_BOOL),
    # Bitwise operations.
    PointwiseDef("bitwise_and.Tensor", 2, Functor("_bitwise_and"), dtypes=_INT_DTYPES),
    PointwiseDef("bitwise_or.Tensor", 2, Functor("_bitwise_or"), dtypes=_INT_DTYPES),
    PointwiseDef("bitwise_xor.Tensor", 2, Functor("_bitwise_xor"), dtypes=_INT_DTYPES),
    PointwiseDef(
        "bitwise_not",
        1,
        Functor("_bitwise_not"),
        dtypes=_INT_DTYPES,
        precondition=_all_same_dtype,
    ),
    PointwiseDef(
        "bitwise_left_shift.Tensor",
        2,
        Functor("_bitwise_left_shift"),
        output_dispatch=True,
        dtypes=_INT_DTYPES,
    ),
    PointwiseDef(
        "bitwise_right_shift.Tensor",
        2,
        Functor("_bitwise_right_shift"),
        output_dispatch=True,
        dtypes=_INT_DTYPES,
    ),
    # Ternary and multiple-output operations.
    PointwiseDef(
        "addcmul", 3, Functor("_addcmul"), scalars=("value",), cpu_scalar_inputs=(2,)
    ),
    PointwiseDef(
        "frexp.Tensor",
        1,
        Functor("_frexp"),
        nout=2,
        skip_out_variant=True,
        # FP64 currently uses ATen.
        dtypes=(torch.float16, torch.bfloat16, torch.float32),
    ),
    PointwiseDef(
        "sgn", 1, Functor("_sign"), int_dtypes=_INT_DTYPES, precondition=_all_same_dtype
    ),
    PointwiseDef("angle", 1, Functor("_angle"), promotion=_INT2FLOAT),
    PointwiseDef(
        "isposinf", 1, Functor("_isposinf"), promotion=_BOOL, precondition=_boolean_out
    ),
    PointwiseDef(
        "isneginf", 1, Functor("_isneginf"), promotion=_BOOL, precondition=_boolean_out
    ),
    PointwiseDef(
        "heaviside",
        2,
        Functor("_heaviside"),
        int_dtypes=_INT_DTYPES,
        precondition=_all_same_dtype,
    ),
    PointwiseDef(
        "logaddexp2",
        2,
        Functor("_logaddexp2"),
        cpu_scalar_inputs=(),
        output_dispatch=True,
    ),
    PointwiseDef("special_entr", 1, Functor("_entr"), promotion=_INT2FLOAT),
    PointwiseDef(
        "special_xlog1py",
        2,
        Functor("_xlog1py"),
        promotion=_INT2FLOAT,
        int_via_float=True,
    ),
    PointwiseDef("hardswish", 1, Functor("_hardswish"), precondition=_all_same_dtype),
    PointwiseDef(
        "leaky_relu",
        1,
        Functor("_leaky_relu"),
        scalars=("negative_slope",),
        scalar_defaults=(0.01,),
        precondition=_all_same_dtype,
    ),
    # ATen rejects integer addcdiv.
    PointwiseDef(
        "addcdiv", 3, Functor("_addcdiv"), scalars=("value",), cpu_scalar_inputs=()
    ),
    # Mode-selected formulas and optional scalars.
    PointwiseDef(
        "gelu",
        1,
        Functor("_gelu_erf"),
        mode_kwarg="approximate",
        mode_fns={
            None: Functor("_gelu_erf"),
            "none": Functor("_gelu_erf"),
            "tanh": Functor("_gelu_tanh"),
        },
        precondition=_all_same_dtype,
    ),
    # True division promotes integers; floor/trunc division preserves them.
    PointwiseDef(
        "div.Tensor_mode",
        2,
        Functor("_div"),
        mode_kwarg="rounding_mode",
        mode_fns={
            None: Functor("_div"),
            "floor": Functor("_floor_divide_int"),
            "trunc": Functor("_div_trunc_int"),
        },
        mode_promotion={None: _INT2FLOAT, "floor": _DEFAULT, "trunc": _DEFAULT},
        int_dtypes=_INT_DTYPES,
        mode_int_only=("floor", "trunc"),
        cpu_scalar_inputs=(),
    ),
    # Omitted clamp bounds use the matching infinity or integer limit.
    PointwiseDef(
        "clamp",
        1,
        Functor("_clamp"),
        scalars=("min", "max"),
        optional_defaults={"min": _lowest, "max": _highest},
        int_dtypes=_INT_DTYPES,
        precondition=_all_same_dtype,
    ),
    # Default infinity replacements depend on the result dtype's finite limits.
    PointwiseDef(
        "nan_to_num",
        1,
        Functor("_nan_to_num"),
        scalars=("nan", "posinf", "neginf"),
        optional_defaults={
            "nan": lambda dt: 0.0,
            "posinf": lambda dt: torch.finfo(dt).max,
            "neginf": lambda dt: -torch.finfo(dt).max,
        },
        precondition=_all_same_dtype,
    ),
    PointwiseDef(
        "lerp.Scalar",
        2,
        Functor("_lerp_scalar"),
        scalars=("weight",),
        precondition=_lerp_dtypes_ok,
        cpu_scalar_inputs=(),
    ),
    PointwiseDef(
        "lerp.Tensor",
        3,
        Functor("_lerp"),
        precondition=_lerp_dtypes_ok,
        cpu_scalar_inputs=(2,),
    ),
)


def variant_aten_name(base_aten: str, variant: PointwiseVariant) -> str | None:
    r"""Find the ATen overload for a functional, out or in-place variant."""
    op, _, ov = base_aten.partition(".")
    ov = ov or "default"
    if variant == "functional":
        return base_aten
    if variant == "out":
        packet = getattr(torch.ops.aten, op)
        arguments = getattr(packet, ov)._schema.arguments
        for name in packet.overloads():
            candidate = getattr(packet, name)._schema.arguments
            outputs = [a for a in candidate if a.alias_info and a.alias_info.is_write]
            inputs = [
                a for a in candidate if not (a.alias_info and a.alias_info.is_write)
            ]
            if not outputs or len(inputs) != len(arguments):
                continue
            if all(
                a.name == b.name
                and a.type == b.type
                and a.kwarg_only == b.kwarg_only
                and a.default_value == b.default_value
                for a, b in zip(arguments, inputs)
            ):
                return f"{op}.{name}"
        return None
    if variant == "inplace":
        packet = getattr(torch.ops.aten, op + "_", None)
        if packet is None:
            return None
        arguments = getattr(getattr(torch.ops.aten, op), ov)._schema.arguments
        for name in packet.overloads():
            candidate = getattr(packet, name)._schema.arguments
            if len(candidate) == len(arguments) and all(
                a.name == b.name
                and a.type == b.type
                and a.kwarg_only == b.kwarg_only
                and a.default_value == b.default_value
                for a, b in zip(arguments, candidate)
            ):
                return op + "_" if name == "default" else f"{op}_.{name}"
        return None
    return None
