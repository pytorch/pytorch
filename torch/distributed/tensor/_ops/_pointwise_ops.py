# Copyright (c) Meta Platforms, Inc. and affiliates
from collections.abc import Callable

import torch
from torch._ops import OpOverload
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.distributed.tensor._op_schema import (
    ArgsType,
    KwargsType,
    RuntimeSchemaInfo,
)
from torch.distributed.tensor._ops.single_dim_strategy import (
    _ShardingPlaceholder,
    register_single_dim_strategy,
)
from torch.distributed.tensor._ops.utils import infer_broadcast_dims_map
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
)


aten = torch.ops.aten
prims = torch.ops.prims
# leave the remaining pointwise_ops list here for convenience,
# Below ops are some pointwise ops that are yet to be supported,
# they might not be a complete list.
# pointwise_ops = [
#     "fake_quantize_per_channel_affine",
#     "fake_quantize_per_tensor_affine",
#     "floor_divide",  # floor_divide is deprecated
#     "frexp",  # multiple output pointwise op, need to add support
#     "gradient",  #  need investigation on this op
#     "imag",  # complex data type only
#     "quantized_batch_norm",
#     "quantized_max_pool1d",
#     "quantized_max_pool2d",
#     "real",  # complex data type only
# ]

# Linear pointwise ops, split by linearity type.
unary_linear_ops = [aten.to.dtype]

binary_additive_ops = [
    aten.add.Tensor,
    aten.add.out,
    aten.add_.Tensor,
    aten.sub.Tensor,
    aten.sub.out,
    aten.sub_.Tensor,
    aten._foreach_add.List,
    aten._foreach_add_.List,
    aten._foreach_sub.List,
    aten._foreach_sub_.List,
]

# mul: partials propagate through either arg. div: only through numerator.
binary_mul_ops = [
    aten.mul.Tensor,
    aten.mul.out,
    aten.mul_.Tensor,
    aten._foreach_mul.List,
    aten._foreach_mul.Tensor,
    aten._foreach_mul_.List,
    aten._foreach_mul_.Tensor,
]
binary_div_ops = [
    aten.div.Tensor,
    aten.div.out,
    aten.div_.Tensor,
    aten._foreach_div.List,
    aten._foreach_div.Tensor,
    aten._foreach_div_.List,
    aten._foreach_div_.Tensor,
]


scalar_linear_ops = [
    aten.div.Scalar,
    aten.div_.Scalar,
    aten.mul.Scalar,
    aten.mul_.Scalar,
    aten._foreach_add.Scalar,
    aten._foreach_add_.Scalar,
    aten._foreach_add_.ScalarList,
    aten._foreach_div.Scalar,
    aten._foreach_div.ScalarList,
    aten._foreach_div_.Scalar,
    aten._foreach_div_.ScalarList,
    aten._foreach_mul.Scalar,
    aten._foreach_mul.ScalarList,
    aten._foreach_mul_.Scalar,
    aten._foreach_mul_.ScalarList,
    aten._foreach_sub.Scalar,
    aten._foreach_sub.ScalarList,
    aten._foreach_sub_.Scalar,
    aten._foreach_sub_.ScalarList,
]

neg_ops = [
    aten.neg.default,
    aten.neg.out,
    aten.neg_.default,
    aten._foreach_neg.default,
    aten._foreach_neg_.default,
]

# Non-decreasing unary ops: f(max(a,b)) = max(f(a),f(b)).
# Only ops that are non-decreasing on their ENTIRE domain belong here.
# Ops with restricted domains (e.g. log on (0,∞), asin on [-1,1]) do NOT qualify
# because P(max) offsets can push inputs outside the valid domain.
non_decreasing_unary_ops = [
    aten.asinh.default,
    aten.asinh.out,
    aten.asinh_.default,
    aten.atan.default,
    aten.atan.out,
    aten.atan_.default,
    aten.ceil.default,
    aten.ceil.out,
    aten.ceil_.default,
    aten.erf.default,
    aten.erf.out,
    aten.erf_.default,
    aten.exp.default,
    aten.exp.out,
    aten.exp_.default,
    aten.exp2.default,
    aten.exp2.out,
    aten.exp2_.default,
    aten.expm1.default,
    aten.expm1.out,
    aten.expm1_.default,
    aten.floor.default,
    aten.floor.out,
    aten.floor_.default,
    aten.relu.default,
    aten.relu.out,
    aten.relu_.default,
    aten.round.decimals,
    aten.round.decimals_out,
    aten.round.default,
    aten.round.out,
    aten.round_.decimals,
    aten.round_.default,
    aten.sgn.default,
    aten.sgn.out,
    aten.sgn_.default,
    aten.sigmoid.default,
    aten.sigmoid.out,
    aten.sigmoid_.default,
    aten.sign.default,
    aten.sign.out,
    aten.sign_.default,
    aten.sinh.default,
    aten.sinh.out,
    aten.sinh_.default,
    aten.tanh.default,
    aten.tanh.out,
    aten.tanh_.default,
    aten.trunc.default,
    aten.trunc.out,
    aten.trunc_.default,
]

# Non-increasing unary ops: f(max(a,b)) = min(f(a),f(b)).
# Note: acos excluded due to domain constraints [-1,1] causing validation failures
non_increasing_unary_ops: list[OpOverload] = [
    aten.erfc.default,
    aten.erfc.out,
    aten.erfc_.default,
    aten.special_erfcx.default,
    aten.special_erfcx.out,
]

# All-partial-preserving unary ops: P(x)->P(x) for all x
# These ops preserve the exact value for each element, only transforming units/representation
all_partial_preserving_unary_ops = [
    aten.deg2rad.default,
    aten.deg2rad.out,
    aten.deg2rad_.default,
    aten.nan_to_num.default,
    aten.nan_to_num.out,
    aten.nan_to_num_.default,
    aten.rad2deg.default,
    aten.rad2deg.out,
    aten.rad2deg_.default,
]

# Binary ops monotonically increasing in both arguments.
# max-preserving: P(max)+P(max)->P(max) because max(max(a),max(b)) = max(a,b)
monotonic_max_preserving_binary_ops = [
    aten.clamp_min.Tensor,
    aten.clamp_min.Tensor_out,
    aten.fmax.default,
    aten.fmax.out,
    aten.maximum.default,
    aten.maximum.out,
    prims.fmax.default,
    aten._foreach_maximum_.List,
]

# min-preserving: P(min)+P(min)->P(min) because min(min(a),min(b)) = min(a,b)
monotonic_min_preserving_binary_ops = [
    aten.clamp_max.Tensor,
    aten.clamp_max.Tensor_out,
    aten.fmin.default,
    aten.fmin.out,
    aten.minimum.default,
    aten.minimum.out,
    prims.fmin.default,
]

# Monotonic increasing in both args but don't preserve any specific partial type.
monotonic_binary_ops = [
    aten.logaddexp.default,
    aten.logaddexp.out,
    aten.logaddexp2.default,
    aten.logaddexp2.out,
]

pointwise_ops = [
    # please keep the entries below alphabetically sorted
    aten.__ilshift__.Scalar,
    aten.__ilshift__.Tensor,
    aten.__irshift__.Scalar,
    aten.__irshift__.Tensor,
    aten.__lshift__.Scalar,
    aten.__lshift__.Tensor,
    aten.__rshift__.Scalar,
    aten.__rshift__.Tensor,
    aten._conj.default,
    aten.abs.default,
    aten.abs.out,
    aten.abs_.default,
    aten.acos.default,
    aten.acos.out,
    aten.acos_.default,
    aten.acosh.default,
    aten.acosh.out,
    aten.acosh_.default,
    aten.add.Scalar,
    aten.add_.Scalar,
    aten.addcdiv.default,
    aten.addcdiv.out,
    aten.addcdiv_.default,
    aten.addcmul.default,
    aten.addcmul.out,
    aten.addcmul_.default,
    aten.angle.default,
    aten.angle.out,
    aten.asin.default,
    aten.asin.out,
    aten.asin_.default,
    aten.atan2.default,
    aten.atan2.out,
    aten.atan2_.default,
    aten.atanh.default,
    aten.atanh.out,
    aten.atanh_.default,
    aten.bitwise_and.Scalar,
    aten.bitwise_and.Scalar_Tensor,
    aten.bitwise_and.Scalar_out,
    aten.bitwise_and.Tensor,
    aten.bitwise_and.Tensor_out,
    aten.bitwise_and_.Scalar,
    aten.bitwise_and_.Tensor,
    aten.bitwise_left_shift.Scalar_Tensor,
    aten.bitwise_left_shift.Tensor,
    aten.bitwise_left_shift.Tensor_Scalar,
    aten.bitwise_left_shift.Tensor_Scalar_out,
    aten.bitwise_left_shift.Tensor_out,
    aten.bitwise_left_shift_.Tensor,
    aten.bitwise_left_shift_.Tensor_Scalar,
    aten.bitwise_not.default,
    aten.bitwise_not.out,
    aten.bitwise_not_.default,
    aten.bitwise_or.Scalar,
    aten.bitwise_or.Scalar_Tensor,
    aten.bitwise_or.Scalar_out,
    aten.bitwise_or.Tensor,
    aten.bitwise_or.Tensor_out,
    aten.bitwise_or_.Scalar,
    aten.bitwise_or_.Tensor,
    aten.bitwise_right_shift.Scalar_Tensor,
    aten.bitwise_right_shift.Tensor,
    aten.bitwise_right_shift.Tensor_Scalar,
    aten.bitwise_right_shift.Tensor_Scalar_out,
    aten.bitwise_right_shift.Tensor_out,
    aten.bitwise_right_shift_.Tensor,
    aten.bitwise_right_shift_.Tensor_Scalar,
    aten.bitwise_xor.Scalar,
    aten.bitwise_xor.Scalar_Tensor,
    aten.bitwise_xor.Scalar_out,
    aten.bitwise_xor.Tensor,
    aten.bitwise_xor.Tensor_out,
    aten.bitwise_xor_.Scalar,
    aten.bitwise_xor_.Tensor,
    aten.clamp.default,
    aten.clamp.Tensor,
    aten.clamp.out,
    aten.clamp_.default,
    aten.clamp_.Tensor,
    aten.clip.default,
    aten.clip.out,
    aten.clip_.default,
    aten.conj_physical.default,
    aten.conj_physical.out,
    aten.conj_physical_.default,
    aten.copysign.Scalar,
    aten.copysign.Scalar_out,
    aten.copysign.Tensor,
    aten.copysign.out,
    aten.copysign_.Scalar,
    aten.copysign_.Tensor,
    aten.cos.default,
    aten.cos.out,
    aten.cos_.default,
    aten.cosh.default,
    aten.cosh.out,
    aten.cosh_.default,
    aten.digamma.default,
    aten.digamma.out,
    aten.digamma_.default,
    aten.div.Tensor_mode,
    aten.div.out_mode,
    aten.div_.Tensor_mode,
    aten.eq.Tensor,
    aten.eq.Tensor_out,
    aten.eq.Scalar,
    aten.eq.Scalar_out,
    aten.erfinv.default,
    aten.erfinv.out,
    aten.erfinv_.default,
    aten.float_power.Scalar,
    aten.float_power.Scalar_out,
    aten.float_power.Tensor_Scalar,
    aten.float_power.Tensor_Scalar_out,
    aten.float_power.Tensor_Tensor,
    aten.float_power.Tensor_Tensor_out,
    aten.float_power_.Scalar,
    aten.float_power_.Tensor,
    aten.fmod.Scalar,
    aten.fmod.Scalar_out,
    aten.fmod.Tensor,
    aten.fmod.Tensor_out,
    aten.fmod_.Scalar,
    aten.fmod_.Tensor,
    aten.frac.default,
    aten.frac.out,
    aten.frac_.default,
    aten.gcd.default,
    aten.gcd.out,
    aten.ge.Scalar,
    aten.ge.Tensor,
    aten.gelu.default,
    aten.gt.Tensor,
    aten.gt.Tensor_out,
    aten.gt.Scalar,
    aten.gt.Scalar_out,
    aten.gt.Scalar,
    aten.gt.Tensor,
    aten.heaviside.default,
    aten.heaviside.out,
    aten.hypot.default,
    aten.hypot.out,
    aten.hypot_.default,
    aten.i0.default,
    aten.i0.out,
    aten.i0_.default,
    aten.igamma.default,
    aten.igamma.out,
    aten.igamma_.default,
    aten.igammac.default,
    aten.igammac.out,
    aten.igammac_.default,
    aten.isinf.default,
    aten.isnan.default,
    aten.isneginf.default,
    aten.isneginf.out,
    aten.isposinf.default,
    aten.isposinf.out,
    aten.ldexp.Tensor,
    aten.ldexp.out,
    aten.ldexp_.default,
    aten.lt.Tensor,
    aten.lt.Tensor_out,
    aten.lt.Scalar,
    aten.lt.Scalar_out,
    aten.le.Scalar,
    aten.le.Tensor,
    aten.lerp.Scalar,
    aten.lerp.Scalar_out,
    aten.lerp.Tensor,
    aten.lerp.Tensor_out,
    aten.lerp_.Scalar,
    aten.lerp_.Tensor,
    aten.lgamma.default,
    aten.lgamma.out,
    aten.lgamma_.default,
    aten.log.default,
    aten.log.out,
    aten.log10.default,
    aten.log10.out,
    aten.log10_.default,
    aten.log1p.default,
    aten.log1p.out,
    aten.log1p_.default,
    aten.log2.default,
    aten.log2.out,
    aten.log2_.default,
    aten.log_.default,
    aten.logical_and.default,
    aten.logical_and.out,
    aten.logical_and_.default,
    aten.logical_not.default,
    aten.logical_not.out,
    aten.logical_not_.default,
    aten.logical_or.default,
    aten.logical_or.out,
    aten.logical_or_.default,
    aten.logical_xor.default,
    aten.logical_xor.out,
    aten.logical_xor_.default,
    aten.logit.default,
    aten.logit.out,
    aten.logit_.default,
    aten.masked_fill.Scalar,
    aten.masked_fill_.Scalar,
    aten.mvlgamma.default,
    aten.mvlgamma.out,
    aten.mvlgamma_.default,
    aten.native_dropout_backward.default,
    aten.native_dropout_backward.out,
    aten.ne.Scalar,
    aten.nextafter.default,
    aten.nextafter.out,
    aten.nextafter_.default,
    aten.polygamma.default,
    aten.polygamma.out,
    aten.polygamma_.default,
    aten.positive.default,
    aten.pow.Scalar,
    aten.pow.Scalar_out,
    aten.pow.Tensor_Scalar,
    aten.pow.Tensor_Scalar_out,
    aten.pow.Tensor_Tensor,
    aten.pow.Tensor_Tensor_out,
    aten.pow_.Scalar,
    aten.pow_.Tensor,
    aten.reciprocal.default,
    aten.reciprocal.out,
    aten.reciprocal_.default,
    aten.remainder.Scalar,
    aten.remainder.Scalar_Tensor,
    aten.remainder.Scalar_out,
    aten.remainder.Tensor,
    aten.remainder.Tensor_out,
    aten.remainder_.Scalar,
    aten.remainder_.Tensor,
    aten.rsqrt.default,
    aten.rsqrt.out,
    aten.rsqrt_.default,
    aten.rsub.Scalar,
    aten.signbit.default,
    aten.signbit.out,
    aten.silu.default,
    aten.silu.out,
    aten.sin.default,
    aten.sin.out,
    aten.sin_.default,
    aten.sinc.default,
    aten.sinc.out,
    aten.sinc_.default,
    aten.sqrt.default,
    aten.sqrt.out,
    aten.sqrt_.default,
    aten.square.default,
    aten.square.out,
    aten.square_.default,
    aten.sub.Scalar,
    aten.sub_.Scalar,
    aten.tan.default,
    aten.tan.out,
    aten.tan_.default,
    aten.true_divide.Tensor,
    aten.where.self,
    aten.where.self_out,
    aten.xlogy.OutScalar_Self,
    aten.xlogy.OutScalar_Other,
    aten.xlogy.OutTensor,
    aten.xlogy.Scalar_Other,
    aten.xlogy.Scalar_Self,
    aten.xlogy.Tensor,
    aten.xlogy_.Scalar_Other,
    aten.xlogy_.Tensor,
    # backward point-wise ops
    # please keep the entries below alphabetically sorted
    aten.gelu_backward.default,
    aten.sigmoid_backward.default,
    aten.silu_backward.default,
    aten.tanh_backward.default,
    aten.threshold_backward.default,
    # prims ops
    # please keep the entries below alphabetically sorted
    prims.bessel_i0e.default,
    prims.bessel_i1.default,
    prims.bessel_i1e.default,
    prims.bessel_j0.default,
    prims.bessel_j1.default,
    prims.div.default,
    prims.erfcx.default,
    prims.gcd.default,
    prims.frexp.default,
    prims.ndtri.default,
    prims.ne.default,
    prims.spherical_bessel_j0.default,
    prims.zeta.default,
    # Foreach ops without specialized partial rules
    aten._foreach_abs.default,
    aten._foreach_abs_.default,
    aten._foreach_addcdiv_.Scalar,
    aten._foreach_addcdiv_.ScalarList,
    aten._foreach_addcdiv_.Tensor,
    aten._foreach_addcmul.Scalar,
    aten._foreach_addcmul_.Scalar,
    aten._foreach_addcmul_.ScalarList,
    aten._foreach_addcmul_.Tensor,
    aten._foreach_clamp_max_.Scalar,
    aten._foreach_clamp_min_.Scalar,
    aten._foreach_lerp_.Scalar,
    aten._foreach_pow.List,
    aten._foreach_pow.ScalarList,
    aten._foreach_reciprocal_.default,
    aten._foreach_sqrt.default,
    aten._foreach_sqrt_.default,
    aten._foreach_zero_.default,
    aten._foreach_exp.default,
    aten._foreach_exp_.default,
    aten._foreach_cos.default,
    aten._foreach_cos_.default,
    aten._foreach_log.default,
    aten._foreach_log_.default,
    aten._amp_foreach_non_finite_check_and_unscale_.default,
]

# Rule constants for partial placement propagation
_UNARY_LINEAR_RULES: list[list[Placement]] = [
    [Partial("sum"), Partial("sum")],
    [Partial("avg"), Partial("avg")],
]

_BINARY_ADDITIVE_RULES: list[list[Placement]] = [
    [Partial("sum"), Partial("sum"), Partial("sum")],
    [Partial("avg"), Partial("avg"), Partial("avg")],
    # P(x), R -> P(x): adding/subtracting a replicated value preserves partial type
    # (the replicated value is constant across ranks, so reduce order is unaffected)
    [Partial("avg"), Partial("avg"), Replicate()],
    [Partial("max"), Partial("max"), Replicate()],
    [Partial("min"), Partial("min"), Replicate()],
    # R, P(avg) -> P(avg): avg is linear so this holds for any alpha
    # (R, P(max/min) excluded: negative alpha would flip the ordering)
    [Partial("avg"), Replicate(), Partial("avg")],
]

_monotone_binary_base_rules: list[list[Placement]] = [
    [Partial("max"), Partial("max"), Replicate()],
    [Partial("max"), Replicate(), Partial("max")],
    [Partial("min"), Partial("min"), Replicate()],
    [Partial("min"), Replicate(), Partial("min")],
]


def single_mesh_dim_common_pointwise_strategy(
    args_schema: ArgsType,
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Generate Shard placement strategies for pointwise ops based on tensor shapes."""
    tensor_arg_metas: list[TensorMeta] = [
        arg for arg in args_schema if isinstance(arg, TensorMeta)
    ]
    common_shape = torch.broadcast_shapes(
        *[arg.shape for arg in args_schema if isinstance(arg, TensorMeta)]
    )
    placements_list: list[list[Placement | _ShardingPlaceholder]] = []
    for i in range(len(common_shape)):
        shard_placements: list[Placement | _ShardingPlaceholder] = [
            _ShardingPlaceholder(i)
        ]
        for arg in tensor_arg_metas:
            common_dim_to_arg_dim = infer_broadcast_dims_map(common_shape, arg.shape)
            if common_dim_to_arg_dim[i] >= 0:
                shard_placements.append(_ShardingPlaceholder(common_dim_to_arg_dim[i]))
            else:
                shard_placements.append(Replicate())
        placements_list.append(shard_placements)
    return placements_list


def _make_partial_strategy(
    extra_rules: list[list[Placement]] | None = None,
) -> Callable[
    [OpOverload, ArgsType, KwargsType], list[list[Placement | _ShardingPlaceholder]]
]:
    """Factory for single-dim strategies that add partial placement rules."""

    def strategy(
        op: OpOverload,
        args_schema: ArgsType,
        kwargs_schema: KwargsType,
    ) -> list[list[Placement | _ShardingPlaceholder]]:
        placements = single_mesh_dim_common_pointwise_strategy(args_schema)
        if extra_rules:
            n_tensors = sum(1 for arg in args_schema if isinstance(arg, TensorMeta))
            expected_len = 1 + n_tensors
            for rule in extra_rules:
                if len(rule) == expected_len:
                    # pyrefly: ignore [bad-argument-type]
                    placements.append(rule)
        return placements

    return strategy


def _register(
    op: OpOverload,
    extra_rules: list[list[Placement]] | None = None,
    static_argnum: int = 0,
) -> None:
    """Register a single-dim strategy for an op, auto-detecting foreach schema."""
    if "_foreach_" in str(op) or "_amp_foreach_" in str(op):
        schema = RuntimeSchemaInfo(needs_pytree=True)
    else:
        schema = RuntimeSchemaInfo(static_argnum, static_kwargkey=["out"])
    register_single_dim_strategy(op, schema_info=schema)(
        _make_partial_strategy(extra_rules=extra_rules)
    )



# Register single-dim strategies for all categorized ops.

_NEG_RULES: list[list[Placement]] = (
    _UNARY_LINEAR_RULES
    + [  # pyrefly: ignore[bad-assignment]
        [Partial("min"), Partial("max")],
        [Partial("max"), Partial("min")],
    ]
)

_MONOTONIC_INCREASING_RULES: list[list[Placement]] = [
    [Partial("max"), Partial("max")],
    [Partial("min"), Partial("min")],
]

_MONOTONIC_DECREASING_RULES: list[list[Placement]] = [
    [Partial("min"), Partial("max")],
    [Partial("max"), Partial("min")],
]

_ALL_PARTIAL_PRESERVING_RULES: list[list[Placement]] = [
    [Partial(r), Partial(r)] for r in ("sum", "avg", "max", "min")
]

for op in unary_linear_ops:
    _register(op, _UNARY_LINEAR_RULES)

for op in binary_additive_ops:
    _register(op, _BINARY_ADDITIVE_RULES)

# _UNARY_LINEAR_RULES handles the scalar promotion case: Python's __mul__/__truediv__
# promote scalars to 0-dim tensors, so aten.mul.Scalar dispatches as aten.mul.Tensor
# with n_tensors=1, matching the length-2 unary rules.
_MUL_RULES: list[list[Placement]] = [
    [Partial("sum"), Partial("sum"), Replicate()],
    [Partial("avg"), Partial("avg"), Replicate()],
    [Partial("sum"), Replicate(), Partial("sum")],
    [Partial("avg"), Replicate(), Partial("avg")],
]

_DIV_RULES: list[list[Placement]] = [
    [Partial("sum"), Partial("sum"), Replicate()],
    [Partial("avg"), Partial("avg"), Replicate()],
]

for op in binary_mul_ops:
    _register(op, _UNARY_LINEAR_RULES + _MUL_RULES)

for op in binary_div_ops:
    _register(op, _UNARY_LINEAR_RULES + _DIV_RULES)

for op in scalar_linear_ops:
    _register(op, _UNARY_LINEAR_RULES, static_argnum=1)

for op in non_decreasing_unary_ops:
    _register(op, _MONOTONIC_INCREASING_RULES)

for op in non_increasing_unary_ops:
    _register(op, _MONOTONIC_DECREASING_RULES)

for op in all_partial_preserving_unary_ops:
    _register(op, _ALL_PARTIAL_PRESERVING_RULES)

# neg: linear + monotonic decreasing
for op in neg_ops:
    _register(op, _NEG_RULES)

# Monotonic binary ops: max-preserving
for op in monotonic_max_preserving_binary_ops:
    _register(
        op,
        # pyrefly: ignore[bad-argument-type]
        _monotone_binary_base_rules
        + [[Partial("max"), Partial("max"), Partial("max")]],
    )

# Monotonic binary ops: min-preserving
for op in monotonic_min_preserving_binary_ops:
    _register(
        op,
        # pyrefly: ignore[bad-argument-type]
        _monotone_binary_base_rules
        + [[Partial("min"), Partial("min"), Partial("min")]],
    )

# Monotonic binary ops: no specific partial preservation
for op in monotonic_binary_ops:
    _register(op, _monotone_binary_base_rules)

# copy_(self, src): preserves all Partial types (2 tensor inputs → 3-element rules)
_register(
    aten.copy_.default,
    [[Partial(r), Partial(r), Partial(r)] for r in ("sum", "avg", "max", "min")],
)

# Generic pointwise ops: just Shard + Replicate strategies (no partial rules)
for op in pointwise_ops:
    _register(op)



fused_ops = [
    aten._fused_adam_.default,
    aten._fused_adam.default,
    aten._fused_adam.tensor_lr,
    aten._fused_adam_.tensor_lr,
    aten._fused_adamw_.default,
    aten._fused_adamw.default,
    aten._fused_adamw.tensor_lr,
    aten._fused_adamw_.tensor_lr,
]

# Fused adam/adamw ops: state_steps (input 5) is a Replicate scalar tensor that may be on
# a different mesh than the other inputs. cross_mesh_indices=[5] tells the expansion
# infrastructure to preserve the input's own mesh and assert Replicate placements.
for op in fused_ops:
    register_single_dim_strategy(
        op,
        schema_info=RuntimeSchemaInfo(needs_pytree=True),
        cross_mesh_indices=[5],
    )(_make_partial_strategy())
