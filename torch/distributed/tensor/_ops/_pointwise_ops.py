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
    aten.acos_.default,
    aten.acos.out,
    aten.acosh.default,
    aten.acosh_.default,
    aten.acosh.out,
    aten.asin.default,
    aten.asin_.default,
    aten.asinh.default,
    aten.asinh_.default,
    aten.atan.default,
    aten.atan_.default,
    aten.add.Scalar,
    aten.add.out,
    aten.add_.Scalar,
    aten.addcdiv.default,
    aten.addcdiv.out,
    aten.addcdiv_.default,
    aten.addcmul.default,
    aten.addcmul.out,
    aten.addcmul_.default,
    aten.angle.default,
    aten.angle.out,
    aten.asin.out,
    aten.asinh.out,
    aten.atan.out,
    aten.atan2.default,
    aten.atan2.out,
    aten.atan2_.default,
    aten.atanh.default,
    aten.atanh_.default,
    aten.atanh.out,
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
    aten.ceil.out,
    aten.clamp.default,
    aten.clamp.Tensor,
    aten.clamp.out,
    aten.clamp_.default,
    aten.clamp_.Tensor,
    aten.clamp_min.default,
    aten.clamp_min.Tensor,
    aten.clamp_max.default,
    aten.clamp_max.Tensor,
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
    aten.deg2rad.out,
    aten.digamma.default,
    aten.digamma.out,
    aten.digamma_.default,
    aten.div.Tensor_mode,
    aten.div.out,
    aten.div.out_mode,
    aten.div_.Tensor_mode,
    aten.eq.Tensor,
    aten.eq.Tensor_out,
    aten.eq.Scalar,
    aten.eq.Scalar_out,
    aten.erf.out,
    aten.erfc.default,
    aten.erfc.out,
    aten.erfc_.default,
    aten.erfinv.default,
    aten.erfinv_.default,
    aten.erfinv.out,
    aten.exp.default,
    aten.exp_.default,
    aten.exp.out,
    aten.exp2.default,
    aten.exp2_.default,
    aten.exp2.out,
    aten.expm1.default,
    aten.expm1_.default,
    aten.expm1.out,
    aten.float_power.Scalar,
    aten.float_power.Scalar_out,
    aten.float_power.Tensor_Scalar,
    aten.float_power.Tensor_Scalar_out,
    aten.float_power.Tensor_Tensor,
    aten.float_power.Tensor_Tensor_out,
    aten.float_power_.Scalar,
    aten.float_power_.Tensor,
    aten.floor.out,
    aten.fmax.out,
    aten.fmin.out,
    aten.fmod.Scalar,
    aten.fmod.Scalar_out,
    aten.fmod.Tensor,
    aten.fmod.Tensor_out,
    aten.fmod_.Scalar,
    aten.fmod_.Tensor,
    aten.frac.default,
    aten.frac.out,
    aten.frac_.default,
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
    aten.i0_.default,
    aten.i0.out,
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
    aten.log_.default,
    aten.log.out,
    aten.log10.default,
    aten.log10_.default,
    aten.log10.out,
    aten.log1p.default,
    aten.log1p_.default,
    aten.log1p.out,
    aten.log2.default,
    aten.log2_.default,
    aten.log2.out,
    aten.logaddexp.out,
    aten.logaddexp2.out,
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
    aten.logit_.default,
    aten.logit.out,
    aten.masked_fill.Scalar,
    aten.masked_fill_.Scalar,
    aten.mul.out,
    aten.mvlgamma.default,
    aten.mvlgamma.out,
    aten.mvlgamma_.default,
    aten.native_dropout_backward.default,
    aten.native_dropout_backward.out,
    aten.nan_to_num.out,
    aten.ne.Scalar,
    aten.neg.out,
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
    aten.rad2deg.out,
    aten.remainder.Scalar,
    aten.remainder.Scalar_Tensor,
    aten.remainder.Scalar_out,
    aten.remainder.Tensor,
    aten.remainder.Tensor_out,
    aten.remainder_.Scalar,
    aten.remainder_.Tensor,
    aten.round.decimals,
    aten.round.decimals_out,
    aten.round.default,
    aten.round.out,
    aten.round_.decimals,
    aten.round_.default,
    aten.rsqrt.default,
    aten.rsqrt_.default,
    aten.rsqrt.out,
    aten.rsub.Scalar,
    aten.sgn.default,
    aten.sgn.out,
    aten.sgn_.default,
    aten.sigmoid.out,
    aten.sign.default,
    aten.sign.out,
    aten.sign_.default,
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
    aten.sinh.out,
    aten.sqrt.default,
    aten.sqrt_.default,
    aten.sqrt.out,
    aten.square.default,
    aten.square.out,
    aten.square_.default,
    aten.sub.Scalar,
    aten.sub.out,
    aten.sub_.Scalar,
    aten.tan.default,
    aten.tan_.default,
    aten.tan.out,
    aten.tanh.out,
    aten.true_divide.Tensor,
    aten.trunc.out,
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
]

# Linear pointwise ops, split by linearity type.
# Partial rules are specified explicitly at registration time via named constants.
unary_linear_ops = [aten.to.dtype]

binary_additive_ops = [
    aten.add.Tensor,
    aten.add_.Tensor,
    aten.sub.Tensor,
    aten.sub_.Tensor,
]

binary_multiplicative_ops = [
    aten.div.Tensor,
    aten.div_.Tensor,
    aten.mul.Tensor,
    aten.mul_.Tensor,
]

# Scalar multiplicative ops: unary linear rules
scalar_multiplicative_ops = [
    aten.div.Scalar,
    aten.div_.Scalar,
    aten.mul.Scalar,
    aten.mul_.Scalar,
]

# Monotone increasing unary ops: P(max)->P(max), P(min)->P(min)
# Only ops that are monotone on their ENTIRE domain belong here.
# Ops with restricted domains (e.g. log on (0,∞), asin on [-1,1]) do NOT qualify
# because P(max) offsets can push inputs outside the valid domain.
monotone_increasing_unary_ops = [
    aten.ceil.default,
    aten.ceil_.default,
    aten.erf.default,
    aten.erf_.default,
    aten.floor.default,
    aten.floor_.default,
    aten.relu.default,
    aten.relu_.default,
    aten.sigmoid.default,
    aten.sigmoid_.default,
    aten.sinh.default,
    aten.sinh_.default,
    aten.tanh.default,
    aten.tanh_.default,
    aten.trunc.default,
    aten.trunc_.default,
]

# Monotone decreasing unary ops: P(max)->P(min), P(min)->P(max)
# Note: acos excluded due to domain constraints [-1,1] causing validation failures
monotone_decreasing_unary_ops: list[OpOverload] = []

# All-partial-preserving unary ops: P(x)->P(x) for all x
# These ops preserve the exact value for each element, only transforming units/representation
all_partial_preserving_unary_ops = [
    aten.deg2rad.default,
    aten.deg2rad_.default,
    aten.nan_to_num.default,
    aten.nan_to_num_.default,
    aten.rad2deg.default,
    aten.rad2deg_.default,
]

# Monotone binary ops: maps op -> which partial to preserve (max, min, or None)
monotone_binary_ops: dict[torch._ops.OpOverload, str | None] = {
    aten.fmax.default: "max",
    aten.fmax.out: "max",
    aten.fmin.default: "min",
    aten.fmin.out: "min",
    aten.logaddexp.default: None,
    aten.logaddexp.out: None,
    aten.logaddexp2.default: None,
    aten.logaddexp2.out: None,
    aten.maximum.default: "max",
    aten.maximum.out: "max",
    aten.minimum.default: "min",
    aten.minimum.out: "min",
}

# Rule constants for partial placement propagation
_UNARY_LINEAR_RULES: list[list[Placement]] = [
    [Partial("sum"), Partial("sum")],
    [Partial("avg"), Partial("avg")],
]

_BINARY_ADDITIVE_RULES: list[list[Placement]] = [
    [Partial("sum"), Partial("sum"), Partial("sum")],
    [Partial("avg"), Partial("avg"), Partial("avg")],
]

_BINARY_MULTIPLICATIVE_RULES: list[list[Placement]] = [
    [Partial("sum"), Partial("sum"), Replicate()],
    [Partial("sum"), Replicate(), Partial("sum")],
    [Partial("avg"), Partial("avg"), Replicate()],
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



# Register new single-dim strategies for categorized ops.

for op in unary_linear_ops:
    register_single_dim_strategy(
        op, schema_info=RuntimeSchemaInfo(static_kwargkey=["out"])
    )(_make_partial_strategy(extra_rules=_UNARY_LINEAR_RULES))

for op in binary_additive_ops:
    register_single_dim_strategy(
        op, schema_info=RuntimeSchemaInfo(static_kwargkey=["out"])
    )(_make_partial_strategy(extra_rules=_BINARY_ADDITIVE_RULES))

for op in binary_multiplicative_ops:
    register_single_dim_strategy(
        op, schema_info=RuntimeSchemaInfo(static_kwargkey=["out"])
    )(
        _make_partial_strategy(
            extra_rules=_UNARY_LINEAR_RULES + _BINARY_MULTIPLICATIVE_RULES
        )
    )

# Scalar multiplicative ops: unary linear rules
# Scalar multiplicative ops: unary linear rules
for op in scalar_multiplicative_ops:
    register_single_dim_strategy(
        op, schema_info=RuntimeSchemaInfo(1, static_kwargkey=["out"])
    )(_make_partial_strategy(extra_rules=_UNARY_LINEAR_RULES))

# Monotone increasing unary: P(max)->P(max), P(min)->P(min)
for op in monotone_increasing_unary_ops:
    register_single_dim_strategy(
        op, schema_info=RuntimeSchemaInfo(static_kwargkey=["out"])
    )(
        _make_partial_strategy(
            extra_rules=[
                [Partial("max"), Partial("max")],
                [Partial("min"), Partial("min")],
            ]
        )
    )

# Monotone decreasing unary: P(max)->P(min), P(min)->P(max)
for op in monotone_decreasing_unary_ops:
    register_single_dim_strategy(
        op, schema_info=RuntimeSchemaInfo(static_kwargkey=["out"])
    )(
        _make_partial_strategy(
            extra_rules=[
                [Partial("min"), Partial("max")],
                [Partial("max"), Partial("min")],
            ]
        )
    )

# All-partial-preserving unary: P(x)->P(x) for all x
for op in all_partial_preserving_unary_ops:
    register_single_dim_strategy(
        op, schema_info=RuntimeSchemaInfo(static_kwargkey=["out"])
    )(
        _make_partial_strategy(
            extra_rules=[[Partial(r), Partial(r)] for r in ("sum", "avg", "max", "min")]
        )
    )

# neg: linear (P(sum)->P(sum), P(avg)->P(avg)) + monotone decreasing
register_single_dim_strategy(
    [aten.neg.default, aten.neg_.default],
    schema_info=RuntimeSchemaInfo(static_kwargkey=["out"]),
)(
    _make_partial_strategy(
        # pyrefly: ignore [bad-argument-type]
        extra_rules=_UNARY_LINEAR_RULES
        + [
            [Partial("min"), Partial("max")],
            [Partial("max"), Partial("min")],
        ]
    )
)

# Monotone binary ops
for op, preserve in monotone_binary_ops.items():
    rules = list(_monotone_binary_base_rules)
    if preserve == "max":
        rules.append([Partial("max"), Partial("max"), Partial("max")])
    elif preserve == "min":
        rules.append([Partial("min"), Partial("min"), Partial("min")])
    register_single_dim_strategy(
        op, schema_info=RuntimeSchemaInfo(static_kwargkey=["out"])
    )(_make_partial_strategy(extra_rules=rules))

# copy_(self, src): preserves all Partial types (2 tensor inputs → 3-element rules)
register_single_dim_strategy(
    aten.copy_.default, schema_info=RuntimeSchemaInfo(static_kwargkey=["out"])
)(
    _make_partial_strategy(
        extra_rules=[[Partial(r), Partial(r), Partial(r)] for r in ("sum", "avg", "max", "min")]
    )
)

# Generic pointwise ops: just Shard + Replicate strategies
for op in pointwise_ops:
    register_single_dim_strategy(
        op, schema_info=RuntimeSchemaInfo(static_kwargkey=["out"])
    )(_make_partial_strategy())

# TODO: add all for_each ops
for_each_ops = [
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
    aten._foreach_div_.List,
    aten._foreach_div_.Scalar,
    aten._foreach_div_.ScalarList,
    aten._foreach_div_.Tensor,
    aten._foreach_div.List,
    aten._foreach_div.Scalar,
    aten._foreach_div.ScalarList,
    aten._foreach_div.Tensor,
    aten._foreach_lerp_.Scalar,
    aten._foreach_maximum_.List,
    aten._foreach_mul.Scalar,
    aten._foreach_mul.ScalarList,
    aten._foreach_mul.Tensor,
    aten._foreach_mul.List,
    aten._foreach_mul_.Scalar,
    aten._foreach_mul_.ScalarList,
    aten._foreach_mul_.Tensor,
    aten._foreach_mul_.List,
    aten._foreach_pow.List,
    aten._foreach_pow.ScalarList,
    aten._foreach_neg.default,
    aten._foreach_neg_.default,
    aten._foreach_reciprocal_.default,
    aten._foreach_sub.Scalar,
    aten._foreach_sub_.Scalar,
    aten._foreach_sub.List,
    aten._foreach_sub_.List,
    aten._foreach_sub.ScalarList,
    aten._foreach_sub_.ScalarList,
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

for_each_linearity_ops = [
    aten._foreach_add.Scalar,
    aten._foreach_add_.Scalar,
    aten._foreach_add_.ScalarList,
    aten._foreach_add.List,
    aten._foreach_add_.List,
]



# Foreach ops: use register_single_dim_strategy (auto-detected by expanded_foreach_strategy)
for op in for_each_ops:
    register_single_dim_strategy(op, schema_info=RuntimeSchemaInfo(needs_pytree=True))(
        _make_partial_strategy()
    )

# Foreach ops with linearity (add/sub with scalars)
for op in for_each_linearity_ops:
    register_single_dim_strategy(op, schema_info=RuntimeSchemaInfo(needs_pytree=True))(
        _make_partial_strategy(extra_rules=_UNARY_LINEAR_RULES)
    )

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
