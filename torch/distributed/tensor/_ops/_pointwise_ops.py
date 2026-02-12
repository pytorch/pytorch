# Copyright (c) Meta Platforms, Inc. and affiliates
from collections.abc import Callable, Sequence
from typing import cast

import torch
from torch._ops import OpOverload
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import (
    ArgsType,
    KwargsType,
    OpSchema,
    OpSpec,
    OpStrategy,
    RuntimeSchemaInfo,
    StrategyType,
    TupleStrategy,
)
from torch.distributed.tensor._ops.single_dim_strategy import (
    _ShardingPlaceholder,
    register_single_dim_strategy,
)
from torch.distributed.tensor._ops.utils import (
    generate_redistribute_costs,
    infer_broadcast_dims_map,
    map_placements_after_broadcast,
    normalize_dim,
    register_op_strategy,
)
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.types import _Number
from torch.utils._typing_utils import not_none


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
    aten.acos.out,
    aten.acosh.out,
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
    aten.erfinv.out,
    aten.exp.out,
    aten.exp2.out,
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
    aten.log.out,
    aten.log10.out,
    aten.log1p.out,
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
    aten.sqrt.out,
    aten.square.default,
    aten.square.out,
    aten.square_.default,
    aten.sub.Scalar,
    aten.sub.out,
    aten.sub_.Scalar,
    aten.tan.out,
    aten.tanh.out,
    aten.true_divide.Tensor,
    aten.trunc.default,
    aten.trunc.out,
    aten.trunc_.default,
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
monotone_increasing_unary_ops = [
    aten.asin.default,
    aten.asin_.default,
    aten.asinh.default,
    aten.asinh_.default,
    aten.atan.default,
    aten.atan_.default,
    aten.atanh.default,
    aten.atanh_.default,
    aten.ceil.default,
    aten.ceil_.default,
    aten.erf.default,
    aten.erf_.default,
    aten.erfinv.default,
    aten.erfinv_.default,
    aten.exp.default,
    aten.exp_.default,
    aten.exp2.default,
    aten.exp2_.default,
    aten.expm1.default,
    aten.expm1_.default,
    aten.floor.default,
    aten.floor_.default,
    aten.i0.default,
    aten.i0_.default,
    aten.log.default,
    aten.log_.default,
    aten.log10.default,
    aten.log10_.default,
    aten.log1p.default,
    aten.log1p_.default,
    aten.log2.default,
    aten.log2_.default,
    aten.logit.default,
    aten.logit_.default,
    aten.relu.default,
    aten.relu_.default,
    aten.sgn.default,
    aten.sgn_.default,
    aten.sigmoid.default,
    aten.sigmoid_.default,
    aten.sign.default,
    aten.sign_.default,
    aten.sinh.default,
    aten.sinh_.default,
    aten.sqrt.default,
    aten.sqrt_.default,
    aten.tan.default,
    aten.tan_.default,
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


def pointwise_strategy(
    op_schema: OpSchema,
    preserve_partial: str | None = None,
) -> OpStrategy:
    """Strategy for pointwise ops on the old registration path."""
    followed_strategy_index = -1
    max_shards = -1
    max_ndim = -1

    if op_schema.is_inplace_op():
        followed_strategy = op_schema.args_schema[0]
        followed_strategy_index = 0
    elif op_schema.is_out_variant_op():
        followed_strategy = op_schema.kwargs_schema["out"]
        followed_strategy_index = 100
    else:
        for idx, arg_strategy in enumerate(op_schema.args_schema):
            if not isinstance(arg_strategy, OpStrategy):
                continue
            arg_max_shards = arg_strategy.max_num_shards()
            arg_max_ndim = arg_strategy.ndim
            if (arg_max_shards > max_shards) or (
                arg_max_shards == max_shards and arg_max_ndim > max_ndim
            ):
                followed_strategy_index = idx
                max_shards = arg_max_shards
                max_ndim = arg_max_ndim
        followed_strategy = op_schema.args_schema[followed_strategy_index]

    assert isinstance(followed_strategy, OpStrategy), (
        f"no strategy to follow for {op_schema}!"
    )
    return common_pointwise_strategy(
        op_schema.op,
        op_schema.args_schema,
        followed_strategy,
        followed_strategy_index,
        preserve_partial=preserve_partial,
    )


def common_pointwise_strategy(
    op,
    args_schema: Sequence[object],
    followed_strategy: OpStrategy,
    followed_strategy_index: int,
    linearity: int = -1,
    scalar_tensor_idx: int | None = None,
    preserve_partial: str | None = None,
) -> OpStrategy:
    """
    Common strategy for pointwise operations.

    Args:
        args_schema: Input arguments schema
        followed_strategy: Strategy to follow for output placement
        followed_strategy_index: Index of the strategy being followed
        linearity: depending on the operator, we support different types of linearity
            -1: the operation does not support linearity
            0: the unary operation that supports linearity, output propagates partial.
            1: the binary operation supports add linearity, where it requires every operand
                to be partial, output propagates partial.
            2: the binary operation supports multiplicative linearity, where it requires
                the primary operand to be partial, and the other operands to be replicate,
                output propagates partial.
        scalar_tensor_idx: Index of the Replicate scalar tensor for which we allow the mesh
            to be different from the mesh of followed_strategy
        preserve_partial: If set, Partial placements with this reduce_op will be preserved
            through the operation (e.g., "max" for torch.maximum, "min" for torch.minimum).
    """
    # handle broadcasting
    common_shape = torch.broadcast_shapes(
        *[arg.shape for arg in args_schema if isinstance(arg, OpStrategy)]
    )
    pointwise_strategy = OpStrategy([])

    for op_spec in followed_strategy.strategies:
        spec_to_follow = op_spec.output_spec

        out_placements: list[Placement] = []
        for placement in spec_to_follow.placements:
            if isinstance(placement, Shard | _StridedShard):
                shard_dim = normalize_dim(placement.dim, len(spec_to_follow.shape))
                common_ndim = len(common_shape)
                new_shard_dim = common_ndim - len(spec_to_follow.shape) + shard_dim
                if isinstance(placement, _StridedShard):
                    out_placements.append(
                        _StridedShard(
                            new_shard_dim, split_factor=placement.split_factor
                        )
                    )
                else:
                    out_placements.append(Shard(new_shard_dim))
            elif isinstance(placement, Partial):
                is_scalar_arg = any(isinstance(arg, _Number) for arg in args_schema)
                propagate_partial = not (
                    op in p_sum_scalar_redistribute_ops and is_scalar_arg
                )

                # Check if this partial type should be preserved
                # preserve_partial="all" preserves any Partial type (used for copy_)
                if preserve_partial == "all":
                    out_placements.append(placement)
                elif preserve_partial is not None and placement.is_partial(
                    preserve_partial
                ):
                    out_placements.append(placement)
                # note that only partial-sum and partial-avg are supported for linearity
                elif (
                    linearity >= 0
                    and (placement.is_partial("sum") or placement.is_partial("avg"))
                    and propagate_partial
                ):
                    # propagate the partial placement
                    out_placements.append(placement)
                else:
                    # clear the partial placement if op does not support linearity
                    # by default we just replicate the partial, need to see if this
                    # is optimal for all cases
                    out_placements.append(Replicate())
            else:
                out_placements.append(placement)

        input_specs: list[DTensorSpec] = []
        redistribute_costs: list[list[float]] = []
        for input_idx, input_arg in enumerate(args_schema):
            if isinstance(input_arg, OpStrategy):
                input_arg_spec = input_arg.strategies[0].output_spec

                # sanity check that all args that follow the same strategy
                # are on the same DeviceMesh
                if input_arg.mesh != followed_strategy.mesh:
                    # For the scalar tensor arg in fused ops, do not follow followed_strategy;
                    # instead, let the input mesh and the Replicate placements propagate through.
                    if input_idx == scalar_tensor_idx:
                        assert all(p == Replicate() for p in input_arg_spec.placements)
                        input_arg_target_spec = DTensorSpec(
                            mesh=input_arg.mesh,
                            placements=input_arg_spec.placements,
                            tensor_meta=input_arg_spec.tensor_meta,
                        )
                        input_specs.append(input_arg_target_spec)
                        redistribute_costs.append(
                            generate_redistribute_costs(
                                input_arg, input_arg_target_spec
                            )
                        )
                        continue
                    else:
                        raise ValueError(
                            f"Could not run pointwise computation across different mesh: "
                            f"Found {input_arg.mesh} and {followed_strategy.mesh}!"
                        )

                # every arg follow the out_placements, but need to handle broadcasting
                input_arg_dims_map = infer_broadcast_dims_map(
                    common_shape, input_arg_spec.shape
                )

                # Determine if this input should convert Partial to Replicate based on linearity
                should_convert_partial = (
                    linearity == 2
                    and input_idx
                    != followed_strategy_index  # Don't convert the "followed" strategy
                )

                # For preserve_partial ops, check if non-followed input has incompatible
                # Partial type. If so, it must be redistributed to Replicate first.
                if (
                    preserve_partial is not None
                    and input_idx != followed_strategy_index
                ):
                    for out_p, in_p in zip(out_placements, input_arg_spec.placements):
                        if (
                            isinstance(out_p, Partial)
                            and isinstance(in_p, Partial)
                            and out_p != in_p
                        ):
                            should_convert_partial = True
                            break

                input_target_placements = map_placements_after_broadcast(
                    tuple(out_placements),
                    common_shape,
                    input_arg_dims_map,
                    partial_to_replicate=should_convert_partial,
                )

                input_arg_target_spec = DTensorSpec(
                    mesh=followed_strategy.mesh,
                    placements=input_target_placements,
                    tensor_meta=input_arg_spec.tensor_meta,
                )
                input_specs.append(input_arg_target_spec)
                redistribute_costs.append(
                    generate_redistribute_costs(input_arg, input_arg_target_spec)
                )

        pointwise_strategy.strategies.append(
            OpSpec(
                output_specs=DTensorSpec(
                    mesh=followed_strategy.mesh,
                    placements=tuple(out_placements),
                ),
                input_specs=input_specs,
                redistribute_cost=redistribute_costs,
            )
        )
    return pointwise_strategy


p_sum_scalar_redistribute_ops = {
    aten.add.Tensor,
    aten.add_.Tensor,
    aten.sub.Tensor,
    aten.sub_.Tensor,
}

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

# copy_: preserves all Partial types
register_single_dim_strategy(
    aten.copy_.default, schema_info=RuntimeSchemaInfo(static_kwargkey=["out"])
)(
    _make_partial_strategy(
        extra_rules=[[Partial(r), Partial(r)] for r in ("sum", "avg", "max", "min")]
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


def list_pointwise_strategy(
    op_schema: OpSchema, linearity: bool = False
) -> StrategyType:
    """
    Apply the pointwise strategy to the zipped arguments. For example, if we
    run a foreach add of two lists l1 and l2, then we apply the pointwise
    strategy on each pair (l1[i], l2[i]). If the first argument is a list but
    the second (or later) one is a tensor, then we broadcast the tensor by
    replicating it into a list with the length of the first argument.

    Args:
        mesh (DeviceMesh): device mesh for pointwise ops
        op_schema (OpSchema): schema of the operator to generate strategy for
        linearity (bool): specify whether op(a) + op(b) = op(a + b)

    Returns:
        OpStrategy: generated strategy
    """

    def args_tuple_strategies(
        args_schema: tuple[object, ...],
    ) -> list[TupleStrategy | None]:
        first_arg = args_schema[0]
        assert isinstance(first_arg, TupleStrategy)
        strategy_len = len(first_arg.children)
        tuple_strategies: list[TupleStrategy | None] = []
        for arg_idx, arg in enumerate(args_schema):
            if isinstance(arg, TupleStrategy):
                # every tuple strategy should have the same length
                assert len(arg.children) == strategy_len
                tuple_strategies.append(arg)
            elif isinstance(arg, OpStrategy):
                if arg_idx > 0:  # implicitly broadcast
                    tuple_strategies.append(
                        TupleStrategy([arg for _ in range(strategy_len)])
                    )
                else:
                    raise RuntimeError(
                        f"list op only supports tuple strategy! {op_schema}"
                    )
            else:
                # insert None as placeholder so that the idx of arg is kept
                tuple_strategies.append(None)
        return tuple_strategies

    args_strategies = args_tuple_strategies(op_schema.args_schema)
    follow_strategy: TupleStrategy = not_none(args_strategies[0])
    list_strategy: list[OpStrategy] = []

    for child_idx, child_strtgy in enumerate(follow_strategy.children):
        assert isinstance(child_strtgy, OpStrategy)
        args_schema: list[OpStrategy | None] = [
            cast(OpStrategy, arg_strategy.children[child_idx]) if arg_strategy else None
            for arg_strategy in args_strategies
        ]
        pointwise_strategy: OpStrategy = common_pointwise_strategy(
            op_schema.op,
            args_schema,
            child_strtgy,
            linearity,
            scalar_tensor_idx=(
                _FUSED_OP_SCALAR_IDX if op_schema.op in fused_ops else None
            ),
        )
        list_strategy.append(pointwise_strategy)
    return TupleStrategy(list_strategy)


def list_linear_pointwise_strategy(op_schema: OpSchema) -> StrategyType:
    """
    for each list op stratgy that supports linearity
    """
    return list_pointwise_strategy(op_schema, linearity=True)


for op in for_each_ops:
    register_op_strategy(op, schema_info=RuntimeSchemaInfo(needs_pytree=True))(
        list_pointwise_strategy
    )

for op in for_each_linearity_ops:
    register_op_strategy(op, schema_info=RuntimeSchemaInfo(needs_pytree=True))(
        list_linear_pointwise_strategy
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


# The state_steps arg of fused adam / adamw is a Replicate scalar tensor, which will be put on
# the compute_mesh of an op across all parameter groups, even when not all parameter groups
# are on the same device mesh. This idx will help avoid hitting exceptions or unnecessary
# redistribute during sharding propagation.
_FUSED_OP_SCALAR_IDX = 5

for op in fused_ops:
    register_op_strategy(op, schema_info=RuntimeSchemaInfo(needs_pytree=True))(
        list_pointwise_strategy
    )
