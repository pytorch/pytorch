# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import List, Tuple

import torch

from torch.distributed._tensor.op_schema import (
    _is_inplace_op,
    _is_out_variant_op,
    OpSchema,
    OpStrategy,
    PlacementStrategy,
    RuntimeSchemaInfo,
    StrategyType,
    TupleStrategy,
)

from torch.distributed._tensor.ops.utils import (
    generate_redistribute_costs,
    infer_broadcast_dims_map,
    is_tensor_partial,
    map_placements_after_broadcast,
    normalize_dim,
    register_op_strategy,
)
from torch.distributed._tensor.placement_types import (
    _Partial,
    DTensorSpec,
    Placement,
    Replicate,
    Shard,
)
from torch.distributed.device_mesh import DeviceMesh


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


linear_pointwise_ops = [
    aten.div.Scalar,  # this op is linear on the first argument, and the second argument is scalar, so it fits as a linear op.
    aten.to.dtype,
    aten.add.Tensor,
]


pointwise_ops = [
    # please keep the entries below alphabetically sorted
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
    aten.add.out,
    aten.add_.Scalar,
    aten.add_.Tensor,
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
    aten.asinh.default,
    aten.asinh.out,
    aten.asinh_.default,
    aten.atan.default,
    aten.atan.out,
    aten.atan2.default,
    aten.atan2.out,
    aten.atan2_.default,
    aten.atan_.default,
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
    aten.ceil.default,
    aten.ceil.out,
    aten.ceil_.default,
    aten.clamp.default,
    aten.clamp.out,
    aten.clamp_.default,
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
    aten.deg2rad.default,
    aten.deg2rad.out,
    aten.deg2rad_.default,
    aten.digamma.default,
    aten.digamma.out,
    aten.digamma_.default,
    aten.div.Tensor,
    aten.div.Tensor_mode,
    aten.div.out,
    aten.div.out_mode,
    aten.div_.Tensor,
    aten.div_.Tensor_mode,
    aten.eq.Tensor,
    aten.eq.Tensor_out,
    aten.eq.Scalar,
    aten.eq.Scalar_out,
    aten.erf.default,
    aten.erf.out,
    aten.erf_.default,
    aten.erfc.default,
    aten.erfc.out,
    aten.erfc_.default,
    aten.erfinv.default,
    aten.erfinv.out,
    aten.erfinv_.default,
    aten.exp.default,
    aten.exp.out,
    aten.exp2.default,
    aten.exp2.out,
    aten.exp2_.default,
    aten.exp_.default,
    aten.expm1.default,
    aten.expm1.out,
    aten.expm1_.default,
    aten.float_power.Scalar,
    aten.float_power.Scalar_out,
    aten.float_power.Tensor_Scalar,
    aten.float_power.Tensor_Scalar_out,
    aten.float_power.Tensor_Tensor,
    aten.float_power.Tensor_Tensor_out,
    aten.float_power_.Scalar,
    aten.float_power_.Tensor,
    aten.floor.default,
    aten.floor.out,
    aten.floor_.default,
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
    aten.isnan.default,
    aten.ldexp.default,
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
    aten.logaddexp.default,
    aten.logaddexp.out,
    aten.logaddexp2.default,
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
    aten.logit.out,
    aten.logit_.default,
    aten.masked_fill.Scalar,
    aten.mul.Scalar,
    aten.mul.Tensor,
    aten.mul.out,
    aten.mul_.Scalar,
    aten.mul_.Tensor,
    aten.mvlgamma.default,
    aten.mvlgamma.out,
    aten.mvlgamma_.default,
    aten.native_dropout_backward.default,
    aten.native_dropout_backward.out,
    aten.nan_to_num.default,
    aten.nan_to_num.out,
    aten.nan_to_num_.default,
    aten.ne.Scalar,
    aten.neg.default,
    aten.neg.out,
    aten.neg_.default,
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
    aten.rad2deg.default,
    aten.rad2deg.out,
    aten.rad2deg_.default,
    aten.relu.default,
    aten.relu_.default,
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
    aten.rsqrt.out,
    aten.rsqrt_.default,
    aten.rsub.Scalar,
    aten.sgn.default,
    aten.sgn.out,
    aten.sgn_.default,
    aten.sigmoid.default,
    aten.sigmoid.out,
    aten.sigmoid_.default,
    aten.sign.default,
    aten.sign.out,
    aten.sign_.default,
    aten.signbit.default,
    aten.signbit.out,
    aten.sin.default,
    aten.sin.out,
    aten.sin_.default,
    aten.sinc.default,
    aten.sinc.out,
    aten.sinc_.default,
    aten.sinh.default,
    aten.sinh.out,
    aten.sinh_.default,
    aten.sqrt.default,
    aten.sqrt.out,
    aten.sqrt_.default,
    aten.square.default,
    aten.square.out,
    aten.square_.default,
    aten.sub.Scalar,
    aten.sub.Tensor,
    aten.sub.out,
    aten.sub_.Scalar,
    aten.sub_.Tensor,
    aten.tan.default,
    aten.tan.out,
    aten.tan_.default,
    aten.tanh.default,
    aten.tanh.out,
    aten.tanh_.default,
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
    aten.tanh_backward.default,
    aten.threshold_backward.default,
]


def pointwise_strategy(
    mesh: DeviceMesh, op_schema: OpSchema, linearity: bool = False
) -> StrategyType:
    max_shards_strategy_index = -1
    max_shards = -1
    # handle broadcasting
    common_shape = torch.broadcast_shapes(
        *[
            arg.output_shape
            for arg in op_schema.args_schema
            if isinstance(arg, OpStrategy)
        ]
    )

    if _is_inplace_op(op_schema.op):
        # inplace op should follow the first arg strategy
        followed_strategy = op_schema.args_schema[0]
    elif _is_out_variant_op(op_schema.op):
        # out variant op should follow the out kwarg strategy
        followed_strategy = op_schema.kwargs_schema["out"]
    else:
        # normal pointwise op, we choose to follow the arg with
        # the max shards in case operands needs reshard
        for idx, arg_strategy in enumerate(op_schema.args_schema):
            if not isinstance(arg_strategy, OpStrategy):
                continue

            arg_max_shards = arg_strategy.max_num_shards()
            if arg_max_shards > max_shards:
                max_shards_strategy_index = idx
                max_shards = arg_max_shards

        followed_strategy = op_schema.args_schema[max_shards_strategy_index]
        assert isinstance(followed_strategy, OpStrategy)
        follow_operand_dims_map = infer_broadcast_dims_map(
            common_shape, followed_strategy.output_shape
        )

    assert isinstance(
        followed_strategy, OpStrategy
    ), f"no strategy to follow for {op_schema}!"
    pointwise_strategy = OpStrategy([])

    for placement_strategy in followed_strategy.strategies:
        spec_to_follow = placement_strategy.output_spec
        out_placements: List[Placement] = []

        for placement in spec_to_follow.placements:
            if isinstance(placement, Shard):
                shard_dim = normalize_dim(placement.dim, len(spec_to_follow.shape))
                common_ndim = len(common_shape)
                new_shard_dim = common_ndim - len(spec_to_follow.shape) + shard_dim
                out_placements.append(Shard(new_shard_dim))
            elif isinstance(placement, _Partial) and not linearity:
                # clear the partial placemnet if op does not support linearity
                # by default we just replicate the partial, need to see if this
                # is optimal for all cases
                out_placements.append(Replicate())
            else:
                out_placements.append(placement)

        input_specs = []
        redistribute_costs: List[List[float]] = []
        for idx, input_arg in enumerate(op_schema.args_schema):
            if isinstance(input_arg, OpStrategy):
                # every arg follow the out_placements, but need to handle broadcasting
                input_arg_spec = input_arg.strategies[0].output_spec
                input_arg_dims_map = infer_broadcast_dims_map(
                    common_shape, input_arg_spec.shape
                )
                input_target_placements = map_placements_after_broadcast(
                    tuple(out_placements),
                    common_shape,
                    input_arg_dims_map,
                )
                input_arg_target_spec = DTensorSpec(
                    mesh=mesh,
                    placements=input_target_placements,
                    tensor_meta=input_arg_spec.tensor_meta,
                )
                input_specs.append(input_arg_target_spec)
                redistribute_costs.append(
                    generate_redistribute_costs(input_arg, input_arg_target_spec)
                )

        pointwise_strategy.strategies.append(
            PlacementStrategy(
                output_specs=DTensorSpec(
                    mesh=mesh,
                    placements=tuple(out_placements),
                ),
                input_specs=input_specs,
                redistribute_cost=redistribute_costs,
            )
        )
    return pointwise_strategy


def linear_pointwise_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    """
    Linear pointwise operators can propagate pending reductions.
    For example, c = add(a, b); if a is pending sum, then c will be
    pending sum as well without any communication overhead.
    """
    return pointwise_strategy(mesh, op_schema, linearity=True)


for op in linear_pointwise_ops:
    register_op_strategy(op, schema_info=RuntimeSchemaInfo(static_kwargkey=["out"]))(
        linear_pointwise_strategy
    )

for op in pointwise_ops:
    register_op_strategy(op, schema_info=RuntimeSchemaInfo(static_kwargkey=["out"]))(
        pointwise_strategy
    )


# TODO: add all for_each ops
for_each_ops = [
    aten._foreach_addcdiv_.Scalar,
    aten._foreach_addcdiv_.ScalarList,
    aten._foreach_addcdiv_.Tensor,
    aten._foreach_addcmul.Scalar,
    aten._foreach_addcmul_.Scalar,
    aten._foreach_addcmul_.ScalarList,
    aten._foreach_addcmul_.Tensor,
    aten._foreach_div_.List,
    aten._foreach_div_.ScalarList,
    aten._foreach_lerp_.Scalar,
    aten._foreach_maximum_.List,
    aten._foreach_mul.Scalar,
    aten._foreach_mul.List,
    aten._foreach_mul_.Scalar,
    aten._foreach_mul_.ScalarList,
    aten._foreach_mul_.List,
    aten._foreach_neg.default,
    aten._foreach_neg_.default,
    aten._foreach_reciprocal_.default,
    aten._foreach_sub_.Scalar,
    aten._foreach_sqrt.default,
    aten._foreach_sqrt_.default,
    aten._foreach_zero_.default,
]

for_each_linearity_ops = [
    aten._foreach_add.Scalar,
    aten._foreach_add_.Scalar,
    aten._foreach_add_.ScalarList,
    aten._foreach_add.List,
    aten._foreach_add_.List,
]


def foreach_list_strategy(
    mesh: DeviceMesh, op_schema: OpSchema, linearity: bool = False
) -> StrategyType:
    """
    for each list op stratgy mostly follow the same logic as pointwise strategy
    except that it handles list of tensors instead, and normally we don't need to
    handle implicit broadcasting
    """

    def args_tuple_strategies(args_schema: Tuple[object, ...]) -> List[TupleStrategy]:
        first_arg = args_schema[0]
        assert isinstance(first_arg, TupleStrategy)
        strategy_len = len(first_arg.childs)
        tuple_strategies: List[TupleStrategy] = []
        for arg in args_schema:
            if isinstance(arg, TupleStrategy):
                # every tuple strategy should have the same length
                assert len(arg.childs) == strategy_len
                tuple_strategies.append(arg)
            elif isinstance(arg, OpStrategy):
                raise RuntimeError("foreach list op only supports tuple strategy!")
        return tuple_strategies

    args_strategies = args_tuple_strategies(op_schema.args_schema)

    # foreach op should follow the first arg strategy
    follow_strategy = args_strategies[0]

    foreach_strategy_list = []
    for idx, child_strtgy in enumerate(follow_strategy.childs):
        assert isinstance(child_strtgy, OpStrategy)

        strategies = []
        for strtgy in child_strtgy.strategies:
            spec_to_follow = strtgy.output_spec
            if not linearity:
                assert not is_tensor_partial(
                    spec_to_follow
                ), f"{op_schema.op} does not support operation on partial tensor!"

            redistribute_costs: List[List[float]] = []

            for arg_strtgy in args_strategies:
                child_strtgy = arg_strtgy.childs[idx]
                assert isinstance(child_strtgy, OpStrategy)
                redistribute_costs.append(
                    generate_redistribute_costs(child_strtgy, spec_to_follow)
                )
            strategies.append(
                PlacementStrategy(
                    output_specs=spec_to_follow, redistribute_cost=redistribute_costs
                )
            )

        foreach_strategy_list.append(OpStrategy(strategies))

    tup_strategy = TupleStrategy(foreach_strategy_list)
    return tup_strategy


def foreach_list_linear_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    """
    for each list op stratgy that supports linearity
    """
    return foreach_list_strategy(mesh, op_schema, linearity=True)


for op in for_each_ops:
    register_op_strategy(op, schema_info=RuntimeSchemaInfo(needs_pytree=True))(
        foreach_list_strategy
    )

for op in for_each_linearity_ops:
    register_op_strategy(op, schema_info=RuntimeSchemaInfo(needs_pytree=True))(
        foreach_list_linear_strategy
    )
