# Copyright (c) Meta Platforms, Inc. and affiliates

import torch
from torch.distributed._tensor.op_schema import (
    _is_inplace_op,
    _is_out_variant_op,
    OpSchema,
    OpStrategy,
    PlacementStrategy,
    StrategyType,
)

from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed._tensor.ops.common_rules import (
    linear_pointwise_rule,
    pointwise_rule,
)
from torch.distributed._tensor.ops.utils import register_op_strategy, register_prop_rule
from torch.distributed._tensor.placement_types import (
    _Partial,
    DTensorSpec,
    Placement,
    Replicate,
    Shard,
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


linear_pointwise_ops = [
    aten.div.Scalar,  # this op is linear on the first argument, and the second argument is scalar, so it fits as a linear op.
    aten.to.dtype,
    aten.add.Tensor,
]


pointwise_ops = [
    # please keep the entries below alphabetically sorted
    aten.abs.default,
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


for op in linear_pointwise_ops:
    register_prop_rule(op)(linear_pointwise_rule)


# for op in pointwise_ops:
#     register_prop_rule(op)(pointwise_rule)

def pointwise_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    follow_placement_strategy_indices = []
    followed_strategy: Optional[OpStrategy] = None
    if _is_inplace_op(op_schema.op):
        # inplace op should follow the first arg strategy
        followed_strategy = op_schema.args_schema[0]
    elif _is_out_variant_op(op_schema.op):
        # out variant op should follow the last kwarg strategy
        followed_strategy = op_schema.kwargs_schema[-1]
    else:
        # normal pointwise op, we choose to follow the arg with
        # the max shards in case operands needs reshard
        # args_max_num_shards = [
        #     arg_strategy.max_num_shards()
        #     for arg_strategy in op_schema.args_schema
        #     if isinstance(arg_strategy, OpStrategy)
        # ]
        # max_shards_index = args_max_num_shards.index(max(args_max_num_shards))
        # followed_strategy = op_schema.args_schema[max_shards_index]

        max_shards_strategy_index = -1
        max_shards = -1

        for idx, arg_strategy in enumerate(op_schema.args_schema):
            if not isinstance(arg_strategy, OpStrategy):
                continue

            arg_max_shards = arg_strategy.max_num_shards()
            if arg_max_shards > max_shards:
                # since pointwise op can have broadcasting semantics,
                # we need to ensure the strategy to follow can shard
                # all the input args
                follow_indices_for_current_strategy = []
                strategy_shardable = False
                for strat_idx, placement_strategy in enumerate(arg_strategy.strategies):
                    all_args_shardable = all(placement_strategy.shardable_shape(arg.output_shape) for arg in op_schema.args_schema if isinstance(arg, OpStrategy))
                    follow_indices_for_current_strategy.append(strat_idx)
                    if all_args_shardable and not strategy_shardable:
                        # if there's a placement strategy can shard all args
                        # this op strategy can be selected as the candidate to follow
                        max_shards_strategy_index = idx
                        max_shards = arg_max_shards
                        strategy_shardable = True
                if strategy_shardable:
                    # record the indices of strategies need to follow
                    follow_placement_strategy_indices = follow_indices_for_current_strategy

        if max_shards_strategy_index < 0:
            # there's no strategy from inputs can be followed, we fallback and generate
            # a default replicate strategy
            followed_strategy = OpStrategy([
                PlacementStrategy(
                    output_spec= DTensorSpec(mesh=mesh, placements=[Replicate()]),
                )
            ])
            print(f">>>> falling back to default replication strategy")
        else:
            followed_strategy = op_schema.args_schema[max_shards_strategy_index]
        if mesh.get_rank() == 0:
            print(f">>>>>> op op: {op_schema.op}, chose {max_shards_strategy_index} as select, strategy to follow: {followed_strategy}")
            for arg in op_schema.args_schema:
                if isinstance(arg, OpStrategy):
                    print(f" arg: {arg}, shape: {arg.strategies[0].output_spec.shape}")

    assert isinstance(followed_strategy, OpStrategy)
    if not follow_placement_strategy_indices:
        follow_placement_strategy_indices = list(range(len(followed_strategy.strategies)))
    pointwise_strategy = OpStrategy([])

    for placement_strategy_idx in follow_placement_strategy_indices:
        placement_strategy_to_follow = followed_strategy.strategies[placement_strategy_idx]
        placements_to_follow = placement_strategy_to_follow.output_spec.placements
        input_specs = []
        for input_arg in op_schema.args_schema:
            if isinstance(input_arg, OpStrategy):
                input_specs.append(
                    DTensorSpec(
                        mesh=mesh,
                        placements=placements_to_follow,
                        shape=input_arg.output_shape,
                        stride=input_arg.output_stride
                    )
                )
        pointwise_strategy.strategies.append(
            PlacementStrategy(
                output_spec=DTensorSpec(
                    mesh=mesh,
                    placements=placements_to_follow,
                ),
                input_specs=input_specs
            )
        )
    print(f">>>>>>> final pointwise strategy generated: {pointwise_strategy}.")
    return pointwise_strategy


for op in pointwise_ops:
    register_op_strategy(op)(pointwise_strategy)
