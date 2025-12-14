# Copyright (c) Meta Platforms, Inc. and affiliates
from collections.abc import Sequence
from typing import cast

import torch
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpSpec,
    OpStrategy,
    RuntimeSchemaInfo,
    StrategyType,
    TupleStrategy,
)
from torch.distributed.tensor._ops.registration import register_op_strategy
from torch.distributed.tensor._ops.utils import (
    generate_redistribute_costs,
    infer_broadcast_dims_map,
    map_placements_after_broadcast,
    normalize_dim,
)
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.utils._typing_utils import not_none


aten = torch.ops.aten


# check torch.Tag.pointwise dynamically.
# Blocklist for pointwise ops that have the tag but need special handling

POINTWISE_OPS_BLOCKLIST = {
    # TODO: Populate this as we discover ops that need special handling
    # Example format:
    # aten.special_op.default,
}


def is_pointwise_op(op: torch._ops.OpOverload) -> bool:
    return torch.Tag.pointwise in op.tags and op not in POINTWISE_OPS_BLOCKLIST


# the linear pointwise ops map, key is op, value is the type of linearity
linear_pointwise_ops = {
    aten.to.dtype: 0,
    aten.add.Tensor: 1,
    aten.add_.Tensor: 1,
    aten.div.Scalar: 0,
    aten.div_.Scalar: 0,
    aten.mul.Scalar: 0,
    aten.mul_.Scalar: 0,
    aten.mul.Tensor: 2,
    aten.mul_.Tensor: 2,
    aten.copy_.default: 1,
}

# Ops that preserve specific Partial types through the operation.
# For example, torch.maximum preserves Partial("max") because
# max(max(a), max(b)) == max(a, b).
partial_preserving_ops: dict[torch._ops.OpOverload, str] = {
    aten.maximum.default: "max",
    aten.maximum.out: "max",
    aten.minimum.default: "min",
    aten.minimum.out: "min",
}


def pointwise_strategy(
    op_schema: OpSchema,
    linearity: int = -1,
    preserve_partial: str | None = None,
) -> OpStrategy:
    followed_strategy_index = -1
    max_shards = -1
    max_ndim = -1

    if op_schema.is_inplace_op():
        # inplace op should follow the first arg strategy
        followed_strategy = op_schema.args_schema[0]
        followed_strategy_index = 0
    elif op_schema.is_out_variant_op():
        # out variant op should follow the out kwarg strategy
        followed_strategy = op_schema.kwargs_schema["out"]
        # out variant is technically a kwarg for the strategy to follow so it does not
        # have an "index", we set it to a reasonably large number just to indicate it's
        # not a valid index
        followed_strategy_index = 100
    else:
        # normal pointwise op, we choose to follow the arg with
        # the max shards in case operands needs reshard
        # in case of multiple operands with max shard, we take
        # the one with the max number of dimensions
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
        op_schema.args_schema,
        followed_strategy,
        followed_strategy_index,
        linearity,
        preserve_partial=preserve_partial,
    )


def linear_pointwise_strategy(op_schema: OpSchema) -> StrategyType:
    """
    Linear pointwise operators can propagate pending reductions.
    For example, c = add(a, b); if a is pending sum, then c will be
    pending sum as well without any communication overhead.

    Note that:
    1. Only unary and binary operations are supported, out variant
      ops are not supported.
    2. There're multiple types of linearity, refer to the doc of
      common_pointwise_strategy for more details.
    """
    linearity_type = linear_pointwise_ops.get(op_schema.op, -1)
    return pointwise_strategy(op_schema, linearity=linearity_type)


def partial_preserving_pointwise_strategy(op_schema: OpSchema) -> StrategyType:
    """
    Strategy for pointwise ops that preserve specific Partial types.

    For example, torch.maximum preserves Partial("max") placements because
    max(max(a), max(b)) == max(a, b). Similarly, torch.minimum preserves
    Partial("min") placements.
    """
    preserve_partial = partial_preserving_ops.get(op_schema.op)
    return pointwise_strategy(op_schema, preserve_partial=preserve_partial)


def common_pointwise_strategy(
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
                # Check if this partial type should be preserved
                if preserve_partial is not None and placement.is_partial(
                    preserve_partial
                ):
                    out_placements.append(placement)
                # note that only partial-sum and partial-avg are supported for linearity
                elif linearity >= 0 and (
                    placement.is_partial("sum") or placement.is_partial("avg")
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


for op in linear_pointwise_ops:
    register_op_strategy(op, schema_info=RuntimeSchemaInfo(static_kwargkey=["out"]))(
        linear_pointwise_strategy
    )


for op in partial_preserving_ops:
    register_op_strategy(op, schema_info=RuntimeSchemaInfo(static_kwargkey=["out"]))(
        partial_preserving_pointwise_strategy
    )


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
