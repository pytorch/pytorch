# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor
import itertools
from typing import cast, List

import torch
from torch.distributed._tensor.op_schema import (
    OpSchema,
    OpStrategy,
    PlacementStrategy,
    StrategyType,
)
from torch.distributed._tensor.ops.utils import (
    generate_redistribute_costs,
    is_tensor_shardable,
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


@register_op_strategy(aten.embedding.default)
def embedding_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    """
    This strategy handles embedding op. We have two possible embedding shardings:
    rowwise and colwise
    # TODO: implement rowwise sharding
    """
    weight_strategy = cast(OpStrategy, op_schema.args_schema[0])
    indices_strategy = cast(OpStrategy, op_schema.args_schema[1])

    weight_shape = weight_strategy.output_shape
    indices_shape = indices_strategy.output_shape
    output_emd_dim = len(indices_shape)

    # guard rowwise sharding not implemented for now
    weight_spec = weight_strategy.strategies[0].output_spec
    if any(placement.is_shard(0) for placement in weight_spec.placements):
        raise NotImplementedError(
            "DTensor does not support row-wise sharded embedding operation yet!"
        )

    all_mesh_dim_strategies = []

    for mesh_dim in range(mesh.ndim):
        single_mesh_dim_strategies = []

        # placement list stores placements of [output, weight, input_indices]
        # first we always have replicate all for inputs and output
        all_replicate: List[Placement] = [Replicate()] * 3
        single_mesh_dim_strategies.append(all_replicate)

        # colwise sharding, output shard on last dim, weight shard on dim 1, input replicate
        colwise_sharding = [Shard(output_emd_dim), Shard(1), Replicate()]
        single_mesh_dim_strategies.append(colwise_sharding)

        # batch dim sharding, weight replicated, input can shard on any dim, output follows input
        for input_dim in range(len(indices_shape)):
            batch_sharding = [Shard(input_dim), Replicate(), Shard(input_dim)]
            single_mesh_dim_strategies.append(batch_sharding)

        all_mesh_dim_strategies.append(single_mesh_dim_strategies)

    strategy_combs = itertools.product(*all_mesh_dim_strategies)

    all_strategies = []
    for strategy_comb in strategy_combs:
        spec_list = []
        for specs in zip(*strategy_comb):
            spec_list.append(DTensorSpec(mesh, tuple(specs)))

        if is_tensor_shardable(weight_shape, spec_list[1]) and is_tensor_shardable(
            indices_shape, spec_list[2]
        ):
            # only add to the strategy list when both weight and indices are shardable
            weight_spec, indices_spec = spec_list[1:]
            redistribute_cost = [
                generate_redistribute_costs(weight_strategy, weight_spec),
                generate_redistribute_costs(indices_strategy, indices_spec),
            ]
            strat = PlacementStrategy(
                output_specs=spec_list[0],
                input_specs=spec_list[1:],
                redistribute_cost=redistribute_cost,
            )
            all_strategies.append(strat)

    return OpStrategy(all_strategies)


@register_op_strategy(aten.embedding_dense_backward.default)
def embedding_dense_backward_strategy(
    mesh: DeviceMesh, op_schema: OpSchema
) -> StrategyType:
    """
    This strategy handles embedding op. We have two possible embedding shardings:
    rowwise and colwise
    # TODO: implement rowwise sharding backward
    """
    grad_out_strategy = cast(OpStrategy, op_schema.args_schema[0])
    indices_strategy = cast(OpStrategy, op_schema.args_schema[1])

    grad_out_shape = grad_out_strategy.output_shape
    indices_shape = indices_strategy.output_shape
    grad_out_ndim = len(grad_out_shape)

    all_mesh_dim_strategies = []

    for mesh_dim in range(mesh.ndim):
        single_mesh_dim_strategies = []

        # placement list stores placements of [output, weight, input_indices]
        # first we always have replicate all for inputs and output
        all_replicate: List[Placement] = [Replicate()] * 3
        single_mesh_dim_strategies.append(all_replicate)

        # colwise sharding backward, grad_out shard on last dim, input replicate,
        # weight grad shard colwise
        colwise_sharding = [Shard(1), Shard(grad_out_ndim - 1), Replicate()]
        single_mesh_dim_strategies.append(colwise_sharding)

        # batch dim sharding, weight replicated, grad_out/input have same sharding
        # that can shard on any dim, weight grad partial
        for input_dim in range(len(indices_shape)):
            batch_sharding = [_Partial(), Shard(input_dim), Shard(input_dim)]
            single_mesh_dim_strategies.append(batch_sharding)

        # grad_out partial, input replicate, weight grad keep partial
        partial_sharding = [_Partial(), _Partial(), Replicate()]
        single_mesh_dim_strategies.append(partial_sharding)

        all_mesh_dim_strategies.append(single_mesh_dim_strategies)

    strategy_combs = itertools.product(*all_mesh_dim_strategies)

    all_strategies = []
    for strategy_comb in strategy_combs:
        spec_list = []
        for specs in zip(*strategy_comb):
            spec_list.append(DTensorSpec(mesh, tuple(specs)))

        if is_tensor_shardable(grad_out_shape, spec_list[1]) and is_tensor_shardable(
            indices_shape, spec_list[2]
        ):
            # only add to the strategy list when both grad_out and indices are shardable
            grad_out_spec, indices_spec = spec_list[1:]
            redistribute_cost = [
                generate_redistribute_costs(grad_out_strategy, grad_out_spec),
                generate_redistribute_costs(indices_strategy, indices_spec),
            ]
            strat = PlacementStrategy(
                output_specs=spec_list[0],
                input_specs=spec_list[1:],
                redistribute_cost=redistribute_cost,
            )
            all_strategies.append(strat)

    return OpStrategy(all_strategies)
