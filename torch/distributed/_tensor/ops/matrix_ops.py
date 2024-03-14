# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor
import itertools
from typing import List, Optional

import torch
from torch.distributed._tensor.op_schema import (
    OpSchema,
    OpStrategy,
    OutputSharding,
    PlacementStrategy,
)
from torch.distributed._tensor.ops.basic_strategy import gen_einsum_strategies
from torch.distributed._tensor.ops.common_rules import einop_rule
from torch.distributed._tensor.ops.utils import (
    generate_redistribute_costs,
    infer_broadcast_dims_map,
    is_tensor_shardable,
    map_placements_after_broadcast,
    register_op_strategy,
    register_prop_rule,
)
from torch.distributed._tensor.placement_types import (
    DTensorSpec,
    Placement,
    Replicate,
    Shard,
)

from torch.distributed.device_mesh import DeviceMesh

aten = torch.ops.aten


@register_prop_rule(aten.t.default)
def transpose_rule(op_schema: OpSchema) -> OutputSharding:
    return einop_rule("ij->ji", op_schema, linearity=True)


def _mm_like_strategy(
    mm_equation: str, mesh: DeviceMesh, op_schema: OpSchema
) -> OpStrategy:
    self_strategy, mat2_strategy = op_schema.args_schema
    assert isinstance(self_strategy, OpStrategy)
    assert isinstance(mat2_strategy, OpStrategy)
    # generate all possible strategies for mm
    mm_strategy = gen_einsum_strategies(mm_equation, mesh)
    # filter out invalid strategies and associate costs
    strategies = mm_strategy.strategies
    filtered_strategies = []
    for strtg in strategies:
        assert strtg.input_specs is not None
        self_spec = strtg.input_specs[0]
        mat2_spec = strtg.input_specs[1]
        if is_tensor_shardable(
            self_strategy.output_shape, self_spec
        ) and is_tensor_shardable(mat2_strategy.output_shape, mat2_spec):
            redistribute_cost = [
                generate_redistribute_costs(self_strategy, self_spec),
                generate_redistribute_costs(mat2_strategy, mat2_spec),
            ]
            strtg.redistribute_cost = redistribute_cost
            filtered_strategies.append(strtg)

    mm_strategy.strategies = filtered_strategies

    return mm_strategy


def _addmm_like_strategy(
    mm_equation: str, mesh: DeviceMesh, op_schema: OpSchema
) -> OpStrategy:
    self_strategy, mat1_strategy, mat2_strategy = op_schema.args_schema
    assert isinstance(self_strategy, OpStrategy)
    assert isinstance(mat1_strategy, OpStrategy)
    assert isinstance(mat2_strategy, OpStrategy)
    self_shape = self_strategy.output_shape
    mm_out_shape = torch.Size(
        [
            mat2_strategy.output_shape[-1]
            if i == len(mat1_strategy.output_shape) - 1
            else dim_size
            for i, dim_size in enumerate(mat1_strategy.output_shape)
        ]
    )
    # generate all possible strategies for mm
    mm_strategy = gen_einsum_strategies(mm_equation, mesh)
    # filter out invalid strategies and associate costs
    strategies = mm_strategy.strategies
    filtered_strategies = []
    for strtg in strategies:
        # construct new strategy by consider the self arg
        assert strtg.input_specs is not None
        mat1_spec = strtg.input_specs[0]
        mat2_spec = strtg.input_specs[1]
        out_spec = strtg.output_spec

        # self arg's spec should follow the output of mm, but need
        # to consider broadcast for the self arg
        broadcast_dims_map = infer_broadcast_dims_map(mm_out_shape, self_shape)
        self_placements = map_placements_after_broadcast(
            out_spec.placements, mm_out_shape, broadcast_dims_map
        )
        self_spec = DTensorSpec(mesh=mesh, placements=self_placements)

        if is_tensor_shardable(
            mat1_strategy.output_shape, mat1_spec
        ) and is_tensor_shardable(mat2_strategy.output_shape, mat2_spec):
            # update input specs with new self spec
            strtg.input_specs = (self_spec, mat1_spec, mat2_spec)

            # associate costs
            redistribute_cost = [
                generate_redistribute_costs(self_strategy, self_spec),
                generate_redistribute_costs(mat1_strategy, mat1_spec),
                generate_redistribute_costs(mat2_strategy, mat2_spec),
            ]
            strtg.redistribute_cost = redistribute_cost
            filtered_strategies.append(strtg)

    mm_strategy.strategies = filtered_strategies

    return mm_strategy


@register_op_strategy(aten.mm.default)
def mm_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    return _mm_like_strategy("mk,kn->mn", mesh, op_schema)


@register_op_strategy(aten.addmm.default)
def addmm_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    return _addmm_like_strategy("mk,kn->mn", mesh, op_schema)


@register_op_strategy(aten.bmm.default)
def bmm_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    return _mm_like_strategy("bmk,bkn->bmn", mesh, op_schema)


@register_op_strategy(aten.baddbmm.default)
def baddmm_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    return _addmm_like_strategy("bmk,bkn->bmn", mesh, op_schema)


@register_op_strategy(aten._scaled_dot_product_flash_attention.default)
def scaled_dot_product_attention_strategy(
    mesh: DeviceMesh, op_schema: OpSchema
) -> OpStrategy:
    # NOTE: currently we only support some simple strategies to support tensor parallelism
    # TODO: sdpa might be a good candidate for us to explore decomposed sharding propagation
    # as it involves: matmul, pointwise, reduction ops together.
    return_debug_mask = len(op_schema.args_schema) >= 6 and op_schema.args_schema[5]
    q_input_strategy = op_schema.args_schema[0]
    assert isinstance(q_input_strategy, OpStrategy)
    # q/k/v have the same shape
    qkv_shape = q_input_strategy.output_shape

    all_mesh_dim_strategies = []

    for mesh_dim in range(mesh.ndim):
        single_mesh_dim_strategies = []

        # placement list stores placements of [outputs, inputs]
        # in the spda case, we have 3 valid tensor outputs and 3 tensor inputs
        # first we can always accept full replication for inputs and output
        all_replicate: List[Placement] = [Replicate()] * 6
        single_mesh_dim_strategies.append(all_replicate)

        # second we can accept the sharding pattern of tensor parallelism, which
        # shard on the num of head dim
        qkv_sharding = Shard(1)  # num head dim
        output_sharding = Shard(1)  # num head dim
        logsumexp_sharding = Shard(1)  # num head dim
        if return_debug_mask:
            debug_attn_mask_sharding: Placement = Shard(1)  # num head dim
        else:
            # empty debug mask, replicated
            debug_attn_mask_sharding = Replicate()

        num_heads_dim_sharding = [
            output_sharding,
            logsumexp_sharding,
            debug_attn_mask_sharding,
            qkv_sharding,
            qkv_sharding,
            qkv_sharding,
        ]
        single_mesh_dim_strategies.append(num_heads_dim_sharding)

        all_mesh_dim_strategies.append(single_mesh_dim_strategies)

    strategy_combs = itertools.product(*all_mesh_dim_strategies)

    all_strategies = []
    for strategy_comb in strategy_combs:
        spec_list = []
        for specs in zip(*strategy_comb):
            spec_list.append(DTensorSpec(mesh, tuple(specs)))

        assert len(spec_list) == 6
        input_expected_specs = spec_list[3:]
        output_specs: List[Optional[DTensorSpec]] = list(spec_list[:3])
        # fix up output_specs and fill in None for the int and empty tensor return values
        for i in range(2, 8):
            output_specs.insert(i, None)
        if all(is_tensor_shardable(qkv_shape, spec) for spec in input_expected_specs):
            # only add to the strategy list when all inputs are shardable
            redistribute_cost = []
            for input_idx, spec in enumerate(input_expected_specs):
                qkv_strategy = op_schema.args_schema[input_idx]
                assert isinstance(qkv_strategy, OpStrategy)
                qkv_tensor_meta = qkv_strategy.strategies[0].output_spec.tensor_meta
                spec.tensor_meta = qkv_tensor_meta
                redistribute_cost.append(
                    generate_redistribute_costs(qkv_strategy, spec)
                )

            strat = PlacementStrategy(
                output_specs=tuple(output_specs),
                input_specs=tuple(input_expected_specs),
                redistribute_cost=redistribute_cost,
            )
            all_strategies.append(strat)

    return OpStrategy(all_strategies)
