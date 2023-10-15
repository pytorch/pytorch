# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor
from typing import List, Tuple

import torch

from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed._tensor.op_schema import OpSchema, OpStrategy, OutputSharding
from torch.distributed._tensor.ops.basic_strategy import gen_einsum_strategies
from torch.distributed._tensor.ops.common_rules import einop_rule
from torch.distributed._tensor.ops.utils import (
    generate_redistribute_costs,
    is_tensor_shardable,
    normalize_dim,
    register_op_strategy,
    register_prop_rule,
)
from torch.distributed._tensor.placement_types import (
    _Partial,
    DTensorSpec,
    Placement,
    Replicate,
    Shard,
)

aten = torch.ops.aten


@register_prop_rule(aten.t.default)
def transpose_rule(op_schema: OpSchema) -> OutputSharding:
    return einop_rule("ij->ji", op_schema, linearity=True)


def _infer_broadcast_dims_map(
    common_shape: torch.Size, input_shape: torch.Size
) -> List[int]:
    # infer the broadcast dims map, where it maps from the common shape dim to the input shape dim
    # this is aligned with the broadcast semantics
    common_ndim = len(common_shape)
    input_ndim = len(input_shape)
    broadcast_dims_map = [-1] * common_ndim
    for idx in range(-1, -1 - input_ndim, -1):
        if input_shape[idx] == common_shape[idx]:
            broadcast_dims_map[common_ndim + idx] = input_ndim + idx
    return broadcast_dims_map


def map_placements_after_broadcast(
    placements: Tuple[Placement, ...],
    shape: torch.Size,
    broadcast_dims_map: List[int],
) -> Tuple[Placement, ...]:
    """
    Map each placement based on the output shape after broadcast.
    """
    new_placements: List[Placement] = []
    for placement in placements:
        if isinstance(placement, (Replicate, _Partial)):
            new_placements.append(placement)
        else:
            assert isinstance(placement, Shard)
            shard_dim = normalize_dim(placement.dim, len(shape))
            new_shard_dim = broadcast_dims_map[shard_dim]
            if new_shard_dim != -1:
                # there's a map from the common shape shard dim to
                # the input shape shard dim before broadcasting,
                # use that instead
                new_placements.append(Shard(new_shard_dim))
            else:
                # there's no map between common shape shard dim and
                # the input shape shard dim before broadcasting,
                # in this case it means implicit broadcasting happen
                # in this dim, so we can just mark it as replicate
                # and implict broadcast will broadcast automatically
                # to the sharded shape
                new_placements.append(Replicate())

    return tuple(new_placements)


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
        broadcast_dims_map = _infer_broadcast_dims_map(mm_out_shape, self_shape)
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
