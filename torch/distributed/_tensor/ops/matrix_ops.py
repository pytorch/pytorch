# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor
import itertools
from typing import List, Optional

import torch
from torch.distributed._tensor._op_schema import OpSchema, OpStrategy, PlacementStrategy
from torch.distributed._tensor.ops.basic_strategy import gen_einsum_strategies
from torch.distributed._tensor.ops.utils import (
    generate_redistribute_costs,
    infer_broadcast_dims_map,
    is_tensor_shardable,
    map_placements_after_broadcast,
    register_op_strategy,
)
from torch.distributed._tensor.placement_types import (
    DTensorSpec,
    Placement,
    Replicate,
    Shard,
)

from torch.distributed.device_mesh import DeviceMesh

aten = torch.ops.aten


@register_op_strategy(aten.t.default)
def transpose_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    self_strategy = op_schema.args_schema[0]
    assert isinstance(self_strategy, OpStrategy)

    transpose_strategies = []
    for input_strategy in self_strategy.strategies:
        input_spec = input_strategy.output_spec
        # follow the input spec but transpose the Shard placements
        output_placements = [
            Shard(1 - p.dim) if isinstance(p, Shard) else p
            for p in input_spec.placements
        ]
        transpose_strategy = PlacementStrategy(
            output_specs=DTensorSpec(
                mesh=input_strategy.output_spec.mesh,
                placements=tuple(output_placements),
            ),
            input_specs=(input_strategy.output_spec,),
        )
        transpose_strategies.append(transpose_strategy)

    return OpStrategy(strategies=transpose_strategies)


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
        if is_tensor_shardable(self_strategy.shape, self_spec) and is_tensor_shardable(
            mat2_strategy.shape, mat2_spec
        ):
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
    self_shape = self_strategy.shape
    mm_out_shape = torch.Size(
        [
            mat2_strategy.shape[-1] if i == len(mat1_strategy.shape) - 1 else dim_size
            for i, dim_size in enumerate(mat1_strategy.shape)
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

        if is_tensor_shardable(mat1_strategy.shape, mat1_spec) and is_tensor_shardable(
            mat2_strategy.shape, mat2_spec
        ):
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
def scaled_dot_product_flash_attention_strategy(
    mesh: DeviceMesh, op_schema: OpSchema
) -> OpStrategy:
    # NOTE: currently we only support some simple strategies to support tensor parallelism
    # TODO: sdpa might be a good candidate for us to explore decomposed sharding propagation
    # as it involves: matmul, pointwise, reduction ops together.
    return_debug_mask = len(op_schema.args_schema) >= 6 and op_schema.args_schema[5]
    q_input_strategy = op_schema.args_schema[0]
    assert isinstance(q_input_strategy, OpStrategy)
    # assuming q/k/v have the same shape
    qkv_shape = q_input_strategy.shape

    all_mesh_dim_strategies = []

    for mesh_dim in range(mesh.ndim):
        single_mesh_dim_strategies = []

        # placement list stores placements of [outputs, inputs]
        # in the spda case, we have 3 valid tensor outputs and 3 tensor inputs
        # first we can always accept full replication for both inputs and outputs
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

        # Context Parallelism: shards on the sequence dim
        single_mesh_dim_strategies.append(
            [
                Shard(2),  # output
                Shard(2),  # logsumexp
                Shard(2),  # debugattn
                Shard(2),  # q
                Shard(2),  # k
                Shard(2),  # v
            ]
        )

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


@register_op_strategy(aten._scaled_dot_product_flash_attention_backward.default)
def scaled_dot_product_flash_attention_backward_strategy(
    mesh: DeviceMesh, op_schema: OpSchema
) -> OpStrategy:
    q_input_strategy = op_schema.args_schema[1]
    assert isinstance(q_input_strategy, OpStrategy)
    # assuming q/k/v have the same shape
    qkv_shape = q_input_strategy.shape

    tensor_input_indices = [
        i
        for i, arg_spec in enumerate(op_schema.args_schema)
        if isinstance(arg_spec, OpStrategy)
    ]
    num_tensor_inputs = len(tensor_input_indices)

    all_mesh_dim_strategies = []

    for mesh_dim in range(mesh.ndim):
        single_mesh_dim_strategies = []

        # placement list stores placements of [outputs, inputs]
        # in the spda backward case, we have 3 tensor outputs and 6 to 10 tensor inputs
        # first we can always accept full replication for both inputs and outputs
        all_replicate: List[Placement] = [Replicate()] * (3 + num_tensor_inputs)

        single_mesh_dim_strategies.append(all_replicate)

        # second we can accept the sharding pattern of tensor parallelism, which
        # shard on the num of head dim
        grad_output_sharding = Shard(1)  # num head dim
        qkv_sharding = Shard(1)  # num head dim
        output_sharding = Shard(1)  # num head dim
        logsumexp_sharding = Shard(1)  # num head dim
        grad_qkv_sharding = Shard(1)  # num head dim

        num_heads_dim_sharding: List[Placement] = [
            grad_qkv_sharding,
            grad_qkv_sharding,
            grad_qkv_sharding,
            grad_output_sharding,
            qkv_sharding,
            qkv_sharding,
            qkv_sharding,
            output_sharding,
            logsumexp_sharding,
        ]
        # accept replicate on the rest tensor inputs, potentially
        # cum_seq_q, cum_seq_k, philox_seed, philox_offset
        # at indices 6, 7, 12, 13, respectively
        num_heads_dim_sharding.extend([Replicate()] * (num_tensor_inputs - 6))
        single_mesh_dim_strategies.append(num_heads_dim_sharding)

        # Context Parallelism: shards on the sequence dim
        seq_dim_sharding: List[Placement] = [
            Shard(2),  # grad_q
            Shard(2),  # grad_k
            Shard(2),  # grad_v
            Shard(2),  # grad_output
            Shard(2),  # q
            Shard(2),  # k
            Shard(2),  # v
            Shard(2),  # output
            Shard(2),  # logsumexp
        ]
        # accept replicate on the rest tensor inputs, potentially
        # cum_seq_q, cum_seq_k, philox_seed, philox_offset
        # at indices 6, 7, 12, 13, respectively
        seq_dim_sharding.extend([Replicate()] * (num_tensor_inputs - 6))
        single_mesh_dim_strategies.append(seq_dim_sharding)

        all_mesh_dim_strategies.append(single_mesh_dim_strategies)

    strategy_combs = itertools.product(*all_mesh_dim_strategies)

    all_strategies = []
    for strategy_comb in strategy_combs:
        spec_list = []
        for specs in zip(*strategy_comb):
            spec_list.append(DTensorSpec(mesh, tuple(specs)))

        assert len(spec_list) == 3 + num_tensor_inputs
        input_expected_specs = spec_list[3:]
        output_specs: List[Optional[DTensorSpec]] = list(spec_list[:3])
        if all(
            is_tensor_shardable(qkv_shape, spec) for spec in input_expected_specs[:6]
        ):
            # only add to the strategy list when all inputs are shardable
            redistribute_cost = []
            for input_idx, spec in enumerate(input_expected_specs):
                qkv_strategy = op_schema.args_schema[tensor_input_indices[input_idx]]
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


@register_op_strategy(aten.constant_pad_nd.default)
def constant_pad_nd_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    # TODO(d4l3k); implement a more correct strategy for constant_pad_nd
    return OpStrategy(
        [
            PlacementStrategy(
                output_specs=DTensorSpec(mesh, (Replicate(),)),
                input_specs=(
                    DTensorSpec(mesh, (Replicate(),)),
                    DTensorSpec(mesh, (Replicate(),)),
                ),
                redistribute_cost=[[1]],
            )
        ]
    )


@register_op_strategy(aten._scaled_dot_product_efficient_attention.default)
def scaled_dot_product_efficient_attention_strategy(
    mesh: DeviceMesh, op_schema: OpSchema
) -> OpStrategy:
    # NOTE: currently we only support some simple strategies to support tensor parallelism
    q_input_strategy = op_schema.args_schema[0]
    assert isinstance(q_input_strategy, OpStrategy)
    # assuming q/k/v have the same shape
    qkv_shape = q_input_strategy.shape
    has_attn_bias = op_schema.args_schema[3] is not None
    compute_log_sumexp = op_schema.args_schema[4]

    all_mesh_dim_strategies = []

    for mesh_dim in range(mesh.ndim):
        single_mesh_dim_strategies = []

        # placement list stores placements of [outputs, inputs]
        # in the spda case, we have 2 valid tensor outputs and 3 or 4 tensor inputs
        # first we can always accept full replication for both inputs and outputs
        all_replicate: List[Placement] = [Replicate()] * (5 + has_attn_bias)
        single_mesh_dim_strategies.append(all_replicate)

        # second we can accept the sharding pattern of tensor parallelism, which
        # shard on the heads dimension
        qkv_sharding = Shard(1)
        output_sharding = Shard(1)
        if compute_log_sumexp:
            logsumexp_sharding: Placement = Shard(1)
        else:
            # empty logsumexp, replicated
            logsumexp_sharding = Replicate()

        num_heads_dim_sharding = [
            output_sharding,
            logsumexp_sharding,
            qkv_sharding,
            qkv_sharding,
            qkv_sharding,
        ]
        if has_attn_bias:
            num_heads_dim_sharding.append(Shard(1))
        single_mesh_dim_strategies.append(num_heads_dim_sharding)

        all_mesh_dim_strategies.append(single_mesh_dim_strategies)

    strategy_combs = itertools.product(*all_mesh_dim_strategies)

    all_strategies = []
    for strategy_comb in strategy_combs:
        spec_list = []
        for specs in zip(*strategy_comb):
            spec_list.append(DTensorSpec(mesh, tuple(specs)))

        assert len(spec_list) == (5 + has_attn_bias)
        input_expected_specs = spec_list[2:]
        output_specs: List[Optional[DTensorSpec]] = list(spec_list[:2])
        # fill in None for the scalar tensor return values
        # namely philox_seed and philox_offset
        output_specs.extend([None, None])
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


@register_op_strategy(aten._scaled_dot_product_efficient_attention_backward.default)
def scaled_dot_product_efficient_attention_backward_strategy(
    mesh: DeviceMesh, op_schema: OpSchema
) -> OpStrategy:
    q_input_strategy = op_schema.args_schema[1]
    assert isinstance(q_input_strategy, OpStrategy)
    # assuming q/k/v have the same shape
    qkv_shape = q_input_strategy.shape
    has_attn_bias = op_schema.args_schema[4] is not None

    tensor_input_indices = [
        i
        for i, arg_spec in enumerate(op_schema.args_schema)
        if isinstance(arg_spec, OpStrategy)
    ]

    all_mesh_dim_strategies = []

    for mesh_dim in range(mesh.ndim):
        single_mesh_dim_strategies = []

        # placement list stores placements of [outputs, inputs]
        # in the spda backward case, we have 4 tensor outputs and 8 or 9 tensor inputs
        # NOTE: Output sharding of grad_bias on heads dim if attn_bias is present;
        #       otherwise grad_bias will be empty and its DTensorSpec will be removed.
        # first we can always accept full replication for both inputs and outputs
        all_replicate: List[Placement] = [Replicate()] * (12 + has_attn_bias)

        single_mesh_dim_strategies.append(all_replicate)

        # second we can accept the sharding pattern of tensor parallelism, which
        # shard on the heads dimension
        grad_output_sharding = Shard(1)
        qkv_sharding = Shard(1)
        output_sharding = Shard(1)
        logsumexp_sharding = Shard(1)
        grad_qkv_sharding = Shard(1)
        grad_bias_sharding = Shard(1)

        num_heads_dim_sharding: List[Placement] = [
            grad_qkv_sharding,
            grad_qkv_sharding,
            grad_qkv_sharding,
            grad_bias_sharding,
            grad_output_sharding,
            qkv_sharding,
            qkv_sharding,
            qkv_sharding,
            # the place for optional input attn_bias,
            output_sharding,
            logsumexp_sharding,
        ]
        # input sharding of attn_bias on heads dim if present
        if has_attn_bias:
            num_heads_dim_sharding.insert(8, Shard(1))
        # accept replicate on the rest scalar tensor inputs
        # namely philox_seed and philox_offset
        num_heads_dim_sharding.extend([Replicate(), Replicate()])
        single_mesh_dim_strategies.append(num_heads_dim_sharding)

        all_mesh_dim_strategies.append(single_mesh_dim_strategies)

    strategy_combs = itertools.product(*all_mesh_dim_strategies)

    all_strategies = []
    for strategy_comb in strategy_combs:
        spec_list = []
        for specs in zip(*strategy_comb):
            spec_list.append(DTensorSpec(mesh, tuple(specs)))

        assert len(spec_list) == (12 + has_attn_bias)
        input_expected_specs = spec_list[4:]
        output_specs: List[Optional[DTensorSpec]] = list(spec_list[:4])
        # remove the DTensorSpec of output grad_bias if it's empty
        if not has_attn_bias:
            output_specs[-1] = None
        if all(is_tensor_shardable(qkv_shape, spec) for spec in input_expected_specs):
            # only add to the strategy list when all inputs are shardable
            redistribute_cost = []
            for input_idx, spec in enumerate(input_expected_specs):
                qkv_strategy = op_schema.args_schema[tensor_input_indices[input_idx]]
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
