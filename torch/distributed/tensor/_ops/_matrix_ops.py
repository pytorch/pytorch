# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor


from typing import Optional

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpStrategy,
    PlacementList,
    PlacementStrategy,
    RuntimeSchemaInfo,
)
from torch.distributed.tensor._ops._einsum_strategy import gen_einsum_strategies
from torch.distributed.tensor._ops.utils import (
    expand_to_full_mesh_op_strategy,
    generate_redistribute_costs,
    infer_broadcast_dims_map,
    is_tensor_shardable,
    map_placements_after_broadcast,
    prod,
    register_op_strategy,
)
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)


aten = torch.ops.aten


@register_op_strategy(aten.t.default)
def transpose_strategy(op_schema: OpSchema) -> OpStrategy:
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
                mesh=input_strategy.mesh,
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


def _scaled_mm_like_strategy(
    mm_equation: str, mesh: DeviceMesh, op_schema: OpSchema
) -> OpStrategy:
    (
        self_strategy,
        mat2_strategy,
        scale_self_strategy,
        scale_mat2_strategy,
        bias_strategy,
        scale_result_strategy,
        *_,
    ) = op_schema.args_schema
    assert isinstance(self_strategy, OpStrategy)
    assert isinstance(mat2_strategy, OpStrategy)
    assert isinstance(scale_self_strategy, OpStrategy)
    assert isinstance(scale_mat2_strategy, OpStrategy)
    # TODO: add support for these later
    assert bias_strategy is None, "_scaled_mm on DTensors doesn't support bias"
    assert scale_result_strategy is None, (
        "_scaled_mm on DTensors doesn't support scale_result"
    )
    # generate all possible strategies for mm
    mm_strategy = gen_einsum_strategies(mm_equation, mesh)
    # filter out invalid strategies and associate costs
    strategies = mm_strategy.strategies
    filtered_strategies = []
    for strtg in strategies:
        assert strtg.input_specs is not None
        self_spec = strtg.input_specs[0]
        mat2_spec = strtg.input_specs[1]
        # propagate the operands' specs to their scales, except for tensor-wise
        # scaling which can have any numbers of dims (legacy...), hence sharding
        # dims won't map. for tensor-wise, anyways, we can only do replication.
        scale_self_spec = (
            DTensorSpec(self_spec.mesh, (Replicate(),))
            if prod(scale_self_strategy.shape) == 1
            else self_spec
        )
        scale_mat2_spec = (
            DTensorSpec(mat2_spec.mesh, (Replicate(),))
            if prod(scale_mat2_strategy.shape) == 1
            else mat2_spec
        )
        strtg.input_specs = list(strtg.input_specs) + [scale_self_spec, scale_mat2_spec]
        if (
            is_tensor_shardable(self_strategy.shape, self_spec)
            and is_tensor_shardable(mat2_strategy.shape, mat2_spec)
            and is_tensor_shardable(scale_self_strategy.shape, scale_self_spec)
            and is_tensor_shardable(scale_mat2_strategy.shape, scale_mat2_spec)
        ):
            redistribute_cost = [
                generate_redistribute_costs(self_strategy, self_spec),
                generate_redistribute_costs(mat2_strategy, mat2_spec),
                generate_redistribute_costs(scale_self_strategy, scale_self_spec),
                generate_redistribute_costs(scale_mat2_strategy, scale_mat2_spec),
            ]
            strtg.redistribute_cost = redistribute_cost
            filtered_strategies.append(strtg)

    mm_strategy.strategies = filtered_strategies

    return mm_strategy


@register_op_strategy(aten.mm.default)
def mm_strategy(op_schema: OpSchema) -> OpStrategy:
    mesh = op_schema.get_mesh_from_args()
    return _mm_like_strategy("mk,kn->mn", mesh, op_schema)


@register_op_strategy(aten.addmm.default)
def addmm_strategy(op_schema: OpSchema) -> OpStrategy:
    mesh = op_schema.get_mesh_from_args()
    return _addmm_like_strategy("mk,kn->mn", mesh, op_schema)


@register_op_strategy(aten.bmm.default)
def bmm_strategy(op_schema: OpSchema) -> OpStrategy:
    mesh = op_schema.get_mesh_from_args()
    return _mm_like_strategy("bmk,bkn->bmn", mesh, op_schema)


@register_op_strategy(aten.baddbmm.default)
def baddmm_strategy(op_schema: OpSchema) -> OpStrategy:
    mesh = op_schema.get_mesh_from_args()
    return _addmm_like_strategy("bmk,bkn->bmn", mesh, op_schema)


@register_op_strategy(aten._scaled_mm.default)
def scaled_mm_strategy(op_schema: OpSchema) -> OpStrategy:
    mesh = op_schema.get_mesh_from_args()
    return _scaled_mm_like_strategy("mk,kn->mn", mesh, op_schema)


@register_op_strategy(
    aten._scaled_dot_product_flash_attention.default, schema_info=RuntimeSchemaInfo(5)
)
def scaled_dot_product_flash_attention_strategy(op_schema: OpSchema) -> OpStrategy:
    # NOTE: currently we only support some simple strategies to support tensor parallelism
    # TODO: sdpa might be a good candidate for us to explore decomposed sharding propagation
    # as it involves: matmul, pointwise, reduction ops together.

    mesh = op_schema.get_mesh_from_args()

    return_debug_mask = len(op_schema.args_schema) >= 6 and op_schema.args_schema[5]
    q_input_strategy = op_schema.args_schema[0]
    assert isinstance(q_input_strategy, OpStrategy)
    # assuming q/k/v have the same shape

    single_mesh_dim_strategies = []

    # placement list stores placements of [outputs, inputs]
    # in the spda case, we have 3 valid tensor outputs and 3 tensor inputs
    # first we can always accept full replication for both inputs and outputs
    all_replicate: PlacementList = [
        Replicate(),
        Replicate(),
        None,  # cum_seq_q
        None,  # cum_seq_k
        None,  # max_q
        None,  # max_k
        Replicate(),  # rng_state
        None,  # unused
        Replicate(),
        Replicate(),
        Replicate(),
        Replicate(),
    ]
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

    num_heads_dim_sharding: PlacementList = [
        output_sharding,
        logsumexp_sharding,
        None,  # cum_seq_q
        None,  # cum_seq_k
        None,  # max_q
        None,  # max_k
        Replicate(),  # rng_state
        None,  # unused
        debug_attn_mask_sharding,
        qkv_sharding,
        qkv_sharding,
        qkv_sharding,
    ]
    single_mesh_dim_strategies.append(num_heads_dim_sharding)

    # Shard on the batch dimension
    single_mesh_dim_strategies.append(
        [
            Shard(0),  # output
            Shard(0),  # logsumexp
            None,  # cum_seq_q
            None,  # cum_seq_k
            None,  # max_q
            None,  # max_k
            Replicate(),  # rng_state
            None,  # unused
            Shard(0),  # debugattn
            Shard(0),  # q
            Shard(0),  # k
            Shard(0),  # v
        ]
    )

    # Context Parallelism: shards on the sequence dim
    single_mesh_dim_strategies.append(
        [
            Shard(2),  # output
            Shard(2),  # logsumexp
            None,  # cum_seq_q
            None,  # cum_seq_k
            None,  # max_q
            None,  # max_k
            Replicate(),  # rng_state
            None,  # unused
            Shard(2),  # debugattn
            Shard(2),  # q
            Shard(2),  # k
            Shard(2),  # v
        ]
    )
    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=9
    )


@register_op_strategy(aten._scaled_dot_product_flash_attention_backward.default)
def scaled_dot_product_flash_attention_backward_strategy(
    op_schema: OpSchema,
) -> OpStrategy:
    # backward op does not need to validate the mesh since forward op has already done it
    mesh = op_schema.get_mesh_from_args(validate=False)

    q_input_strategy = op_schema.args_schema[1]
    assert isinstance(q_input_strategy, OpStrategy)
    # assuming q/k/v have the same shape

    tensor_input_indices = [
        i
        for i, arg_spec in enumerate(op_schema.args_schema)
        if isinstance(arg_spec, OpStrategy)
    ]
    num_tensor_inputs = len(tensor_input_indices)

    single_mesh_dim_strategies = []

    # placement list stores placements of [outputs, inputs]
    # in the spda backward case, we have 3 tensor outputs and 6 to 10 tensor inputs
    # first we can always accept full replication for both inputs and outputs
    all_replicate: PlacementList = [Replicate()] * (3 + num_tensor_inputs)

    single_mesh_dim_strategies.append(all_replicate)

    # second we can accept the sharding pattern of tensor parallelism, which
    # shard on the num of head dim
    grad_output_sharding = Shard(1)  # num head dim
    qkv_sharding = Shard(1)  # num head dim
    output_sharding = Shard(1)  # num head dim
    logsumexp_sharding = Shard(1)  # num head dim
    grad_qkv_sharding = Shard(1)  # num head dim

    num_heads_dim_sharding: PlacementList = [
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

    # Batch sharding
    batch_dim_sharding: PlacementList = [
        Shard(0),  # grad_q
        Shard(0),  # grad_k
        Shard(0),  # grad_v
        Shard(0),  # grad_output
        Shard(0),  # q
        Shard(0),  # k
        Shard(0),  # v
        Shard(0),  # output
        Shard(0),  # logsumexp
    ]
    # accept replicate on the rest tensor inputs, potentially
    # cum_seq_q, cum_seq_k, philox_seed, philox_offset
    # at indices 6, 7, 12, 13, respectively
    batch_dim_sharding.extend([Replicate()] * (num_tensor_inputs - 6))
    single_mesh_dim_strategies.append(batch_dim_sharding)

    # Context Parallelism: shards on the sequence dim
    seq_dim_sharding: PlacementList = [
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

    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=3
    )


@register_op_strategy(aten.constant_pad_nd.default)
def constant_pad_nd_strategy(op_schema: OpSchema) -> OpStrategy:
    mesh = op_schema.get_mesh_from_args(validate=False)

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


@register_op_strategy(
    aten._scaled_dot_product_efficient_attention.default,
    schema_info=RuntimeSchemaInfo(4),
)
def scaled_dot_product_efficient_attention_strategy(op_schema: OpSchema) -> OpStrategy:
    # NOTE: currently we only support some simple strategies to support tensor parallelism
    mesh = op_schema.get_mesh_from_args()
    q_input_strategy = op_schema.args_schema[0]
    assert isinstance(q_input_strategy, OpStrategy)
    # assuming q/k/v have the same shape

    has_attn_bias = op_schema.args_schema[3] is not None
    compute_log_sumexp = op_schema.args_schema[4]

    single_mesh_dim_strategies: list[PlacementList] = []

    # placement list stores placements of [outputs, inputs]
    # in the spda case, we have 2 valid tensor outputs and 3 or 4 tensor inputs
    # first we can always accept full replication for both inputs and outputs
    all_replicate: PlacementList = [
        Replicate(),
        Replicate(),
        None,
        None,
        Replicate(),
        Replicate(),
        Replicate(),
    ]
    if has_attn_bias:
        all_replicate.append(Replicate())  # attn bias

    # Context Parallelism: shards on the sequence dim
    single_mesh_dim_strategies.append(
        [
            Shard(2),  # output
            Shard(2),  # logsumexp
            None,  # philox_seed
            None,  # philox_offset
            Shard(2),  # q
            Shard(2),  # k
            Shard(2),  # v
        ]
    )

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
        None,
        None,
        qkv_sharding,
        qkv_sharding,
        qkv_sharding,
    ]
    if has_attn_bias:
        num_heads_dim_sharding.append(Shard(1))
    single_mesh_dim_strategies.append(num_heads_dim_sharding)

    # batch sharding
    if compute_log_sumexp:
        logsumexp_sharding_dp: Placement = Shard(0)
    else:
        # empty logsumexp, replicated
        logsumexp_sharding_dp = Replicate()
    batch_sharding = [
        Shard(0),  # output
        logsumexp_sharding_dp,  # logsumexp
        None,  # philox_seed
        None,  # philox_offset
        Shard(0),  # q
        Shard(0),  # k
        Shard(0),  # v
    ]
    if has_attn_bias:
        batch_sharding.append(Shard(0))

    single_mesh_dim_strategies.append(batch_sharding)

    return expand_to_full_mesh_op_strategy(
        mesh,
        op_schema,
        single_mesh_dim_strategies,
        input_index=4,
    )


@register_op_strategy(aten._scaled_dot_product_efficient_attention_backward.default)
def scaled_dot_product_efficient_attention_backward_strategy(
    op_schema: OpSchema,
) -> OpStrategy:
    # backward op does not need to validate the mesh since forward op has already done it
    mesh = op_schema.get_mesh_from_args(validate=False)

    q_input_strategy = op_schema.args_schema[1]
    assert isinstance(q_input_strategy, OpStrategy)
    # assuming q/k/v have the same shape
    has_attn_bias = op_schema.args_schema[4] is not None

    single_mesh_dim_strategies = []

    # placement list stores placements of [outputs, inputs]
    # in the spda backward case, we have 4 tensor outputs and 8 or 9 tensor inputs
    # NOTE: Output sharding of grad_bias on heads dim if attn_bias is present;
    #       otherwise grad_bias will be empty and its DTensorSpec will be removed.
    # first we can always accept full replication for both inputs and outputs
    all_replicate: PlacementList = [Replicate()] * (12 + has_attn_bias)

    if not has_attn_bias:
        all_replicate[3] = None  # grad bias is None if attn_bias is not present

    single_mesh_dim_strategies.append(all_replicate)

    # second we can accept the sharding pattern of tensor parallelism, which
    # shard on the heads dimension
    grad_output_sharding = Shard(1)
    qkv_sharding = Shard(1)
    output_sharding = Shard(1)
    logsumexp_sharding = Shard(1)
    grad_qkv_sharding = Shard(1)
    grad_bias_sharding = Shard(1) if has_attn_bias else None

    num_heads_dim_sharding: PlacementList = [
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

    # Shards on batch dim
    batch_dim_sharding: PlacementList = [
        Shard(0),  # grad_q
        Shard(0),  # grad_k
        Shard(0),  # grad_v
        Shard(0) if has_attn_bias else None,  # grad_bias
        Shard(0),  # grad_output
        Shard(0),  # q
        Shard(0),  # k
        Shard(0),  # v
        Shard(0),  # output
        Shard(0),  # logsumexp
    ]
    # accept replicate on the rest tensor inputs, potentially
    # cum_seq_q, cum_seq_k, philox_seed, philox_offset
    # at indices 6, 7, 12, 13, respectively
    if has_attn_bias:
        batch_dim_sharding.insert(8, Shard(0))
    batch_dim_sharding.extend([Replicate(), Replicate()])
    single_mesh_dim_strategies.append(batch_dim_sharding)

    # Context Parallelism: shards on the sequence dim
    seq_dim_sharding: PlacementList = [
        Shard(2),  # grad_q
        Shard(2),  # grad_k
        Shard(2),  # grad_v
        Shard(1) if has_attn_bias else None,  # grad_bias
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
    if has_attn_bias:
        num_heads_dim_sharding.insert(8, Shard(1))
    seq_dim_sharding.extend([Replicate(), Replicate()])
    single_mesh_dim_strategies.append(seq_dim_sharding)

    return expand_to_full_mesh_op_strategy(
        mesh,
        op_schema,
        single_mesh_dim_strategies,
        input_index=4,
    )


@register_op_strategy(
    aten._scaled_dot_product_cudnn_attention.default,
    schema_info=RuntimeSchemaInfo(4),
)
def scaled_dot_product_cudnn_attention_strategy(op_schema: OpSchema) -> OpStrategy:
    mesh = op_schema.get_mesh_from_args()

    (
        query_strategy,  # query
        _,  # key
        _,  # value
        attn_bias_strategy,
        compute_log_sumexp,  # compute_log_sumexp
        *rest_args,  # optional args: dropout_p, is_causal, return_debug_mask, scale
    ) = op_schema.args_schema
    return_debug_mask = len(op_schema.args_schema) >= 8 and rest_args[2]
    has_attn_bias = attn_bias_strategy is not None
    debug_attn_mask_sharding: Optional[Placement] = (
        Replicate() if return_debug_mask else None
    )

    assert isinstance(query_strategy, OpStrategy)
    # assuming q/k/v have the same shape

    single_mesh_dim_strategies = []

    # placement list stores placements of [outputs, inputs]
    # in the spda case, we have 2 valid tensor outputs and 3 tensor inputs
    # first we can always accept full replication for both inputs and outputs
    all_replicate: PlacementList = [
        Replicate(),  # output
        Replicate(),  # logsumexp
        None,  # cum_seq_q
        None,  # cum_seq_k
        None,  # max_q
        None,  # max_k
        None,  # philox_seed
        None,  # philox_offset
        # NOTE: debug_attn_mask is not supproted by pytorch and is always an empty tensor
        # https://github.com/pytorch/pytorch/blob/60205b0eb2602317856312a66d955c88334ade0b/aten/src/ATen/native/transformers/cuda/attention.cu#L839-L840
        debug_attn_mask_sharding,  # debug_attn_mask
        Replicate(),  # q
        Replicate(),  # k
        Replicate(),  # v
    ]
    if has_attn_bias:
        all_replicate.append(Replicate())  # attn bias

    single_mesh_dim_strategies.append(all_replicate)

    # second we can accept the sharding pattern of tensor parallelism, which
    # shard on the num of head dim
    tp_sharding = Shard(1)  # num head dim
    qkv_sharding = tp_sharding
    output_sharding = tp_sharding
    logsumexp_sharding = tp_sharding if compute_log_sumexp else Replicate()
    debug_attn_mask_sharding = tp_sharding if return_debug_mask else None

    num_heads_dim_sharding: PlacementList = [
        output_sharding,
        logsumexp_sharding,
        None,  # cum_seq_q
        None,  # cum_seq_k
        None,  # max_q
        None,  # max_k
        None,  # philox_seed
        None,  # philox_offset
        debug_attn_mask_sharding,
        qkv_sharding,
        qkv_sharding,
        qkv_sharding,
    ]
    single_mesh_dim_strategies.append(num_heads_dim_sharding)

    # batch parallelism
    logsumexp_sharding = Shard(0) if compute_log_sumexp else Replicate()
    debug_attn_mask_sharding = Shard(0) if return_debug_mask else None
    batch_dim_sharding: PlacementList = [
        Shard(0),  # output
        logsumexp_sharding,
        None,  # cum_seq_q
        None,  # cum_seq_k
        None,  # max_q
        None,  # max_k
        None,  # philox_seed
        None,  # philox_offset
        debug_attn_mask_sharding,
        Shard(0),  # q
        Shard(0),  # k
        Shard(0),  # v
    ]
    single_mesh_dim_strategies.append(batch_dim_sharding)

    # Context Parallelism: shards on the sequence dim
    cp_sharding = Shard(2)  # seq dim
    logsumexp_sharding = cp_sharding if compute_log_sumexp else Replicate()
    debug_attn_mask_sharding = cp_sharding if return_debug_mask else None

    single_mesh_dim_strategies.append(
        [
            cp_sharding,  # output
            logsumexp_sharding,  # logsumexp
            None,  # cum_seq_q
            None,  # cum_seq_k
            None,  # max_q
            None,  # max_k
            None,  # philox_seed
            None,  # philox_offset
            debug_attn_mask_sharding,  # debug_attn_mask
            cp_sharding,  # q
            cp_sharding,  # k
            cp_sharding,  # v
        ]
    )
    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=9
    )


@register_op_strategy(aten._scaled_dot_product_cudnn_attention_backward.default)
def scaled_scaled_dot_product_cudnn_attention_backward_strategy(
    op_schema: OpSchema,
) -> OpStrategy:
    # backward op does not need to validate the mesh since forward op has already done it
    mesh = op_schema.get_mesh_from_args(validate=False)

    assert len(op_schema.args_schema) >= 15
    has_attn_bias = op_schema.args_schema[8] is not None
    has_scale = len(op_schema.args_schema) >= 16 and False

    query_strategy = op_schema.args_schema[1]
    assert isinstance(query_strategy, OpStrategy)
    # assuming q/k/v have the same shape

    single_mesh_dim_strategies = []

    # placement list stores placements of [outputs, inputs]
    # cudnn outputs: (Tensor dq, Tensor dk, Tensor dv)
    # cudnn inputs: (
    #   Tensor grad_out,
    #   Tensor query,
    #   Tensor key,
    #   Tensor value,
    #   Tensor out,
    #   Tensor logsumexp,
    #   Tensor philox_seed,
    #   Tensor philox_offset,
    #   Tensor attn_bias,
    #   Tensor cum_seq_q,
    #   Tensor cum_seq_k,
    #   SymInt max_q,
    #   SymInt max_k,
    #   float dropout_p,
    #   bool is_causal,
    #   int? scale,
    # )

    # case 1: we can always accept full replication for both inputs and outputs
    all_replicate_out: PlacementList = [
        Replicate(),  # dq
        Replicate(),  # dk
        Replicate(),  # dv
    ]
    all_replicate_inp: PlacementList = [Replicate()] * 6
    all_replicate_inp += [
        Replicate()
    ] * 2  # philox_seed, philox_offset is casted to Replicate() in DTensor
    all_replicate_inp += [Replicate() if has_attn_bias else None]
    all_replicate_inp += [None] * 6
    if has_scale:
        all_replicate_inp.append(None)

    all_replicate: PlacementList = all_replicate_out + all_replicate_inp
    single_mesh_dim_strategies.append(all_replicate)

    # case 2: we can accept the sharding pattern of tensor parallelism, which
    #   shards on the num of head dim
    qkv_sharding = Shard(1)  # num head dim
    output_sharding = Shard(1)  # num head dim
    logsumexp_sharding = Shard(1)  # num head dim

    num_heads_dim_sharding_out: PlacementList = [qkv_sharding] * 3
    num_heads_dim_sharding_inp: PlacementList = [qkv_sharding] * 4
    num_heads_dim_sharding_inp += [output_sharding]
    num_heads_dim_sharding_inp += [logsumexp_sharding]
    num_heads_dim_sharding_inp += [
        Replicate()
    ] * 2  # philox_seed, philox_offset is casted to Replicate() in DTensor
    num_heads_dim_sharding_inp += [Shard(1) if has_attn_bias else None]
    num_heads_dim_sharding_inp += [None] * 6
    if has_scale:
        num_heads_dim_sharding_inp.append(None)

    num_heads_dim_sharding = num_heads_dim_sharding_out + num_heads_dim_sharding_inp
    single_mesh_dim_strategies.append(num_heads_dim_sharding)

    # case 3: Context Parallelism which shards on the sequence dim
    context_parallel_sharding_out: PlacementList = [Shard(2)] * 3
    context_parallel_sharding_inp: PlacementList = [Shard(2)] * 6
    context_parallel_sharding_inp += [
        Replicate()
    ] * 2  # philox_seed, philox_offset is casted to Replicate() in DTensor
    context_parallel_sharding_inp += [Shard(2) if has_attn_bias else None]
    context_parallel_sharding_inp += [None] * 6
    if has_scale:
        context_parallel_sharding_inp.append(None)

    context_parallel_sharding = (
        context_parallel_sharding_out + context_parallel_sharding_inp
    )
    single_mesh_dim_strategies.append(context_parallel_sharding)

    # case 4: we can accept the sharding pattern of batch parallelism, which
    #   shards on the batch dimension
    qkv_sharding = Shard(0)
    output_sharding = Shard(0)
    logsumexp_sharding = Shard(0)

    batch_dim_sharding_out: PlacementList = [qkv_sharding] * 3
    batch_dim_sharding_inp: PlacementList = [qkv_sharding] * 4
    batch_dim_sharding_inp += [output_sharding]
    batch_dim_sharding_inp += [logsumexp_sharding]
    batch_dim_sharding_inp += [
        Replicate()
    ] * 2  # philox_seed, philox_offset is casted to Replicate() in DTensor
    batch_dim_sharding_inp += [Shard(0) if has_attn_bias else None]
    batch_dim_sharding_inp += [None] * 6
    if has_scale:
        batch_dim_sharding_inp.append(None)

    batch_dim_sharding = batch_dim_sharding_out + batch_dim_sharding_inp
    single_mesh_dim_strategies.append(batch_dim_sharding)

    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=3
    )


@register_op_strategy(aten._grouped_mm.default)
def grouped_mm_strategy(op_schema: OpSchema) -> OpStrategy:
    mesh = op_schema.get_mesh_from_args()

    mat1_strategy = op_schema.args_schema[0]
    assert isinstance(mat1_strategy, OpStrategy)
    mat2_strategy = op_schema.args_schema[1]
    assert isinstance(mat2_strategy, OpStrategy)
    if len(op_schema.args_schema) > 3:
        bias_strategy = op_schema.args_schema[3]
        assert bias_strategy is None, "grouped_mm doesn't support bias yet"

    single_mesh_dim_strategies = []

    offs_placement = None
    if len(op_schema.args_schema) > 2 and op_schema.args_schema[2] is not None:
        offs_placement = Replicate()  # offs should always be replicated

    all_replicate: PlacementList = [
        Replicate(),
        Replicate(),  # mat1
        Replicate(),  # mat2
        offs_placement,  # offs
        None,  # bias
    ]
    partial_replicate: PlacementList = [
        Partial(),
        Partial(),  # mat1
        Replicate(),  # mat2
        offs_placement,  # offs
        None,  # bias
    ]
    replicate_partial: PlacementList = [
        Partial(),
        Replicate(),  # mat1
        Partial(),  # mat2
        offs_placement,  # offs
        None,  # bias
    ]
    single_mesh_dim_strategies = [all_replicate, partial_replicate, replicate_partial]

    if mat1_strategy.ndim == 2 and mat2_strategy.ndim == 3:
        # rowwise_replicate for 2dx3d not supported
        replicate_colwise_2x3: PlacementList = [
            Shard(1),
            Replicate(),  # mat1
            Shard(2),  # mat2
            offs_placement,  # offs
            None,  # bias
        ]
        colwise_rowwise_2x3: PlacementList = [
            Partial(),
            Shard(1),  # mat1
            Shard(1),  # mat2
            offs_placement,  # offs
            None,  # bias
        ]
        single_mesh_dim_strategies.extend([replicate_colwise_2x3, colwise_rowwise_2x3])

    if mat1_strategy.ndim == 3 and mat2_strategy.ndim == 2:
        # replicate_colwise for 3dx2d not supported
        colwise_rowwise_3x2: PlacementList = [
            Partial(),
            Shard(2),  # mat1
            Shard(0),  # mat2
            offs_placement,  # offs
            None,  # bias
        ]
        rowwise_replicate_3x2: PlacementList = [
            Shard(0),
            Shard(1),  # mat1
            Replicate(),  # mat2
            offs_placement,  # offs
            None,  # bias
        ]
        single_mesh_dim_strategies.extend([colwise_rowwise_3x2, rowwise_replicate_3x2])

    if mat1_strategy.ndim == 2 and mat2_strategy.ndim == 2:
        # colwise_rowwise for 2dx2d not supported
        replicate_colwise_2x2: PlacementList = [
            Shard(2),
            Replicate(),  # mat1
            Shard(1),  # mat2
            offs_placement,  # offs
            None,  # bias
        ]
        rowwise_replicate_2x2: PlacementList = [
            Shard(1),
            Shard(0),  # mat1
            Replicate(),  # mat2
            offs_placement,  # offs
            None,  # bias
        ]
        single_mesh_dim_strategies.extend(
            [replicate_colwise_2x2, rowwise_replicate_2x2]
        )

    if mat1_strategy.ndim == 3 and mat2_strategy.ndim == 3:
        replicate_colwise_3x3: PlacementList = [
            Shard(2),
            Replicate(),  # mat1
            Shard(2),  # mat2
            offs_placement,  # offs
            None,  # bias
        ]
        rowwise_replicate_3x3: PlacementList = [
            Shard(1),
            Shard(1),  # mat1
            Replicate(),  # mat2
            offs_placement,  # offs
            None,  # bias
        ]
        colwise_rowwise_3x3: PlacementList = [
            Partial(),
            Shard(2),  # mat1
            Shard(1),  # mat2
            offs_placement,  # offs
            None,  # bias
        ]
        batch_dim_sharding: PlacementList = [
            Shard(0),
            Shard(0),  # mat1
            Shard(0),  # mat2
            offs_placement,  # offs
            None,  # bias
        ]
        single_mesh_dim_strategies.extend(
            [
                replicate_colwise_3x3,
                rowwise_replicate_3x3,
                colwise_rowwise_3x3,
                batch_dim_sharding,
            ]
        )

    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=1
    )
