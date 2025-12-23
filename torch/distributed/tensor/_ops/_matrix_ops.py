# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor


import torch
from torch._ops import OpOverload
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import (
    ArgsType,
    KwargsType,
    OpSchema,
    OpSpec,
    OpStrategy,
    PlacementList,
    RuntimeSchemaInfo,
)
from torch.distributed.tensor._ops._einsum_strategy import gen_einsum_strategies
from torch.distributed.tensor._ops.single_dim_strategy import _ShardingPlaceholder
from torch.distributed.tensor._ops.utils import (
    expand_to_full_mesh_op_strategy,
    generate_redistribute_costs,
    infer_broadcast_dims_map,
    is_tensor_shardable,
    map_placements_after_broadcast,
    prod,
    register_op_strategy,
)
from torch.distributed.tensor._utils import (
    compute_local_shape_and_global_offset,
    compute_local_stride,
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
    if not isinstance(self_strategy, OpStrategy):
        raise AssertionError(f"Expected OpStrategy, got {type(self_strategy)}")

    transpose_strategies = []
    for input_strategy in self_strategy.strategies:
        input_spec = input_strategy.output_spec
        # follow the input spec but transpose the Shard placements
        output_placements = [
            Shard(1 - p.dim) if isinstance(p, Shard) else p
            for p in input_spec.placements
        ]
        transpose_strategy = OpSpec(
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
    if not isinstance(self_strategy, OpStrategy):
        raise AssertionError(f"Expected OpStrategy, got {type(self_strategy)}")
    if not isinstance(mat2_strategy, OpStrategy):
        raise AssertionError(f"Expected OpStrategy, got {type(mat2_strategy)}")
    # generate all possible strategies for mm
    mm_strategy = gen_einsum_strategies(mm_equation, mesh)
    # filter out invalid strategies and associate costs
    strategies = mm_strategy.strategies
    filtered_strategies = []
    for strtg in strategies:
        if strtg.input_specs is None:
            raise AssertionError(
                f"Expected input_specs to be not None, got {strtg.input_specs}"
            )
        self_spec = strtg.input_specs[0]
        mat2_spec = strtg.input_specs[1]
        if is_tensor_shardable(
            self_strategy.shape, self_spec, allow_unbacked_sharding=True
        ) and is_tensor_shardable(
            mat2_strategy.shape, mat2_spec, allow_unbacked_sharding=True
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
    if not isinstance(self_strategy, OpStrategy):
        raise AssertionError(f"Expected OpStrategy, got {type(self_strategy)}")
    if not isinstance(mat1_strategy, OpStrategy):
        raise AssertionError(f"Expected OpStrategy, got {type(mat1_strategy)}")
    if not isinstance(mat2_strategy, OpStrategy):
        raise AssertionError(f"Expected OpStrategy, got {type(mat2_strategy)}")
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
        if strtg.input_specs is None:
            raise AssertionError(
                f"Expected input_specs to be not None, got {strtg.input_specs}"
            )
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
            mat1_strategy.shape, mat1_spec, allow_unbacked_sharding=True
        ) and is_tensor_shardable(
            mat2_strategy.shape, mat2_spec, allow_unbacked_sharding=True
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
    if not isinstance(self_strategy, OpStrategy):
        raise AssertionError(f"Expected OpStrategy, got {type(self_strategy)}")
    if not isinstance(mat2_strategy, OpStrategy):
        raise AssertionError(f"Expected OpStrategy, got {type(mat2_strategy)}")
    if not isinstance(scale_self_strategy, OpStrategy):
        raise AssertionError(f"Expected OpStrategy, got {type(scale_self_strategy)}")
    if not isinstance(scale_mat2_strategy, OpStrategy):
        raise AssertionError(f"Expected OpStrategy, got {type(scale_mat2_strategy)}")
    # TODO: add support for these later
    if bias_strategy is not None:
        raise AssertionError("_scaled_mm on DTensors doesn't support bias")
    if scale_result_strategy is not None:
        raise AssertionError("_scaled_mm on DTensors doesn't support scale_result")
    # generate all possible strategies for mm
    mm_strategy = gen_einsum_strategies(mm_equation, mesh)
    # filter out invalid strategies and associate costs
    strategies = mm_strategy.strategies
    filtered_strategies = []
    for strtg in strategies:
        if strtg.input_specs is None:
            raise AssertionError(
                f"Expected input_specs to be not None, got {strtg.input_specs}"
            )
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
            is_tensor_shardable(
                self_strategy.shape, self_spec, allow_unbacked_sharding=True
            )
            and is_tensor_shardable(
                mat2_strategy.shape, mat2_spec, allow_unbacked_sharding=True
            )
            and is_tensor_shardable(
                scale_self_strategy.shape, scale_self_spec, allow_unbacked_sharding=True
            )
            and is_tensor_shardable(
                scale_mat2_strategy.shape, scale_mat2_spec, allow_unbacked_sharding=True
            )
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


@register_op_strategy(aten.dot.default)
def dot_strategy(op_schema: OpSchema) -> OpStrategy:
    mesh = op_schema.get_mesh_from_args()
    return _mm_like_strategy("i,i->", mesh, op_schema)


@register_op_strategy(aten.mm.default)
def mm_strategy(op_schema: OpSchema) -> OpStrategy:
    mesh = op_schema.get_mesh_from_args()
    return _mm_like_strategy("mk,kn->mn", mesh, op_schema)


from ._einsum_strategy import EinsumDims


def gen_single_dim_einsum_strategies(
    equation: str,
    *,
    linearity: bool = False,
) -> list[list[Placement | _ShardingPlaceholder]]:
    """
    Generate a strategy list for the ops that follow einsum style notation.

    In principle, each mesh dim is independent of other device mesh dim when we
    generate strategies. So we generate strategy over each device mesh dim and
    do product combination on all mesh dims. We basically follow the below rule
    for each device mesh dim:

    1. Shard on contracting dim: When both inputs shard on contracting dim over
       the same device dim. The result will be Partial over that device dim.

    2. Shard on noncontracting dim:
        2.1: Shard on batch dim: output, both inputs all should shard on batch
        dim.
        2.2: Shard on lhs only dim or rhs only dim: both output and lhs or rhs
        input should shard on this free dim.

    3. Linearity (Partial): If enabled, set Partial on output and inputs over
       the same device mesh dim.
    """
    # parse einop equation and extract dims
    input_dims, output_dim = EinsumDims.parse_equation(equation)
    edims = EinsumDims.parse_dims(input_dims, output_dim)

    # generate strategies for each mesh dim and do cartesian product for final strategy. E.g., for a 2D mesh, we can have [P(),R,R]
    strategies_over_one_mesh_dim: list[list[Placement | _ShardingPlaceholder]] = []
    placement_list: list[Placement | _ShardingPlaceholder]
    # split batch dim
    for batch_dim in edims.batch_dims:
        output_batch_dim = output_dim.index(batch_dim)
        placement_list = [_ShardingPlaceholder(output_batch_dim)]
        for input_dim in input_dims:
            input_batch_dim = input_dim.index(batch_dim)
            placement_list.append(_ShardingPlaceholder(input_batch_dim))

        strategies_over_one_mesh_dim.append(placement_list)

    # split contracting dim
    for contracting_dim in edims.contracting_dims:
        # Contracting dim can shard on same device axis for both inputs. This
        # results in the output being Partial on that device axis. For example:
        # bmk_{x},k_{x}n -> bmn{Ux} (becomes partial over device axis x)
        placement_list = [Partial()]
        for input_dim in input_dims:
            input_contracting_dim = input_dim.index(contracting_dim)
            placement_list.append(_ShardingPlaceholder(input_contracting_dim))

        strategies_over_one_mesh_dim.append(placement_list)

    # split lhs free dim
    for lhs_dim in edims.lhs_out_only_dims:
        lhs_free_dim_output = output_dim.index(lhs_dim)
        lhs_free_dim_input = input_dims[0].index(lhs_dim)
        # this means split the lhs input and output
        # i.e. S(0), R -> S(0)
        lhs_placement_list: list[Placement | _ShardingPlaceholder] = [
            _ShardingPlaceholder(lhs_free_dim_output),
            _ShardingPlaceholder(lhs_free_dim_input),
            Replicate(),
        ]
        strategies_over_one_mesh_dim.append(lhs_placement_list)

    # split rhs free dim
    for rhs_dim in edims.rhs_out_only_dims:
        rhs_free_dim_output = output_dim.index(rhs_dim)
        rhs_free_dim_input = input_dims[1].index(rhs_dim)
        rhs_placement_list: list[Placement | _ShardingPlaceholder] = [
            _ShardingPlaceholder(rhs_free_dim_output),
            Replicate(),
            _ShardingPlaceholder(rhs_free_dim_input),
        ]
        strategies_over_one_mesh_dim.append(rhs_placement_list)

    # linearity strategy
    if linearity:
        linearity_placement_list: list[Placement | _ShardingPlaceholder] = [Partial()]
        for _ in input_dims:
            linearity_placement_list.append(Partial())
        strategies_over_one_mesh_dim.append(linearity_placement_list)

    return strategies_over_one_mesh_dim


# TODO enable in a separate PR along with more extensive validation.
# currently just used in test_single_dim_strategy.py to help validate the single-dim expansion infra
# @register_single_dim_strategy(aten.mm.default)
def mm_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    return gen_single_dim_einsum_strategies("mk,kn->mn")


@register_op_strategy(aten.addmm.default)
def addmm_strategy(op_schema: OpSchema) -> OpStrategy:
    mesh = op_schema.get_mesh_from_args()
    return _addmm_like_strategy("mk,kn->mn", mesh, op_schema)


@register_op_strategy(aten.bmm.default)
def bmm_strategy(op_schema: OpSchema) -> OpStrategy:
    mesh = op_schema.get_mesh_from_args()
    return _mm_like_strategy("bmk,bkn->bmn", mesh, op_schema)


@register_op_strategy(aten.baddbmm.default)
def baddbmm_strategy(op_schema: OpSchema) -> OpStrategy:
    mesh = op_schema.get_mesh_from_args()
    return _addmm_like_strategy("bmk,bkn->bmn", mesh, op_schema)


@register_op_strategy(aten._scaled_mm.default)
def scaled_mm_strategy(op_schema: OpSchema) -> OpStrategy:
    mesh = op_schema.get_mesh_from_args()
    return _scaled_mm_like_strategy("mk,kn->mn", mesh, op_schema)


def _scaled_dot_product_flash_attention_base_strategies(
    op_schema: OpSchema,
) -> list[PlacementList]:
    """Helper that returns list of base placement strategies (without CP)."""
    return_debug_mask = len(op_schema.args_schema) >= 6 and op_schema.args_schema[5]
    q_input_strategy = op_schema.args_schema[0]
    if not isinstance(q_input_strategy, OpStrategy):
        raise AssertionError(f"Expected OpStrategy, got {type(q_input_strategy)}")
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
    debug_attn_mask_sharding = Shard(0) if return_debug_mask else Replicate()
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
            debug_attn_mask_sharding,  # debugattn
            Shard(0),  # q
            Shard(0),  # k
            Shard(0),  # v
        ]
    )
    return single_mesh_dim_strategies


@register_op_strategy(
    aten._scaled_dot_product_flash_attention.default, schema_info=RuntimeSchemaInfo(5)
)
def scaled_dot_product_flash_attention_strategy(op_schema: OpSchema) -> OpStrategy:
    # NOTE: currently we only support some simple strategies to support tensor parallelism
    # TODO: sdpa might be a good candidate for us to explore decomposed sharding propagation
    # as it involves: matmul, pointwise, reduction ops together.

    mesh = op_schema.get_mesh_from_args()
    single_mesh_dim_strategies = _scaled_dot_product_flash_attention_base_strategies(
        op_schema
    )
    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=9
    )


def _scaled_dot_product_flash_attention_backward_base_strategies(
    op_schema: OpSchema,
) -> list[PlacementList]:
    """Helper that returns list of base placement strategies (without CP)."""
    q_input_strategy = op_schema.args_schema[1]
    if not isinstance(q_input_strategy, OpStrategy):
        raise AssertionError(f"Expected OpStrategy, got {type(q_input_strategy)}")
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

    return single_mesh_dim_strategies


@register_op_strategy(aten._scaled_dot_product_flash_attention_backward.default)
def scaled_dot_product_flash_attention_backward_strategy(
    op_schema: OpSchema,
) -> OpStrategy:
    # backward op does not need to validate the mesh since forward op has already done it
    mesh = op_schema.get_mesh_from_args(validate=False)
    single_mesh_dim_strategies = (
        _scaled_dot_product_flash_attention_backward_base_strategies(op_schema)
    )
    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=3
    )


@register_op_strategy(aten.constant_pad_nd.default)
def constant_pad_nd_strategy(op_schema: OpSchema) -> OpStrategy:
    mesh = op_schema.get_mesh_from_args(validate=False)

    # TODO(d4l3k); implement a more correct strategy for constant_pad_nd
    return OpStrategy(
        [
            OpSpec(
                output_specs=DTensorSpec(mesh, (Replicate(),)),
                input_specs=(
                    DTensorSpec(mesh, (Replicate(),)),
                    DTensorSpec(mesh, (Replicate(),)),
                ),
                redistribute_cost=[[1]],
            )
        ]
    )


def _scaled_dot_product_efficient_attention_base_strategies(
    op_schema: OpSchema,
) -> list[PlacementList]:
    """Helper that returns list of base placement strategies (without CP)."""
    q_input_strategy = op_schema.args_schema[0]
    if not isinstance(q_input_strategy, OpStrategy):
        raise AssertionError(f"Expected OpStrategy, got {type(q_input_strategy)}")
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

    return single_mesh_dim_strategies


@register_op_strategy(
    aten._scaled_dot_product_efficient_attention.default,
    schema_info=RuntimeSchemaInfo(4),
)
def scaled_dot_product_efficient_attention_strategy(op_schema: OpSchema) -> OpStrategy:
    # NOTE: currently we only support some simple strategies to support tensor parallelism
    mesh = op_schema.get_mesh_from_args()
    single_mesh_dim_strategies = (
        _scaled_dot_product_efficient_attention_base_strategies(op_schema)
    )
    return expand_to_full_mesh_op_strategy(
        mesh,
        op_schema,
        single_mesh_dim_strategies,
        input_index=4,
    )


def _scaled_dot_product_efficient_attention_backward_base_strategies(
    op_schema: OpSchema,
) -> list[PlacementList]:
    """Helper that returns list of base placement strategies (without CP)."""
    q_input_strategy = op_schema.args_schema[1]
    if not isinstance(q_input_strategy, OpStrategy):
        raise AssertionError(f"Expected OpStrategy, got {type(q_input_strategy)}")
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

    return single_mesh_dim_strategies


@register_op_strategy(aten._scaled_dot_product_efficient_attention_backward.default)
def scaled_dot_product_efficient_attention_backward_strategy(
    op_schema: OpSchema,
) -> OpStrategy:
    # backward op does not need to validate the mesh since forward op has already done it
    mesh = op_schema.get_mesh_from_args(validate=False)
    single_mesh_dim_strategies = (
        _scaled_dot_product_efficient_attention_backward_base_strategies(op_schema)
    )
    return expand_to_full_mesh_op_strategy(
        mesh,
        op_schema,
        single_mesh_dim_strategies,
        input_index=4,
    )


def _scaled_dot_product_cudnn_attention_base_strategies(
    op_schema: OpSchema,
) -> list[PlacementList]:
    """Helper that returns list of base placement strategies (without CP)."""
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
    debug_attn_mask_sharding: Placement | None = (
        Replicate() if return_debug_mask else None
    )

    if not isinstance(query_strategy, OpStrategy):
        raise AssertionError(f"Expected OpStrategy, got {type(query_strategy)}")
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
        # NOTE: debug_attn_mask is not supported by pytorch and is always an empty tensor
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

    return single_mesh_dim_strategies


@register_op_strategy(
    aten._scaled_dot_product_cudnn_attention.default,
    schema_info=RuntimeSchemaInfo(4),
)
def scaled_dot_product_cudnn_attention_strategy(op_schema: OpSchema) -> OpStrategy:
    mesh = op_schema.get_mesh_from_args()
    single_mesh_dim_strategies = _scaled_dot_product_cudnn_attention_base_strategies(
        op_schema
    )
    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=9
    )


def _scaled_dot_product_cudnn_attention_backward_base_strategies(
    op_schema: OpSchema,
) -> list[PlacementList]:
    """Helper that returns list of base placement strategies (without CP)."""
    if len(op_schema.args_schema) < 15:
        raise AssertionError(
            f"Expected at least 15 args_schema, got {len(op_schema.args_schema)}"
        )
    has_attn_bias = op_schema.args_schema[8] is not None
    has_scale = len(op_schema.args_schema) >= 16 and False

    query_strategy = op_schema.args_schema[1]
    if not isinstance(query_strategy, OpStrategy):
        raise AssertionError(f"Expected OpStrategy, got {type(query_strategy)}")
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

    # case 3: we can accept the sharding pattern of batch parallelism, which
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

    return single_mesh_dim_strategies


@register_op_strategy(aten._scaled_dot_product_cudnn_attention_backward.default)
def scaled_scaled_dot_product_cudnn_attention_backward_strategy(
    op_schema: OpSchema,
) -> OpStrategy:
    # backward op does not need to validate the mesh since forward op has already done it
    mesh = op_schema.get_mesh_from_args(validate=False)
    single_mesh_dim_strategies = (
        _scaled_dot_product_cudnn_attention_backward_base_strategies(op_schema)
    )
    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=3
    )


@register_op_strategy(aten._grouped_mm.default)
def grouped_mm_strategy(op_schema: OpSchema) -> OpStrategy:
    mesh = op_schema.get_mesh_from_args()

    mat1_strategy = op_schema.args_schema[0]
    if not isinstance(mat1_strategy, OpStrategy):
        raise AssertionError(f"Expected OpStrategy, got {type(mat1_strategy)}")
    mat2_strategy = op_schema.args_schema[1]
    if not isinstance(mat2_strategy, OpStrategy):
        raise AssertionError(f"Expected OpStrategy, got {type(mat2_strategy)}")
    if len(op_schema.args_schema) > 3:
        bias_strategy = op_schema.args_schema[3]
        if bias_strategy is not None:
            raise AssertionError("grouped_mm doesn't support bias yet")

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

    def valid_grouped_mm_strides(
        input_specs: list[DTensorSpec], output_specs: tuple[DTensorSpec | None, ...]
    ) -> bool:
        # 1. compute the local-tensor shape/strides given this sharding proposal
        # 2. apply the logic from the groped_mm meta function
        # UGH the input DTensorSpecs are missing their tensormetas... so i can get them another way
        def local_meta(spec: OpSpec, placements: tuple[Placement, ...]) -> TensorMeta:
            if not isinstance(spec.output_specs, DTensorSpec):
                raise AssertionError(
                    f"Expected DTensorSpec, got {type(spec.output_specs)}"
                )
            if not isinstance(spec.output_specs.tensor_meta, TensorMeta):
                raise AssertionError(
                    f"Expected TensorMeta, got {type(spec.output_specs.tensor_meta)}"
                )
            meta: TensorMeta = spec.output_specs.tensor_meta
            local_stride = compute_local_stride(meta.stride, mesh, placements)
            local_shape, _ = compute_local_shape_and_global_offset(
                meta.shape, mesh, placements, skip_offset=True
            )
            return TensorMeta(torch.Size(local_shape), local_stride, meta.dtype)

        # pyrefly: ignore [missing-attribute]
        mat1_meta = local_meta(mat1_strategy.strategies[0], input_specs[0].placements)
        # pyrefly: ignore [missing-attribute]
        mat2_meta = local_meta(mat2_strategy.strategies[0], input_specs[1].placements)

        def check_valid_strides(meta: TensorMeta) -> bool:
            # copied from `_meta_grouped_mm_common` in meta_registrations.py
            end_dim = len(meta.shape) - 1
            alignment = 16 // meta.dtype.itemsize
            if meta.stride[end_dim - 1] == 1 and meta.stride[end_dim] >= max(
                1, meta.shape[end_dim - 1]
            ):
                if meta.stride[end_dim] % alignment != 0:
                    return False
            elif meta.stride[end_dim] == 1 and meta.stride[end_dim - 1] >= max(
                1, meta.shape[end_dim]
            ):
                if meta.stride[end_dim - 1] % alignment != 0:
                    return False
            else:
                return False
            return True

        mat1_valid = check_valid_strides(mat1_meta)
        mat2_valid = check_valid_strides(mat2_meta)
        return mat1_valid and mat2_valid

    return expand_to_full_mesh_op_strategy(
        mesh,
        op_schema,
        single_mesh_dim_strategies,
        input_index=1,
        is_valid_strategy_cb=valid_grouped_mm_strides,
    )
