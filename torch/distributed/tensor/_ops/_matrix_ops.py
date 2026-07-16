# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor


import copy

import torch
from torch._ops import OpOverload
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import (
    ArgsType,
    KwargsType,
    OpSchema,
    OpStrategy,
    PlacementList,
    RuntimeSchemaInfo,
)
from torch.distributed.tensor._ops.single_dim_strategy import (
    _ShardingPlaceholder,
    register_single_dim_strategy,
)
from torch.distributed.tensor._ops.utils import prod
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
from torch.fx.experimental.symbolic_shapes import guard_or_false


aten = torch.ops.aten


@register_single_dim_strategy(aten.t.default, allow_unbacked_sharding=False)
def transpose_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    if ndim <= 1:
        for dim in range(ndim):
            strategies.append([_ShardingPlaceholder(dim), _ShardingPlaceholder(dim)])
    else:
        strategies.append([_ShardingPlaceholder(1), _ShardingPlaceholder(0)])
        strategies.append([_ShardingPlaceholder(0), _ShardingPlaceholder(1)])

    for reduce_op in ("sum", "avg", "max", "min"):
        strategies.append([Partial(reduce_op), Partial(reduce_op)])
    return strategies


def _scaled_mm_scale_placement(
    data_placement: Placement | _ShardingPlaceholder,
    scale_shape: torch.Size,
    contracting_dim: int,
) -> Placement | _ShardingPlaceholder | None:
    """
    Derive scale placement from data operand placement for _scaled_mm.

    Handles three cases:

    1. Tensor-wise scale (single element): always Replicate.
    2. 2D (or higher) scale, e.g. row-wise [M,1]: copy data placement directly.
    3. 1D blockwise scale, e.g. MX format [M*K/block_size]: map
       non-contracting shard to Shard(0)/_ShardingPlaceholder(0), and reject
       contracting-dim shards (returns None).
    """
    if prod(scale_shape) == 1:
        return Replicate()

    if len(scale_shape) != 1:
        return data_placement

    # 1D blockwise scale: Shard(>=1) is invalid on a 1D tensor, so we need
    # to map the data operand's placement to a valid 1D placement.
    if isinstance(data_placement, _ShardingPlaceholder):
        if data_placement.dim == contracting_dim:
            return None
        return _ShardingPlaceholder(0)
    # NOTE: isinstance(_, Shard) does not match _StridedShard; see _is_shard_like().
    elif isinstance(data_placement, Shard):
        if data_placement.dim == contracting_dim:
            return None
        return Shard(0)
    elif isinstance(data_placement, (Replicate, Partial)):
        return Replicate()
    return data_placement


from ._einsum_strategy import EinsumDims


def gen_single_dim_einsum_strategies(
    equation: str,
    *,
    bias_shape: torch.Size | None = None,
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

    3. Per-input linearity (Partial): Since matmul is linear in each input
       independently, one input can remain Partial while others are Replicate,
       producing a Partial output.

    4. Batch-dimension linearity (all-Partial): When all dims are batch dims
       (no contracting or free dims), the operation is element-wise and linear
       in all inputs simultaneously, so all inputs can be Partial.

    5. Bias input (optional): If bias_shape is provided, a bias placement
       is inserted after the output placement. The bias placement is derived from
       the output placement, accounting for broadcast semantics (based on ndim
       difference between output and bias). This is used for addmm-like ops
       (addmm, baddbmm) where bias + mat1 @ mat2.
    """
    # parse einop equation and extract dims
    input_dims, output_dim = EinsumDims.parse_equation(equation)
    edims = EinsumDims.parse_dims(input_dims, output_dim)

    # Compute broadcast dims map for bias if provided
    # Maps output dims to bias dims, -1 for broadcast dims (dims that don't exist in bias
    # or have size 1)
    broadcast_dims_map: list[int] | None = None
    if bias_shape is not None:
        output_ndim = len(output_dim)
        bias_ndim = len(bias_shape)
        pad_size = output_ndim - bias_ndim
        broadcast_dims_map = []
        for i in range(output_ndim):
            if i < pad_size:
                # Padded dimension (not in bias)
                broadcast_dims_map.append(-1)
            else:
                bias_dim_idx = i - pad_size
                if bias_shape[bias_dim_idx] == 1:
                    # Size-1 dimension (broadcasts)
                    broadcast_dims_map.append(-1)
                else:
                    broadcast_dims_map.append(bias_dim_idx)

    def _derive_bias_placement(
        output_placement: Placement | _ShardingPlaceholder,
    ) -> Placement | _ShardingPlaceholder:
        """Derive bias placement from output placement, accounting for broadcast."""
        if broadcast_dims_map is None:
            return copy.copy(output_placement)
        if isinstance(output_placement, _ShardingPlaceholder):
            output_dim_idx = output_placement.dim
            bias_dim = broadcast_dims_map[output_dim_idx]
            if bias_dim == -1:
                # Dim doesn't exist in bias (broadcast), replicate
                return Replicate()
            else:
                return _ShardingPlaceholder(bias_dim)
        else:
            # Clone Partial, Replicate, or other placements
            return copy.copy(output_placement)

    def _maybe_add_bias(
        placement_list: list[Placement | _ShardingPlaceholder],
    ) -> list[Placement | _ShardingPlaceholder]:
        """Insert bias placement after output if bias_shape is provided."""
        if bias_shape is None:
            return placement_list
        output_placement = placement_list[0]
        bias_placement = _derive_bias_placement(output_placement)
        return [placement_list[0], bias_placement] + placement_list[1:]

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

        strategies_over_one_mesh_dim.append(_maybe_add_bias(placement_list))

    # split contracting dim
    for contracting_dim in edims.contracting_dims:
        # Contracting dim can shard on same device axis for both inputs. This
        # results in the output being Partial on that device axis. For example:
        # bmk_{x},k_{x}n -> bmn{Ux} (becomes partial over device axis x)
        placement_list = [Partial()]
        for input_dim in input_dims:
            input_contracting_dim = input_dim.index(contracting_dim)
            placement_list.append(_ShardingPlaceholder(input_contracting_dim))

        strategies_over_one_mesh_dim.append(_maybe_add_bias(placement_list))

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
        strategies_over_one_mesh_dim.append(_maybe_add_bias(lhs_placement_list))

    # split rhs free dim
    for rhs_dim in edims.rhs_out_only_dims:
        rhs_free_dim_output = output_dim.index(rhs_dim)
        rhs_free_dim_input = input_dims[1].index(rhs_dim)
        rhs_placement_list: list[Placement | _ShardingPlaceholder] = [
            _ShardingPlaceholder(rhs_free_dim_output),
            Replicate(),
            _ShardingPlaceholder(rhs_free_dim_input),
        ]
        strategies_over_one_mesh_dim.append(_maybe_add_bias(rhs_placement_list))

    # Per-input linearity: matmul is linear in each input independently.
    # One input Partial, the other Replicate → output Partial.
    for reduce_op in Partial.LINEAR_REDUCE_OPS:
        output_placement = Partial(reduce_op)
        strategies_over_one_mesh_dim.append(
            _maybe_add_bias([output_placement, Partial(reduce_op), Replicate()])
        )
        strategies_over_one_mesh_dim.append(
            _maybe_add_bias([output_placement, Replicate(), Partial(reduce_op)])
        )

    # Batch-dimension linearity: when the einsum has no contracting dims and
    # no free dims (all dims are batch dims), the operation is element-wise
    # and linear in all inputs simultaneously. Add all-Partial strategies.
    if (
        not edims.contracting_dims
        and not edims.lhs_out_only_dims
        and not edims.rhs_out_only_dims
    ):
        for reduce_op in Partial.LINEAR_REDUCE_OPS:
            linearity_placements: list[Placement | _ShardingPlaceholder] = [
                Partial(reduce_op)
            ] + [Partial(reduce_op) for _ in input_dims]
            strategies_over_one_mesh_dim.append(_maybe_add_bias(linearity_placements))

    return strategies_over_one_mesh_dim


@register_single_dim_strategy(aten.dot.default, allow_unbacked_sharding=True)
def dot_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    return gen_single_dim_einsum_strategies("i,i->")


@register_single_dim_strategy(aten.mm.default, allow_unbacked_sharding=True)
def mm_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    return gen_single_dim_einsum_strategies("mk,kn->mn")


@register_single_dim_strategy(aten.addmm.default, allow_unbacked_sharding=True)
def addmm_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    bias_meta = args_schema[0]
    if not isinstance(bias_meta, TensorMeta):
        raise AssertionError
    return gen_single_dim_einsum_strategies("mk,kn->mn", bias_shape=bias_meta.shape)


@register_single_dim_strategy(aten.bmm.default, allow_unbacked_sharding=True)
def bmm_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    return gen_single_dim_einsum_strategies("bmk,bkn->bmn")


@register_single_dim_strategy(aten.baddbmm.default, allow_unbacked_sharding=True)
def baddbmm_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    bias_meta = args_schema[0]
    if not isinstance(bias_meta, TensorMeta):
        raise AssertionError
    return gen_single_dim_einsum_strategies("bmk,bkn->bmn", bias_shape=bias_meta.shape)


@register_single_dim_strategy(aten._scaled_mm.default, allow_unbacked_sharding=True)
def scaled_mm_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    scale_self_meta = args_schema[2]
    scale_mat2_meta = args_schema[3]
    if not isinstance(scale_self_meta, TensorMeta):
        raise AssertionError
    if not isinstance(scale_mat2_meta, TensorMeta):
        raise AssertionError
    if args_schema[4] is not None:
        raise AssertionError("_scaled_mm on DTensors doesn't support bias")
    if args_schema[5] is not None:
        raise AssertionError("_scaled_mm on DTensors doesn't support scale_result")

    # "mk,kn->mn": self_contracting_dim=1, mat2_contracting_dim=0
    base_strategies = gen_single_dim_einsum_strategies("mk,kn->mn")
    result = []
    for strat in base_strategies:
        # strat is [output, self, mat2]; derive scale placements
        scale_self_p = _scaled_mm_scale_placement(
            strat[1], scale_self_meta.shape, contracting_dim=1
        )
        scale_mat2_p = _scaled_mm_scale_placement(
            strat[2], scale_mat2_meta.shape, contracting_dim=0
        )
        if scale_self_p is None or scale_mat2_p is None:
            continue
        result.append(strat + [scale_self_p, scale_mat2_p])
    return result


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


@register_single_dim_strategy(
    aten._scaled_dot_product_flash_attention.default, schema_info=RuntimeSchemaInfo(5)
)
def scaled_dot_product_flash_attention_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    q_meta = args_schema[0]
    if not isinstance(q_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(q_meta)}")

    return_debug_mask = len(args_schema) >= 6 and args_schema[5]
    debug_attn_mask_head: Placement | _ShardingPlaceholder = (
        _ShardingPlaceholder(1) if return_debug_mask else Replicate()
    )
    debug_attn_mask_batch: Placement | _ShardingPlaceholder = (
        _ShardingPlaceholder(0) if return_debug_mask else Replicate()
    )

    return [
        [
            _ShardingPlaceholder(1),  # output
            _ShardingPlaceholder(1),  # logsumexp
            None,  # cum_seq_q
            None,  # cum_seq_k
            None,  # max_q
            None,  # max_k
            Replicate(),  # rng_state
            None,  # unused
            debug_attn_mask_head,
            _ShardingPlaceholder(1),  # q
            _ShardingPlaceholder(1),  # k
            _ShardingPlaceholder(1),  # v
        ],
        [
            _ShardingPlaceholder(0),  # output
            _ShardingPlaceholder(0),  # logsumexp
            None,  # cum_seq_q
            None,  # cum_seq_k
            None,  # max_q
            None,  # max_k
            Replicate(),  # rng_state
            None,  # unused
            debug_attn_mask_batch,
            _ShardingPlaceholder(0),  # q
            _ShardingPlaceholder(0),  # k
            _ShardingPlaceholder(0),  # v
        ],
    ]


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


@register_single_dim_strategy(aten._scaled_dot_product_flash_attention_backward.default)
def scaled_dot_product_flash_attention_backward_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    num_tensor_inputs = sum(isinstance(arg, TensorMeta) for arg in args_schema)
    if num_tensor_inputs < 6:
        raise AssertionError(
            f"Expected at least 6 tensor inputs, got {num_tensor_inputs}"
        )

    num_heads_dim_sharding: list[Placement | _ShardingPlaceholder] = [
        _ShardingPlaceholder(1),  # grad_q
        _ShardingPlaceholder(1),  # grad_k
        _ShardingPlaceholder(1),  # grad_v
        _ShardingPlaceholder(1),  # grad_out
        _ShardingPlaceholder(1),  # q
        _ShardingPlaceholder(1),  # k
        _ShardingPlaceholder(1),  # v
        _ShardingPlaceholder(1),  # output
        _ShardingPlaceholder(1),  # logsumexp
    ]
    num_heads_dim_sharding.extend([Replicate()] * (num_tensor_inputs - 6))

    batch_dim_sharding: list[Placement | _ShardingPlaceholder] = [
        _ShardingPlaceholder(0),  # grad_q
        _ShardingPlaceholder(0),  # grad_k
        _ShardingPlaceholder(0),  # grad_v
        _ShardingPlaceholder(0),  # grad_out
        _ShardingPlaceholder(0),  # q
        _ShardingPlaceholder(0),  # k
        _ShardingPlaceholder(0),  # v
        _ShardingPlaceholder(0),  # output
        _ShardingPlaceholder(0),  # logsumexp
    ]
    batch_dim_sharding.extend([Replicate()] * (num_tensor_inputs - 6))

    return [num_heads_dim_sharding, batch_dim_sharding]


@register_single_dim_strategy(
    aten.constant_pad_nd.default, schema_info=RuntimeSchemaInfo(1)
)
def constant_pad_nd_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    # Allow sharding on non-padded dimensions; ban sharding on dims
    # that have non-zero padding (where the pad value must be inserted).
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)
    pad = args_schema[1]
    if not isinstance(pad, (list, tuple)):
        raise AssertionError(f"Expected list or tuple, got {type(pad)}")

    # pad is [dim_{n-1}_left, dim_{n-1}_right, dim_{n-2}_left, ...] from
    # the last dim backwards. Determine which dims have non-zero padding.
    padded_dims = set()
    for i in range(len(pad) // 2):
        if not (
            guard_or_false(pad[i * 2] == 0) and guard_or_false(pad[i * 2 + 1] == 0)
        ):
            padded_dims.add(ndim - 1 - i)

    # Shard on any non-padded dim: output and input share the same placement.
    # All-Replicate is added automatically by the framework.
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(ndim):
        if dim not in padded_dims:
            strategies.append([_ShardingPlaceholder(dim), _ShardingPlaceholder(dim)])

    # Partial rules: at padded positions every rank writes the same constant v,
    # so reduce(v, v, ..., v) = v for avg/max/min (idempotent). P(sum) only
    # works when v=0 since sum(v, ..., v) = N*v != v otherwise.
    # When all pad amounts are zero the op is a no-op, so all reduce ops hold.
    value = args_schema[2] if len(args_schema) > 2 else 0
    no_padding = all(guard_or_false(pad[i] == 0) for i in range(len(pad)))
    if no_padding or guard_or_false(value == 0):
        reduce_ops = ("sum", "avg", "max", "min")
    else:
        reduce_ops = ("avg", "max", "min")
    for reduce_op in reduce_ops:
        strategies.append([Partial(reduce_op), Partial(reduce_op)])

    return strategies


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


@register_single_dim_strategy(
    aten._scaled_dot_product_efficient_attention.default,
    schema_info=RuntimeSchemaInfo(4),
)
def scaled_dot_product_efficient_attention_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    q_meta = args_schema[0]
    if not isinstance(q_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(q_meta)}")

    has_attn_bias = args_schema[3] is not None
    compute_log_sumexp = args_schema[4]

    logsumexp_head: Placement | _ShardingPlaceholder = (
        _ShardingPlaceholder(1) if compute_log_sumexp else Replicate()
    )
    logsumexp_batch: Placement | _ShardingPlaceholder = (
        _ShardingPlaceholder(0) if compute_log_sumexp else Replicate()
    )

    num_heads_dim_sharding: list[Placement | _ShardingPlaceholder | None] = [
        _ShardingPlaceholder(1),  # output
        logsumexp_head,
        None,  # philox_seed
        None,  # philox_offset
        _ShardingPlaceholder(1),  # q
        _ShardingPlaceholder(1),  # k
        _ShardingPlaceholder(1),  # v
    ]
    if has_attn_bias:
        num_heads_dim_sharding.append(_ShardingPlaceholder(1))

    batch_dim_sharding: list[Placement | _ShardingPlaceholder | None] = [
        _ShardingPlaceholder(0),  # output
        logsumexp_batch,
        None,  # philox_seed
        None,  # philox_offset
        _ShardingPlaceholder(0),  # q
        _ShardingPlaceholder(0),  # k
        _ShardingPlaceholder(0),  # v
    ]
    if has_attn_bias:
        batch_dim_sharding.append(_ShardingPlaceholder(0))

    return [num_heads_dim_sharding, batch_dim_sharding]


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


@register_single_dim_strategy(
    aten._scaled_dot_product_efficient_attention_backward.default
)
def scaled_dot_product_efficient_attention_backward_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    has_attn_bias = args_schema[4] is not None

    num_heads_dim_sharding: list[Placement | _ShardingPlaceholder | None] = [
        _ShardingPlaceholder(1),  # grad_q
        _ShardingPlaceholder(1),  # grad_k
        _ShardingPlaceholder(1),  # grad_v
        _ShardingPlaceholder(1) if has_attn_bias else None,  # grad_bias
        _ShardingPlaceholder(1),  # grad_out
        _ShardingPlaceholder(1),  # q
        _ShardingPlaceholder(1),  # k
        _ShardingPlaceholder(1),  # v
    ]
    if has_attn_bias:
        num_heads_dim_sharding.append(_ShardingPlaceholder(1))
    num_heads_dim_sharding.extend(
        [
            _ShardingPlaceholder(1),  # output
            _ShardingPlaceholder(1),  # logsumexp
            Replicate(),  # philox_seed
            Replicate(),  # philox_offset
        ]
    )

    batch_dim_sharding: list[Placement | _ShardingPlaceholder | None] = [
        _ShardingPlaceholder(0),  # grad_q
        _ShardingPlaceholder(0),  # grad_k
        _ShardingPlaceholder(0),  # grad_v
        _ShardingPlaceholder(0) if has_attn_bias else None,  # grad_bias
        _ShardingPlaceholder(0),  # grad_out
        _ShardingPlaceholder(0),  # q
        _ShardingPlaceholder(0),  # k
        _ShardingPlaceholder(0),  # v
    ]
    if has_attn_bias:
        batch_dim_sharding.append(_ShardingPlaceholder(0))
    batch_dim_sharding.extend(
        [
            _ShardingPlaceholder(0),  # output
            _ShardingPlaceholder(0),  # logsumexp
            Replicate(),  # philox_seed
            Replicate(),  # philox_offset
        ]
    )

    return [num_heads_dim_sharding, batch_dim_sharding]


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


@register_single_dim_strategy(
    aten._scaled_dot_product_cudnn_attention.default,
    schema_info=RuntimeSchemaInfo(4),
)
def scaled_dot_product_cudnn_attention_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    query_meta = args_schema[0]
    if not isinstance(query_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(query_meta)}")

    has_attn_bias = args_schema[3] is not None
    compute_log_sumexp = args_schema[4]
    return_debug_mask = len(args_schema) >= 8 and args_schema[7]

    logsumexp_head: Placement | _ShardingPlaceholder = (
        _ShardingPlaceholder(1) if compute_log_sumexp else Replicate()
    )
    logsumexp_batch: Placement | _ShardingPlaceholder = (
        _ShardingPlaceholder(0) if compute_log_sumexp else Replicate()
    )
    debug_attn_mask_head: Placement | _ShardingPlaceholder | None = (
        _ShardingPlaceholder(1) if return_debug_mask else None
    )
    debug_attn_mask_batch: Placement | _ShardingPlaceholder | None = (
        _ShardingPlaceholder(0) if return_debug_mask else None
    )

    num_heads_dim_sharding: list[Placement | _ShardingPlaceholder | None] = [
        _ShardingPlaceholder(1),  # output
        logsumexp_head,
        None,  # cum_seq_q
        None,  # cum_seq_k
        None,  # max_q
        None,  # max_k
        None,  # philox_seed
        None,  # philox_offset
        debug_attn_mask_head,
        _ShardingPlaceholder(1),  # q
        _ShardingPlaceholder(1),  # k
        _ShardingPlaceholder(1),  # v
    ]
    if has_attn_bias:
        num_heads_dim_sharding.append(_ShardingPlaceholder(1))

    batch_dim_sharding: list[Placement | _ShardingPlaceholder | None] = [
        _ShardingPlaceholder(0),  # output
        logsumexp_batch,
        None,  # cum_seq_q
        None,  # cum_seq_k
        None,  # max_q
        None,  # max_k
        None,  # philox_seed
        None,  # philox_offset
        debug_attn_mask_batch,
        _ShardingPlaceholder(0),  # q
        _ShardingPlaceholder(0),  # k
        _ShardingPlaceholder(0),  # v
    ]
    if has_attn_bias:
        batch_dim_sharding.append(_ShardingPlaceholder(0))

    return [num_heads_dim_sharding, batch_dim_sharding]


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


@register_single_dim_strategy(aten._scaled_dot_product_cudnn_attention_backward.default)
def scaled_dot_product_cudnn_attention_backward_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    if len(args_schema) < 15:
        raise AssertionError(f"Expected at least 15 args, got {len(args_schema)}")

    for arg_index in range(6):
        arg = args_schema[arg_index]
        if not isinstance(arg, TensorMeta):
            raise AssertionError(f"Expected TensorMeta, got {type(arg)}")

    philox_placements: list[Placement] = []
    for arg_index in (6, 7):
        arg = args_schema[arg_index]
        if isinstance(arg, TensorMeta):
            philox_placements.append(Replicate())
        elif not isinstance(arg, torch.Tensor):
            raise AssertionError(f"Expected TensorMeta or Tensor, got {type(arg)}")

    has_attn_bias = args_schema[8] is not None
    if has_attn_bias and not isinstance(args_schema[8], (TensorMeta, torch.Tensor)):
        raise AssertionError(
            f"Expected TensorMeta or Tensor, got {type(args_schema[8])}"
        )

    cum_seq_placements: list[None] = []
    for arg_index in (9, 10):
        arg = args_schema[arg_index]
        if isinstance(arg, TensorMeta):
            cum_seq_placements.append(None)
        elif arg is None or isinstance(arg, torch.Tensor):
            pass
        else:
            raise AssertionError(f"Expected TensorMeta or Tensor, got {type(arg)}")

    num_heads_dim_sharding: list[Placement | _ShardingPlaceholder | None] = [
        _ShardingPlaceholder(1),  # grad_q
        _ShardingPlaceholder(1),  # grad_k
        _ShardingPlaceholder(1),  # grad_v
        _ShardingPlaceholder(1),  # grad_out
        _ShardingPlaceholder(1),  # q
        _ShardingPlaceholder(1),  # k
        _ShardingPlaceholder(1),  # v
        _ShardingPlaceholder(1),  # output
        _ShardingPlaceholder(1),  # logsumexp
    ]
    num_heads_dim_sharding.extend(philox_placements)
    if has_attn_bias and isinstance(args_schema[8], TensorMeta):
        num_heads_dim_sharding.append(_ShardingPlaceholder(1))
    num_heads_dim_sharding.extend(cum_seq_placements)

    batch_dim_sharding: list[Placement | _ShardingPlaceholder | None] = [
        _ShardingPlaceholder(0),  # grad_q
        _ShardingPlaceholder(0),  # grad_k
        _ShardingPlaceholder(0),  # grad_v
        _ShardingPlaceholder(0),  # grad_out
        _ShardingPlaceholder(0),  # q
        _ShardingPlaceholder(0),  # k
        _ShardingPlaceholder(0),  # v
        _ShardingPlaceholder(0),  # output
        _ShardingPlaceholder(0),  # logsumexp
    ]
    batch_dim_sharding.extend(philox_placements)
    if has_attn_bias and isinstance(args_schema[8], TensorMeta):
        batch_dim_sharding.append(_ShardingPlaceholder(0))
    batch_dim_sharding.extend(cum_seq_placements)

    return [num_heads_dim_sharding, batch_dim_sharding]


def _valid_grouped_mm_strides(
    mesh: DeviceMesh,
    op_schema: OpSchema,
    input_specs: list[DTensorSpec],
    output_specs: DTensorSpec | tuple[DTensorSpec | None, ...],
) -> bool:
    def local_meta(spec: DTensorSpec) -> TensorMeta:
        if not isinstance(spec.tensor_meta, TensorMeta):
            raise AssertionError(f"Expected TensorMeta, got {type(spec.tensor_meta)}")
        meta: TensorMeta = spec.tensor_meta
        local_shape, _ = compute_local_shape_and_global_offset(
            meta.shape, mesh, spec.placements, skip_offset=True
        )
        local_stride = compute_local_stride(meta.stride, local_shape)
        return TensorMeta(torch.Size(local_shape), local_stride, meta.dtype)

    def check_valid_strides(meta: TensorMeta) -> bool:
        # copied from `_meta_grouped_mm_common` in meta_registrations.py
        end_dim = len(meta.shape) - 1
        alignment = 16 // meta.dtype.itemsize
        if meta.stride[end_dim - 1] == 1 and meta.stride[end_dim] >= max(
            1, meta.shape[end_dim - 1]
        ):
            return meta.stride[end_dim] % alignment == 0
        elif meta.stride[end_dim] == 1 and meta.stride[end_dim - 1] >= max(
            1, meta.shape[end_dim]
        ):
            return meta.stride[end_dim - 1] % alignment == 0
        else:
            return False

    mat1_meta = local_meta(input_specs[0])
    mat2_meta = local_meta(input_specs[1])
    return check_valid_strides(mat1_meta) and check_valid_strides(mat2_meta)


@register_single_dim_strategy(
    aten._grouped_mm.default,
    full_mesh_strategy_filter=_valid_grouped_mm_strides,
)
def grouped_mm_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    mat1_meta = args_schema[0]
    if not isinstance(mat1_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(mat1_meta)}")
    mat2_meta = args_schema[1]
    if not isinstance(mat2_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(mat2_meta)}")

    if len(args_schema) > 3 and args_schema[3] is not None:
        raise AssertionError("grouped_mm doesn't support bias yet")

    tensor_inputs_tail: list[Placement | None] = []
    if len(args_schema) > 2 and args_schema[2] is not None:
        if not isinstance(args_schema[2], TensorMeta):
            raise AssertionError(f"Expected TensorMeta, got {type(args_schema[2])}")
        tensor_inputs_tail.append(Replicate())

    strategies: list[list[Placement | _ShardingPlaceholder | None]] = [
        [Partial(), Partial(), Replicate(), *tensor_inputs_tail],
        [Partial(), Replicate(), Partial(), *tensor_inputs_tail],
    ]

    mat1_ndim = len(mat1_meta.shape)
    mat2_ndim = len(mat2_meta.shape)

    if mat1_ndim == 2 and mat2_ndim == 3:
        strategies.extend(
            [
                [
                    _ShardingPlaceholder(1),
                    Replicate(),
                    _ShardingPlaceholder(2),
                    *tensor_inputs_tail,
                ],
                [
                    Partial(),
                    _ShardingPlaceholder(1),
                    _ShardingPlaceholder(1),
                    *tensor_inputs_tail,
                ],
            ]
        )

    if mat1_ndim == 3 and mat2_ndim == 2:
        strategies.extend(
            [
                [
                    Partial(),
                    _ShardingPlaceholder(2),
                    _ShardingPlaceholder(0),
                    *tensor_inputs_tail,
                ],
                [
                    _ShardingPlaceholder(0),
                    _ShardingPlaceholder(1),
                    Replicate(),
                    *tensor_inputs_tail,
                ],
            ]
        )

    if mat1_ndim == 2 and mat2_ndim == 2:
        strategies.extend(
            [
                [
                    _ShardingPlaceholder(2),
                    Replicate(),
                    _ShardingPlaceholder(1),
                    *tensor_inputs_tail,
                ],
                [
                    _ShardingPlaceholder(1),
                    _ShardingPlaceholder(0),
                    Replicate(),
                    *tensor_inputs_tail,
                ],
            ]
        )

    if mat1_ndim == 3 and mat2_ndim == 3:
        strategies.extend(
            [
                [
                    _ShardingPlaceholder(2),
                    Replicate(),
                    _ShardingPlaceholder(2),
                    *tensor_inputs_tail,
                ],
                [
                    _ShardingPlaceholder(1),
                    _ShardingPlaceholder(1),
                    Replicate(),
                    *tensor_inputs_tail,
                ],
                [
                    Partial(),
                    _ShardingPlaceholder(2),
                    _ShardingPlaceholder(1),
                    *tensor_inputs_tail,
                ],
                [
                    _ShardingPlaceholder(0),
                    _ShardingPlaceholder(0),
                    _ShardingPlaceholder(0),
                    *tensor_inputs_tail,
                ],
            ]
        )

    return strategies
