# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
from collections.abc import Sequence, Sized
from typing import cast, Optional

import torch
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpSpec,
    OpStrategy,
    OutputSharding,
    PlacementList,
    RuntimeSchemaInfo,
    StrategyType,
    TupleStrategy,
)
from torch.distributed.tensor._ops._common_rules import pointwise_rule
from torch.distributed.tensor._ops._embedding_ops import _MaskPartial
from torch.distributed.tensor._ops.utils import (
    expand_to_full_mesh_op_strategy,
    generate_redistribute_costs,
    is_tensor_dim_sharded,
    is_tensor_evenly_shardable,
    is_tensor_partial,
    normalize_dim,
    register_op_strategy,
    register_prop_rule,
)
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)


aten = torch.ops.aten


def default_strategy(op_schema: OpSchema) -> StrategyType:
    # Default strategy by default just propagate the first input strategy
    select_strategy = op_schema.args_schema[0]
    assert isinstance(select_strategy, OpStrategy)
    # we create new DTensorSpecs even for default strategy to assure that
    # the tensor metas are distinct between the arguments and outputs
    default_strategy = [
        OpSpec(
            output_specs=DTensorSpec(
                mesh=select_strategy.mesh,
                placements=strategy.output_spec.placements,
            )
        )
        for strategy in select_strategy.strategies
    ]
    return OpStrategy(default_strategy)


register_op_strategy(
    [
        aten.clone.default,
        aten.contiguous.default,
        aten.copy_.default,
        aten.detach.default,
        aten.fill_.Scalar,
        aten.view.dtype,
        aten.zero_.default,
    ]
)(default_strategy)


register_op_strategy(
    aten._to_copy.default, schema_info=RuntimeSchemaInfo(static_kwargkey=["dtype"])
)(default_strategy)


@register_op_strategy(
    [
        aten.equal.default,
        aten.is_same_size.default,
    ]
)
def equal_strategy(op_schema: OpSchema) -> StrategyType:
    # equal_strategy deals with ops that comparing two tensor, we need to make sure
    # sharding layout the same with two operands, we choose to follow the arg with max
    # num of shards, still keep is_same_size here for completeness as they share the
    # same strategy in theory.
    mesh = op_schema.get_mesh_from_args()
    self_strategy, other_strategy = op_schema.args_schema
    assert isinstance(self_strategy, OpStrategy)
    assert isinstance(other_strategy, OpStrategy)

    select_strategy = (
        self_strategy
        if self_strategy.max_num_shards() >= other_strategy.max_num_shards()
        else other_strategy
    )
    equal_strategy = OpStrategy([])

    for arg_strategy in select_strategy.strategies:
        arg_spec = arg_strategy.output_spec
        if is_tensor_partial(arg_spec):
            # if the arg_spec have partial, reshard to replicate
            # otherwise local shard tensor comparison would be invalid
            output_spec = DTensorSpec(
                mesh=mesh,
                placements=tuple(
                    Replicate() if isinstance(p, Partial) else p
                    for p in arg_spec.placements
                ),
            )
            equal_strategy.strategies.append(OpSpec(output_specs=output_spec))
        else:
            equal_strategy.strategies.append(OpSpec(arg_spec))
    return equal_strategy


@register_op_strategy(
    [
        aten.empty_like.default,
        aten.ones_like.default,
        aten.rand_like.default,
        aten.randn_like.default,
        aten.zeros_like.default,
    ],
    schema_info=RuntimeSchemaInfo(1, ["dtype"]),
)
@register_op_strategy(
    [aten.full_like.default],
    schema_info=RuntimeSchemaInfo(2, ["dtype"]),
)
@register_op_strategy(
    [
        aten.randint_like.default,
        aten.randint_like.low_dtype,
        aten.randint_like.low_dtype_out,
    ],
    schema_info=RuntimeSchemaInfo(3, ["dtype"]),
)
def create_like_strategy(op_schema: OpSchema) -> StrategyType:
    # create_like_strategy deals with ops that creating tensors with same
    # shape as input, but with specific content that does not depend on
    # the input, we can propagate sharding, but we have to make sure we
    # move from partial to replicated.
    select_strategy = op_schema.args_schema[0]
    create_like_strategy = OpStrategy([])
    assert isinstance(select_strategy, OpStrategy)
    for arg_strategy in select_strategy.strategies:
        arg_spec = arg_strategy.output_spec
        output_spec = DTensorSpec(
            mesh=select_strategy.mesh,
            placements=tuple(
                Replicate() if isinstance(p, Partial) else p
                for p in arg_spec.placements
            ),
        )
        create_like_strategy.strategies.append(
            OpSpec(output_specs=output_spec, input_specs=(arg_spec,))
        )

    return create_like_strategy


@register_op_strategy(
    [
        aten.new_empty.default,
        aten.new_full.default,
        aten.new_ones.default,
        aten.new_zeros.default,
        aten.new_empty_strided.default,
    ],
    schema_info=RuntimeSchemaInfo(1, ["dtype"]),
)
def new_factory_strategy(op_schema: OpSchema) -> StrategyType:
    # Currently there are two strategies:
    # 1. let the output be replicated
    # 2. let the output follow the input if input and output have the same shape
    input_strategy = op_schema.args_schema[0]
    assert isinstance(input_strategy, OpStrategy)

    mesh = input_strategy.mesh
    input_shape = input_strategy.shape
    output_shape = op_schema.args_schema[1]
    assert isinstance(output_shape, list)

    new_factory_strategy = OpStrategy([])
    for arg_strategy in input_strategy.strategies:
        input_spec = arg_strategy.output_spec
        replica_spec = DTensorSpec(mesh, tuple([Replicate()] * mesh.ndim))
        new_factory_strategy.strategies.append(
            OpSpec(
                output_specs=replica_spec,
                input_specs=(input_spec,),
                redistribute_cost=[[0.0] * mesh.ndim],
            )
        )

        if tuple(input_shape) == tuple(output_shape) and input_spec.is_sharded():
            # NOTE: for new_empty_strided, currently the non-replicate sharding
            #       is supported only when the shape is evenly shardable
            if (
                op_schema.op == aten.new_empty_strided.default
                and not is_tensor_evenly_shardable(input_shape, input_spec)
            ):
                continue

            new_factory_strategy.strategies.append(
                OpSpec(
                    output_specs=input_spec,
                    input_specs=(input_spec,),
                    # encouraging new tensor placement to be the same as input
                    redistribute_cost=[[-0.1] * mesh.ndim],
                )
            )

    return new_factory_strategy


@register_op_strategy(aten.bucketize.Tensor)
def gen_bucketize_strategy(op_schema: OpSchema) -> StrategyType:
    """Just propagate input sharding, but expect replicated for boundaries input."""
    mesh = op_schema.get_mesh_from_args()
    input_strategy = op_schema.args_schema[0]
    bucketize_strategy = OpStrategy([])
    assert isinstance(input_strategy, OpStrategy)
    for arg_strategy in input_strategy.strategies:
        arg_spec = DTensorSpec(mesh, arg_strategy.output_spec.placements)
        replica_spec = DTensorSpec(mesh, tuple([Replicate()] * mesh.ndim))
        bucketize_strategy.strategies.append(
            OpSpec(output_specs=arg_spec, input_specs=(arg_spec, replica_spec))
        )

    return bucketize_strategy


@register_op_strategy(aten.select.int, schema_info=RuntimeSchemaInfo(1))
def select_int_strategy(op_schema: OpSchema) -> StrategyType:
    """
    In this select op, first determine the input specs, then determine the output specs.
    - Input specs:
        - If the input is sharded on the selected dim, unshard it and change to replicate.
        - Otherwise, keep the original input specs.
    - Output specs:
        - It checks the input specs with the following cases:
        - Case 1 shard_dim == selected_dim: not possible as the input is already unsharded.
        - Case 2 shard_dim < selected_dim: keep the input specs.
        - Case 3 shard_dim > selected_dim: shard_dim -= 1.
    """
    input_strategy = op_schema.args_schema[0]
    assert isinstance(input_strategy, OpStrategy)
    assert len(op_schema.args_schema) == 3
    selected_dim, index = (
        cast(int, op_schema.args_schema[1]),
        cast(int, op_schema.args_schema[2]),
    )
    input_shape = input_strategy.shape
    input_ndim = input_strategy.ndim
    selected_dim = normalize_dim(selected_dim, input_ndim)
    index = normalize_dim(index, input_shape[selected_dim])

    select_strategy = OpStrategy([])
    for arg_strategy in input_strategy.strategies:
        arg_spec = arg_strategy.output_spec

        # determine input spec
        input_specs = arg_spec
        if is_tensor_dim_sharded(arg_spec, dim=selected_dim):
            # if input is sharded on the selected dim, need to unshard it, change to replicate
            arg_target_placements = unshard_tensor_dim(
                arg_spec.placements, dim=selected_dim
            )
            input_specs = DTensorSpec(arg_spec.mesh, arg_target_placements)  # R

        # determine output spec
        output_specs = input_specs
        if input_specs.is_sharded():
            # handle cases with sharded_dim != selected_dim
            output_spec_placements = []
            for placement in input_specs.placements:
                if placement.is_shard():
                    shard_dim = cast(Shard, placement).dim
                    if shard_dim > selected_dim:
                        shard_dim -= 1
                    placement = Shard(dim=shard_dim)
                output_spec_placements.append(placement)
            output_specs = DTensorSpec(
                arg_spec.mesh, placements=tuple(output_spec_placements)
            )

        select_strategy.strategies.append(
            OpSpec(
                output_specs=output_specs,
                input_specs=(input_specs,),
            )
        )
    return select_strategy


@register_op_strategy(
    aten.select_backward.default,
    schema_info=RuntimeSchemaInfo(1),
)
def select_backward_strategy(op_schema: OpSchema) -> OpStrategy:
    # func: select_backward(Tensor grad_output, SymInt[] input_sizes, int dim, SymInt index) -> Tensor
    args_schema = op_schema.args_schema
    input_strategy, dim = args_schema[0], args_schema[2]
    assert isinstance(input_strategy, OpStrategy), f"{input_strategy}"
    assert isinstance(dim, int)
    output_strategies: list[OpSpec] = []
    for placement_strategy in input_strategy.strategies:
        input_spec = placement_strategy.output_spec
        output_spec_placements: list[Placement] = []
        for placement in input_spec.placements:
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                if shard_dim >= dim:
                    # NOTE: shard_dim is guaranteed to exist because
                    # grad_input has one more dim than grad_output
                    output_spec_placements.append(Shard(shard_dim + 1))
                else:
                    output_spec_placements.append(Shard(shard_dim))
            else:
                output_spec_placements.append(placement)
        output_specs = DTensorSpec(input_spec.mesh, tuple(output_spec_placements))
        output_strategies.append(
            OpSpec(output_specs=output_specs, input_specs=(input_spec,))
        )
    return OpStrategy(output_strategies)


@register_op_strategy(aten.slice.Tensor, schema_info=RuntimeSchemaInfo(1))
def gen_slice_strategy(op_schema: OpSchema) -> StrategyType:
    """Forward all shardings except the slice dimension."""
    defaults = (None, 0, None, None, 1)
    input_strategy, dim, start, end, step = (
        op_schema.args_schema + defaults[len(op_schema.args_schema) :]
    )
    assert isinstance(input_strategy, OpStrategy)

    mesh = input_strategy.mesh
    input_shape = input_strategy.shape
    input_ndim = input_strategy.ndim
    assert isinstance(dim, int)
    if start is None:
        start = 0
    if end is None or end > input_shape[dim]:
        end = input_shape[dim]
    assert isinstance(start, int)
    assert isinstance(end, int)
    assert isinstance(step, int)

    # normalize args
    slice_dim = normalize_dim(dim, input_ndim)
    start = normalize_dim(start, input_shape[dim])
    end = normalize_dim(end, input_shape[dim])

    redundant_slice = start == 0 and end == input_shape[dim] and step == 1

    slice_strategy = OpStrategy([])

    for arg_strategy in input_strategy.strategies:
        arg_spec = arg_strategy.output_spec
        if not is_tensor_dim_sharded(arg_spec, dim=slice_dim) or redundant_slice:
            # only add the strategy if the slice dim is not sharded
            out_spec = DTensorSpec(mesh, arg_spec.placements)
            slice_strategy.strategies.append(OpSpec(output_specs=out_spec))
    if not slice_strategy.strategies:
        # if all strategies are filtered out, unsharding all specs on slice dim
        # of the input strategy, and use that as the op strategy
        for arg_strategy in input_strategy.strategies:
            arg_spec = arg_strategy.output_spec
            unshard_spec = DTensorSpec(
                mesh, unshard_tensor_dim(arg_spec.placements, dim=slice_dim)
            )
            slice_strategy.strategies.append(OpSpec(output_specs=unshard_spec))
    return slice_strategy


@register_op_strategy(
    aten.slice_backward.default,
    schema_info=RuntimeSchemaInfo(1),
)
def slice_backward_rules(op_schema: OpSchema) -> OpStrategy:
    # func: slice_backward(Tensor grad_output, SymInt[] input_sizes, int dim, SymInt start, SymInt end, SymInt step) -> Tensor
    args_schema = op_schema.args_schema
    input_strategy, dim = args_schema[0], args_schema[2]
    assert isinstance(input_strategy, OpStrategy), f"{input_strategy}"
    output_strategies: list[OpSpec] = []
    for placement_strategy in input_strategy.strategies:
        output_spec = placement_strategy.output_spec
        new_placements: list[Placement] = []
        for placement in output_spec.placements:
            # Redistribute to replicate only if the dim is sharded and matches the slice dim
            if isinstance(placement, Shard) and placement.dim == dim:
                new_placements.append(Replicate())
            else:
                new_placements.append(placement)
        new_spec = DTensorSpec(output_spec.mesh, tuple(new_placements))
        redistribute_cost = [generate_redistribute_costs(input_strategy, new_spec)]
        placement_strategy.redistribute_cost = redistribute_cost
        new_strategy = OpSpec(output_specs=new_spec)
        output_strategies.append(new_strategy)
    return OpStrategy(output_strategies)


def unshard_tensor_dim(
    placements: Sequence[Placement], dim: int
) -> tuple[Placement, ...]:
    """Disallow the given tensor dimension to be sharded."""
    return tuple(
        p if (not isinstance(p, Shard) or p.dim != dim) else Replicate()
        for p in placements
    )


def replicate_tensor_dim(
    placements: Sequence[Placement], dim: int
) -> tuple[Placement, ...]:
    """Force the given tensor dimension to be replicated."""
    # Not using p.is_shard() to avoid mypy complain about Placement not having
    # attribute dim.
    return tuple(
        Replicate() if p.is_partial() or isinstance(p, Shard) and p.dim == dim else p
        for p in placements
    )


@register_op_strategy(aten.slice_scatter.default, schema_info=RuntimeSchemaInfo(2))
def gen_slice_scatter_strategy(op_schema: OpSchema) -> StrategyType:
    # 1. number of dimensions in input and src need to match.
    # 2. number of elements on all non-dim need to match between input and src.
    # 3. numer of elements in src in dim need to match the slice size.
    # Given the above:
    # - We suggest for src to follow the sharding of input, except on the scatter dimension,
    #   where our best bet for now is to make them replicated as a fall-back.
    #   TODO: Ideally we'd like to make sure the output is re-sharded afterwards to keep input sharding.
    mesh = op_schema.get_mesh_from_args()
    input_strategy = op_schema.args_schema[0]
    assert isinstance(input_strategy, OpStrategy)
    input_ndim = input_strategy.ndim
    slice_dim = (
        cast(int, op_schema.args_schema[2]) if len(op_schema.args_schema) > 2 else 0
    )
    slice_dim = normalize_dim(slice_dim, input_ndim)

    slice_scatter_strategy = OpStrategy([])
    # by default follow the input strategy for both input and src
    for arg_strategy in input_strategy.strategies:
        arg_spec = arg_strategy.output_spec
        if not (
            is_tensor_dim_sharded(arg_spec, dim=slice_dim)
            or is_tensor_partial(arg_spec)
        ):
            # only add the strategy if the slice_scatter dim is not sharded or partial
            slice_scatter_strategy.strategies.append(OpSpec(output_specs=arg_spec))

    if not slice_scatter_strategy.strategies:
        # if all strategies are filtered out, replicating all specs on slice_scatter dim
        # of the input strategy, and use that as the op strategy
        for arg_strategy in input_strategy.strategies:
            arg_spec = arg_strategy.output_spec
            replicate_spec = DTensorSpec(
                mesh, replicate_tensor_dim(arg_spec.placements, dim=slice_dim)
            )
            slice_scatter_strategy.strategies.append(
                OpSpec(output_specs=replicate_spec)
            )
    return slice_scatter_strategy


@register_op_strategy(aten._local_scalar_dense.default)
def replica_only_strategy(op_schema: OpSchema) -> StrategyType:
    """Only allow replication on the input/output."""
    input_strategy = op_schema.args_schema[0]
    assert isinstance(input_strategy, OpStrategy)
    mesh = input_strategy.mesh
    replicate_spec = DTensorSpec(mesh, tuple([Replicate()] * mesh.ndim))
    return OpStrategy([OpSpec(replicate_spec)])


@register_op_strategy(
    [aten.scatter_.value, aten.scatter.value, aten.scatter_.src, aten.scatter.src],
    schema_info=RuntimeSchemaInfo(1),
)
def scatter_strategy(op_schema: OpSchema) -> StrategyType:
    mesh = op_schema.get_mesh_from_args()
    single_mesh_dim_strategies = []

    # placement list stores placements of [output, input, index, src]
    # first we always have replicate all for inputs and output
    if len(op_schema.args_strategy) < 3:
        # scatter_.src/scatter.src with src be float number instead of tensor
        all_replicate: PlacementList = [Replicate()] * 3
    else:
        all_replicate = [Replicate()] * 4
    single_mesh_dim_strategies.append(all_replicate)

    # TODO: see if we can support input sharding pattern
    op_strategy = expand_to_full_mesh_op_strategy(
        mesh,
        op_schema,
        single_mesh_dim_strategies,
        inplace_op=op_schema.is_inplace_op(),
    )
    return op_strategy


@register_op_strategy(aten.gather.default)
def gather_strategy(op_schema: OpSchema) -> StrategyType:
    mesh = op_schema.get_mesh_from_args()
    input_strategy = cast(OpStrategy, op_schema.args_schema[0])
    dim = cast(int, op_schema.args_schema[1])
    index_strategy = cast(OpStrategy, op_schema.args_schema[2])

    input_shape = input_strategy.shape
    index_shape = index_strategy.shape

    single_mesh_dim_strategies = []

    # placement list stores placements of [output, input, index]
    # first we always have replicate all for inputs and output
    all_replicate: PlacementList = [Replicate()] * 3
    single_mesh_dim_strategies.append(all_replicate)

    # input sharding, input sharded, index accepts mask partial, output follows index
    # this only works when the input is sharded on the gather dimension, and
    # index has size 1 on the gather dimension
    if index_shape[dim] == 1:
        index_partial_placement = _MaskPartial(offset_shape=input_shape, offset_dim=dim)
        input_sharding: PlacementList = [
            index_partial_placement,
            Shard(dim),
            index_partial_placement,
        ]
        single_mesh_dim_strategies.append(input_sharding)

    # index sharding, input replicated, index sharded, output follows index
    # this only works when the sharding dimension is the gather dimension
    index_sharding: PlacementList = [Shard(dim), Replicate(), Shard(dim)]
    single_mesh_dim_strategies.append(index_sharding)

    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_mesh_dim_strategies, input_index=1
    )


def _derive_follow_placements_from_tuple_strategy(
    op: torch._ops.OpOverload,
    tuple_strategy: TupleStrategy,
) -> Sequence[Placement]:
    """
    derive the placements to follow from the tuple strategy, mainly used by
    aten.stack, aten.cat, where each operand have the same shape, and correspondingly
    expecting the same sharding
    """

    def merge_placement(
        cur_placement: Placement, new_placement: Placement
    ) -> Placement:
        # semantic if we already have a follow placement, we
        # check each placement for the current arg placement
        # to see if we want to merge/adjust the placement to follow
        # the priority: Partial -> Shard -> Replicate
        if cur_placement == new_placement:
            return cur_placement

        if cur_placement.is_partial():
            if new_placement.is_shard():
                # follow new placement
                return new_placement
            elif new_placement.is_partial():
                # different partial types, we can't merge and have to replicate all here
                return Replicate()
            else:
                # follow partial
                return cur_placement
        elif cur_placement.is_shard():
            if new_placement.is_shard():
                # cur/new placement are different sharding (i.e. different shard dim)
                # currently fallback to replicate all args
                return Replicate()
            else:
                # for partial/replicate, follow the current shard placement
                return cur_placement
        else:
            # current replicate, just follow new placement
            return new_placement

    follow_placements: Optional[list[Placement]] = None
    mesh = tuple_strategy.child_mesh(0)
    for arg_strategy in tuple_strategy.childs:
        assert isinstance(arg_strategy, OpStrategy)
        if arg_strategy.mesh != mesh:
            raise ValueError(
                f"All operands in {op} must have the same mesh, "
                f"but got {arg_strategy.mesh} and {mesh}."
            )

        for placement_strategy in arg_strategy.strategies:
            arg_placements = placement_strategy.output_spec.placements
            if follow_placements is None:
                follow_placements = list(arg_placements)
                continue
            assert follow_placements is not None
            for mesh_idx in range(mesh.ndim):
                # merge placements with the priority
                follow_placements[mesh_idx] = merge_placement(
                    follow_placements[mesh_idx], arg_placements[mesh_idx]
                )
    assert follow_placements is not None, "follow placements should not be None!"
    return follow_placements


def normalize_shard_for_stack(
    placements: Sequence[Placement], insert_dim: int = 0
) -> Sequence[Placement]:
    # stack op would "insert" new dim, so all sharded dim >= the inserted dim need to
    # be normalized with the new Shard placement
    normalized_placements: list[Placement] = []
    for placement in placements:
        if isinstance(placement, Shard) and placement.dim >= insert_dim:
            normalized_placements.append(Shard(placement.dim + 1))
        else:
            normalized_placements.append(placement)
    return normalized_placements


@register_op_strategy(aten.stack.default, RuntimeSchemaInfo(1, needs_pytree=True))
def stack_strategy(op_schema: OpSchema) -> StrategyType:
    args_schema = op_schema.args_schema
    input_tuple_strategy = args_schema[0]
    assert isinstance(input_tuple_strategy, TupleStrategy), f"{input_tuple_strategy}"
    first_input_strategy = input_tuple_strategy.childs[0]
    assert isinstance(first_input_strategy, OpStrategy), f"{first_input_strategy}"
    common_input_ndim = first_input_strategy.ndim
    dim = cast(int, args_schema[1]) if len(args_schema) > 1 else 0
    # normalize the dim to be within the common input ndim
    dim = normalize_dim(dim, common_input_ndim)

    mesh = first_input_strategy.mesh

    follow_placements = _derive_follow_placements_from_tuple_strategy(
        op_schema.op, input_tuple_strategy
    )

    # create op strategy base on the follow placements
    op_strategy = OpStrategy([])

    input_specs = tuple(
        DTensorSpec(mesh, tuple(follow_placements))
        for _ in range(len(input_tuple_strategy.childs))
    )

    follow_placements = normalize_shard_for_stack(follow_placements, dim)

    op_strategy.strategies.append(
        OpSpec(
            output_specs=DTensorSpec(mesh, tuple(follow_placements)),
            input_specs=input_specs,
        )
    )
    return op_strategy


@register_op_strategy(aten.cat.default, RuntimeSchemaInfo(1, needs_pytree=True))
def cat_strategy(op_schema: OpSchema) -> StrategyType:
    args_schema = op_schema.args_schema
    input_tuple_strategy = args_schema[0]
    assert isinstance(input_tuple_strategy, TupleStrategy), f"{input_tuple_strategy}"
    first_input_strategy = input_tuple_strategy.childs[0]
    assert isinstance(first_input_strategy, OpStrategy), f"{first_input_strategy}"
    common_input_ndim = first_input_strategy.ndim
    dim = cast(int, args_schema[1]) if len(args_schema) > 1 else 0
    # normalize the dim to be within the common input ndim
    dim = normalize_dim(dim, common_input_ndim)

    mesh = first_input_strategy.mesh

    follow_placements = _derive_follow_placements_from_tuple_strategy(
        op_schema.op, input_tuple_strategy
    )
    # for cat we unshard the cat dim if it is sharded
    follow_placements = unshard_tensor_dim(follow_placements, dim)

    # create op strategy base on the follow placements
    op_strategy = OpStrategy([])

    input_specs = tuple(
        DTensorSpec(mesh, tuple(follow_placements))
        for _ in range(len(input_tuple_strategy.childs))
    )
    op_strategy.strategies.append(
        OpSpec(
            output_specs=DTensorSpec(mesh, tuple(follow_placements)),
            input_specs=input_specs,
        )
    )
    return op_strategy


@register_prop_rule(aten.index_select.default, schema_info=RuntimeSchemaInfo(1))
def prop_index_select(op_schema: OpSchema) -> OutputSharding:
    values_spec, dim, indices_spec = op_schema.args_schema

    assert isinstance(values_spec, DTensorSpec)
    assert isinstance(dim, int)
    assert isinstance(indices_spec, DTensorSpec)

    all_indices_spec: list[Optional[DTensorSpec]] = [
        indices_spec if dim == i else None for i in range(values_spec.ndim)
    ]

    result = prop_index(
        OpSchema(
            op=op_schema.op,
            args_schema=(values_spec, all_indices_spec),
            kwargs_schema=op_schema.kwargs_schema,
        )
    )
    if result.redistribute_schema:
        schema_suggestion = result.redistribute_schema
        result.redistribute_schema = OpSchema(
            op=op_schema.op,
            args_schema=(
                schema_suggestion.args_schema[0],
                dim,
                schema_suggestion.args_schema[1][dim],  # type: ignore[index]
            ),
            kwargs_schema=op_schema.kwargs_schema,
        )
    return result


@register_op_strategy(
    [
        aten.index_put.default,
        aten._index_put_impl_.default,
    ],
    schema_info=RuntimeSchemaInfo(needs_pytree=True),
)
def prop_index_put(op_schema: OpSchema) -> StrategyType:
    # We have 3 DTensor spec from argument `in`, `indices` and `values`
    # accordingly.
    in_spec, indices_spec, values_spec = op_schema.args_schema
    assert isinstance(in_spec, OpStrategy)
    # `indices`` is a tuple of scalar LongTensor, so we use TupleStrategy.
    assert isinstance(indices_spec, TupleStrategy)
    assert isinstance(values_spec, OpStrategy)
    mesh = values_spec.mesh
    op_strategy = OpStrategy([])
    # 1. `indices` should all be replicated first.
    indices_redistribute_costs = []
    new_indices_spec: list[Optional[DTensorSpec]] = []
    for indices_spec_child in indices_spec.childs:
        assert isinstance(indices_spec_child, OpStrategy)

        replicated_spec = DTensorSpec(
            mesh=mesh,
            placements=tuple([Replicate()] * mesh.ndim),
            tensor_meta=indices_spec_child.strategies[0].output_spec.tensor_meta,
        )
        new_indices_spec.append(replicated_spec)
        child_costs = generate_redistribute_costs(indices_spec_child, replicated_spec)
        indices_redistribute_costs.append(child_costs)

    # 2. For placement rule of `values` and `in`, assume `values` shape =
    # [a,b,c,d,e,f], `in` shape = [d,e,f]. Then `values`'s a,b,c (selected dim)
    # must be replicated and d,e,f (nonselected dim) in both `values` and `in`
    # should follow the same sharding (replicate or shard, but not partial).
    size_offset = (
        in_spec.strategies[0].output_spec.ndim
        - values_spec.strategies[0].output_spec.ndim
    )
    # We can either let `values` follow `in`'s placements or reverse.
    for exemplar_spec in [in_spec, values_spec]:
        # use exemplar_spec as the target spec
        for strategy in exemplar_spec.strategies:
            in_spec_new_placements: list[Placement] = []
            values_spec_new_placements: list[Placement] = []
            placements = strategy.output_spec.placements
            for placement in placements:
                if placement.is_shard():
                    assert isinstance(placement, Shard)
                    if exemplar_spec is in_spec:
                        # let `values_spce` follow `in_spec`
                        if placement.dim < size_offset:
                            # sharded on selected dim, need to change to replicate
                            in_spec_new_placements.append(Replicate())
                            values_spec_new_placements.append(Replicate())
                        else:
                            in_spec_new_placements.append(placement)
                            values_spec_new_placements.append(
                                Shard(placement.dim - size_offset)
                            )
                    else:
                        # let `in_spec` follow `values_spec`
                        in_spec_new_placements.append(
                            Shard(placement.dim + size_offset)
                        )
                        values_spec_new_placements.append(placement)
                else:
                    in_spec_new_placements.append(Replicate())
                    values_spec_new_placements.append(Replicate())
            new_in_spec = DTensorSpec(
                mesh=mesh,
                placements=tuple(in_spec_new_placements),
                tensor_meta=in_spec.strategies[0].output_spec.tensor_meta,
            )
            new_values_spec = DTensorSpec(
                mesh=mesh,
                placements=tuple(values_spec_new_placements),
                tensor_meta=values_spec.strategies[0].output_spec.tensor_meta,
            )
            output_spec = DTensorSpec(
                mesh=mesh,
                placements=tuple(in_spec_new_placements),
                tensor_meta=in_spec.strategies[0].output_spec.tensor_meta,
            )
            cost_in_spec = generate_redistribute_costs(in_spec, new_in_spec)
            cost_values_spec = generate_redistribute_costs(values_spec, new_values_spec)
            op_strategy.strategies.append(
                OpSpec(
                    input_specs=(
                        new_in_spec,
                        *new_indices_spec,  # type: ignore[arg-type]
                        new_values_spec,
                    ),
                    output_specs=output_spec,
                    redistribute_cost=[
                        cost_in_spec,
                        *indices_redistribute_costs,
                        cost_values_spec,
                    ],
                )
            )
    return op_strategy


@register_prop_rule(aten.index.Tensor, schema_info=RuntimeSchemaInfo(needs_pytree=True))
def prop_index(op_schema: OpSchema) -> OutputSharding:
    """
    Expect replicated on the first input; _mostly_ pointwise on the second input.

    TODO: exception: when the dtype of second input is "bool", then a torch.nonzero needs to be triggered first.
    """
    # Current sharding constraints:
    # For values:
    #   1. We currently require that the dimension of values_spec be replicated or partial
    #      if they are being indexed on.
    #   2. Other dimensions of values_spec can remain sharded if they are so.
    # For indices:
    #   Indices can be either sharded or replicated. All index tensors need to be sharded
    #   in a compatible way, following the pointwise rule (including resolving Partial
    #   into either sharded or replicated)

    values_spec, multi_indices_spec = op_schema.args_schema
    assert isinstance(values_spec, DTensorSpec)
    assert isinstance(multi_indices_spec, list)
    multi_indices_spec = cast(list[Optional[DTensorSpec]], multi_indices_spec)
    valid_indices_spec: list[tuple[int, DTensorSpec]] = [
        (i, a) for i, a in enumerate(multi_indices_spec) if a is not None
    ]

    # 1. All indices have to be sharded equally. Moreover, indices can be broadcast.
    #    Here, we piggyback on the pointwise sharding rule for indices.
    indices_out = pointwise_rule(
        OpSchema(
            op=op_schema.op,
            args_schema=tuple(v[1] for v in valid_indices_spec),
            kwargs_schema={},
        )
    )
    need_reshard_on_indices = indices_out.output_spec is None

    if not need_reshard_on_indices:
        # this means that our inputs are already sharded properly and we will use that as our indices_spec
        assert isinstance(indices_out.output_spec, DTensorSpec)
        indices_spec: DTensorSpec = indices_out.output_spec
    else:
        assert indices_out.redistribute_schema is not None
        valid_indices_suggestion = indices_out.redistribute_schema
        for i, v in enumerate(valid_indices_suggestion.args_spec):
            multi_indices_spec[valid_indices_spec[i][0]] = v
        # we'll need to call pointwise_rule again to see what's our ideal indices_spec and then
        # use that to compute our ideal values_spec
        indices_output_spec = pointwise_rule(valid_indices_suggestion).output_spec
        assert isinstance(indices_output_spec, DTensorSpec)
        indices_spec = indices_output_spec

    lookup_dims = {v[0] for v in valid_indices_spec}

    need_reshard_on_values = tuple(
        (isinstance(vp, Shard) and (vp.dim in lookup_dims or isinstance(ip, Shard)))
        for vp, ip in zip(values_spec.placements, indices_spec.placements)
    )

    if not need_reshard_on_indices and not any(need_reshard_on_values):
        value_placements = values_spec.placements

        all_dims_consecutive = all(
            b[0] - a[0] == 1
            for b, a in zip(valid_indices_spec[1:], valid_indices_spec[:-1])
        )
        if all_dims_consecutive:
            # if all index vectors are consecutives, insert at the dimension of the first index
            insert_dim: int = valid_indices_spec[0][0]
        else:
            # else, insert on the first dimension
            insert_dim = 0

        def place(vp: Placement, ip: Placement) -> Placement:
            if isinstance(vp, Shard):
                return Shard(
                    vp.dim
                    if vp.dim < insert_dim
                    # accounts for the offset in output dimensions
                    else vp.dim
                    + indices_spec.ndim
                    - sum(1 if vp.dim > v[0] else 0 for v in valid_indices_spec)
                )
            if isinstance(ip, Shard):
                return Shard(ip.dim + insert_dim)
            # Partial or Replicated
            return vp

        value_placements = tuple(
            place(vp, ip)
            for vp, ip in zip(values_spec.placements, indices_spec.placements)
        )
        result = OutputSharding(
            output_spec=DTensorSpec(
                mesh=values_spec.mesh,
                placements=value_placements,
            )
        )
        return result
    else:
        result = OutputSharding(
            output_spec=None,
            redistribute_schema=OpSchema(
                op=op_schema.op,
                args_schema=(
                    DTensorSpec(
                        mesh=values_spec.mesh,
                        placements=tuple(
                            [
                                Replicate() if need_reshard_on_values[i] else v
                                for i, v in enumerate(values_spec.placements)
                            ]
                        ),
                        tensor_meta=values_spec.tensor_meta,
                    ),
                    multi_indices_spec,
                ),
                kwargs_schema=op_schema.kwargs_schema,
            ),
        )
        return result


@register_op_strategy(
    [
        aten.split.Tensor,
        aten.split_with_sizes.default,
        aten.split_with_sizes_copy.default,
    ],
    RuntimeSchemaInfo(1),
)
def split_strategy(op_schema: OpSchema) -> TupleStrategy:
    input_strategy = op_schema.args_schema[0]
    split_size_or_sections = op_schema.args_schema[1]
    assert isinstance(input_strategy, OpStrategy)
    input_ndim = input_strategy.ndim
    split_dim = (
        cast(int, op_schema.args_schema[2]) if len(op_schema.args_schema) > 2 else 0
    )
    dim = normalize_dim(split_dim, input_ndim)

    # tensor to split cannot have Partial for now
    for arg_strategy in input_strategy.strategies:
        arg_spec = arg_strategy.output_spec
        if is_tensor_partial(arg_spec):
            raise NotImplementedError(
                f"splitting distributed tensor with "
                f"Partial placement is not implemented!\n"
                f"DTensorSpec={arg_strategy}"
            )

    def size_split(N, i) -> list:
        # Last chunk will be smaller if the tensor size N
        # along the given dimension dim is not divisible by i.
        assert i > 0
        return [i] * (N // i) + ([N % i] if N % i != 0 else [])

    output_size_list = (
        size_split(input_strategy.shape[dim], split_size_or_sections)
        if isinstance(split_size_or_sections, int)
        else split_size_or_sections
    )
    assert isinstance(output_size_list, Sized)

    split_strategies = []

    for _ in range(len(output_size_list)):
        op_strategy = OpStrategy([])

        for strategy in input_strategy.strategies:
            spec = strategy.output_spec
            placements = spec.placements
            if is_tensor_dim_sharded(spec, dim=dim):
                # if the input is sharded on the split dim, we need to unshard it
                placements = unshard_tensor_dim(spec.placements, dim=dim)

            spec = DTensorSpec(spec.mesh, placements)

            op_strategy.strategies.append(
                OpSpec(output_specs=spec, input_specs=([spec]))
            )
        split_strategies.append(op_strategy)

    return TupleStrategy(split_strategies)
