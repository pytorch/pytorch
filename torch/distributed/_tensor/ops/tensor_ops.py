# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import cast, Dict, List, Optional, Sequence, Tuple

import torch

from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.api import (
    _Partial,
    DTensorSpec,
    Placement,
    Replicate,
    Shard,
)
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed._tensor.op_schema import (
    OpSchema,
    OpStrategy,
    OutputSharding,
    PlacementStrategy,
    StrategyType,
)
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import (
    normalize_dim,
    prod,
    register_op_strategy,
    register_prop_rule,
)
from torch.fx import Node


aten = torch.ops.aten


@register_op_strategy(
    [
        aten._to_copy.default,
        aten.clone.default,
        aten.contiguous.default,
        aten.copy_.default,
        aten.detach.default,
        aten.new_empty_strided.default,  # TODO: re-think new_empty_strided
    ]
)
def default_strategy(
    node: Node, mesh: DeviceMesh, node_to_strategy: Dict[Node, StrategyType]
) -> StrategyType:
    # Default strategy by default just propagate the first input strategy
    select_strategy = node_to_strategy[node.all_input_nodes[0]]
    assert isinstance(select_strategy, OpStrategy)
    return OpStrategy(
        [
            PlacementStrategy(arg_strategy.output_spec)
            for arg_strategy in select_strategy.strategies
        ]
    )


@register_op_strategy(
    [
        aten.empty_like.default,
        aten.fill_.Scalar,
        aten.full_like.default,
        aten.ones_like.default,
        aten.zero_.default,
        aten.zeros_like.default,
    ]
)
def create_like_strategy(
    node: Node, mesh: DeviceMesh, node_to_strategy: Dict[Node, StrategyType]
) -> StrategyType:
    # create_like_strategy deals with ops that creating tensors with same
    # shape as input, but with specific content that does not depend on
    # the input, we can propagate sharding, but we have to make sure we
    # move from partial to replicated.
    select_strategy = node_to_strategy[node.all_input_nodes[0]]
    create_like_strategy = OpStrategy([])
    assert isinstance(select_strategy, OpStrategy)
    for arg_strategy in select_strategy.strategies:
        arg_spec = arg_strategy.output_spec
        if arg_spec.sums:
            # if the arg_spec have partial, accept partial
            # in the input_specs but output replicate for
            # those corresponding mesh dims
            output_spec = DTensorSpec(
                mesh=arg_spec.mesh,
                placements=tuple(
                    Replicate() if isinstance(p, _Partial) else p
                    for p in arg_spec.placements
                ),
            )
            create_like_strategy.strategies.append(
                PlacementStrategy(output_spec=output_spec, input_specs=(arg_spec,))
            )

        else:
            create_like_strategy.strategies.append(PlacementStrategy(arg_spec))

    return create_like_strategy


@register_prop_rule(aten._local_scalar_dense.default)
def no_shard_prop_rule(op_schema: OpSchema) -> OutputSharding:
    # some tensor ops should not support shard, i.e. local_scalar_dense
    # shouldn't work for shard as it requires numel == 1
    # by default prop the first arg spec
    tensor_spec = op_schema.args_spec[0]
    for placement in tensor_spec.placements:
        if placement.is_shard():
            return OutputSharding(
                None,
                failed_reason=f"Op does not support input placements "
                f"with `Shard`, but found placements: "
                f"{tensor_spec.placements}",
            )
    # otherwise default prop as None as it would not return
    # a DTensor
    return OutputSharding(None)


def new_factory_rule(op_schema: OpSchema) -> OutputSharding:
    # this op would benefit from backward sharding propagation!
    # Since we cannot do that yet, just return replicated
    input = op_schema.args_schema[0]
    assert isinstance(input, DTensorSpec)

    return OutputSharding(
        output_spec=DTensorSpec(
            mesh=input.mesh,
            placements=[Replicate()] * input.mesh.ndim,
            tensor_meta=input.tensor_meta,
        )
    )


@register_prop_rule([aten.equal.default, aten.is_same_size.default])
def non_tensor_prop_rule(op_schema: OpSchema) -> OutputSharding:
    # simply return None as it does not return DTensor
    return OutputSharding(output_spec=None)


new_factory_ops = [
    aten.new_full.default,
    aten.new_ones.default,
    aten.new_zeros.default,
]

for op in new_factory_ops:
    register_prop_rule(op)(new_factory_rule)


@register_prop_rule(aten.bucketize.Tensor)
def prop_bucketize(op_schema: OpSchema) -> OutputSharding:
    """
    Point-wise on the first input (just propagate input sharding).
    Expect replicated for second input.
    """
    input_schema, boundaries = op_schema.args_schema
    assert isinstance(input_schema, DTensorSpec)
    assert isinstance(boundaries, DTensorSpec)

    if all(isinstance(p, Replicate) for p in boundaries.placements):
        return OutputSharding(output_spec=input_schema)
    else:
        return OutputSharding(
            output_spec=None,
            schema_suggestions=[
                OpSchema(
                    func_schema=op_schema.func_schema,
                    args_schema=(
                        input_schema,
                        DTensorSpec(
                            mesh=boundaries.mesh,
                            placements=[Replicate()] * len(boundaries.placements),
                            tensor_meta=boundaries.tensor_meta,
                        ),
                    ),
                    kwargs_schema=op_schema.kwargs_schema,
                )
            ],
        )


def unshard_tensor_dim(
    placements: Sequence[Placement], dim: int
) -> Sequence[Placement]:
    """Disallow the given tensor dimension to be sharded"""
    return tuple(
        p if (not isinstance(p, Shard) or p.dim != dim) else Replicate()
        for p in placements
    )


def replicate_tensor_dim(
    placements: Sequence[Placement], dim: int
) -> Sequence[Placement]:
    """Force the given tensor dimension to be replicated"""
    # Not using p.is_shard() to avoid mypy complain about Placement not having
    # attribute dim.
    return tuple(
        Replicate() if p.is_partial() or isinstance(p, Shard) and p.dim == dim else p
        for p in placements
    )


def is_tensor_dim_sharded(spec: DTensorSpec, dim: int) -> bool:
    """Return True if tensor dim is sharded"""
    return (dim < spec.ndim) and spec.dim_map[dim] >= 0


def is_tensor_partial(spec: DTensorSpec) -> bool:
    return any(p.is_partial() for p in spec.placements)


def _prop_all_but_dim(op_schema: OpSchema, dim: int) -> OutputSharding:
    """
    Considering an op that takes its input as first argument, forwards all shardings
    except for the given dimension.
    """
    input_spec = op_schema.args_schema[0]
    assert isinstance(input_spec, DTensorSpec)

    output_placements = unshard_tensor_dim(input_spec.placements, dim=dim)
    output_spec = DTensorSpec(
        mesh=input_spec.mesh,
        placements=output_placements,
    )

    if input_spec.placements == output_placements:
        out = OutputSharding(output_spec=output_spec)
    else:
        suggested_input_spec = DTensorSpec(
            mesh=input_spec.mesh,
            placements=output_placements,
            tensor_meta=input_spec.tensor_meta,
        )
        out = OutputSharding(
            output_spec=None,
            schema_suggestions=[
                OpSchema(
                    func_schema=op_schema.func_schema,
                    args_schema=(suggested_input_spec,) + op_schema.args_schema[1:],
                    kwargs_schema=op_schema.kwargs_schema,
                ),
            ],
        )
    return out


@register_prop_rule(aten.slice.Tensor)
def prop_slice(op_schema: OpSchema) -> OutputSharding:
    """NOTE: can be further optimized (right now it replicates before slicing on a sharded dimension)"""
    defaults = (None, 0, None, None, 1)
    input_spec, dim, start, end, step = (
        op_schema.args_schema + defaults[len(op_schema.args_schema) :]
    )
    assert isinstance(input_spec, DTensorSpec)
    assert isinstance(dim, int)
    assert start is None or isinstance(start, int)
    assert end is None or isinstance(end, int)
    assert isinstance(step, int)

    # normalize arguments
    if dim < 0:
        dim += input_spec.ndim
    if start is None:
        start = 0
    if step is None:
        step = 1
    if end is None or end > input_spec.shape[dim]:
        end = input_spec.shape[dim]
    if start < 0:
        start += input_spec.shape[dim]
    if end < 0:
        end += input_spec.shape[dim]

    if start == 0 and end == input_spec.shape[dim] and step == 1:
        return OutputSharding(output_spec=input_spec)

    return _prop_all_but_dim(op_schema, dim=dim)


@register_prop_rule(aten.slice_scatter.default)
def prop_slice_scatter(op_schema: OpSchema) -> OutputSharding:
    # 1. number of dimensions in input and src need to match.
    # 2. number of elements on all non-dim need to match between input and src.
    # 3. numer of elements in src in dim need to match the slice size.
    # Given the above:
    # - We suggest for src to follow the sharding of input, except on the scatter dimension,
    #   where our best bet for now is to make them replicated as a fall-back.
    #   TODO: Ideally we'd like to make sure the output is re-sharded afterwards to keep input sharding.

    defaults = (None, None, 0, None, None, 1)
    input, src, dim, start, end, step = (
        op_schema.args_schema + defaults[len(op_schema.args_schema) :]
    )
    assert isinstance(input, DTensorSpec)
    assert isinstance(src, DTensorSpec)
    assert isinstance(dim, int)

    if dim < 0:
        dim += input.ndim

    # first, we keep the input sharding, except for the input dimension
    # also, we cannot allow partial sum anymore.
    input_suggestion = tuple(
        Replicate()
        if isinstance(p, _Partial) or (isinstance(p, Shard) and p.dim == dim)
        else p
        for p in input.placements
    )

    if input_suggestion == tuple(input.placements) and src.placements == tuple(
        input.placements
    ):
        # if our sharding is correct, the output sharding will be the same as the input.
        return OutputSharding(
            output_spec=DTensorSpec(
                mesh=input.mesh,
                placements=input.placements,
            )
        )
    else:
        # otherwise, return the suggestion.
        return OutputSharding(
            output_spec=None,
            schema_suggestions=[
                OpSchema(
                    func_schema=op_schema.func_schema,
                    args_schema=(
                        DTensorSpec(
                            mesh=input.mesh,
                            placements=input_suggestion,
                            tensor_meta=input.tensor_meta,
                        ),
                        DTensorSpec(
                            mesh=src.mesh,
                            placements=input_suggestion,
                            tensor_meta=src.tensor_meta,
                        ),
                    )
                    + op_schema.args_schema[2:],
                    kwargs_schema=op_schema.kwargs_schema,
                )
            ],
        )


@register_prop_rule(aten.index_select.default)
def prop_index_select(op_schema: OpSchema) -> OutputSharding:
    values_spec, dim, indices_spec = op_schema.args_schema

    assert isinstance(values_spec, DTensorSpec)
    assert isinstance(dim, int)
    assert isinstance(indices_spec, DTensorSpec)

    all_indices_spec: List[Optional[DTensorSpec]] = [
        indices_spec if dim == i else None for i in range(values_spec.ndim)
    ]

    result = prop_index(
        OpSchema(
            func_schema=op_schema.func_schema,
            args_schema=(values_spec, all_indices_spec),
            kwargs_schema=op_schema.kwargs_schema,
        )
    )
    if result.schema_suggestions:
        result.schema_suggestions = [
            OpSchema(
                func_schema=op_schema.func_schema,
                args_schema=(s.args_schema[0], dim, s.args_schema[1][dim]),
                kwargs_schema=op_schema.kwargs_schema,
            )
            for s in result.schema_suggestions
        ]
    return result


@register_prop_rule(aten.index.Tensor)
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
    #   in a compatible way, following the pointwise rule (including resolving _Partial
    #   into either sharded or replicated)

    values_spec, multi_indices_spec = op_schema.args_schema
    assert isinstance(values_spec, DTensorSpec)
    assert isinstance(multi_indices_spec, list)
    multi_indices_spec = cast(List[Optional[DTensorSpec]], multi_indices_spec)
    valid_indices_spec: List[Tuple[int, DTensorSpec]] = [
        (i, a) for i, a in enumerate(multi_indices_spec) if a is not None
    ]

    # 1. All indices have to be sharded equally. Moreover, indices can be broadcast.
    #    Here, we piggyback on the pointwise sharding rule for indices.
    indices_out = pointwise_rule(
        OpSchema(
            func_schema=op_schema.func_schema,
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
        assert indices_out.schema_suggestions is not None
        valid_indices_suggestion = indices_out.schema_suggestions[0]
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
            # _Partial or Replicated
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
            schema_suggestions=[
                OpSchema(
                    func_schema=op_schema.func_schema,
                    args_schema=(
                        DTensorSpec(
                            mesh=values_spec.mesh,
                            placements=[
                                Replicate() if need_reshard_on_values[i] else v
                                for i, v in enumerate(values_spec.placements)
                            ],
                            tensor_meta=values_spec.tensor_meta,
                        ),
                        multi_indices_spec,
                    ),
                    kwargs_schema=op_schema.kwargs_schema,
                )
            ],
        )
        return result


@register_prop_rule(aten.cat.default)
def cat_rule(op_schema: OpSchema) -> OutputSharding:
    # torch.cat requires all tensors must either have the same shape (except
    # in the concatenating dimension) or be "empty". "Empty" here strictly means
    # tensor.shape is torch.Size([0]). When tensor.ndim > 1, it will be treated
    # as a non-empty tensor and the shape must match on non-cat dimensions.
    def is_empty(spec: DTensorSpec) -> bool:
        return list(spec.shape) == [0]

    # the first arg is a list of input tensor specs
    tensor_list_specs = cast(List[DTensorSpec], op_schema.args_schema[0])
    assert len(tensor_list_specs) > 0, "torch.cat expects a non-empty list of tensors"
    non_empty_specs = [spec for spec in tensor_list_specs if not is_empty(spec)]

    if len(non_empty_specs) == 0:
        # all tensors are empty, we can return any output sharding
        return OutputSharding(
            output_spec=DTensorSpec(
                mesh=tensor_list_specs[0].mesh,
                placements=tensor_list_specs[0].placements,
            )
        )

    assert all(
        spec.ndim == non_empty_specs[0].ndim for spec in non_empty_specs
    ), f"Expect all tensors to have same shape or empty, but got {tensor_list_specs}"
    assert all(
        spec.mesh == tensor_list_specs[0].mesh for spec in tensor_list_specs
    ), f"Expect all tensors to have same mesh, but got {tensor_list_specs}"

    # ndim will also be the result's ndim
    ndim = 1
    for spec in tensor_list_specs:
        ndim = max(ndim, spec.ndim)

    dim = 0  # default dim = 0
    if len(op_schema.args_schema) > 1:
        dim = cast(int, op_schema.args_schema[1])
    dim = normalize_dim(dim, ndim)

    # Make sure all tensors are replciated on cat dimension
    need_reshard = False
    tensor_list_specs_after: List[DTensorSpec] = []
    for spec in tensor_list_specs:
        if not is_empty(spec) and (
            is_tensor_dim_sharded(spec, dim=dim) or is_tensor_partial(spec)
        ):
            need_reshard = True
            tensor_list_specs_after.append(
                DTensorSpec(
                    mesh=spec.mesh,
                    placements=replicate_tensor_dim(spec.placements, dim=dim),
                    tensor_meta=spec.tensor_meta,
                )
            )
        else:
            tensor_list_specs_after.append(spec)

    tensor_list_specs = tensor_list_specs_after

    # align non-cat dimensions placements based on reshard cost
    non_empty_specs = [spec for spec in tensor_list_specs if not is_empty(spec)]
    mesh = non_empty_specs[0].mesh
    ndim = non_empty_specs[0].ndim
    new_placements: List[Placement] = []
    for mesh_dim in range(mesh.ndim):
        # compute the minimum cost of resharding on this mesh_dim
        if any(
            spec.placements[mesh_dim] != non_empty_specs[0].placements[mesh_dim]
            for spec in non_empty_specs
        ):
            # only reshard if there is a mismatch
            need_reshard = True
            reshard_cost = []
            for shard_dim in range(ndim):
                # compute the cost of resharding on this shard_dim
                cost: float = 0.0
                for spec in non_empty_specs:
                    global_shape = spec.shape
                    if global_shape[shard_dim] < mesh.size(mesh_dim):
                        # found one tensor where the shard_dim is smaller than
                        # mesh_dim. In this case, we cannot shard on this shard_dim,
                        # and hence set cost to infinity.
                        cost = +float("inf")
                    elif (
                        is_tensor_dim_sharded(spec, dim=shard_dim)
                        or prod(global_shape) == 0
                    ):
                        continue
                    else:
                        local_shape = compute_local_shape(
                            global_shape, spec.mesh, spec.placements
                        )
                        cost += prod(local_shape) * spec.mesh.size(mesh_dim)
                reshard_cost.append(cost)
            best_dim = reshard_cost.index(min(reshard_cost))
            new_placements.append(Shard(best_dim))
        else:
            # no mismatch, keep the original placement
            new_placements.append(non_empty_specs[0].placements[mesh_dim])

    if need_reshard:
        tensor_list_specs_after = []
        for spec in tensor_list_specs:
            if is_empty(spec):
                tensor_list_specs_after.append(spec)
            else:
                tensor_list_specs_after.append(
                    DTensorSpec(
                        mesh=spec.mesh,
                        placements=new_placements,
                        tensor_meta=spec.tensor_meta,
                    )
                )

        return OutputSharding(
            output_spec=None,
            schema_suggestions=[
                OpSchema(
                    func_schema=op_schema.func_schema,
                    args_schema=(
                        tuple(tensor_list_specs_after),
                        *op_schema.args_schema[1:],
                    ),
                    kwargs_schema=op_schema.kwargs_schema,
                ),
            ],
        )
    else:
        # at this point, the cat dim is not sharded,
        return OutputSharding(
            output_spec=DTensorSpec(
                mesh=non_empty_specs[0].mesh,
                placements=non_empty_specs[0].placements,
            ),
        )


def _update_schema_suggestion_for_cat(
    output_sharding: OutputSharding,
    op_schema: OpSchema,
) -> OutputSharding:
    assert output_sharding.schema_suggestions is not None
    suggestion_specs = output_sharding.schema_suggestions[0].args_spec

    args_schema = (suggestion_specs,) + op_schema.args_schema[1:]

    output_sharding.schema_suggestions = [
        OpSchema(
            func_schema=op_schema.func_schema,
            args_schema=args_schema,
            kwargs_schema=op_schema.kwargs_schema,
        )
    ]
    return output_sharding


@register_prop_rule([aten.split.Tensor, aten.split_with_sizes.default])
def split_rule(op_schema: OpSchema) -> OutputSharding:
    output_spec_list: List[DTensorSpec] = []
    input_spec = cast(DTensorSpec, op_schema.args_schema[0])
    ndim = input_spec.ndim
    split_size_or_sections = op_schema.args_schema[1]
    dim = cast(int, op_schema.args_schema[2]) if len(op_schema.args_schema) > 2 else 0
    dim = normalize_dim(dim, ndim)

    # TODO: tensor to split cannot have _Partial
    # in its placements for now. Will need to
    # support in future.
    if input_spec.sums:
        raise NotImplementedError(
            f"splitting distributed tensor with "
            f"_Partial placement is not implemented!\n"
            f"DTensorSpec={input_spec}"
        )

    # TODO: just like slice op, split replicates before
    # splitting on a sharded dimension
    need_reshard = False
    if is_tensor_dim_sharded(input_spec, dim=dim):
        need_reshard = True
        input_spec = DTensorSpec(
            mesh=input_spec.mesh,
            placements=unshard_tensor_dim(input_spec.placements, dim=dim),
            tensor_meta=input_spec.tensor_meta,
        )

    if need_reshard:
        return OutputSharding(
            None,
            schema_suggestions=[
                OpSchema(
                    func_schema=op_schema.func_schema,
                    args_schema=(input_spec,) + op_schema.args_schema[1:],
                    kwargs_schema=op_schema.kwargs_schema,
                ),
            ],
        )

    def size_split(N, i):
        # Last chunk will be smaller if the tensor size N
        # along the given dimension dim is not divisible by i.
        assert i > 0
        return [i] * (N // i) + ([N % i] if N % i != 0 else [])

    output_size_list = (
        size_split(input_spec.shape[dim], split_size_or_sections)
        if isinstance(split_size_or_sections, int)
        else split_size_or_sections
    )
    output_spec_list = [
        DTensorSpec(
            mesh=input_spec.mesh,
            placements=input_spec.placements,
        )
        for _ in range(len(output_size_list))
    ]
    return OutputSharding(output_spec_list)
