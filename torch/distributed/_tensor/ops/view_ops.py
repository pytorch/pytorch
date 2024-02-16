# Copyright (c) Meta Platforms, Inc. and affiliates
from dataclasses import dataclass
from typing import Callable, cast, Dict, Iterable, Optional, Sequence, Set, Tuple, Union

import torch

from torch import Tensor
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.api import Shard
from torch.distributed._tensor.op_schema import (
    OpSchema,
    OutputSharding,
    RuntimeSchemaInfo,
)
from torch.distributed._tensor.ops.utils import (
    normalize_dim,
    normalize_dims,
    prod,
    register_prop_rule,
)

from torch.distributed._tensor.placement_types import DTensorSpec, Placement, Replicate
from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing

aten = torch.ops.aten

Shape = Tuple[int, ...]


@dataclass
class DimSpec:
    """Specifies how an output dimension maps to an input dimension."""

    def inputs(self) -> Iterable["DimSpec"]:
        return ()


# Rules that map each dimension of the output to dimensions of the input tensor
DimMap = Tuple[DimSpec, ...]


@dataclass
class Singleton(DimSpec):
    """Output dimension is a singleton."""

    pass


@dataclass
class InputDim(DimSpec):
    """Output dimension maps directly to an input dimension."""

    input_dim: int


@dataclass
class Broadcast(DimSpec):
    """Output is the broadcast of a singleton input dimension."""

    dim: DimSpec
    dim_size: int

    @classmethod
    def new(cls, dim: DimSpec, dim_size: int) -> DimSpec:
        return Broadcast(dim, dim_size)

    def inputs(self) -> Iterable[DimSpec]:
        return (self.dim,)


@dataclass
class NewDim(DimSpec):
    """This is a new dimension created by the op."""

    size: int

    @classmethod
    def new(cls, size: int) -> DimSpec:
        return Singleton() if size == 1 else NewDim(size)


@dataclass
class Repeat(DimSpec):
    """Output dimension is the input dimension repeated n-times."""

    input_dim: DimSpec
    times: int

    @classmethod
    def new(cls, dim: DimSpec, times: int) -> DimSpec:
        if times == 1:
            return dim
        elif isinstance(dim, Singleton):
            # repeating a singleton is the same as broadcasting it
            return Broadcast(dim, times)
        else:
            return Repeat(dim, times)

    def inputs(self) -> Iterable[DimSpec]:
        return (self.input_dim,)


@dataclass
class Flatten(DimSpec):
    """Flatten a set of input dimensions, ensuring right-most adjacent elements remain adjacent in the output."""

    input_dims: Sequence[DimSpec]

    @classmethod
    def new(cls, dims: Sequence[DimSpec]) -> DimSpec:
        if len(dims) == 0:
            # flattening a scalar leads to a singleton
            return Singleton()
        elif len(dims) == 1:
            # flattening a single dimension is no-op
            return dims[0]
        else:
            return Flatten(dims)

    def inputs(self) -> Iterable[DimSpec]:
        return self.input_dims


@dataclass
class Split(DimSpec):
    """
    This dimension is a member of a decomposition of the input dim.

    Note that input_dim itself could be a Flattened set of input dims.
    """

    input_dim: DimSpec
    group_shape: Shape
    split_id: int

    @classmethod
    def new(cls, dim: DimSpec, group_shape: Tuple[int, ...], idx: int) -> DimSpec:
        assert len(group_shape) > 0
        if len(group_shape) == 1:
            # not really a group, just return the input dim back
            assert idx == 0
            return dim
        elif group_shape[idx] == 1:
            return Singleton()
        else:
            # remove singletons from group
            # group_mapping = [(new_index, (shape, old_index)) ...]
            group_mapping = list(
                enumerate((s, i) for i, s in enumerate(group_shape) if s != 1)
            )
            new_group_shape = tuple(m[1][0] for m in group_mapping)
            new_idx = next(filter(lambda x: x[1][1] == idx, group_mapping))[0]
            return Split(dim, new_group_shape, new_idx)

    def inputs(self) -> Iterable[DimSpec]:
        return (self.input_dim,)


def dim_pad_left(ndim: int, min_dims: int) -> DimMap:
    return (Singleton(),) * max(0, min_dims - ndim) + tuple(
        InputDim(i) for i in range(ndim)
    )


def dim_atleast_3d(ndim: int) -> DimMap:
    if ndim == 0:
        return (Singleton(), Singleton(), Singleton())
    elif ndim == 1:
        return (Singleton(), InputDim(0), Singleton())
    elif ndim == 2:
        return (InputDim(0), InputDim(1), Singleton())
    else:
        return tuple(InputDim(i) for i in range(ndim))


def expand(input_shape: Shape, shape: Shape) -> DimMap:
    """Implement broadcast on multiple dimensions."""
    assert len(shape) >= len(input_shape)

    # 1. create padded input dimensions
    padded_input = dim_pad_left(len(input_shape), len(shape))
    # 2. check that input shapes are compatible
    mapping = []
    for p, desired_s in zip(padded_input, shape):
        if isinstance(p, Singleton):
            actual_s = 1
            assert desired_s >= 0
        else:
            assert isinstance(p, InputDim), f"DimSpec not supported in expand: {p}"
            actual_s = input_shape[p.input_dim]
            assert actual_s == 1 or desired_s == -1 or desired_s == actual_s
        mapping.append(
            p
            if desired_s in (1, -1) or desired_s == actual_s
            else Broadcast.new(p, desired_s)
        )
    return tuple(mapping)


def normalize_sizes(sizes: Union[Shape, Tuple[Shape]]) -> Shape:
    if isinstance(sizes[0], int):
        return cast(Shape, sizes)
    elif len(sizes) == 1:
        return cast(Shape, sizes[0])  # type: ignore[redundant-cast]
    else:
        raise RuntimeError("Size must be int... or tuple")


def dim_flatten(ndim: int) -> DimMap:
    if ndim == 0:
        return (Singleton(),)
    elif ndim == 1:
        return (InputDim(0),)
    else:
        return (Flatten.new(tuple(InputDim(i) for i in range(ndim))),)


def dim_movedim(
    ndim: int,
    input: Union[int, Sequence[int]],
    destination: Union[int, Sequence[int]],
) -> DimMap:
    input = normalize_dims(input, ndim)
    destination = normalize_dims(destination, ndim)

    assert len(input) == len(destination)
    input_set = set(input)
    assert len(input_set) == len(input), "Found repeated input dims"
    assert len(set(destination)) == len(destination), "Found repeated output dims"
    assert max(input) < ndim
    assert max(destination) < ndim

    dest = [-1] * ndim
    for i, d in zip(input, destination):
        dest[d] = i

    unused_inputs_iter = iter(i for i in range(ndim) if i not in input_set)
    for i in range(ndim):
        if dest[i] == -1:
            dest[i] = next(unused_inputs_iter)

    return tuple(InputDim(i) for i in dest)


def dim_repeat(ndim: int, sizes: Shape) -> DimMap:
    sizes = normalize_sizes(sizes)
    assert (
        len(sizes) >= ndim
    ), f"Number of dimensions of repeat dims {sizes} can not be smaller than number of dimensions of tensor {ndim}."
    pad = len(sizes) - ndim
    return tuple(Repeat.new(Singleton(), s) for s in sizes[:pad]) + tuple(
        Repeat.new(InputDim(i), s) for i, s in enumerate(sizes[pad:])
    )


def infer_size(total_size: int, sizes: Shape) -> Shape:
    """
    One dimension input to view may be "-1".

    Infer the size of this dimension given the total_size.
    """
    infers = [i for i, s in enumerate(sizes) if s == -1]
    size = prod(sizes)
    assert len(infers) <= 1, "can only infer one size"
    if infers:
        size = -size
        missing_size = total_size // size
        assert (
            total_size % size == 0
        ), f"size inferred for -1 is not integral {sizes} should have {total_size} elements."
        return tuple(s if s != -1 else missing_size for s in sizes)
    assert size == total_size, f"sizes do not match {total_size} vs {size}"
    return sizes


def view_groups(from_size: Shape, to_size: Shape) -> DimMap:
    """
    Decompose a reshape operation into forwarding, flattening, or splitting dimensions for each output dimension.

    A view or reshape operation can be decomposed into a set of 3 types of smaller operations:
    1) Forward a dimension from input to output
    2) Flatten a set of dimensions into a single dimension
    3) Split one dimension into multiple dimensions

    view_groups identifies these operations and returns, for each output dimension, what
    is operation was performed in the input dimension. For example:

        view_groups([2, 3, 4], [2, 12]) -> (
            InputDim(0),
            Flatten((InputDim(1), InputDim(2)))
        )

    - ouptut dimension 0 maps to input dimension 0
    - output dimension 1 maps to a flattened input dimensions 1 and 2


        view_groups([2, 3], [3, 2]) -> (
            Split(Flatten((InputDim(0), InputDim(1))), (3, 2), 0),
            Split(Flatten((InputDim(0), InputDim(1))), (3, 2), 1),
        )

    - in the above, input is flattened into a single dimension and then split
      into two separate dimensions with different sizes from the input.
    """
    from_nelem = prod(from_size)
    to_size = infer_size(from_nelem, normalize_sizes(to_size))

    assert from_nelem == prod(to_size), "Total view shape does not add up"

    from_idx = 0
    to_idx = 0
    from_len = len(from_size)
    to_len = len(to_size)

    result_pp = []

    while from_idx < from_len or to_idx < to_len:
        from_group_dim, to_group_shape = [], []

        if from_idx >= from_len:
            f = 1
        else:
            f = from_size[from_idx]
            from_group_dim.append(from_idx)
            from_idx += 1

        if to_idx >= to_len:
            t = 1
        else:
            t = to_size[to_idx]
            to_group_shape.append(t)
            to_idx += 1

        # if any of the groups is singleton, great, we need to backtrack though
        if f == 1 and t != 1:
            # produces ([1], [])
            to_idx -= 1
            to_group_shape = []
        elif f != 1 and t == 1:
            # produces ([], [1])
            from_idx -= 1
            from_group_dim = []
        else:
            # produces ([1], [1]),  ([2], [2]), ([2,3], [6])
            while f != t:
                if f < t:
                    nf = from_size[from_idx]
                    from_group_dim.append(from_idx)
                    from_idx += 1
                    f *= nf
                else:
                    nt = to_size[to_idx]
                    to_group_shape.append(nt)
                    to_idx += 1
                    t *= nt

        if len(to_group_shape) > 0:
            flattened = Flatten.new(
                tuple(InputDim(fi) for fi in from_group_dim if from_size[fi] > 1)
            )
            result_pp += [
                Split.new(flattened, tuple(to_group_shape), i)
                for i in range(len(to_group_shape))
            ]

    return tuple(result_pp)


def dim_tile(ndim: int, dims: Tuple[int, ...]) -> DimMap:
    if len(dims) < ndim:
        dims = (1,) * (ndim - len(dims)) + dims
    return dim_repeat(ndim, dims)


def dim_transpose(ndim: int, dim1: int, dim2: int) -> DimMap:
    dim1 = normalize_dim(dim1, ndim)
    dim2 = normalize_dim(dim2, ndim)
    assert dim1 < ndim
    assert dim2 < ndim
    dimmap = [InputDim(i) for i in range(ndim)]
    swapdim = dimmap[dim1]
    dimmap[dim1] = dimmap[dim2]
    dimmap[dim2] = swapdim
    return tuple(dimmap)


def dim_squeeze(shape: Shape, dim: Optional[int] = None) -> DimMap:
    # FIXME: this is wrong when dim=None and one of the dimensions
    # equals size of the mesh. For example squeeze(DTensor(tensor(4), Shard[0])) could
    # end up as squeeze(tensor(1)) if we have 4 devices; this would lead to
    # removal of a dimension that is not actually a singleton.
    return tuple(
        InputDim(i)
        for i, s in enumerate(shape)
        if s > 1 or (dim is not None and i != normalize_dim(dim, len(shape)))
    )


def dim_unsqueeze(ndim: int, dim: int) -> DimMap:
    dims = tuple(InputDim(i) for i in range(ndim))
    if dim < 0:
        dim += ndim + 1
    return dims[:dim] + (Singleton(),) + dims[dim:]


def dim_reduction(
    ndim: int, dim_or_dims: Optional[Union[int, Sequence[int]]], keepdim: bool
) -> DimMap:
    """
    General fallback for reduction ops where _Partial() does not apply.

    This will cause incoming tensor to be replicated on the reducing dimensions.
    """
    if dim_or_dims is None:
        dim_or_dims = tuple(range(ndim))
    if isinstance(dim_or_dims, int):
        dim_or_dims = (dim_or_dims,)
    dim_or_dims = tuple(d if d >= 0 else d + ndim for d in dim_or_dims)
    return tuple(
        InputDim(i) if i not in dim_or_dims else Singleton()
        for i in range(ndim)
        if i not in dim_or_dims or keepdim
    )


@dataclass
class Op:
    dim_map: Callable[..., DimMap]
    shape_argnum: Optional[int] = None


ops: Dict[Callable[..., torch.Tensor], Op] = {
    torch.atleast_1d: Op(dim_map=lambda x: dim_pad_left(x.ndim, 1)),
    torch.atleast_2d: Op(dim_map=lambda x: dim_pad_left(x.ndim, 2)),
    torch.atleast_3d: Op(dim_map=lambda x: dim_atleast_3d(x.ndim)),
    torch.broadcast_to: Op(
        dim_map=lambda input, shape: expand(input.shape, shape), shape_argnum=1
    ),
    Tensor.expand: Op(
        dim_map=lambda self, *sizes: expand(self.shape, normalize_sizes(sizes)),
        shape_argnum=1,
    ),
    torch.flatten: Op(dim_map=lambda tensor: dim_flatten(tensor.ndim)),
    torch.movedim: Op(
        dim_map=lambda input, source, destination: dim_movedim(
            input.ndim, source, destination
        )
    ),
    torch.permute: Op(
        dim_map=lambda input, dims: tuple(
            InputDim(i) for i in normalize_dims(dims, input.ndim)
        )
    ),
    torch.ravel: Op(dim_map=lambda tensor: dim_flatten(tensor.ndim)),
    Tensor.repeat: Op(dim_map=lambda self, *sizes: dim_repeat(self.ndim, sizes)),
    torch.reshape: Op(
        dim_map=lambda input, shape: view_groups(input.shape, shape),
        shape_argnum=1,
    ),
    torch.squeeze: Op(dim_map=lambda input, dim=None: dim_squeeze(input.shape, dim)),
    torch.tile: Op(dim_map=lambda input, dims: dim_tile(input.ndim, dims)),
    torch.transpose: Op(
        dim_map=lambda input, dim0, dim1: dim_transpose(input.ndim, dim0, dim1)
    ),
    torch.unsqueeze: Op(dim_map=lambda input, dim: dim_unsqueeze(input.ndim, dim)),
    Tensor.view: Op(
        dim_map=lambda input, *shape: view_groups(input.shape, shape),
        shape_argnum=1,
    ),
}


def propagate_shape_and_sharding(
    in_shard: Sequence[Placement],
    local_in_shape: Shape,
    rule: DimMap,
    mesh_sizes: Shape,
) -> Tuple[Shape, Optional[Sequence[Placement]], torch.Tensor]:
    """
    Determine output sharding and tensor shape based on given global tensor shape and input sharding.

    Takes as input the global shape of the tensor, and the input sharding,
    and produce corresponding output sharding and shape of the output tensor.

    Sharding propagation follows mapped dimensions:
    - An output dimension that maps directly to an input dimension is sharded equally
    - An output dimension that is a flattened set of input dimensions can only be
      sharded if only the leftmost flattened dimension is sharded.
    - An output dimension that is a split of the input dimension can only be sharded
      if the leftmost split size is divisible by the mesh dimension
    """
    assert len(in_shard) == len(mesh_sizes)
    sharded_in_dims: Set[int] = {s.dim for s in in_shard if isinstance(s, Shard)}
    # for each input dim, for each mesh dim, provides a list of possible shardable dimensions
    shardable_dims: torch.Tensor = torch.ones(
        (len(local_in_shape), len(mesh_sizes)), dtype=torch.bool
    )

    # in case an input dimension disappears (e.g. collapsing, reduction)
    # we cannot shard in that dimension (we need a replication fall-back rule)

    seen_input_dims: Set[int] = set()

    def collect_used_inputs(cmd: DimSpec) -> None:
        if isinstance(cmd, InputDim):
            seen_input_dims.add(cmd.input_dim)
        for inp in cmd.inputs():
            collect_used_inputs(inp)

    for cmd in rule:
        collect_used_inputs(cmd)
    for dim in range(len(local_in_shape)):
        shardable_dims[dim, :] = dim in seen_input_dims

    def get_dim_size(cmd: DimSpec) -> Tuple[int, Optional[InputDim]]:
        if isinstance(cmd, InputDim):
            seen_input_dims.add(cmd.input_dim)
            return (
                local_in_shape[cmd.input_dim],
                cmd if cmd.input_dim in sharded_in_dims else None,
            )
        elif isinstance(cmd, Flatten):
            for dim in cmd.input_dims[1:]:
                if isinstance(dim, InputDim):
                    shardable_dims[dim.input_dim, :] = False
            dim0 = cmd.input_dims[0]
            return (
                prod(get_dim_size(a)[0] for a in cmd.input_dims),
                dim0
                if isinstance(dim0, InputDim) and dim0.input_dim in sharded_in_dims
                else None,
            )
        elif isinstance(cmd, Split):
            _, in_dim = get_dim_size(cmd.input_dim)
            out_size = cmd.group_shape[cmd.split_id]
            if cmd.split_id == 0 and in_dim is not None:
                # we need to check that the input dimension is divisible
                # by the size of the submesh we're sharding it on
                # NOTE: it would be possible to shard the same input dimension
                # on more than one mesh dimension. In that case, the dimension
                # needs to be divisible by the product of mesh sizes.
                # In order to keep the problem more tractable, we will not consider
                # double resharding as a suggestion (e.g. [Shard(0), Shard(0) ])
                # but we will allow it if that's the input and it's compatible

                # 1. is this dimension shardable on each individual mesh dim?
                for mesh_dim, mesh_dim_size in enumerate(mesh_sizes):
                    shardable_dims[in_dim.input_dim, mesh_dim] = (
                        out_size % mesh_dim_size == 0
                    )

                # 2. here we special case things like [Shard(0), Shard(0)]
                submesh_size = 1
                for size, shard in zip(mesh_sizes, in_shard):
                    if isinstance(shard, Shard) and shard.dim == in_dim:
                        submesh_size *= size
                assert (
                    out_size % submesh_size == 0
                ), f"Resulting dimension size {out_size} is not divisible by its mesh dimension {submesh_size}."

            # we will only shard our first component of the split
            return out_size, in_dim if cmd.split_id == 0 else None
        elif isinstance(cmd, Singleton):
            return 1, None
        elif isinstance(cmd, Broadcast):
            return cmd.dim_size, None
        elif isinstance(cmd, NewDim):
            return cmd.size, None
        elif isinstance(cmd, Repeat):
            size, in_dim = get_dim_size(cmd.input_dim)
            if in_dim is not None:
                shardable_dims[in_dim.input_dim, :] = False
            return size * cmd.times, None
        else:
            raise RuntimeError(f"cmd not found: {cmd}, in rule: {rule}")

    dim_map = {}
    out_shape = []
    for dim, cmd in enumerate(rule):
        out_size, in_dim = get_dim_size(cmd)
        out_shape.append(out_size)
        if in_dim is not None:
            dim_map[in_dim.input_dim] = dim

    needs_reshard = any(
        isinstance(placement, Shard) and not shardable_dims[placement.dim][mesh_dim]
        for mesh_dim, placement in enumerate(in_shard)
    )

    output_placements = (
        None
        if needs_reshard
        else [Shard(dim_map[s.dim]) if isinstance(s, Shard) else s for s in in_shard]
    )

    return (tuple(out_shape), output_placements, shardable_dims)


def register_prop_rule_map(
    aten_op_overload: torch._ops.OpOverload,
    local_op_name: Callable[..., torch.Tensor],
    schema_info: Optional[RuntimeSchemaInfo] = None,
) -> None:
    spec: Op = ops[local_op_name]

    @register_prop_rule(aten_op_overload, schema_info=schema_info)
    def reshape_prop(op_schema: OpSchema) -> OutputSharding:
        rules = spec.dim_map(*op_schema.args_schema, **op_schema.kwargs_schema)
        input_dtensor_spec = cast(DTensorSpec, op_schema.args_schema[0])
        mesh = input_dtensor_spec.mesh

        assert isinstance(
            input_dtensor_spec, DTensorSpec
        ), "Expected first input to be a DTensorSpec"
        global_in_shape = input_dtensor_spec.shape
        assert global_in_shape is not None, "Shape required."

        with disable_proxy_modes_tracing(), unset_fake_temporarily():
            (
                global_out_shape,
                shard_out,
                shardable_dims,
            ) = propagate_shape_and_sharding(
                input_dtensor_spec.placements,
                tuple(global_in_shape),
                rules,
                mesh.shape,
            )

        if shard_out is not None:
            # no reshard needed
            output_dtensor_spec = DTensorSpec(mesh=mesh, placements=tuple(shard_out))

            # We only need the local shape to lower the call into the local op
            args = op_schema.args_schema
            shape_argnum = spec.shape_argnum
            if shape_argnum is not None:
                # compute the local shape from the global shape, then return
                # a resharding even if we don't really reshard, the only reason
                # for this type of resharding is to lower the global shape to
                # local shape
                local_out_shape = compute_local_shape(
                    list(global_out_shape), mesh, shard_out
                )

                suggested_schema = OpSchema(
                    op=op_schema.op,
                    args_schema=args[:shape_argnum]
                    + (tuple(local_out_shape),)
                    + args[shape_argnum + 1 :],
                    kwargs_schema=op_schema.kwargs_schema,
                )
                return OutputSharding(
                    output_spec=output_dtensor_spec,
                    schema_suggestions=[suggested_schema],
                    needs_redistribute=True,
                )

            return OutputSharding(output_spec=output_dtensor_spec)

        else:
            # TODO: optimize this. we shouldn't simply blindly replicate
            #       unshardable dims ...
            # FIXME: this can be wrong for situations where we have
            #        [Shard(0), Shard(0)]
            suggested_placements = [
                p
                if not isinstance(p, Shard) or shardable_dims[p.dim][mesh_dim]
                else Replicate()
                for mesh_dim, p in enumerate(input_dtensor_spec.placements)
            ]
            return OutputSharding(
                output_spec=None,
                schema_suggestions=[
                    OpSchema(
                        op=op_schema.op,
                        args_schema=(
                            DTensorSpec(
                                placements=tuple(suggested_placements),
                                mesh=input_dtensor_spec.mesh,
                                tensor_meta=input_dtensor_spec.tensor_meta,
                            ),
                        )
                        + op_schema.args_schema[1:],
                        kwargs_schema=op_schema.kwargs_schema,
                    )
                ],
            )


register_prop_rule_map(aten.squeeze.default, torch.squeeze)
register_prop_rule_map(
    aten.squeeze.dim, torch.squeeze, schema_info=RuntimeSchemaInfo(1)
)
register_prop_rule_map(aten.view.default, Tensor.view, schema_info=RuntimeSchemaInfo(1))
register_prop_rule_map(
    aten.reshape.default, torch.reshape, schema_info=RuntimeSchemaInfo(1)
)
register_prop_rule_map(
    aten._unsafe_view.default, Tensor.view, schema_info=RuntimeSchemaInfo(1)
)
register_prop_rule_map(
    aten.unsqueeze.default, torch.unsqueeze, schema_info=RuntimeSchemaInfo(1)
)
register_prop_rule_map(
    aten.expand.default, Tensor.expand, schema_info=RuntimeSchemaInfo(1)
)
register_prop_rule_map(
    aten.permute.default, torch.permute, schema_info=RuntimeSchemaInfo(1)
)
register_prop_rule_map(
    aten.repeat.default, Tensor.repeat, schema_info=RuntimeSchemaInfo(1)
)
register_prop_rule_map(
    aten.transpose.int, torch.transpose, schema_info=RuntimeSchemaInfo(1)
)
