# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor
from torch._prims_common import DimsType
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpSpec,
    OpStrategy,
    RuntimeSchemaInfo,
    StrategyType,
)
from torch.distributed.tensor._ops.utils import (
    generate_redistribute_costs,
    normalize_dim,
    normalize_dims,
    prod,
    register_op_strategy,
)
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Placement,
    Replicate,
    Shard,
)


aten = torch.ops.aten

Shape = tuple[int, ...]


@dataclass
class DimSpec:
    """Specifies how an output dimension maps to an input dimension."""

    def inputs(self) -> Iterable["DimSpec"]:
        return ()


# Rules that map each dimension of the output to dimensions of the input tensor
DimMap = tuple[DimSpec, ...]


@dataclass
class Singleton(DimSpec):
    """Output dimension is a singleton."""


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
    def new(cls, dim: DimSpec, group_shape: tuple[int, ...], idx: int) -> DimSpec:
        if not len(group_shape) > 0:
            raise AssertionError(
                f"Expected group_shape length > 0, got {len(group_shape)}"
            )
        if len(group_shape) == 1:
            # not really a group, just return the input dim back
            if not idx == 0:
                raise AssertionError(f"Expected idx == 0, got {idx}")
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
    if not len(shape) >= len(input_shape):
        raise AssertionError(
            f"Expected len(shape) >= len(input_shape), got {len(shape)} < {len(input_shape)}"
        )

    # 1. create padded input dimensions
    padded_input = dim_pad_left(len(input_shape), len(shape))
    # 2. check that input shapes are compatible
    mapping = []
    for p, desired_s in zip(padded_input, shape):
        if isinstance(p, Singleton):
            actual_s = 1
            if not desired_s >= 0:
                raise AssertionError(f"Expected desired_s >= 0, got {desired_s}")
        else:
            if not isinstance(p, InputDim):
                raise AssertionError(f"DimSpec not supported in expand: {p}")
            actual_s = input_shape[p.input_dim]
            if not (actual_s == 1 or desired_s == -1 or desired_s == actual_s):
                raise AssertionError(
                    f"Expected actual_s == 1 or desired_s == -1 or "
                    f"desired_s == actual_s, got actual_s={actual_s}, desired_s={desired_s}"
                )
        mapping.append(
            p
            if desired_s in (1, -1) or desired_s == actual_s
            else Broadcast.new(p, desired_s)
        )
    return tuple(mapping)


def normalize_sizes(sizes: Shape | tuple[Shape]) -> Shape:
    if isinstance(sizes[0], int):
        return cast(Shape, sizes)
    elif len(sizes) == 1:
        return sizes[0]
    else:
        raise RuntimeError("Size must be int... or tuple")


def dim_flatten(ndim: int, start_dim=0, end_dim=-1) -> DimMap:
    if ndim == 0:
        return (Singleton(),)
    elif ndim == 1:
        return (InputDim(0),)
    else:
        # only flattening dims from start_dim to end_dim (inclusive)
        # other dims are passed through
        if end_dim < 0:
            end_dim += ndim
        results: list[DimSpec] = [InputDim(i) for i in range(start_dim)]
        results.append(
            Flatten.new(tuple(InputDim(i) for i in range(start_dim, end_dim + 1)))
        )
        results.extend([InputDim(i) for i in range(end_dim + 1, ndim)])
        return tuple(results)


def dim_movedim(
    ndim: int,
    input: DimsType,
    destination: DimsType,
) -> DimMap:
    input = normalize_dims(input, ndim)
    destination = normalize_dims(destination, ndim)

    if not len(input) == len(destination):
        raise AssertionError(
            f"Expected len(input) == len(destination), got {len(input)} != {len(destination)}"
        )
    input_set = set(input)
    if not len(input_set) == len(input):
        raise AssertionError("Found repeated input dims")
    if not len(set(destination)) == len(destination):
        raise AssertionError("Found repeated output dims")
    if not max(input) < ndim:
        raise AssertionError(f"Expected max(input) < ndim, got {max(input)} >= {ndim}")
    if not max(destination) < ndim:
        raise AssertionError(
            f"Expected max(destination) < ndim, got {max(destination)} >= {ndim}"
        )

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
    if not len(sizes) >= ndim:
        raise AssertionError(
            f"Number of dimensions of repeat dims {sizes} can not be smaller than number of dimensions of tensor {ndim}."
        )
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
    if not len(infers) <= 1:
        raise AssertionError("can only infer one size")
    if infers:
        size = -size
        missing_size = total_size // size
        if not total_size % size == 0:
            raise AssertionError(
                f"size inferred for -1 is not integral {sizes} should have {total_size} elements."
            )
        return tuple(s if s != -1 else missing_size for s in sizes)
    if not size == total_size:
        raise AssertionError(f"sizes do not match {total_size} vs {size}")
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

    - output dimension 0 maps to input dimension 0
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

    if not from_nelem == prod(to_size):
        raise AssertionError("Total view shape does not add up")

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
                tuple(InputDim(fi) for fi in from_group_dim if from_size[fi] >= 1)
            )
            result_pp += [
                Split.new(flattened, tuple(to_group_shape), i)
                for i in range(len(to_group_shape))
            ]

    return tuple(result_pp)


def dim_tile(ndim: int, dims: tuple[int, ...]) -> DimMap:
    if len(dims) < ndim:
        dims = (1,) * (ndim - len(dims)) + dims
    return dim_repeat(ndim, dims)


def dim_transpose(ndim: int, dim1: int, dim2: int) -> DimMap:
    dim1 = normalize_dim(dim1, ndim)
    dim2 = normalize_dim(dim2, ndim)
    if not dim1 < ndim:
        raise AssertionError(f"Expected dim1 < ndim, got {dim1} >= {ndim}")
    if not dim2 < ndim:
        raise AssertionError(f"Expected dim2 < ndim, got {dim2} >= {ndim}")
    dimmap = [InputDim(i) for i in range(ndim)]
    swapdim = dimmap[dim1]
    dimmap[dim1] = dimmap[dim2]
    dimmap[dim2] = swapdim
    return tuple(dimmap)


def dim_squeeze(shape: Shape, dim: int | None = None) -> DimMap:
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


def dim_view_as_real(shape: Shape) -> DimMap:
    ndim = len(shape)
    results: list[DimSpec] = [InputDim(i) for i in range(ndim - 1)]
    # each complex number is split into two real numbers,
    # resulting in one more dimension of size 2
    results.append(Split(InputDim(ndim - 1), (shape[-1], 2), 0))
    results.append(Split(InputDim(ndim - 1), (shape[-1], 2), 1))
    return tuple(results)


def dim_reduction(ndim: int, dim_or_dims: DimsType | None, keepdim: bool) -> DimMap:
    """
    General fallback for reduction ops where Partial() does not apply.

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


dim_maps: dict[Callable[..., torch.Tensor], Callable[..., DimMap]] = {
    torch.atleast_1d: lambda x: dim_pad_left(x.ndim, 1),
    torch.atleast_2d: lambda x: dim_pad_left(x.ndim, 2),
    torch.atleast_3d: lambda x: dim_atleast_3d(x.ndim),
    torch.broadcast_to: lambda input, shape: expand(input.shape, shape),
    Tensor.expand: lambda self, *sizes: expand(self.shape, normalize_sizes(sizes)),
    torch.flatten: lambda tensor: dim_flatten(tensor.ndim),
    torch.movedim: lambda input, source, destination: dim_movedim(
        input.ndim, source, destination
    ),
    torch.permute: lambda input, dims: tuple(
        InputDim(i) for i in normalize_dims(dims, input.ndim)
    ),
    torch.ravel: lambda tensor: dim_flatten(tensor.ndim),
    Tensor.repeat: lambda self, *sizes: dim_repeat(self.ndim, sizes),
    torch.reshape: lambda input, shape: view_groups(input.shape, shape),
    torch.squeeze: lambda input, dim=None: dim_squeeze(input.shape, dim),
    torch.tile: lambda input, dims: dim_tile(input.ndim, dims),
    torch.transpose: lambda input, dim0, dim1: dim_transpose(input.ndim, dim0, dim1),
    torch.unsqueeze: lambda input, dim: dim_unsqueeze(input.ndim, dim),
    Tensor.view: lambda input, *shape: view_groups(input.shape, shape),
    torch.view_as_complex: lambda input: dim_flatten(input.ndim, input.ndim - 2),
    torch.view_as_real: lambda input: dim_view_as_real(input.shape),
}


def _is_last_shard_on_tensor_dim(mesh_dim, placements):
    """Check if mesh_dim is the last mesh dimension that shards on the same (or higher) tensor dim."""
    tensor_dim = placements[mesh_dim].dim
    return not any(
        isinstance(p, (Shard, _StridedShard)) and p.dim >= tensor_dim
        for p in placements[mesh_dim + 1 :]
    )


def propagate_shape_and_sharding(
    input_src_spec,
    global_input_shape: Shape,
    rule: DimMap,
    mesh_sizes: Shape,
    strict_view: bool = False,
) -> tuple[Sequence[Placement], Sequence[Placement]]:
    """
    Determine input target sharding and output sharding based on
    given global tensor shape and input source sharding.

    Sharding propagation follows mapped dimensions:
    - An output dimension that maps directly to an input dimension is sharded equally
    - An output dimension that is a flattened set of input dimensions can only be
      sharded if only the leftmost flattened dimension is sharded.
    - An output dimension that is a split of the input dimension can only be sharded
      if the leftmost split size is divisible by the mesh dimension
    """
    input_src_placements: Sequence[Placement] = input_src_spec.placements
    if not len(input_src_placements) == len(mesh_sizes):
        raise AssertionError(f"{input_src_placements} != {mesh_sizes}")
    # for each input dim, for each mesh dim, provides a list of possible shardable dimensions
    mesh = input_src_spec.mesh
    mesh_ndim = len(mesh_sizes)
    shardable_dims: dict[int, list[bool]] = {}

    # Track which mesh dims have been matched to an output dimension in Split operations
    # For _StridedShard placements, we need to match each mesh dim to exactly one output dim
    matched_mesh_dims: set[int] = set()

    # in case an input dimension disappears (e.g. collapsing, reduction)
    # we cannot shard in that dimension (we need a replication fall-back rule)
    seen_input_dims: set[int] = set()

    def collect_used_inputs(cmd: DimSpec) -> None:
        if isinstance(cmd, InputDim):
            seen_input_dims.add(cmd.input_dim)
        for inp in cmd.inputs():
            collect_used_inputs(inp)

    for cmd in rule:
        collect_used_inputs(cmd)
    for dim in range(len(global_input_shape)):
        shardable_dims[dim] = [dim in seen_input_dims] * mesh_ndim

    def maybe_get_shard_mesh_dim_and_placement(
        input_dim: InputDim,
    ) -> tuple[int | None, Shard | _StridedShard | None]:
        for mesh_dim, placement in enumerate(input_src_placements):
            if isinstance(placement, Shard) and placement.dim == input_dim.input_dim:
                return mesh_dim, placement
        return None, None

    def maybe_get_shard_mesh_dim_and_placement_split(
        current_dim: int,
        cmd: Split,
        placements,
    ) -> tuple[int | None, Shard | _StridedShard | None]:
        """Find the mesh dim and placement for an input dim in Split ops.

        Handles multi-mesh sharding (e.g. [Shard(0), Shard(0)]) by matching
        _StridedShard split_factors and skipping already-matched mesh dims.
        """
        for mesh_dim, placement in enumerate(placements):
            if not isinstance(placement, Shard | _StridedShard):
                continue
            if placement.dim != current_dim:
                continue
            if mesh_dim in matched_mesh_dims:
                # This mesh dim already matched a previous output dimension
                continue

            if isinstance(placement, _StridedShard):
                # Compute expected split_factor for this mesh dim at this split_id
                expected_split_factor = math.prod(cmd.group_shape[0 : cmd.split_id])
                # Divide by mesh sizes of earlier mesh dims that shard the same input dim
                for m in range(mesh_dim):
                    p = placements[m]
                    if isinstance(p, Shard | _StridedShard) and p.dim == current_dim:
                        expected_split_factor = expected_split_factor // mesh_sizes[m]
                if placement.split_factor == expected_split_factor:
                    return mesh_dim, placement
                # Split factor doesn't match - this mesh dim is for a different output dim
            else:
                # For regular Shard, just return the first unmatched one
                return mesh_dim, placement
        return None, None

    # This function fills shardable_dims, returns the input dim(s) to shard on,
    # and raises errors for strict_view when sharding is incompatible.
    def get_in_dim_to_shard(cmd: DimSpec) -> InputDim | list[InputDim] | None:
        if isinstance(cmd, InputDim):
            return cmd
        elif isinstance(cmd, Flatten):
            sharded_dims = []
            num_input_dims = len(cmd.input_dims)
            for i, dim in enumerate(cmd.input_dims):
                # so far all Flatten is always composed of InputDims; revisit this if needed
                if not isinstance(dim, InputDim):
                    raise AssertionError(f"Expected InputDim, got {type(dim)}")
                can_shard_dim = True
                shard_mesh_dim, shard_placement = (
                    maybe_get_shard_mesh_dim_and_placement(dim)
                )
                input_sharded = shard_mesh_dim is not None
                is_last_input_dim = i == num_input_dims - 1
                if i > 0:
                    if strict_view and input_sharded:
                        assert shard_placement is not None
                        # Check for uneven sharding on non-last dimensions
                        if not is_last_input_dim:
                            tensor_dim_size = global_input_shape[shard_placement.dim]
                            mesh_dim_size = mesh_sizes[shard_mesh_dim]
                            if tensor_dim_size % mesh_dim_size != 0:
                                raise RuntimeError(
                                    f"Cannot flatten unevenly sharded tensor: "
                                    f"dimension {dim.input_dim} (size {tensor_dim_size}) "
                                    f"is not evenly divisible by mesh dimension {shard_mesh_dim} (size {mesh_dim_size}). "
                                    f"Please redistribute the tensor before this operation."
                                )
                        for x in range(dim.input_dim + 1):
                            shardable_dims[x] = [True] * mesh_ndim
                        sharded_dims.append(dim)
                elif input_sharded:
                    assert shard_placement is not None
                    tensor_dim_size = global_input_shape[shard_placement.dim]
                    mesh_dim_size = mesh_sizes[shard_mesh_dim]
                    sharded_dims.append(dim)
                    if tensor_dim_size % mesh_dim_size != 0:
                        can_shard_dim = False
                        if strict_view:
                            raise RuntimeError(
                                f"Cannot flatten unevenly sharded tensor: "
                                f"dimension {dim.input_dim} (size {tensor_dim_size}) "
                                f"is not evenly divisible by mesh dimension {shard_mesh_dim} (size {mesh_dim_size}). "
                                f"Please redistribute the tensor before this operation."
                            )
                shardable_dims[dim.input_dim] = [can_shard_dim] * mesh_ndim

            if len(sharded_dims) > 0:
                return sharded_dims
            else:
                if not isinstance(cmd.input_dims[0], InputDim):
                    raise AssertionError(
                        f"Expected InputDim, got {type(cmd.input_dims[0])}"
                    )
                return cmd.input_dims[0]
        elif isinstance(cmd, Split):
            in_dim = get_in_dim_to_shard(cmd.input_dim)
            # Flatten returns a list of sharded InputDims; unwrap to the
            # first one so the Split handler can use in_dim.input_dim.
            if isinstance(in_dim, list):
                in_dim = in_dim[0] if len(in_dim) > 0 else None
            out_size = cmd.group_shape[cmd.split_id]
            if in_dim is not None:
                shard_mesh_dim, input_src_placement = (
                    maybe_get_shard_mesh_dim_and_placement_split(
                        in_dim.input_dim,
                        cmd,
                        input_src_placements,
                    )
                )
                if shard_mesh_dim is not None and isinstance(
                    input_src_placement, _StridedShard
                ):
                    is_last_split_dim = cmd.split_id == len(cmd.group_shape) - 1
                    if (
                        strict_view
                        and not is_last_split_dim
                        and out_size % mesh_sizes[shard_mesh_dim] != 0
                    ):
                        raise RuntimeError(
                            f"Cannot unflatten unevenly sharded tensor: "
                            f"output dimension {cmd.split_id} (size {out_size}) "
                            f"is not evenly divisible by mesh dimension {shard_mesh_dim} "
                            f"(size {mesh_sizes[shard_mesh_dim]}). "
                            f"Please redistribute the tensor before this operation."
                        )
                    matched_mesh_dims.add(shard_mesh_dim)
                    if in_dim.input_dim in shardable_dims:
                        is_shardable = (
                            out_size % mesh_sizes[shard_mesh_dim] == 0
                            or is_last_split_dim
                        )
                        shardable_dims[in_dim.input_dim][shard_mesh_dim] = is_shardable
            if cmd.split_id == 0 and in_dim is not None:
                # Check that the first split output dim is shardable per mesh dim.
                # split_id == 0 with group_shape >= 2 means is_last_split_dim is always
                # False, so uneven sharding is not allowed on this dimension.
                shardable_dims[in_dim.input_dim] = [
                    out_size % mesh_dim_size == 0
                    for (mesh_dim, mesh_dim_size) in enumerate(mesh_sizes)
                ]

                shard_mesh_dim, _ = maybe_get_shard_mesh_dim_and_placement(in_dim)
                if strict_view and shard_mesh_dim is not None:
                    if not shardable_dims[in_dim.input_dim][shard_mesh_dim]:
                        raise RuntimeError(
                            f"Cannot unflatten unevenly sharded tensor: "
                            f"output dimension {cmd.split_id} (size {out_size}) "
                            f"is not evenly divisible by mesh dimension {shard_mesh_dim} (size {mesh_sizes[shard_mesh_dim]}). "
                            f"Please redistribute the tensor before this operation."
                        )

            # we will only shard our first component of the split
            return in_dim if cmd.split_id == 0 else None
        elif isinstance(cmd, Repeat):
            in_dim = get_in_dim_to_shard(cmd.input_dim)
            if isinstance(in_dim, list):
                for d in in_dim:
                    shardable_dims[d.input_dim] = [False] * mesh_ndim
            elif in_dim is not None:
                shardable_dims[in_dim.input_dim] = [False] * mesh_ndim
            return None
        else:
            return None

    # for each output dim, find the corresponding input dim in terms of sharding prop
    input_dim_to_output_dims = {}
    for output_dim, cmd in enumerate(rule):
        in_dims = get_in_dim_to_shard(cmd)
        if isinstance(in_dims, list) and len(in_dims) > 0:
            for in_dim in in_dims:
                if in_dim is not None:
                    assert in_dim.input_dim not in input_dim_to_output_dims
                    input_dim_to_output_dims[in_dim.input_dim] = [output_dim]
        else:
            if in_dims is not None and isinstance(in_dims, InputDim):
                if in_dims.input_dim not in input_dim_to_output_dims:
                    input_dim_to_output_dims[in_dims.input_dim] = [output_dim]
                else:
                    input_dim_to_output_dims[in_dims.input_dim].append(output_dim)
            # For Split with split_id > 0, in_dims is None but we still need to track
            # the output dim for _StridedShard unflatten support
            elif isinstance(cmd, Split):
                input_dim = cmd.input_dim
                while isinstance(input_dim, (Flatten, Split)):
                    if isinstance(input_dim, Flatten):
                        input_dim = input_dim.input_dims[0]
                    else:
                        input_dim = input_dim.input_dim
                if isinstance(input_dim, InputDim):
                    if input_dim.input_dim in input_dim_to_output_dims:
                        input_dim_to_output_dims[input_dim.input_dim].append(output_dim)

    input_tgt_placements: list[Placement] = []
    for mesh_dim, p in enumerate(input_src_placements):
        if isinstance(p, Shard | _StridedShard) and not shardable_dims[p.dim][mesh_dim]:
            input_tgt_placements.append(Replicate())
        else:
            input_tgt_placements.append(p)

    def _rewrite_shard_dim(
        p: Shard | _StridedShard, local_tensor_shapes, rule, mesh_dim, placements
    ):
        """Rewrite shard dim to the corresponding output dim.

        For _StridedShard, the split_factor is kept unchanged because view()
        doesn't redistribute — only the dim attribute changes.
        """

        if isinstance(p, _StridedShard):
            tgt_shard_dims = input_dim_to_output_dims[p.dim]
            found_split_cmd = False
            # Phase 1: prefix-product matching — the split_factor encodes
            # position within the unflatten (pure SS+SS case).
            prefix_match_idx = None
            for idx, candidate_dim in enumerate(tgt_shard_dims):
                cmd = rule[candidate_dim]
                if isinstance(cmd, Split):
                    found_split_cmd = True
                    expected_sf = math.prod(cmd.group_shape[0 : cmd.split_id])
                    for m in range(mesh_dim):
                        other_p = placements[m]
                        if (
                            isinstance(other_p, (_StridedShard, Shard))
                            and other_p.dim == p.dim
                        ):
                            expected_sf //= mesh.size(m)
                    if expected_sf == p.split_factor:
                        prefix_match_idx = idx
                        break

            if prefix_match_idx is not None:
                # SS resolves into contiguous sharding → Shard.
                tgt_shard_dim = tgt_shard_dims.pop(prefix_match_idx)
                local_tensor_shapes[p.dim] = local_tensor_shapes[p.dim] // mesh.size(
                    mesh_dim
                )
                return Shard(tgt_shard_dim)

            # Phase 2: chunk-based matching — the split_factor encodes
            # mesh structure (SS+S on same dim). The SS stays as SS.
            chunk = global_input_shape[p.dim] // (mesh_sizes[mesh_dim] * p.split_factor)
            chunk_match_idx = None
            for idx, candidate_dim in enumerate(tgt_shard_dims):
                cmd = rule[candidate_dim]
                if isinstance(cmd, Split):
                    inner_size = math.prod(cmd.group_shape[cmd.split_id + 1 :])
                    # When Split operates on a Flatten, scale chunk by
                    # trailing input dims so it's in flattened space.
                    effective_chunk = chunk
                    if isinstance(cmd.input_dim, Flatten):
                        trailing_size = 1
                        found_p_dim = False
                        for flat_dim in cmd.input_dim.input_dims:
                            if isinstance(flat_dim, InputDim):
                                if flat_dim.input_dim == p.dim:
                                    found_p_dim = True
                                elif found_p_dim:
                                    trailing_size *= global_input_shape[
                                        flat_dim.input_dim
                                    ]
                        effective_chunk = chunk * trailing_size
                    if (
                        effective_chunk >= inner_size
                        and effective_chunk % inner_size == 0
                    ):
                        chunk_match_idx = idx
                        break

            if chunk_match_idx is not None:
                tgt_shard_dim = tgt_shard_dims[chunk_match_idx]
            elif found_split_cmd and strict_view:
                raise RuntimeError(
                    f"Cannot unflatten tensor with _StridedShard placement: "
                    f"split_factor={p.split_factor} does not match any output dimension. "
                    f"This typically means the _StridedShard placement was constructed "
                    f"with a split_factor that is incompatible with the unflatten shape. "
                    f"Please redistribute the tensor before this operation."
                )
            else:
                tgt_shard_dim = tgt_shard_dims[0]
            local_tensor_shapes[p.dim] = local_tensor_shapes[p.dim] // mesh.size(
                mesh_dim
            )
            return _StridedShard(tgt_shard_dim, split_factor=p.split_factor)
        else:
            assert isinstance(p, Shard)
            tgt_shard_dims = input_dim_to_output_dims[p.dim]
            if len(tgt_shard_dims) == 1:
                tgt_shard_dim = tgt_shard_dims[0]
            else:
                # Unflatten: find the output dim with split_id=0.
                tgt_shard_dim = None
                for candidate_dim in tgt_shard_dims:
                    cmd = rule[candidate_dim]
                    if isinstance(cmd, Split) and cmd.split_id == 0:
                        tgt_shard_dim = candidate_dim
                        break
                if tgt_shard_dim is None:
                    tgt_shard_dim = tgt_shard_dims[0]
            cmd = rule[tgt_shard_dim]
            if isinstance(cmd, Split):
                assert tgt_shard_dim == p.dim
                output_placement = Shard(tgt_shard_dim)
            elif isinstance(cmd, InputDim):
                output_placement = Shard(tgt_shard_dim)
            else:
                # flatten from S to S/SS
                assert isinstance(cmd, Flatten)
                first_dim = cmd.input_dims[0]
                assert isinstance(first_dim, InputDim)
                input_start_idx = first_dim.input_dim
                if p.dim == input_start_idx:
                    output_placement = Shard(tgt_shard_dim)
                else:
                    split_factor = math.prod(
                        local_tensor_shapes[input_start_idx : p.dim]
                    )
                    output_placement = _StridedShard(
                        tgt_shard_dim, split_factor=split_factor
                    )

            if local_tensor_shapes[p.dim] % mesh.size(
                mesh_dim
            ) != 0 and not _is_last_shard_on_tensor_dim(mesh_dim, placements):
                raise RuntimeError(
                    f"Cannot shard unevenly distributed tensor: "
                    f"dimension {p.dim} (size {local_tensor_shapes[p.dim]}) "
                    f"is not evenly divisible by mesh dimension {mesh_dim} (size {mesh.size(mesh_dim)}). "
                    f"Please redistribute the tensor before this operation."
                )
            else:
                local_tensor_shapes[p.dim] = local_tensor_shapes[p.dim] // mesh.size(
                    mesh_dim
                )
            return output_placement

    local_tensor_shapes = list(input_src_spec.shape)
    output_placements = [
        _rewrite_shard_dim(p, local_tensor_shapes, rule, mesh_dim, input_tgt_placements)
        if isinstance(p, Shard | _StridedShard)
        else p
        for mesh_dim, p in enumerate(input_tgt_placements)
    ]

    return input_tgt_placements, output_placements


def register_op_strategy_map(
    aten_op_overload: torch._ops.OpOverload,
    local_op_name: Callable[..., torch.Tensor],
    schema_info: RuntimeSchemaInfo | None = None,
    strict_view: bool = False,
) -> None:
    """
    Helper that registers strategies for view-like operators that follow a pattern:
      (1) define the way input dims are split/combined to form output dims (dim_maps)
      (2) register a strategy for the op schema that uses the dim_map as a sharding prop rule

    strict_view: if True, we will error out if the view-operation would require resharding the input.
       Currently, this should be set to 'true' for any "view" ops.
       We could diverge behavior for "reshape" ops which could perform a redistribute implicitly.
    """
    dim_map: Callable[..., DimMap] = dim_maps[local_op_name]

    @register_op_strategy(aten_op_overload, schema_info=schema_info)
    def reshape_strategy(op_schema: OpSchema) -> StrategyType:
        rules = dim_map(*op_schema.args_schema, **op_schema.kwargs_schema)
        input_strategy = cast(OpStrategy, op_schema.args_schema[0])
        mesh = op_schema.get_mesh_from_args(validate=False)

        global_in_shape = input_strategy.shape
        if global_in_shape is None:
            raise AssertionError("Shape required.")

        output_strategy = OpStrategy([])
        for input_placement_strategy in input_strategy.strategies:
            input_src_spec = input_placement_strategy.output_spec

            input_tgt_placements, output_placements = propagate_shape_and_sharding(
                input_src_spec,
                tuple(global_in_shape),
                rules,
                mesh.shape,
                strict_view,
            )

            # TODO: optimize this. we shouldn't simply blindly replicate
            #       unshardable dims ...
            # FIXME: this can be wrong for situations where we have
            #        [Shard(0), Shard(0)]
            input_tgt_spec = DTensorSpec(
                placements=tuple(input_tgt_placements),
                mesh=mesh,
                tensor_meta=input_src_spec.tensor_meta,
            )
            redistribute_costs: list[list[float]] = [
                generate_redistribute_costs(input_strategy, input_tgt_spec)
            ]

            output_spec = DTensorSpec(mesh=mesh, placements=tuple(output_placements))
            output_strategy.strategies.append(
                OpSpec(
                    output_specs=output_spec,
                    input_specs=(input_tgt_spec,),
                    redistribute_cost=redistribute_costs,
                )
            )

        return output_strategy


register_op_strategy_map(aten.squeeze.default, torch.squeeze)
register_op_strategy_map(
    aten.squeeze_.dim, torch.squeeze, schema_info=RuntimeSchemaInfo(1)
)
register_op_strategy_map(
    aten.squeeze.dim, torch.squeeze, schema_info=RuntimeSchemaInfo(1)
)
register_op_strategy_map(
    aten.view.default,
    Tensor.view,
    schema_info=RuntimeSchemaInfo(1),
    strict_view=True,
)
register_op_strategy_map(
    aten.reshape.default, torch.reshape, schema_info=RuntimeSchemaInfo(1)
)
register_op_strategy_map(
    aten._unsafe_view.default,
    Tensor.view,
    schema_info=RuntimeSchemaInfo(1),
    strict_view=True,
)
register_op_strategy_map(
    aten.unsqueeze.default, torch.unsqueeze, schema_info=RuntimeSchemaInfo(1)
)
register_op_strategy_map(
    aten.expand.default, Tensor.expand, schema_info=RuntimeSchemaInfo(1)
)
register_op_strategy_map(
    aten.permute.default, torch.permute, schema_info=RuntimeSchemaInfo(1)
)
register_op_strategy_map(
    aten.repeat.default, Tensor.repeat, schema_info=RuntimeSchemaInfo(1)
)
register_op_strategy_map(
    aten.transpose.int, torch.transpose, schema_info=RuntimeSchemaInfo(1)
)
register_op_strategy_map(aten.view_as_complex.default, torch.view_as_complex)
register_op_strategy_map(aten.view_as_real.default, torch.view_as_real)
