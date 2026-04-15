# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import cast, NamedTuple

import torch
from torch import Tensor
from torch._prims_common import DimsType
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpSpec,
    OpStrategy,
    RuntimeSchemaInfo,
    StrategyType,
)
from torch.distributed.tensor._ops.single_dim_strategy import (
    _ShardingPlaceholder,
    register_single_dim_strategy,
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
    Partial,
    Placement,
    Replicate,
    Shard,
)


aten = torch.ops.aten

Shape = tuple[int, ...]


class ClaimedDim(NamedTuple):
    """An (input_dim, output_dim) pair claimed by a mesh dim's _StridedShard rewrite."""

    input_dim: int
    output_dim: int


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


@dataclass(eq=False)
class InputDim(DimSpec):
    """Output dimension maps directly to an input dimension."""

    input_dim: int

    def __eq__(self, other: object) -> bool:
        """Raises TypeError for non-DimSpec comparisons to catch accidental
        ``shard.dim == input_dim`` bugs where ``.input_dim`` was intended."""
        if isinstance(other, InputDim):
            return self.input_dim == other.input_dim
        if not isinstance(other, DimSpec):
            raise TypeError(
                f"Cannot compare InputDim with {type(other).__name__}. "
                f"Did you mean to use .input_dim?"
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash((InputDim, self.input_dim))


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
        from torch.fx.experimental.symbolic_shapes import guard_or_false

        return Singleton() if guard_or_false(size == 1) else NewDim(size)


@dataclass
class Repeat(DimSpec):
    """Output dimension is the input dimension repeated n-times."""

    input_dim: DimSpec
    times: int

    @classmethod
    def new(cls, dim: DimSpec, times: int) -> DimSpec:
        from torch.fx.experimental.symbolic_shapes import guard_or_false

        if guard_or_false(times == 1):
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
        from torch.fx.experimental.symbolic_shapes import guard_or_false, guard_or_true

        if not len(group_shape) > 0:
            raise AssertionError(
                f"Expected group_shape length > 0, got {len(group_shape)}"
            )
        if len(group_shape) == 1:
            # not really a group, just return the input dim back
            if not idx == 0:
                raise AssertionError(f"Expected idx == 0, got {idx}")
            return dim
        elif guard_or_false(group_shape[idx] == 1):
            return Singleton()
        else:
            # remove singletons from group
            # group_mapping = [(new_index, (shape, old_index)) ...]
            group_mapping = list(
                enumerate(
                    (s, i) for i, s in enumerate(group_shape) if guard_or_true(s != 1)
                )
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
    from torch.fx.experimental.symbolic_shapes import guard_or_false

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
            if not (
                guard_or_false(actual_s == 1)
                or guard_or_false(desired_s == -1)
                or guard_or_false(desired_s == actual_s)
            ):
                raise AssertionError(
                    f"Expected actual_s == 1 or desired_s == -1 or "
                    f"desired_s == actual_s, got actual_s={actual_s}, desired_s={desired_s}"
                )
        mapping.append(
            p
            if (
                guard_or_false(desired_s == 1)
                or guard_or_false(desired_s == -1)
                or guard_or_false(desired_s == actual_s)
            )
            else Broadcast.new(p, desired_s)
        )
    return tuple(mapping)


def normalize_sizes(sizes: Shape | tuple[Shape]) -> Shape:
    if isinstance(sizes[0], (int, torch.SymInt)):
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
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    infers = [i for i, s in enumerate(sizes) if guard_or_false(s == -1)]
    size = prod(sizes)
    if not len(infers) <= 1:
        raise AssertionError("can only infer one size")
    if infers:
        size = -size
        missing_size = total_size // size
        torch._check(
            total_size % size == 0,
            lambda: f"size inferred for -1 is not integral {sizes} should have {total_size} elements.",
        )
        return tuple(s if not guard_or_false(s == -1) else missing_size for s in sizes)
    torch._check(
        size == total_size,
        lambda: f"sizes do not match {total_size} vs {size}",
    )
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
    from torch.fx.experimental.symbolic_shapes import guard_or_false, guard_or_true

    from_nelem = prod(from_size)
    to_size = infer_size(from_nelem, normalize_sizes(to_size))

    torch._check(
        from_nelem == prod(to_size),
        lambda: "Total view shape does not add up",
    )

    from_idx = 0
    to_idx = 0
    from_len = len(from_size)
    to_len = len(to_size)

    result_pp: list[DimSpec] = []

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
        if guard_or_false(f == 1) and guard_or_true(t != 1):
            # produces ([1], [])
            to_idx -= 1
            to_group_shape = []
        elif guard_or_true(f != 1) and guard_or_false(t == 1):
            # produces ([], [1])
            from_idx -= 1
            from_group_dim = []
        else:
            # produces ([1], [1]),  ([2], [2]), ([2,3], [6])
            while guard_or_true(f != t):
                if (
                    t % f == 0 or t > f
                ):  # for easier symbolic comparisons, e.g. u0*u1 > u0
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
                tuple(
                    InputDim(fi)
                    for fi in from_group_dim
                    if guard_or_true(from_size[fi] >= 1)
                )
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


def dim_squeeze(shape: Shape, dim: DimsType | None = None) -> DimMap:
    # Operates on local shape; sharding_prop rewrites squeeze ops to squeeze.dims
    # with only globally-singleton dims before this is called.
    from torch.fx.experimental.symbolic_shapes import guard_or_true

    ndim = len(shape)
    if dim is None:
        target_dims = set(range(ndim))
    elif isinstance(dim, int):
        target_dims = {normalize_dim(dim, ndim)}
    else:
        target_dims = set(normalize_dims(dim, ndim))
    return tuple(
        InputDim(i)
        for i, s in enumerate(shape)
        if guard_or_true(s > 1) or i not in target_dims
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


def propagate_shape_and_sharding(
    input_src_placements: Sequence[Placement],
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
    - An output dimension that is a flattened set of input dimensions can be sharded:
      the first sharded dim stays as Shard, non-first sharded dims become _StridedShard
    - An output dimension that is a split of the input dimension can only be sharded
      if the leftmost split size is divisible by the mesh dimension
    """
    propagator = _ViewShardingPropagator(
        input_src_placements, global_input_shape, rule, mesh_sizes, strict_view
    )
    input_tgt_placements, input_to_output_tensor_dims = propagator.analyze()
    output_placements = propagator.rewrite_output_placements(
        input_tgt_placements, input_to_output_tensor_dims
    )
    return input_tgt_placements, output_placements


class _ViewShardingPropagator:
    """Two-phase sharding propagator for view ops.

    Phase 1 — ``analyze()``:
      Walks the DimMap rule and returns:
      - ``input_tgt_placements``: input placements with unshardable dims
        demoted to Replicate.
      - ``input_to_output_tensor_dims``: maps each input tensor dim to its
        output dim(s).  Cardinality encodes the op type: 1→1 for InputDim,
        N→1 for Flatten, 1→N for Split/unflatten.

    Phase 2 — ``rewrite_output_placements()``:
      Consumes both Phase 1 outputs.  Iterates mesh dims 0..n-1, maintaining:
      - ``strided_shard_claimed_dims``: (input_dim, output_dim) pairs already assigned
        to a mesh dim by _StridedShard rewriting.
      - ``local_tensor_shapes``: global shape progressively divided by each
        mesh dim's shard size.
      For each surviving Shard/_StridedShard, looks up the output dim(s) and
      produces the final output placement.
    """

    def __init__(
        self,
        input_src_placements: Sequence[Placement],
        global_input_shape: Shape,
        rule: DimMap,
        mesh_sizes: Shape,
        strict_view: bool,
    ) -> None:
        self.input_src_placements = input_src_placements
        self.global_input_shape = global_input_shape
        self.rule = rule
        self.mesh_sizes = mesh_sizes
        self.strict_view = strict_view
        self.mesh_ndim = len(mesh_sizes)

        # shard_allowed[input_dim][mesh_dim]: whether input_dim can stay
        # sharded on mesh_dim.  Populated by _analyze_dim and its helpers.
        self.shard_allowed: dict[int, list[bool]] = {}
        # Mesh dims whose _StridedShard has already been matched to an output dim.
        # Populated by _analyze_split.
        self.matched_strided_mesh_dims: set[int] = set()

    # ------------------------------------------------------------------
    # Public API: analyze → rewrite_output_placements
    # ------------------------------------------------------------------

    def analyze(
        self,
    ) -> tuple[Sequence[Placement], dict[int, list[int]]]:
        """Phase 1: walk the DimMap rule, return (input_tgt_placements, input_to_output_tensor_dims)."""
        input_dims_in_rule = self._input_dims_in_rule(self.rule)

        # Default: shardable if the dim appears in the rule. Refined by _analyze_*.
        for dim in range(len(self.global_input_shape)):
            self.shard_allowed[dim] = [dim in input_dims_in_rule] * self.mesh_ndim

        # Walk the rule to refine shard_allowed and build input_to_output_tensor_dims.
        #
        # Flatten example: view([2, 3, 4], [6, 4])
        #   rule = (Flatten(InputDim(0), InputDim(1)), InputDim(2))
        #   output_dim=0 (Flatten): hits the isinstance(cmd, Flatten) branch.
        #     Maps input dims 0 and 1 to output dim 0.  Result: {0: [0], 1: [0]}
        #   output_dim=1 (InputDim(2)): hits the len(in_dims) > 0 branch.
        #     Maps input dim 2 to output dim 1.  Result: {0: [0], 1: [0], 2: [1]}
        #
        # Split example: view([6], [2, 3])
        #   rule = (Split(InputDim(0), (2,3), 0), Split(InputDim(0), (2,3), 1))
        #   output_dim=0 (split_id=0): hits the len(in_dims) > 0 branch.
        #     Maps input dim 0 to output dim 0.  Result: {0: [0]}
        #   output_dim=1 (split_id=1): hits the isinstance(cmd, Split) branch
        #     because _analyze_split returns [] for split_id>0.  Chases root
        #     InputDim(0) and appends output dim 1.  Result: {0: [0, 1]}
        input_to_output_tensor_dims: dict[int, list[int]] = {}
        for output_dim, cmd in enumerate(self.rule):
            in_dims = self._analyze_dim(cmd)
            if isinstance(cmd, Flatten):
                for in_dim in in_dims:
                    if in_dim.input_dim in input_to_output_tensor_dims:
                        raise AssertionError(
                            f"Input dim {in_dim.input_dim} already mapped to output dims "
                            f"{input_to_output_tensor_dims[in_dim.input_dim]}"
                        )
                    input_to_output_tensor_dims[in_dim.input_dim] = [output_dim]
            elif len(in_dims) > 0:
                # InputDim (identity) or Split(split_id=0).
                in_dim = in_dims[0]
                if in_dim.input_dim not in input_to_output_tensor_dims:
                    input_to_output_tensor_dims[in_dim.input_dim] = [output_dim]
                else:
                    input_to_output_tensor_dims[in_dim.input_dim].append(output_dim)
            elif isinstance(cmd, Split):
                # Split(split_id>0): _analyze_split returned [], so chase the
                # root input dim and append this output dim to its existing entry.
                #
                # Flatten+Split example: view([2, 3], [3, 2])
                #   rule = (Split(Flatten(InputDim(0), InputDim(1)), (3,2), 0),
                #           Split(Flatten(InputDim(0), InputDim(1)), (3,2), 1))
                #   output_dim=0 (split_id=0): same as Split example above.
                #     Result: {0: [0]}
                #   output_dim=1 (split_id=1): same as Split example, but
                #     the chase unwraps the inner Flatten to find InputDim(0).
                #     Result: {0: [0, 1]}
                root_spec = cmd.input_dim
                while isinstance(root_spec, (Flatten, Split)):
                    if isinstance(root_spec, Flatten):
                        # _analyze_flatten always returns input_dims[0] as
                        # the first element (either as the only shardable dim
                        # in non-strict mode, or as the fallback when nothing
                        # is sharded), so split_id=0 uses it as the key in
                        # input_to_output_tensor_dims. Use [0] here to match.
                        root_spec = root_spec.input_dims[0]
                    else:
                        root_spec = root_spec.input_dim
                root = root_spec if isinstance(root_spec, InputDim) else None
                if root is not None and root.input_dim in input_to_output_tensor_dims:
                    input_to_output_tensor_dims[root.input_dim].append(output_dim)

        input_tgt_placements: list[Placement] = []
        for mesh_dim, p in enumerate(self.input_src_placements):
            if (
                isinstance(p, Shard | _StridedShard)
                and not self.shard_allowed[p.dim][mesh_dim]
            ):
                input_tgt_placements.append(Replicate())
            else:
                input_tgt_placements.append(p)
        return input_tgt_placements, input_to_output_tensor_dims

    def rewrite_output_placements(
        self,
        input_tgt_placements: Sequence[Placement],
        input_to_output_tensor_dims: dict[int, list[int]],
    ) -> list[Placement]:
        """Phase 2: consume analyze() outputs, return final output placements."""
        # (input_dim, output_dim) pairs claimed by earlier mesh dims
        # (via _rewrite_strided_shard), to avoid double-assignment.
        strided_shard_claimed_dims: set[ClaimedDim] = set()
        # Starts as global_input_shape; each mesh dim divides its sharded dim.
        local_tensor_shapes: list[int] = list(self.global_input_shape)

        output_placements: list[Placement] = []
        # Process mesh dims in order; _rewrite_*_shard relies on this for
        # truncating division safety in local_tensor_shapes.
        for mesh_dim, p in enumerate(input_tgt_placements):
            if isinstance(p, Shard):
                placement, local_tensor_shapes = self._rewrite_plain_shard(
                    p,
                    mesh_dim,
                    input_tgt_placements,
                    strided_shard_claimed_dims,
                    local_tensor_shapes,
                    input_to_output_tensor_dims,
                )
                output_placements.append(placement)
            elif isinstance(p, _StridedShard):
                placement, local_tensor_shapes = self._rewrite_strided_shard(
                    p,
                    mesh_dim,
                    input_tgt_placements,
                    strided_shard_claimed_dims,
                    local_tensor_shapes,
                    input_to_output_tensor_dims,
                )
                output_placements.append(placement)
            else:
                output_placements.append(p)
        return output_placements

    # ------------------------------------------------------------------
    # Analysis phase helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _input_dims_in_rule(rule: DimMap) -> set[int]:
        """Walk the DimMap rule tree and return all input dim indices that appear in it."""
        seen: set[int] = set()

        def _walk(cmd: DimSpec) -> None:
            if isinstance(cmd, InputDim):
                seen.add(cmd.input_dim)
            for inp in cmd.inputs():
                _walk(inp)

        for cmd in rule:
            _walk(cmd)
        return seen

    def _find_plain_shard(
        self, input_dim: InputDim
    ) -> tuple[int | None, Shard | _StridedShard | None]:
        """Find the mesh dim with a plain Shard on ``input_dim``.

        Only matches Shard, not _StridedShard.  Used by both _analyze_flatten
        and _analyze_split.  _find_shard_for_split is the counterpart that
        also matches _StridedShard with split_factor validation.
        """
        for mesh_dim, placement in enumerate(self.input_src_placements):
            if isinstance(placement, Shard) and placement.dim == input_dim.input_dim:
                return mesh_dim, placement
        return None, None

    def _find_shard_for_split(
        self,
        current_dim: int,
        cmd: Split,
        placements: Sequence[Placement],
    ) -> tuple[int | None, Shard | _StridedShard | None]:
        """Find the mesh dim and placement for an input dim in Split ops.

        Matches both Shard and _StridedShard:
        - Shard: plain unflatten, e.g. [6] Shard(0) → [2, 3].
        - _StridedShard: unflatten after a prior flatten that produced
          _StridedShard, e.g. [2,3,4] Shard(1) → flatten → [6,4]
          _StridedShard(0,sf=2) → unflatten → [2,3,4].  Validates that
          the split_factor matches the expected value for this split_id.
        """
        for mesh_dim, placement in enumerate(placements):
            if not isinstance(placement, Shard | _StridedShard):
                continue
            if placement.dim != current_dim:
                continue
            if mesh_dim in self.matched_strided_mesh_dims:
                continue

            if isinstance(placement, _StridedShard):
                expected_sf = self._expected_split_factor(
                    cmd, current_dim, mesh_dim, placements
                )
                if expected_sf == placement.split_factor:
                    return mesh_dim, placement
            else:
                return mesh_dim, placement
        return None, None

    def _analyze_flatten(self, cmd: Flatten) -> list[InputDim]:
        """Fill self.shard_allowed for Flatten; return sharded input dims."""
        from torch.fx.experimental.symbolic_shapes import guard_or_true

        sharded_dims: list[InputDim] = []
        num_input_dims = len(cmd.input_dims)
        for i, dim in enumerate(cmd.input_dims):
            if not isinstance(dim, InputDim):
                raise AssertionError(f"Expected InputDim, got {type(dim)}")
            shard_mesh_dim, shard_placement = self._find_plain_shard(dim)
            if shard_mesh_dim is None or shard_placement is None:
                continue  # default from analyze() already covers this
            tensor_dim_size = self.global_input_shape[shard_placement.dim]
            mesh_dim_size = self.mesh_sizes[shard_mesh_dim]
            can_shard_dim = True
            if self.strict_view:
                is_last_input_dim = i == num_input_dims - 1
                if not is_last_input_dim and guard_or_true(
                    tensor_dim_size % mesh_dim_size != 0
                ):
                    raise RuntimeError(
                        f"Cannot flatten unevenly sharded tensor: "
                        f"dimension {dim.input_dim} (size {tensor_dim_size}) "
                        f"is not evenly divisible by mesh dimension "
                        f"{shard_mesh_dim} (size {mesh_dim_size}). "
                        f"Please redistribute the tensor before this operation."
                    )
                sharded_dims.append(dim)
            else:
                # TODO: non-strict (reshape) should allow can_shard_dim = True
                # for non-first flatten dims, since strict_view already does.
                # Currently forces redistribution because the rewrite phase
                # wasn't originally implemented for this case.
                if i == 0:
                    sharded_dims.append(dim)
                    if guard_or_true(tensor_dim_size % mesh_dim_size != 0):
                        can_shard_dim = False
                else:
                    can_shard_dim = False
            self.shard_allowed[dim.input_dim] = [can_shard_dim] * self.mesh_ndim

        if len(sharded_dims) > 0:
            return sharded_dims
        # No sharded dims: e.g. Flatten([InputDim(0), InputDim(1)]) where
        # neither dim is sharded.  Return the first input dim so that
        # input_to_output_tensor_dims is populated for identity rewrites.
        if not isinstance(cmd.input_dims[0], InputDim):
            raise AssertionError(f"Expected InputDim, got {type(cmd.input_dims[0])}")
        return [cmd.input_dims[0]]

    def _analyze_split(self, cmd: Split) -> list[InputDim]:
        """Fill self.shard_allowed for Split; return shardable input dims."""
        from torch.fx.experimental.symbolic_shapes import guard_or_false, guard_or_true

        in_dims = self._analyze_dim(cmd.input_dim)
        if len(in_dims) == 0:
            return []
        in_dim = in_dims[0]
        out_size = cmd.group_shape[cmd.split_id]
        shard_mesh_dim, input_src_placement = self._find_shard_for_split(
            in_dim.input_dim, cmd, self.input_src_placements
        )
        # split_id == 0 sets the base shard_allowed for this input dim.
        # Later split_ids (processed in subsequent rule iterations) refine
        # individual mesh_dim entries via the _StridedShard branch below.
        if cmd.split_id == 0:
            self.shard_allowed[in_dim.input_dim] = [
                guard_or_false(out_size % mesh_dim_size == 0)
                for mesh_dim_size in self.mesh_sizes
            ]
            plain_mesh_dim, _ = self._find_plain_shard(in_dim)
            # Non-strict silently redistributes via shard_allowed=False above;
            # strict raises so the user knows to redistribute before view().
            if self.strict_view and plain_mesh_dim is not None:
                if not self.shard_allowed[in_dim.input_dim][plain_mesh_dim]:
                    raise RuntimeError(
                        f"Cannot unflatten unevenly sharded tensor: "
                        f"output dimension {cmd.split_id} (size {out_size}) "
                        f"is not evenly divisible by mesh dimension "
                        f"{plain_mesh_dim} (size {self.mesh_sizes[plain_mesh_dim]}). "
                        f"Please redistribute the tensor before this operation."
                    )
        if shard_mesh_dim is not None and isinstance(
            input_src_placement, _StridedShard
        ):
            # The last split dim doesn't require even divisibility because
            # its local size is inferred: local_last = local_flat / product
            # of earlier dims, and DTensor handles uneven local sizes.
            # Non-last dims must be evenly divisible because they appear as
            # fixed sizes in the local reshape — uneven division would make
            # the stride pattern inconsistent across devices.
            # E.g. [12] → [3, 4], _StridedShard targeting dim 1 (last),
            # mesh=3: 4%3≠0, but local shapes [3,2],[3,1],[3,1] are valid.
            is_last_split_dim = cmd.split_id == len(cmd.group_shape) - 1
            if (
                self.strict_view
                and not is_last_split_dim
                and guard_or_true(out_size % self.mesh_sizes[shard_mesh_dim] != 0)
            ):
                raise RuntimeError(
                    f"Cannot unflatten unevenly sharded tensor: "
                    f"output dimension {cmd.split_id} (size {out_size}) "
                    f"is not evenly divisible by mesh dimension {shard_mesh_dim} "
                    f"(size {self.mesh_sizes[shard_mesh_dim]}). "
                    f"Please redistribute the tensor before this operation."
                )
            # Prevents _find_shard_for_split from matching this mesh dim
            # again for a later split_id of the same Split group.
            self.matched_strided_mesh_dims.add(shard_mesh_dim)
            if in_dim.input_dim in self.shard_allowed:
                self.shard_allowed[in_dim.input_dim][shard_mesh_dim] = (
                    guard_or_false(out_size % self.mesh_sizes[shard_mesh_dim] == 0)
                    or is_last_split_dim
                )
        # Only split_id==0 returns the input dim for input_to_output_tensor_dims.
        # Later split_ids refine shard_allowed above but return [] — their
        # output dims are linked via the root-input-dim chase in analyze().
        return [in_dim] if cmd.split_id == 0 else []

    def _analyze_dim(self, cmd: DimSpec) -> list[InputDim]:
        """Dispatch one DimSpec: update self.shard_allowed, return input dim(s) to shard on."""
        if isinstance(cmd, InputDim):
            return [cmd]
        elif isinstance(cmd, Flatten):
            return self._analyze_flatten(cmd)
        elif isinstance(cmd, Split):
            return self._analyze_split(cmd)
        elif isinstance(cmd, Repeat):
            in_dims = self._analyze_dim(cmd.input_dim)
            for d in in_dims:
                self.shard_allowed[d.input_dim] = [False] * self.mesh_ndim
            return []
        else:
            return []

    # ------------------------------------------------------------------
    # Rewrite phase helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_last_shard_in_flatten_range(
        mesh_dim: int,
        placements: Sequence[Placement],
        flatten_start: int,
        flatten_end: int,
    ) -> bool:
        """Check if no later mesh dim shards a dim within the flatten range at or above this one.

        Uneven sharding on dim d breaks stride computation for all earlier dims
        that flatten together with d. Only dims within [flatten_start, flatten_end)
        matter; shards on dims outside the flatten range are independent.

        Requires: placements[mesh_dim] must be Shard or _StridedShard.
        """
        p = placements[mesh_dim]
        if not isinstance(p, (Shard, _StridedShard)):
            raise AssertionError(
                f"Expected Shard or _StridedShard at mesh_dim {mesh_dim}, got {type(p)}"
            )
        tensor_dim = p.dim
        return not any(
            isinstance(other_p, (Shard, _StridedShard))
            and flatten_start <= other_p.dim < flatten_end
            and other_p.dim >= tensor_dim
            for other_p in placements[mesh_dim + 1 :]
        )

    def _expected_split_factor(
        self,
        cmd: Split,
        sharded_dim: int,
        mesh_dim: int,
        placements: Sequence[Placement],
    ) -> int | None:
        """Compute the residual split factor for ``cmd`` after earlier mesh dims.

        Starts from ``math.prod(cmd.group_shape[:cmd.split_id])`` and divides
        out each earlier mesh dim that shards the same input dim.  Returns
        ``None`` if any earlier mesh size doesn't divide evenly.
        """
        sf = math.prod(cmd.group_shape[: cmd.split_id])
        for m in range(mesh_dim):
            other_p = placements[m]
            if (
                isinstance(other_p, (_StridedShard, Shard))
                and other_p.dim == sharded_dim
            ):
                if sf % self.mesh_sizes[m] != 0:
                    return None
                sf //= self.mesh_sizes[m]
        return sf

    def _find_keep_ss_dim(
        self,
        tgt_shard_dims: list[int],
        p: _StridedShard,
        mesh_dim: int,
    ) -> int | None:
        """Find an output dim where SS stays as SS.

        Returns the first output dim whose Split can accommodate the combined
        sharding (mesh_size * split_factor), or ``None`` if no dim fits.
        """
        total_shard = self.mesh_sizes[mesh_dim] * p.split_factor
        if self.global_input_shape[p.dim] % total_shard != 0:
            return None
        shard_size = self.global_input_shape[p.dim] // total_shard
        for candidate_dim in tgt_shard_dims:
            cmd = self.rule[candidate_dim]
            if isinstance(cmd, Split):
                inner_size = math.prod(cmd.group_shape[cmd.split_id + 1 :])
                # When a Split wraps a Flatten, the per-shard chunk covers
                # the sharded dim plus trailing dims flattened together.
                trailing_size = 1
                if isinstance(cmd.input_dim, Flatten):
                    found = False
                    for flat_dim in cmd.input_dim.input_dims:
                        if not isinstance(flat_dim, InputDim):
                            raise AssertionError(
                                f"Expected InputDim, got {type(flat_dim)}"
                            )
                        if flat_dim.input_dim == p.dim:
                            found = True
                        elif found:
                            trailing_size *= self.global_input_shape[flat_dim.input_dim]
                flattened_shard_size = shard_size * trailing_size
                if (
                    flattened_shard_size >= inner_size
                    and flattened_shard_size % inner_size == 0
                ):
                    return candidate_dim
        return None

    def _rewrite_plain_shard(
        self,
        p: Shard,
        mesh_dim: int,
        placements: Sequence[Placement],
        strided_shard_claimed_dims: set[ClaimedDim],
        local_tensor_shapes: list[int],
        input_to_output_tensor_dims: dict[int, list[int]],
    ) -> tuple[Placement, list[int]]:
        """Given a plain Shard(dim=X) input placement on a specific mesh dim,
        determine what output placement it maps to after the view op.

        For identity and unflatten, produces Shard on the output dim.  For
        flatten, Shard on the first flattened dim stays Shard, while Shard on
        a non-first dim produces _StridedShard (consumed later by
        _rewrite_strided_shard).

        Returns the output placement and a new local_tensor_shapes with this
        mesh dim's division applied.
        """
        # Output dims that input dim p.dim maps to, filtering out any
        # already claimed by _StridedShard rewriting on earlier mesh dims.
        tgt_shard_dims = [
            d
            for d in input_to_output_tensor_dims[p.dim]
            if ClaimedDim(p.dim, d) not in strided_shard_claimed_dims
        ]
        if len(tgt_shard_dims) == 0:
            raise AssertionError(
                f"No output dim available for Shard(dim={p.dim}) on mesh dim "
                f"{mesh_dim}. All output dims already claimed by earlier mesh dims."
            )
        if len(tgt_shard_dims) == 1:
            tgt_shard_dim = tgt_shard_dims[0]
        else:
            # Unflatten: one input dim maps to multiple output dims
            # (e.g. (24,) → (2, 3, 4) gives 3 splits). Plain Shard
            # always targets the split_id=0 output dim.
            tgt_shard_dim = next(
                (
                    d
                    for d in tgt_shard_dims
                    if isinstance(self.rule[d], Split)
                    and cast(Split, self.rule[d]).split_id == 0
                ),
                None,
            )
            if tgt_shard_dim is None:
                raise AssertionError(
                    f"No Split(split_id=0) found among unclaimed output dims "
                    f"{tgt_shard_dims} for Shard(dim={p.dim}) on mesh dim {mesh_dim}."
                )
        cmd = self.rule[tgt_shard_dim]
        if isinstance(cmd, Split) and isinstance(cmd.input_dim, Flatten):
            first_dim = cmd.input_dim.input_dims[0]
            if isinstance(first_dim, InputDim) and p.dim != first_dim.input_dim:
                raise RuntimeError(
                    f"Shard(dim={p.dim}) through Split(Flatten(...), {cmd.group_shape}) "
                    f"is not supported yet for non-first flatten dims."
                )
        if isinstance(cmd, (Split, InputDim)):
            # Split/InputDim: 1:1 dim mapping, sharding transfers directly.
            # Flatten needs stride computation below (multiple dims merge).
            new_shapes = list(local_tensor_shapes)
            new_shapes[p.dim] //= self.mesh_sizes[mesh_dim]
            return Shard(tgt_shard_dim), new_shapes
        if not isinstance(cmd, Flatten):
            raise AssertionError(f"Expected Flatten, got {type(cmd)}")
        first_dim = cmd.input_dims[0]
        last_dim = cmd.input_dims[-1]
        if not isinstance(first_dim, InputDim):
            raise AssertionError(f"Expected InputDim, got {type(first_dim)}")
        if not isinstance(last_dim, InputDim):
            raise AssertionError(f"Expected InputDim, got {type(last_dim)}")
        input_start_idx = first_dim.input_dim
        if p.dim == input_start_idx:
            output_placement: Placement = Shard(tgt_shard_dim)
        else:
            split_factor = math.prod(local_tensor_shapes[input_start_idx : p.dim])
            output_placement = _StridedShard(tgt_shard_dim, split_factor=split_factor)
        # Uneven sharding on a non-last flatten dim breaks _StridedShard:
        # split_factor (number of groups) must be the same on all devices,
        # but uneven division of a non-last dim makes group count vary.
        # E.g. [3,4]→[12] Shard(0) mesh=2: device 0 has 2 groups of 4,
        # device 1 has 1 group of 4 — no consistent split_factor.
        # The last dim is exempt: only group *size* varies, not count.
        flatten_end = last_dim.input_dim + 1
        if local_tensor_shapes[p.dim] % self.mesh_sizes[
            mesh_dim
        ] != 0 and not self._is_last_shard_in_flatten_range(
            mesh_dim, placements, input_start_idx, flatten_end
        ):
            raise RuntimeError(
                f"Cannot shard unevenly distributed tensor: "
                f"dimension {p.dim} (size {local_tensor_shapes[p.dim]}) "
                f"is not evenly divisible by mesh dimension "
                f"{mesh_dim} (size {self.mesh_sizes[mesh_dim]}). "
                f"Please redistribute the tensor before this operation."
            )
        new_shapes = list(local_tensor_shapes)
        new_shapes[p.dim] //= self.mesh_sizes[mesh_dim]
        return output_placement, new_shapes

    def _rewrite_strided_shard(
        self,
        p: _StridedShard,
        mesh_dim: int,
        placements: Sequence[Placement],
        strided_shard_claimed_dims: set[ClaimedDim],
        local_tensor_shapes: list[int],
        input_to_output_tensor_dims: dict[int, list[int]],
    ) -> tuple[Placement, list[int]]:
        """Rewrite _StridedShard placement to target the correct output dim.

        _StridedShard inputs arise from a prior flatten on a non-first dim
        (produced by _rewrite_plain_shard above).  The interesting case is
        unflatten (Split rule): the split_factor may resolve to contiguous
        sharding (producing Shard) or stay as _StridedShard.  For
        identity/flatten rules, falls through to the fallback and keeps the
        placement as-is.

        Returns the output placement and a new local_tensor_shapes with this
        mesh dim's division applied.
        """
        tgt_shard_dims = [
            d
            for d in input_to_output_tensor_dims[p.dim]
            if ClaimedDim(p.dim, d) not in strided_shard_claimed_dims
        ]
        # Phase 1: resolve SS → Shard.  If an output dim's Split has a
        # group_shape prefix matching the split_factor, the strided pattern
        # is fully captured by the Split, so SS simplifies to Shard.
        # E.g. unflatten (6, 4) → (2, 3, 4) with SS(0, sf=2) on mesh (3):
        # sf=2 means 2 groups of contiguous data in dim 0.  Split into
        # (2, 3, 4) gives group_shape=(2, 3); prod(group_shape[:1])=2==sf,
        # so the strided pattern lands exactly on output dim 1 → Shard(1).
        for candidate_dim in tgt_shard_dims:
            cmd = self.rule[candidate_dim]
            if isinstance(cmd, Split):
                expected_sf = self._expected_split_factor(
                    cmd, p.dim, mesh_dim, placements
                )
                if expected_sf != p.split_factor:
                    continue
                strided_shard_claimed_dims.add(ClaimedDim(p.dim, candidate_dim))
                new_shapes = list(local_tensor_shapes)
                new_shapes[p.dim] //= self.mesh_sizes[mesh_dim]
                return Shard(candidate_dim), new_shapes

        # Phase 2: keep SS as SS.  Phase 1 is tried first because we prefer
        # resolving to the simpler Shard when possible.
        tgt_shard_dim = self._find_keep_ss_dim(tgt_shard_dims, p, mesh_dim)

        if tgt_shard_dim is None:
            if self.strict_view and any(
                isinstance(self.rule[d], Split) for d in tgt_shard_dims
            ):
                raise RuntimeError(
                    f"Cannot unflatten tensor with _StridedShard placement: "
                    f"split_factor={p.split_factor} does not match any output "
                    f"dimension. This typically means the _StridedShard placement "
                    f"was constructed with a split_factor that is incompatible "
                    f"with the unflatten shape. Please redistribute the tensor "
                    f"before this operation."
                )
            if len(tgt_shard_dims) == 0:
                raise AssertionError(
                    f"No unclaimed output dims for _StridedShard(dim={p.dim}) "
                    f"on mesh dim {mesh_dim}."
                )
            # Fallback for identity/flatten: tgt_shard_dims has exactly one
            # element, so [0] is correct.  For Split rules this is unreachable
            # in practice — the analysis phase rejects mismatched split_factors
            # via shard_allowed, forcing redistribution before we get here.
            tgt_shard_dim = tgt_shard_dims[0]
        new_shapes = list(local_tensor_shapes)
        new_shapes[p.dim] //= self.mesh_sizes[mesh_dim]
        return _StridedShard(tgt_shard_dim, split_factor=p.split_factor), new_shapes


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
                input_src_spec.placements,
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
                use_strided_shard_as_shard_order=False,
            )
            redistribute_costs: list[list[float]] = [
                generate_redistribute_costs(input_strategy, input_tgt_spec)
            ]

            output_spec = DTensorSpec(
                mesh=mesh,
                placements=tuple(output_placements),
                use_strided_shard_as_shard_order=False,
            )
            output_strategy.strategies.append(
                OpSpec(
                    output_specs=output_spec,
                    input_specs=(input_tgt_spec,),
                    redistribute_cost=redistribute_costs,
                )
            )

        return output_strategy


register_op_strategy_map(aten.squeeze.default, torch.squeeze)
register_op_strategy_map(aten.squeeze_.default, torch.squeeze)
register_op_strategy_map(
    aten.squeeze_.dim, torch.squeeze, schema_info=RuntimeSchemaInfo(1)
)
register_op_strategy_map(
    aten.squeeze.dim, torch.squeeze, schema_info=RuntimeSchemaInfo(1)
)
register_op_strategy_map(
    aten.squeeze.dims, torch.squeeze, schema_info=RuntimeSchemaInfo(1)
)
register_op_strategy_map(
    aten.squeeze_.dims, torch.squeeze, schema_info=RuntimeSchemaInfo(1)
)
register_op_strategy_map(
    aten.view.default,
    Tensor.view,
    schema_info=RuntimeSchemaInfo(1),
    strict_view=True,
)
register_op_strategy_map(
    aten.view_copy.default,
    Tensor.view,
    schema_info=RuntimeSchemaInfo(1),
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
    aten.expand_copy.default, Tensor.expand, schema_info=RuntimeSchemaInfo(1)
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


@register_single_dim_strategy(aten.view_as_complex.default)
def view_as_complex_single_dim_strategy(op, args_schema, kwargs_schema):
    # view_as_complex: float [..., 2] -> complex [...]
    # Dims 0..ndim-2 map 1:1; last dim (real/imag pair) is consumed.
    # P(max)/P(min) invalid: complex numbers have no total ordering.
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for d in range(ndim - 1):
        strategies.append([_ShardingPlaceholder(d), _ShardingPlaceholder(d)])
    strategies.append([Partial("sum"), Partial("sum")])
    strategies.append([Partial("avg"), Partial("avg")])
    return strategies


register_op_strategy_map(aten.view_as_real.default, torch.view_as_real)
