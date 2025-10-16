"""Utility for reconstructing a sequence of view operations that is equivalent
to an as_strided call (change a source tensor's size/stride/offset to a
target size/stride/offset).

In some situations, it can be helpful to be able to construct a sequence of
"normal" view operations (e.g., view and permute) which gets you
from one size/stride/offset to another size/stride/offset, instead of brute
forcing it with as_strided.  The reason is that as_strided is extremely flexible,
making it difficult for backends to implement: for example, using it you can
generate views with strange overlap patterns (e.g., rolling windows) or
generate an output view which is out of bounds for the original view.  If you
are implementing a tensor subclass, it may be possible to implement simple
views but not as_strided (e.g., DTensor sharding propagation).  These
utilities help you reconstruct view operations in some situations where it is
possible.

The algorithm we implement is almost complete assuming the as_strided call was
derived from these view operations:

- view; or equivalently, these four operations:
  - squeeze
  - unsqueeze
  - unflatten
  - flatten (output view only)
- permute (this subsumes transpose, movedim, etc.)

Furthermore, we assume that the source tensor is a non-overlapping view, i.e.,
that for every physical location there is a unique coordinate that maps to it.
(This implies the destination tensor is non-overlapping, as we did not include
broadcast in our set of valid view operations.)

Narrow is another useful view operation we might consider supporting, but
narrow is actually quite complicated so we leave it to future work.

# Definition of view operations

To start, it's helpful to review how these view operations affect the size/stride
of an operator:

- squeeze(dim)

      new_size =   size[:dim] +   size[dim+1:]
    new_stride = stride[:dim] + stride[dim+1:]

        if size[dim] == 1

- unsqueeze(dim):

      new_size =   size[:dim] +                       [1] + size[dim:]
    new_stride = stride[:dim] + [size[dim] * stride[dim]] + stride[dim:]

        NB: new_stride[dim] = 1 if dim == len(size)

- unflatten(dim, unflat_sizes):

      new_size =   size[:dim] +                unflat_sizes               +   size[dim+1:]
    new_stride = stride[:dim] + contig_strides(unflat_sizes, stride[dim]) + stride[dim+1:]

        if product(unflat_sizes) == size[dim]

- flatten(start_dim, end_dim-1)  # here, end_dim is the first NON-flattened dim

      new_size =   size[:start_dim] + product(size[start_dim:end_dim]) +   size[end_dim:]
    new_stride = stride[:start_dim] +                stride[end_dim-1] + stride[end_dim:]

        if contig_strides(size[start_dim:end_dim], stride[end_dim]) == stride[start_dim:end_dim]

- permute(dims)

      new_size[i] =   size[dims[i]]
    new_stride[i] = stride[dims[i]]

Where contig_strides(sizes, stride) is simply the contiguous strides for sizes
with innermost stride 'stride' (this coincides with contiguous strides when
stride = 1).

# Examples to build intuition

There are a few things to notice looking over these view operations.  First,
there are only a few ways for "new" strides to show up when you do operations:
unsqueeze and unflatten.  In particular, we can only generate strides which
are *contiguous* when we unflatten, which means no matter what view operations
we do, the result is always non-overlapping.

Unsqueeze is a funny duck: depending on the permutation of the tensor at the
time unsqueeze is run, we can generate 1-size dims with different strides:

    >>> a = torch.empty_strided((2,2),(50,30)).unsqueeze(0)
    >>> b = torch.empty_strided((2,2),(50,30)).mT.unsqueeze(0).mT
    >>> a.size(), a.stride()
    (torch.Size([1, 2, 2]), (100, 50, 30))
    >>> b.size(), b.stride()
    (torch.Size([1, 2, 2]), (60, 50, 30))

By the way, the stride on a 1-size dim doesn't affect the contents of a
tensor, but it can influence decisions about memory layout, so it is better to
preserve it.  But we can generate nearly arbitrary strides for a size one dimension
by narrowing, unsqueezing, and then further narrowing that dimension.

Conversely, any stride can be deleted from the original tensor: we simply
narrow a dim so it is size 1, and then squeeze it away.

# Proof

The proof strategy:

1. We first need to show that, if we are limited to the above view operations,
   the target size/stride/offset must have certain STRUCTURE.

2. Given this STRUCTURE, we have enough information to generate a de novo
   set of view operations that achieves the target size/stride/offset.

Let us first talk about the STRUCTURE.  First, remember that although we
assume that the input tensor is non-overlapping, we don't assume that it is
contiguous; its dimensions will form several contiguous subspaces (on which
flattens and unflattens are possible).  Ignoring size 1 dimensions, we can
only ever generate new stride values by flattening or unflattening.  By the
fundamental theorem of arithmetic, 

# Some notes about narrow

- narrow(dim, start, length)

      new_size =   size[:dim] +      [length] +   size[dim+1:]
    new_stride = stride[:dim] + [stride[dim]] + stride[dim+1:]
    new_offset = offset + stride[dim] * start

Finally, we see narrow is the only view operation that affects offset.
Because of our assumption that the input is non-overlapping, a delta in offset
can always be uniquely factored into a sum over strides multiplied by start
indices.  However, this set of strides to factor by may not necessarily be
exactly the source or the target tensor:

    >>> a = torch.empty_strided((8,),(1,)).view(2,2,2).select(1,1)
    >>> a.size(), a.stride(), a.storage_offset()
    (torch.Size([2, 2]), (4, 1), 2)

Here, the correct stride to factor is 2, but it occurs neither in the source
(1,) or the target (4,1); the dim with stride 2 is generated when we unflatten
the (8,) tensor into a (2,2,2) tensor.

However, it's important to observe that narrows are irreversible: unlike the
other operations which have inverses, once you narrow a tensor, you cannot
un-narrow it.  Furthermore, when you narrow a dimension, it is no longer
contiguous with its adjacent outer dimension: this means you can no longer
flatten with the outer dimension, permanently reducing the space of possible
operations you can do.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypedDict

import torch


class _AssignedDim(TypedDict):
    """Metadata for an assigned dimension."""
    index: int
    size: int
    stride: int


class _DimInfo(TypedDict):
    """Information about a dimension after splitting."""
    split_size: int
    target_size: int
    start: int
    keep: bool
    output_index: int | None

__all__ = [
    "ViewOp",
    "ViewRecomputeError",
    "recompute_view_ops",
    "apply_view_ops",
    "recompute_view",
]


@dataclass(frozen=True)
class ViewOp:
    """Represents a single view operation.

    Attributes:
        kind: The type of operation ('view', 'permute', or 'narrow')
        args: The arguments for the operation
    """
    kind: str
    args: tuple[int, ...]


class ViewRecomputeError(RuntimeError):
    """Raised when view recomputation fails."""
    pass


def _canonicalize(tensor: torch.Tensor) -> tuple[torch.Tensor, list[ViewOp]]:
    """Canonicalize tensor metadata for view recomputation.

    Performs the following transformations:
    1. Squeeze away every size-1 dimension
    2. Permute axes so strides are in strictly descending order
    3. Coalesce adjacent contiguous dimensions
    4. Prepend an extra leading dimension of size 1

    Args:
        tensor: Input tensor to canonicalize

    Returns:
        Tuple of (canonicalized tensor, list of operations applied)
    """
    ops: list[ViewOp] = []
    # Step 1 (Spec): Canonicalize the input tensor.
    #   * Squeeze away every size-1 dimension.
    #   * Permute axes so the strides are in strictly descending order (unique by non-overlap).
    #   * Coalesce adjacent contiguous dimensions where size[j] * stride[j] == stride[i].
    #   * Prepend an extra leading dimension of size 1 to simplify later boundary cases.
    trimmed_sizes: list[int] = [int(s) for s in tensor.shape if s != 1]
    target_shape: tuple[int, ...] = (1, *trimmed_sizes) if trimmed_sizes else (1,)
    if tuple(int(s) for s in tensor.shape) != target_shape:
        tensor = tensor.view(*target_shape)
        ops.append(ViewOp("view", tuple(map(int, target_shape))))
    order: list[int] = sorted(range(tensor.dim()), key=lambda i: tensor.stride()[i], reverse=True)
    if tuple(order) != tuple(range(tensor.dim())):
        tensor = tensor.permute(*order)
        ops.append(ViewOp("permute", tuple(order)))
    sizes: list[int] = list(tensor.shape)
    strides: list[int] = list(tensor.stride())
    if sizes:
        new_sizes: list[int] = [sizes[0]]
        new_strides: list[int] = [strides[0]]
        for i in range(1, len(sizes)):
            if len(new_sizes) > 1 and new_strides[-1] == sizes[i] * strides[i]:
                new_sizes[-1] *= sizes[i]
                new_strides[-1] = strides[i]
            else:
                new_sizes.append(sizes[i])
                new_strides.append(strides[i])
        if new_sizes != sizes:
            tensor = tensor.view(*new_sizes)
            ops.append(ViewOp("view", tuple(map(int, new_sizes))))
    return tensor, ops


def _compute_coords(strides: Sequence[int], delta: int) -> list[int]:
    """Compute coordinates from stride information and offset delta.

    Finds coord such that output.storage_offset == input.storage_offset +
    sum(coord[i] * stride[i]) using the canonical strides.

    Args:
        strides: Stride values for each dimension
        delta: Offset delta to decompose

    Returns:
        List of coordinates

    Raises:
        ViewRecomputeError: If offset decomposition fails
    """
    coords: list[int] = []
    remaining: int = delta
    for stride in strides:
        if stride == 0:
            coords.append(0)
            continue
        coord: int = remaining // stride
        coords.append(int(coord))
        remaining -= coord * stride
    if remaining != 0:
        raise ViewRecomputeError("storage offset decomposition failed")
    return coords


def _assign_dims(
    canon_strides: Sequence[int], output_meta: Sequence[tuple[int, int, int]]
) -> list[list[_AssignedDim]]:
    """Assign each output stride to the canonical dimension whose stride range contains it.

    Groups output dimensions by which canonical dimension's stride range they fall into:
    (stride[i+1] <= output stride < stride[i]).

    Args:
        canon_strides: Canonical stride values
        output_meta: List of (index, size, stride) tuples for output tensor

    Returns:
        List of lists, where assigned[i] contains all output dims assigned to canonical dim i

    Raises:
        ViewRecomputeError: If a stride cannot be assigned to any dimension
    """
    assigned: list[list[_AssignedDim]] = [[] for _ in canon_strides]
    sorted_meta = sorted(output_meta, key=lambda item: (-item[2], -item[1], item[0]))
    for index, size, stride in sorted_meta:
        chosen: int | None = None
        for dim, canon_stride in enumerate(canon_strides):
            if stride % canon_stride == 0 and (dim == 0 or stride < canon_strides[dim - 1]):
                chosen = dim
                break
        if chosen is None:
            raise ViewRecomputeError(f"cannot assign stride {stride}")
        assigned[chosen].append(_AssignedDim(index=index, size=size, stride=stride))
    return assigned


def _split_sizes(
    canon_size: int, canon_stride: int, assigned: list[_AssignedDim]
) -> tuple[list[int], list[_AssignedDim], list[_AssignedDim]]:
    """Compute split sizes for a canonical dimension.

    Applies all splits for a canonical dimension in one shot instead of per-target,
    imputing contiguous sizes that realize the requested output strides.

    Args:
        canon_size: Size of the canonical dimension
        canon_stride: Stride of the canonical dimension
        assigned: List of assigned output dimensions

    Returns:
        Tuple of (split sizes, filtered assigned dims, removed size-1 entries)

    Raises:
        ViewRecomputeError: If splitting fails
    """
    if not assigned:
        return [canon_size], assigned, []
    working: list[_AssignedDim] = list(assigned)
    removed: list[_AssignedDim] = []
    while True:
        m: int = len(working)
        if m == 0:
            return [canon_size], working, removed
        target_strides: list[int] = [int(item["stride"]) for item in working]
        splits: list[int] = [0] * m
        if m == 1:
            # Only one target stride in range: the entire canonical dim maps to it.
            if target_strides[0] % canon_stride != 0:
                raise ViewRecomputeError("stride mismatch during split")
            splits[0] = canon_size
            return splits, working, removed
        for k in range(m - 1):
            num: int = target_strides[k]
            den: int = target_strides[k + 1]
            if num % den != 0:
                raise ViewRecomputeError("non integer split")
            splits[k + 1] = num // den
        prod_tail: int = 1
        for value in splits[1:]:
            prod_tail *= value
        if prod_tail != 0 and canon_size % prod_tail == 0:
            splits[0] = canon_size // prod_tail
            if target_strides[-1] % canon_stride != 0:
                raise ViewRecomputeError("stride mismatch during split")
            return splits, working, removed
        # if we reach here, split failed; try removing size-1 duplicate stride
        removed_idx: int | None = None
        for idx in range(len(working) - 1):
            if working[idx]["size"] == 1 and working[idx]["stride"] == working[idx + 1]["stride"]:
                removed_idx = idx
                break
        if removed_idx is None and working and working[-1]["size"] == 1:
            removed_idx = len(working) - 1
        if removed_idx is None:
            raise ViewRecomputeError("canonical size mismatch during split")
        removed.append(working.pop(removed_idx))


def _expand_coord(coord: int, splits: Sequence[int]) -> list[int]:
    """Expand a coordinate into split dimensions.

    Args:
        coord: Coordinate to expand
        splits: Split sizes for each dimension

    Returns:
        List of coordinate values for each split dimension
    """
    suffix_products: list[int] = [1] * len(splits)
    acc: int = 1
    for i in range(len(splits) - 1, -1, -1):
        suffix_products[i] = acc
        acc *= splits[i]
    digits: list[int] = []
    remaining: int = coord
    for size, suffix in zip(splits, suffix_products):
        idx: int = 0 if suffix == 0 else remaining // suffix
        digits.append(int(idx))
        remaining -= idx * suffix
    return digits


def recompute_view_ops(
    input_tensor: torch.Tensor, output_tensor: torch.Tensor
) -> list[ViewOp]:
    """Compute a sequence of view operations to transform input_tensor to match output_tensor.

    Given two tensors that share the same storage, computes a sequence of view operations
    (view, permute, narrow) that will transform the input tensor to have the same
    size, stride, and storage offset as the output tensor.

    Args:
        input_tensor: The starting tensor
        output_tensor: The target tensor (must share storage with input_tensor)

    Returns:
        List of ViewOp operations that can transform input_tensor to output_tensor

    Raises:
        ViewRecomputeError: If tensors don't share storage or transformation is not possible

    Example:
        >>> base = torch.arange(12).view(3, 4)
        >>> out = base.permute(1, 0)
        >>> ops = recompute_view_ops(base, out)
        >>> rebuilt = apply_view_ops(base, ops)
        >>> assert rebuilt.shape == out.shape
        >>> assert rebuilt.stride() == out.stride()
    """
    if input_tensor.untyped_storage().data_ptr() != output_tensor.untyped_storage().data_ptr():
        raise ViewRecomputeError("input and output tensors do not share storage")
    current: torch.Tensor = input_tensor
    ops: list[ViewOp] = []
    canonical_tensor, canonical_ops = _canonicalize(current)
    for op in canonical_ops:
        if op.kind == "view":
            current = current.view(*op.args)
        elif op.kind == "permute":
            current = current.permute(*op.args)
        ops.append(op)
    canon_sizes: list[int] = list(map(int, current.shape))
    canon_strides: list[int] = list(map(int, current.stride()))
    out_strides: list[int] = [int(st) for st in output_tensor.stride()]
    if canon_sizes and canon_sizes[0] == 1 and canon_strides[0] not in out_strides:
        # Step 4 (Spec): Handle the extra leading dimension. If the output never references its stride,
        #   squeeze it away; otherwise keep it (the output size there must be 1 to remain a subset).
        if current.dim() == 1:
            raise ViewRecomputeError("cannot drop sole canonical dimension")
        current = current.view(*current.shape[1:])
        ops.append(ViewOp("view", tuple(map(int, current.shape))))
        canon_sizes = list(map(int, current.shape))
        canon_strides = list(map(int, current.stride()))
    out_meta_all: list[tuple[int, int, int]] = [
        (i, int(sz), int(st))
        for i, (sz, st) in enumerate(zip(output_tensor.shape, output_tensor.stride()))
    ]
    delta: int = output_tensor.storage_offset() - current.storage_offset()
    coords: list[int] = _compute_coords(canon_strides, delta)
    assigned: list[list[_AssignedDim]] = _assign_dims(canon_strides, out_meta_all)
    # Step 3 (Spec): For each canonical dimension, split it to cover all target strides in the
    #   range stride[i+1] <= target_stride < stride[i], and use the Step 2 coordinates to narrow.
    #   The split infers sizes that would make each target stride contiguous; removed size-1
    #   duplicates are deferred and reintroduced later as unit dimensions.
    deferred_unit_indices: set[int] = set()
    dim_infos: list[_DimInfo] = []
    for dim_index, (canon_size, canon_stride) in enumerate(zip(canon_sizes, canon_strides)):
        splits: list[int]
        filtered_assigned: list[_AssignedDim]
        removed_entries: list[_AssignedDim]
        splits, filtered_assigned, removed_entries = _split_sizes(
            canon_size, canon_stride, assigned[dim_index]
        )
        assigned[dim_index] = filtered_assigned
        for removed_entry in removed_entries:
            deferred_unit_indices.add(removed_entry["index"])
        expanded: list[int] = _expand_coord(coords[dim_index], splits)
        if not filtered_assigned:
            start: int = expanded[0] if expanded else 0
            dim_infos.append(
                _DimInfo(
                    split_size=splits[0],
                    target_size=1,
                    start=start,
                    keep=False,
                    output_index=None,
                )
            )
        else:
            if len(expanded) != len(filtered_assigned):
                raise ViewRecomputeError("coordinate rank mismatch")
            for meta, split_size, start_val in zip(filtered_assigned, splits, expanded):
                dim_infos.append(
                    _DimInfo(
                        split_size=split_size,
                        target_size=meta["size"],
                        start=start_val,
                        keep=True,
                        output_index=meta["index"],
                    )
                )
    split_shape: tuple[int, ...] = tuple(int(info["split_size"]) for info in dim_infos)
    if split_shape != current.shape:
        current = current.view(*split_shape)
        ops.append(ViewOp("view", tuple(map(int, split_shape))))
    for dim, info in enumerate(dim_infos):
        start_pos: int = int(info["start"])
        length: int = int(info["target_size"])
        if length > info["split_size"]:
            raise ViewRecomputeError("narrow length exceeds dimension")
        if start_pos != 0 or length != info["split_size"]:
            current = current.narrow(dim, start_pos, length)
            ops.append(ViewOp("narrow", (dim, start_pos, length)))
            info["split_size"] = length
    core_indices: list[int] = [idx for idx in range(len(out_meta_all)) if idx not in deferred_unit_indices]
    if not core_indices:
        core_indices = [0]
        deferred_unit_indices.discard(0)
    core_meta: list[tuple[int, int, int]] = [out_meta_all[idx] for idx in core_indices]
    unit_indices_list: list[int] = sorted(deferred_unit_indices)

    keep_indices: list[int] = [idx for idx, info in enumerate(dim_infos) if info["keep"]]
    if len(keep_indices) != len(core_meta):
        raise ViewRecomputeError("dimension count mismatch")

    keep_shape: list[int] = [dim_infos[idx]["split_size"] for idx in keep_indices]
    if len(keep_shape) < len(dim_infos):
        for idx, info in enumerate(dim_infos):
            if idx not in keep_indices and info["split_size"] != 1:
                raise ViewRecomputeError("attempting to drop non unit dimension")
        if keep_shape:
            current = current.view(*map(int, keep_shape))
            ops.append(ViewOp("view", tuple(map(int, keep_shape))))
        else:
            current = current.view(1)
            ops.append(ViewOp("view", (1,)))
        dim_infos = [dim_infos[idx] for idx in keep_indices]
    else:
        dim_infos = [dim_infos[idx] for idx in keep_indices]

    position_map: dict[int, int] = {}
    for out_index in range(len(core_meta)):
        for idx, info in enumerate(dim_infos):
            if info["output_index"] == core_meta[out_index][0]:
                position_map[out_index] = idx
                break
        else:
            raise ViewRecomputeError("missing output dimension mapping")

    perm: list[int] = [position_map[i] for i in range(len(core_meta))]
    # Step 5 (Spec): Restore the target ordering by permuting back to the output's dimension order.
    if perm != list(range(len(core_meta))):
        current = current.permute(*perm)
        ops.append(ViewOp("permute", tuple(perm)))
        dim_infos = [dim_infos[i] for i in perm]

    expected_core_shape: tuple[int, ...] = tuple(meta[1] for meta in core_meta)
    if current.shape != expected_core_shape:
        current = current.view(*expected_core_shape)
        ops.append(ViewOp("view", tuple(map(int, expected_core_shape))))

    # Reintroduce any deferred size-1 dimensions so the final tensor covers the same subset as the output.
    combined_indices: list[int] = list(core_indices)
    extended_shape: tuple[int, ...]
    if unit_indices_list:
        extended_shape = tuple(list(expected_core_shape) + [1] * len(unit_indices_list))
        if current.shape != extended_shape:
            current = current.view(*extended_shape)
            ops.append(ViewOp("view", tuple(map(int, extended_shape))))
        combined_indices.extend(unit_indices_list)
    else:
        extended_shape = expected_core_shape

    if combined_indices:
        perm_order: list[int] = sorted(range(len(combined_indices)), key=lambda pos: combined_indices[pos])
        if perm_order != list(range(len(combined_indices))):
            current = current.permute(*perm_order)
            ops.append(ViewOp("permute", tuple(perm_order)))

    if (
        current.shape != output_tensor.shape
        or current.stride() != output_tensor.stride()
        or current.storage_offset() != output_tensor.storage_offset()
    ):
        raise ViewRecomputeError("result does not match target metadata")
    return ops


def apply_view_ops(tensor: torch.Tensor, ops: Sequence[ViewOp]) -> torch.Tensor:
    """Apply a sequence of view operations to a tensor.

    Args:
        tensor: The tensor to apply operations to
        ops: Sequence of ViewOp operations to apply

    Returns:
        The resulting tensor after applying all operations

    Raises:
        ViewRecomputeError: If an operation is invalid

    Example:
        >>> base = torch.arange(12).view(3, 4)
        >>> ops = [ViewOp("permute", (1, 0))]
        >>> result = apply_view_ops(base, ops)
        >>> assert result.shape == (4, 3)
    """
    result: torch.Tensor = tensor
    for op in ops:
        if op.kind == "view":
            result = result.view(*op.args)
        elif op.kind == "permute":
            result = result.permute(*op.args)
        elif op.kind == "narrow":
            if len(op.args) != 3:
                raise ViewRecomputeError("invalid narrow args")
            dim: int
            start_narrow: int
            length_narrow: int
            dim, start_narrow, length_narrow = op.args
            result = result.narrow(dim, start_narrow, length_narrow)
        else:
            raise ViewRecomputeError(f"unknown op kind {op.kind}")
    return result


def recompute_view(
    input_tensor: torch.Tensor, output_tensor: torch.Tensor
) -> torch.Tensor:
    """Convenience function to recompute and apply view operations in one step.

    This is equivalent to calling recompute_view_ops followed by apply_view_ops.

    Args:
        input_tensor: The starting tensor
        output_tensor: The target tensor (must share storage with input_tensor)

    Returns:
        A new tensor with the same metadata as output_tensor, derived from input_tensor

    Raises:
        ViewRecomputeError: If tensors don't share storage or transformation is not possible

    Example:
        >>> base = torch.arange(30).view(5, 3, 2)
        >>> out = base.view(3, 5, 2).permute(1, 2, 0)
        >>> rebuilt = recompute_view(base, out)
        >>> assert torch.equal(rebuilt, out)
    """
    ops: list[ViewOp] = recompute_view_ops(input_tensor, output_tensor)
    return apply_view_ops(input_tensor, ops)
