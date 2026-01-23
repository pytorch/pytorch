# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import cast, TypeVar

import torch
from torch import sym_min
from torch.distributed import RankType
from torch.distributed._local_tensor import maybe_run_for_local_tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._collective_utils import (
    fill_empty_tensor_to_shards,
    pad_tensor,
    unpad_tensor,
)


_RankTypeT = TypeVar("_RankTypeT", bound=RankType)


class PadType(Enum):
    """
    Type of padding operation based on collective direction and shard type.

    - OLD_*: Padding/unpadding for the source (existing) shard dimension
    - NEW_*: Padding/unpadding for the target (new) shard dimension
    - *_SHARD: Regular Shard placement
    - *_STRIDED: _StridedShard placement with split_factor
    """

    OLD_SHARD = "old_shard"
    NEW_SHARD = "new_shard"
    OLD_STRIDED = "old_strided"
    NEW_STRIDED = "new_strided"


@dataclass
class PaddingOp:
    """A paired padding/unpadding operation specification."""

    pad_type: PadType
    shard_dim: int
    dim_logical_size: int
    split_factor: int = 1  # Only used for strided shard types


# ================== Shard Padding Functions ==================


def _shard_compute_padding_info(
    logical_size_on_dim: int,
    num_chunks: int,
) -> tuple[bool, int]:
    """
    Compute padding information for regular Shard placement.

    Args:
        logical_size_on_dim: The logical size of the tensor on the shard dimension.
        num_chunks: Number of chunks (typically mesh dimension size).

    Returns:
        A tuple of (needs_padding, max_chunk_size).
    """
    dim_padding = logical_size_on_dim % num_chunks != 0
    dim_full_chunk_size = (logical_size_on_dim + num_chunks - 1) // num_chunks
    return dim_padding, dim_full_chunk_size


@maybe_run_for_local_tensor
def _shard_local_size_and_offset(
    curr_local_size: int,
    num_chunks: int,
    rank: _RankTypeT,
) -> tuple[int, _RankTypeT]:
    """
    Given the size of the current local tensor (which may already be sharded on some dimensions),
    computes the new local shard size and offset given the desired number of chunks
    (num_chunks is generally equal to the size of the current sharding dim).

    Note: new local shard offset is relative to the current sharded tensor, not the global tensor.
    See `_utils.compute_local_shape_and_global_offset` for computing global offset.

    Returns (new local shard size, offset)

    """
    # Compute the chunk size inline with ``torch.chunk``
    if curr_local_size % num_chunks == 0:
        full_chunk_size = curr_local_size // num_chunks
        # pyrefly: ignore[bad-assignment] # pyrefly bug?
        shard_starting_idx: _RankTypeT = full_chunk_size * rank
        return full_chunk_size, shard_starting_idx

    # uneven sharding case
    full_chunk_size = (curr_local_size + num_chunks - 1) // num_chunks
    # pyrefly: ignore[bad-assignment] # pyrefly bug?
    shard_starting_idx: _RankTypeT = full_chunk_size * rank

    if curr_local_size < shard_starting_idx:
        return 0, cast(_RankTypeT, curr_local_size)
    else:
        local_shard_size = (
            sym_min(curr_local_size, shard_starting_idx + full_chunk_size)
            - shard_starting_idx
        )
        return local_shard_size, shard_starting_idx


@maybe_run_for_local_tensor
def _pad_for_old_shard_dim(
    local_tensor: torch.Tensor,
    shard_dim: int,
    logical_dim_size: int,
    num_chunks: int,
) -> torch.Tensor:
    """
    Pad the local tensor on the existing (old) shard dimension before a collective operation.

    This ensures all ranks have equal-sized chunks for collective operations like
    all_gather. The padding is applied to the end of the tensor dimension when the
    logical size is not evenly divisible by the number of ranks.

    "Old" refers to the source shard dimension - the dimension that is currently
    sharded before the collective operation.

    Args:
        local_tensor: The local shard tensor to pad.
        shard_dim: The tensor dimension that is sharded.
        logical_dim_size: The global/logical size of the tensor on shard_dim.
        num_chunks: Number of chunks (typically mesh dimension size).

    Returns:
        The padded tensor (contiguous). If no padding is needed, returns
        the original tensor made contiguous.
    """
    dim_padding, dim_max_chunk_size = _shard_compute_padding_info(
        logical_dim_size, num_chunks
    )
    if dim_padding:
        dim_pad_size = dim_max_chunk_size - local_tensor.size(shard_dim)
        if dim_pad_size > 0:
            local_tensor = pad_tensor(local_tensor, shard_dim, dim_pad_size)
    if not local_tensor.is_contiguous():
        local_tensor = local_tensor.contiguous()
    return local_tensor


@maybe_run_for_local_tensor
def _pad_for_new_shard_dim(
    local_tensor: torch.Tensor,
    shard_dim: int,
    logical_dim_size: int,
    num_chunks: int,
) -> torch.Tensor:
    """
    Pad the local tensor on the target (new) shard dimension before an alltoall operation.

    This is used before alltoall when redistributing to a new shard dimension. The
    padding ensures the dimension size equals num_chunks * max_chunk_size so that
    after alltoall, each rank receives equal-sized chunks.

    "New" refers to the target shard dimension - the dimension that will be sharded
    after the collective operation.

    Args:
        local_tensor: The local tensor to pad.
        shard_dim: The target tensor dimension for sharding.
        logical_dim_size: The global/logical size of the tensor on shard_dim.
        num_chunks: Number of chunks (typically mesh dimension size).

    Returns:
        The padded tensor (contiguous). If no padding is needed, returns
        the original tensor made contiguous.
    """
    dim_padding, dim_max_chunk_size = _shard_compute_padding_info(
        logical_dim_size, num_chunks
    )
    if dim_padding:
        dim_pad_size = num_chunks * dim_max_chunk_size - local_tensor.size(shard_dim)
        if dim_pad_size > 0:
            local_tensor = pad_tensor(local_tensor, shard_dim, dim_pad_size)
    if not local_tensor.is_contiguous():
        local_tensor = local_tensor.contiguous()
    return local_tensor


@maybe_run_for_local_tensor
def _unpad_for_old_shard_dim(
    local_tensor: torch.Tensor,
    shard_dim: int,
    logical_dim_size: int,
    num_chunks: int,
) -> torch.Tensor:
    """
    Remove padding from the old shard dimension after a collective operation.

    After an all_gather, the gathered tensor may have extra padding at the end
    that was added to ensure uniform chunk sizes. This method removes that padding
    to restore the logical size.

    "Old" refers to the source shard dimension - the dimension that was sharded
    before the collective operation.

    Args:
        local_tensor: The tensor with potential padding to remove.
        shard_dim: The tensor dimension that was sharded.
        logical_dim_size: The expected global/logical size on shard_dim.
        num_chunks: Number of chunks used in the collective.

    Returns:
        The tensor with padding removed, having size logical_dim_size on shard_dim.
    """
    dim_padding, max_chunk_size = _shard_compute_padding_info(
        logical_dim_size, num_chunks
    )
    if dim_padding:
        dim_unpad_size = max_chunk_size * num_chunks - logical_dim_size
        local_tensor = unpad_tensor(local_tensor, shard_dim, dim_unpad_size)
    return local_tensor


@maybe_run_for_local_tensor
def _unpad_for_new_shard_dim(
    local_tensor: torch.Tensor,
    shard_dim: int,
    logical_dim_size: int,
    num_chunks: int,
    current_rank: int,
) -> torch.Tensor:
    """
    Remove padding from the new shard dimension after a collective operation.

    After collective op, each rank receives a chunk that may have padding at the end.
    This method removes that padding based on the rank's expected local shard size,
    which varies for uneven sharding (last ranks may have smaller or empty shards).

    "New" refers to the target shard dimension - the dimension that is now sharded
    after the collective operation.

    Args:
        local_tensor: The local tensor with potential padding to remove.
        shard_dim: The tensor dimension that is now sharded.
        logical_dim_size: The global/logical size of the tensor on shard_dim.
        num_chunks: Number of chunks used in the collective.
        current_rank: The rank index on the mesh dimension.

    Returns:
        The tensor with padding removed, having the correct local shard size.
    """
    dim_padding, _ = _shard_compute_padding_info(logical_dim_size, num_chunks)
    if dim_padding:
        expected_local_shard_size, _ = _shard_local_size_and_offset(
            logical_dim_size, num_chunks, current_rank
        )
        pad_size = local_tensor.size(shard_dim) - expected_local_shard_size
        local_tensor = unpad_tensor(local_tensor, shard_dim, pad_size)
    return local_tensor


# ================== StridedShard Padding Functions ==================


def _strided_compute_padding_info(
    logical_size_on_dim: int,
    num_chunks: int,
    split_factor: int = 1,
) -> tuple[bool, int]:
    """
    Compute padding information for _StridedShard collective operations.

    This method calculates whether padding is needed and the maximum chunk size
    for collective operations (e.g., all-to-all) involving _StridedShard tensors.

    For _StridedShard with split_factor > 1, the tensor undergoes two-level splitting:
    1. First level: split into ``split_factor`` pieces
    2. Second level: each piece is split into ``num_chunks`` pieces

    The resulting shards are interleaved, so padding must account for both levels.
    Unlike regular Shard where at most one partition (the last) requires padding,
    _StridedShard can have multiple trailing partitions that need padding due to
    the interleaved structure.

    When split_factor=1, this behaves identically to regular Shard padding logic.

    Args:
        logical_size_on_dim: The logical shape size of the tensor on the shard dimension.
        num_chunks: Number of chunks to split into (typically the mesh dim size).
        split_factor: The split_factor in the _StridedShard.

    Returns:
        A tuple of (needs_padding_on_dim, max_chunk_size):
            - needs_padding_on_dim: Whether padding is required on the shard dimension.
            - max_chunk_size: The maximum chunk size per rank after both levels of splitting.
    """

    def _ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b

    if split_factor != 1:
        # Computing padding info for StridedShard tensor dim
        # First level: split into split_factor pieces
        first_chunk_size = _ceil_div(logical_size_on_dim, split_factor)
        num_full_first_chunks = logical_size_on_dim // first_chunk_size
        remainder = logical_size_on_dim - num_full_first_chunks * first_chunk_size

        # Determine if padding is needed:
        # - remainder > 0 means one chunk has partial size
        has_partial_chunk = remainder > 0

        # Second level: each first-level chunk is split into num_chunks pieces
        # Calculate the per-rank chunk size after both levels of splitting
        max_chunk_size = _ceil_div(first_chunk_size, num_chunks) * num_full_first_chunks
        if has_partial_chunk:
            max_chunk_size += _ceil_div(remainder, num_chunks)

        needs_padding_on_dim = (logical_size_on_dim % split_factor) != 0 or (
            (logical_size_on_dim // split_factor) % num_chunks
        ) != 0

    else:
        # Compute padding info for normal shard, no split_factor impact.
        needs_padding_on_dim = logical_size_on_dim % num_chunks != 0
        max_chunk_size = _ceil_div(logical_size_on_dim, num_chunks)

    return needs_padding_on_dim, max_chunk_size


def _strided_split_tensor(
    tensor: torch.Tensor,
    shard_dim: int,
    split_factor: int,
    num_chunks: int,
    *,
    with_padding: bool = True,
    contiguous: bool = True,
) -> tuple[list[torch.Tensor], list[int]]:
    """
    Split a tensor using _StridedShard's interleaved pattern.

    This performs a two-level split:
    1. First split: chunk into split_factor pieces
    2. Second split: chunk each piece into num_chunks pieces
    Then reassemble in transposed left-first order.

    Args:
        tensor: The tensor to split.
        shard_dim: The dimension to split along.
        split_factor: The split factor for _StridedShard.
        num_chunks: Number of chunks for the second split.
        with_padding: Whether to compute padding sizes.
        contiguous: Whether to make shards contiguous.

    Returns:
        A tuple of (shard_list, pad_sizes).
    """
    # First split: chunk into split_factor pieces
    first_split = list(torch.chunk(tensor, split_factor, dim=shard_dim))
    first_split = fill_empty_tensor_to_shards(
        first_split, shard_dim, split_factor - len(first_split)
    )

    # Second split: chunk each piece into num_chunks pieces
    second_split = []
    for s in first_split:
        chunks = list(torch.chunk(s, num_chunks, dim=shard_dim))
        chunks = fill_empty_tensor_to_shards(
            chunks, shard_dim, num_chunks - len(chunks)
        )
        second_split.append(chunks)

    shard_list: list[torch.Tensor] = []
    for i in range(num_chunks):
        shard = torch.cat(
            [second_split[j][i] for j in range(split_factor)],
            dim=shard_dim,
        )
        if contiguous:
            shard = shard.contiguous()
        shard_list.append(shard)

    # The amount of padding is determined by the local chunk with the largest size.
    pad_sizes: list[int] = []
    max_chunk_size = max([shard.size(shard_dim) for shard in shard_list])
    if with_padding:
        pad_sizes = [max_chunk_size - shard.size(shard_dim) for shard in shard_list]

    return shard_list, pad_sizes


@maybe_run_for_local_tensor
def _strided_local_size_and_offset(
    shard_dim,
    split_factor,
    curr_local_size: int,
    num_chunks: int,
    rank: RankType,
    return_first_offset: bool = True,
) -> tuple[int, list[int] | int]:
    """
    Compute the local shard size and offset(s) for a _StridedShard placement.

    Unlike the regular Shard placement which produces contiguous offsets, _StridedShard
    produces non-contiguous (strided) offsets due to the right-to-left sharding semantics.
    This method computes the actual indices that belong to the local shard.

    Args:
        self (_StridedShard): The _StridedShard placement instance.
        curr_local_size (int): The current size of the tensor dimension to be sharded.
        num_chunks (int): Number of chunks to split the dimension into (typically the mesh dimension size).
        rank (int): The rank index to compute the shard for.
        return_first_offset (bool): If True, return only the first offset as an int. If False,
            return all offsets as a list. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - local_shard_size (int): The number of elements in the local shard for this rank.
            - offset (int | list[int]): If return_first_offset is True, returns the first offset
                as an int. If False or if the shard size is 0, returns a list of all offsets
                (which may be empty for empty shards).
    """
    # indices_tensor is 1D torch.arange(logical_dim_size) unsqueezed
    # so that we can reuse self._split_tensor which splits on self.dim
    shape = [1] * shard_dim + [curr_local_size]
    indices_tensor = torch.arange(
        curr_local_size,
    ).view(shape)

    sharded_indices, _ = _strided_split_tensor(
        indices_tensor,
        shard_dim,
        split_factor,
        num_chunks,
        with_padding=False,
        contiguous=False,
    )
    # squeeze back to 1D indices tensor
    sharded_indices = [shard.view(-1) for shard in sharded_indices]

    local_shard_size = len(sharded_indices[rank])
    if local_shard_size > 0:
        offsets = sharded_indices[rank].tolist()
    else:
        offsets = []

    if return_first_offset:
        # Always return an int for consistency across ranks.
        # For empty shards, return -1 as an invalid offset indicator.
        offsets = offsets[0] if len(offsets) > 0 else -1

    return local_shard_size, offsets


@maybe_run_for_local_tensor
def _pad_for_old_strided_shard_dim(
    local_tensor: torch.Tensor,
    shard_dim: int,
    split_factor: int,
    logical_dim_size: int,
    num_chunks: int,
) -> torch.Tensor:
    """
    Pad the local tensor on the existing (old) _StridedShard dimension before a collective.

    This is used before collective operations when redistributing from _StridedShard to
    another placement. Unlike regular Shard padding, this accounts for the split_factor
    which affects chunk sizes due to the interleaved strided sharding pattern.

    "Old" refers to the source _StridedShard dimension - the dimension that is
    currently sharded with _StridedShard before the collective operation.

    Args:
        local_tensor: The local shard tensor to pad.
        shard_dim: The tensor dimension with _StridedShard placement.
        split_factor: The split factor for the _StridedShard placement.
        logical_dim_size: The global/logical size of the tensor on shard_dim.
        num_chunks: Number of chunks (typically mesh dimension size).

    Returns:
        The padded tensor (contiguous). If no padding is needed, returns
        the original tensor made contiguous.
    """
    dim_padding, dim_max_chunk_size = _strided_compute_padding_info(
        logical_dim_size, num_chunks, split_factor
    )
    if dim_padding:
        dim_pad_size = dim_max_chunk_size - local_tensor.size(shard_dim)
        if dim_pad_size > 0:
            local_tensor = pad_tensor(local_tensor, shard_dim, dim_pad_size)
    if not local_tensor.is_contiguous():
        local_tensor = local_tensor.contiguous()
    return local_tensor


@maybe_run_for_local_tensor
def _pad_for_new_strided_shard_dim(
    local_tensor: torch.Tensor,
    shard_dim: int,
    split_factor: int,
    logical_dim_size: int,
    num_chunks: int,
) -> torch.Tensor:
    """
    Pad the local tensor on the target (new) _StridedShard dimension before alltoall.

    This is used before alltoall when redistributing to a _StridedShard placement.
    The method first splits the tensor using _StridedShard's interleaved pattern,
    pads each chunk to uniform size, then concatenates them back.

    "New" refers to the target _StridedShard dimension - the dimension that will
    be sharded with _StridedShard after the collective operation.

    Args:
        local_tensor: The local tensor to pad.
        shard_dim: The target tensor dimension for _StridedShard.
        split_factor: The split factor for the target _StridedShard placement.
        logical_dim_size: The global/logical size of the tensor on shard_dim.
        num_chunks: Number of chunks (typically mesh dimension size).

    Returns:
        The padded tensor with all chunks having uniform size.
    """
    strided_sharded_chunks, strided_sharded_paddings = _strided_split_tensor(
        local_tensor,
        shard_dim,
        split_factor,
        num_chunks,
        with_padding=True,
        contiguous=True,
    )
    for idx, num_paddings in enumerate(strided_sharded_paddings):
        if num_paddings > 0:
            strided_sharded_chunks[idx] = pad_tensor(
                strided_sharded_chunks[idx], shard_dim, num_paddings
            )
    local_tensor = torch.cat(strided_sharded_chunks, dim=shard_dim)
    return local_tensor


@maybe_run_for_local_tensor
def _unpad_for_old_strided_shard_dim(
    local_tensor: torch.Tensor,
    old_strided_shard_dim: int,
    split_factor: int,
    logical_dim_size: int,
    num_chunks: int,
) -> torch.Tensor:
    """
    Remove padding and reorder elements after transforming from _StridedShard.

    After a collective operation that transforms from _StridedShard to another
    placement, the resulting tensor has elements in a strided order with potential
    padding. This method uses index_select to both:
    1. Extract only the non-padding elements
    2. Reorder them back to the correct logical order

    The approach builds an index mapping by replaying the _StridedShard split
    operation to understand which positions in the padded tensor correspond to
    which logical indices.

    Args:
        local_tensor: The local tensor after collective with strided ordering and padding.
        old_strided_shard_dim: The tensor dimension that was sharded with _StridedShard.
        split_factor: The split factor of the source _StridedShard placement.
        logical_dim_size: The global/logical size of the tensor on shard_dim.
        num_chunks: Number of chunks used in the collective.

    Returns:
        The tensor with elements reordered to logical order and padding removed.
    """
    # Build sharded indices to understand the strided pattern
    shape = [1] * old_strided_shard_dim + [logical_dim_size]
    indices_tensor = torch.arange(logical_dim_size, device=local_tensor.device).view(
        shape
    )
    sharded_indices, _ = _strided_split_tensor(
        indices_tensor,
        old_strided_shard_dim,
        split_factor,
        num_chunks,
        with_padding=False,
        contiguous=False,
    )
    sharded_indices = [shard.view(-1) for shard in sharded_indices]

    max_chunk_size = len(sharded_indices[0])

    # After all_gather, the tensor is [chunk0, chunk1 (may padded), ...].
    # Each chunk may have padding at the end. Use a single index_select to
    # both extract non-padding data and reorder to the original positions.
    #
    # Build select_indices where select_indices[original_pos] = position in
    # the padded tensor that holds the element for original_pos.
    padded_positions = []

    # Compute positions in the padded local_tensor for each actual element.
    # Each shard i starts at position i * max_chunk_size in local_tensor.
    for i, shard in enumerate(sharded_indices):
        base_offset = i * max_chunk_size
        positions = base_offset + torch.arange(len(shard), device=local_tensor.device)
        padded_positions.append(positions)

    permutation = torch.cat(sharded_indices)
    select_positions = torch.cat(padded_positions)

    inv_permutation = torch.argsort(permutation)
    select_indices = select_positions.index_select(0, inv_permutation)
    local_tensor = torch.index_select(
        local_tensor, old_strided_shard_dim, select_indices
    )
    return local_tensor


@maybe_run_for_local_tensor
def _unpad_for_new_strided_shard_dim(
    local_tensor: torch.Tensor,
    shard_dim: int,
    split_factor: int,
    logical_dim_size: int,
    num_chunks: int,
    current_rank: int,
) -> torch.Tensor:
    """
    Remove padding from the new _StridedShard dimension after alltoall.

    After alltoall when the target is a _StridedShard placement, the local tensor
    may have padding that was added to ensure uniform chunk sizes. This method
    removes that padding based on the rank's expected local shard size, which
    accounts for the strided interleaving pattern.

    "New" refers to the target _StridedShard dimension - the dimension that is
    now sharded with _StridedShard after the collective operation.

    Args:
        local_tensor: The local tensor with potential padding to remove.
        shard_dim: The tensor dimension with _StridedShard placement.
        split_factor: The split factor for the _StridedShard placement.
        logical_dim_size: The global/logical size of the tensor on shard_dim.
        num_chunks: Number of chunks used in the collective.
        current_rank: The rank index on the mesh dimension.

    Returns:
        The tensor with padding removed, having the correct local shard size.
    """
    dim_padding, dim_max_chunk_size = _strided_compute_padding_info(
        logical_dim_size, num_chunks, split_factor
    )
    if dim_padding:
        local_size, _ = _strided_local_size_and_offset(
            shard_dim, split_factor, logical_dim_size, num_chunks, current_rank
        )
        padding_size = dim_max_chunk_size - local_size
        local_tensor = unpad_tensor(local_tensor, shard_dim, padding_size)
    return local_tensor


# ================== CollectivePaddingContext ==================


@dataclass
class CollectivePaddingContext:
    """
    Context for managing padding/unpadding around collective operations.

    Padding operations are paired with their corresponding unpadding automatically.
    Unpadding is applied in reverse order (LIFO - last padded, first unpadded).

    This context simplifies the pad → collective → unpad pattern by:
    1. Automatically pairing each pad with its corresponding unpad
    2. Managing the order of operations (LIFO for unpadding)
    3. Providing a fluent API for easy composition

    Example usage::

        # Shard → Shard alltoall (pad old, unpad old+new)
        result = (
            CollectivePaddingContext(mesh, mesh_dim)
            .pad_old_shard(self.dim, current_logical_shape[self.dim])
            .pad_new_shard(new_shard_dim, current_logical_shape[new_shard_dim])
            .run(
                local_tensor,
                lambda t: shard_dim_alltoall(
                    t, self.dim, new_shard_dim, mesh, mesh_dim
                ),
            )
        )

        # Shard → Replicate all_gather (pad old, unpad old)
        result = (
            CollectivePaddingContext(mesh, mesh_dim)
            .pad_old_shard(self.dim, current_logical_shape[self.dim])
            .run(
                local_tensor,
                lambda t: funcol.all_gather_tensor(
                    t, gather_dim=self.dim, group=(mesh, mesh_dim)
                ),
            )
        )
    """

    mesh: DeviceMesh
    mesh_dim: int
    _ops: list[PaddingOp] = field(default_factory=list)

    def __post_init__(self):
        self.num_chunks = self.mesh.size(mesh_dim=self.mesh_dim)
        coord = self.mesh.get_coordinate()
        self.current_rank = coord[self.mesh_dim] if coord else 0

    def pad_old_shard(
        self, shard_dim: int, dim_logical_size: int
    ) -> "CollectivePaddingContext":
        """
        Pad the source Shard dimension before a collective.

        Use this for the dimension that is currently sharded (source placement).
        Example: In Shard(0) → Replicate, dim 0 is the "old" shard dimension.
        """
        self._ops.append(PaddingOp(PadType.OLD_SHARD, shard_dim, dim_logical_size))
        return self

    def pad_new_shard(
        self, shard_dim: int, dim_logical_size: int
    ) -> "CollectivePaddingContext":
        """
        Pad the target Shard dimension before a collective.

        Use this for the dimension that will be sharded after the collective (target placement).
        Example: In Shard(0) → Shard(1), dim 1 is the "new" shard dimension.
        """
        self._ops.append(PaddingOp(PadType.NEW_SHARD, shard_dim, dim_logical_size))
        return self

    def pad_old_strided(
        self, shard_dim: int, dim_logical_size: int, split_factor: int
    ) -> "CollectivePaddingContext":
        """
        Pad the source _StridedShard dimension before a collective.

        Use this for the dimension that is currently sharded with _StridedShard.
        Example: In _StridedShard(0, sf=2) → Shard(1), dim 0 is the "old" strided shard dimension.
        """
        self._ops.append(
            PaddingOp(PadType.OLD_STRIDED, shard_dim, dim_logical_size, split_factor)
        )
        return self

    def pad_new_strided(
        self, shard_dim: int, dim_logical_size: int, split_factor: int
    ) -> "CollectivePaddingContext":
        """
        Pad the target _StridedShard dimension before a collective.

        Use this for the dimension that will be sharded with _StridedShard after the collective.
        Example: In Shard(0) → _StridedShard(1, sf=2), dim 1 is the "new" strided shard dimension.
        """
        self._ops.append(
            PaddingOp(PadType.NEW_STRIDED, shard_dim, dim_logical_size, split_factor)
        )
        return self

    def _apply_pad(self, tensor: torch.Tensor, op: PaddingOp) -> torch.Tensor:
        """Apply a single padding operation."""
        if op.pad_type == PadType.OLD_SHARD:
            return _pad_for_old_shard_dim(
                tensor, op.shard_dim, op.dim_logical_size, self.num_chunks
            )
        elif op.pad_type == PadType.NEW_SHARD:
            return _pad_for_new_shard_dim(
                tensor, op.shard_dim, op.dim_logical_size, self.num_chunks
            )
        elif op.pad_type == PadType.OLD_STRIDED:
            return _pad_for_old_strided_shard_dim(
                tensor,
                op.shard_dim,
                op.split_factor,
                op.dim_logical_size,
                self.num_chunks,
            )
        elif op.pad_type == PadType.NEW_STRIDED:
            return _pad_for_new_strided_shard_dim(
                tensor,
                op.shard_dim,
                op.split_factor,
                op.dim_logical_size,
                self.num_chunks,
            )
        raise ValueError(f"Unknown pad type: {op.pad_type}")

    def _apply_unpad(self, tensor: torch.Tensor, op: PaddingOp) -> torch.Tensor:
        """Apply a single unpadding operation (paired with the corresponding pad)."""
        if op.pad_type == PadType.OLD_SHARD:
            return _unpad_for_old_shard_dim(
                tensor, op.shard_dim, op.dim_logical_size, self.num_chunks
            )
        elif op.pad_type == PadType.NEW_SHARD:
            return _unpad_for_new_shard_dim(
                tensor,
                op.shard_dim,
                op.dim_logical_size,
                self.num_chunks,
                self.current_rank,
            )
        elif op.pad_type == PadType.OLD_STRIDED:
            return _unpad_for_old_strided_shard_dim(
                tensor,
                op.shard_dim,
                op.split_factor,
                op.dim_logical_size,
                self.num_chunks,
            )
        elif op.pad_type == PadType.NEW_STRIDED:
            return _unpad_for_new_strided_shard_dim(
                tensor,
                op.shard_dim,
                op.split_factor,
                op.dim_logical_size,
                self.num_chunks,
                self.current_rank,
            )
        raise ValueError(f"Unknown pad type: {op.pad_type}")

    def run(
        self,
        local_tensor: torch.Tensor,
        collective_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Execute collective with automatic padding before and unpadding after.

        Args:
            local_tensor: The local tensor to process.
            collective_fn: The collective operation to run (e.g., alltoall, all_gather).

        Returns:
            The result tensor after the collective and unpadding.

        Note:
            Padding is applied in the order operations were added.
            Unpadding is applied in reverse order (LIFO - last padded, first unpadded).
        """
        # Apply all padding operations in order
        for op in self._ops:
            local_tensor = self._apply_pad(local_tensor, op)

        # Run collective
        result = collective_fn(local_tensor)

        # Apply all unpadding operations in reverse order
        for op in reversed(self._ops):
            result = self._apply_unpad(result, op)

        return result
