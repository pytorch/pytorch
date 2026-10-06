from __future__ import annotations

import math
from typing import Protocol, runtime_checkable, TypeGuard

import torch

from .metadata import ChunkStorageMetadata, MetadataIndex


__all__ = ["CheckpointableTensor"]


@runtime_checkable
class CheckpointableTensor(Protocol):
    """Protocol fields for checkpointing a local tensor as global tensor shards.

    A tensor does not need to be wrapped or subclassed for DCP to checkpoint it
    as a shard. It can stay a regular local ``torch.Tensor``; implementing
    these fields is enough for DCP to map one or more slices of that tensor
    into a logical global tensor.

    The local tensor may have fewer dims than ``global_shape`` when it stores
    rows of a view that merges leading dims, e.g. ``[rows, *shape[k:]]`` for
    merged ``shape[:k]``. Each shard then spans the local tensor's dims after
    the first, and is the run of local rows starting at its local offset,
    viewed as its local size.

    Attributes:
        global_shape: Full logical tensor shape to write into checkpoint
            metadata; needed because ``tensor.size()`` is only the local buffer
            size.
        global_offsets: Global start coordinate for each local shard; needed to
            name checkpoint chunks and match load requests by global offset.
        local_offsets: Start coordinate for each local shard inside the local
            tensor; needed when one tensor stores multiple shards or includes
            padding. With merged leading dims, only the row may be nonzero.
        local_sizes: Shape of each local shard; needed to build checkpoint
            chunks and slice the local tensor during load. With merged leading
            dims, it has ``global_shape``'s dims, and its trailing dims match
            the local tensor's dims after the first.
    """

    global_shape: tuple[int, ...]
    global_offsets: tuple[tuple[int, ...], ...]
    local_offsets: tuple[tuple[int, ...], ...]
    local_sizes: tuple[tuple[int, ...], ...]


def _is_checkpointable_tensor(obj: object) -> TypeGuard[CheckpointableTensor]:
    return isinstance(obj, torch.Tensor) and isinstance(obj, CheckpointableTensor)


def _copy_checkpointable_tensor_metadata(
    src: CheckpointableTensor, dst: torch.Tensor
) -> None:
    setattr(dst, "global_shape", src.global_shape)  # noqa: B010
    setattr(dst, "global_offsets", src.global_offsets)  # noqa: B010
    setattr(dst, "local_offsets", src.local_offsets)  # noqa: B010
    setattr(dst, "local_sizes", src.local_sizes)  # noqa: B010


def _get_checkpointable_tensor_chunks(
    tensor: CheckpointableTensor,
) -> list[ChunkStorageMetadata]:
    _validate_checkpointable_tensor_metadata(tensor)
    return [
        ChunkStorageMetadata(
            offsets=torch.Size(global_offset),
            sizes=torch.Size(local_size),
        )
        for global_offset, local_size in zip(
            tensor.global_offsets,
            tensor.local_sizes,
            strict=True,
        )
    ]


def _get_checkpointable_tensor_shard(
    tensor: CheckpointableTensor,
    index: MetadataIndex,
) -> torch.Tensor:
    _validate_checkpointable_tensor_metadata(tensor)

    if index.offset is None:
        if len(tensor.global_offsets) == 1:
            shard_idx = 0
        else:
            raise ValueError(
                f"Cannot lookup {index.fqn} with multiple checkpointable shards and no offset"
            )
    elif (
        index.index is not None
        and index.index < len(tensor.global_offsets)
        and torch.Size(tensor.global_offsets[index.index]) == index.offset
    ):
        shard_idx = index.index
    else:
        shard_idx = -1
        for idx, global_offset in enumerate(tensor.global_offsets):
            if torch.Size(global_offset) == index.offset:
                shard_idx = idx
                break
        if shard_idx < 0:
            raise ValueError(
                f"Could not find checkpointable tensor shard at '{index.offset}' "
                f"for FQN: '{index.fqn}'"
            )

    local_offset = tensor.local_offsets[shard_idx]
    local_size = tensor.local_sizes[shard_idx]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("CheckpointableTensor must also be a torch.Tensor")
    local_tensor = tensor
    if local_tensor.dim() < len(local_size):
        # The shard's leading dims are merged into rows of the local tensor
        rows = math.prod(local_size[: len(local_size) - local_tensor.dim() + 1])
        return local_tensor.narrow(0, local_offset[0], rows).view(local_size)
    if not local_offset:
        return local_tensor
    return local_tensor[
        tuple(
            slice(offset, offset + size)
            for offset, size in zip(local_offset, local_size, strict=True)
        )
    ]


def _validate_checkpointable_tensor_metadata(tensor: CheckpointableTensor) -> None:
    num_shards = len(tensor.global_offsets)
    if len(tensor.local_offsets) != num_shards:
        raise ValueError("global_offsets and local_offsets must have the same length")
    if len(tensor.local_sizes) != num_shards:
        raise ValueError("global_offsets and local_sizes must have the same length")

    global_shape = tensor.global_shape
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("CheckpointableTensor must also be a torch.Tensor")
    tensor_shape = tuple(tensor.size())
    for idx, (global_offset, local_offset, local_size) in enumerate(
        zip(
            tensor.global_offsets,
            tensor.local_offsets,
            tensor.local_sizes,
            strict=True,
        )
    ):
        if len(global_offset) != len(global_shape):
            raise ValueError(
                f"global_offsets[{idx}] must have {len(global_shape)} dimensions"
            )
        if len(local_offset) != len(tensor_shape):
            raise ValueError(
                f"local_offsets[{idx}] must have {len(tensor_shape)} dimensions"
            )
        if len(local_size) != len(global_shape):
            raise ValueError(
                f"local_sizes[{idx}] must have {len(global_shape)} dimensions"
            )
        # Fewer local dims means merged rows, which need a local row dim
        if len(local_size) < len(tensor_shape) or (local_size and not tensor_shape):
            raise ValueError(
                f"local_sizes[{idx}] must have {len(tensor_shape)} local dimensions"
            )

        for dim, (offset, size, global_dim) in enumerate(
            zip(global_offset, local_size, global_shape, strict=True)
        ):
            if offset < 0 or size < 0 or offset + size > global_dim:
                raise ValueError(
                    f"global shard {idx} dimension {dim} is outside global_shape"
                )

        if len(local_size) > len(tensor_shape):
            # Leading dims merged into local rows
            num_merged = len(local_size) - len(tensor_shape) + 1
            if tuple(local_size[num_merged:]) != tensor_shape[1:]:
                raise ValueError(
                    f"local_sizes[{idx}] must end with the local dims {tensor_shape[1:]}"
                )
            if any(local_offset[1:]):
                raise ValueError(f"local_offsets[{idx}] may only offset local rows")
            rows = math.prod(local_size[:num_merged])
            if local_offset[0] < 0 or local_offset[0] + rows > tensor_shape[0]:
                raise ValueError(f"local shard {idx} rows are outside tensor shape")
            continue

        for dim, (offset, size, local_dim) in enumerate(
            zip(local_offset, local_size, tensor_shape, strict=True)
        ):
            if offset < 0 or size < 0 or offset + size > local_dim:
                raise ValueError(
                    f"local shard {idx} dimension {dim} is outside tensor shape"
                )
