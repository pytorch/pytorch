from __future__ import annotations

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

    Attributes:
        global_shape: Full logical tensor shape to write into checkpoint
            metadata; needed because ``tensor.size()`` is only the local buffer
            size.
        global_offsets: Global start coordinate for each local shard; needed to
            name checkpoint chunks and match load requests by global offset.
        local_offsets: Start coordinate for each local shard inside the local
            tensor; needed when one tensor stores multiple shards or includes
            padding.
        local_sizes: Shape of each local shard; needed to build checkpoint
            chunks and slice the local tensor during load.
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
        if len(local_size) != len(tensor_shape):
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

        for dim, (offset, size, local_dim) in enumerate(
            zip(local_offset, local_size, tensor_shape, strict=True)
        ):
            if offset < 0 or size < 0 or offset + size > local_dim:
                raise ValueError(
                    f"local shard {idx} dimension {dim} is outside tensor shape"
                )
