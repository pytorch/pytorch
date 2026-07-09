from __future__ import annotations

from typing import cast, Protocol, runtime_checkable

import torch

from .metadata import ChunkStorageMetadata, MetadataIndex


__all__ = ["CheckpointableTensor"]


@runtime_checkable
class CheckpointableTensor(Protocol):
    """Protocol for tensor state-dict values with checkpoint shard metadata."""

    global_shape: tuple[int, ...]
    global_offsets: tuple[tuple[int, ...], ...]
    local_offsets: tuple[tuple[int, ...], ...]
    local_sizes: tuple[tuple[int, ...], ...]


def _is_checkpointable_tensor(obj: object) -> bool:
    return isinstance(obj, torch.Tensor) and isinstance(obj, CheckpointableTensor)


def _get_checkpointable_tensor_chunks(
    tensor: torch.Tensor,
) -> list[ChunkStorageMetadata]:
    _validate_checkpointable_tensor_metadata(tensor)
    checkpointable_tensor = cast(CheckpointableTensor, tensor)
    return [
        ChunkStorageMetadata(
            offsets=torch.Size(global_offset),
            sizes=torch.Size(local_size),
        )
        for global_offset, local_size in zip(
            checkpointable_tensor.global_offsets,
            checkpointable_tensor.local_sizes,
        )
    ]


def _get_checkpointable_tensor_shard(
    tensor: torch.Tensor,
    index: MetadataIndex,
) -> torch.Tensor:
    _validate_checkpointable_tensor_metadata(tensor)
    checkpointable_tensor = cast(CheckpointableTensor, tensor)

    if index.offset is None:
        if len(checkpointable_tensor.global_offsets) == 1:
            shard_idx = 0
        else:
            raise ValueError(
                f"Cannot lookup {index.fqn} with multiple checkpointable shards and no offset"
            )
    elif (
        index.index is not None
        and index.index < len(checkpointable_tensor.global_offsets)
        and torch.Size(checkpointable_tensor.global_offsets[index.index])
        == index.offset
    ):
        shard_idx = index.index
    else:
        for idx, global_offset in enumerate(checkpointable_tensor.global_offsets):
            if torch.Size(global_offset) == index.offset:
                shard_idx = idx
                break
        else:
            raise ValueError(
                f"Could not find checkpointable tensor shard at '{index.offset}' "
                f"for FQN: '{index.fqn}'"
            )

    local_offset = checkpointable_tensor.local_offsets[shard_idx]
    local_size = checkpointable_tensor.local_sizes[shard_idx]
    if not local_offset:
        return tensor
    return tensor[
        tuple(
            slice(offset, offset + size)
            for offset, size in zip(local_offset, local_size)
        )
    ]


def _validate_checkpointable_tensor_metadata(tensor: torch.Tensor) -> None:
    checkpointable_tensor = cast(CheckpointableTensor, tensor)
    num_shards = len(checkpointable_tensor.global_offsets)
    if len(checkpointable_tensor.local_offsets) != num_shards:
        raise ValueError("global_offsets and local_offsets must have the same length")
    if len(checkpointable_tensor.local_sizes) != num_shards:
        raise ValueError("global_offsets and local_sizes must have the same length")

    global_shape = checkpointable_tensor.global_shape
    tensor_shape = tuple(tensor.size())
    for idx, (global_offset, local_offset, local_size) in enumerate(
        zip(
            checkpointable_tensor.global_offsets,
            checkpointable_tensor.local_offsets,
            checkpointable_tensor.local_sizes,
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
            zip(global_offset, local_size, global_shape)
        ):
            if offset < 0 or size < 0 or offset + size > global_dim:
                raise ValueError(
                    f"global shard {idx} dimension {dim} is outside global_shape"
                )

        for dim, (offset, size, local_dim) in enumerate(
            zip(local_offset, local_size, tensor_shape)
        ):
            if offset < 0 or size < 0 or offset + size > local_dim:
                raise ValueError(
                    f"local shard {idx} dimension {dim} is outside tensor shape"
                )
