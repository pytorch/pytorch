# Copyright (c) Meta Platforms, Inc. and affiliates
"""Lower distributed tensor shard metadata to logical RNG indices."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod
from typing import TYPE_CHECKING, TypeAlias

import torch


if TYPE_CHECKING:
    from torch.distributed.checkpoint import CheckpointableTensor


_IntLike: TypeAlias = int | torch.SymInt
_RNGIndexBlock: TypeAlias = tuple[_IntLike, _IntLike, _IntLike, _IntLike]


@dataclass(frozen=True)
class _RNGLayoutChunk:
    local_slices: tuple[slice, ...]
    local_shape: tuple[int, ...]
    numel: int
    index_blocks: tuple[_RNGIndexBlock, ...]


@dataclass(frozen=True)
class _CheckpointableRNGLayout:
    logical_numel: int
    chunks: tuple[_RNGLayoutChunk, ...]
    is_direct: bool

    @property
    def index_blocks(self) -> tuple[_RNGIndexBlock, ...]:
        return tuple(block for chunk in self.chunks for block in chunk.index_blocks)

    @property
    def local_numel(self) -> int:
        return sum(chunk.numel for chunk in self.chunks)


def _int_tuple(name: str, values: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple, got {type(values).__name__}")
    for dim, value in enumerate(values):
        if type(value) is not int:
            raise TypeError(f"{name}[{dim}] must be an int, got {type(value).__name__}")
    return values


def _rectangles_overlap(
    first_offset: tuple[int, ...],
    first_size: tuple[int, ...],
    second_offset: tuple[int, ...],
    second_size: tuple[int, ...],
) -> bool:
    if prod(first_size) == 0 or prod(second_size) == 0:
        return False
    if not first_size:
        return True
    return all(
        first_start < second_start + second_length
        and second_start < first_start + first_length
        for first_start, first_length, second_start, second_length in zip(
            first_offset,
            first_size,
            second_offset,
            second_size,
            strict=True,
        )
    )


def _chunk_to_index_blocks(
    global_shape: tuple[int, ...],
    global_offset: tuple[int, ...],
    chunk_shape: tuple[int, ...],
) -> tuple[_RNGIndexBlock, ...]:
    chunk_numel = prod(chunk_shape)
    if chunk_numel == 0:
        return ()
    if not global_shape:
        return ((0, 1, 1, 1),)

    global_strides = tuple(
        prod(global_shape[dim + 1 :]) for dim in range(len(global_shape))
    )
    base_index = sum(
        offset * stride
        for offset, stride in zip(global_offset, global_strides, strict=True)
    )
    last_partial_dim = max(
        (
            dim
            for dim, (offset, size, global_size) in enumerate(
                zip(global_offset, chunk_shape, global_shape, strict=True)
            )
            if offset != 0 or size != global_size
        ),
        default=-1,
    )
    if last_partial_dim < 0:
        return ((0, chunk_numel, chunk_numel, 1),)

    block_size = chunk_shape[last_partial_dim] * global_strides[last_partial_dim]
    if last_partial_dim == 0:
        return ((base_index, block_size, block_size, 1),)

    last_outer_partial_dim = max(
        (
            dim
            for dim in range(last_partial_dim)
            if global_offset[dim] != 0 or chunk_shape[dim] != global_shape[dim]
        ),
        default=-1,
    )
    repeat_start_dim = max(last_outer_partial_dim, 0)
    prefix_shape = chunk_shape[:repeat_start_dim]
    block_count = prod(chunk_shape[repeat_start_dim:last_partial_dim])
    block_stride = global_strides[last_partial_dim - 1]

    blocks: list[_RNGIndexBlock] = []
    for prefix_coordinate in product(*(range(size) for size in prefix_shape)):
        start_index = base_index + sum(
            coordinate * global_strides[dim]
            for dim, coordinate in enumerate(prefix_coordinate)
        )
        blocks.append((start_index, block_size, block_stride, block_count))
    return tuple(blocks)


def _derive_checkpointable_rng_layout(
    tensor: CheckpointableTensor,
) -> _CheckpointableRNGLayout:
    """Validate and lower checkpoint shard metadata before RNG reservation."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("CheckpointableTensor must also be a torch.Tensor")

    global_shape = _int_tuple("global_shape", tensor.global_shape)
    local_shape = tuple(tensor.shape)
    if len(global_shape) != len(local_shape):
        raise ValueError(
            "global_shape and the local tensor must have the same number of dimensions"
        )
    if any(size < 0 for size in global_shape):
        raise ValueError("global_shape must be non-negative")

    logical_numel = prod(global_shape)
    if logical_numel > torch.iinfo(torch.int32).max:
        raise ValueError("global_shape has more than INT_MAX elements")

    num_chunks = len(tensor.global_offsets)
    if len(tensor.local_offsets) != num_chunks:
        raise ValueError("global_offsets and local_offsets must have the same length")
    if len(tensor.local_sizes) != num_chunks:
        raise ValueError("global_offsets and local_sizes must have the same length")

    chunks: list[_RNGLayoutChunk] = []
    global_rectangles: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    local_rectangles: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for chunk_index, (raw_global_offset, raw_local_offset, raw_local_size) in enumerate(
        zip(
            tensor.global_offsets,
            tensor.local_offsets,
            tensor.local_sizes,
            strict=True,
        )
    ):
        global_offset = _int_tuple(f"global_offsets[{chunk_index}]", raw_global_offset)
        local_offset = _int_tuple(f"local_offsets[{chunk_index}]", raw_local_offset)
        local_size = _int_tuple(f"local_sizes[{chunk_index}]", raw_local_size)
        if not (
            len(global_offset)
            == len(local_offset)
            == len(local_size)
            == len(global_shape)
        ):
            raise ValueError(
                f"shard {chunk_index} metadata must have {len(global_shape)} dimensions"
            )

        for dim, (offset, size, global_size) in enumerate(
            zip(global_offset, local_size, global_shape, strict=True)
        ):
            if offset < 0 or size < 0 or offset + size > global_size:
                raise ValueError(
                    f"global shard {chunk_index} dimension {dim} is outside global_shape"
                )
        for dim, (offset, size, tensor_size) in enumerate(
            zip(local_offset, local_size, local_shape, strict=True)
        ):
            if offset < 0 or offset + size > tensor_size:
                raise ValueError(
                    f"local shard {chunk_index} dimension {dim} is outside tensor shape"
                )

        global_rectangles.append((global_offset, local_size))
        local_rectangles.append((local_offset, local_size))
        chunk_numel = prod(local_size)
        if chunk_numel == 0:
            continue
        chunks.append(
            _RNGLayoutChunk(
                tuple(
                    slice(offset, offset + size)
                    for offset, size in zip(local_offset, local_size, strict=True)
                ),
                local_size,
                chunk_numel,
                _chunk_to_index_blocks(global_shape, global_offset, local_size),
            )
        )

    for name, rectangles in (
        ("global", global_rectangles),
        ("local", local_rectangles),
    ):
        for first in range(len(rectangles)):
            for second in range(first + 1, len(rectangles)):
                if _rectangles_overlap(*rectangles[first], *rectangles[second]):
                    raise ValueError(
                        f"{name} shards {first} and {second} must not overlap"
                    )

    local_numel = sum(chunk.numel for chunk in chunks)
    if local_numel != tensor.numel():
        raise ValueError(
            "local shard metadata must cover the entire tensor: "
            f"described {local_numel} of {tensor.numel()} elements"
        )

    is_direct = tensor.is_contiguous() and (
        not chunks
        or (
            len(chunks) == 1
            and all(block.start in (None, 0) for block in chunks[0].local_slices)
            and chunks[0].local_shape == local_shape
        )
    )
    return _CheckpointableRNGLayout(logical_numel, tuple(chunks), is_direct)


def _rng_target_for_layout(
    tensor: torch.Tensor,
    layout: _CheckpointableRNGLayout,
) -> torch.Tensor:
    if layout.is_direct:
        return tensor
    return tensor.new_empty((layout.local_numel,))


def _scatter_rng_result_(
    tensor: torch.Tensor,
    result: torch.Tensor,
    layout: _CheckpointableRNGLayout,
) -> None:
    if result is tensor:
        return
    offset = 0
    for chunk in layout.chunks:
        values = result.narrow(0, offset, chunk.numel).reshape(chunk.local_shape)
        tensor[chunk.local_slices].copy_(values)
        offset += chunk.numel
