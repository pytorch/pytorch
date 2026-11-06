# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import triton
import triton.language as tl

import torch


@triton.jit
def _pack_segments_fp32_to_bf16_kernel(
    input_ptrs,
    tensor_indices,
    src_offsets,
    numels,
    dst_offsets,
    output,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    chunk_id = tl.program_id(0)
    segment_id = tl.program_id(1)
    offsets = chunk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    tensor_index = tl.load(tensor_indices + segment_id)
    src_offset = tl.load(src_offsets + segment_id)
    numel = tl.load(numels + segment_id)
    dst_offset = tl.load(dst_offsets + segment_id)
    mask = offsets < numel

    src_base_i64 = tl.load(input_ptrs + tensor_index)
    src_base = src_base_i64.to(tl.pointer_type(tl.float32))
    values = tl.load(src_base + src_offset + offsets, mask=mask, other=0.0)
    tl.store(output + dst_offset + offsets, values, mask=mask)


@triton.jit
def _pack_segments_bf16_to_fp32_kernel(
    input_ptrs,
    tensor_indices,
    src_offsets,
    numels,
    dst_offsets,
    output,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    chunk_id = tl.program_id(0)
    segment_id = tl.program_id(1)
    offsets = chunk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    tensor_index = tl.load(tensor_indices + segment_id)
    src_offset = tl.load(src_offsets + segment_id)
    numel = tl.load(numels + segment_id)
    dst_offset = tl.load(dst_offsets + segment_id)
    mask = offsets < numel

    src_base_i64 = tl.load(input_ptrs + tensor_index)
    src_base = src_base_i64.to(tl.pointer_type(tl.bfloat16))
    values = tl.load(src_base + src_offset + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    tl.store(output + dst_offset + offsets, values, mask=mask)


@triton.jit
def _pack_segments_fp32_to_fp32_kernel(
    input_ptrs,
    tensor_indices,
    src_offsets,
    numels,
    dst_offsets,
    output,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    chunk_id = tl.program_id(0)
    segment_id = tl.program_id(1)
    offsets = chunk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    tensor_index = tl.load(tensor_indices + segment_id)
    src_offset = tl.load(src_offsets + segment_id)
    numel = tl.load(numels + segment_id)
    dst_offset = tl.load(dst_offsets + segment_id)
    mask = offsets < numel

    src_base_i64 = tl.load(input_ptrs + tensor_index)
    src_base = src_base_i64.to(tl.pointer_type(tl.float32))
    values = tl.load(src_base + src_offset + offsets, mask=mask, other=0.0)
    tl.store(output + dst_offset + offsets, values, mask=mask)


@triton.jit
def _pack_segments_bf16_to_bf16_kernel(
    input_ptrs,
    tensor_indices,
    src_offsets,
    numels,
    dst_offsets,
    output,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    chunk_id = tl.program_id(0)
    segment_id = tl.program_id(1)
    offsets = chunk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    tensor_index = tl.load(tensor_indices + segment_id)
    src_offset = tl.load(src_offsets + segment_id)
    numel = tl.load(numels + segment_id)
    dst_offset = tl.load(dst_offsets + segment_id)
    mask = offsets < numel

    src_base_i64 = tl.load(input_ptrs + tensor_index)
    src_base = src_base_i64.to(tl.pointer_type(tl.bfloat16))
    values = tl.load(src_base + src_offset + offsets, mask=mask, other=0.0)
    tl.store(output + dst_offset + offsets, values, mask=mask)


@triton.jit
def _copy_segments_kernel(
    source,
    src_offsets,
    numels,
    dst_offsets,
    output,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    chunk_id = tl.program_id(0)
    segment_id = tl.program_id(1)
    offsets = chunk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    src_offset = tl.load(src_offsets + segment_id)
    numel = tl.load(numels + segment_id)
    dst_offset = tl.load(dst_offsets + segment_id)
    mask = offsets < numel

    values = tl.load(source + src_offset + offsets, mask=mask)
    tl.store(output + dst_offset + offsets, values, mask=mask)


@triton.jit
def _zero_segments_kernel(
    numels,
    dst_offsets,
    output,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    chunk_id = tl.program_id(0)
    segment_id = tl.program_id(1)
    offsets = chunk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    numel = tl.load(numels + segment_id)
    dst_offset = tl.load(dst_offsets + segment_id)
    mask = offsets < numel

    zeros = tl.zeros([BLOCK_SIZE], dtype=output.dtype.element_ty)
    tl.store(output + dst_offset + offsets, zeros, mask=mask)


def _staged_cuda_i64_tensor(
    values: Sequence[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    host = torch.tensor(values, dtype=torch.int64, pin_memory=True)
    cuda = torch.empty(host.shape, dtype=torch.int64, device=device)
    if host.numel() > 0:
        cuda.copy_(host, non_blocking=True)
    return cuda, host


@dataclass
class SegmentPackDescriptor:
    input_ptrs: torch.Tensor
    tensor_indices: torch.Tensor
    src_offsets: torch.Tensor
    numels: torch.Tensor
    dst_offsets: torch.Tensor
    max_segment_numel: int
    nsegments: int
    _staged_hosts: tuple[torch.Tensor, ...]
    _input_ptr_values: tuple[int, ...] = ()

    @classmethod
    def create(
        cls,
        *,
        ninputs: int,
        tensor_indices: Sequence[int],
        src_offsets: Sequence[int],
        numels: Sequence[int],
        dst_offsets: Sequence[int],
        device: torch.device,
    ) -> SegmentPackDescriptor:
        input_ptrs = torch.empty((ninputs,), dtype=torch.int64, device=device)
        tensor_indices_t, tensor_indices_host = _staged_cuda_i64_tensor(
            tensor_indices, device
        )
        src_offsets_t, src_offsets_host = _staged_cuda_i64_tensor(src_offsets, device)
        numels_t, numels_host = _staged_cuda_i64_tensor(numels, device)
        dst_offsets_t, dst_offsets_host = _staged_cuda_i64_tensor(dst_offsets, device)
        return cls(
            input_ptrs=input_ptrs,
            tensor_indices=tensor_indices_t,
            src_offsets=src_offsets_t,
            numels=numels_t,
            dst_offsets=dst_offsets_t,
            max_segment_numel=max(numels, default=0),
            nsegments=len(tensor_indices),
            _staged_hosts=(
                tensor_indices_host,
                src_offsets_host,
                numels_host,
                dst_offsets_host,
            ),
        )

    def update_input_ptrs(
        self,
        inputs: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        input_ptr_values = tuple(tensor.data_ptr() for tensor in inputs)
        if input_ptr_values == self._input_ptr_values:
            return []
        input_ptrs_host = torch.tensor(
            input_ptr_values,
            dtype=torch.int64,
            pin_memory=True,
        )
        if input_ptrs_host.numel() > 0:
            self.input_ptrs.copy_(input_ptrs_host, non_blocking=True)
        self._input_ptr_values = input_ptr_values
        return [input_ptrs_host]

    @property
    def tensors(self) -> list[torch.Tensor]:
        return [
            self.input_ptrs,
            self.tensor_indices,
            self.src_offsets,
            self.numels,
            self.dst_offsets,
        ]


def pack_segments_into_flat_buffer_triton(
    inputs: list[torch.Tensor],
    tensor_indices: Sequence[int],
    src_offsets: Sequence[int],
    numels: Sequence[int],
    dst_offsets: Sequence[int],
    output: torch.Tensor,
    *,
    descriptor: SegmentPackDescriptor | None = None,
    block_size: int = 1024,
    num_warps: int = 4,
) -> list[torch.Tensor] | None:
    """Pack flat source segments into ``output`` with one Triton kernel.

    ``inputs`` are flat contiguous views. Segment metadata is in elements:
    input tensor index, source offset, length, and destination offset.
    """
    if torch.compiler.is_compiling() or block_size <= 0:
        return None
    if output.dim() != 1 or output.device.type != "cuda" or not output.is_contiguous():
        return None
    if not (len(tensor_indices) == len(src_offsets) == len(numels) == len(dst_offsets)):
        raise ValueError("Segment descriptor lists must have the same length.")
    if not inputs:
        if len(tensor_indices) == 0:
            return []
        return None

    device = output.device
    input_dtype = inputs[0].dtype
    if input_dtype not in (torch.float32, torch.bfloat16):
        return None
    if output.dtype not in (torch.float32, torch.bfloat16):
        return None
    for tensor in inputs:
        if tensor.device != device or tensor.dtype != input_dtype:
            return None
        if tensor.dim() != 1 or not tensor.is_contiguous():
            return None

    nsegments = len(tensor_indices)
    if descriptor is None:
        descriptor = SegmentPackDescriptor.create(
            ninputs=len(inputs),
            tensor_indices=tensor_indices,
            src_offsets=src_offsets,
            numels=numels,
            dst_offsets=dst_offsets,
            device=device,
        )
    elif (
        descriptor.input_ptrs.device != device
        or descriptor.input_ptrs.numel() != len(inputs)
        or descriptor.nsegments != nsegments
    ):
        return None

    input_ptr_hosts = descriptor.update_input_ptrs(inputs)
    scratch = [*descriptor.tensors, *input_ptr_hosts, *inputs]

    if output.numel() == 0 or nsegments == 0 or descriptor.max_segment_numel == 0:
        return scratch

    grid = (triton.cdiv(descriptor.max_segment_numel, block_size), nsegments)
    if input_dtype == torch.float32 and output.dtype == torch.bfloat16:
        _pack_segments_fp32_to_bf16_kernel[grid](
            descriptor.input_ptrs,
            descriptor.tensor_indices,
            descriptor.src_offsets,
            descriptor.numels,
            descriptor.dst_offsets,
            output,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    elif input_dtype == torch.bfloat16 and output.dtype == torch.float32:
        _pack_segments_bf16_to_fp32_kernel[grid](
            descriptor.input_ptrs,
            descriptor.tensor_indices,
            descriptor.src_offsets,
            descriptor.numels,
            descriptor.dst_offsets,
            output,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    elif input_dtype == torch.float32 and output.dtype == torch.float32:
        _pack_segments_fp32_to_fp32_kernel[grid](
            descriptor.input_ptrs,
            descriptor.tensor_indices,
            descriptor.src_offsets,
            descriptor.numels,
            descriptor.dst_offsets,
            output,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    elif input_dtype == torch.bfloat16 and output.dtype == torch.bfloat16:
        _pack_segments_bf16_to_bf16_kernel[grid](
            descriptor.input_ptrs,
            descriptor.tensor_indices,
            descriptor.src_offsets,
            descriptor.numels,
            descriptor.dst_offsets,
            output,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    else:
        return None
    return scratch


# CUDA caps grid dimension 1, the segment axis, at 65535.
_MAX_GRID_SEGMENTS = 65535


@dataclass
class SegmentCopyDescriptor:
    """Device copy plan for segments of one source tensor, in elements.

    Segments longer than ``max_chunks_per_segment`` chunks are split, so the
    chunk axis of the grid stays short when segment sizes differ a lot: a
    grid sized by the longest segment would otherwise leave most programs
    idle.
    """

    src_offsets: torch.Tensor
    numels: torch.Tensor
    dst_offsets: torch.Tensor
    max_segment_numel: int
    nsegments: int
    block_size: int
    _staged_hosts: tuple[torch.Tensor, ...]

    @classmethod
    def create(
        cls,
        *,
        src_offsets: Sequence[int],
        numels: Sequence[int],
        dst_offsets: Sequence[int],
        device: torch.device,
        block_size: int = 4096,
        max_chunks_per_segment: int = 8,
    ) -> SegmentCopyDescriptor:
        if not (len(src_offsets) == len(numels) == len(dst_offsets)):
            raise ValueError("Segment descriptor lists must have the same length.")
        if len(numels) > _MAX_GRID_SEGMENTS:
            raise ValueError(
                f"At most {_MAX_GRID_SEGMENTS} segments are supported, got "
                f"{len(numels)}."
            )
        split_numel = max_chunks_per_segment * block_size
        while (
            sum(triton.cdiv(numel, split_numel) for numel in numels)
            > _MAX_GRID_SEGMENTS
        ):
            split_numel *= 2
        split_src_offsets: list[int] = []
        split_numels: list[int] = []
        split_dst_offsets: list[int] = []
        for src_offset, numel, dst_offset in zip(
            src_offsets, numels, dst_offsets, strict=True
        ):
            for start in range(0, numel, split_numel):
                split_src_offsets.append(src_offset + start)
                split_numels.append(min(split_numel, numel - start))
                split_dst_offsets.append(dst_offset + start)
        src_offsets_t, src_offsets_host = _staged_cuda_i64_tensor(
            split_src_offsets, device
        )
        numels_t, numels_host = _staged_cuda_i64_tensor(split_numels, device)
        dst_offsets_t, dst_offsets_host = _staged_cuda_i64_tensor(
            split_dst_offsets, device
        )
        return cls(
            src_offsets=src_offsets_t,
            numels=numels_t,
            dst_offsets=dst_offsets_t,
            max_segment_numel=max(split_numels, default=0),
            nsegments=len(split_numels),
            block_size=block_size,
            _staged_hosts=(src_offsets_host, numels_host, dst_offsets_host),
        )


def copy_segments_triton(
    source: torch.Tensor,
    output: torch.Tensor,
    descriptor: SegmentCopyDescriptor,
    *,
    num_warps: int = 4,
) -> None:
    """Copy ``descriptor``'s segments from ``source`` into ``output``.

    One kernel launch for any number of segments. Offsets count elements from
    each tensor's first element, so ``source`` may be a strided view: the kernel
    reads only its data pointer and dtype.
    """
    if descriptor.nsegments == 0 or descriptor.max_segment_numel == 0:
        return
    block_size = descriptor.block_size
    grid = (triton.cdiv(descriptor.max_segment_numel, block_size), descriptor.nsegments)
    _copy_segments_kernel[grid](
        source,
        descriptor.src_offsets,
        descriptor.numels,
        descriptor.dst_offsets,
        output,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )


def zero_segments_triton(
    output: torch.Tensor,
    descriptor: SegmentCopyDescriptor,
    *,
    num_warps: int = 4,
) -> None:
    """Zero ``descriptor``'s destination segments of ``output``, in one launch.

    Reads only the descriptor's lengths and destination offsets.
    """
    if descriptor.nsegments == 0 or descriptor.max_segment_numel == 0:
        return
    block_size = descriptor.block_size
    grid = (triton.cdiv(descriptor.max_segment_numel, block_size), descriptor.nsegments)
    _zero_segments_kernel[grid](
        descriptor.numels,
        descriptor.dst_offsets,
        output,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )


__all__ = [
    "SegmentCopyDescriptor",
    "SegmentPackDescriptor",
    "copy_segments_triton",
    "pack_segments_into_flat_buffer_triton",
    "zero_segments_triton",
]
