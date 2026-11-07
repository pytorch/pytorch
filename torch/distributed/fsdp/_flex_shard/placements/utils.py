# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib.util
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist


if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..placement_contract import GradientReduceOp, GradientReduction


def foreach_copy_(
    dst_tensors: list[torch.Tensor],
    src_tensors: list[torch.Tensor],
) -> None:
    """Copy tensors with one foreach runtime boundary when eager."""
    if len(dst_tensors) != len(src_tensors):
        raise AssertionError(
            f"Expected {len(dst_tensors)} destination tensors to match "
            f"{len(src_tensors)} source tensors."
        )
    if not dst_tensors:
        return
    if torch.compiler.is_compiling():
        for dst, src in zip(dst_tensors, src_tensors, strict=True):
            dst.copy_(src)
    else:
        torch._foreach_copy_(dst_tensors, src_tensors)


def copy_tensor_to_dtype(
    tensor: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a contiguous tensor with dtype, using foreach copy for casts."""
    if tensor.is_contiguous() and tensor.dtype == dtype:
        return tensor

    out = torch.empty(
        tensor.shape,
        dtype=dtype,
        device=tensor.device,
    )
    if tensor.numel() > 0:
        foreach_copy_([out], [tensor])
    return out


def pack_tensors_into_flat_buffer(
    tensors: list[torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Pack tensors into one flat buffer using foreach copy/cast."""
    if not tensors:
        raise AssertionError("Expected at least one tensor to pack.")

    total_numel = sum(tensor.numel() for tensor in tensors)
    out = torch.empty(
        total_numel,
        dtype=dtype,
        device=tensors[0].device,
    )
    copy_tensors_into_flat_buffer(tensors, out)
    return out


def pack_tensors_into_flat_buffer_with_scratch(
    tensors: list[torch.Tensor],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Pack tensors into one flat buffer and return async scratch to retain."""
    if not tensors:
        raise AssertionError("Expected at least one tensor to pack.")

    return pack_tensors_into_flat_buffer(tensors, dtype), []


def copy_tensors_into_flat_buffer(
    tensors: list[torch.Tensor],
    out: torch.Tensor,
) -> None:
    """Copy tensors into preallocated flat out buffer with foreach copy/cast."""
    copy_srcs: list[torch.Tensor] = []
    copy_dsts: list[torch.Tensor] = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        if numel > 0:
            copy_srcs.append(tensor)
            copy_dsts.append(out.narrow(0, offset, numel).view(tensor.shape))
        offset += numel

    if offset != out.numel():
        raise AssertionError(
            f"Packed tensor numel {offset} does not match output numel {out.numel()}."
        )
    foreach_copy_(copy_dsts, copy_srcs)


def _to_dist_reduce_op(op: GradientReduceOp) -> dist.ReduceOp.RedOpType:
    if op == "avg" or op == dist.ReduceOp.AVG:
        return dist.ReduceOp.AVG
    if op == "sum" or op == dist.ReduceOp.SUM:
        return dist.ReduceOp.SUM
    raise ValueError(f"Unsupported gradient reduce op: {op!r}")


def _gradient_reduce_ops(
    reduction: GradientReduction,
    group_size: int,
    dtype: torch.dtype,
) -> tuple[Any, float | None, float | None, Any, float | None]:
    """Return ``(reduce-scatter op, pre-divide factor, post-reduce-scatter divide
    factor, all-reduce op, post-all-reduce divide factor)`` that make a
    reduce-scatter over ``group_size`` ranks, then an all-reduce over
    ``reduction.replicate_size`` replicas if more than one, produce
    ``reduction``, as FSDP2's ``_get_gradient_divide_factors``."""
    sum_op = dist.ReduceOp.SUM
    if _to_dist_reduce_op(reduction.op) == sum_op:
        return sum_op, None, None, sum_op, None
    replicate_size = reduction.replicate_size
    data_parallel_size = group_size * replicate_size
    factor = reduction.divide_factor
    if dtype in (torch.float32, torch.bfloat16) and not reduction.force_sum:
        if factor is None:
            if data_parallel_size == 1:
                return sum_op, None, None, sum_op, None
            # NCCL's AVG may produce incorrect results with world size 1 (FSDP2).
            avg = dist.ReduceOp.AVG
            return (
                avg if group_size > 1 else sum_op,
                None,
                None,
                avg if replicate_size > 1 else sum_op,
                None,
            )
        if group_size == 1:
            # No reduce-scatter on one rank: divide its copy, as FSDP2 does
            return sum_op, None, None if factor == 1 else factor, sum_op, None
        if factor == group_size:
            return dist.ReduceOp.AVG, None, None, sum_op, None
        # One NCCL call, which multiplies each input by 1/factor before summing.
        return dist._make_nccl_premul_sum(1.0 / factor), None, None, sum_op, None
    if factor is None:
        factor = float(data_parallel_size)
    pre_factor: float | None = None
    if dtype not in (torch.float32, torch.bfloat16):
        # fp16 has a narrow range: divide by about sqrt(factor) before the sum
        # and by the rest after it.
        pre_factor = 1
        while factor % pre_factor == 0 and factor / pre_factor > pre_factor:
            pre_factor *= 2
        post_factor = factor / pre_factor
    else:
        post_factor = factor
    if pre_factor == 1:
        pre_factor = None
    if post_factor == 1:
        post_factor = None
    if replicate_size > 1:
        return sum_op, pre_factor, None, sum_op, post_factor
    return sum_op, pre_factor, post_factor, sum_op, None


def _gradient_reduce_scatter_op(
    reduction: GradientReduction,
    group_size: int,
    dtype: torch.dtype,
) -> tuple[Any, float | None, float | None]:
    """Return ``(reduce op, pre-divide factor, post-divide factor)`` of the
    reduce-scatter step of ``_gradient_reduce_ops``."""
    return _gradient_reduce_ops(reduction, group_size, dtype)[:3]


def reduce_grads_on_one_rank(grads: torch.Tensor, reduction: GradientReduction) -> None:
    """Apply ``reduction`` in place to packed gradients of a 1-rank group, with
    the same result as ``reduce_scatter_grads`` without the collective."""
    _, pre_factor, post_factor = _gradient_reduce_scatter_op(reduction, 1, grads.dtype)
    if pre_factor is not None:
        grads.div_(pre_factor)
    if post_factor is not None:
        grads.div_(post_factor)


def reduce_scatter_grads(
    output: torch.Tensor,
    input: torch.Tensor,
    reduction: GradientReduction,
    group: Any,
    comm: Any = None,
) -> None:
    """Reduce-scatter packed gradients into ``output`` as ``reduction`` specifies,
    for placements whose gradient reduction is one reduce-scatter. ``comm``
    replaces the collective, as FSDP2's custom ``ReduceScatter``."""
    reduce_op, pre_factor, post_factor = _gradient_reduce_scatter_op(
        reduction, group.size(), input.dtype
    )
    if pre_factor is not None:
        input.div_(pre_factor)
    if comm is not None:
        comm(
            output_tensor=output,
            input_tensor=input,
            group=group,
            op=reduce_op,
            async_op=False,
        )
    else:
        dist.reduce_scatter_single(
            output=output, input=input, op=reduce_op, group=group
        )
    if post_factor is not None:
        output.div_(post_factor)


def reduce_scatter_grads_uneven(
    output: torch.Tensor,
    inputs: list[torch.Tensor],
    reduction: GradientReduction,
    group: Any,
) -> None:
    """``reduce_scatter_grads`` for uneven chunks: ``inputs[r]`` is reduced into
    rank r's ``output``, as one NCCL reduce per rank in one group."""
    reduce_op, pre_factor, post_factor = _gradient_reduce_scatter_op(
        reduction, group.size(), output.dtype
    )
    if pre_factor is not None:
        inputs = [chunk / pre_factor for chunk in inputs]
    dist.reduce_scatter(output, inputs, op=reduce_op, group=group)
    if post_factor is not None:
        output.div_(post_factor)


def pack_segments_into_flat_buffer_triton_if_supported(
    inputs: list[torch.Tensor],
    tensor_indices: Sequence[int],
    src_offsets: Sequence[int],
    numels: Sequence[int],
    dst_offsets: Sequence[int],
    out: torch.Tensor,
    descriptor: Any | None = None,
) -> list[torch.Tensor] | None:
    """Use Triton descriptor packing for supported inputs.

    Triton is required for this backend. The function returns ``None`` only when
    the current inputs should use the deterministic foreach copy fallback.
    """
    if torch.compiler.is_compiling():
        return None
    if importlib.util.find_spec("triton") is None:
        raise AssertionError(
            "BucketedOwned Triton segment packing requires the triton package."
        )

    from ._copy_kernels import pack_segments_into_flat_buffer_triton

    return pack_segments_into_flat_buffer_triton(
        inputs,
        tensor_indices,
        src_offsets,
        numels,
        dst_offsets,
        out,
        descriptor=descriptor,
    )


def make_segment_pack_descriptor_if_supported(
    *,
    ninputs: int,
    tensor_indices: Sequence[int],
    src_offsets: Sequence[int],
    numels: Sequence[int],
    dst_offsets: Sequence[int],
    device: torch.device,
) -> Any | None:
    if torch.compiler.is_compiling():
        return None
    if importlib.util.find_spec("triton") is None:
        raise AssertionError(
            "BucketedOwned Triton segment packing requires the triton package."
        )

    from ._copy_kernels import SegmentPackDescriptor

    return SegmentPackDescriptor.create(
        ninputs=ninputs,
        tensor_indices=tensor_indices,
        src_offsets=src_offsets,
        numels=numels,
        dst_offsets=dst_offsets,
        device=device,
    )


__all__ = [
    "copy_tensor_to_dtype",
    "copy_tensors_into_flat_buffer",
    "foreach_copy_",
    "make_segment_pack_descriptor_if_supported",
    "pack_segments_into_flat_buffer_triton_if_supported",
    "pack_tensors_into_flat_buffer",
    "pack_tensors_into_flat_buffer_with_scratch",
    "reduce_scatter_grads",
    "_gradient_reduce_ops",
    "_gradient_reduce_scatter_op",
    "_to_dist_reduce_op",
]
