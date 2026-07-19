# Copyright (c) Meta Platforms, Inc. and affiliates
"""Stateful RNG support for local views of logical tensors."""

from __future__ import annotations

from typing import cast

import torch


aten = torch.ops.aten


def _validate_stateful_rng_parameters(
    op_call: torch._ops.OpOverload,
    tensor: torch.Tensor,
    op_args: list[object],
) -> None:
    if op_call is aten.normal_.default:
        std = cast(float, op_args[1])
        torch._check(
            std >= 0.0,
            lambda: f"normal expects std >= 0.0, but found std {std}",
        )
        return

    if op_call is not aten.uniform_.default:
        raise AssertionError(f"Unsupported stateful RNG op {op_call}")
    low, high = cast(tuple[float, float], tuple(op_args))
    finfo = torch.finfo(tensor.dtype)
    torch._check(
        low >= finfo.min and low <= finfo.max,
        lambda: f"from is out of bounds for {tensor.dtype}",
    )
    torch._check(
        high >= finfo.min and high <= finfo.max,
        lambda: f"to is out of bounds for {tensor.dtype}",
    )
    torch._check(
        low <= high,
        lambda: (
            "uniform_ expects to return a [from, to) range, but found "
            f"from={low} > to={high}"
        ),
    )
    torch._check(
        high - low <= finfo.max,
        lambda: (
            f"uniform_ expects to-from <= {finfo.max}, but found "
            f"to={high} and from={low}"
        ),
    )


def _run_stateful_rng_op(
    tensor: torch.Tensor,
    global_numel: int,
    index_blocks: tuple[tuple[int, int, int, int], ...],
    flat_slice_op_call: torch._ops.OpOverload,
    generator: torch.Generator | None,
    *op_args: object,
) -> torch.Tensor:
    if not tensor.is_contiguous():
        raise RuntimeError(
            f"{flat_slice_op_call}: expected a contiguous local tensor, "
            f"got stride {tensor.stride()}"
        )
    return flat_slice_op_call(
        tensor,
        global_numel,
        [start_index for start_index, _, _, _ in index_blocks],
        [block_size for _, block_size, _, _ in index_blocks],
        [block_stride for _, _, block_stride, _ in index_blocks],
        [num_blocks for _, _, _, num_blocks in index_blocks],
        *op_args,
        generator=generator,
    )
