# Copyright (c) Meta Platforms, Inc. and affiliates
"""Stateful RNG support for local views of logical tensors."""

from __future__ import annotations

from typing import Any, cast, TYPE_CHECKING

import torch
from torch._library.utils import fill_defaults
from torch.distributed._local_tensor import (
    enabled_local_tensor_mode,
    maybe_run_for_local_tensor,
)


if TYPE_CHECKING:
    from torch.distributed import RNGIndexBlock


aten = torch.ops.aten

_PHILOX_DISTRIBUTION_NORMAL = 0
_PHILOX_DISTRIBUTION_UNIFORM = 1

_SUPPORTED_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


def _validate_normal_parameters(
    tensor: torch.Tensor,
    op_args: list[object],
) -> None:
    std = cast(float, op_args[1])
    torch._check(
        std >= 0.0,
        lambda: f"normal expects std >= 0.0, but found std {std}",
    )


def _validate_uniform_parameters(
    tensor: torch.Tensor,
    op_args: list[object],
) -> None:
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


_STATEFUL_RNG_OP_SPECS = {
    aten.normal_.default: (
        _PHILOX_DISTRIBUTION_NORMAL,
        _validate_normal_parameters,
    ),
    aten.uniform_.default: (
        _PHILOX_DISTRIBUTION_UNIFORM,
        _validate_uniform_parameters,
    ),
}


def _is_supported_stateful_rng_op(
    op_call: torch._ops.OpOverload,
    tensor: torch.Tensor,
) -> bool:
    """Return whether ``tensor`` can use logical-index Generator replay."""
    return (
        op_call in _STATEFUL_RNG_OP_SPECS
        and not tensor.is_meta
        and tensor.device.type == "cuda"
        and tensor.dtype in _SUPPORTED_DTYPES
        and tensor.is_contiguous()
    )


@maybe_run_for_local_tensor
def _run_stateful_rng_op_rankwise(
    tensor: torch.Tensor,
    logical_numel: int,
    start_indices: list[int | torch.SymInt],
    block_sizes: list[int | torch.SymInt],
    block_strides: list[int | torch.SymInt],
    block_counts: list[int | torch.SymInt],
    kind: int,
    generator: torch.Generator | None,
    generator_state: torch.Tensor | None,
    params: tuple[object, ...],
) -> torch.Tensor:
    if generator_state is not None:
        if generator is None:
            raise AssertionError
        # LocalTensor runs every virtual rank in one process. Replay each rank
        # from the same explicit state, leaving one logical draw consumed.
        generator.set_state(generator_state)
    return aten._philox_distribution_flat_slice_.default(
        tensor,
        logical_numel,
        start_indices,
        block_sizes,
        block_strides,
        block_counts,
        kind,
        params,
        generator=generator,
    )


def _run_stateful_rng_op(
    op_call: torch._ops.OpOverload,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
    logical_numel: int,
    index_blocks: tuple[RNGIndexBlock, ...],
) -> torch.Tensor:
    """Run an in-place RNG op for selected indices of one logical CUDA draw."""
    if op_call not in _STATEFUL_RNG_OP_SPECS:
        raise NotImplementedError(f"Unsupported stateful RNG op {op_call}")
    if kwargs is None:
        kwargs = {}

    filled_args, filled_kwargs = fill_defaults(op_call._schema, args, kwargs)
    tensor, *op_args = filled_args
    filled_kwargs = dict(filled_kwargs)
    generator = filled_kwargs.pop("generator")
    if filled_kwargs:
        raise AssertionError(f"Unexpected keyword arguments: {filled_kwargs}")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a Tensor for {op_call}, got {type(tensor)}")
    if not (generator is None or isinstance(generator, torch.Generator)):
        raise TypeError(
            f"Expected generator to be a torch.Generator or None, got {type(generator)}"
        )
    if tensor.is_meta or tensor.device.type != "cuda":
        raise NotImplementedError(
            f"{op_call} logical-index replay only supports CUDA tensors"
        )
    if tensor.dtype not in _SUPPORTED_DTYPES:
        raise RuntimeError(
            f"{op_call} logical-index replay does not support dtype {tensor.dtype}"
        )
    if not tensor.is_contiguous():
        raise RuntimeError(
            f"{op_call}: expected a contiguous local tensor, got stride {tensor.stride()}"
        )

    # Validate before entering the rankwise helper or reserving generator state.
    kind, validate = _STATEFUL_RNG_OP_SPECS[op_call]
    validate(tensor, op_args)
    params = tuple(op_args)
    generator_state = (
        generator.get_state()
        if generator is not None and enabled_local_tensor_mode()
        else None
    )
    start_indices: list[int | torch.SymInt] = []
    block_sizes: list[int | torch.SymInt] = []
    block_strides: list[int | torch.SymInt] = []
    block_counts: list[int | torch.SymInt] = []
    for start_index, block_size, block_stride, num_blocks in index_blocks:
        start_indices.append(start_index)
        block_sizes.append(block_size)
        block_strides.append(block_stride)
        block_counts.append(num_blocks)
    if generator_state is None:
        aten._philox_distribution_flat_slice_.default(
            tensor,
            logical_numel,
            start_indices,
            block_sizes,
            block_strides,
            block_counts,
            kind,
            params,
            generator=generator,
        )
    else:
        _run_stateful_rng_op_rankwise(
            tensor,
            logical_numel,
            start_indices,
            block_sizes,
            block_strides,
            block_counts,
            kind,
            generator,
            generator_state,
            params,
        )
    return tensor
