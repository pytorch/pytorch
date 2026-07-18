# Copyright (c) Meta Platforms, Inc. and affiliates
"""Stateful RNG support for local views of logical tensors."""

from __future__ import annotations

from typing import Any, cast, TypeGuard

import torch
from torch._library.utils import fill_defaults
from torch.distributed import StatefulRNGTensor
from torch.distributed._local_tensor import (
    enabled_local_tensor_mode,
    maybe_run_for_local_tensor,
)
from torch.utils._python_dispatch import TorchDispatchMode


aten = torch.ops.aten

__all__ = [
    "StatefulRNGMode",
]


_SUPPORTED_STATEFUL_RNG_OPS = {
    aten.normal_.default: aten._philox_normal_flat_slice_.default,
    aten.uniform_.default: aten._philox_uniform_flat_slice_.default,
}

_SUPPORTED_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


def _is_stateful_rng_tensor(obj: object) -> TypeGuard[StatefulRNGTensor]:
    return isinstance(obj, torch.Tensor) and isinstance(obj, StatefulRNGTensor)


def _is_supported_stateful_rng_op(
    op_call: torch._ops.OpOverload,
    tensor: torch.Tensor,
) -> bool:
    """Return whether ``tensor`` can use logical-index Generator replay."""
    return (
        op_call in _SUPPORTED_STATEFUL_RNG_OPS
        and not tensor.is_meta
        and tensor.device.type == "cuda"
        and tensor.dtype in _SUPPORTED_DTYPES
        and tensor.is_contiguous()
    )


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


@maybe_run_for_local_tensor
def _run_stateful_rng_op_rankwise(
    tensor: torch.Tensor,
    logical_numel: int,
    start_indices: tuple[int | torch.SymInt, ...],
    block_sizes: tuple[int | torch.SymInt, ...],
    block_strides: tuple[int | torch.SymInt, ...],
    block_counts: tuple[int | torch.SymInt, ...],
    flat_slice_op_call: torch._ops.OpOverload,
    generator: torch.Generator | None,
    generator_state: torch.Tensor | None,
    op_args: tuple[object, ...],
) -> torch.Tensor:
    if generator_state is not None:
        if generator is None:
            raise AssertionError
        # LocalTensor runs every virtual rank in one process. Replay each rank
        # from the same explicit state, leaving one logical draw consumed.
        generator.set_state(generator_state)
    return flat_slice_op_call(
        tensor,
        logical_numel,
        start_indices,
        block_sizes,
        block_strides,
        block_counts,
        *op_args,
        generator=generator,
    )


def _run_stateful_rng_op(
    op_call: torch._ops.OpOverload,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
    logical_numel: int,
    index_blocks: tuple[
        tuple[
            int | torch.SymInt,
            int | torch.SymInt,
            int | torch.SymInt,
            int | torch.SymInt,
        ],
        ...,
    ],
) -> torch.Tensor:
    """Run an in-place RNG op for selected indices of one logical CUDA draw."""
    if op_call not in _SUPPORTED_STATEFUL_RNG_OPS:
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
    _validate_stateful_rng_parameters(op_call, tensor, op_args)
    flat_slice_op_call = _SUPPORTED_STATEFUL_RNG_OPS[op_call]
    generator_state = (
        generator.get_state()
        if generator is not None and enabled_local_tensor_mode()
        else None
    )
    start_indices = tuple(block[0] for block in index_blocks)
    block_sizes = tuple(block[1] for block in index_blocks)
    block_strides = tuple(block[2] for block in index_blocks)
    block_counts = tuple(block[3] for block in index_blocks)
    if generator_state is None:
        flat_slice_op_call(
            tensor,
            logical_numel,
            start_indices,
            block_sizes,
            block_strides,
            block_counts,
            *op_args,
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
            flat_slice_op_call,
            generator,
            generator_state,
            tuple(op_args),
        )
    return tensor


class StatefulRNGMode(TorchDispatchMode):
    def __torch_dispatch__(
        self,
        func: torch._ops.OpOverload,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}
        if func not in (aten.normal_.default, aten.uniform_.default):
            return func(*args, **kwargs)

        filled_args, _ = fill_defaults(func._schema, args, kwargs)
        tensor_arg = filled_args[0]
        if not isinstance(tensor_arg, torch.Tensor):
            return func(*args, **kwargs)
        if tensor_arg.is_meta or tensor_arg.device.type != "cuda":
            return func(*args, **kwargs)
        if not _is_stateful_rng_tensor(tensor_arg):
            return func(*args, **kwargs)
        rng_metadata = cast(StatefulRNGTensor, tensor_arg)

        return _run_stateful_rng_op(
            func,
            args,
            kwargs,
            rng_metadata.rng_global_numel,
            rng_metadata.rng_index_blocks,
        )
