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
    from torch.distributed.checkpoint import CheckpointableTensor


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
        and tensor.layout == torch.strided
    )


def _flatten_shard_metadata(
    global_shape: tuple[int | torch.SymInt, ...],
    global_offsets: tuple[tuple[int | torch.SymInt, ...], ...],
    local_offsets: tuple[tuple[int | torch.SymInt, ...], ...],
    local_sizes: tuple[tuple[int | torch.SymInt, ...], ...],
) -> tuple[
    int,
    list[int | torch.SymInt],
    list[int | torch.SymInt],
    list[int | torch.SymInt],
]:
    rank = len(global_shape)
    chunk_count = len(global_offsets)

    def flatten(
        name: str,
        chunks: tuple[tuple[int | torch.SymInt, ...], ...],
    ) -> list[int | torch.SymInt]:
        if len(chunks) != chunk_count:
            raise ValueError(f"global_offsets and {name} must have the same length")
        if any(len(chunk) != rank for chunk in chunks):
            raise ValueError(f"each {name} entry must have {rank} dimensions")
        return [value for chunk in chunks for value in chunk]

    return (
        chunk_count,
        flatten("global_offsets", global_offsets),
        flatten("local_offsets", local_offsets),
        flatten("local_sizes", local_sizes),
    )


@maybe_run_for_local_tensor
def _run_stateful_rng_op_rankwise(
    tensor: torch.Tensor,
    global_shape: list[int | torch.SymInt],
    global_offsets: list[int | torch.SymInt],
    local_offsets: list[int | torch.SymInt],
    local_sizes: list[int | torch.SymInt],
    chunk_count: int,
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
    return aten._philox_distribution_shards_.default(
        tensor,
        global_shape,
        global_offsets,
        local_offsets,
        local_sizes,
        chunk_count,
        kind,
        params,
        generator=generator,
    )


def _run_stateful_rng_op(
    op_call: torch._ops.OpOverload,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
    global_shape: tuple[int | torch.SymInt, ...],
    global_offsets: tuple[tuple[int | torch.SymInt, ...], ...],
    local_offsets: tuple[tuple[int | torch.SymInt, ...], ...],
    local_sizes: tuple[tuple[int | torch.SymInt, ...], ...],
) -> torch.Tensor:
    """Run one logical CUDA draw for rectangular local tensor shards."""
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
    if tensor.layout != torch.strided:
        raise RuntimeError(
            f"{op_call}: expected a strided local tensor, got layout {tensor.layout}"
        )

    # Validate before entering the rankwise helper or reserving generator state.
    kind, validate = _STATEFUL_RNG_OP_SPECS[op_call]
    validate(tensor, op_args)
    params = tuple(op_args)
    chunk_count, flat_global_offsets, flat_local_offsets, flat_local_sizes = (
        _flatten_shard_metadata(
            global_shape,
            global_offsets,
            local_offsets,
            local_sizes,
        )
    )
    generator_state = (
        generator.get_state()
        if generator is not None and enabled_local_tensor_mode()
        else None
    )
    if generator_state is None:
        aten._philox_distribution_shards_.default(
            tensor,
            global_shape,
            flat_global_offsets,
            flat_local_offsets,
            flat_local_sizes,
            chunk_count,
            kind,
            params,
            generator=generator,
        )
    else:
        _run_stateful_rng_op_rankwise(
            tensor,
            list(global_shape),
            flat_global_offsets,
            flat_local_offsets,
            flat_local_sizes,
            chunk_count,
            kind,
            generator,
            generator_state,
            params,
        )
    return tensor


def _run_stateful_rng_op_for_checkpointable_tensor(
    op_call: torch._ops.OpOverload,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
    tensor: CheckpointableTensor,
) -> torch.Tensor:
    """Run one dense-equivalent RNG draw using checkpoint shard metadata."""
    local_tensor = cast(torch.Tensor, tensor)
    if not args or args[0] is not local_tensor:
        raise AssertionError("Expected the checkpointable tensor as the first argument")

    return _run_stateful_rng_op(
        op_call,
        args,
        kwargs,
        tensor.global_shape,
        tensor.global_offsets,
        tensor.local_offsets,
        tensor.local_sizes,
    )
