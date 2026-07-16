# Copyright (c) Meta Platforms, Inc. and affiliates
"""Stateful RNG support for local views of logical tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast, Protocol, runtime_checkable, TypeGuard

import torch
from torch._library.utils import fill_defaults
from torch.utils._python_dispatch import TorchDispatchMode


aten = torch.ops.aten

__all__ = [
    "RNGIndexBlock",
    "StatefulRNGMode",
    "StatefulRNGTensor",
    "set_stateful_rng_metadata",
]


@dataclass(frozen=True)
class RNGIndexBlock:
    start_index: int
    block_size: int
    block_stride: int
    num_blocks: int


@runtime_checkable
class StatefulRNGTensor(Protocol):
    rng_global_numel: int
    rng_index_blocks: tuple[RNGIndexBlock, ...]


def _is_stateful_rng_tensor(obj: object) -> TypeGuard[StatefulRNGTensor]:
    return isinstance(obj, torch.Tensor) and isinstance(obj, StatefulRNGTensor)


def set_stateful_rng_metadata(
    tensor: torch.Tensor,
    global_numel: int,
    index_blocks: tuple[RNGIndexBlock, ...],
) -> None:
    setattr(tensor, "rng_global_numel", global_numel)  # noqa: B010
    setattr(tensor, "rng_index_blocks", index_blocks)  # noqa: B010


def _run_stateful_rng_op(
    tensor: torch.Tensor,
    global_numel: int,
    index_blocks: tuple[RNGIndexBlock, ...],
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
        [block.start_index for block in index_blocks],
        [block.block_size for block in index_blocks],
        [block.block_stride for block in index_blocks],
        [block.num_blocks for block in index_blocks],
        *op_args,
        generator=generator,
    )


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

        filled_args, filled_kwargs = fill_defaults(func._schema, args, kwargs)
        tensor_arg = filled_args[0]
        if not isinstance(tensor_arg, torch.Tensor):
            return func(*args, **kwargs)
        if tensor_arg.is_meta or tensor_arg.device.type != "cuda":
            return func(*args, **kwargs)
        if not _is_stateful_rng_tensor(tensor_arg):
            return func(*args, **kwargs)
        rng_metadata = cast(StatefulRNGTensor, tensor_arg)

        tensor, *op_args = filled_args
        generator = filled_kwargs.pop("generator")
        if filled_kwargs:
            raise AssertionError
        if not isinstance(tensor, torch.Tensor):
            raise AssertionError
        if not (generator is None or isinstance(generator, torch.Generator)):
            raise AssertionError

        flat_slice_op_call = (
            aten._philox_normal_flat_slice_.default
            if func is aten.normal_.default
            else aten._philox_uniform_flat_slice_.default
        )
        _run_stateful_rng_op(
            tensor,
            rng_metadata.rng_global_numel,
            rng_metadata.rng_index_blocks,
            flat_slice_op_call,
            generator,
            *op_args,
        )
        return tensor
