import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ._fsdp_api import AllGatherInput


@dataclass(frozen=True)
class _AllGatherOutputLayout:
    input_size: torch.Size
    dim: int
    output_size: torch.Size
    num_prefixes: int


def _get_all_gather_output_layout(
    input_size: torch.Size,
    dim: int,
    world_size: int,
    output_size: torch.Size | None = None,
) -> _AllGatherOutputLayout:
    input_size = torch.Size(input_size or (1,))
    ndim = len(input_size)
    if not -ndim <= dim < ndim:
        raise ValueError(f"All-gather dim {dim} is invalid for input size {input_size}")
    if world_size <= 0:
        raise ValueError(f"All-gather world size must be positive, got {world_size}")
    dim %= ndim
    gathered_size = list(input_size)
    gathered_size[dim] *= world_size
    output_size = torch.Size(gathered_size if output_size is None else output_size)
    if any(size < 0 for size in output_size):
        raise ValueError(
            f"All-gather output size must be nonnegative, got {output_size}"
        )
    input_numel = input_size.numel()
    if output_size.numel() != input_numel * world_size:
        raise ValueError(
            f"All-gather output size {output_size} must contain "
            f"{input_numel * world_size} elements"
        )
    num_prefixes = math.prod(input_size[:dim]) if input_numel else 1
    return _AllGatherOutputLayout(input_size, dim, output_size, num_prefixes)


def _normalize_all_gather_inputs(
    inputs: Sequence[torch.Tensor | AllGatherInput],
    *,
    world_size: int,
    shard_dim: int,
    padded_sharded_size: torch.Size,
    require_padding: bool,
) -> tuple[list[torch.Tensor], tuple[_AllGatherOutputLayout, ...]]:
    if len(inputs) == 0:
        raise ValueError(
            "fsdp_pre_all_gather must return at least one all-gather input"
        )
    tensors: list[torch.Tensor] = []
    layouts: list[_AllGatherOutputLayout] = []
    for inp in inputs:
        tensor = inp.tensor if isinstance(inp, AllGatherInput) else inp
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Expected an all-gather input Tensor, got {type(tensor).__name__}"
            )
        input_size = tensor.size()
        if isinstance(inp, AllGatherInput):
            layout = _get_all_gather_output_layout(
                input_size, inp.dim, world_size, inp.output_size
            )
        else:
            if require_padding and input_size != padded_sharded_size:
                raise AssertionError(
                    "fsdp_pre_all_gather must return all-gather inputs with the padded sharded size "
                    f"{padded_sharded_size} but got {input_size}"
                )
            output_size = list(input_size or (1,))
            output_size[0] *= world_size
            dim = shard_dim if world_size > 1 and tensor.numel() else 0
            if dim != 0:
                if tensor.numel() != padded_sharded_size.numel():
                    raise ValueError(
                        f"Legacy Shard({shard_dim}) all-gather input size {input_size} "
                        f"must have the same number of elements as {padded_sharded_size}"
                    )
                input_size = padded_sharded_size
            # Legacy hooks receive a dim-0-expanded view after parameter-layout copying.
            layout = _get_all_gather_output_layout(
                input_size, dim, world_size, torch.Size(output_size)
            )
        tensors.append(tensor)
        layouts.append(layout)
    return tensors, tuple(layouts)
