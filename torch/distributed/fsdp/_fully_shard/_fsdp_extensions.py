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
    all_gather_outputs: Sequence[torch.Tensor] = (),
) -> tuple[list[torch.Tensor], tuple[_AllGatherOutputLayout, ...]]:
    legacy_numels = [inp.numel() for inp in inputs if isinstance(inp, torch.Tensor)]
    legacy_dim = shard_dim if world_size > 1 and any(legacy_numels) else 0
    padded_numel = padded_sharded_size.numel()
    validate_legacy_outputs = (
        legacy_dim != 0
        and any(numel != padded_numel for numel in legacy_numels)
        and (not all_gather_outputs or len(all_gather_outputs) == len(inputs))
    )
    legacy_prefixes = math.prod(padded_sharded_size[:legacy_dim]) if legacy_dim else 1
    tensors: list[torch.Tensor] = []
    layouts: list[_AllGatherOutputLayout] = []
    for i, inp in enumerate(inputs):
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
            input_numel = tensor.numel()
            if validate_legacy_outputs:
                output_numel = (
                    all_gather_outputs[i].numel()
                    if all_gather_outputs
                    else input_numel * world_size
                )
                expected_numel = padded_numel * world_size
                if output_numel != expected_numel and (
                    input_numel != 0 or output_numel != 0
                ):
                    raise RuntimeError(
                        f"Shard({shard_dim}) all-gather output must have "
                        f"{expected_numel} elements for padded local size "
                        f"{padded_sharded_size} and world size "
                        f"{world_size}, but got {output_numel}"
                    )
            # Legacy hooks reshape cached outputs using the current input's tail.
            output_size = torch.Size((-1, *input_size[1:]))
            if 0 in input_size[1:]:
                output_size = torch.Size((input_size[0] * world_size, *input_size[1:]))
            dim = legacy_dim if input_numel else 0
            layout = _AllGatherOutputLayout(
                padded_sharded_size if dim else torch.Size(input_size or (1,)),
                dim,
                output_size,
                legacy_prefixes if dim else 1,
            )
        tensors.append(tensor)
        layouts.append(layout)
    return tensors, tuple(layouts)
