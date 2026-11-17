r"""Host launch geometry and state for ATen-compatible Philox distributions."""

from __future__ import annotations

import functools

import torch

from ...cutedsl.config import RNG_BLOCK as _BLOCK
from ...cutedsl.hw_caps import caps


@functools.cache
def _device_launch_caps(device_index: int) -> tuple[int, int]:
    hw = caps(device_index)
    return hw.sm_count, hw.blocks_per_sm(_BLOCK)


def launch_plan(n: int, device_index: int) -> tuple[int, int, int]:
    r"""Match ATen's grid, iteration count and Philox reservation."""
    sm_count, blocks_per_sm = _device_launch_caps(device_index)
    grid = min((n + _BLOCK - 1) // _BLOCK, sm_count * blocks_per_sm)
    # Each curand4 call reserves four offsets regardless of unrolling.
    num_iters = (n - 1) // (_BLOCK * grid * 4) + 1
    return grid, num_iters * 4, num_iters


def philox_args(
    device_index: int, counter_offset: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    r"""Reserve Philox state; capture returns device-backed seed and offset."""
    gen = torch.cuda.default_generators[device_index]
    seed_t, offset_t, intra_t = gen.philox_state(counter_offset)
    return seed_t, offset_t, int(intra_t.item())


@functools.cache
def state_dummy(device_index: int) -> torch.Tensor:
    r"""Return an unused device placeholder for eager seed/offset slots."""
    return torch.zeros(1, dtype=torch.int64, device=f"cuda:{device_index}")
