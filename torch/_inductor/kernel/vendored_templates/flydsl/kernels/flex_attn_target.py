# mypy: allow-untyped-defs
"""The device limits FlyDSL flex kernels are planned against.

The kernels use gfx950 MFMA and LDS instructions laid out for 64-lane waves.
The LDS budget and CU count come from the device, so lowering and kernel
builds size tiles and grids for the GPU they run on.
"""

import functools
from typing import NamedTuple


FLEX_ARCH = "gfx950"
WAVE_SIZE = 64


class TargetLimits(NamedTuple):
    # LDS bytes one workgroup may allocate.
    lds_bytes: int
    # Compute units the grid heuristics try to fill.
    num_cus: int


# Planning without a gfx950 device, e.g. compile-only builds and CPU tests.
GFX950_LIMITS = TargetLimits(lds_bytes=160 << 10, num_cus=256)


@functools.cache
def _device_limits(device_index: int) -> TargetLimits:
    import torch

    props = torch.cuda.get_device_properties(device_index)
    if str(getattr(props, "gcnArchName", "")).split(":", 1)[0] != FLEX_ARCH:
        return GFX950_LIMITS
    return TargetLimits(props.shared_memory_per_block, props.multi_processor_count)


def target_limits(device=None) -> TargetLimits:
    """The limits of ``device`` (default: the current device), or
    ``GFX950_LIMITS`` when it is not a gfx950 GPU."""
    import torch

    if torch.version.hip is None or not torch.cuda.is_available():
        return GFX950_LIMITS
    index = None if device is None else torch.device(device).index
    return _device_limits(torch.cuda.current_device() if index is None else index)


def resolve_target(target=None) -> TargetLimits:
    """``target`` as ``TargetLimits``; ``None`` reads the current device."""
    return target_limits() if target is None else TargetLimits(*target)
