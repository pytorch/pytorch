"""Shared utilities for learned reduction heuristics.

Used by both the FB-internal logging module and the OSS inference path.
"""

from __future__ import annotations

import functools
import math

import torch


@functools.lru_cache(None)
def get_gpu_family() -> str:
    """Return a canonical GPU architecture string.

    NVIDIA GPUs are identified by compute capability (e.g. ``"sm90"``).
    AMD GPUs use the GCN architecture name.
    """
    assert torch.cuda.is_available(), "get_gpu_family requires a GPU"
    if torch.version.hip:
        return torch.cuda.get_device_properties(0).gcnArchName
    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


def _noop_autotune_log(autotuner: object, timings: object) -> None:
    pass


try:
    from torch._inductor.fb.reduction_autotune_logging import (  # type: ignore[import-not-found]
        enqueue_autotune_log,
    )
except ImportError:
    enqueue_autotune_log = _noop_autotune_log


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def compute_derived_features(
    xnumel: int,
    ynumel: int,
    xblock: int,
    yblock: int,
    grid_size: int,
    num_sms: int,
) -> dict[str, float]:
    """Compute tile/wave quantization features from problem and config dims.

    These features capture how well a config tiles the problem space and
    utilises the GPU's streaming multiprocessors.  The same computation is
    used during training (on Hive data) and inference (in the scoring path)
    so the model sees identical features.
    """
    tile_utilization_x = (
        xnumel / (_ceildiv(xnumel, xblock) * xblock) if xblock > 0 else 1.0
    )
    tile_utilization_y = (
        ynumel / (_ceildiv(ynumel, yblock) * yblock)
        if ynumel > 0 and yblock > 0
        else 1.0
    )
    num_waves = _ceildiv(grid_size, num_sms) if num_sms > 0 else 1
    wave_utilization = (
        grid_size / (num_waves * num_sms) if num_sms > 0 and num_waves > 0 else 1.0
    )

    return {
        "tile_utilization_x": tile_utilization_x,
        "tile_utilization_y": tile_utilization_y,
        "grid_size": float(grid_size),
        "wave_utilization": wave_utilization,
        "num_waves": float(num_waves),
        "log_xnumel": math.log1p(xnumel),
        "log_ynumel": math.log1p(ynumel),
    }
