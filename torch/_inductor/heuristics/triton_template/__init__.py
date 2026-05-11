"""
Triton template heuristics for pre-written template kernels (mm, conv, flex_attention...).

These heuristics are used at compile time to generate autotuning configs
for template-based Triton kernels.
"""

from __future__ import annotations

import torch

# NOTE: add new template heuristics here, so they get imported and registered
from . import (
    aten,
    base,
    contiguous_mm,
    decompose_k,
    nv_universal_gemm,
    registry,
    tlx,
    triton,
)

# expose the entry function
from .registry import get_template_heuristic
from .triton import (
    BaseConfigHeuristic,
    CPUConfigHeuristic,
    CUDAConfigHeuristic,
    MTIAConfigHeuristic,
    ROCmConfigHeuristic,
    XPUConfigHeuristic,
)


def get_config_heuristic_for_device(device_type: str | None) -> BaseConfigHeuristic:
    """Return the device-specific template config heuristic instance."""
    if device_type == "cuda":
        if torch.version.hip is None:
            return CUDAConfigHeuristic()
        else:
            return ROCmConfigHeuristic()
    elif device_type == "xpu":
        return XPUConfigHeuristic()
    elif device_type == "cpu":
        return CPUConfigHeuristic()
    elif device_type == "mtia":
        return MTIAConfigHeuristic()
    else:
        return BaseConfigHeuristic()


__all__ = [
    "get_config_heuristic_for_device",
    "get_template_heuristic",
    "BaseConfigHeuristic",
    "CPUConfigHeuristic",
    "CUDAConfigHeuristic",
    "MTIAConfigHeuristic",
    "ROCmConfigHeuristic",
    "XPUConfigHeuristic",
]
