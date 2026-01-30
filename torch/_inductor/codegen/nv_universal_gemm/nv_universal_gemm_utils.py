# mypy: allow-untyped-defs
"""
Utility functions for NVIDIA Universal GEMM.
"""

from functools import lru_cache
from typing import Any, Optional

import torch
from torch.nn.functional import ScalingType, SwizzleType


@lru_cache(maxsize=1)
def get_nvmatmul_gpu_enum():
    """
    Detect the current GPU and return the corresponding nvMatmulHeuristics enum.
    Returns None if GPU cannot be detected or is not in the predefined list.
    """
    if not torch.cuda.is_available():
        return None

    try:
        import nvMatmulHeuristics
    except ImportError:
        return None

    device_name = torch.cuda.get_device_name(0).upper()

    # Map device names to nvMatmulHeuristics GPU enums
    # Priority: check specific models first, then fallback to architecture
    gpu_mapping = {
        # Blackwell
        "B200": nvMatmulHeuristics.NvMatmulHeuristicsNvidiaGpu.B200,
        "GB200": nvMatmulHeuristics.NvMatmulHeuristicsNvidiaGpu.GB200_NVL,
        "GB300": nvMatmulHeuristics.NvMatmulHeuristicsNvidiaGpu.GB300_NVL,
        "RTX 5090": nvMatmulHeuristics.NvMatmulHeuristicsNvidiaGpu.RTX_5090,
        "RTX 5080": nvMatmulHeuristics.NvMatmulHeuristicsNvidiaGpu.RTX_5080,
        # Hopper
        "H100 SXM": nvMatmulHeuristics.NvMatmulHeuristicsNvidiaGpu.H100_SXM,
        "H100 PCIE": nvMatmulHeuristics.NvMatmulHeuristicsNvidiaGpu.H100_PCIE,
        "H100 NVL": nvMatmulHeuristics.NvMatmulHeuristicsNvidiaGpu.H100_NVL,
        "H200": nvMatmulHeuristics.NvMatmulHeuristicsNvidiaGpu.H200_SXM,
        "H20": nvMatmulHeuristics.NvMatmulHeuristicsNvidiaGpu.H20_SXM,
        # Ada
        "L40S": nvMatmulHeuristics.NvMatmulHeuristicsNvidiaGpu.L40S,
        "L40": nvMatmulHeuristics.NvMatmulHeuristicsNvidiaGpu.L40,
        "L20": nvMatmulHeuristics.NvMatmulHeuristicsNvidiaGpu.L20,
        "L4": nvMatmulHeuristics.NvMatmulHeuristicsNvidiaGpu.L4,
        "RTX 4090": nvMatmulHeuristics.NvMatmulHeuristicsNvidiaGpu.RTX_4090,
        # Ampere
        "A100 SXM": nvMatmulHeuristics.NvMatmulHeuristicsNvidiaGpu.A100_SXM_80GB,
        "A100 PCIE": nvMatmulHeuristics.NvMatmulHeuristicsNvidiaGpu.A100_PCIE_80GB,
        "A100": nvMatmulHeuristics.NvMatmulHeuristicsNvidiaGpu.A100_SXM_80GB,
        "A40": nvMatmulHeuristics.NvMatmulHeuristicsNvidiaGpu.A40_PCIE,
        "A30": nvMatmulHeuristics.NvMatmulHeuristicsNvidiaGpu.A30_PCIE,
        "A10": nvMatmulHeuristics.NvMatmulHeuristicsNvidiaGpu.A10_PCIE,
        "RTX 3090": nvMatmulHeuristics.NvMatmulHeuristicsNvidiaGpu.RTX_3090,
    }

    for pattern, gpu_enum in gpu_mapping.items():
        if pattern in device_name:
            return gpu_enum

    return None


def to_cutlass_scale_mode(
    scale_type: Any, swizzle_type: Any
) -> tuple[Optional[Any], Optional[Any]]:
    """
    Map PyTorch ScalingType/SwizzleType to cutlass_api ScaleMode/ScaleSwizzleMode.

    Args:
        scale_type: ScalingType from torch.nn.functional
        swizzle_type: SwizzleType from torch.nn.functional

    Returns:
        Tuple of (ScaleMode, ScaleSwizzleMode) from cutlass_api.library,
        or (None, None) if the types are not supported.

    The returned enum objects can be used directly with cutlass_api, or their
    .name attribute can be used for codegen (e.g., scale_mode.name -> "Blockwise1x32").

    NOTE:
        Currently on Blackwell (SM100), NVGEMM only supports MXFP8 scaling modes.
        Update this mapping when additional scaling modes are added.
    """
    from cutlass_api.library import ScaleMode, ScaleSwizzleMode

    scale_mode_map = {
        ScalingType.BlockWise1x32: ScaleMode.Blockwise1x32,
        ScalingType.BlockWise1x16: ScaleMode.Blockwise1x16,
    }
    swizzle_mode_map = {
        SwizzleType.SWIZZLE_32_4_4: ScaleSwizzleMode.Swizzle32x4x4,
        SwizzleType.NO_SWIZZLE: ScaleSwizzleMode.SwizzleNone,
    }
    return scale_mode_map.get(scale_type), swizzle_mode_map.get(swizzle_type)
