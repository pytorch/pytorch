# mypy: allow-untyped-defs
"""
Utility functions for NVIDIA Universal GEMM.
"""

from typing import Any

from torch.nn.functional import ScalingType, SwizzleType


def to_cutlass_scale_mode(
    scale_type: Any, swizzle_type: Any
) -> tuple[Any | None, Any | None]:
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
