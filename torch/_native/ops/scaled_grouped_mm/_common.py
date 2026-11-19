"""Shared helpers for DeepSeek grouped mm."""

from torch._meta_registrations import round_up
from torch.nn.functional import ScalingType, SwizzleType


BLOCKWISE_1X128 = ScalingType.BlockWise1x128.value
BLOCKWISE_128X128 = ScalingType.BlockWise128x128.value
NO_SWIZZLE = SwizzleType.NO_SWIZZLE.value


# cp.async.bulk needs 16-byte-aligned addresses, so a copy may start up to
# align-1 elements before the tile.
BULK_COPY_ALIGN_BYTES = 16
FP32_SCALE_COPY_ALIGN_ELEMS = BULK_COPY_ALIGN_BYTES // 4


def fp32_scale_stage_size(tile_size: int) -> int:
    return round_up(
        tile_size + FP32_SCALE_COPY_ALIGN_ELEMS - 1, FP32_SCALE_COPY_ALIGN_ELEMS
    )
