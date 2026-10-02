"""Shared helpers for DeepSeek grouped mm."""

import torch
from torch.nn.functional import ScalingType, SwizzleType


BLOCKWISE_1X128 = ScalingType.BlockWise1x128.value
BLOCKWISE_128X128 = ScalingType.BlockWise128x128.value
NO_SWIZZLE = SwizzleType.NO_SWIZZLE.value


def read_only(tensor: torch.Tensor) -> torch.Tensor:
    """Export-const view; COW is not materialized. Wrap only at the call
    site: the wrapper raises on non-DLPack torch ops."""
    from torch.utils.dlpack import ReadOnlyTensorWrapper

    return ReadOnlyTensorWrapper(tensor)


def _make_fake_1d_tensor(dtype):
    import cutlass.cute as cute

    return cute.runtime.make_fake_tensor(dtype, (cute.sym_int(),), stride=(1,))


def _make_fake_2d_tensor(dtype, cols: int):
    import cutlass.cute as cute

    return cute.runtime.make_fake_tensor(
        dtype, (cute.sym_int(), cols), stride=(cols, 1)
    )


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def round_up(a: int, b: int) -> int:
    return ceil_div(a, b) * b


# cp.async.bulk needs 16-byte-aligned addresses, so a copy may start up to
# align-1 elements before the tile.
BULK_COPY_ALIGN_BYTES = 16
FP32_SCALE_COPY_ALIGN_ELEMS = BULK_COPY_ALIGN_BYTES // 4


def fp32_scale_stage_size(tile_size: int) -> int:
    return round_up(
        tile_size + FP32_SCALE_COPY_ALIGN_ELEMS - 1, FP32_SCALE_COPY_ALIGN_ELEMS
    )
