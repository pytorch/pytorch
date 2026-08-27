"""Shared helpers for DeepSeek grouped mm."""

import torch


# Ordinal values of at::blas::ScalingType.
BLOCKWISE_1X128 = 4
BLOCKWISE_128X128 = 5
# at::blas::SwizzleType::NO_SWIZZLE.
NO_SWIZZLE = 0


def any_cow(*tensors: torch.Tensor) -> bool:
    return any(
        torch._C._is_cow_tensor(t)  # pyrefly: ignore[missing-attribute]
        for t in tensors
    )


def _make_fake_1d_tensor(dtype):
    import cutlass.cute as cute

    return cute.runtime.make_fake_tensor(dtype, (cute.sym_int(),), stride=(1,))


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def round_up(a: int, b: int) -> int:
    return ceil_div(a, b) * b


# CopyBulkG2SOp needs 16-byte-aligned addresses; Float32 scale copies must
# start on a multiple of 4 elements, widening the tile by up to align-1.
SCALE_BULK_COPY_ALIGN = 4


def scale_stage_size(tile_size: int) -> int:
    return round_up(tile_size + SCALE_BULK_COPY_ALIGN - 1, SCALE_BULK_COPY_ALIGN)
