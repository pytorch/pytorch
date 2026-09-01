"""Eligibility check and execution for the SM90 DeepSeek FP8 grouped mm."""

import functools

import torch

from ._common import BLOCKWISE_128X128, BLOCKWISE_1X128, ceil_div, NO_SWIZZLE, round_up


_DEEPSEEK_RECIPES = {
    (BLOCKWISE_1X128, BLOCKWISE_1X128),
    (BLOCKWISE_1X128, BLOCKWISE_128X128),
    (BLOCKWISE_128X128, BLOCKWISE_1X128),
}


@functools.cache
def _sm90_only(device: int) -> bool:
    major, _ = torch.cuda.get_device_capability(device)
    return major == 9


def _expected_scale_b_shape(
    recipe_b: int, group_count: int, k: int, n: int, b_is_2d: bool = False
) -> tuple[int, ...]:
    if recipe_b == BLOCKWISE_1X128:
        inner: tuple[int, ...] = (n, ceil_div(k, 128))
    else:
        inner = (round_up(ceil_div(k, 128), 4), ceil_div(n, 128))
    return inner if b_is_2d else (group_count, *inner)


def _expected_scale_a_shape(recipe_a: int, total_m: int, k: int) -> tuple[int, int]:
    if recipe_a == BLOCKWISE_1X128:
        return (total_m, ceil_div(k, 128))
    return (round_up(ceil_div(k, 128), 4), ceil_div(total_m, 128))


def _valid_blockwise_scale_strides(
    scale: torch.Tensor, unit_dim: int, outer_dim: int, expected_outer_stride: int
) -> bool:
    if scale.stride(unit_dim) != 1:
        return False
    if scale.stride(outer_dim) != expected_outer_stride and not (
        scale.size(outer_dim) == 1 and scale.stride(outer_dim) == 1
    ):
        return False
    return True


def _should_use_cutedsl_scaled_grouped_mm_deepseek(
    self: torch.Tensor,
    mat2: torch.Tensor,
    scale_a,
    recipe_a,
    swizzle_a,
    scale_b,
    recipe_b,
    swizzle_b,
    offs,
    bias,
    out_dtype,
    contraction_dim,
    use_fast_accum,
) -> bool:
    if not self.is_cuda:
        return False
    if self.dtype != torch.float8_e4m3fn or mat2.dtype != torch.float8_e4m3fn:
        return False
    if self.dim() != 2 or mat2.dim() not in (2, 3):
        return False
    if bias is not None:
        return False
    if out_dtype not in (None, torch.bfloat16):
        return False
    if len(contraction_dim) != 0:
        return False
    if use_fast_accum:
        return False
    if (
        offs is None
        or offs.dtype != torch.int32
        or offs.dim() != 1
        or offs.device != self.device
        or offs.stride(0) != 1
    ):
        return False
    if (
        len(scale_a) != 1
        or len(recipe_a) != 1
        or len(scale_b) != 1
        or len(recipe_b) != 1
    ):
        return False
    if len(swizzle_a) != 1 or len(swizzle_b) != 1:
        return False
    recipe_a0 = recipe_a[0]
    recipe_b0 = recipe_b[0]
    if swizzle_a[0] != NO_SWIZZLE or swizzle_b[0] != NO_SWIZZLE:
        return False
    if (recipe_a0, recipe_b0) not in _DEEPSEEK_RECIPES:
        return False
    if not _sm90_only(self.device.index or 0):
        return False
    scale_a0 = scale_a[0]
    scale_b0 = scale_b[0]
    if scale_a0.dtype != torch.float32 or scale_b0.dtype != torch.float32:
        return False
    if (
        mat2.device != self.device
        or scale_a0.device != self.device
        or scale_b0.device != self.device
    ):
        return False
    if self.data_ptr() % 16 != 0 or mat2.data_ptr() % 16 != 0:
        return False

    total_m, k = self.shape
    b_is_2d = mat2.dim() == 2
    group_count = offs.shape[0]
    if b_is_2d:
        k2, n = mat2.shape
    else:
        b_groups, k2, n = mat2.shape
        if b_groups != group_count:
            return False
    if k2 != k:
        return False
    if k % 128 != 0:
        return False
    self_stride0 = self.stride(0)
    if self.stride(1) != 1 or self_stride0 < max(1, k) or self_stride0 % 16 != 0:
        return False
    if b_is_2d:
        if (
            mat2.stride(0) != 1
            or mat2.stride(1) < max(1, k)
            or mat2.stride(1) % 16 != 0
        ):
            return False
    else:
        mat2_stride0, mat2_stride2 = mat2.stride(0), mat2.stride(2)
        if (
            mat2.stride(1) != 1
            or mat2_stride2 < max(1, k)
            or mat2_stride0 % 16 != 0
            or mat2_stride2 % 16 != 0
        ):
            return False
    if scale_a0.shape != _expected_scale_a_shape(recipe_a0, total_m, k):
        return False
    expected_a_outer_stride = (
        total_m if recipe_a0 == BLOCKWISE_1X128 else scale_a0.size(0)
    )
    if not _valid_blockwise_scale_strides(scale_a0, 0, 1, expected_a_outer_stride):
        return False
    # At offset 0 alignment rounding can only move forward, so row/col 0 is
    # only covered if these strides are already 4-aligned. A single k-block
    # never multiplies the stride, so it is exempt.
    if (
        recipe_a0 == BLOCKWISE_1X128
        and scale_a0.size(1) > 1
        and scale_a0.stride(1) % 4 != 0
    ):
        return False
    if recipe_a0 == BLOCKWISE_1X128 and scale_a0.data_ptr() % 16 != 0:
        return False
    if scale_b0.shape != _expected_scale_b_shape(recipe_b0, group_count, k, n, b_is_2d):
        return False
    b_unit_dim = 0 if b_is_2d else 1
    expected_b_outer_stride = (
        n if recipe_b0 == BLOCKWISE_1X128 else scale_b0.size(b_unit_dim)
    )
    if not _valid_blockwise_scale_strides(
        scale_b0, b_unit_dim, b_unit_dim + 1, expected_b_outer_stride
    ):
        return False
    if (
        recipe_b0 == BLOCKWISE_1X128
        and scale_b0.size(b_unit_dim + 1) > 1
        and scale_b0.stride(b_unit_dim + 1) % 4 != 0
    ):
        return False
    if recipe_b0 == BLOCKWISE_1X128 and not b_is_2d and scale_b0.stride(0) % 4 != 0:
        return False
    if recipe_b0 == BLOCKWISE_1X128 and scale_b0.data_ptr() % 16 != 0:
        return False
    return True
