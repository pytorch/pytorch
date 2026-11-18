"""Launch policy for TMA-based block-scaled quantization kernels."""

from typing import NamedTuple

import torch


_QUANT_ORIENTATION_DIM_K = 0
_QUANT_ORIENTATION_DIM_M = 1
_QUANT_ORIENTATION_DIM_KM = 2

_DIM_K_TILE_M_SIZE_128 = 128
_DIM_K_MAX_TILE_K_SIZE_128 = 128
_DIM_M_KM_SMALL_TILE_32_128 = (32, 128)
_DIM_M_LARGE_TILE_64_256 = (64, 256)
_DIM_M_FLOAT32_LARGE_TILE_64_128 = (64, 128)
_DIM_KM_LARGE_TILE_64_128 = (64, 128)


class BlockscaledTmaPlan(NamedTuple):
    """Compile-time launch parameters and runtime grid for one invocation."""

    tile_m_size: int
    tile_k_size: int
    cluster_k: int
    needs_boundary_masking: bool
    grid_m: int
    grid_k: int


def _ceil_div(num: int, den: int) -> int:
    return (num + den - 1) // den


def select_blockscaled_tma_plan(
    M: int,
    K: int,
    *,
    input_dtype: torch.dtype,
    quant_orientation_id: int,
    is_square_scaling: bool,
    quant_block_size: tuple[int, int],
) -> BlockscaledTmaPlan:
    """Select the existing B200-tuned tile, cluster, masking, and grid policy."""
    if quant_block_size != (1, 32):
        raise AssertionError(f"unsupported quantization block size: {quant_block_size}")
    if quant_orientation_id not in (
        _QUANT_ORIENTATION_DIM_K,
        _QUANT_ORIENTATION_DIM_M,
        _QUANT_ORIENTATION_DIM_KM,
    ):
        raise ValueError(
            f"unsupported quantization orientation: {quant_orientation_id}"
        )
    do_dim_k = quant_orientation_id != _QUANT_ORIENTATION_DIM_M
    do_dim_m = quant_orientation_id != _QUANT_ORIENTATION_DIM_K
    scale_group_size = quant_block_size[1]

    # The unmasked kernel assumes both matrix dimensions are divisible by 128.
    padded_M = _ceil_div(M, 128) * 128
    padded_K = _ceil_div(K, 128) * 128
    needs_boundary_masking = M != padded_M or K != padded_K

    if do_dim_k and not do_dim_m:
        s_num_col_blk_k = _ceil_div(K // scale_group_size, 4)
        # First choose the original adaptive K width. For small problems that would use K=64,
        # rotate the same-size 128x64 tile to 32x128: it keeps 128 one-group threads but gives TMA
        # contiguous rows and exposes more M-parallel CTAs.
        num_128_tiles = _ceil_div(M, _DIM_K_TILE_M_SIZE_128) * _ceil_div(
            K, _DIM_K_MAX_TILE_K_SIZE_128
        )
        if num_128_tiles <= 64:
            tile_k_size = 32
        elif num_128_tiles <= 512:
            tile_k_size = 64
        else:
            tile_k_size = 128

        # Do not compute padding merely to reach the selected width for very narrow matrices.
        if K <= 32:
            tile_k_size = 32
        elif K <= 64:
            tile_k_size = min(tile_k_size, 64)
        if M * K <= 2048 * 2048 and tile_k_size >= 64:
            tile_m_size, tile_k_size = 32, 128
        else:
            tile_m_size = 128
        grid_k = _ceil_div(K, tile_k_size)
        cluster_k = (
            # Square scaling has enough independent CTAs at small/medium sizes that clustering only
            # constrains scheduling; large 16-bit shapes retain v2's K-locality-oriented clusters.
            # FP32's larger shared-memory tiles reduce residency enough that clustering loses.
            1
            if input_dtype == torch.float32
            or (is_square_scaling and M * K <= 4096 * 4096)
            else next(
                c for c in (16, 8, 4, 2, 1) if c <= s_num_col_blk_k and grid_k % c == 0
            )
        )
    elif do_dim_m and not do_dim_k:
        tile_m_size, tile_k_size = (
            _DIM_M_KM_SMALL_TILE_32_128
            if M * K <= 2048 * 2048
            else (
                _DIM_M_FLOAT32_LARGE_TILE_64_128
                if input_dtype == torch.float32
                else _DIM_M_LARGE_TILE_64_256
            )
        )
        tile_padded_K = _ceil_div(K, tile_k_size) * tile_k_size
        grid_m = padded_M // tile_m_size
        grid_k = tile_padded_K // tile_k_size
        cluster_k = (
            next(c for c in (16, 8, 4, 2, 1) if c <= grid_k and grid_k % c == 0)
            if grid_m * grid_k <= 512
            else 1
        )
        needs_boundary_masking = needs_boundary_masking or K != tile_padded_K
    else:
        tile_m_size, tile_k_size = (
            _DIM_M_KM_SMALL_TILE_32_128
            if M * K <= 2048 * 2048
            else _DIM_KM_LARGE_TILE_64_128
        )
        cluster_k = 1

    grid_k = _ceil_div(K, tile_k_size)
    grid_m = _ceil_div(M, 128) * (128 // tile_m_size)

    return BlockscaledTmaPlan(
        tile_m_size,
        tile_k_size,
        cluster_k,
        needs_boundary_masking,
        grid_m,
        grid_k,
    )
