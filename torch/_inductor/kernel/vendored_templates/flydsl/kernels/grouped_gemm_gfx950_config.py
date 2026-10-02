"""Pure-Python gfx950 grouped GEMM configuration helpers."""

from typing import Any


GFX950_DMA_BYTES = 16
GFX950_WAVE_SIZE = 64


def is_grouped_gemm_gfx950_layout_valid(
    tile_m: int,
    tile_n: int,
    m_waves: int,
    n_waves: int,
    use_half_tile_interleaved: bool,
) -> bool:
    """Return whether a config satisfies gfx950 B-LDS and C-shuffle layouts."""
    divisor = 2 if use_half_tile_interleaved else 1
    effective_tile_m = tile_m // divisor
    effective_tile_n = tile_n // divisor
    # LDSReadTrans16_64b requires effective N to span a full 64-lane wave.
    if (
        effective_tile_m <= 0
        or effective_tile_n < GFX950_WAVE_SIZE
        or m_waves <= 0
        or n_waves <= 0
    ):
        return False

    # Each C-shuffle thread writes one 16-byte N vector. Its thread count must
    # divide the workgroup, and the remaining thread dimension must tile M.
    cshuffle_vec_size = GFX950_DMA_BYTES // 2
    cshuffle_x_threads = effective_tile_n // cshuffle_vec_size
    block_threads = m_waves * n_waves * GFX950_WAVE_SIZE
    m_threads, remainder = divmod(block_threads, cshuffle_x_threads)
    return remainder == 0 and effective_tile_m % m_threads == 0


def get_grouped_gemm_persistent_grid_size(
    param: Any,
    total_m: int,
    n: int,
    group_count: int,
    device_properties: Any,
) -> int:
    """Choose a persistent grid from work and occupancy upper bounds.

    A grid that is too small serializes independent tiles, while an oversized
    grid launches workgroups that immediately exit. Per-group M sizes are not
    available on the host, so ``m_tiles_upper`` safely overestimates
    ``sum(ceil(group_m / block_m))`` from ``total_m`` and ``group_count``.
    Overestimating adds scheduling cost; underestimating reduces parallelism.
    The occupancy bound uses the unioned LDS footprint and thread limit. Light
    tiles use a power-of-two occupancy capped at eight blocks per CU. Larger
    tiles use a conservative one-or-two block cap because register pressure is
    not modeled. The final grid is the smaller of both bounds.
    """
    num_cus = device_properties.multi_processor_count
    if total_m <= 0 or n <= 0 or group_count <= 0:
        return 1

    ab_smem_bytes = param.stages * (param.block_m + param.block_n)
    ab_smem_bytes *= param.block_k * param.in_data_bytes
    c_smem_bytes = param.block_m * param.block_n * param.out_data_bytes
    smem_bytes = max(ab_smem_bytes, c_smem_bytes)
    shared_memory_per_cu = device_properties.shared_memory_per_multiprocessor
    max_threads_per_cu = device_properties.max_threads_per_multi_processor
    resource_blocks_per_cu = min(
        max(shared_memory_per_cu // smem_bytes, 1),
        max(max_threads_per_cu // param.block_threads, 1),
    )

    light_tile = param.block_m <= 64 and param.block_n <= 128
    n_tiles = (n - 1) // param.block_n + 1
    light_blocks = min(1 << (resource_blocks_per_cu.bit_length() - 1), 8)
    blocks_per_cu = light_blocks if light_tile else min(resource_blocks_per_cu, 2)
    nonempty = min(group_count, total_m)
    m_tiles_upper = nonempty + (total_m - nonempty) // param.block_m
    return max(1, min(num_cus * blocks_per_cu, m_tiles_upper * n_tiles))
