from dataclasses import asdict, dataclass
from itertools import product

import torch._inductor.config as config


# Keep in sync with make_gemm_gfx950_param in
# torch/_inductor/kernel/vendored_templates/flydsl/kernels/gemm_gfx950.py
_SMEM_CAPACITY_BY_ARCH = {
    "gfx942": 65536,
    "gfx950": 163840,
}
_DEFAULT_SMEM_CAPACITY = 65536


def _smem_capacity() -> int:
    """Best-effort per-arch LDS capacity, matching the vendored gfx950 kernel.

    Falls back to the conservative gfx942 value so configs that only fit the
    larger gfx950 LDS are never proposed on an unknown/smaller device.
    """
    try:
        import torch

        if torch.cuda.is_available():
            gcn_arch = (
                torch.cuda.get_device_properties(0).gcnArchName or ""
            ).split(":", 1)[0]
            return _SMEM_CAPACITY_BY_ARCH.get(gcn_arch, _DEFAULT_SMEM_CAPACITY)
    except Exception:
        pass
    return _DEFAULT_SMEM_CAPACITY


@dataclass(frozen=True)
class FlyDSLGemmConfig:
    TILE_M: int = 128
    TILE_N: int = 128
    TILE_K: int = 64
    STAGES: int = 2
    SPLIT_K: int = 1
    BLOCK_M_WARPS: int = 4
    BLOCK_N_WARPS: int = 4
    BLOCK_K_WARPS: int = 1
    GROUP_M: int = 0
    B_TO_LDS: bool = True
    USE_HALF_TILE_INTERLEAVED: bool = False


def _is_valid_gemm_config(gemm_config: dict[str, int | bool]) -> bool:
    block_m = int(gemm_config["TILE_M"])
    block_n = int(gemm_config["TILE_N"])
    block_k = int(gemm_config["TILE_K"])
    stages = int(gemm_config["STAGES"])
    m_waves = int(gemm_config["BLOCK_M_WARPS"])
    n_waves = int(gemm_config["BLOCK_N_WARPS"])
    group_m = int(gemm_config["GROUP_M"])
    use_half_tile_interleaved = bool(
        gemm_config.get("USE_HALF_TILE_INTERLEAVED", False)
    )
    mma_m = 16
    mma_n = 16
    mma_k = 32

    if block_m <= 0 or block_n <= 0 or block_k <= 0 or stages <= 0:
        return False
    if stages < 2:
        return False
    if m_waves <= 0 or n_waves <= 0:
        return False
    if group_m < 0:
        return False

    in_dbytes = 2
    out_dbytes = 2
    cshuffle_vec_size = 16 // out_dbytes
    if use_half_tile_interleaved:
        half_block_m = block_m // 2
        half_block_n = block_n // 2
        if stages != 2:
            return False
        if m_waves != 2 or n_waves < 2:
            return False
        if half_block_m * 2 != block_m or half_block_n * 2 != block_n:
            return False
        mma_m_half_repeat = half_block_m // m_waves // mma_m
        mma_n_half_repeat = half_block_n // n_waves // mma_n
        if mma_m_half_repeat * m_waves * mma_m != half_block_m:
            return False
        if mma_n_half_repeat * n_waves * mma_n != half_block_n:
            return False
        if mma_n_half_repeat != 2:
            return False
        if half_block_n % cshuffle_vec_size != 0:
            return False
    elif block_n % cshuffle_vec_size != 0:
        return False

    smem_capacity = _smem_capacity()
    smem_bytes = stages * (block_m + block_n) * block_k * in_dbytes
    smem_bytes = max(smem_bytes, block_m * block_n * out_dbytes)
    if smem_bytes > smem_capacity:
        return False

    async_load_vec_size = 16 // in_dbytes
    ldg_x_threads = block_k // async_load_vec_size
    if ldg_x_threads * async_load_vec_size != block_k:
        return False

    block_threads = m_waves * n_waves * 64
    load_elems_per_iter = block_threads * async_load_vec_size
    if (block_m * block_k) % load_elems_per_iter != 0:
        return False
    if (block_n * block_k) % load_elems_per_iter != 0:
        return False
    ldg_a_iters = (block_m * block_k) // load_elems_per_iter
    ldg_b_iters = (block_n * block_k) // load_elems_per_iter
    if use_half_tile_interleaved:
        half_ldg_a_iters = ((block_m // 2) * block_k) // load_elems_per_iter
        half_ldg_b_iters = ((block_n // 2) * block_k) // load_elems_per_iter
        if half_ldg_a_iters * load_elems_per_iter != (block_m // 2) * block_k:
            return False
        if half_ldg_b_iters * load_elems_per_iter != (block_n // 2) * block_k:
            return False
    if ldg_a_iters <= 0 or ldg_b_iters <= 0:
        return False
    if (stages - 2) * (ldg_a_iters + ldg_b_iters) >= 63:
        return False

    mma_m_repeat = block_m // m_waves // mma_m
    mma_n_repeat = block_n // n_waves // mma_n
    if mma_m_repeat * m_waves * mma_m != block_m:
        return False
    if mma_n_repeat * n_waves * mma_n != block_n:
        return False
    if block_k % mma_k != 0:
        return False
    return True


def get_exhaustive_gemm_configs() -> list[FlyDSLGemmConfig]:
    """
    Returns the exhaustive configuration set for the gfx950 FlyDSL HGEMM kernel.
    """
    selections = {
        "TILE_M": [16, 32, 48, 64, 96, 128, 256],
        "TILE_N": [16, 32, 64, 96, 128, 256],
        "TILE_K": [64, 128, 256],
        "STAGES": [i for i in range(2, 10)],
        "BLOCK_M_WARPS": [1, 2, 4],
        "BLOCK_N_WARPS": [1, 2, 4],
        "SPLIT_K": [1],
        "BLOCK_K_WARPS": [1],
        "GROUP_M": [0, 4],
        "B_TO_LDS": [True],
        "USE_HALF_TILE_INTERLEAVED": [False, True],
    }
    keys = selections.keys()
    values = selections.values()
    configs = [dict(zip(keys, combo)) for combo in product(*values)]
    valid_configs: list[FlyDSLGemmConfig] = []
    for gemm_config in configs:
        mma_m_iters = gemm_config["TILE_M"] // gemm_config["BLOCK_M_WARPS"] // 16
        mma_n_iters = gemm_config["TILE_N"] // gemm_config["BLOCK_N_WARPS"] // 16
        if mma_m_iters > 4 or mma_n_iters > 4:
            continue
        if not _is_valid_gemm_config(gemm_config):
            continue
        try:
            valid_configs.append(FlyDSLGemmConfig(**gemm_config))
        except Exception:
            pass
    return valid_configs


def get_default_gemm_configs() -> list[FlyDSLGemmConfig]:
    """
    Returns the default configuration set for the gfx950 FlyDSL HGEMM kernel.
    """
    config_tuples = [
        (128, 128, 64, 2, 1, 4, 4, 1, 0, True),
        (128, 128, 64, 4, 1, 4, 4, 1, 0, True),
        (256, 256, 64, 2, 1, 4, 4, 1, 0, True),
        (128, 256, 64, 2, 1, 4, 4, 1, 0, True),
        (256, 128, 64, 2, 1, 4, 4, 1, 0, True),
        (64, 256, 64, 2, 1, 2, 4, 1, 0, True),
        (256, 64, 64, 2, 1, 4, 2, 1, 0, True),
        (64, 128, 64, 2, 1, 2, 4, 1, 0, True),
        (128, 64, 64, 2, 1, 4, 2, 1, 0, True),
        (96, 128, 64, 2, 1, 2, 4, 1, 0, True),
        (128, 96, 64, 2, 1, 4, 2, 1, 0, True),
        (64, 64, 64, 2, 1, 2, 2, 1, 0, True),
        (128, 128, 128, 2, 1, 4, 4, 1, 0, True),
        (64, 128, 128, 2, 1, 2, 4, 1, 0, True),
        (128, 64, 128, 2, 1, 4, 2, 1, 0, True),
        (64, 64, 128, 2, 1, 2, 2, 1, 0, True),
        (64, 64, 256, 2, 1, 2, 2, 1, 0, True),
        (128, 128, 64, 4, 1, 4, 4, 1, 4, True),
        (256, 256, 64, 2, 1, 4, 4, 1, 4, True),
        # Small-N tiles help small-M decode GEMMs.
        (16, 16, 128, 8, 1, 1, 1, 1, 4, True),
        (16, 16, 64, 8, 1, 1, 1, 1, 0, True),
        (32, 32, 64, 8, 1, 2, 2, 1, 4, True),
        (64, 32, 128, 4, 1, 4, 2, 1, 4, True),
        (64, 64, 64, 7, 1, 4, 2, 1, 4, True),
        (64, 128, 64, 6, 1, 2, 4, 1, 4, True),
        (128, 128, 64, 4, 1, 2, 4, 1, 4, True),
        (128, 256, 64, 3, 1, 4, 4, 1, 4, True),
        (32, 64, 64, 8, 1, 2, 2, 1, 0, True),
        (16, 64, 128, 3, 1, 1, 4, 1, 4, True),
        (64, 64, 64, 6, 1, 4, 2, 1, 4, True),
        # Trailing True enables the half-tile interleaved kernel.
        (128, 128, 64, 2, 1, 2, 2, 1, 0, True, True),
        (128, 128, 64, 2, 1, 2, 2, 1, 4, True, True),
        (128, 256, 64, 2, 1, 2, 4, 1, 0, True, True),
        (256, 128, 64, 2, 1, 2, 2, 1, 0, True, True),
        (256, 256, 64, 2, 1, 2, 4, 1, 0, True, True),
    ]
    configs = [FlyDSLGemmConfig(*args) for args in config_tuples]
    return [
        gemm_config
        for gemm_config in configs
        if _is_valid_gemm_config(asdict(gemm_config))
    ]


def get_gemm_configs() -> list[dict[str, object]]:
    """
    Returns the configuration set for the gfx950 FlyDSL HGEMM kernel.

    Shape compatibility is checked in the lowering before this function is called.
    By default, autotuning is disabled and we return only a single baseline config.
    """
    if (
        config.flydsl_enable_autotuning
        and config.max_autotune_gemm_search_space == "EXHAUSTIVE"
    ):
        configs = get_exhaustive_gemm_configs()
    elif config.flydsl_enable_autotuning:
        configs = get_default_gemm_configs()
    else:
        configs = [get_default_gemm_configs()[0]]
    return [asdict(gemm_config) for gemm_config in configs]
