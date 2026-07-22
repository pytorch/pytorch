from dataclasses import asdict, dataclass
from itertools import product

import torch._inductor.config as config


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

    try:
        GFX950_DMA_BYTES = 16
        GFX950_WAVE_SIZE = 64
        if block_m <= 0 or block_n <= 0 or block_k <= 0 or stages <= 0:
            raise ValueError("block_m, block_n, block_k, and stages must be positive")
        if (mma_m, mma_n, mma_k) != (16, 16, 32):
            raise ValueError("the gfx950 layout kernel currently requires mma=16x16x32")
        if stages < 2:
            raise ValueError("stages must be at least 2 for the staged LDS pipeline")
        if m_waves <= 0 or n_waves <= 0:
            raise ValueError("m_waves, and n_waves must be positive")
        if group_m < 0:
            raise ValueError("group_m must be non-negative")
        in_dbytes = out_dbytes = 2  # for hgemm
        cshuffle_vec_size = 16 // in_dbytes
        if use_half_tile_interleaved:
            half_block_m = block_m // 2
            half_block_n = block_n // 2
            assert stages == 2
            assert m_waves == 2 and n_waves >= 2
            assert half_block_m * 2 == block_m
            assert half_block_n * 2 == block_n
            mma_m_half_repeat = half_block_m // m_waves // mma_m
            mma_n_half_repeat = half_block_n // n_waves // mma_n
            assert mma_m_half_repeat * m_waves * mma_m == half_block_m
            assert mma_n_half_repeat * n_waves * mma_n == half_block_n
            assert half_block_n % cshuffle_vec_size == 0
        else:
            assert block_n % cshuffle_vec_size == 0
        smem_bytes = stages * (block_m + block_n) * block_k * in_dbytes
        smem_bytes = max(smem_bytes, block_m * block_n * in_dbytes)
        smem_capacity = 163840
        if smem_bytes > smem_capacity:
            raise ValueError(
                "staged LDS buffers exceed the device shared-memory capacity: "
                f"stages={stages}, block_m={block_m}, block_n={block_n}, "
                f"block_k={block_k}, smem_bytes={smem_bytes}, "
                f"capacity={smem_capacity} for arch={"gfx950"}"
            )
        async_load_vec_size = GFX950_DMA_BYTES // in_dbytes
        ldg_x_threads = block_k // async_load_vec_size
        if ldg_x_threads * async_load_vec_size != block_k:
            raise ValueError(
                "block_k must be divisible by the async load vector size: "
                f"block_k={block_k}, async_load_vec_size={async_load_vec_size}, "
                f"covered_k={ldg_x_threads * async_load_vec_size}"
            )
        block_threads = m_waves * n_waves * GFX950_WAVE_SIZE
        ldg_a_iters = (block_m * block_k) // (block_threads * async_load_vec_size)
        ldg_b_iters = (block_n * block_k) // (block_threads * async_load_vec_size)
        if use_half_tile_interleaved:
            half_ldg_a_iters = ((block_m // 2) * block_k) // (
                block_threads * async_load_vec_size
            )
            half_ldg_b_iters = ((block_n // 2) * block_k) // (
                block_threads * async_load_vec_size
            )
            if (
                half_ldg_a_iters * block_threads * async_load_vec_size
                != (block_m // 2) * block_k
            ):
                raise ValueError(
                    "Half-tile A async load tile must be exactly covered by whole-thread vector loads: "
                    f"half_block_m={block_m // 2}, block_k={block_k}, "
                    f"block_threads={block_threads}, async_load_vec_size={async_load_vec_size}, "
                    f"half_ldg_a_iters={half_ldg_a_iters}"
                )
            if (
                half_ldg_b_iters * block_threads * async_load_vec_size
                != (block_n // 2) * block_k
            ):
                raise ValueError(
                    "Half-tile B async load tile must be exactly covered by whole-thread vector loads: "
                    f"half_block_n={block_n // 2}, block_k={block_k}, "
                    f"block_threads={block_threads}, async_load_vec_size={async_load_vec_size}, "
                    f"half_ldg_b_iters={half_ldg_b_iters}"
                )
        if ldg_a_iters * block_threads * async_load_vec_size != block_m * block_k:
            raise ValueError(
                "A async load tile must be exactly covered by whole-thread vector loads: "
                f"block_m={block_m}, block_k={block_k}, "
                f"block_threads={block_threads}, async_load_vec_size={async_load_vec_size}, "
                f"ldg_a_iters={ldg_a_iters}, "
                f"covered={ldg_a_iters * block_threads * async_load_vec_size}, "
                f"required={block_m * block_k}"
            )
        if ldg_b_iters * block_threads * async_load_vec_size != block_n * block_k:
            raise ValueError(
                "B async load tile must be exactly covered by whole-thread vector loads: "
                f"block_n={block_n}, block_k={block_k}, "
                f"block_threads={block_threads}, async_load_vec_size={async_load_vec_size}, "
                f"ldg_b_iters={ldg_b_iters}, "
                f"covered={ldg_b_iters * block_threads * async_load_vec_size}, "
                f"required={block_n * block_k}"
            )
        assert (stages - 2) * (ldg_a_iters + ldg_b_iters) < 63
        mma_m_repeat = block_m // m_waves // mma_m
        mma_n_repeat = block_n // n_waves // mma_n
        if mma_m_repeat * m_waves * mma_m != block_m:
            raise ValueError(
                "block_m must be divisible by m_waves * mma_m: "
                f"block_m={block_m}, m_waves={m_waves}, mma_m={mma_m}, "
                f"mma_m_repeat={mma_m_repeat}, covered_m={mma_m_repeat * m_waves * mma_m}"
            )
        if mma_n_repeat * n_waves * mma_n != block_n:
            raise ValueError(
                "block_n must be divisible by n_waves * mma_n: "
                f"block_n={block_n}, n_waves={n_waves}, mma_n={mma_n}, "
                f"mma_n_repeat={mma_n_repeat}, covered_n={mma_n_repeat * n_waves * mma_n}"
            )
    except Exception:
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
