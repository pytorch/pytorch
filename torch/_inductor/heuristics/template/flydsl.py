import logging
from dataclasses import asdict, dataclass
from itertools import product

import torch._inductor.config as config


log = logging.getLogger(__name__)


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


def _config_from_mapping(values: dict[str, int | bool]) -> FlyDSLGemmConfig:
    return FlyDSLGemmConfig(
        TILE_M=int(values["TILE_M"]),
        TILE_N=int(values["TILE_N"]),
        TILE_K=int(values["TILE_K"]),
        STAGES=int(values["STAGES"]),
        SPLIT_K=int(values["SPLIT_K"]),
        BLOCK_M_WARPS=int(values["BLOCK_M_WARPS"]),
        BLOCK_N_WARPS=int(values["BLOCK_N_WARPS"]),
        BLOCK_K_WARPS=int(values["BLOCK_K_WARPS"]),
        GROUP_M=int(values["GROUP_M"]),
        B_TO_LDS=bool(values["B_TO_LDS"]),
        USE_HALF_TILE_INTERLEAVED=bool(values.get("USE_HALF_TILE_INTERLEAVED", False)),
    )


def _config_from_tuple(values: tuple[int | bool, ...]) -> FlyDSLGemmConfig:
    if len(values) not in (10, 11):
        raise ValueError(f"expected 10 or 11 FlyDSL config values, got {len(values)}")
    return FlyDSLGemmConfig(
        TILE_M=int(values[0]),
        TILE_N=int(values[1]),
        TILE_K=int(values[2]),
        STAGES=int(values[3]),
        SPLIT_K=int(values[4]),
        BLOCK_M_WARPS=int(values[5]),
        BLOCK_N_WARPS=int(values[6]),
        BLOCK_K_WARPS=int(values[7]),
        GROUP_M=int(values[8]),
        B_TO_LDS=bool(values[9]),
        USE_HALF_TILE_INTERLEAVED=bool(values[10]) if len(values) == 11 else False,
    )


def _make_gemm_param(gemm_config: dict[str, int | bool]):
    # Keep FlyDSL optional when this heuristics module is imported.
    from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
        make_gemm_gfx950_param,
    )

    return make_gemm_gfx950_param(
        block_m=int(gemm_config["TILE_M"]),
        block_n=int(gemm_config["TILE_N"]),
        block_k=int(gemm_config["TILE_K"]),
        stages=int(gemm_config["STAGES"]),
        m_waves=int(gemm_config["BLOCK_M_WARPS"]),
        n_waves=int(gemm_config["BLOCK_N_WARPS"]),
        group_m=int(gemm_config["GROUP_M"]),
        use_half_tile_interleaved=bool(
            gemm_config.get("USE_HALF_TILE_INTERLEAVED", False)
        ),
    )


def is_gemm_config_valid_for_shape(
    m: int,
    n: int,
    k: int,
    dtype_id: int,
    gemm_config: dict[str, int | bool],
) -> bool:
    """Return whether a FlyDSL config supports this concrete GEMM shape."""
    from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
        infer_has_k_tail,
        make_gemm_param_and_validate,
    )

    block_k = int(gemm_config["TILE_K"])
    stages = int(gemm_config["STAGES"])
    use_half_tile_interleaved = bool(
        gemm_config.get("USE_HALF_TILE_INTERLEAVED", False)
    )
    has_k_tail = infer_has_k_tail(k, block_k, stages)
    if use_half_tile_interleaved:
        k_tiles = (k + block_k - 1) // block_k
        has_k_tail = has_k_tail or (k_tiles % 2 != 0)

    return (
        make_gemm_param_and_validate(
            m,
            n,
            k,
            {
                "dtype_id": dtype_id,
                "block_m": int(gemm_config["TILE_M"]),
                "block_n": int(gemm_config["TILE_N"]),
                "block_k": block_k,
                "stages": stages,
                "m_waves": int(gemm_config["BLOCK_M_WARPS"]),
                "n_waves": int(gemm_config["BLOCK_N_WARPS"]),
                "group_m": int(gemm_config["GROUP_M"]),
                "use_half_tile_interleaved": use_half_tile_interleaved,
                "has_bias": False,
                "has_k_tail": has_k_tail,
            },
        )
        is not None
    )


def get_exhaustive_gemm_configs() -> list[FlyDSLGemmConfig]:
    """
    Returns the exhaustive configuration set for the gfx950 FlyDSL GEMM kernel.
    """
    selections = {
        "TILE_M": [16, 32, 48, 64, 80, 96, 128, 256],
        "TILE_N": [16, 32, 64, 80, 96, 128, 256],
        "TILE_K": [64, 128, 256],
        "STAGES": list(range(2, 10)),
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
        if not gemm_config["USE_HALF_TILE_INTERLEAVED"]:
            mma_m_iters = gemm_config["TILE_M"] // gemm_config["BLOCK_M_WARPS"] // 16
            mma_n_iters = gemm_config["TILE_N"] // gemm_config["BLOCK_N_WARPS"] // 16
            if mma_m_iters > 4 or mma_n_iters > 4:
                continue
        try:
            config = _config_from_mapping(gemm_config)
            _make_gemm_param(asdict(config))
            valid_configs.append(config)
        except Exception as e:
            log.debug(
                "Skipping invalid exhaustive FlyDSL config %s: %s", gemm_config, e
            )
    return valid_configs


def get_default_gemm_configs() -> list[FlyDSLGemmConfig]:
    """
    Returns the default configuration set for the gfx950 FlyDSL GEMM kernel.
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
        (256, 256, 64, 2, 1, 2, 4, 1, 4, True, True),
    ]
    configs = [_config_from_tuple(args) for args in config_tuples]
    valid_configs: list[FlyDSLGemmConfig] = []
    for gemm_config in configs:
        try:
            _make_gemm_param(asdict(gemm_config))
            valid_configs.append(gemm_config)
        except Exception as e:
            log.debug("Skipping invalid default FlyDSL config %s: %s", gemm_config, e)
    return valid_configs


def get_gemm_configs() -> list[dict[str, int | bool]]:
    """
    Returns the configuration set for the gfx950 FlyDSL GEMM kernel.

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
        configs = get_default_gemm_configs()
        configs = configs[:1]
    if not configs:
        log.warning("No valid FlyDSL GEMM configuration is available")
        return []
    return [asdict(gemm_config) for gemm_config in configs]
