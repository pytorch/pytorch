import functools
import logging
from dataclasses import asdict, dataclass
from itertools import product
from typing import cast, TypedDict

import torch._inductor.config as config


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FlyDSLGemmConfig:
    TILE_M: int = 128
    TILE_N: int = 128
    TILE_K: int = 64
    STAGES: int = 2
    BLOCK_M_WARPS: int = 4
    BLOCK_N_WARPS: int = 4
    GROUP_M: int = 0
    USE_HALF_TILE_INTERLEAVED: bool = False


class FlyDSLGemmConfigDict(TypedDict):
    TILE_M: int
    TILE_N: int
    TILE_K: int
    STAGES: int
    BLOCK_M_WARPS: int
    BLOCK_N_WARPS: int
    GROUP_M: int
    USE_HALF_TILE_INTERLEAVED: bool


FlyDSLGemmConfigArgs = tuple[int, int, int, int, int, int, int]
FlyDSLHTIGemmConfigArgs = tuple[int, int, int, int, int, int, int, bool]


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


@functools.cache
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
        "GROUP_M": [0, 4],
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
            candidate = FlyDSLGemmConfig(**cast(FlyDSLGemmConfigDict, gemm_config))
            _make_gemm_param(asdict(candidate))
            valid_configs.append(candidate)
        except Exception as e:
            log.debug(
                "Skipping invalid exhaustive FlyDSL config %s: %s", gemm_config, e
            )
    return valid_configs


@functools.cache
def get_default_gemm_configs() -> list[FlyDSLGemmConfig]:
    """
    Returns the default configuration set for the gfx950 FlyDSL GEMM kernel.
    """
    config_tuples: list[FlyDSLGemmConfigArgs] = [
        (128, 128, 64, 2, 4, 4, 0),
        (128, 128, 64, 4, 4, 4, 0),
        (256, 256, 64, 2, 4, 4, 0),
        (128, 256, 64, 2, 4, 4, 0),
        (256, 128, 64, 2, 4, 4, 0),
        (64, 256, 64, 2, 2, 4, 0),
        (256, 64, 64, 2, 4, 2, 0),
        (64, 128, 64, 2, 2, 4, 0),
        (128, 64, 64, 2, 4, 2, 0),
        (96, 128, 64, 2, 2, 4, 0),
        (128, 96, 64, 2, 4, 2, 0),
        (64, 64, 64, 2, 2, 2, 0),
        (128, 128, 128, 2, 4, 4, 0),
        (64, 128, 128, 2, 2, 4, 0),
        (128, 64, 128, 2, 4, 2, 0),
        (64, 64, 128, 2, 2, 2, 0),
        (64, 64, 256, 2, 2, 2, 0),
        (128, 128, 64, 4, 4, 4, 4),
        (256, 256, 64, 2, 4, 4, 4),
        # Small-N tiles help small-M decode GEMMs.
        (16, 16, 128, 8, 1, 1, 4),
        (16, 16, 64, 8, 1, 1, 0),
        (32, 32, 64, 8, 2, 2, 4),
        (64, 32, 128, 4, 4, 2, 4),
        (64, 64, 64, 7, 4, 2, 4),
        (64, 128, 64, 6, 2, 4, 4),
        (128, 128, 64, 4, 2, 4, 4),
        (128, 256, 64, 3, 4, 4, 4),
        (32, 64, 64, 8, 2, 2, 0),
        (16, 64, 128, 3, 1, 4, 4),
        (64, 64, 64, 6, 4, 2, 4),
    ]
    hti_config_tuples: list[FlyDSLHTIGemmConfigArgs] = [
        (128, 128, 64, 2, 2, 2, 0, True),
        (128, 128, 64, 2, 2, 2, 4, True),
        (128, 256, 64, 2, 2, 4, 0, True),
        (256, 128, 64, 2, 2, 2, 0, True),
        (256, 256, 64, 2, 2, 4, 0, True),
        (256, 256, 64, 2, 2, 4, 4, True),
    ]
    # Tuple order must match the FlyDSLGemmConfig field declaration order.
    configs = [FlyDSLGemmConfig(*args) for args in config_tuples]
    configs.extend(FlyDSLGemmConfig(*args) for args in hti_config_tuples)
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
    else:
        configs = get_default_gemm_configs()
        if not config.flydsl_enable_autotuning:
            configs = [c for c in configs if c == FlyDSLGemmConfig()]
    if not configs:
        log.warning("No valid FlyDSL GEMM configuration is available")
        return []
    return [asdict(gemm_config) for gemm_config in configs]


def _is_grouped_gemm_layout_valid(
    tile_m: int,
    tile_n: int,
    m_waves: int,
    n_waves: int,
    use_half_tile_interleaved: bool,
) -> bool:
    """Return whether a config satisfies grouped B-LDS and C-shuffle layouts."""
    divisor = 2 if use_half_tile_interleaved else 1
    effective_tile_m = tile_m // divisor
    effective_tile_n = tile_n // divisor
    if (
        effective_tile_m <= 0
        or effective_tile_n < 64
        or effective_tile_n & (effective_tile_n - 1)
        or m_waves <= 0
        or n_waves <= 0
    ):
        return False

    cshuffle_x_threads = effective_tile_n // 8
    block_threads = m_waves * n_waves * 64
    if block_threads % cshuffle_x_threads != 0:
        return False
    return effective_tile_m % (block_threads // cshuffle_x_threads) == 0


def get_exhaustive_grouped_gemm_configs() -> list[FlyDSLGemmConfig]:
    """Return exhaustive configs for the gfx950 FlyDSL grouped GEMM kernel."""
    return [
        gemm_config
        for gemm_config in get_exhaustive_gemm_configs()
        if _is_grouped_gemm_layout_valid(
            gemm_config.TILE_M,
            gemm_config.TILE_N,
            gemm_config.BLOCK_M_WARPS,
            gemm_config.BLOCK_N_WARPS,
            gemm_config.USE_HALF_TILE_INTERLEAVED,
        )
    ]


# Baseline used when FlyDSL autotuning is disabled. The dataclass defaults
# describe the dense kernel, whose 4x4 wave split does not apply here, so the
# grouped baseline is named explicitly; it must stay in the candidate list below.
DEFAULT_GROUPED_GEMM_CONFIG = FlyDSLGemmConfig(128, 128, 64, 2, 1, 4, 0)


@functools.cache
def get_default_grouped_gemm_configs() -> list[FlyDSLGemmConfig]:
    """Return default configs for the gfx950 FlyDSL grouped GEMM kernel.

    The grouped kernel always stages N-contiguous B vectors into LDS and reads
    them back transposed.
    """
    config_tuples: list[FlyDSLGemmConfigArgs] = [
        # Small-M grouped/decode configs.  These reduce wasted work when each
        # group has far fewer than 32 rows.
        (16, 64, 64, 2, 1, 2, 0),
        (16, 128, 64, 2, 1, 2, 0),
        (32, 64, 64, 2, 1, 2, 0),
        (32, 256, 64, 2, 1, 4, 0),
        (32, 128, 64, 2, 1, 4, 0),
        (64, 128, 64, 2, 1, 4, 0),
        (64, 256, 64, 2, 1, 4, 0),
        (128, 128, 64, 2, 1, 4, 0),
        (128, 256, 64, 2, 1, 4, 0),
        # Deeper pipelines, autotuned for the multi-stage overlap.
        (64, 128, 64, 3, 1, 4, 0),
        (128, 128, 64, 3, 1, 4, 0),
        (128, 256, 64, 3, 1, 4, 0),
        # Swizzled group-M variant remaps sufficiently large per-group tile grids.
        (128, 128, 64, 2, 1, 4, 4),
    ]
    hti_config_tuples: list[FlyDSLHTIGemmConfigArgs] = [
        # 2x2 half-tile-interleaved variant (stages=2 only): four half-block
        # accumulators + per-quadrant cshuffle store for better register tiling
        # and MMA scheduling. Requires m_waves=2, n_waves>=2 and even tiles.
        (64, 128, 64, 2, 2, 2, 0, True),
        (128, 128, 64, 2, 2, 2, 0, True),
        (128, 128, 64, 2, 2, 2, 4, True),
        (128, 256, 64, 2, 2, 4, 0, True),
        (256, 128, 64, 2, 2, 2, 0, True),
        (256, 256, 64, 2, 2, 4, 0, True),
    ]
    # Tuple order must match the FlyDSLGemmConfig field declaration order.
    candidates = [FlyDSLGemmConfig(*args) for args in config_tuples]
    candidates.extend(FlyDSLGemmConfig(*args) for args in hti_config_tuples)
    valid_configs: list[FlyDSLGemmConfig] = []
    for gemm_config in candidates:
        try:
            _make_gemm_param(asdict(gemm_config))
            valid_configs.append(gemm_config)
        except Exception as e:
            log.debug(
                "Skipping invalid default FlyDSL grouped config %s: %s",
                gemm_config,
                e,
            )
    return valid_configs


def is_grouped_gemm_config_valid_for_shape(
    m: int,
    n: int,
    k: int,
    dtype_id: int,
    gemm_config: dict[str, int | bool],
) -> bool:
    """Return whether a FlyDSL config supports this grouped GEMM shape."""
    tile_m = int(gemm_config["TILE_M"])
    tile_n = int(gemm_config["TILE_N"])
    tile_k = int(gemm_config["TILE_K"])
    stages = int(gemm_config["STAGES"])
    m_waves = int(gemm_config["BLOCK_M_WARPS"])
    n_waves = int(gemm_config["BLOCK_N_WARPS"])
    use_half_tile_interleaved = bool(gemm_config["USE_HALF_TILE_INTERLEAVED"])
    has_enough_k = use_half_tile_interleaved or k // tile_k >= stages
    return (
        tile_m <= max(128, m)
        and n >= tile_n
        and n % tile_n == 0
        and has_enough_k
        and _is_grouped_gemm_layout_valid(
            tile_m, tile_n, m_waves, n_waves, use_half_tile_interleaved
        )
        and is_gemm_config_valid_for_shape(m, n, k, dtype_id, gemm_config)
    )


def get_grouped_gemm_configs() -> list[dict[str, int | bool]]:
    """Return configs for the persistent multi-stage grouped kernel.

    Shape compatibility is checked in the lowering before this function is called.
    By default, autotuning is disabled and we return only a single baseline config.
    """
    if (
        config.flydsl_enable_autotuning
        and config.max_autotune_gemm_search_space == "EXHAUSTIVE"
    ):
        candidates = get_exhaustive_grouped_gemm_configs()
    else:
        candidates = get_default_grouped_gemm_configs()
        if not config.flydsl_enable_autotuning:
            candidates = [c for c in candidates if c == DEFAULT_GROUPED_GEMM_CONFIG]

    if not candidates:
        log.warning("No valid FlyDSL grouped GEMM configuration is available")
        return []
    return [asdict(gemm_config) for gemm_config in candidates]
