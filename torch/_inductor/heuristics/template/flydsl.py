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
    M_WAVES: int = 4
    N_WAVES: int = 4
    GROUP_M: int = 0
    USE_HALF_TILE_INTERLEAVED: bool = False


class FlyDSLGemmConfigDict(TypedDict):
    TILE_M: int
    TILE_N: int
    TILE_K: int
    STAGES: int
    M_WAVES: int
    N_WAVES: int
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
        tile_m=int(gemm_config["TILE_M"]),
        tile_n=int(gemm_config["TILE_N"]),
        tile_k=int(gemm_config["TILE_K"]),
        stages=int(gemm_config["STAGES"]),
        m_waves=int(gemm_config["M_WAVES"]),
        n_waves=int(gemm_config["N_WAVES"]),
        group_m=int(gemm_config["GROUP_M"]),
        use_half_tile_interleaved=bool(
            gemm_config.get("USE_HALF_TILE_INTERLEAVED", False)
        ),
        # Tile validation is layout-independent; runtime callers pass the layout.
        a_is_transposed=False,
        b_is_transposed=False,
    )


def is_gemm_config_valid_for_shape(
    m: int,
    n: int,
    k: int,
    dtype_id: int,
    gemm_config: dict[str, int | bool],
    *,
    a_is_transposed: bool,
    b_is_transposed: bool,
) -> bool:
    """Return whether a FlyDSL config supports this concrete GEMM shape."""
    from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
        infer_has_k_tail,
        make_gemm_param_and_validate,
    )

    tile_k = int(gemm_config["TILE_K"])
    stages = int(gemm_config["STAGES"])
    use_half_tile_interleaved = bool(
        gemm_config.get("USE_HALF_TILE_INTERLEAVED", False)
    )
    has_k_tail = infer_has_k_tail(k, tile_k, stages)
    if use_half_tile_interleaved:
        k_tiles = (k + tile_k - 1) // tile_k
        has_k_tail = has_k_tail or (k_tiles % 2 != 0)

    return (
        make_gemm_param_and_validate(
            m,
            n,
            k,
            {
                "dtype_id": dtype_id,
                "tile_m": int(gemm_config["TILE_M"]),
                "tile_n": int(gemm_config["TILE_N"]),
                "tile_k": tile_k,
                "stages": stages,
                "m_waves": int(gemm_config["M_WAVES"]),
                "n_waves": int(gemm_config["N_WAVES"]),
                "group_m": int(gemm_config["GROUP_M"]),
                "use_half_tile_interleaved": use_half_tile_interleaved,
                "a_is_transposed": a_is_transposed,
                "b_is_transposed": b_is_transposed,
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
        "M_WAVES": [1, 2, 4],
        "N_WAVES": [1, 2, 4],
        "GROUP_M": [0, 4],
        "USE_HALF_TILE_INTERLEAVED": [False, True],
    }
    keys = selections.keys()
    values = selections.values()
    configs = [dict(zip(keys, combo)) for combo in product(*values)]
    valid_configs: list[FlyDSLGemmConfig] = []
    for gemm_config in configs:
        if not gemm_config["USE_HALF_TILE_INTERLEAVED"]:
            mma_m_iters = gemm_config["TILE_M"] // gemm_config["M_WAVES"] // 16
            mma_n_iters = gemm_config["TILE_N"] // gemm_config["N_WAVES"] // 16
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
