import logging
from dataclasses import asdict, dataclass
from itertools import product
from typing import cast, TypedDict

import torch._inductor.config as config


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FlyDSLMXFP8Config:
    TILE_M: int = 128
    TILE_N: int = 128
    TILE_K: int = 128
    STAGES: int = 2
    BLOCK_M_WARPS: int = 2
    BLOCK_N_WARPS: int = 2
    GROUP_M: int = 0


class FlyDSLMXFP8ConfigDict(TypedDict):
    TILE_M: int
    TILE_N: int
    TILE_K: int
    STAGES: int
    BLOCK_M_WARPS: int
    BLOCK_N_WARPS: int
    GROUP_M: int


FlyDSLMXFP8ConfigArgs = tuple[int, int, int, int, int, int, int]


def _check_gemm_config(gemm_config: dict[str, int]) -> None:
    """Raise if the kernel cannot build this tile config."""
    # Keep FlyDSL optional when this heuristics module is imported.
    from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
        mxfp8_gemm_derived,
    )

    mxfp8_gemm_derived(
        block_m=int(gemm_config["TILE_M"]),
        block_n=int(gemm_config["TILE_N"]),
        block_k=int(gemm_config["TILE_K"]),
        stages=int(gemm_config["STAGES"]),
        m_waves=int(gemm_config["BLOCK_M_WARPS"]),
        n_waves=int(gemm_config["BLOCK_N_WARPS"]),
        group_m=int(gemm_config["GROUP_M"]),
    )


def is_mxfp8_config_valid_for_shape(
    m: int,
    n: int,
    k: int,
    out_dtype: str,
    gemm_config: dict[str, int],
) -> bool:
    """Return whether a FlyDSL MXFP8 config supports this concrete shape."""
    from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
        make_mxfp8_param_and_validate,
    )

    return make_mxfp8_param_and_validate(m, n, k, out_dtype, gemm_config) is not None


def get_exhaustive_mxfp8_gemm_configs() -> list[FlyDSLMXFP8Config]:
    """
    Returns the exhaustive configuration set for the gfx950 FlyDSL MXFP8 kernel.
    """
    selections = {
        "TILE_M": [16, 32, 64, 96, 128, 256],
        "TILE_N": [16, 32, 64, 96, 128, 256],
        "TILE_K": [128, 256, 512],
        "STAGES": [2, 3, 4, 5, 6],
        "BLOCK_M_WARPS": [1, 2, 4],
        "BLOCK_N_WARPS": [1, 2, 4],
        "GROUP_M": [0, 4],
    }
    keys = selections.keys()
    values = selections.values()
    configs = [dict(zip(keys, combo)) for combo in product(*values)]
    valid_configs: list[FlyDSLMXFP8Config] = []
    for gemm_config in configs:
        try:
            gemm = FlyDSLMXFP8Config(**cast(FlyDSLMXFP8ConfigDict, gemm_config))
            _check_gemm_config(asdict(gemm))
            valid_configs.append(gemm)
        except Exception as e:
            log.debug(
                "Skipping invalid exhaustive FlyDSL MXFP8 config %s: %s", gemm_config, e
            )
    return valid_configs


def get_default_mxfp8_gemm_configs() -> list[FlyDSLMXFP8Config]:
    """
    Returns the default configuration set for the gfx950 FlyDSL MXFP8 kernel.
    """
    tile_tuples: list[tuple[int, int, int, int, int, int, int]] = [
        # Deep register blocking (8x8 / 8x4 repeats): fewer, fatter waves so
        # each LDS read feeds four MFMAs instead of two. These win every shape
        # from M=2048 up -- 256x256 with a 2x2 wave grid is the large-GEMM
        # sweet spot -- and are the reason MXFP8_MAX_MMA_REPEAT is 8.
        (256, 256, 128, 2, 2, 2, 4),
        (256, 256, 128, 2, 2, 2, 0),
        (128, 128, 128, 2, 1, 2, 4),
        (128, 128, 128, 2, 2, 1, 4),
        (128, 128, 128, 2, 1, 1, 4),
        (128, 256, 128, 2, 1, 4, 4),
        (256, 128, 128, 2, 4, 1, 4),
        # Square-ish tiles: the throughput sweet spot for large GEMMs.
        (128, 128, 128, 2, 2, 2, 0),
        (128, 128, 128, 2, 2, 2, 4),
        (128, 128, 128, 3, 2, 2, 0),
        (256, 128, 128, 2, 4, 2, 0),
        (128, 256, 128, 2, 2, 4, 0),
        (256, 256, 128, 2, 4, 4, 0),
        (128, 128, 256, 2, 2, 2, 0),
        (64, 64, 128, 2, 2, 2, 0),
        (64, 128, 128, 2, 1, 2, 0),
        (128, 64, 128, 2, 2, 1, 0),
        (64, 64, 256, 2, 1, 1, 0),
        (64, 64, 512, 2, 1, 1, 0),
        # Small-M decode shapes: thin tiles keep the N sweep parallel.
        (16, 128, 128, 4, 1, 2, 0),
        (16, 256, 128, 4, 1, 4, 0),
        (32, 128, 128, 4, 1, 2, 0),
        (32, 256, 128, 3, 1, 4, 0),
        (32, 128, 256, 3, 1, 2, 0),
        (64, 256, 128, 2, 1, 4, 0),
        # Small-N (e.g. 4096x256x4096) wants the opposite aspect ratio.
        (256, 64, 128, 2, 4, 1, 0),
        (128, 32, 128, 3, 2, 1, 0),
        # Tiny catch-all tiles. The kernel has no boundary predication, so a
        # shape only gets a FlyDSL choice when some tile divides it exactly;
        # these keep any 16-aligned shape covered.
        (32, 32, 128, 2, 1, 1, 0),
        (16, 16, 128, 2, 1, 1, 0),
    ]
    # Tuple order must match the FlyDSLMXFP8Config field declaration order.
    configs = [FlyDSLMXFP8Config(*args) for args in tile_tuples]
    valid_configs: list[FlyDSLMXFP8Config] = []
    for gemm_config in configs:
        try:
            _check_gemm_config(asdict(gemm_config))
            valid_configs.append(gemm_config)
        except Exception as e:
            log.debug(
                "Skipping invalid default FlyDSL MXFP8 config %s: %s", gemm_config, e
            )
    return valid_configs


def get_mxfp8_gemm_configs() -> list[dict[str, int]]:
    """
    Returns the shape-independent configuration set for the gfx950 FlyDSL
    MXFP8 kernel.
    """
    if (
        config.flydsl_enable_autotuning
        and config.max_autotune_gemm_search_space == "EXHAUSTIVE"
    ):
        configs = get_exhaustive_mxfp8_gemm_configs()
    else:
        configs = get_default_mxfp8_gemm_configs()
    if not configs:
        log.warning("No valid FlyDSL MXFP8 GEMM configuration is available")
        return []
    return [asdict(gemm_config) for gemm_config in configs]


def get_mxfp8_gemm_configs_for_shape(
    m: int, n: int, k: int, out_dtype: str
) -> list[dict[str, int]]:
    """
    Returns the configurations to autotune over for one concrete shape.

    Unlike the FP16 GEMM template, this kernel has no boundary predication, so
    a config is only usable when its tile divides the shape exactly. That makes
    "the baseline config" shape-dependent: with autotuning disabled we return
    the single default config that fits, preferring FlyDSLMXFP8Config() when it
    is valid, rather than a fixed tuple that many shapes would reject outright.
    """
    configs = [
        gemm_config
        for gemm_config in get_mxfp8_gemm_configs()
        if is_mxfp8_config_valid_for_shape(m, n, k, out_dtype, gemm_config)
    ]
    if config.flydsl_enable_autotuning or not configs:
        return configs
    baseline = asdict(FlyDSLMXFP8Config())
    return [baseline] if baseline in configs else configs[:1]
