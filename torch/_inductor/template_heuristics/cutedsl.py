from dataclasses import dataclass
from enum import auto, Enum
from itertools import product

import torch._inductor.config as config


class TensorMapUpdateMode(Enum):
    """Enum mirroring cutlass.utils.TensorMapUpdateMode to decouple this file from a cutlass dependency."""

    SMEM = auto()
    GMEM = auto()


@dataclass(frozen=True)
class CuTeGemmConfig:
    TILE_M: int = 128
    TILE_N: int = 192
    CLUSTER_M: int = 2
    CLUSTER_N: int = 1
    USE_2_CTA: bool = False
    TENSORMAP_UPDATE_MODE: TensorMapUpdateMode = TensorMapUpdateMode.SMEM


def get_exhaustive_groupgemm_configs() -> list[CuTeGemmConfig]:
    """
    Returns the exhaustive configuration set for the Blackwell CuTeDSL Grouped GEMM kernel.
    For information regarding valid config sets, see:
    https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/grouped_gemm.py
    """

    # Tile_n is always the same regardless of 2cta
    tile_n_vals = [32, 64, 96, 128, 160, 192, 224, 256]

    # Valid clusters
    clusters_no_2cta = [
        (1, 1),
        (1, 2),
        (1, 4),
        (1, 8),
        (1, 16),
        (2, 1),
        (2, 2),
        (2, 4),
        (2, 8),
        (4, 1),
        (4, 2),
        (4, 4),
        (8, 1),
        (8, 2),
        (16, 1),
    ]
    clusters_2cta = [
        (2, 1),
        (2, 2),
        (2, 4),
        (2, 8),
        (4, 1),
        (4, 2),
        (4, 4),
        (8, 1),
        (8, 2),
        (16, 1),
    ]

    configs: list[CuTeGemmConfig] = []

    for use_2cta, cluster_set, tile_m_range in [
        (False, clusters_no_2cta, [64, 128]),
        (True, clusters_2cta, [128, 256]),
    ]:
        for tensormap_update_mode, tile_m, tile_n, (cluster_m, cluster_n) in product(
            [TensorMapUpdateMode.SMEM, TensorMapUpdateMode.GMEM],
            tile_m_range,
            tile_n_vals,
            cluster_set,
        ):
            configs.append(
                CuTeGemmConfig(
                    tile_m,
                    tile_n,
                    cluster_m,
                    cluster_n,
                    USE_2_CTA=use_2cta,
                    TENSORMAP_UPDATE_MODE=tensormap_update_mode,
                )
            )

    return configs


def get_default_groupgemm_configs() -> list[CuTeGemmConfig]:
    """
    Returns the default configuration set for the Blackwell CuTeDSL Grouped GEMM kernel.
    """

    config_tuples = [
        (128, 256, 2, 1, False, TensorMapUpdateMode.SMEM),
        (256, 160, 2, 1, True, TensorMapUpdateMode.GMEM),
        (256, 256, 2, 1, True, TensorMapUpdateMode.GMEM),
        (64, 32, 1, 1, False, TensorMapUpdateMode.GMEM),
        (64, 256, 1, 2, False, TensorMapUpdateMode.SMEM),
        (128, 256, 1, 2, False, TensorMapUpdateMode.SMEM),
        (256, 256, 2, 2, True, TensorMapUpdateMode.GMEM),
        (128, 256, 1, 2, False, TensorMapUpdateMode.GMEM),
        (64, 32, 1, 1, False, TensorMapUpdateMode.SMEM),
        (256, 256, 2, 1, True, TensorMapUpdateMode.SMEM),
        (128, 256, 1, 1, False, TensorMapUpdateMode.GMEM),
        (256, 256, 8, 1, True, TensorMapUpdateMode.GMEM),
        (64, 32, 1, 2, False, TensorMapUpdateMode.SMEM),
        (256, 192, 2, 1, True, TensorMapUpdateMode.GMEM),
        (256, 256, 2, 2, True, TensorMapUpdateMode.SMEM),
        (128, 96, 1, 2, False, TensorMapUpdateMode.SMEM),
        (64, 192, 1, 1, False, TensorMapUpdateMode.SMEM),
        (64, 64, 1, 1, False, TensorMapUpdateMode.GMEM),
        (64, 192, 1, 1, False, TensorMapUpdateMode.GMEM),
        (128, 64, 1, 1, False, TensorMapUpdateMode.GMEM),
        (64, 160, 1, 1, False, TensorMapUpdateMode.GMEM),
        (64, 256, 1, 1, False, TensorMapUpdateMode.GMEM),
    ]

    return [CuTeGemmConfig(*args) for args in config_tuples]


def get_groupgemm_configs() -> list[CuTeGemmConfig]:
    """
    Returns the configuration set for the Blackwell CuTeDSL Grouped GEMM kernel.

    Note: CuTeDSL autotuning is still experimental — enabling it may trigger kernel launch failures
    or unstable results. By default, autotuning is disabled and we return only
    a single baseline config.
    """
    if (
        config.cutedsl_enable_autotuning
        and config.max_autotune_gemm_search_space == "EXHAUSTIVE"
    ):
        return get_exhaustive_groupgemm_configs()
    elif config.cutedsl_enable_autotuning:
        return get_default_groupgemm_configs()
    else:
        return [get_default_groupgemm_configs()[0]]
