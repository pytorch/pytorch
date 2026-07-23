# Copyright (C) 2025, Tri Dao.
import itertools
from typing import Optional, List
from functools import partial
from dataclasses import dataclass


@dataclass(frozen=True)
class GemmConfig:
    tile_m: int = 128
    tile_n: int = 192
    tile_k: int | None = None
    num_warps: int | None = None
    pingpong: bool = True
    # by default, we use dynamic persistent tile scheduler on SM100 but not on SM90
    is_dynamic_persistent: bool = True
    cluster_m: int = 2
    cluster_n: int = 1
    cluster_k: int = 1
    swap_ab: bool = False
    # raster_order: int = 1
    max_swizzle_size: int = 8
    device_capacity: int = 9
    # whether to use TMA gather (vs normal cp.async) for gather_A on SM100
    use_tma_gather: bool = False


def _get_sm90_configs(
    epilogue: Optional[str] = None,
    tune_coop: bool = True,
) -> List[GemmConfig]:
    tile_n_vals = [128, 160, 192, 208]
    tile_mn_vals_coop = [(256, tile_n) for tile_n in tile_n_vals] + [
        (128, 224),
        (128, 256),
        # (192, 256),  # Getting IOT instruction (core dumped) in the bwd
    ]
    tile_mn_vals_pingpong = [(128, tile_n) for tile_n in tile_n_vals] + [(192, 128)]
    if epilogue in ["gated"]:
        tile_mn_vals_coop = [(m, n) for m, n in tile_mn_vals_coop if n % 32 == 0 and m != 192]
        tile_mn_vals_pingpong = [(m, n) for m, n in tile_mn_vals_pingpong if n % 32 == 0]
    elif epilogue in ["lse"]:
        tile_mn_vals_coop = [(m, n) for m, n in tile_mn_vals_coop if m != 192]
    tile_mn_vals = []
    if tune_coop:
        tile_mn_vals += [(m, n, False) for m, n in tile_mn_vals_coop]
    tile_mn_vals += [(m, n, True) for m, n in tile_mn_vals_pingpong]
    cluster = [(1, 2), (2, 1)]
    # cluster = [(1, 1), (1, 2), (2, 1)]
    if epilogue in ["lse"]:
        cluster = [(1, 2), (2, 1)]
    swap_ab_vals = [False, True]
    if epilogue in ["lse", "gated"]:
        swap_ab_vals = [False]

    return [
        GemmConfig(
            tile_m=tile_m,
            tile_n=tile_n,
            pingpong=pingpong,
            cluster_m=cluster_m,
            cluster_n=cluster_n,
            swap_ab=swap_ab,
            device_capacity=9,
            is_dynamic_persistent=False,  # default to not use dynamic persistent on SM90
            use_tma_gather=False,  # TMA gather not supported on SM90
        )
        for (tile_m, tile_n, pingpong), (cluster_m, cluster_n), swap_ab in itertools.product(
            tile_mn_vals,
            cluster,
            swap_ab_vals,
        )
    ]


def _get_sm80_configs() -> List[GemmConfig]:
    tile_mn_warps_vals = [
        (128, 128, 4),
        (128, 128, 8),
        (128, 160, 4),
        # TODO: Make 128x160 work with 8 warps. It currently makes the accumulator
        # N layout odd and fails epilogue retile.
        (128, 192, 4),
        (128, 192, 8),
        (128, 256, 8),
        (128, 64, 4),
        (64, 128, 4),
    ]
    return [
        GemmConfig(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            num_warps=num_warps,
            pingpong=False,
            cluster_m=1,
            cluster_n=1,
            swap_ab=swap_ab,
            device_capacity=8,
            is_dynamic_persistent=False,
            use_tma_gather=False,
        )
        for (tile_m, tile_n, num_warps), tile_k, swap_ab in itertools.product(
            tile_mn_warps_vals, [32, 64], [False, True]
        )
    ]


def _get_sm100_configs(
    epilogue: Optional[str] = None,
) -> List[GemmConfig]:
    tile_n_vals = [64, 128, 160, 192, 224, 256]
    tile_mn_cluster_vals = (
        [(128, tile_n, (1, 1)) for tile_n in tile_n_vals]
        + [(128, tile_n, (1, 2)) for tile_n in tile_n_vals]
        + [(128, tile_n, (2, 1)) for tile_n in tile_n_vals]
        + [(128, tile_n, (2, 2)) for tile_n in tile_n_vals]
        + [(256, tile_n, (2, 1)) for tile_n in tile_n_vals]
        + [(256, tile_n, (2, 2)) for tile_n in tile_n_vals]
        + [(256, 512, (2, 1))]
    )
    swap_ab_vals = [False, True]
    if epilogue in ["lse", "gated"]:
        swap_ab_vals = [False]
    GemmConfigCls = partial(
        GemmConfig, pingpong=False, device_capacity=10
    )  # There's no pingpong on Sm100
    use_clc_vals = [True, False]
    use_tma_gather_vals = [True, False]
    return [
        GemmConfigCls(
            tile_m=m,
            tile_n=n,
            cluster_m=cm,
            cluster_n=cn,
            swap_ab=sab,
            max_swizzle_size=8,
            is_dynamic_persistent=use_clc,
            use_tma_gather=use_tma_gather,
        )
        for (m, n, (cm, cn)), sab, use_clc, use_tma_gather in itertools.product(
            tile_mn_cluster_vals, swap_ab_vals, use_clc_vals, use_tma_gather_vals
        )
    ]


def _get_sm120_configs(
    epilogue: Optional[str] = None,
    tune_coop: bool = True,
) -> List[GemmConfig]:
    tile_mn_vals_coop = [(128, 128), (128, 64), (64, 128), (128, 160), (128, 192)]
    tile_mn_vals_pingpong = [(128, 128), (128, 64), (64, 128), (128, 160)]
    tile_mn_vals = []
    if tune_coop:
        tile_mn_vals += [(m, n, False) for m, n in tile_mn_vals_coop]
    tile_mn_vals += [(m, n, True) for m, n in tile_mn_vals_pingpong]
    swap_ab_vals = [False, True]
    if epilogue in ["lse", "gated"]:
        swap_ab_vals = [False]
    return [
        GemmConfig(
            tile_m=tile_m,
            tile_n=tile_n,
            pingpong=pingpong,
            cluster_m=1,
            cluster_n=1,
            swap_ab=swap_ab,
            device_capacity=12,
            is_dynamic_persistent=True,
            use_tma_gather=False,  # TMA gather not supported on SM120
        )
        for (tile_m, tile_n, pingpong), swap_ab in itertools.product(tile_mn_vals, swap_ab_vals)
    ]


def get_all_configs(
    epilogue: Optional[str] = None,
    tune_coop: bool = True,
) -> List[GemmConfig]:
    """Return autotuning configs for all supported device capabilities.

    Each GemmConfig is tagged with its target device_capacity, so the caller can
    filter at runtime based on the actual device. This avoids querying the device
    (and initializing a CUDA context) at import time.
    """
    return (
        _get_sm80_configs()
        + _get_sm90_configs(epilogue, tune_coop)
        + _get_sm100_configs(epilogue)
        + _get_sm120_configs(epilogue, tune_coop)
    )
