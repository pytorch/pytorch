"""Config for SM90 DeepSeek grouped mm."""

import math
from typing import NamedTuple


class HopperDeepSeekConfig(NamedTuple):
    tile_m: int
    tile_n: int
    cluster_m: int
    cluster_n: int
    # Compile-time in the kernel: tx_count must be a Python value. The wide
    # copies are always safe; the narrow ones need a provably aligned start.
    a_scale_wide: bool = True
    b_scale_wide: bool = True


_FIXED_CONFIG = HopperDeepSeekConfig(tile_m=64, tile_n=128, cluster_m=1, cluster_n=1)

_TILE_MS = (128, 64)
_TILE_NS = (128, 64)
_DEFAULT_NUM_SMS = 132
# cluster_n=2 multicasts A across the CTA pair: it pays when A is refetched a
# lot, i.e. L2-pressuring routed shapes or a single group below one M tile.
_CLUSTER_N_MIN_TOTAL_M = 32768
_CLUSTER_N_MIN_TILES_N = 32


def _validate_config(config: HopperDeepSeekConfig) -> HopperDeepSeekConfig:
    if config.tile_m not in (64, 128):
        raise ValueError(f"unsupported Hopper tile_m: {config.tile_m}")
    if config.tile_n not in (64, 128):
        raise ValueError(f"unsupported Hopper tile_n: {config.tile_n}")
    if config.cluster_m not in (1, 2):
        raise ValueError(
            f"unsupported Hopper cluster_m: {config.cluster_m} (only 1 or 2 is implemented)"
        )
    if config.cluster_n not in (1, 2):
        raise ValueError(
            f"unsupported Hopper cluster_n: {config.cluster_n} (only 1 or 2 is implemented)"
        )
    if config.cluster_m > 1 and config.cluster_n > 1:
        raise ValueError("only one clustered dimension may be greater than 1")
    return config


def _tile_rank(
    tile_m: int, tile_n: int, avg_group_m: int, n: int, group_count: int, num_sms: int
) -> tuple[int, float, int]:
    # Tiling is per group, so the M extent that matters is the average group,
    # not total_m: a 128-row tile over 4-row groups wastes twice a 64-row tile.
    m_tiles = -(-avg_group_m // tile_m)
    n_tiles = -(-n // tile_n)
    tiles = group_count * m_tiles * n_tiles
    work = tiles * tile_m * tile_n
    waves = -(-tiles // num_sms)
    wave_fill = tiles / (waves * num_sms)
    return -work, wave_fill * math.sqrt(tile_m * tile_n), tile_n


def select_kernel_config(
    total_m: int | None = None,
    n: int | None = None,
    k: int | None = None,
    group_count: int | None = None,
    num_sms: int | None = None,
    groups_split_k: bool = False,
) -> HopperDeepSeekConfig:
    del k
    if not total_m or not n or not group_count:
        return _validate_config(_FIXED_CONFIG)
    if not num_sms:
        num_sms = _DEFAULT_NUM_SMS

    # Splitting K leaves every group spanning all of total_m.
    if groups_split_k:
        avg_group_m = total_m
    else:
        avg_group_m = max(1, total_m // group_count)
    best_rank = None
    best_tile = None
    for tile_m in _TILE_MS:
        for tile_n in _TILE_NS:
            rank = _tile_rank(tile_m, tile_n, avg_group_m, n, group_count, num_sms)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_tile = (tile_m, tile_n)
    if best_tile is None:
        return _validate_config(_FIXED_CONFIG)

    tile_m, tile_n = best_tile
    cluster_n = 1
    tiles_n = -(-n // tile_n)
    tiny_single_group = (
        group_count == 1
        and not groups_split_k
        and total_m < tile_m
        and tiles_n >= _CLUSTER_N_MIN_TILES_N
    )
    cluster_n_eligible = total_m >= _CLUSTER_N_MIN_TOTAL_M or tiny_single_group
    if cluster_n_eligible and tiles_n % 2 == 0:
        cluster_n = 2
    # The narrow A-scale copy needs an aligned M start: splitting K keeps it at
    # m_tile * tile_m, splitting M does not.
    if groups_split_k:
        a_scale_wide = total_m % tile_m != 0
    else:
        a_scale_wide = not (group_count == 1 and total_m % tile_m == 0)
    return _validate_config(
        HopperDeepSeekConfig(
            tile_m=tile_m,
            tile_n=tile_n,
            cluster_m=1,
            cluster_n=cluster_n,
            a_scale_wide=a_scale_wide,
            b_scale_wide=n % tile_n != 0,
        )
    )
