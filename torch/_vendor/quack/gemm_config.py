# Copyright (C) 2025, Tri Dao.
import itertools
from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import IntEnum
from functools import lru_cache, partial
from typing import List, Optional, get_args


class SplitKMode(IntEnum):
    """How split-K partial results are combined into the output.

    The canonical public import path is ``from quack.gemm_interface import SplitKMode``
    (re-exported there next to the gemm entry points that take it). The definition has
    to live in this leaf module: the kernel layer (gemm_base / gemm_sm*) consumes it at
    import time, and gemm_interface sits above quack.gemm in the import graph, so
    defining it there would be a circular import.

    IntEnum (like RoundingMode) so values cross the torch.library custom-op schema
    boundary as plain ints and pickle stably for the jit-cache key.
    """

    # All modes commit raw f32 partials; the full epilogue runs exactly once, on the
    # entity that owns the completed sum (f32 accumulation in every mode).
    # Per-output-tile turnstile orders the partial commits in split order; the last
    # split finalizes: bitwise deterministic run to run.
    SERIAL = 0
    # Partials commit in arrival order (no waiting); the last split finalizes after an
    # arrival counter fills: lowest latency, NOT deterministic run to run.
    PARALLEL = 1
    # Each split stores f32 partials to its own workspace slice; a separate reduction
    # kernel (quack/split_k_reduce.py) sums them in a deterministic order and applies
    # the epilogue.
    SEPARATE = 2


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
    split_k: int = 1
    swap_ab: bool = False
    # raster_order: int = 1
    max_swizzle_size: int = 8
    device_capacity: int = 9
    # whether to use TMA gather (vs normal cp.async) for gather_A on SM100
    use_tma_gather: bool = False


def cta_tile_shape_m(
    tile_m: int, cluster_m: int, device_capacity: int, blockscaled: bool = False
) -> int:
    """Per-CTA M tile. Mirrors GemmSm100.use_2cta_instrs (keep in sync): on
    SM100/SM103 an even cluster_m with MMA tiler M in {128, 256} ({256} only
    when blockscaled) selects the 2-CTA MMA, which splits tile_m across the
    CTA pair. Tile schedulers, OOB limits, and reduce-sink partial slots all
    count M in this unit — host-side buffers sized per M tile must use it too."""
    if device_capacity not in (10, 11) or cluster_m % 2:
        return tile_m
    valid_2cta_m = (256,) if blockscaled else (128, 256)
    return tile_m // 2 if tile_m in valid_2cta_m else tile_m


def blockscaled_config_ok(c: GemmConfig) -> bool:
    """Can this config run a blockscaled GEMM (SM100 tcgen05 MMA, or SM120
    warp MMA)? THE single statement of the constraint set — both autotune
    prune paths call this."""
    if c.device_capacity == 12:
        # SM120 warp-MMA blockscaled (MXFP8): the SF smem layouts and fragment
        # partition helpers are whole-128-tile granular.
        return (
            not c.swap_ab  # untested with blockscaled; SFA/SFB would swap too
            and c.tile_k is None  # tile_k is derived from the SF atom column
            and c.tile_m in (128, 256)
            and c.tile_n in (128, 256)
        )
    return (
        c.device_capacity in (10, 11)
        and not c.swap_ab  # untested with blockscaled; SFA/SFB would swap too
        and c.tile_k is None  # tile_k is derived from the MMA instruction
        and c.tile_m in (128, 256)
        # SF tmem datapath is 64-N granular; tcgen05 MMA N is capped at 256
        and c.tile_n % 64 == 0
        and 64 <= c.tile_n <= 256
        # SF multicast is limited to 4 CTAs per cluster dim
        and c.cluster_m <= 4
        and c.cluster_n <= 4
    )


def canonicalize_config_constraints(config_constraints) -> tuple[tuple[str, object], ...]:
    """Validate partial GemmConfig fields and return a stable tuple key."""
    if config_constraints is None:
        return ()
    if isinstance(config_constraints, Mapping):
        items = config_constraints.items()
    elif isinstance(config_constraints, tuple):
        items = config_constraints
    else:
        raise TypeError("config_constraints must be a mapping or tuple of (field, value) pairs")

    field_types = {field.name: field.type for field in fields(GemmConfig)}
    constraints = {}
    for item in items:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("config_constraints entries must be (field, value) tuples")
        name, value = item
        if not isinstance(name, str):
            raise TypeError("config_constraints field names must be strings")
        if name not in field_types:
            valid = ", ".join(field_types)
            raise ValueError(f"unknown GemmConfig constraint {name!r}; choose one of {valid}")
        if name in constraints:
            raise ValueError(f"duplicate GemmConfig constraint {name!r}")
        expected_types = get_args(field_types[name]) or (field_types[name],)
        if type(value) not in expected_types:
            expected = " | ".join(
                "None" if expected_type is type(None) else expected_type.__name__
                for expected_type in expected_types
            )
            raise TypeError(
                f"GemmConfig constraint {name!r} must have exact type {expected}, "
                f"got {type(value).__name__}"
            )
        constraints[name] = value
    return tuple(sorted(constraints.items()))


def config_supports(config: GemmConfig, *, gather_A: bool = False, varlen_m: bool = False) -> bool:
    """Structural validity of a config for gather_A / varlen_m.

    Single source of truth shared by the autotune pruner
    (gemm_interface.prune_invalid_gemm_configs) and the analytic heuristic's
    candidate spaces (gemm_heuristic; enforced by tests/test_gemm_heuristic.py).
    """
    if (gather_A or varlen_m) and config.swap_ab:
        return False
    if gather_A:
        if config.cluster_n != 1:
            return False
        if config.device_capacity == 9 and (config.tile_n == 208 or config.is_dynamic_persistent):
            return False
    return True


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
    tile_n_vals = [16, 32, 64, 128, 160, 192, 224, 256]
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


def default_config(device) -> GemmConfig:
    """Per-arch default config (canonical home; gemm_interface re-exports)."""
    from torch._vendor.quack.cute_dsl_utils import get_device_capacity

    return _default_config_for_cap(get_device_capacity(device)[0])


def blockscaled_default_config(m: int, n: int, device_capacity: int = 10) -> GemmConfig:
    """Default config for blockscaled GEMM (SM100 unless ``device_capacity``
    says otherwise).

    On SM100, large shapes use a (256, 256) tile: it makes num_acc_stage == 1,
    which turns on ``overlap_accum_sf`` (a second TMEM accumulator stage) so
    the per-tile scale-apply + TMEM drain overlaps the next tile's MMA instead
    of serializing after it.

    On SM120 (warp MMA, no clusters), (128, 128) pingpong — riding CLC via
    is_dynamic_persistent — measured best across mxfp8/nvfp4/mxfp4 at 2048³
    through 8192³ on RTX 5090 (interleaved medians, 2026-07-30). The older
    snapshot codebase's fp4-at->=8192 (256, 128) cooperative rule (see
    AI/sm120_blockscaled_gemm_worklog.md) no longer holds: (256, 128) is now
    the WORST of the three candidates at 8192³ (nvfp4 1051 vs pingpong 1354
    TF); the sole exception is mxfp4 8192³ where (128, 128) coop leads
    pingpong by ~5% — not enough for a format-special rule.
    """
    if device_capacity == 12:
        return _blockscaled_config(128, 128, (1, 1), device_capacity=12, pingpong=True)
    if m >= 512 and n >= 256:
        tile_m, tile_n, cluster = 256, 256, (2, 1)
    elif m >= 512 and n >= 128:
        tile_m, tile_n, cluster = 256, 128, (2, 1)
    else:
        tile_m, tile_n, cluster = 128, 128, (1, 1)
    return _blockscaled_config(tile_m, tile_n, cluster)


@lru_cache(maxsize=None)
def _blockscaled_config(tile_m, tile_n, cluster, device_capacity=10, pingpong=False):
    return GemmConfig(
        tile_m=tile_m,
        tile_n=tile_n,
        cluster_m=cluster[0],
        cluster_n=cluster[1],
        pingpong=pingpong,
        is_dynamic_persistent=True,
        device_capacity=device_capacity,
    )


@lru_cache(maxsize=None)
def _default_config_for_cap(cap):
    if cap == 8:
        return GemmConfig(
            tile_m=128,
            tile_n=128,
            tile_k=32,
            num_warps=4,
            cluster_m=1,
            cluster_n=1,
            pingpong=False,
            is_dynamic_persistent=False,
            device_capacity=8,
        )
    elif cap in [10, 11]:
        return GemmConfig(
            tile_m=256,
            tile_n=256,
            cluster_m=2,
            cluster_n=1,
            pingpong=False,
            is_dynamic_persistent=True,
            device_capacity=10,
        )
    elif cap == 12:
        return GemmConfig(
            tile_m=128,
            tile_n=128,
            cluster_m=1,
            cluster_n=1,
            pingpong=True,
            is_dynamic_persistent=True,
            device_capacity=12,
        )
    else:
        return GemmConfig(
            tile_m=128,
            tile_n=192,
            cluster_m=2,
            cluster_n=1,
            pingpong=True,
            is_dynamic_persistent=False,
        )
