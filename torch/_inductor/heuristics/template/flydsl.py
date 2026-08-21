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


# ---------------------------------------------------------------------------
# MXFP8 grouped GEMM (gfx950)
#
# Shares this module with the BF16 GEMM heuristics above but no knobs: the
# MXFP8 kernel is not a scaled variant of that kernel. See
# FlyDSLMXFP8GroupedGemmConfig for which dimensions are fixed and why.
# ---------------------------------------------------------------------------

# OCP MX: 32 elements share one E8M0 scale. Fixed by the spec, so it is declared
# here rather than imported from the vendored kernel (which pulls in the FlyDSL
# runtime) and the lowering's shape gate stays importable without it.
MXFP8_SCALE_BLOCK = 32


@dataclass(frozen=True)
class FlyDSLMXFP8GroupedGemmConfig:
    """Tile shape of the MXFP8 grouped GEMM kernel.

    The MXFP8 kernel is not a scaled variant of the BF16 grouped kernel and
    does not share its knobs: its contraction step is fixed at 128 elements
    (one scaled-MFMA K), its wave split is fixed at 4x4 over one 256-thread
    block, and its LDS ping-pong depth is fixed at 2. What remains tunable is
    the output tile, so that is the whole config.
    """

    BLOCK_R: int = 256
    BLOCK_C: int = 256


class FlyDSLMXFP8GroupedGemmConfigDict(TypedDict):
    BLOCK_R: int
    BLOCK_C: int


# Every tile the kernel implements. BLOCK_R below 64 puts a wave-dim half under
# one MFMA tile (16 rows x 2 wave-rows) and the row structure stops holding;
# BLOCK_C below 128 does the same on the column side.
_MXFP8_GROUPED_BLOCK_R = (64, 128, 256)
_MXFP8_GROUPED_BLOCK_C = (128, 256)


def _make_mxfp8_grouped_gemm_param(
    k: int, n: int, group_count: int, gemm_config: dict[str, int]
):
    # Keep FlyDSL optional when this heuristics module is imported.
    from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
        make_mxfp8_grouped_gemm_param,
    )

    return make_mxfp8_grouped_gemm_param(
        k,
        n,
        group_count,
        block_r=int(gemm_config["BLOCK_R"]),
        block_c=int(gemm_config["BLOCK_C"]),
    )


def is_mxfp8_grouped_gemm_config_valid_for_shape(
    n: int,
    k: int,
    group_count: int,
    gemm_config: dict[str, int],
) -> bool:
    """Return whether a config supports this MXFP8 grouped GEMM shape.

    M is deliberately not an argument: the per-group row counts live on the
    device and the kernel is built not to sync on them, so nothing here may
    depend on them.
    """
    from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
        make_mxfp8_grouped_gemm_param_and_validate,
    )

    return (
        make_mxfp8_grouped_gemm_param_and_validate(
            k,
            n,
            group_count,
            int(gemm_config["BLOCK_R"]),
            int(gemm_config["BLOCK_C"]),
        )
        is not None
    )


def get_exhaustive_mxfp8_grouped_gemm_configs() -> list[FlyDSLMXFP8GroupedGemmConfig]:
    """Return every tile the MXFP8 grouped GEMM kernel implements."""
    return [
        FlyDSLMXFP8GroupedGemmConfig(BLOCK_R=block_r, BLOCK_C=block_c)
        for block_r, block_c in product(_MXFP8_GROUPED_BLOCK_R, _MXFP8_GROUPED_BLOCK_C)
    ]


def get_default_mxfp8_grouped_gemm_config(
    m: int, n: int, group_count: int
) -> FlyDSLMXFP8GroupedGemmConfig:
    """The tile the kernel's own measured heuristic picks for this shape.

    Unlike the dense and BF16-grouped defaults, this one is shape-dependent.
    It has to be: the tile trades per-block MFMA density against how much of
    a row tile a group actually fills and against how many blocks the grid
    ends up with, and both terms move with (M/G, N). ``pick_tile`` is the
    upstream kernel's own measured answer, so the single non-autotuned choice
    is the one it makes rather than a fixed tile that is wrong at both ends.
    """
    from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
        pick_mxfp8_grouped_gemm_tile,
    )

    block_r, block_c = pick_mxfp8_grouped_gemm_tile(m, group_count, n)
    return FlyDSLMXFP8GroupedGemmConfig(BLOCK_R=block_r, BLOCK_C=block_c)


def get_mxfp8_grouped_gemm_configs(
    m: int, n: int, k: int, group_count: int
) -> list[dict[str, int]]:
    """Return configs for the MXFP8 ragged grouped GEMM kernel.

    Shape compatibility beyond the tile itself is checked in the lowering
    before this function is called. By default, autotuning is disabled and we
    return only the tile ``pick_tile`` chooses.
    """
    if config.flydsl_enable_autotuning:
        candidates = get_exhaustive_mxfp8_grouped_gemm_configs()
    else:
        candidates = [get_default_mxfp8_grouped_gemm_config(m, n, group_count)]

    valid_configs: list[FlyDSLMXFP8GroupedGemmConfig] = []
    for gemm_config in candidates:
        try:
            _make_mxfp8_grouped_gemm_param(
                k,
                n,
                group_count,
                cast(dict[str, int], asdict(gemm_config)),
            )
            valid_configs.append(gemm_config)
        except Exception as e:
            log.debug(
                "Skipping invalid FlyDSL MXFP8 grouped config %s: %s", gemm_config, e
            )

    if not valid_configs:
        log.warning("No valid FlyDSL MXFP8 grouped GEMM configuration is available")
        return []
    return [asdict(gemm_config) for gemm_config in valid_configs]
