import functools
import logging
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Callable, Literal, TypeVar

import torch._inductor.config as config


log = logging.getLogger(__name__)

MXFPFormat = Literal["mxfp4", "mxfp8"]


@dataclass(frozen=True)
class FlyDSLMXFPConfig:
    TILE_M: int = 128
    TILE_N: int = 128
    TILE_K: int = 128
    STAGES: int = 2
    M_WAVES: int = 2
    N_WAVES: int = 2
    GROUP_M: int = 0
    LDS_SCALE: int = 0


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


FlyDSLGemmConfigArgs = tuple[int, int, int, int, int, int, int]
FlyDSLHTIGemmConfigArgs = tuple[int, int, int, int, int, int, int, bool]
FlyDSLMXFPConfigArgs = tuple[int, int, int, int, int, int, int, int]


_MXFP_DEFAULT_CONFIG_ARGS: dict[
    MXFPFormat, tuple[FlyDSLMXFPConfigArgs, ...]
] = {
    "mxfp4": (
        (16, 16, 512, 6, 1, 1, 4, 0),
        (32, 64, 512, 4, 1, 2, 4, 1),
        (32, 64, 512, 2, 1, 2, 0, 1),
        (32, 32, 512, 2, 1, 2, 0, 1),
        (32, 32, 512, 2, 2, 1, 0, 1),
        (64, 64, 512, 3, 2, 2, 0, 1),
        (64, 64, 512, 2, 2, 2, 4, 1),
        (128, 128, 512, 2, 2, 4, 0, 1),
        (128, 128, 256, 2, 2, 2, 0, 1),
        (64, 64, 512, 3, 2, 2, 4, 1),
        (256, 256, 256, 2, 2, 2, 4, 1),
    ),
    "mxfp8": (
        (16, 16, 512, 3, 1, 1, 4, 1),
        (32, 32, 256, 3, 1, 1, 4, 1),
        (32, 64, 256, 3, 1, 1, 4, 1),
        (32, 32, 512, 4, 1, 2, 4, 1),
        (32, 32, 512, 2, 2, 1, 4, 1),
        (64, 64, 512, 2, 2, 2, 0, 1),
        (64, 64, 256, 2, 1, 2, 4, 1),
        (128, 128, 256, 2, 4, 1, 4, 1),
        (128, 128, 128, 2, 1, 2, 0, 1),
        (64, 64, 512, 2, 2, 2, 4, 1),
        (256, 256, 128, 2, 2, 2, 4, 1),
        (256, 256, 128, 2, 2, 2, 0, 1),
    ),
}


_BASELINE_CONFIG: dict[MXFPFormat, FlyDSLMXFPConfig] = {
    "mxfp4": FlyDSLMXFPConfig(TILE_K=256),
    "mxfp8": FlyDSLMXFPConfig(),
}


_Config = TypeVar("_Config")


def _expand_config_space(
    config_type: Callable[..., _Config], selections: dict[str, list[Any]]
) -> list[_Config]:
    keys = selections.keys()
    return [
        config_type(**dict(zip(keys, values)))
        for values in product(*selections.values())
    ]


def _valid_configs(
    candidates: list[_Config],
    validator: Callable[[_Config], None],
    exception_types: type[Exception] | tuple[type[Exception], ...],
    label: str,
) -> list[_Config]:
    valid_configs = []
    for candidate in candidates:
        try:
            validator(candidate)
            valid_configs.append(candidate)
        except exception_types as error:
            log.debug("Skipping invalid %s config %s: %s", label, candidate, error)
    return valid_configs


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


def _check_mxfp_gemm_config(
    mxfp_format: MXFPFormat, gemm_config: dict[str, int]
) -> None:
    from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
        mxfp_gemm_derived,
    )

    mxfp_gemm_derived(
        mxfp_format,
        block_m=int(gemm_config["TILE_M"]),
        block_n=int(gemm_config["TILE_N"]),
        block_k=int(gemm_config["TILE_K"]),
        stages=int(gemm_config["STAGES"]),
        m_waves=int(gemm_config["M_WAVES"]),
        n_waves=int(gemm_config["N_WAVES"]),
        group_m=int(gemm_config["GROUP_M"]),
        lds_scale_req=int(gemm_config.get("LDS_SCALE", 0)),
    )


def is_mxfp_config_valid_for_shape(
    mxfp_format: MXFPFormat,
    m: int,
    n: int,
    k: int,
    out_dtype: str,
    gemm_config: dict[str, int],
    a_is_transposed: bool = False,
    b_is_transposed: bool = True,
) -> bool:
    from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
        make_mxfp_param_and_validate,
    )

    return (
        make_mxfp_param_and_validate(
            mxfp_format,
            m,
            n,
            k,
            out_dtype,
            gemm_config,
            a_is_transposed=a_is_transposed,
            b_is_transposed=b_is_transposed,
        )
        is not None
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
    candidates = _expand_config_space(FlyDSLGemmConfig, selections)
    candidates = [
        candidate
        for candidate in candidates
        if candidate.USE_HALF_TILE_INTERLEAVED
        or (
            candidate.TILE_M // candidate.M_WAVES // 16 <= 4
            and candidate.TILE_N // candidate.N_WAVES // 16 <= 4
        )
    ]
    return _valid_configs(
        candidates,
        lambda candidate: _make_gemm_param(asdict(candidate)),
        Exception,
        "exhaustive FlyDSL",
    )


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
    return _valid_configs(
        configs,
        lambda candidate: _make_gemm_param(asdict(candidate)),
        Exception,
        "default FlyDSL",
    )


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


def _project_mxfp_gemm_configs(
    mxfp_format: MXFPFormat, gemm_configs: list[FlyDSLGemmConfig]
) -> list[FlyDSLMXFPConfig]:
    tile_k_multiplier = 4 if mxfp_format == "mxfp4" else 2
    return [
        FlyDSLMXFPConfig(
            TILE_M=gemm_config.TILE_M,
            TILE_N=gemm_config.TILE_N,
            TILE_K=gemm_config.TILE_K * tile_k_multiplier,
            STAGES=gemm_config.STAGES,
            M_WAVES=gemm_config.M_WAVES,
            N_WAVES=gemm_config.N_WAVES,
            GROUP_M=gemm_config.GROUP_M,
            LDS_SCALE=lds_scale,
        )
        for gemm_config in gemm_configs
        if not gemm_config.USE_HALF_TILE_INTERLEAVED
        for lds_scale in (0, 1)
    ]


def _get_valid_mxfp_gemm_configs(
    mxfp_format: MXFPFormat, candidates: list[FlyDSLMXFPConfig]
) -> list[FlyDSLMXFPConfig]:
    return _valid_configs(
        list(dict.fromkeys(candidates)),
        lambda candidate: _check_mxfp_gemm_config(
            mxfp_format, asdict(candidate)
        ),
        ValueError,
        f"FlyDSL {mxfp_format}",
    )


@functools.cache
def get_exhaustive_mxfp_gemm_configs(
    mxfp_format: MXFPFormat,
) -> list[FlyDSLMXFPConfig]:
    selections = {
        "TILE_M": [16, 32, 64, 96, 128, 256],
        "TILE_N": [16, 32, 64, 96, 128, 256],
        "TILE_K": [128, 256, 512, 1024]
        if mxfp_format == "mxfp4"
        else [128, 256, 512],
        "STAGES": list(range(2, 7)),
        "M_WAVES": [1, 2, 4],
        "N_WAVES": [1, 2, 4],
        "GROUP_M": [0, 4],
        "LDS_SCALE": [0, 1],
    }
    candidates = _expand_config_space(FlyDSLMXFPConfig, selections)
    return _get_valid_mxfp_gemm_configs(mxfp_format, candidates)


@functools.cache
def get_default_mxfp_gemm_configs(
    mxfp_format: MXFPFormat,
) -> list[FlyDSLMXFPConfig]:
    candidates = [_BASELINE_CONFIG[mxfp_format]]
    candidates.extend(
        FlyDSLMXFPConfig(*args) for args in _MXFP_DEFAULT_CONFIG_ARGS[mxfp_format]
    )
    candidates.extend(
        _project_mxfp_gemm_configs(mxfp_format, get_default_gemm_configs())
    )
    return _get_valid_mxfp_gemm_configs(mxfp_format, candidates)


def get_mxfp_gemm_configs(mxfp_format: MXFPFormat) -> list[dict[str, int]]:
    if (
        config.flydsl_enable_autotuning
        and config.max_autotune_gemm_search_space == "EXHAUSTIVE"
    ):
        configs = get_exhaustive_mxfp_gemm_configs(mxfp_format)
    else:
        configs = get_default_mxfp_gemm_configs(mxfp_format)
    if not configs:
        log.warning("No valid FlyDSL %s GEMM configuration is available", mxfp_format)
        return []
    return [asdict(gemm_config) for gemm_config in configs]


def _get_exhaustive_gfx950_grouped_gemm_configs() -> list[FlyDSLGemmConfig]:
    """Return exhaustive configs for the gfx950 FlyDSL grouped GEMM kernel."""
    from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
        is_grouped_gemm_gfx950_layout_valid,
    )

    return [
        gemm_config
        for gemm_config in get_exhaustive_gemm_configs()
        if is_grouped_gemm_gfx950_layout_valid(
            gemm_config.TILE_M,
            gemm_config.TILE_N,
            gemm_config.M_WAVES,
            gemm_config.N_WAVES,
            gemm_config.USE_HALF_TILE_INTERLEAVED,
        )
    ]


@functools.cache
def get_exhaustive_grouped_gemm_configs() -> list[FlyDSLGemmConfig]:
    """Return exhaustive configs for the active FlyDSL grouped GEMM backend."""
    return _get_exhaustive_gfx950_grouped_gemm_configs()


# Baseline used when FlyDSL autotuning is disabled. The dataclass defaults
# describe the dense kernel, whose 4x4 wave split does not apply here, so the
# grouped baseline is named explicitly; it must stay in the candidate list below.
DEFAULT_GROUPED_GEMM_CONFIG = FlyDSLGemmConfig(128, 128, 64, 2, 1, 4, 0)


def _get_default_gfx950_grouped_gemm_configs() -> list[FlyDSLGemmConfig]:
    """Return default configs for the gfx950 grouped B-LDS transpose kernel."""
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
    return _valid_configs(
        candidates,
        lambda candidate: _make_gemm_param(asdict(candidate)),
        Exception,
        "default FlyDSL grouped",
    )


@functools.cache
def get_default_grouped_gemm_configs() -> list[FlyDSLGemmConfig]:
    """Return default configs for the active FlyDSL grouped GEMM backend."""
    return _get_default_gfx950_grouped_gemm_configs()


def is_grouped_gemm_config_valid_for_shape(
    m: int,
    n: int,
    k: int,
    dtype_id: int,
    gemm_config: dict[str, int | bool],
) -> bool:
    """Return whether a FlyDSL config supports this grouped GEMM shape."""
    from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
        is_grouped_gemm_gfx950_layout_valid,
    )

    tile_m = int(gemm_config["TILE_M"])
    tile_n = int(gemm_config["TILE_N"])
    tile_k = int(gemm_config["TILE_K"])
    stages = int(gemm_config["STAGES"])
    m_waves = int(gemm_config["M_WAVES"])
    n_waves = int(gemm_config["N_WAVES"])
    use_half_tile_interleaved = bool(gemm_config["USE_HALF_TILE_INTERLEAVED"])
    k_tiles = (k + tile_k - 1) // tile_k
    # The staged kernel prefetches stages-1 K tiles before the main loop.
    has_enough_k = use_half_tile_interleaved or k_tiles >= stages - 1
    # B uses a max-size buffer descriptor, so partial N tiles can issue
    # unguarded vector loads past the allocation. Predicated C stores do not
    # make those reads safe.
    return (
        tile_m <= max(128, m)
        and n >= tile_n
        and n % tile_n == 0
        and has_enough_k
        and is_grouped_gemm_gfx950_layout_valid(
            tile_m, tile_n, m_waves, n_waves, use_half_tile_interleaved
        )
        # Grouped GEMM only accepts row-major A [M, K] and B [G, K, N].
        and is_gemm_config_valid_for_shape(
            m,
            n,
            k,
            dtype_id,
            gemm_config,
            a_is_transposed=False,
            b_is_transposed=False,
        )
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


def get_mxfp_gemm_configs_for_shape(
    mxfp_format: MXFPFormat,
    m: int,
    n: int,
    k: int,
    out_dtype: str,
    *,
    a_is_transposed: bool = False,
    b_is_transposed: bool = True,
) -> list[dict[str, int]]:
    configs = [
        gemm_config
        for gemm_config in get_mxfp_gemm_configs(mxfp_format)
        if is_mxfp_config_valid_for_shape(
            mxfp_format,
            m,
            n,
            k,
            out_dtype,
            gemm_config,
            a_is_transposed=a_is_transposed,
            b_is_transposed=b_is_transposed,
        )
    ]
    if config.flydsl_enable_autotuning or not configs:
        return configs
    baseline = asdict(_BASELINE_CONFIG[mxfp_format])
    return [baseline] if baseline in configs else configs[:1]
