import logging
from dataclasses import asdict, dataclass
from itertools import product
from typing import cast, Literal, TypedDict

import torch._inductor.config as config


log = logging.getLogger(__name__)

MXFPFormat = Literal["mxfp4", "mxfp8"]


@dataclass(frozen=True)
class FlyDSLMXFPConfig:
    """Tile configuration shared by the gfx950 MXFP4 and MXFP8 kernels."""

    TILE_M: int = 128
    TILE_N: int = 128
    # TILE_K is always logical elements. MXFP4 stores half as many K bytes.
    TILE_K: int = 128
    STAGES: int = 2
    BLOCK_M_WARPS: int = 2
    BLOCK_N_WARPS: int = 2
    GROUP_M: int = 0
    # Scale staging is shared but remains an autotune choice for each format.
    LDS_SCALE: int = 0


class FlyDSLMXFPConfigDict(TypedDict):
    TILE_M: int
    TILE_N: int
    TILE_K: int
    STAGES: int
    BLOCK_M_WARPS: int
    BLOCK_N_WARPS: int
    GROUP_M: int
    LDS_SCALE: int


FlyDSLMXFPConfigArgs = tuple[int, int, int, int, int, int, int, int]


_DEFAULT_CONFIG_ARGS: dict[MXFPFormat, tuple[FlyDSLMXFPConfigArgs, ...]] = {
    "mxfp8": (
        (256, 256, 128, 2, 2, 2, 4, 0),
        (256, 256, 128, 2, 2, 2, 0, 0),
        (128, 128, 128, 2, 1, 2, 4, 0),
        (128, 128, 128, 2, 2, 1, 4, 0),
        (128, 128, 128, 2, 1, 1, 4, 0),
        (128, 256, 128, 2, 1, 4, 4, 0),
        (256, 128, 128, 2, 4, 1, 4, 0),
        (128, 128, 128, 2, 2, 2, 0, 0),
        (128, 128, 128, 2, 2, 2, 4, 0),
        (128, 128, 128, 3, 2, 2, 0, 0),
        (256, 128, 128, 2, 4, 2, 0, 0),
        (128, 256, 128, 2, 2, 4, 0, 0),
        (256, 256, 128, 2, 4, 4, 0, 0),
        (128, 128, 256, 2, 2, 2, 0, 0),
        (64, 64, 128, 2, 2, 2, 0, 0),
        (64, 128, 128, 2, 1, 2, 0, 0),
        (128, 64, 128, 2, 2, 1, 0, 0),
        (64, 64, 256, 2, 1, 1, 0, 0),
        (64, 64, 512, 2, 1, 1, 0, 0),
        (16, 128, 128, 4, 1, 2, 0, 0),
        (16, 256, 128, 4, 1, 4, 0, 0),
        (32, 128, 128, 4, 1, 2, 0, 0),
        (32, 256, 128, 3, 1, 4, 0, 0),
        (32, 128, 256, 3, 1, 2, 0, 0),
        (64, 256, 128, 2, 1, 4, 0, 0),
        (256, 64, 128, 2, 4, 1, 0, 0),
        (128, 32, 128, 3, 2, 1, 0, 0),
        (32, 32, 128, 2, 1, 1, 0, 0),
        (16, 16, 128, 2, 1, 1, 0, 0),
    ),
    "mxfp4": (
        (256, 256, 256, 2, 2, 4, 4, 0),
        (256, 256, 256, 2, 2, 4, 0, 0),
        (256, 256, 256, 2, 2, 2, 0, 0),
        (128, 256, 256, 2, 1, 4, 4, 0),
        (256, 128, 256, 2, 4, 1, 4, 0),
        (128, 128, 256, 2, 2, 2, 0, 0),
        (128, 128, 256, 2, 2, 2, 4, 0),
        (256, 256, 128, 4, 2, 4, 0, 0),
        (256, 256, 128, 5, 2, 4, 0, 0),
        (256, 256, 128, 4, 2, 4, 4, 0),
        (128, 128, 128, 4, 2, 2, 0, 0),
        (256, 256, 128, 2, 4, 4, 0, 0),
        (256, 256, 128, 4, 4, 4, 0, 0),
        (256, 256, 256, 2, 4, 4, 0, 0),
        (128, 128, 512, 2, 2, 2, 0, 0),
        (64, 128, 512, 2, 1, 2, 0, 0),
        (16, 128, 256, 4, 1, 2, 0, 0),
        (16, 256, 256, 4, 1, 4, 0, 0),
        (32, 128, 256, 4, 1, 2, 0, 0),
        (32, 256, 256, 3, 1, 4, 0, 0),
        (64, 256, 256, 2, 1, 4, 0, 0),
        (256, 64, 256, 2, 4, 1, 0, 0),
        (128, 32, 256, 3, 2, 1, 0, 0),
        (64, 64, 256, 2, 1, 1, 0, 0),
        (32, 32, 128, 2, 1, 1, 0, 0),
        (16, 16, 128, 2, 1, 1, 0, 0),
        (32, 32, 512, 6, 2, 1, 4, 0),
        (16, 32, 256, 4, 1, 1, 0, 0),
        (32, 64, 1024, 3, 1, 4, 0, 0),
        (32, 32, 512, 2, 1, 2, 0, 0),
        (32, 64, 256, 2, 1, 2, 0, 0),
        (64, 32, 512, 2, 1, 1, 4, 0),
        (64, 64, 512, 2, 2, 2, 4, 0),
        (64, 128, 512, 2, 2, 2, 4, 0),
        (128, 128, 512, 2, 2, 2, 4, 0),
        (256, 256, 256, 2, 4, 4, 4, 0),
        (256, 256, 256, 2, 4, 2, 0, 0),
        (32, 32, 512, 2, 1, 2, 0, 1),
        (32, 32, 512, 2, 1, 1, 0, 1),
        (32, 64, 512, 2, 1, 2, 0, 1),
        (64, 64, 512, 2, 2, 2, 4, 1),
        (64, 128, 512, 2, 2, 2, 4, 1),
        (128, 64, 512, 2, 1, 2, 0, 1),
        (64, 128, 512, 2, 2, 2, 0, 1),
        (32, 64, 1024, 3, 1, 4, 0, 1),
    ),
}


_BASELINE_CONFIG: dict[MXFPFormat, FlyDSLMXFPConfig] = {
    "mxfp8": FlyDSLMXFPConfig(),
    "mxfp4": FlyDSLMXFPConfig(TILE_K=256),
}


def _check_mxfp_gemm_config(
    mxfp_format: MXFPFormat, gemm_config: dict[str, int]
) -> None:
    """Raise if the unified kernel cannot build this format and tile."""
    from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
        mxfp_gemm_derived,
    )

    mxfp_gemm_derived(
        mxfp_format,
        block_m=int(gemm_config["TILE_M"]),
        block_n=int(gemm_config["TILE_N"]),
        block_k=int(gemm_config["TILE_K"]),
        stages=int(gemm_config["STAGES"]),
        m_waves=int(gemm_config["BLOCK_M_WARPS"]),
        n_waves=int(gemm_config["BLOCK_N_WARPS"]),
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
) -> bool:
    from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
        make_mxfp_param_and_validate,
    )

    return (
        make_mxfp_param_and_validate(
            mxfp_format, m, n, k, out_dtype, gemm_config
        )
        is not None
    )


def get_exhaustive_mxfp_gemm_configs(
    mxfp_format: MXFPFormat,
) -> list[FlyDSLMXFPConfig]:
    selections = {
        "TILE_M": [16, 32, 64, 96, 128, 256],
        "TILE_N": [16, 32, 64, 96, 128, 256],
        "TILE_K": [128, 256, 512, 1024]
        if mxfp_format == "mxfp4"
        else [128, 256, 512],
        "STAGES": [2, 3, 4, 5, 6],
        "BLOCK_M_WARPS": [1, 2, 4],
        "BLOCK_N_WARPS": [1, 2, 4],
        "GROUP_M": [0, 4],
        "LDS_SCALE": [0, 1],
    }
    keys = selections.keys()
    configs = [dict(zip(keys, combo)) for combo in product(*selections.values())]
    valid_configs: list[FlyDSLMXFPConfig] = []
    for gemm_config in configs:
        try:
            gemm = FlyDSLMXFPConfig(
                **cast(FlyDSLMXFPConfigDict, gemm_config)
            )
            _check_mxfp_gemm_config(mxfp_format, asdict(gemm))
            valid_configs.append(gemm)
        except Exception as error:
            log.debug(
                "Skipping invalid exhaustive FlyDSL %s config %s: %s",
                mxfp_format,
                gemm_config,
                error,
            )
    return valid_configs


def get_default_mxfp_gemm_configs(
    mxfp_format: MXFPFormat,
) -> list[FlyDSLMXFPConfig]:
    configs = [FlyDSLMXFPConfig(*args) for args in _DEFAULT_CONFIG_ARGS[mxfp_format]]
    valid_configs: list[FlyDSLMXFPConfig] = []
    for gemm_config in configs:
        try:
            _check_mxfp_gemm_config(mxfp_format, asdict(gemm_config))
            valid_configs.append(gemm_config)
        except Exception as error:
            log.debug(
                "Skipping invalid default FlyDSL %s config %s: %s",
                mxfp_format,
                gemm_config,
                error,
            )
    return valid_configs


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


def get_mxfp_gemm_configs_for_shape(
    mxfp_format: MXFPFormat, m: int, n: int, k: int, out_dtype: str
) -> list[dict[str, int]]:
    """Return format-specific configurations valid for one concrete shape."""
    configs = [
        gemm_config
        for gemm_config in get_mxfp_gemm_configs(mxfp_format)
        if is_mxfp_config_valid_for_shape(
            mxfp_format, m, n, k, out_dtype, gemm_config
        )
    ]
    if config.flydsl_enable_autotuning or not configs:
        return configs
    baseline = asdict(_BASELINE_CONFIG[mxfp_format])
    return [baseline] if baseline in configs else configs[:1]
