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


def _check_mxfp8_gemm_config(gemm_config: dict[str, int]) -> None:
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
            _check_mxfp8_gemm_config(asdict(gemm))
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
            _check_mxfp8_gemm_config(asdict(gemm_config))
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


@dataclass(frozen=True)
class FlyDSLMXFP4Config:
    # TILE_K counts E2M1 *elements*; the LDS row it produces is TILE_K // 2
    # bytes wide, which is what every DMA and swizzle constant is derived from.
    TILE_M: int = 128
    TILE_N: int = 128
    TILE_K: int = 256
    STAGES: int = 2
    BLOCK_M_WARPS: int = 2
    BLOCK_N_WARPS: int = 2
    GROUP_M: int = 0
    # 1 stages the E8M0 block scales through LDS and reads each one back with a
    # single ds_read_u8; 0 keeps the in-register lane-group transpose. Searched
    # rather than derived -- see the comment in mxfp4_gemm_derived. Declared
    # last and defaulted so the 7-tuples in the default list stay valid.
    LDS_SCALE: int = 0


class FlyDSLMXFP4ConfigDict(TypedDict):
    TILE_M: int
    TILE_N: int
    TILE_K: int
    STAGES: int
    BLOCK_M_WARPS: int
    BLOCK_N_WARPS: int
    GROUP_M: int
    LDS_SCALE: int


FlyDSLMXFP4ConfigArgs = tuple[int, int, int, int, int, int, int]


def _check_mxfp4_gemm_config(gemm_config: dict[str, int]) -> None:
    """Raise if the kernel cannot build this tile config."""
    # Keep FlyDSL optional when this heuristics module is imported.
    from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
        mxfp4_gemm_derived,
    )

    mxfp4_gemm_derived(
        block_m=int(gemm_config["TILE_M"]),
        block_n=int(gemm_config["TILE_N"]),
        block_k=int(gemm_config["TILE_K"]),
        stages=int(gemm_config["STAGES"]),
        m_waves=int(gemm_config["BLOCK_M_WARPS"]),
        n_waves=int(gemm_config["BLOCK_N_WARPS"]),
        group_m=int(gemm_config["GROUP_M"]),
        lds_scale_req=int(gemm_config.get("LDS_SCALE", 0)),
    )


def is_mxfp4_config_valid_for_shape(
    m: int,
    n: int,
    k: int,
    out_dtype: str,
    gemm_config: dict[str, int],
) -> bool:
    """Return whether a FlyDSL MXFP4 config supports this concrete shape."""
    from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
        make_mxfp4_param_and_validate,
    )

    return make_mxfp4_param_and_validate(m, n, k, out_dtype, gemm_config) is not None


def get_exhaustive_mxfp4_gemm_configs() -> list[FlyDSLMXFP4Config]:
    """
    Returns the exhaustive configuration set for the gfx950 FlyDSL MXFP4 kernel.
    """
    selections = {
        "TILE_M": [16, 32, 64, 96, 128, 256],
        "TILE_N": [16, 32, 64, 96, 128, 256],
        # Elements. 128 gives a 64-byte LDS row, 1024 a 512-byte one; the range
        # is wider than the MXFP8 kernel's because an E2M1 tile costs half the
        # LDS of the E4M3 tile with the same K.
        "TILE_K": [128, 256, 512, 1024],
        "STAGES": [2, 3, 4, 5, 6],
        "BLOCK_M_WARPS": [1, 2, 4],
        "BLOCK_N_WARPS": [1, 2, 4],
        "GROUP_M": [0, 4],
        # Both variants are generated; a config that cannot express the LDS path
        # raises in _check_mxfp4_gemm_config and is dropped, so this adds
        # candidates only where the two are genuinely different kernels.
        "LDS_SCALE": [0, 1],
    }
    keys = selections.keys()
    values = selections.values()
    configs = [dict(zip(keys, combo)) for combo in product(*values)]
    valid_configs: list[FlyDSLMXFP4Config] = []
    for gemm_config in configs:
        try:
            gemm = FlyDSLMXFP4Config(**cast(FlyDSLMXFP4ConfigDict, gemm_config))
            _check_mxfp4_gemm_config(asdict(gemm))
            valid_configs.append(gemm)
        except Exception as e:
            log.debug(
                "Skipping invalid exhaustive FlyDSL MXFP4 config %s: %s", gemm_config, e
            )
    return valid_configs


def get_default_mxfp4_gemm_configs() -> list[FlyDSLMXFP4Config]:
    """
    Returns the default configuration set for the gfx950 FlyDSL MXFP4 kernel.
    """
    tile_tuples: list[FlyDSLMXFP4ConfigArgs] = [
        # TILE_K=256 elements reproduces the byte geometry the MXFP8 kernel
        # settled on (128-byte LDS rows, 8 DMA and 24 ds_read per K-tile) while
        # covering twice the K, and its two MFMA K steps let the compiler merge
        # the pair of E8M0 scale dwords into one dwordx2.
        (256, 256, 256, 2, 2, 4, 4),
        (256, 256, 256, 2, 2, 4, 0),
        (256, 256, 256, 2, 2, 2, 0),
        (128, 256, 256, 2, 1, 4, 4),
        (256, 128, 256, 2, 4, 1, 4),
        (128, 128, 256, 2, 2, 2, 0),
        (128, 128, 256, 2, 2, 2, 4),
        # Half the LDS per K-tile instead: deep pipelines that the MXFP8 kernel
        # could never fit, which is what turns the K-tile boundary from a full
        # vmcnt(0) drain into a counted wait.
        (256, 256, 128, 4, 2, 4, 0),
        (256, 256, 128, 5, 2, 4, 0),
        (256, 256, 128, 4, 2, 4, 4),
        (128, 128, 128, 4, 2, 2, 0),
        # 4 waves/SIMD. Only reachable with 4-bit operands: the same tile in
        # E4M3 needs more than the 128 VGPRs a wave gets at this occupancy.
        (256, 256, 128, 2, 4, 4, 0),
        (256, 256, 128, 4, 4, 4, 0),
        (256, 256, 256, 2, 4, 4, 0),
        # TILE_K=512 elements: four MFMA K steps per tile, so the scale dwords
        # merge into a dwordx4.
        (128, 128, 512, 2, 2, 2, 0),
        (64, 128, 512, 2, 1, 2, 0),
        # Small-M decode shapes: thin tiles keep the N sweep parallel.
        (16, 128, 256, 4, 1, 2, 0),
        (16, 256, 256, 4, 1, 4, 0),
        (32, 128, 256, 4, 1, 2, 0),
        (32, 256, 256, 3, 1, 4, 0),
        (64, 256, 256, 2, 1, 4, 0),
        # Small-N (e.g. 4096x256x4096) wants the opposite aspect ratio.
        (256, 64, 256, 2, 4, 1, 0),
        (128, 32, 256, 3, 2, 1, 0),
        # Tiny catch-all tiles. The kernel has no boundary predication, so a
        # shape only gets a FlyDSL choice when some tile divides it exactly.
        (64, 64, 256, 2, 1, 1, 0),
        (32, 32, 128, 2, 1, 1, 0),
        (16, 16, 128, 2, 1, 1, 0),
        # Measured winners. Every entry below beat this list's previous best on
        # at least one of the 13 benchmark shapes once the packed-unit scale
        # path removed the per-byte scale fallback that used to make shallow
        # register blocking unusable. Without them the default search cannot
        # reach the numbers the EXHAUSTIVE space does -- 11 of the 13 per-shape
        # winners were absent here.
        (32, 32, 512, 6, 2, 1, 4),    # 32x4096x4096
        (16, 32, 256, 4, 1, 1, 0),    # 64x4096x4096
        (32, 64, 1024, 3, 1, 4, 0),   # 32x14336x4096
        (32, 32, 512, 2, 1, 2, 0),    # 128x4096x4096
        (32, 64, 256, 2, 1, 2, 0),    # 32x28672x4096
        (64, 32, 512, 2, 1, 1, 4),    # 4096x256x4096
        (64, 64, 512, 2, 2, 2, 4),    # 256x4096x4096
        (64, 128, 512, 2, 2, 2, 4),   # 512x4096x4096
        (128, 128, 512, 2, 2, 2, 4),  # 1024x4096x4096
        (256, 256, 256, 2, 4, 4, 4),  # 4096x4096x4096
        (256, 256, 256, 2, 4, 2, 0),  # 8192x8192x8192
        # LDS-staged block scales (8th field = LDS_SCALE). Each of these was
        # measured against its own LDS_SCALE=0 twin in a paired A/B and won by
        # more than the 3.4% noise floor of this box:
        #   32,32,512,2,1,2,0   +18.5%  (128 x 4096 x 4096)
        #   32,32,512,2,1,1,0   +18.9%  (4096^3)
        #   32,64,512,2,1,2,0   +17.9%  (4096^3)
        #   64,64,512,2,2,2,4   +13.0%  (256 x 4096 x 4096)
        #   64,128,512,2,2,2,4  +11.5%  (512 x 4096 x 4096)
        #   128,64,512,2,1,2,0  +11.1%  (4096^3)
        #   64,128,512,2,2,2,0   +8.7%  (4096^3)
        #   32,64,1024,3,1,4,0   +5.8%  (32 x 14336 x 4096)
        (32, 32, 512, 2, 1, 2, 0, 1),
        (32, 32, 512, 2, 1, 1, 0, 1),
        (32, 64, 512, 2, 1, 2, 0, 1),
        (64, 64, 512, 2, 2, 2, 4, 1),
        (64, 128, 512, 2, 2, 2, 4, 1),
        (128, 64, 512, 2, 1, 2, 0, 1),
        (64, 128, 512, 2, 2, 2, 0, 1),
        (32, 64, 1024, 3, 1, 4, 0, 1),
    ]
    # Tuple order must match the FlyDSLMXFP4Config field declaration order.
    configs = [FlyDSLMXFP4Config(*args) for args in tile_tuples]
    valid_configs: list[FlyDSLMXFP4Config] = []
    for gemm_config in configs:
        try:
            _check_mxfp4_gemm_config(asdict(gemm_config))
            valid_configs.append(gemm_config)
        except Exception as e:
            log.debug(
                "Skipping invalid default FlyDSL MXFP4 config %s: %s", gemm_config, e
            )
    return valid_configs


def get_mxfp4_gemm_configs() -> list[dict[str, int]]:
    """
    Returns the shape-independent configuration set for the gfx950 FlyDSL
    MXFP4 kernel.
    """
    if (
        config.flydsl_enable_autotuning
        and config.max_autotune_gemm_search_space == "EXHAUSTIVE"
    ):
        configs = get_exhaustive_mxfp4_gemm_configs()
    else:
        configs = get_default_mxfp4_gemm_configs()
    if not configs:
        log.warning("No valid FlyDSL MXFP4 GEMM configuration is available")
        return []
    return [asdict(gemm_config) for gemm_config in configs]


def get_mxfp4_gemm_configs_for_shape(
    m: int, n: int, k: int, out_dtype: str
) -> list[dict[str, int]]:
    """
    Returns the configurations to autotune over for one concrete shape.

    This kernel has no boundary predication, so a config is only usable when its
    tile divides the shape exactly. That makes "the baseline config" shape
    dependent: with autotuning disabled we return the single default config that
    fits, preferring FlyDSLMXFP4Config() when it is valid, rather than a fixed
    tuple that many shapes would reject outright.
    """
    configs = [
        gemm_config
        for gemm_config in get_mxfp4_gemm_configs()
        if is_mxfp4_config_valid_for_shape(m, n, k, out_dtype, gemm_config)
    ]
    if config.flydsl_enable_autotuning or not configs:
        return configs
    baseline = asdict(FlyDSLMXFP4Config())
    return [baseline] if baseline in configs else configs[:1]
