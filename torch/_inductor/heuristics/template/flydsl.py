"""FlyDSL template heuristics for the gfx950 MXFP8 grouped GEMM.

The FlyDSL template compilation infrastructure landed in #192877; this module
supplies the config schema and shape validation for the MXFP8 grouped GEMM
kernel vendored alongside it.
"""

import logging
from dataclasses import asdict, dataclass
from itertools import product
from typing import cast, TypedDict

import torch._inductor.config as config


log = logging.getLogger(__name__)


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
