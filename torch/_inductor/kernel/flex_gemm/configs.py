# mypy: allow-untyped-defs
"""Inductor-owned QuACK GemmConfig search space for FlexGEMM autotuning.

QuACK decides which configs are legal for a generated epilogue; Inductor decides
which of them are worth benchmarking. The default search space is the measured
dense FlexGEMM preference order below, in priority order; the EXHAUSTIVE search
space benchmarks every legal config. Varlen-M (grouped_mm) calls use their own
measured order and default, since ragged per-group M and per-group B re-reads
favor smaller tiles than the dense default.
"""

from __future__ import annotations

from typing import Any

from torch._inductor import config as inductor_config


QuackConfigKey = tuple[tuple[str, Any], ...]


def _sm100_priority_rank(
    order: tuple[tuple[int, int, int, int, bool], ...],
) -> dict[tuple[Any, ...], int]:
    # Keys are tile M/N, cluster M/N, pingpong, swap_ab, dynamic persistence,
    # and device capacity.
    return {
        (tile_m, tile_n, cluster_m, cluster_n, False, False, dynamic, 10): rank
        for rank, (tile_m, tile_n, cluster_m, cluster_n, dynamic) in enumerate(order)
    }


# Measured dense FlexGEMM preference order on SM100.
_PRIORITY_RANK = _sm100_priority_rank(
    (
        (128, 256, 2, 1, True),
        (128, 192, 2, 1, True),
        (256, 256, 2, 1, True),
        (256, 256, 2, 2, True),
        (256, 192, 2, 1, True),
        (128, 128, 1, 1, False),
        (128, 256, 1, 1, True),
        (128, 256, 1, 1, False),
        (128, 128, 2, 1, True),
        (256, 128, 2, 1, True),
        (128, 224, 1, 1, True),
        (128, 160, 1, 1, True),
    )
)

# Measured varlen-M (grouped_mm) order on SM100 over DeepSeek-V3 16B/671B
# expert shapes (E in 8..256, balanced and skewed offs); the first entry is
# the untuned varlen default. The last two entries are the only ones a
# grouped-main store accepts (cluster_n == 1, tile_m 128 -> cluster_m 1), so
# grouped SwiGLU defaults to 128x128 c1x1 (measured at E=64) and autotunes
# 256x128 c2x1 (E=8).
_VARLEN_PRIORITY_RANK = _sm100_priority_rank(
    (
        (128, 128, 2, 1, True),
        (128, 256, 2, 2, False),
        (256, 256, 2, 2, False),
        (128, 128, 2, 1, False),
        (128, 128, 1, 1, True),
        (256, 128, 2, 1, True),
    )
)


def _priority_rank(config: QuackConfigKey, *, varlen: bool) -> int | None:
    fields = dict(config)
    return (_VARLEN_PRIORITY_RANK if varlen else _PRIORITY_RANK).get(
        (
            fields["tile_m"],
            fields["tile_n"],
            fields["cluster_m"],
            fields["cluster_n"],
            fields["pingpong"],
            fields["swap_ab"],
            fields["is_dynamic_persistent"],
            fields["device_capacity"],
        )
    )


def _prioritized(
    legal_configs: tuple[QuackConfigKey, ...], *, varlen: bool
) -> list[QuackConfigKey]:
    ranked = {
        config: rank
        for config in legal_configs
        if (rank := _priority_rank(config, varlen=varlen)) is not None
    }
    return sorted(ranked, key=ranked.__getitem__)


def flex_gemm_search_space(
    legal_configs: tuple[QuackConfigKey, ...], *, varlen: bool = False
) -> tuple[QuackConfigKey, ...]:
    """Return the legal configs Inductor benchmarks, best-known first.

    ``legal_configs`` comes from QuACK with its untuned default first. The
    default search space keeps that default plus the measured priority configs;
    EXHAUSTIVE keeps everything. Constrained calls whose legal set misses the
    priority list entirely (for example pinned ``swap_ab``) benchmark every
    legal config, since the constraints already narrowed the space. Varlen-M
    calls rank by the grouped_mm order and only add QuACK's dense default when
    it is measured (it is the slowest common choice for ragged groups).
    """
    if inductor_config.max_autotune_gemm_search_space == "EXHAUSTIVE":
        return legal_configs
    prioritized = _prioritized(legal_configs, varlen=varlen)
    if not prioritized:
        return legal_configs
    default = legal_configs[0]
    if default not in prioritized and not varlen:
        prioritized.insert(0, default)
    return tuple(prioritized)


def flex_gemm_default_config(
    legal_configs: tuple[QuackConfigKey, ...], *, varlen: bool = False
) -> QuackConfigKey:
    """Return the config pinned when autotuning is off.

    Dense calls keep QuACK's untuned default (``legal_configs[0]``); varlen-M
    calls take the best-ranked legal grouped_mm config instead.
    """
    prioritized = _prioritized(legal_configs, varlen=True) if varlen else []
    return prioritized[0] if prioritized else legal_configs[0]
