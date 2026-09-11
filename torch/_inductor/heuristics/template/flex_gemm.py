# mypy: allow-untyped-defs
"""Inductor-owned QuACK GemmConfig search space for FlexGEMM autotuning.

QuACK decides which configs are legal for a generated epilogue; Inductor decides
which of them are worth benchmarking. The default search space is the measured
dense FlexGEMM preference order below, in priority order; the EXHAUSTIVE search
space benchmarks every legal config.
"""

from __future__ import annotations

from typing import Any

from torch._inductor import config as inductor_config


QuackConfigKey = tuple[tuple[str, Any], ...]

# Measured dense FlexGEMM preference order on SM100. Keys are tile M/N,
# cluster M/N, pingpong, swap_ab, dynamic persistence, and device capacity.
_PRIORITY_RANK = {
    (tile_m, tile_n, cluster_m, cluster_n, False, False, dynamic, 10): rank
    for rank, (tile_m, tile_n, cluster_m, cluster_n, dynamic) in enumerate(
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
}


def _priority_rank(config: QuackConfigKey) -> int | None:
    fields = dict(config)
    return _PRIORITY_RANK.get(
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


def flex_gemm_search_space(
    legal_configs: tuple[QuackConfigKey, ...],
) -> tuple[QuackConfigKey, ...]:
    """Return the legal configs Inductor benchmarks, best-known first.

    ``legal_configs`` comes from QuACK with its untuned default first. The
    default search space keeps that default plus the measured priority configs;
    EXHAUSTIVE keeps everything. Constrained calls whose legal set misses the
    priority list entirely (for example pinned ``swap_ab``) benchmark every
    legal config, since the constraints already narrowed the space.
    """
    if inductor_config.max_autotune_gemm_search_space == "EXHAUSTIVE":
        return legal_configs
    ranked = {
        config: rank
        for config in legal_configs
        if (rank := _priority_rank(config)) is not None
    }
    prioritized = sorted(ranked, key=ranked.__getitem__)
    if not prioritized:
        return legal_configs
    default = legal_configs[0]
    if default not in prioritized:
        prioritized.insert(0, default)
    return tuple(prioritized)
