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

# Unranked legal sets (capabilities without a priority list, or constrained calls
# whose legal set misses it) benchmark at most this many configs.
_MAX_UNRANKED_CANDIDATES = 12


def _sm100_priority_rank(
    order: tuple[tuple[int, int, int, int, bool], ...],
) -> dict[tuple[Any, ...], int]:
    # Keys are tile M/N, cluster M/N, pingpong, swap_ab, dynamic persistence,
    # and device capacity.
    return {
        (tile_m, tile_n, cluster_m, cluster_n, False, False, dynamic, 10): rank
        for rank, (tile_m, tile_n, cluster_m, cluster_n, dynamic) in enumerate(order)
    }


# Dense FlexGEMM preference order measured on B200 (SM100) with bf16 dense
# mm epilogues, square M=N=K in 256..1024 (#187108, 2026-06); re-tune there.
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
# the untuned varlen default.
_VARLEN_PRIORITY_RANK = _sm100_priority_rank(
    (
        (128, 128, 2, 1, True),
        (128, 256, 2, 2, False),
        (256, 256, 2, 2, False),
        (128, 128, 2, 1, False),
    )
)


def _rank_key(config: QuackConfigKey) -> tuple[Any, ...]:
    fields = dict(config)
    return (
        fields["tile_m"],
        fields["tile_n"],
        fields["cluster_m"],
        fields["cluster_n"],
        fields["pingpong"],
        fields["swap_ab"],
        fields["is_dynamic_persistent"],
        fields["device_capacity"],
    )


def _priority_rank(config: QuackConfigKey, *, varlen: bool) -> int | None:
    rank = _VARLEN_PRIORITY_RANK if varlen else _PRIORITY_RANK
    return rank.get(_rank_key(config))


# Untuned dense pick by shape hint (same measurement as _PRIORITY_RANK): ranks
# into _PRIORITY_RANK, tried in order, falling back to QuACK's default.
_DENSE_DEFAULT, _DENSE_SKINNY, _DENSE_LARGE_RECT, _DENSE_LARGE = range(4)


def _dense_default_ranks(m: int, n: int) -> tuple[int, ...]:
    min_dim, max_dim = min(m, n), max(m, n)
    if min_dim < 512 or (m == 1024 and n == 1024):
        return (_DENSE_SKINNY, _DENSE_DEFAULT)
    if max_dim >= 4096 and 768 <= min_dim < 1024:
        return (_DENSE_LARGE, _DENSE_DEFAULT)
    if max_dim >= 4096 and min_dim == 1024:
        return (_DENSE_LARGE_RECT, _DENSE_DEFAULT)
    if min_dim >= 2048:
        return (_DENSE_LARGE, _DENSE_DEFAULT)
    return (_DENSE_DEFAULT,)


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
    EXHAUSTIVE keeps everything. Legal sets that miss the priority list
    entirely (other capabilities, or constrained calls such as pinned ``swap_ab``)
    benchmark QuACK's order, capped at ``_MAX_UNRANKED_CANDIDATES``. Varlen-M
    calls rank by the grouped_mm order and only add QuACK's dense default when
    it is measured (it is the slowest common choice for ragged groups).
    """
    if inductor_config.max_autotune_gemm_search_space == "EXHAUSTIVE":
        return legal_configs
    prioritized = _prioritized(legal_configs, varlen=varlen)
    if not prioritized:
        return legal_configs[:_MAX_UNRANKED_CANDIDATES]
    default = legal_configs[0]
    if default not in prioritized and not varlen:
        prioritized.insert(0, default)
    return tuple(prioritized)


def flex_gemm_default_config(
    legal_configs: tuple[QuackConfigKey, ...],
    *,
    varlen: bool = False,
    dense_shape: tuple[int, int] | None = None,
) -> QuackConfigKey:
    """Return the config pinned when autotuning is off.

    ``dense_shape`` is the ``(M, N)`` optimization hint of a dense call; it picks
    from the measured dense table without guarding (a different runtime shape
    still runs a legal config). Varlen-M calls take the best-ranked legal
    grouped_mm config. Otherwise QuACK's untuned default (``legal_configs[0]``).
    """
    if varlen:
        if prioritized := _prioritized(legal_configs, varlen=True):
            return prioritized[0]
        return legal_configs[0]
    if dense_shape is not None:
        by_rank = {_PRIORITY_RANK.get(_rank_key(c)): c for c in legal_configs}
        for rank in _dense_default_ranks(*dense_shape):
            if (config := by_rank.get(rank)) is not None:
                return config
    return legal_configs[0]
