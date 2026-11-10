"""The TD scorer at runtime: the TD job's `scorer` block in td_results.json."""

from __future__ import annotations

import json
import math
from typing import Any, TYPE_CHECKING

from tools.testing.target_determination.scorer_model import (
    change_features,
    ModelError,
    parse_model,
    risk_score,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from tools.testing.target_determination.heuristics.interface import (
        AggregatedHeuristics,
        HeuristicInterface,
    )


MAX_REASON_CHARS = 500


def error_reason(e: BaseException) -> str:
    return f"error: {type(e).__name__}: {e}"[:MAX_REASON_CHARS]


def invalid_reason(e: ModelError) -> str:
    return f"invalid: {e}"[:MAX_REASON_CHARS]


def heuristic_file_scores(
    aggregated: AggregatedHeuristics,
) -> dict[str, dict[str, float]]:
    """Each heuristic's non-zero scores per test file (the max over its TestRuns).

    Non-finite scores are dropped.
    """
    out: dict[str, dict[str, float]] = {}
    for heuristic, tp in aggregated._heuristic_results.items():
        best: dict[str, float] = {}
        for test_run, score in tp._test_scores.items():
            if not math.isfinite(score):
                continue
            f = test_run.test_file
            best[f] = max(best[f], score) if f in best else score
        out[heuristic.name] = {f: s for f, s in best.items() if s != 0}
    return out


def _load_model(fetch_model: Callable[[], bytes | None]) -> tuple[Any, str]:
    raw = fetch_model()
    if raw is None:
        return None, "missing"
    try:
        model = parse_model(raw)
    except ModelError as e:
        return None, invalid_reason(e)
    if not model["enabled"]:
        return None, "disabled"
    return model, "ok"


def _build_block(
    aggregated: AggregatedHeuristics,
    heuristics: Iterable[HeuristicInterface],
    fetch_model: Callable[[], bytes | None],
    changed_files: list[str] | None,
) -> dict[str, Any]:
    model, reason = _load_model(fetch_model)
    ran = {h.name for h in aggregated._heuristic_results}
    change, risk = None, None
    try:
        if changed_files is not None:
            change = change_features(changed_files)
            if model is not None:
                risk = risk_score(model, change)
    except Exception as e:
        print(f"TD scorer: risk not computed: {error_reason(e)!r}")
    return {
        "model": model,
        "reason": reason,
        "heuristics": heuristic_file_scores(aggregated),
        "raised": [h.name for h in heuristics if h.name not in ran],
        "change": change,
        "risk": risk,
    }


def build_scorer_block(
    aggregated: AggregatedHeuristics,
    heuristics: Iterable[HeuristicInterface],
    fetch_model: Callable[[], bytes | None],
    changed_files: list[str] | None,
) -> dict[str, Any] | None:
    """The `scorer` block for td_results.json; never raises.

    Test jobs `needs:` the TD job, so an exception escaping here would skip every
    test job CI-wide. On any error the block is reduced to the reason, or None if
    even that fails, and TD's own scores are written regardless. changed_files is
    None when they could not be listed; risk is then null.
    """
    try:
        try:
            block = _build_block(aggregated, heuristics, fetch_model, changed_files)
            json.dumps(block, allow_nan=False)
        except Exception as e:
            block = {"model": None, "reason": error_reason(e)}
        model = block["model"]
        version = model["version"] if model is not None else None
        reason = str(block["reason"])[:MAX_REASON_CHARS]
        print(
            f"TD scorer: reason={reason!r} version={version!r} "
            f"risk={block.get('risk')!r}"
        )
        return block
    except Exception as e:
        print(f"TD scorer: omitted: {type(e).__name__}")
        return None
