"""The TD scorer at runtime: the TD job's `scorer` block in td_results.json and
run_test.py's cut."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from datetime import date, datetime, UTC
from pathlib import Path
from typing import Any, TYPE_CHECKING

from tools.testing.target_determination.heuristics import HEURISTICS
from tools.testing.target_determination.heuristics.edited_by_pr import EditedByPR
from tools.testing.target_determination.heuristics.previously_failed_in_pr import (
    PreviouslyFailedInPR,
)
from tools.testing.target_determination.scorer_model import (
    change_features,
    file_probability,
    ModelError,
    parse_model,
    risk_score,
    start_up_seconds,
    validate_model,
)
from tools.testing.test_run import TestRun


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from tools.testing.target_determination.heuristics.interface import (
        AggregatedHeuristics,
        HeuristicInterface,
    )


REPO_ROOT = Path(__file__).resolve().parents[3]
TD_RESULTS_PATH = REPO_ROOT / ".additional_ci_files" / "td_results.json"
MAX_REASON_CHARS = 500
MUST_RUN_HEURISTICS = (EditedByPR.__name__, PreviouslyFailedInPR.__name__)
TRIAL_HEURISTICS = frozenset(h.name for h in HEURISTICS if h.trial_mode)
MAX_MODEL_AGE_DAYS = 7


class NoModel(Exception):
    pass


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


def _is_stale(version: str, today: date) -> bool:
    """version is the refit's ISO date; one from the future by more than a day,
    or that does not parse, is stale too."""
    try:
        age = (today - date.fromisoformat(version)).days
    except (ValueError, TypeError):
        return True
    return not -1 <= age <= MAX_MODEL_AGE_DAYS


def _load_model(
    fetch_model: Callable[[], bytes | None], today: date | None
) -> tuple[Any, str]:
    raw = fetch_model()
    if raw is None:
        return None, "missing"
    try:
        model = parse_model(raw)
    except ModelError as e:
        return None, invalid_reason(e)
    if not model["enabled"]:
        return None, "disabled"
    if today is None:
        today = datetime.now(UTC).date()
    if _is_stale(model["version"], today):
        return None, f"stale: {model['version']}"
    return model, "ok"


def _build_block(
    aggregated: AggregatedHeuristics,
    heuristics: Iterable[HeuristicInterface],
    fetch_model: Callable[[], bytes | None],
    changed_files: list[str] | None,
    today: date | None,
) -> dict[str, Any]:
    heuristics = list(heuristics)
    model, reason = _load_model(fetch_model, today)
    ran = {h.name for h in aggregated._heuristic_results}
    unknown = None
    if model is not None:
        known = {h.name for h in heuristics}
        unknown = sorted(set(model["scorer"]["heuristics"]) - known)
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
        "unknown": unknown,
        "change": change,
        "risk": risk,
    }


def build_scorer_block(
    aggregated: AggregatedHeuristics,
    heuristics: Iterable[HeuristicInterface],
    fetch_model: Callable[[], bytes | None],
    changed_files: list[str] | None,
    today: date | None = None,
) -> dict[str, Any] | None:
    """The `scorer` block for td_results.json; never raises.

    Test jobs `needs:` the TD job, so an exception escaping here would skip every
    test job CI-wide. On any error the block is reduced to the reason, or None if
    even that fails, and TD's own scores are written regardless. changed_files is
    None when they could not be listed; risk is then null. A model whose version
    is more than MAX_MODEL_AGE_DAYS before today (UTC by default) is dropped as
    stale. `unknown` lists the model's heuristics that are not in `heuristics`:
    they score 0 for every file.
    """
    try:
        try:
            block: dict[str, Any] = _build_block(
                aggregated, heuristics, fetch_model, changed_files, today
            )
            json.dumps(block, allow_nan=False)
        except Exception as e:
            block = {"model": None, "reason": error_reason(e)}
        model = block["model"]
        version = model["version"] if model is not None else None
        reason = str(block["reason"])[:MAX_REASON_CHARS]
        print(
            f"TD scorer: reason={reason!r} version={version!r} "
            f"risk={block.get('risk')!r} unknown={block.get('unknown')!r}"
        )
        return block
    except Exception as e:
        print(f"TD scorer: omitted: {type(e).__name__}")
        return None


def load_scorer_block(td_results_path: Path) -> Any:
    """The raw `scorer` block of td_results.json, or None if it has none."""
    if not td_results_path.exists():
        return None
    with open(td_results_path) as f:
        return json.load(f).get("scorer")


def _selection_hash(files: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(files)).encode()).hexdigest()[:12]


def _unique_files(tests: Iterable[TestRun]) -> list[str]:
    return list(dict.fromkeys(t.test_file for t in tests))


def _seconds_lookup(times: Mapping[str, float]) -> Callable[[str], float]:
    median = statistics.median(times.values()) if times else 0.0
    return lambda f: times.get(f, median)


def _info(
    cut: str,
    reason: str,
    td_files: list[str],
    td_seconds: float | None,
    version: str | None = None,
    budget_ms: int | None = None,
    cost_ms: int | None = None,
    start_up: float | None = None,
    files: list[str] | None = None,
    seconds: float | None = None,
) -> dict[str, Any]:
    if files is None:
        files, seconds = td_files, td_seconds
    return {
        "cut": cut,
        "reason": reason[:MAX_REASON_CHARS],
        "version": version,
        "budget_ms": budget_ms,
        "cost_ms": cost_ms,
        "start_up": start_up,
        "n": len(files),
        "seconds": seconds,
        "hash": _selection_hash(files),
        "td_n": len(td_files),
        "td_seconds": td_seconds,
    }


def select_tests(
    block: Any,
    td_include: list[TestRun],
    td_exclude: list[TestRun],
    times: Mapping[str, float],
    build_env_key: str,
    test_config: str,
) -> tuple[list[TestRun], list[TestRun], dict[str, Any]]:
    """The scorer's cut at the cost of TD's own cut, as whole test files.

    A file's cost is its recorded seconds plus the model's start_up_seconds for
    this build_env_key and test_config; the budget is the cost of TD's include
    list. Files with a non-zero score from a must-run heuristic come first, by
    name, whatever their cost; the rest fill what is left of the budget by
    p / cost. Costs are summed in integer milliseconds so the order of summation
    cannot change what fits. Raises when the block carries no usable model or
    the model has no start-up for this job. Trial-mode heuristics score nothing,
    as they reorder nothing in TD. Every shard computes the same selection: this
    depends only on its arguments.
    """
    if not isinstance(block, dict):
        raise NoModel("no scorer block")
    if block.get("model") is None:
        raise NoModel(f"no model: {block.get('reason')}")
    model = validate_model(block["model"])
    if not model["enabled"]:
        raise NoModel("disabled")
    start_up = start_up_seconds(model, build_env_key, test_config)
    if start_up is None:
        raise NoModel(f"no start-up for {build_env_key}/{test_config}")
    heuristics = block["heuristics"]
    raised = set(block["raised"]) | TRIAL_HEURISTICS
    seconds = _seconds_lookup(times)

    def cost(f: str) -> float:
        return seconds(f) + start_up

    def cost_ms(f: str) -> int:
        return round(cost(f) * 1000)

    td_files = _unique_files(td_include)
    budget = sum(cost_ms(f) for f in td_files)
    files = _unique_files(td_include + td_exclude)
    must_run = {
        f
        for name in MUST_RUN_HEURISTICS
        if name not in raised
        for f, score in heuristics.get(name, {}).items()
        if score != 0
    }
    selected = sorted(f for f in files if f in must_run)
    used = sum(cost_ms(f) for f in selected)

    ranked = []
    for f in files:
        if f in must_run:
            continue
        scores = {name: h.get(f, 0.0) for name, h in heuristics.items()}
        c = cost(f)
        p = file_probability(model, scores, c, build_env_key, test_config, raised)
        ranked.append((-p / c, f))
    ranked.sort()

    rest: list[str] = []
    for _, f in ranked:
        c_ms = cost_ms(f)
        if used + c_ms <= budget:
            selected.append(f)
            used += c_ms
        else:
            rest.append(f)
    info = _info(
        "scorer",
        "ok",
        td_files,
        sum(seconds(f) for f in td_files),
        version=model["version"],
        budget_ms=budget,
        cost_ms=used,
        start_up=start_up,
        files=selected,
        seconds=sum(seconds(f) for f in selected),
    )
    return [TestRun(f) for f in selected], [TestRun(f) for f in rest], info


def scorer_cut(
    td_results_path: Path,
    td_include: list[TestRun],
    td_exclude: list[TestRun],
    times: Mapping[str, float],
    build_env_key: str,
    test_config: str,
) -> tuple[list[TestRun], list[TestRun], dict[str, Any]]:
    """select_tests on td_results.json's block; TD's own cut on any failure."""
    try:
        block = load_scorer_block(td_results_path)
        return select_tests(
            block, td_include, td_exclude, times, build_env_key, test_config
        )
    except Exception as e:
        if isinstance(e, NoModel):
            reason = str(e)
        elif isinstance(e, ModelError):
            reason = invalid_reason(e)
        else:
            reason = error_reason(e)
        td_files = _unique_files(td_include)
        td_seconds = None
        try:
            seconds = _seconds_lookup(times)
            td_seconds = sum(seconds(f) for f in td_files)
        except Exception as seconds_error:
            print(f"TD scorer cut: no test times: {error_reason(seconds_error)!r}")
        return td_include, td_exclude, _info("td", reason, td_files, td_seconds)
