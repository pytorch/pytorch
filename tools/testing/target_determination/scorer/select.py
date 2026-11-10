"""run_test.py's TD cut with the learned scorer: the same number of test files
as TD's cut, ordered by p / cost ** a, with `a` picked on the job so the cut
costs what TD's own cut would. Any problem gives TD's cut."""

from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, NamedTuple, TYPE_CHECKING

from tools.stats.import_test_stats import ADDITIONAL_CI_FILES_FOLDER
from tools.testing.target_determination.exploration import estimate_costs
from tools.testing.test_selections import get_duration


if TYPE_CHECKING:
    from collections.abc import Sequence

    from tools.testing.target_determination.heuristics.interface import (
        TestPrioritizations,
    )
    from tools.testing.test_run import TestRun


REPO_ROOT = Path(__file__).resolve().parents[4]
TD_RESULTS_PATH = REPO_ROOT / ADDITIONAL_CI_FILES_FOLDER / "td_results.json"
EXPONENTS = tuple(i / 20 for i in range(21))
MAX_GAP = 0.25
MAX_REASON_CHARS = 300


class Match(NamedTuple):
    exponent: float
    order: list[int]  # indices into the tests, best first
    td_cut_seconds: float
    cut_seconds: float


def import_scorer(path: Path | None = None) -> dict[str, Any] | None:
    """The "scorer" block of the TD job's td_results.json, or None."""
    try:
        with open(TD_RESULTS_PATH if path is None else path) as f:
            block = json.load(f).get("scorer")
    except FileNotFoundError:
        return None
    except Exception:
        print(
            f"Could not read the TD scorer block:\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        return None
    return block if isinstance(block, dict) else None


def match_order(
    names: Sequence[str], p: Sequence[float], costs: Sequence[float], n: int
) -> Match:
    """For tests in TD's order, the exponent `a` whose top `n` by
    p / cost ** a (ties by name) costs closest to TD's top `n` (ties: the
    smaller `a`), and that order."""
    target = sum(costs[:n])
    best: Match | None = None
    best_gap = math.inf
    for a in EXPONENTS:
        keys = [p[i] / costs[i] ** a for i in range(len(names))]
        order = sorted(range(len(names)), key=lambda i: (-keys[i], names[i]))
        cut_seconds = sum(costs[i] for i in order[:n])
        gap = abs(cut_seconds - target)
        if gap < best_gap:
            best, best_gap = Match(a, order, target, cut_seconds), gap
    if best is None:
        raise ValueError("no exponent gives a finite gap")
    return best


def _p_per_test(
    block: dict[str, Any] | None, tests: Sequence[TestRun]
) -> list[float] | str:
    """p for each test, or why there is none."""
    if block is None:
        return "no scorer block"
    if "fallback" in block:
        return f"scorer fell back: {block['fallback']}"
    p = block.get("p")
    if not isinstance(p, dict):
        return "no p"
    values = []
    for test in tests:
        value = p.get(test.test_file)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not (math.isfinite(value) and value > 0)
        ):
            return f"no usable p for {test.test_file}"
        values.append(float(value))
    return values


def _select(
    test_prioritizations: TestPrioritizations,
    scorer_block: dict[str, Any] | None,
    percent: int,
    test_file_times: dict[str, float],
    test_class_times: dict[str, dict[str, float]],
    info: dict[str, Any],
) -> tuple[list[TestRun], list[TestRun]] | str:
    if percent >= 100:
        return "TD is off"
    if scorer_block is not None and isinstance(scorer_block.get("trained_at"), str):
        info["model"] = scorer_block["trained_at"]
    tests = test_prioritizations.get_all_tests()
    p = _p_per_test(scorer_block, tests)
    if isinstance(p, str):
        return p
    costs = estimate_costs(tests, test_file_times, test_class_times)
    if costs is None:
        return "no test times"
    timed = sum(
        get_duration(test, test_file_times, test_class_times) is not None
        for test in tests
    )
    if 2 * timed < len(tests):
        return f"only {timed} of {len(tests)} tests have a time"
    n = len(test_prioritizations.get_top_per_tests(percent)[0])
    match = match_order([str(test) for test in tests], p, costs, n)
    info["td_cut_seconds"] = match.td_cut_seconds
    if abs(match.cut_seconds - match.td_cut_seconds) > MAX_GAP * match.td_cut_seconds:
        return f"no exponent within {MAX_GAP:.0%} of TD's cut time"
    info["exponent"] = match.exponent
    info["cut_seconds"] = match.cut_seconds
    info["order"] = [tests[i] for i in match.order]
    picked = set(match.order[:n])
    # exclude keeps TD's order: exploration draws from it in that order.
    exclude = [test for i, test in enumerate(tests) if i not in picked]
    return [tests[i] for i in match.order[:n]], exclude


def select_tests(
    test_prioritizations: TestPrioritizations,
    scorer_block: dict[str, Any] | None,
    percent: int,
    test_file_times: dict[str, float],
    test_class_times: dict[str, dict[str, float]],
) -> tuple[list[TestRun], list[TestRun], dict[str, Any]]:
    """(include, exclude, info): the model's cut when it is usable (include
    in the model's order, exclude in TD's, info["order"] the model's whole
    order), else TD's get_top_per_tests(percent) with info["fallback"] saying
    why."""
    info: dict[str, Any] = {
        "model": None,
        "exponent": None,
        "td_cut_seconds": None,
        "cut_seconds": None,
        "fallback": None,
        "order": None,
    }
    try:
        selected = _select(
            test_prioritizations,
            scorer_block,
            percent,
            test_file_times,
            test_class_times,
            info,
        )
    except Exception as e:
        print(f"TD scorer cut failed:\n{traceback.format_exc()}", file=sys.stderr)
        selected = f"{type(e).__name__}: {e}"[:MAX_REASON_CHARS]
    if isinstance(selected, str):
        include, exclude = test_prioritizations.get_top_per_tests(percent)
        info.update(exponent=None, cut_seconds=None, fallback=selected, order=None)
        return include, exclude, info
    return selected[0], selected[1], info


def cut_metric(info: dict[str, Any], n_tests: int) -> dict[str, Any]:
    """The td_scorer_cut metric: one row per run_test.py invocation."""
    return {
        "model": info["model"],
        "exponent": info["exponent"],
        "fallback": info["fallback"],
        "td_cut_seconds": info["td_cut_seconds"],
        "cut_seconds": info["cut_seconds"],
        "n_tests": n_tests,
    }


def describe_cut(info: dict[str, Any]) -> str:
    if info["fallback"] is not None:
        return f"TD cut: TD's ranking ({info['fallback']})"
    return (
        f"TD cut: model {info['model']}, exponent {info['exponent']:.2f}, "
        f"{info['cut_seconds']:.0f}s vs TD's {info['td_cut_seconds']:.0f}s"
    )


def failure_position_info(
    info: dict[str, Any], failure: TestRun
) -> dict[str, Any] | None:
    """td_test_failure_stats_v2's "scorer" field: the model, the exponent and
    the failing test's 0-based position in the model's order. None when the
    model's cut did not run."""
    order = info.get("order")
    if info["fallback"] is not None or order is None:
        return None
    position = next((i for i, test in enumerate(order) if test & failure), None)
    return {"model": info["model"], "exponent": info["exponent"], "position": position}
