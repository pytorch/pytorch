"""run_test.py's TD cut with the learned scorer: the job's tests ranked by
p / cost (ties by name), taken while the time taken so far is below
budget_share of the job's total test time, so the test that crosses the
budget is kept. The TD job sets budget_share from the change's risk. Any
problem gives TD's own cut."""

from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, TYPE_CHECKING

from tools.stats.import_test_stats import ADDITIONAL_CI_FILES_FOLDER
from tools.testing.target_determination.exploration import estimate_costs
from tools.testing.target_determination.scorer.model import SHARE_CEILING, SHARE_FLOOR
from tools.testing.test_selections import get_duration


if TYPE_CHECKING:
    from collections.abc import Sequence

    from tools.testing.target_determination.heuristics.interface import (
        TestPrioritizations,
    )
    from tools.testing.test_run import TestRun


REPO_ROOT = Path(__file__).resolve().parents[4]
TD_RESULTS_PATH = REPO_ROOT / ADDITIONAL_CI_FILES_FOLDER / "td_results.json"
MAX_REASON_CHARS = 300


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


def budget_order(
    names: Sequence[str], p: Sequence[float], costs: Sequence[float]
) -> list[int]:
    """Indices of the tests by p / cost, highest first, ties by name."""
    return sorted(range(len(names)), key=lambda i: (-p[i] / costs[i], names[i]))


def budget_cut(order: Sequence[int], costs: Sequence[float], budget: float) -> int:
    """How many tests of `order` are taken: each while the cost taken so far
    is below `budget`."""
    taken = 0.0
    for n, i in enumerate(order):
        if taken >= budget:
            return n
        taken += costs[i]
    return len(order)


def _budget_share(block: dict[str, Any]) -> float | str:
    """The block's budget_share, or why it cannot be used."""
    share = block.get("budget_share")
    if share is None:
        return "no budget share"
    if isinstance(share, bool) or not isinstance(share, (int, float)):
        return "budget share is not a number"
    if not SHARE_FLOOR <= share <= SHARE_CEILING:
        return f"budget share {share} outside [{SHARE_FLOOR}, {SHARE_CEILING}]"
    return float(share)


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
    share = _budget_share(scorer_block or {})
    if isinstance(share, str):
        return share
    info["budget_share"] = share
    costs = estimate_costs(tests, test_file_times, test_class_times)
    if costs is None:
        return "no test times"
    timed = sum(
        get_duration(test, test_file_times, test_class_times) is not None
        for test in tests
    )
    if 2 * timed < len(tests):
        return f"only {timed} of {len(tests)} tests have a time"
    n_td = len(test_prioritizations.get_top_per_tests(percent)[0])
    info["td_cut_seconds"] = math.fsum(costs[:n_td])
    total = math.fsum(costs)
    budget = share * total
    order = budget_order([str(test) for test in tests], p, costs)
    n = budget_cut(order, costs, budget)
    info["total_seconds"] = total
    info["budget_seconds"] = budget
    info["selected_seconds"] = math.fsum(costs[i] for i in order[:n])
    info["order"] = [tests[i] for i in order]
    picked = set(order[:n])
    # exclude keeps TD's order: exploration draws from it in that order.
    exclude = [test for i, test in enumerate(tests) if i not in picked]
    return [tests[i] for i in order[:n]], exclude


def select_tests(
    test_prioritizations: TestPrioritizations,
    scorer_block: dict[str, Any] | None,
    percent: int,
    test_file_times: dict[str, float],
    test_class_times: dict[str, dict[str, float]],
) -> tuple[list[TestRun], list[TestRun], dict[str, Any]]:
    """(include, exclude, info): the model's budget cut when it is usable
    (include in the model's order, exclude in TD's, info["order"] the model's
    whole order), else TD's get_top_per_tests(percent) with info["fallback"]
    saying why and info["budget_share"] None."""
    info: dict[str, Any] = {
        "model": None,
        "budget_share": None,
        "fallback": None,
        "total_seconds": None,
        "budget_seconds": None,
        "selected_seconds": None,
        "td_cut_seconds": None,
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
        info.update(
            budget_share=None,
            fallback=selected,
            total_seconds=None,
            budget_seconds=None,
            selected_seconds=None,
            order=None,
        )
    else:
        include, exclude = selected
    info["n_tests"] = len(include) + len(exclude)
    info["n_selected"] = len(include)
    return include, exclude, info


def cut_metric(info: dict[str, Any]) -> dict[str, Any]:
    """The td_scorer_cut metric: one row per run_test.py invocation.
    td_cut_seconds is what TD's own cut would cost on the job."""
    return {
        key: info[key]
        for key in (
            "model",
            "budget_share",
            "fallback",
            "total_seconds",
            "budget_seconds",
            "selected_seconds",
            "td_cut_seconds",
            "n_tests",
            "n_selected",
        )
    }


def describe_cut(info: dict[str, Any]) -> str:
    if info["fallback"] is not None:
        return f"TD cut: TD's ranking ({info['fallback']})"
    return (
        f"TD cut: model {info['model']}, budget share {info['budget_share']:.3f}, "
        f"{info['n_selected']} of {info['n_tests']} tests, "
        f"{info['selected_seconds']:.0f}s for a {info['budget_seconds']:.0f}s budget "
        f"of {info['total_seconds']:.0f}s (TD's cut {info['td_cut_seconds']:.0f}s)"
    )


def failure_position_info(
    info: dict[str, Any], failure: TestRun
) -> dict[str, Any] | None:
    """td_test_failure_stats_v2's "scorer" field: the model, the budget share
    and the failing test's 0-based position in the model's order. None when
    the model's cut did not run."""
    order = info.get("order")
    if info["fallback"] is not None or order is None:
        return None
    position = next((i for i, test in enumerate(order) if test & failure), None)
    return {
        "model": info["model"],
        "budget_share": info["budget_share"],
        "position": position,
    }
