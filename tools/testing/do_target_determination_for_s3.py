# /// script
# dependencies = ["boto3==1.35.42"]
# ///
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.stats.import_test_stats import (
    copy_additional_previous_failures,
    copy_pytest_cache,
    get_td_heuristic_historial_edited_files_json,
    get_td_heuristic_profiling_json,
    get_td_scorer,
    get_test_class_ratings,
    get_test_class_times,
    get_test_file_ratings,
    get_test_times,
    TD_SCORER_BUDGET_FILE,
    TD_SCORER_FILE,
)
from tools.stats.upload_metrics import emit_metric
from tools.testing.discover_tests import TESTS
from tools.testing.target_determination.determinator import get_test_prioritizations
from tools.testing.target_determination.heuristics import HEURISTICS
from tools.testing.target_determination.heuristics.interface import (
    AggregatedHeuristics,
    TestPrioritizations,
)
from tools.testing.target_determination.heuristics.utils import query_changed_files
from tools.testing.target_determination.scorer import (
    build_scorer_block,
    TD_RESULTS_PATH,
)
from tools.testing.target_determination.scorer_model import MAX_MODEL_BYTES


sys.path.remove(str(REPO_ROOT))


def choose_scorer_block(
    aggregated: AggregatedHeuristics, changed_files: list[str] | None
) -> dict[str, Any] | None:
    """The scorer block from TD_SCORER_BUDGET_FILE when it and TD_SCORER_FILE both
    give a usable model of the same version (one refit), else from TD_SCORER_FILE
    with any budget block dropped.

    TD_SCORER_FILE decides alone whether the scorer runs: disabled, missing or
    stale, no budget applies. Never raises: on any error the TD_SCORER_FILE
    block stands, or None.
    """
    plain = None
    try:
        plain = build_scorer_block(
            aggregated,
            HEURISTICS,
            lambda: get_td_scorer(MAX_MODEL_BYTES, TD_SCORER_FILE),
            changed_files,
        )
        if plain is None or plain.get("model") is None:
            return plain
        if "budget" in plain["model"]:
            model = {k: v for k, v in plain["model"].items() if k != "budget"}
            plain = {**plain, "model": model}
        budget = build_scorer_block(
            aggregated,
            HEURISTICS,
            lambda: get_td_scorer(MAX_MODEL_BYTES, TD_SCORER_BUDGET_FILE),
            changed_files,
        )
        model = None if budget is None else budget.get("model")
        if model is not None and model["version"] == plain["model"]["version"]:
            print(f"TD scorer: model from {TD_SCORER_BUDGET_FILE}")
            return budget
        print(f"TD scorer: model from {TD_SCORER_FILE}")
    except Exception as e:
        print(f"TD scorer: {type(e).__name__} choosing the model file")
    return plain


def import_results() -> TestPrioritizations:
    if not TD_RESULTS_PATH.exists():
        print("No TD results found")
        return TestPrioritizations([], {})
    with open(TD_RESULTS_PATH) as f:
        td_results = json.load(f)
        tp = TestPrioritizations.from_json(td_results)

    return tp


def main() -> None:
    selected_tests = TESTS

    aggregated_heuristics: AggregatedHeuristics = AggregatedHeuristics(selected_tests)

    get_test_times()
    get_test_class_times()
    get_test_file_ratings()
    get_test_class_ratings()
    get_td_heuristic_historial_edited_files_json()
    get_td_heuristic_profiling_json()
    copy_pytest_cache()
    copy_additional_previous_failures()

    aggregated_heuristics = get_test_prioritizations(selected_tests)

    test_prioritizations = aggregated_heuristics.get_aggregated_priorities()

    print("Aggregated Heuristics")
    print(test_prioritizations.get_info_str(verbose=False))

    if os.getenv("CI") == "true":
        print("Emitting metrics")
        # Split into 3 due to size constraints
        emit_metric(
            "td_results_final_test_prioritizations",
            {"test_prioritizations": test_prioritizations.to_json()},
        )
        emit_metric(
            "td_results_aggregated_heuristics",
            {"aggregated_heuristics": aggregated_heuristics.to_json()},
        )

    try:
        changed_files: list[str] | None = query_changed_files()
    except Exception as e:
        print(f"TD scorer: changed files unavailable: {e!r}")
        changed_files = None

    td_results = test_prioritizations.to_json()
    scorer_block = choose_scorer_block(aggregated_heuristics, changed_files)
    if scorer_block is not None:
        td_results["scorer"] = scorer_block

    with open(REPO_ROOT / "td_results.json", "w") as f:
        f.write(json.dumps(td_results))


if __name__ == "__main__":
    main()
