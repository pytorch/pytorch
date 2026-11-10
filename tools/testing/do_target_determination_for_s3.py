# /// script
# dependencies = ["boto3==1.35.42", "requests==2.32.3"]
# ///
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.stats.import_test_stats import (
    ADDITIONAL_CI_FILES_FOLDER,
    copy_additional_previous_failures,
    copy_pytest_cache,
    get_td_heuristic_historial_edited_files_json,
    get_td_heuristic_profiling_json,
    get_td_scorer,
    get_test_class_ratings,
    get_test_class_times,
    get_test_file_ratings,
    get_test_times,
    TD_SCORER_FILE,
)
from tools.stats.upload_metrics import emit_metric
from tools.testing.discover_tests import TESTS
from tools.testing.target_determination.determinator import get_test_prioritizations
from tools.testing.target_determination.heuristics.interface import (
    AggregatedHeuristics,
    TestPrioritizations,
)
from tools.testing.target_determination.scorer.infer import compute_scorer


sys.path.remove(str(REPO_ROOT))


def import_results() -> TestPrioritizations:
    if not (REPO_ROOT / ".additional_ci_files/td_results.json").exists():
        print("No TD results found")
        return TestPrioritizations([], {})
    with open(REPO_ROOT / ".additional_ci_files/td_results.json") as f:
        td_results = json.load(f)
        tp = TestPrioritizations.from_json(td_results)

    return tp


def main() -> None:
    selected_tests = TESTS

    aggregated_heuristics: AggregatedHeuristics = AggregatedHeuristics(selected_tests)

    test_times: dict[str, Any] = get_test_times()
    get_test_class_times()
    get_test_file_ratings()
    get_test_class_ratings()
    get_td_heuristic_historial_edited_files_json()
    get_td_heuristic_profiling_json()
    copy_pytest_cache()
    copy_additional_previous_failures()
    get_td_scorer()

    aggregated_heuristics = get_test_prioritizations(selected_tests)

    test_prioritizations = aggregated_heuristics.get_aggregated_priorities()

    print("Aggregated Heuristics")
    print(test_prioritizations.get_info_str(verbose=False))

    scorer = compute_scorer(
        selected_tests,
        aggregated_heuristics,
        test_times,
        REPO_ROOT / ADDITIONAL_CI_FILES_FOLDER / TD_SCORER_FILE,
    )

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
            emit_metric("td_scorer", scorer.metric)
        except Exception:
            print(f"Failed to emit td_scorer:\n{traceback.format_exc()}")

    with open(REPO_ROOT / "td_results.json", "w") as f:
        f.write(json.dumps({**test_prioritizations.to_json(), "scorer": scorer.block}))


if __name__ == "__main__":
    main()
