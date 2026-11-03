# /// script
# dependencies = ["boto3==1.35.42"]
# ///
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.stats.import_test_stats import (
    copy_additional_previous_failures,
    copy_pytest_cache,
    get_td_heuristic_historial_edited_files_json,
    get_td_heuristic_profiling_json,
    get_test_class_ratings,
    get_test_class_times,
    get_test_file_ratings,
    get_test_times,
)
from tools.stats.upload_metrics import emit_metric
from tools.testing.discover_tests import TESTS
from tools.testing.llm_td.selection import previous_failures, PREVIOUS_FAILURES_KEY
from tools.testing.target_determination.determinator import get_test_prioritizations
from tools.testing.target_determination.heuristics.interface import (
    AggregatedHeuristics,
    TestPrioritizations,
)


sys.path.remove(str(REPO_ROOT))

TD_RESULTS = REPO_ROOT / ".additional_ci_files/td_results.json"


def import_results() -> TestPrioritizations:
    if not TD_RESULTS.exists():
        print("No TD results found")
        return TestPrioritizations([], {})
    with open(TD_RESULTS) as f:
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

    # Stored with the ranking so that every shard sees the same previous
    # failures; TestPrioritizations.from_json ignores the extra key.
    td_results = test_prioritizations.to_json()
    td_results[PREVIOUS_FAILURES_KEY] = sorted(previous_failures())
    with open(REPO_ROOT / "td_results.json", "w") as f:
        f.write(json.dumps(td_results))


if __name__ == "__main__":
    main()
