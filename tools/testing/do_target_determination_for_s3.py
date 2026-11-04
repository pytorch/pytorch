# /// script
# dependencies = ["boto3==1.35.42"]
# ///
import json
import os
import sys
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
    get_test_class_ratings,
    get_test_class_times,
    get_test_file_ratings,
    get_test_times,
)
from tools.stats.upload_metrics import emit_metric
from tools.testing.discover_tests import TESTS
from tools.testing.target_determination.determinator import get_test_prioritizations
from tools.testing.target_determination.heuristics.interface import (
    AggregatedHeuristics,
    TestPrioritizations,
)
from tools.testing.target_determination.heuristics.previously_failed_in_pr import (
    previous_failures_or_none,
)


sys.path.remove(str(REPO_ROOT))

# main() writes TD_RESULTS_FILE to REPO_ROOT; CI uploads it and the test jobs'
# download-td-artifacts moves it to TD_RESULTS.
TD_RESULTS_FILE = "td_results.json"
TD_RESULTS = REPO_ROOT / ADDITIONAL_CI_FILES_FOLDER / TD_RESULTS_FILE
# When the pytest cache download step succeeded, main() stores the previous
# failures with TD's ranking, so that every shard sees the same ones;
# TestPrioritizations.from_json ignores the key.
PREVIOUS_FAILURES_KEY = "previous_failures"


def import_results(
    path: Path = TD_RESULTS,
) -> tuple[TestPrioritizations, set[str] | None]:
    """TD's ranking and the previous failures stored with it, or None for
    those if they are missing (no TD results, an older writer, or a TD job
    whose cache download step failed or that could not read them) or
    malformed. The download step also succeeds when there is no cache, so
    stored failures can be empty on a PR that did fail before."""
    if not path.exists():
        print("No TD results found")
        return TestPrioritizations([], {}), None
    with open(path) as f:
        td_results = json.load(f)
    tp = TestPrioritizations.from_json(td_results)
    if PREVIOUS_FAILURES_KEY not in td_results:
        print(f"Previous failures not recorded in {path}", file=sys.stderr)
        return tp, None
    failed = td_results[PREVIOUS_FAILURES_KEY]
    if not isinstance(failed, list) or not all(isinstance(x, str) for x in failed):
        print(
            f"Ignoring the previous failures in {path}: {failed!r:.200}",
            file=sys.stderr,
        )
        return tp, None
    return tp, set(failed)


def td_results_json(
    test_prioritizations: TestPrioritizations, previous_failures: set[str] | None
) -> dict[str, Any]:
    """The td_results.json that import_results reads: TD's ranking and the
    previous failures, unless they are unknown."""
    td_results = test_prioritizations.to_json()
    if previous_failures is not None:
        td_results[PREVIOUS_FAILURES_KEY] = sorted(previous_failures)
    return td_results


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

    # The download step is continue-on-error, and a failed download leaves no
    # cache, as does a PR without one yet; only the step's outcome tells them
    # apart. A success can still miss failures of jobs that upload no cache.
    if os.getenv("PYTEST_CACHE_DOWNLOAD_OUTCOME") == "success":
        previous_failures = previous_failures_or_none()
    else:
        print("The pytest cache was not downloaded", file=sys.stderr)
        previous_failures = None
    td_results = td_results_json(test_prioritizations, previous_failures)
    with open(REPO_ROOT / TD_RESULTS_FILE, "w") as f:
        f.write(json.dumps(td_results))


if __name__ == "__main__":
    main()
