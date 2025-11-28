import argparse
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
from tools.testing.target_determination import (
    AggregatedHeuristics,
    get_job_info_from_workflow_file,
    get_test_prioritizations,
    TestsToRun,
)


sys.path.remove(str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run target determination with job info"
    )
    parser.add_argument(
        "--workflow-ref",
        type=str,
        default="",
        help="Path to the GitHub workflow file to parse, this should correspond to github.workflow_ref in the github context",
    )
    args = parser.parse_args()
    return args


def import_results(job_name: str) -> TestsToRun:
    if not (REPO_ROOT / ".additional_ci_files/td_results.json").exists():
        print("No TD results found")
        return TestsToRun([], [])
    with open(REPO_ROOT / ".additional_ci_files/td_results.json") as f:
        td_results = json.load(f)
        res = {k: TestsToRun.from_json(v) for k, v in td_results.items()}
        if job_name not in res:
            print(f"Job name {job_name} not found in TD results, using default")
        return res.get(job_name, res.get("default", TestsToRun([], [])))


def main() -> None:
    args = parse_args()
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

    job_info = get_job_info_from_workflow_file(args.workflow_ref)
    print(f"Job info: {json.dumps(job_info, indent=2)}")

    aggregated_heuristics = get_test_prioritizations(selected_tests)

    test_prioritizations = aggregated_heuristics.get_aggregated_priorities()

    recommended_cutoffs_per_job = test_prioritizations.get_recommended_cutoffs(job_info)

    json_serialized_cutoffs = {
        k: v.to_json() for k, v in recommended_cutoffs_per_job.items()
    }

    print("Recommended Cutoffs Per Job:")
    print(json.dumps(json_serialized_cutoffs, indent=2))

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

    with open(REPO_ROOT / "td_results.json", "w") as f:
        f.write(json.dumps(json_serialized_cutoffs, indent=2))


if __name__ == "__main__":
    main()
