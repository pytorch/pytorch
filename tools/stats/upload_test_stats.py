import argparse
import sys
from typing import Any, Dict, List

from tools.stats.test_dashboard import upload_additional_info
from tools.stats.upload_stats_lib import get_tests, upload_workflow_stats_to_s3


def summarize_test_cases(test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group test cases by classname, file, and job_id. We perform the aggregation
    manually instead of using the `test-suite` XML tag because xmlrunner does
    not produce reliable output for it.
    """

    def get_key(test_case: Dict[str, Any]) -> Any:
        return (
            test_case.get("file"),
            test_case.get("classname"),
            test_case["job_id"],
            test_case["workflow_id"],
            test_case["workflow_run_attempt"],
            # [see: invoking file]
            test_case["invoking_file"],
        )

    def init_value(test_case: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "file": test_case.get("file"),
            "classname": test_case.get("classname"),
            "job_id": test_case["job_id"],
            "workflow_id": test_case["workflow_id"],
            "workflow_run_attempt": test_case["workflow_run_attempt"],
            # [see: invoking file]
            "invoking_file": test_case["invoking_file"],
            "tests": 0,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "successes": 0,
            "time": 0.0,
        }

    ret = {}
    for test_case in test_cases:
        key = get_key(test_case)
        if key not in ret:
            ret[key] = init_value(test_case)

        ret[key]["tests"] += 1

        if "failure" in test_case:
            ret[key]["failures"] += 1
        elif "error" in test_case:
            ret[key]["errors"] += 1
        elif "skipped" in test_case:
            ret[key]["skipped"] += 1
        else:
            ret[key]["successes"] += 1

        ret[key]["time"] += test_case["time"]
    return list(ret.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload test stats to Rockset")
    parser.add_argument(
        "--workflow-run-id",
        required=True,
        help="id of the workflow to get artifacts from",
    )
    parser.add_argument(
        "--workflow-run-attempt",
        type=int,
        required=True,
        help="which retry of the workflow this is",
    )
    parser.add_argument(
        "--head-branch",
        required=True,
        help="Head branch of the workflow",
    )
    parser.add_argument(
        "--head-repository",
        required=True,
        help="Head repository of the workflow",
    )
    parser.add_argument(
        "--head-sha",
        required=False,
        help="Head sha of the workflow",
    )
    args = parser.parse_args()

    print(f"Workflow id is: {args.workflow_run_id}")

    test_cases = get_tests(args.workflow_run_id, args.workflow_run_attempt)

    # Flush stdout so that any errors in Rockset upload show up last in the logs.
    sys.stdout.flush()

    # For PRs, only upload a summary of test_runs. This helps lower the
    # volume of writes we do to Rockset.
    test_case_summary = summarize_test_cases(test_cases)

    upload_workflow_stats_to_s3(
        args.workflow_run_id,
        args.workflow_run_attempt,
        "test_run_summary",
        test_case_summary,
    )

    # Separate out the failed test cases.
    # Uploading everything is too data intensive most of the time,
    # but these will be just a tiny fraction.
    failed_tests_cases = []
    for test_case in test_cases:
        if "rerun" in test_case or "failure" in test_case or "error" in test_case:
            failed_tests_cases.append(test_case)

    upload_workflow_stats_to_s3(
        args.workflow_run_id,
        args.workflow_run_attempt,
        "failed_test_runs",
        failed_tests_cases,
    )

    if args.head_branch == "main" and args.head_repository == "pytorch/pytorch":
        # For jobs on main branch, upload everything.
        upload_workflow_stats_to_s3(
            args.workflow_run_id, args.workflow_run_attempt, "test_run", test_cases
        )

    upload_additional_info(
        args.workflow_run_id, args.workflow_run_attempt, args.head_sha, test_cases
    )
