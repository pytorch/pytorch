import json
import os
import subprocess

from tools.stats.s3_stat_parser import (
    get_previous_reports_for_branch,
    Report,
    Version2Report,
    HAVE_BOTO3,
)
from tools.stats.import_test_stats import get_disabled_tests, get_slow_tests

from typing import Any, Dict, List, Optional, Tuple, cast
from typing_extensions import TypedDict


class JobTimeJSON(TypedDict):
    commit: str
    JOB_BASE_NAME: str
    job_times: Dict[str, float]


def _get_stripped_CI_job() -> str:
    return os.environ.get("BUILD_ENVIRONMENT", "")


def _get_job_times_json(job_times: Dict[str, float]) -> JobTimeJSON:
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], encoding="ascii"
        ).strip(),
        "JOB_BASE_NAME": _get_stripped_CI_job(),
        "job_times": job_times,
    }


def _calculate_job_times(reports: List["Report"]) -> Dict[str, float]:
    """Compute test runtime by filename: ("test_file_name" -> (current_avg, # values))"""
    jobs_to_times: Dict[str, Tuple[float, int]] = dict()
    for report in reports:
        v_report = cast(Version2Report, report)
        assert (
            "format_version" in v_report.keys() and v_report.get("format_version") == 2
        ), "S3 format currently handled is version 2 only"
        files: Dict[str, Any] = v_report["files"]
        for name, test_file in files.items():
            if name not in jobs_to_times:
                jobs_to_times[name] = (test_file["total_seconds"], 1)
            else:
                curr_avg, curr_count = jobs_to_times[name]
                new_count = curr_count + 1
                new_avg = (
                    curr_avg * curr_count + test_file["total_seconds"]
                ) / new_count
                jobs_to_times[name] = (new_avg, new_count)

    return {job: time for job, (time, _) in jobs_to_times.items()}


def calculate_shards(
    num_shards: int, test_files: List[str], test_file_times: Dict[str, float]
) -> List[Tuple[float, List[str]]]:
    filtered_test_file_times: Dict[str, float] = dict()
    unknown_test_files: List[str] = []
    for test_file in test_files:
        if test_file in test_file_times:
            filtered_test_file_times[test_file] = test_file_times[test_file]
        else:
            unknown_test_files.append(test_file)

    # The following attempts to implement a partition approximation greedy algorithm
    # See more at https://en.wikipedia.org/wiki/Greedy_number_partitioning
    sorted_test_files = sorted(
        filtered_test_file_times,
        key=lambda j: filtered_test_file_times[j],
        reverse=True,
    )
    sharded_test_files: List[Tuple[float, List[str]]] = [
        (0.0, []) for _ in range(num_shards)
    ]
    for test_file in sorted_test_files:
        min_shard_index = sorted(
            range(num_shards), key=lambda i: sharded_test_files[i][0]
        )[0]
        curr_shard_time, curr_shard_test_files = sharded_test_files[min_shard_index]
        curr_shard_test_files.append(test_file)
        sharded_test_files[min_shard_index] = (
            curr_shard_time + filtered_test_file_times[test_file],
            curr_shard_test_files,
        )

    # Round robin the unknown test files starting with the smallest shard
    index = sorted(range(num_shards), key=lambda i: sharded_test_files[i][0])[0]
    for test_file in unknown_test_files:
        sharded_test_files[index][1].append(test_file)
        index = (index + 1) % num_shards
    return sharded_test_files


def _pull_job_times_from_S3() -> Dict[str, float]:
    if HAVE_BOTO3:
        ci_job_prefix = _get_stripped_CI_job()
        s3_reports: List["Report"] = get_previous_reports_for_branch(
            "origin/viable/strict", ci_job_prefix
        )
    else:
        print(
            "Uh oh, boto3 is not found. Either it is not installed or we failed to import s3_stat_parser."
        )
        print(
            "If not installed, please install boto3 for automatic sharding and test categorization."
        )
        s3_reports = []

    if len(s3_reports) == 0:
        print("::warning:: Gathered no reports from S3. Please proceed without them.")
        return dict()

    return _calculate_job_times(s3_reports)


def _query_changed_test_files() -> List[str]:
    default_branch = f"origin/{os.environ.get('GIT_DEFAULT_BRANCH', 'master')}"
    cmd = ["git", "diff", "--name-only", default_branch, "HEAD"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if proc.returncode != 0:
        raise RuntimeError("Unable to get changed files")

    lines = proc.stdout.decode().strip().split("\n")
    lines = [line.strip() for line in lines]
    return lines


def get_reordered_tests(tests: List[str]) -> List[str]:
    """Get the reordered test filename list based on github PR history or git changed file."""
    prioritized_tests: List[str] = []
    if len(prioritized_tests) == 0:
        try:
            changed_files = _query_changed_test_files()
        except Exception:
            # If unable to get changed files from git, quit without doing any sorting
            return tests

        prefix = f"test{os.path.sep}"
        prioritized_tests = [
            f for f in changed_files if f.startswith(prefix) and f.endswith(".py")
        ]
        prioritized_tests = [f[len(prefix) :] for f in prioritized_tests]
        prioritized_tests = [f[: -len(".py")] for f in prioritized_tests]
        print("Prioritized test from test file changes.")

    bring_to_front = []
    the_rest = []

    for test in tests:
        if test in prioritized_tests:
            bring_to_front.append(test)
        else:
            the_rest.append(test)
    if len(tests) == len(bring_to_front) + len(the_rest):
        print(
            f"reordering tests for PR:\n"
            f"prioritized: {bring_to_front}\nthe rest: {the_rest}\n"
        )
        return bring_to_front + the_rest
    else:
        print(
            f"Something went wrong in CI reordering, expecting total of {len(tests)}:\n"
            f"but found prioritized: {len(bring_to_front)}\nthe rest: {len(the_rest)}\n"
        )
        return tests


# TODO Refactor this and unify with tools.stats.export_slow_tests
def export_S3_test_times(test_times_filename: Optional[str] = None) -> Dict[str, float]:
    test_times: Dict[str, float] = _pull_job_times_from_S3()
    if test_times_filename is not None:
        print(f"Exporting S3 test stats to {test_times_filename}.")
        if os.path.exists(test_times_filename):
            print(f"Overwriting existent file: {test_times_filename}")
        with open(test_times_filename, "w+") as file:
            job_times_json = _get_job_times_json(test_times)
            json.dump(job_times_json, file, indent="    ", separators=(",", ": "))
            file.write("\n")
    return test_times


def get_test_case_configs(dirpath: str) -> None:
    get_slow_tests(dirpath=dirpath)
    get_disabled_tests(dirpath=dirpath)
