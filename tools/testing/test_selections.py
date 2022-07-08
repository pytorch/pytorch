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
    num_shards: int, tests: List[str], job_times: Dict[str, float]
) -> List[Tuple[float, List[str]]]:
    filtered_job_times: Dict[str, float] = dict()
    unknown_jobs: List[str] = []
    for test in tests:
        if test in job_times:
            filtered_job_times[test] = job_times[test]
        else:
            unknown_jobs.append(test)

    # The following attempts to implement a partition approximation greedy algorithm
    # See more at https://en.wikipedia.org/wiki/Greedy_number_partitioning
    sorted_jobs = sorted(
        filtered_job_times, key=lambda j: filtered_job_times[j], reverse=True
    )
    sharded_jobs: List[Tuple[float, List[str]]] = [(0.0, []) for _ in range(num_shards)]
    for job in sorted_jobs:
        min_shard_index = sorted(range(num_shards), key=lambda i: sharded_jobs[i][0])[0]
        curr_shard_time, curr_shard_jobs = sharded_jobs[min_shard_index]
        curr_shard_jobs.append(job)
        sharded_jobs[min_shard_index] = (
            curr_shard_time + filtered_job_times[job],
            curr_shard_jobs,
        )

    # Round robin the unknown jobs starting with the smallest shard
    index = sorted(range(num_shards), key=lambda i: sharded_jobs[i][0])[0]
    for job in unknown_jobs:
        sharded_jobs[index][1].append(job)
        index = (index + 1) % num_shards
    return sharded_jobs


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


# Get sharded test allocation based on historic S3 data.
def get_shard_based_on_S3(
    which_shard: int, num_shards: int, tests: List[str], test_times_file: str
) -> List[str]:
    # Short circuit and don't do any work if there's only 1 shard
    if num_shards == 1:
        return tests

    with open(test_times_file) as file:
        jobs_to_times = json.load(file)

    # Got no stats from S3, returning early to save runtime
    if len(jobs_to_times) == 0:
        print(
            "::warning:: Gathered no stats from S3. Proceeding with default sharding plan."
        )
        return tests[which_shard - 1 :: num_shards]

    shards = calculate_shards(num_shards, tests, jobs_to_times)
    _, tests_from_shard = shards[which_shard - 1]
    return tests_from_shard


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
