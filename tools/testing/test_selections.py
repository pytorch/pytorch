import os
import subprocess

from typing import Dict, List, Tuple

from tools.stats.import_test_stats import get_disabled_tests, get_slow_tests

NUM_PROCS = 3


def calculate_shards(
    num_shards: int, tests: List[str], job_times: Dict[str, float]
) -> List[Tuple[float, List[str]]]:
    filtered_job_times: Dict[str, float] = {}
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


def get_test_case_configs(dirpath: str) -> None:
    get_slow_tests(dirpath=dirpath)
    get_disabled_tests(dirpath=dirpath)
