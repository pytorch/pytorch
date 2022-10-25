import os
import subprocess

from typing import Callable, Dict, List, Optional, Tuple

from tools.stats.import_test_stats import get_disabled_tests, get_slow_tests

NUM_PROCS = 2


class ShardJob:
    def __init__(self, test_times: Dict[str, float]):
        self.test_times = test_times
        self.serial: List[str] = []
        self.parallel: List[str] = []

    def get_total_time(self) -> float:
        procs = [0.0 for _ in range(NUM_PROCS)]
        for test in self.parallel:
            test_time = self.test_times.get(test, 0)
            min_index = procs.index(min(procs))
            procs[min_index] += test_time
        time = max(procs) + sum(self.test_times.get(test, 0) for test in self.serial)
        return time

    def convert_to_tuple(self) -> Tuple[float, List[str]]:
        return (self.get_total_time(), self.serial + self.parallel)


def calculate_shards(
    num_shards: int,
    tests: List[str],
    test_file_times: Dict[str, float],
    must_serial: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[float, List[str]]]:
    must_serial = must_serial or (lambda x: True)

    known_tests = [x for x in tests if x in test_file_times]
    unknown_tests: List[str] = [x for x in tests if x not in known_tests]

    sorted_tests = sorted(known_tests, key=lambda j: test_file_times[j], reverse=True)

    sharded_jobs: List[ShardJob] = [
        ShardJob(test_file_times) for _ in range(num_shards)
    ]
    for test in sorted_tests:
        if must_serial(test):
            min_sharded_job = min(sharded_jobs, key=lambda j: j.get_total_time())
            min_sharded_job.serial.append(test)
        else:
            min_sharded_job = min(sharded_jobs, key=lambda j: j.get_total_time())
            min_sharded_job.parallel.append(test)

    # Round robin the unknown jobs starting with the smallest shard
    index = min(range(num_shards), key=lambda i: sharded_jobs[i].get_total_time())
    for test in unknown_tests:
        sharded_jobs[index].serial.append(test)
        index = (index + 1) % num_shards
    return [job.convert_to_tuple() for job in sharded_jobs]


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
