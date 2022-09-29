import os
import subprocess

from typing import Dict, List, Tuple

from tools.stats.import_test_stats import get_disabled_tests, get_slow_tests

LARGE_TEST = 45 * 60

TestJob = Tuple[str, int, int, float]


class ShardJob:
    def __init__(self, test_times: Dict[str, float]):
        self.test_times = test_times
        self.test_jobs: List[TestJob] = []

    def get_total_time(self) -> float:
        return sum(x[3] for x in self.test_jobs)

    def convert_to_tuple(self) -> Tuple[float, List[TestJob]]:
        return (self.get_total_time(), self.test_jobs)


def calculate_shards(
    num_shards: int,
    tests: List[str],
    test_times: Dict[str, float],
) -> List[Tuple[float, List[TestJob]]]:
    known_tests = [x for x in tests if x in test_times]
    unknown_tests = [x for x in tests if x not in known_tests]

    test_jobs: List[TestJob] = []
    for test in known_tests:
        test_time = test_times[test]
        if test_time > LARGE_TEST:
            test_shards = int(test_time // LARGE_TEST + 1)
            for i in range(test_shards):
                test_jobs.append((test, i, test_shards, test_time / test_shards))
        else:
            test_jobs.append((test, 0, 1, test_time))

    test_jobs = sorted(test_jobs, key=lambda x: x[3], reverse=True)

    sharded_jobs: List[ShardJob] = [ShardJob(test_times) for _ in range(num_shards)]

    for test_job in test_jobs:
        min_sharded_job = sorted(sharded_jobs, key=lambda j: j.get_total_time())[0]
        min_sharded_job.test_jobs.append(test_job)

    # Round robin the unknown jobs starting with the smallest shard
    index = sorted(range(num_shards), key=lambda i: sharded_jobs[i].get_total_time())[0]
    for test in unknown_tests:
        sharded_jobs[index].test_jobs.append((test, 0, 1, 0.0))
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


def get_reordered_tests(tests: List[TestJob]) -> List[TestJob]:
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
        if test[0] in prioritized_tests:
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
