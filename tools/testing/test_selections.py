import json
import math
import os
import pathlib
import subprocess

from typing import Callable, Dict, List, NamedTuple, Optional, Set, Tuple

from tools.shared.logging_utils import pluralize, to_time_str

from tools.stats.import_test_stats import get_disabled_tests, get_slow_tests

IS_MEM_LEAK_CHECK = os.getenv("PYTORCH_TEST_CUDA_MEM_LEAK_CHECK", "0") == "1"

NUM_PROCS = 1 if IS_MEM_LEAK_CHECK else 2
THRESHOLD = 60 * 10  # 10 minutes

# See Note [ROCm parallel CI testing]
# Special logic for ROCm GHA runners to query number of GPUs available.
# torch.version.hip was not available to check if this was a ROCm self-hosted runner.
# Must check for ROCm runner in another way. We look for /opt/rocm directory.
if os.path.exists("/opt/rocm") and not IS_MEM_LEAK_CHECK:
    try:
        # This is the same logic used in GHA health check, see .github/templates/common.yml.j2
        lines = (
            subprocess.check_output(["rocminfo"], encoding="ascii").strip().split("\n")
        )
        count = 0
        for line in lines:
            if " gfx" in line:
                count += 1
        assert count > 0  # there must be at least 1 GPU
        # Limiting to 8 GPUs(PROCS)
        NUM_PROCS = 8 if count > 8 else count
    except subprocess.CalledProcessError as e:
        # The safe default for ROCm GHA runners is to run tests serially.
        NUM_PROCS = 1


class ShardedTest(NamedTuple):
    name: str
    shard: int
    num_shards: int
    time: Optional[float]

    def __str__(self) -> str:
        return f"{self.name} {self.shard}/{self.num_shards}"

    def get_time(self) -> float:
        return self.time or 0


class ShardJob:
    def __init__(self) -> None:
        self.serial: List[ShardedTest] = []
        self.parallel: List[ShardedTest] = []

    def get_total_time(self) -> float:
        procs = [0.0 for _ in range(NUM_PROCS)]
        for test in self.parallel:
            min_index = procs.index(min(procs))
            procs[min_index] += test.get_time()
        time = max(procs) + sum(test.get_time() for test in self.serial)
        return time

    def convert_to_tuple(self) -> Tuple[float, List[ShardedTest]]:
        return (self.get_total_time(), self.serial + self.parallel)


def get_with_pytest_shard(
    tests: List[str], test_file_times: Dict[str, float]
) -> List[ShardedTest]:
    sharded_tests: List[ShardedTest] = []
    for test in tests:
        duration = test_file_times[test]
        if duration > THRESHOLD:
            num_shards = math.ceil(duration / THRESHOLD)
            for i in range(num_shards):
                sharded_tests.append(
                    ShardedTest(test, i + 1, num_shards, duration / num_shards)
                )
        else:
            sharded_tests.append(ShardedTest(test, 1, 1, duration))
    return sharded_tests


def calculate_shards(
    num_shards: int,
    tests: List[str],
    test_file_times: Dict[str, float],
    must_serial: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[float, List[ShardedTest]]]:
    must_serial = must_serial or (lambda x: True)

    known_tests = [x for x in tests if x in test_file_times]
    unknown_tests: List[str] = [x for x in tests if x not in known_tests]

    sorted_tests = sorted(
        get_with_pytest_shard(known_tests, test_file_times),
        key=lambda j: j.get_time(),
        reverse=True,
    )

    sharded_jobs: List[ShardJob] = [ShardJob() for _ in range(num_shards)]
    for test in sorted_tests:
        if must_serial(test.name):
            min_sharded_job = min(sharded_jobs, key=lambda j: j.get_total_time())
            min_sharded_job.serial.append(test)
        else:
            min_sharded_job = min(sharded_jobs, key=lambda j: j.get_total_time())
            min_sharded_job.parallel.append(test)

    # Round robin the unknown jobs starting with the smallest shard
    index = min(range(num_shards), key=lambda i: sharded_jobs[i].get_total_time())
    for unknown_test in unknown_tests:
        sharded_jobs[index].serial.append(ShardedTest(unknown_test, 1, 1, None))
        index = (index + 1) % num_shards
    return [job.convert_to_tuple() for job in sharded_jobs]


def _query_changed_test_files() -> List[str]:
    default_branch = f"origin/{os.environ.get('GIT_DEFAULT_BRANCH', 'main')}"
    merge_base = (
        subprocess.check_output(["git", "merge-base", default_branch, "HEAD"])
        .decode()
        .strip()
    )
    proc = subprocess.run(
        ["git", "diff", "--name-only", merge_base, "HEAD"], capture_output=True
    )

    if proc.returncode != 0:
        raise RuntimeError("Unable to get changed files")

    lines = proc.stdout.decode().strip().split("\n")
    lines = [line.strip() for line in lines]
    return lines


def _get_previously_failing_tests() -> Set[str]:
    PYTORCH_FAILED_TESTS_CACHE_FILE_PATH = pathlib.Path(
        ".pytorch_cache/v/cache/lastfailed"
    )

    if not PYTORCH_FAILED_TESTS_CACHE_FILE_PATH.exists():
        return []

    with open(PYTORCH_FAILED_TESTS_CACHE_FILE_PATH, "r") as f:
        last_failed_tests = json.load(f)

    prioritized_tests = _parse_prev_failing_test_files(last_failed_tests)

    print(
        f"Prioritized {pluralize(len(prioritized_tests), 'test')} from test file changes."
    )

    return prioritized_tests


def _parse_prev_failing_test_files(last_failed_tests: Dict[str, bool]) -> Set[str]:
    prioritized_tests = set()

    # The keys are formatted as "test_file.py::test_class::test_method[params]"
    # We just need the test_file part, without the extension
    for test in last_failed_tests:
        parts = test.split(".py::")  # For now, only support reordering python tests.
        if len(parts) > 1:
            test_file = parts[0]
            print(f"Adding part: {test_file}. Parts had len {len(parts)}")
            prioritized_tests.add(test_file)

    return prioritized_tests


def _get_test_prioritized_due_to_test_file_changes() -> Set[str]:
    try:
        changed_files = _query_changed_test_files()
    except Exception:
        # If unable to get changed files from git, quit without doing any sorting
        return []

    prefix = f"test{os.path.sep}"
    # TODO: Make prioritization work with C++ test as well
    prioritized_tests = {
        f for f in changed_files if f.startswith(prefix) and f.endswith(".py")
    }
    prioritized_tests = [f[len(prefix) :] for f in prioritized_tests]
    prioritized_tests = [f[: -len(".py")] for f in prioritized_tests]

    print(
        f"Prioritized {pluralize(len(prioritized_tests), 'test')} from test file changes."
    )

    return prioritized_tests


def get_reordered_tests(
    tests: List[ShardedTest],
) -> Tuple[List[ShardedTest], List[ShardedTest]]:
    """
    Get the reordered test filename list based on github PR history or git changed file.
    We prioritize running test files that were changed.
    """
    prioritized_tests: Set[str] = set()
    prioritized_tests |= _get_previously_failing_tests()
    prioritized_tests |= _get_test_prioritized_due_to_test_file_changes()

    bring_to_front = []
    the_rest = []

    test_time_for_regular_tests_so_far = 0.0
    # how much sooner did we run prioritized tests compared to a naive ordering
    time_savings = 0.0

    for test in tests:
        if test.name in prioritized_tests:
            bring_to_front.append(test)
            # Calculate approx time saved by reordering
            time_savings = test_time_for_regular_tests_so_far
        else:
            the_rest.append(test)
            test_time_for_regular_tests_so_far += test.get_time()

    if len(tests) != len(bring_to_front) + len(the_rest):
        print(
            f"Something went wrong in CI reordering, expecting total of {len(tests)}:\n"
            f"but found prioritized: {len(bring_to_front)}\nthe rest: {len(the_rest)}\n"
        )
        return ([], tests)

    test_cnt_str = pluralize(len(tests), "test")
    print(f"Reordering tests: Prioritizing {len(bring_to_front)} of {test_cnt_str}")
    print(f"Prioritized tests will run up to {to_time_str(time_savings)} faster")
    print(f"Prioritized: {bring_to_front}")
    print(f"The rest: {the_rest}")

    return (bring_to_front, the_rest)


def get_test_case_configs(dirpath: str) -> None:
    get_slow_tests(dirpath=dirpath)
    get_disabled_tests(dirpath=dirpath)
