import heapq
import json
import math
import os
import subprocess
from pathlib import Path

from typing import Callable, Dict, List, NamedTuple, Optional, Set, Tuple
from warnings import warn

from tools.shared.logging_utils import duration_to_str, pluralize

from tools.stats.import_test_stats import get_disabled_tests, get_slow_tests

IS_MEM_LEAK_CHECK = os.getenv("PYTORCH_TEST_CUDA_MEM_LEAK_CHECK", "0") == "1"

# NUM_PROCS_FOR_SHARDING_CALC must remain consistent across all shards of a job
# to ensure that sharding is consistent, NUM_PROCS is the actual number of procs
# used to run tests.  If they are not equal, the only consequence should be
# unequal shards.
IS_ROCM = os.path.exists("/opt/rocm")
NUM_PROCS = 1 if IS_MEM_LEAK_CHECK else 2
NUM_PROCS_FOR_SHARDING_CALC = NUM_PROCS if not IS_ROCM or IS_MEM_LEAK_CHECK else 2
THRESHOLD = 60 * 10  # 10 minutes

# See Note [ROCm parallel CI testing]
# Special logic for ROCm GHA runners to query number of GPUs available.
# torch.version.hip was not available to check if this was a ROCm self-hosted runner.
# Must check for ROCm runner in another way. We look for /opt/rocm directory.
if IS_ROCM and not IS_MEM_LEAK_CHECK:
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
    time: Optional[float]  # In seconds

    def __str__(self) -> str:
        return f"{self.name} {self.shard}/{self.num_shards}"

    def get_time(self) -> float:
        return self.time or 0


class ShardJob:
    def __init__(self) -> None:
        self.serial: List[ShardedTest] = []
        self.parallel: List[ShardedTest] = []

    def get_total_time(self) -> float:
        procs = [0.0 for _ in range(NUM_PROCS_FOR_SHARDING_CALC)]
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
    debug: bool = False,
) -> List[Tuple[float, List[ShardedTest]]]:
    must_serial = must_serial or (lambda x: True)

    if debug:
        print(test_file_times)
        print(tests)
        print(num_shards)
        print([x for x in tests if must_serial(x)])

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

    if debug:
        for j in sharded_jobs:
            print(j.convert_to_tuple()[1])

    return [job.convert_to_tuple() for job in sharded_jobs]


def _query_changed_test_files() -> List[str]:
    default_branch = f"origin/{os.environ.get('GIT_DEFAULT_BRANCH', 'main')}"
    merge_base = (
        subprocess.check_output(["git", "merge-base", default_branch, "HEAD"])
        .decode()
        .strip()
    )

    head = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

    base_commit = merge_base
    if base_commit == head:
        # We are on the default branch, so check for changes since the last commit
        base_commit = "HEAD^"

    proc = subprocess.run(
        ["git", "diff", "--name-only", base_commit, "HEAD"], capture_output=True
    )

    if proc.returncode != 0:
        raise RuntimeError("Unable to get changed files")

    lines = proc.stdout.decode().strip().split("\n")
    lines = [line.strip() for line in lines]
    return lines


def _get_previously_failing_tests() -> Set[str]:
    PYTEST_FAILED_TESTS_CACHE_FILE_PATH = Path(".pytest_cache/v/cache/lastfailed")

    if not PYTEST_FAILED_TESTS_CACHE_FILE_PATH.exists():
        warn(
            f"No pytorch cache found at {PYTEST_FAILED_TESTS_CACHE_FILE_PATH.absolute()}"
        )
        return set()

    with open(PYTEST_FAILED_TESTS_CACHE_FILE_PATH, "r") as f:
        last_failed_tests = json.load(f)

    prioritized_tests = _parse_prev_failing_test_files(last_failed_tests)
    return _python_test_file_to_test_name(prioritized_tests)


def _parse_prev_failing_test_files(last_failed_tests: Dict[str, bool]) -> Set[str]:
    prioritized_tests = set()

    # The keys are formatted as "test_file.py::test_class::test_method[params]"
    # We just need the test_file part
    for test in last_failed_tests:
        parts = test.split("::")
        if len(parts) > 1:
            test_file = parts[0]
            prioritized_tests.add(test_file)

    return prioritized_tests


def _get_modified_tests() -> Set[str]:
    try:
        changed_files = _query_changed_test_files()
    except Exception as e:
        warn(f"Can't query changed test files due to {e}")
        # If unable to get changed files from git, quit without doing any sorting
        return set()

    return _python_test_file_to_test_name(set(changed_files))


def _python_test_file_to_test_name(tests: Set[str]) -> Set[str]:
    prefix = f"test{os.path.sep}"
    valid_tests = {f for f in tests if f.startswith(prefix) and f.endswith(".py")}
    valid_tests = {f[len(prefix) : -len(".py")] for f in valid_tests}

    return valid_tests


class PoolTimes:
    def __init__(self, num_procs: int) -> None:
        self.pool_times = [0.0 for _ in range(num_procs)]
        self.serial_times = 0.0

    def next_test_start_time(self, serial: bool) -> float:
        if serial:
            # Serial tests are run after all parallel tests complete
            return max(self.pool_times) + self.serial_times

        return self.pool_times[0]

    def schedule_test(self, test: ShardedTest, serial: bool) -> None:
        if serial:
            self.serial_times += test.get_time()
        else:
            # pool_times[0] is always the thread with the least amount of time scheduled
            heapq.heappushpop(self.pool_times, self.pool_times[0] + test.get_time())


def log_time_savings(
    selected_tests: List[ShardedTest],
    prioritized_tests: List[ShardedTest],
    is_serial_test_fn: Callable[[str], bool],
    num_procs: int = NUM_PROCS,  # make this customizable for testing
) -> float:
    # The tests will be run in [num_procs] parallel threads, so we assume each test
    # is allocated to the thread that'll free up first.
    # This isn't an exact match (since other factors could change which thread
    # pool a test gets scheduled on) but it's a good approximation.

    # Simulates the scheduled tests on each thread pool
    default_pool = PoolTimes(num_procs)  # originally scheduled run
    prioritized_pool = PoolTimes(num_procs)  # run for prioritized tests
    max_time_savings_sec = 0.0

    # It's easier to look up prioritized tests by name
    prioritized_test_names = {test.name for test in prioritized_tests}

    for test in selected_tests:
        serial = is_serial_test_fn(test.name)
        if test.name in prioritized_test_names:
            # Successive tests will always have a greater time savings
            max_time_savings_sec = default_pool.next_test_start_time(
                serial
            ) - prioritized_pool.next_test_start_time(serial)

            # "schedule" this test on the prioritized pool to get time savings for future prioritized tests
            prioritized_pool.schedule_test(test, serial)

        # always schedule on the default pool to know what the unprioritized timeline would've looked like
        default_pool.schedule_test(test, serial)

    print(
        f"Prioritized tests will run about {duration_to_str(max_time_savings_sec)} sooner than they would've otherwise"
    )

    # Return value used by tests
    return max_time_savings_sec


def get_reordered_tests(
    tests: List[ShardedTest],
) -> Tuple[List[ShardedTest], List[ShardedTest]]:
    """
    Get the reordered test filename list based on github PR history or git changed file.
    We prioritize running test files that were changed.
    """

    def print_tests(tests: Set[str], test_group_description: str) -> None:
        if not tests:
            return

        print(f"{test_group_description}:")
        for test in tests:
            print(f"  {test}")

    prioritized_tests: Set[str] = set()

    pri_test = _get_previously_failing_tests()
    print_tests(
        pri_test, "If run, these tests will prioritized because they previously failed"
    )
    prioritized_tests |= pri_test

    pri_test |= _get_modified_tests()
    print_tests(
        pri_test, "If run, these tests will be prioritized because they were modified"
    )
    prioritized_tests |= pri_test

    bring_to_front = []
    the_rest = []

    for test in tests:
        if test.name in prioritized_tests:
            bring_to_front.append(test)
        else:
            the_rest.append(test)

    if len(tests) != len(bring_to_front) + len(the_rest):
        print(
            f"Something went wrong in CI reordering, expecting total of {len(tests)}:\n"
            f"but found prioritized: {len(bring_to_front)}\nthe rest: {len(the_rest)}\n"
        )
        return ([], tests)

    # TODO: Would be great to upload these stats to RDS/Rockset!
    if bring_to_front:
        test_cnt_str = pluralize(len(tests), "test")
        print(f"Reordering tests: Prioritizing {len(bring_to_front)} of {test_cnt_str}")

        prioritized_test_names = [t.name for t in bring_to_front]
        print(f"Prioritized: {prioritized_test_names}")
        remaining_test_names = [t.name for t in the_rest]
        print(f"The Rest: {remaining_test_names}")
    else:
        print("Didn't find any tests to prioritize")

    return (bring_to_front, the_rest)


def get_test_case_configs(dirpath: str) -> None:
    get_slow_tests(dirpath=dirpath)
    get_disabled_tests(dirpath=dirpath)
