import math
import os
import subprocess
from pathlib import Path

from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from tools.stats.import_test_stats import get_disabled_tests, get_slow_tests
from tools.testing.test_run import ShardedTest, TestRun

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

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
        NUM_PROCS = min(count, 8)
    except subprocess.CalledProcessError as e:
        # The safe default for ROCm GHA runners is to run tests serially.
        NUM_PROCS = 1


class ShardJob:
    def __init__(self) -> None:
        self.serial: List[ShardedTest] = []
        self.parallel: List[ShardedTest] = []

    def get_total_time(self) -> float:
        """Default is the value for which to substitute if a test has no time"""
        procs = [0.0 for _ in range(NUM_PROCS_FOR_SHARDING_CALC)]
        for test in self.parallel:
            min_index = procs.index(min(procs))
            procs[min_index] += test.get_time()
        time = max(procs) + sum(test.get_time() for test in self.serial)
        return time

    def convert_to_tuple(self) -> Tuple[float, List[ShardedTest]]:
        return (self.get_total_time(), self.serial + self.parallel)


def get_with_pytest_shard(
    tests: Sequence[TestRun],
    test_file_times: Dict[str, float],
    test_class_times: Optional[Dict[str, Dict[str, float]]],
) -> List[ShardedTest]:
    sharded_tests: List[ShardedTest] = []

    for test in tests:
        duration = get_duration(test, test_file_times, test_class_times or {})

        if duration and duration > THRESHOLD:
            num_shards = math.ceil(duration / THRESHOLD)
            for i in range(num_shards):
                sharded_tests.append(
                    ShardedTest(test, i + 1, num_shards, duration / num_shards)
                )
        else:
            sharded_tests.append(ShardedTest(test, 1, 1, duration))
    return sharded_tests


def get_duration(
    test: TestRun,
    test_file_times: Dict[str, float],
    test_class_times: Dict[str, Dict[str, float]],
) -> Optional[float]:
    """Calculate the time for a TestRun based on the given test_file_times and
    test_class_times.  Returns None if the time is unknown."""
    file_duration = test_file_times.get(test.test_file, None)
    if test.is_full_file():
        return file_duration

    def get_duration_for_classes(
        test_file: str, test_classes: FrozenSet[str]
    ) -> Optional[float]:
        duration: float = 0

        for test_class in test_classes:
            class_duration = test_class_times.get(test_file, {}).get(test_class, None)
            if class_duration is None:
                return None
            duration += class_duration
        return duration

    included = test.included()
    excluded = test.excluded()
    included_classes_duration = get_duration_for_classes(test.test_file, included)
    excluded_classes_duration = get_duration_for_classes(test.test_file, excluded)

    if included_classes_duration is None or excluded_classes_duration is None:
        # Didn't get the time for all classes, so time is unknown
        return None

    if included:
        return included_classes_duration
    assert (
        excluded
    ), f"TestRun {test} is not full file but doesn't have included or excluded classes"
    if file_duration is None:
        return None
    return file_duration - excluded_classes_duration


def shard(
    sharded_jobs: List[ShardJob],
    pytest_sharded_tests: Sequence[ShardedTest],
    estimated_time_limit: Optional[float] = None,
    serial: bool = False,
) -> None:
    # Modifies sharded_jobs in place
    if len(sharded_jobs) == 0:
        assert (
            len(pytest_sharded_tests) == 0
        ), "No shards provided but there are tests to shard"
        return

    round_robin_index = 0

    def _get_min_sharded_job(
        sharded_jobs: List[ShardJob], test: ShardedTest
    ) -> ShardJob:
        if test.time is None:
            nonlocal round_robin_index
            job = sharded_jobs[round_robin_index % len(sharded_jobs)]
            round_robin_index += 1
            return job
        return min(sharded_jobs, key=lambda j: j.get_total_time())

    def _shard_serial(
        tests: Sequence[ShardedTest], sharded_jobs: List[ShardJob]
    ) -> None:
        assert estimated_time_limit is not None, "Estimated time limit must be provided"
        new_sharded_jobs = sharded_jobs
        for test in tests:
            if (
                len(sharded_jobs) > 1
                and sharded_jobs[-1].get_total_time() > estimated_time_limit
            ):
                new_sharded_jobs = sharded_jobs[:-1]
            min_sharded_job = _get_min_sharded_job(new_sharded_jobs, test)
            min_sharded_job.serial.append(test)

    def _shard_parallel(
        tests: Sequence[ShardedTest], sharded_jobs: List[ShardJob]
    ) -> None:
        for test in tests:
            min_sharded_job = _get_min_sharded_job(sharded_jobs, test)
            min_sharded_job.parallel.append(test)

    if serial:
        _shard_serial(pytest_sharded_tests, sharded_jobs)
    else:
        _shard_parallel(pytest_sharded_tests, sharded_jobs)

    return


def calculate_shards(
    num_shards: int,
    tests: Sequence[TestRun],
    test_file_times: Dict[str, float],
    test_class_times: Optional[Dict[str, Dict[str, float]]],
    must_serial: Optional[Callable[[str], bool]] = None,
    sort_by_time: bool = True,
) -> List[Tuple[float, List[ShardedTest]]]:
    must_serial = must_serial or (lambda x: True)
    test_class_times = test_class_times or {}

    # Divide tests into pytest shards
    if sort_by_time:
        known_tests = [
            x
            for x in tests
            if get_duration(x, test_file_times, test_class_times) is not None
        ]
        unknown_tests = [x for x in tests if x not in known_tests]

        pytest_sharded_tests = sorted(
            get_with_pytest_shard(known_tests, test_file_times, test_class_times),
            key=lambda j: j.get_time(),
            reverse=True,
        ) + get_with_pytest_shard(unknown_tests, test_file_times, test_class_times)
    else:
        pytest_sharded_tests = get_with_pytest_shard(
            tests, test_file_times, test_class_times
        )
    del tests

    serial_tests = [test for test in pytest_sharded_tests if must_serial(test.name)]
    parallel_tests = [test for test in pytest_sharded_tests if test not in serial_tests]

    serial_time = sum(test.get_time() for test in serial_tests)
    parallel_time = sum(test.get_time() for test in parallel_tests)
    total_time = serial_time + parallel_time / NUM_PROCS_FOR_SHARDING_CALC
    estimated_time_per_shard = total_time / num_shards
    # Separate serial tests from parallel tests as much as possible to maximize
    # parallelism by putting all the serial tests on the first num_serial_shards
    # shards. The estimated_time_limit is the estimated time it should take for
    # the least filled serial shard. Ex if we have 8 min of serial tests, 20 min
    # of parallel tests, 6 shards, and 2 procs per machine, we would expect each
    # machine to take 3 min and should aim for 3 serial shards, with shards 1
    # and 2 taking 3 min and shard 3 taking 2 min.  The estimated time limit
    # would be 2 min. This ensures that the first few shard contains as many
    # serial tests as possible and as few parallel tests as possible. The least
    # filled/last (in the example, the 3rd) shard may contain a lot of both
    # serial and parallel tests.
    estimated_time_limit = 0.0
    if estimated_time_per_shard != 0:
        estimated_time_limit = serial_time % estimated_time_per_shard
    if estimated_time_limit <= 0.01:
        estimated_time_limit = estimated_time_per_shard
    if total_time == 0:
        num_serial_shards = num_shards
    else:
        num_serial_shards = max(math.ceil(serial_time / total_time * num_shards), 1)

    sharded_jobs = [ShardJob() for _ in range(num_shards)]
    shard(
        sharded_jobs=sharded_jobs[:num_serial_shards],
        pytest_sharded_tests=serial_tests,
        estimated_time_limit=estimated_time_limit,
        serial=True,
    )
    shard(
        sharded_jobs=sharded_jobs,
        pytest_sharded_tests=parallel_tests,
        serial=False,
    )

    return [job.convert_to_tuple() for job in sharded_jobs]


def get_test_case_configs(dirpath: str) -> None:
    get_slow_tests(dirpath=dirpath)
    get_disabled_tests(dirpath=dirpath)
