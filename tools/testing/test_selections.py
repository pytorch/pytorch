import math
import os
import subprocess
from pathlib import Path

from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

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
    file_duration = test_file_times.get(test.test_file, None)
    if test.is_full_file():
        return file_duration

    def get_duration_for_classes(
        test_file: str, test_classes: Set[str]
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
    tests: Sequence[TestRun],
    test_file_times: Dict[str, float],
    test_class_times: Dict[str, Dict[str, float]],
    estimated_time_limit: Optional[float] = None,
    sort_by_time: bool = True,
    serial: bool = False,
) -> None:
    if len(sharded_jobs) == 0:
        assert len(tests) == 0, "No shards provided but there are tests to shard"
        return
    # Modifies sharded_jobs in place
    known_tests = tests
    unknown_tests = []
    if sort_by_time:
        known_tests = [
            x
            for x in tests
            if get_duration(x, test_file_times, test_class_times) is not None
        ]
        unknown_tests = [x for x in tests if x not in known_tests]

        assert (
            unknown_tests == [] or serial
        ), f"Attmempting to parallelize unknown tests {unknown_tests}"
    del tests

    known_tests = get_with_pytest_shard(known_tests, test_file_times, test_class_times)

    if sort_by_time:
        known_tests = sorted(known_tests, key=lambda j: j.get_time(), reverse=True)

    def _shard_serial(tests: List[ShardedTest], sharded_jobs: List[ShardJob]) -> None:
        assert estimated_time_limit is not None, "Estimated time limit must be provided"
        new_sharded_jobs = sharded_jobs
        for test in tests:
            if (
                len(sharded_jobs) > 1
                and sharded_jobs[-1].get_total_time() > estimated_time_limit
            ):
                new_sharded_jobs = sharded_jobs[:-1]
            min_sharded_job = min(new_sharded_jobs, key=lambda j: j.get_total_time())
            min_sharded_job.serial.append(test)

    def _shard_parallel(tests: List[ShardedTest], sharded_jobs: List[ShardJob]) -> None:
        for test in tests:
            min_sharded_job = min(sharded_jobs, key=lambda j: j.get_total_time())
            min_sharded_job.parallel.append(test)

    if serial:
        _shard_serial(known_tests, sharded_jobs)
    else:
        _shard_parallel(known_tests, sharded_jobs)

    # Round robin the unknown jobs starting with the smallest shard
    num_shards = len(sharded_jobs)
    index = min(range(num_shards), key=lambda i: sharded_jobs[i].get_total_time())
    for unknown_test in unknown_tests:
        sharded_jobs[index].serial.append(ShardedTest(unknown_test, 1, 1, None))
        index = (index + 1) % num_shards

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
    serial_tests = [
        test
        for test in tests
        if get_duration(test, test_file_times, test_class_times) is None
        or must_serial(test.test_file)
    ]
    parallel_tests = [test for test in tests if test not in serial_tests]

    serial_time = sum(
        get_duration(test, test_file_times, test_class_times) or 0
        for test in serial_tests
    )
    parallel_time = sum(
        get_duration(test, test_file_times, test_class_times) or 0
        for test in parallel_tests
    )
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
        num_serial_shards = math.ceil(serial_time / total_time * num_shards)

    sharded_jobs = [ShardJob() for _ in range(num_shards)]
    shard(
        sharded_jobs[:num_serial_shards],
        serial_tests,
        test_file_times,
        test_class_times,
        estimated_time_limit=estimated_time_limit,
        sort_by_time=sort_by_time,
        serial=True,
    )
    shard(
        sharded_jobs,
        parallel_tests,
        test_file_times,
        test_class_times,
        sort_by_time=sort_by_time,
        serial=False,
    )

    return [job.convert_to_tuple() for job in sharded_jobs]


def get_test_case_configs(dirpath: str) -> None:
    get_slow_tests(dirpath=dirpath)
    get_disabled_tests(dirpath=dirpath)
