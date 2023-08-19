import math
import os
import subprocess
from pathlib import Path

from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

from tools.stats.import_test_stats import get_disabled_tests, get_slow_tests
from tools.stats.upload_stats_lib import emit_metric

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
        duration = test_file_times.get(test, None)
        if duration and duration > THRESHOLD:
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
    sort_by_time: bool = True,
) -> List[Tuple[float, List[ShardedTest]]]:
    must_serial = must_serial or (lambda x: True)

    known_tests = tests
    unknown_tests = []

    if sort_by_time:
        known_tests = [x for x in tests if x in test_file_times]
        unknown_tests = [x for x in tests if x not in known_tests]

    known_tests = get_with_pytest_shard(known_tests, test_file_times)

    if sort_by_time:
        known_tests = sorted(known_tests, key=lambda j: j.get_time(), reverse=True)

    sharded_jobs: List[ShardJob] = [ShardJob() for _ in range(num_shards)]
    for test in known_tests:
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


def get_test_case_configs(dirpath: str) -> None:
    get_slow_tests(dirpath=dirpath)
    get_disabled_tests(dirpath=dirpath)
