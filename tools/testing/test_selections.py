from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

from tools.stats.import_test_stats import ADDITIONAL_CI_FILES_FOLDER, get_disabled_tests
from tools.testing.target_determination.gen_artifact import (
    gen_ci_artifact,
    LLM_SELECTION_SOURCE,
    TD_SOURCE,
)
from tools.testing.test_run import ShardedTest, TestRun


try:
    import torch
    from torch.testing._internal.common_cuda import SM80OrLater
    from torch.testing._internal.common_utils import TEST_CUDA
except ImportError:
    torch = None
    TEST_CUDA = False
    SM80OrLater = False


if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    from tools.testing.llm_selector.selection import AppliedSelection
    from tools.testing.target_determination.heuristics.interface import (
        TestPrioritizations,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
# tools/testing/llm_selector/selection.py's LLM_SELECTION_FILE, which is imported
# only when the file exists, so the experimental package cannot break run_test.py.
LLM_SELECTION_FILE = "llm_selection.json"

IS_MEM_LEAK_CHECK = os.getenv("PYTORCH_TEST_CUDA_MEM_LEAK_CHECK", "0") == "1"
BUILD_ENVIRONMENT = os.getenv("BUILD_ENVIRONMENT", "")

# NUM_PROCS_FOR_SHARDING_CALC must remain consistent across all shards of a job
# to ensure that sharding is consistent, NUM_PROCS is the actual number of procs
# used to run tests.  If they are not equal, the only consequence should be
# unequal shards.
# Detect ROCm via torch.version.hip, which is set for both system installs and
# ROCm wheels (e.g. TheRock preview wheels which have no /opt/rocm). This must
# reach the rocminfo-based NUM_PROCS clamp below; otherwise NUM_PROCS stays
# >GPU-count and maybe_set_hip_visible_devies() in run_test.py assigns workers
# to nonexistent device indices -> torch.cuda.device_count()==0.
IS_ROCM = torch is not None and torch.version.hip is not None
NUM_PROCS = 1 if IS_MEM_LEAK_CHECK else 3 if not TEST_CUDA or SM80OrLater else 2
NUM_PROCS_FOR_SHARDING_CALC = NUM_PROCS if not IS_ROCM or IS_MEM_LEAK_CHECK else 2
THRESHOLD = 60 * 10  # 10 minutes

# See Note [ROCm parallel CI testing]
# Special logic for ROCm GHA runners to query number of GPUs available.
# hipInfo gcnArchName lines also contain " gfx", so the same count works on Windows.
if IS_ROCM and not IS_MEM_LEAK_CHECK:
    gpu_info_cmds = ["hipInfo", "rocminfo"] if os.name == "nt" else ["rocminfo"]
    gpu_count = 0
    tool_ran = False
    for gpu_info_cmd in gpu_info_cmds:
        try:
            # errors="replace" tolerates non-ASCII device marketing names on Windows.
            output = subprocess.check_output(
                [gpu_info_cmd], encoding="utf-8", errors="replace"
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
        tool_ran = True
        gpu_count = sum(" gfx" in line for line in output.strip().split("\n"))
        if gpu_count > 0:
            break
    if gpu_count > 0:
        # Limiting to 8 GPUs(PROCS)
        NUM_PROCS = min(gpu_count, 8)
    elif tool_ran:
        # A GPU runner whose query tools see no devices is unhealthy; fail loudly.
        raise AssertionError("There must be at least 1 GPU")
    else:
        # No GPU query tool available; the safe default is to run tests serially.
        NUM_PROCS = 1


class ShardJob:
    def __init__(self) -> None:
        self.serial: list[ShardedTest] = []
        self.parallel: list[ShardedTest] = []

    def get_total_time(self) -> float:
        """Default is the value for which to substitute if a test has no time"""
        procs = [0.0 for _ in range(NUM_PROCS_FOR_SHARDING_CALC)]
        for test in self.parallel:
            min_index = procs.index(min(procs))
            procs[min_index] += test.get_time()
        time = max(procs) + sum(test.get_time() for test in self.serial)
        return time

    def convert_to_tuple(self) -> tuple[float, list[ShardedTest]]:
        return (self.get_total_time(), self.serial + self.parallel)


def get_with_pytest_shard(
    tests: Sequence[TestRun],
    test_file_times: dict[str, float],
    test_class_times: dict[str, dict[str, float]] | None,
    *,
    allow_pytest_sharding: bool = True,
) -> list[ShardedTest]:
    sharded_tests: list[ShardedTest] = []

    for test in tests:
        duration = get_duration(test, test_file_times, test_class_times or {})

        if allow_pytest_sharding and duration and duration > THRESHOLD:
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
    test_file_times: dict[str, float],
    test_class_times: dict[str, dict[str, float]],
) -> float | None:
    """Calculate the time for a TestRun based on the given test_file_times and
    test_class_times.  Returns None if the time is unknown."""
    file_duration = test_file_times.get(test.test_file, None)
    if test.is_full_file():
        return file_duration

    def get_duration_for_classes(
        test_file: str, test_classes: frozenset[str]
    ) -> float | None:
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
    if not excluded:
        raise AssertionError(
            f"TestRun {test} is not full file but doesn't have included or excluded classes"
        )
    if file_duration is None:
        return None
    return file_duration - excluded_classes_duration


def shard(
    sharded_jobs: list[ShardJob],
    pytest_sharded_tests: Sequence[ShardedTest],
    estimated_time_limit: float | None = None,
    serial: bool = False,
) -> None:
    # Modifies sharded_jobs in place
    if len(sharded_jobs) == 0:
        if len(pytest_sharded_tests) != 0:
            raise AssertionError("No shards provided but there are tests to shard")
        return

    round_robin_index = 0

    def _get_min_sharded_job(
        sharded_jobs: list[ShardJob], test: ShardedTest
    ) -> ShardJob:
        if test.time is None:
            nonlocal round_robin_index
            job = sharded_jobs[round_robin_index % len(sharded_jobs)]
            round_robin_index += 1
            return job
        return min(sharded_jobs, key=lambda j: j.get_total_time())

    def _shard_serial(
        tests: Sequence[ShardedTest], sharded_jobs: list[ShardJob]
    ) -> None:
        if estimated_time_limit is None:
            raise AssertionError("Estimated time limit must be provided")
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
        tests: Sequence[ShardedTest], sharded_jobs: list[ShardJob]
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
    test_file_times: dict[str, float],
    test_class_times: dict[str, dict[str, float]] | None,
    must_serial: Callable[[str], bool] | None = None,
    sort_by_time: bool = True,
    allow_pytest_sharding: bool = True,
) -> list[tuple[float, list[ShardedTest]]]:
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
            get_with_pytest_shard(
                known_tests,
                test_file_times,
                test_class_times,
                allow_pytest_sharding=allow_pytest_sharding,
            ),
            key=lambda j: j.get_time(),
            reverse=True,
        ) + get_with_pytest_shard(
            unknown_tests,
            test_file_times,
            test_class_times,
            allow_pytest_sharding=allow_pytest_sharding,
        )
    else:
        pytest_sharded_tests = get_with_pytest_shard(
            tests,
            test_file_times,
            test_class_times,
            allow_pytest_sharding=allow_pytest_sharding,
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
    get_disabled_tests(dirpath=dirpath)


# Strip the "test"/"test-osdc" target suffix to recover the build env, the key
# tools/torchci writes to test-times.json. Must match the write-side extraction.
JOB_BASE_NAME_RE = re.compile(r" / test(?:-osdc)? \(")


def get_job_base_name(job_name: str) -> str:
    return JOB_BASE_NAME_RE.split(job_name, maxsplit=1)[0]


@dataclass
class CallSplit:
    """The split of one run_test.py call's tests that runs, and the LLM test
    selection applied to TD's split if CI downloaded one."""

    include: list[TestRun]
    exclude: list[TestRun]
    llm: AppliedSelection | None

    @property
    def llm_enforced(self) -> bool:
        return self.llm is not None and self.llm.enforced

    def selection_metrics(self, is_ci: bool) -> list[tuple[str, dict[str, Any]]]:
        """The (name, payload) metrics run_test.py emits before running the
        tests: llm_selection, in CI with a selection."""
        if not is_ci or (llm := self.llm) is None:
            return []
        return _llm_metric("llm_selection", lambda: llm.selection_metric)

    def failure_metrics(
        self, test: TestRun, selected_tests: list[str], test_stats: dict[str, Any]
    ) -> list[tuple[str, dict[str, Any]]]:
        """The (name, payload) metrics run_test.py emits in CI for a failed
        `test`: td_test_failure_stats_v2 and, with a selection, llm_selection_failure."""
        stats = {
            "selected_tests": selected_tests,
            "failure": str(test),
            "llm_enforced": self.llm_enforced,
            **test_stats,
        }
        metrics = [("td_test_failure_stats_v2", stats)]
        if (llm := self.llm) is not None:
            metrics += _llm_metric("llm_selection_failure", lambda: llm.failure_metric(test))  # fmt: skip
        return metrics


def _llm_metric(
    name: str, payload: Callable[[], dict[str, Any]]
) -> list[tuple[str, dict[str, Any]]]:
    """[(name, payload())], or none, with a ::warning::, if the experimental
    llm_selector package raises."""
    try:
        return [(name, payload())]
    except Exception as e:
        print(f"::warning::Not emitting {name}: {e!r}", file=sys.stderr)
        return []


def split_tests(
    test_prioritizations: TestPrioritizations,
    test_times: dict[str, float],
    *,
    selection_file: Path,
    test_config: str,
    td_enabled: bool,
    previous_failures: Collection[str] | None,
    record: bool,
) -> CallSplit:
    """TD's split, 25% with TD enabled and everything otherwise, replaced by
    the LLM test selection in `selection_file` when apply_selection enforces
    it (see tools/testing/llm_selector). With `record` (in CI), gen_ci_artifact
    records the split that runs and what TD's split excluded. A selection
    that fails to import, load or apply keeps TD's split. run_test.py emits
    CallSplit's metrics, since emit_metric records its caller."""
    percent_to_run = 25 if td_enabled else 100
    include, exclude = test_prioritizations.get_top_per_tests(percent_to_run)
    llm: AppliedSelection | None = None
    if selection_file.exists():
        try:
            from tools.testing.llm_selector.selection import apply_selection, Selection

            selection = Selection.load(selection_file)
            if selection is not None:
                llm = apply_selection(
                    selection,
                    include,
                    exclude,
                    test_prioritizations.get_score,
                    test_times,
                    test_config=test_config,
                    td_enabled=td_enabled,
                    previous_failures=previous_failures,
                )
        except Exception as e:
            print(f"::warning::Ignoring the LLM selection: {e!r}", file=sys.stderr)
    if llm is not None and llm.enforced:
        split = CallSplit(llm.split.include, llm.split.exclude, llm)
    else:
        split = CallSplit(include, exclude, llm)
        td_line = f"Running {percent_to_run}% of tests based on TD"
        print(td_line if td_enabled else "Running all tests", file=sys.stderr)
    if record:
        gen_ci_artifact(
            [x.to_json() for x in split.include],
            [x.to_json() for x in split.exclude],
            source=LLM_SELECTION_SOURCE if split.llm_enforced else TD_SOURCE,
            td_excluded=[x.to_json() for x in exclude],
        )
    return split


def run_test_split(
    test_prioritizations: TestPrioritizations,
    test_times: dict[str, float],
    *,
    previous_failures: Collection[str] | None,
    test_config: str,
    td_enabled: bool,
    is_ci: bool,
    repo_root: Path = REPO_ROOT,
) -> CallSplit:
    """split_tests as run_test.py calls it: the selection CI downloaded, and
    the split recorded only in CI. `previous_failures` is import_results'
    value, None when unknown, which keeps TD's split."""
    return split_tests(
        test_prioritizations,
        test_times,
        selection_file=repo_root / ADDITIONAL_CI_FILES_FOLDER / LLM_SELECTION_FILE,
        test_config=test_config,
        td_enabled=td_enabled,
        previous_failures=previous_failures,
        record=is_ci,
    )
