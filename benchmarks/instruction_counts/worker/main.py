"""File invoked through subprocess to actually carry out measurements.

`worker/main.py` is deliberately isolated from the rest of the benchmark
infrastructure. Other parts of the benchmark rely on this file, but
`worker/` has only one Python file and does not import ANYTHING from the rest
of the benchmark suite. The reason that this is important is that we can't
rely on paths to access the other files (namely `core.api`) since a source
command might change the CWD. It also helps keep startup time down by limiting
spurious definition work.

The life of a worker is very simple:
    It receives a file containing a `WorkerTimerArgs` telling it what to run,
    and writes a `WorkerOutput` result back to the same file.

Because this file only expects to run in a child context, error handling means
plumbing failures up to the caller, not raising in this process.
"""

import argparse
import dataclasses
import io
import os
import pickle
import sys
import timeit
import traceback
from typing import Any, Tuple, TYPE_CHECKING, Union


if TYPE_CHECKING:
    # Benchmark utils are only partially strict compliant, so MyPy won't follow
    # imports using the public namespace. (Due to an exclusion rule in
    # mypy-strict.ini)
    from torch.utils.benchmark.utils.timer import Language, Timer
    from torch.utils.benchmark.utils.valgrind_wrapper.timer_interface import (
        CallgrindStats,
    )

else:
    from torch.utils.benchmark import CallgrindStats, Language, Timer


WORKER_PATH = os.path.abspath(__file__)


# =============================================================================
# == Interface ================================================================
# =============================================================================

# While the point of this is mainly to collect instruction counts, we're going
# to have to compile C++ timers anyway (as they're used as a check before
# calling Valgrind), so we may as well grab wall times for reference. They
# are comparatively inexpensive.
MIN_RUN_TIME = 5

# Repeats are inexpensive as long as they are all run in the same process. This
# also lets us filter outliers (e.g. malloc arena reorganization), so we don't
# need a high CALLGRIND_NUMBER to get good data.
CALLGRIND_NUMBER = 100
CALLGRIND_REPEATS = 5


@dataclasses.dataclass(frozen=True)
class WorkerTimerArgs:
    """Container for Timer constructor arguments.

    This dataclass serves two roles. First, it is a simple interface for
    defining benchmarks. (See core.api.GroupedStmts and core.api.GroupedModules
    for the advanced interfaces.) Second, it provides serialization for
    controlling workers. `Timer` is not pickleable, so instead the main process
    will pass `WorkerTimerArgs` instances to workers for processing.
    """

    stmt: str
    setup: str = "pass"
    global_setup: str = ""
    num_threads: int = 1
    language: Language = Language.PYTHON


@dataclasses.dataclass(frozen=True)
class WorkerOutput:
    # Only return values to reduce communication between main process and workers.
    wall_times: Tuple[float, ...]
    instructions: Tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class WorkerFailure:
    # If a worker fails, we attach the string contents of the Exception
    # rather than the Exception object itself. This is done for two reasons:
    #   1) Depending on the type thrown, `e` may or may not be pickleable
    #   2) If we re-throw in the main process, we lose the true stack trace.
    failure_trace: str


class WorkerUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        """Resolve import for pickle.

        When the main runner uses a symbol `foo` from this file, it sees it as
        `worker.main.foo`. However the worker (called as a standalone file)
        sees the same symbol as `__main__.foo`. We have to help pickle
        understand that they refer to the same symbols.
        """
        symbol_map = {
            # Only blessed interface Enums and dataclasses need to be mapped.
            "WorkerTimerArgs": WorkerTimerArgs,
            "WorkerOutput": WorkerOutput,
            "WorkerFailure": WorkerFailure,
        }

        if name in symbol_map:
            return symbol_map[name]

        return super().find_class(module, name)

    def load_input(self) -> WorkerTimerArgs:
        result = self.load()
        assert isinstance(result, WorkerTimerArgs)
        return result

    def load_output(self) -> Union[WorkerTimerArgs, WorkerOutput, WorkerFailure]:
        """Convenience method for type safe loading."""
        result = self.load()
        assert isinstance(result, (WorkerTimerArgs, WorkerOutput, WorkerFailure))
        return result


# =============================================================================
# == Execution ================================================================
# =============================================================================


def _run(timer_args: WorkerTimerArgs) -> WorkerOutput:
    timer = Timer(
        stmt=timer_args.stmt,
        setup=timer_args.setup or "pass",
        global_setup=timer_args.global_setup,
        # Prevent NotImplementedError on GPU builds and C++ snippets.
        timer=timeit.default_timer,
        num_threads=timer_args.num_threads,
        language=timer_args.language,
    )

    m = timer.blocked_autorange(min_run_time=MIN_RUN_TIME)

    stats: Tuple[CallgrindStats, ...] = timer.collect_callgrind(
        number=CALLGRIND_NUMBER,
        collect_baseline=False,
        repeats=CALLGRIND_REPEATS,
        retain_out_file=False,
    )

    return WorkerOutput(
        wall_times=tuple(m.times),
        instructions=tuple(s.counts(denoise=True) for s in stats),
    )


def main(communication_file: str) -> None:
    result: Union[WorkerOutput, WorkerFailure]
    try:
        with open(communication_file, "rb") as f:
            timer_args: WorkerTimerArgs = WorkerUnpickler(f).load_input()
            assert isinstance(timer_args, WorkerTimerArgs)
        result = _run(timer_args)

    except KeyboardInterrupt:
        # Runner process sent SIGINT.
        sys.exit()

    except BaseException:
        trace_f = io.StringIO()
        traceback.print_exc(file=trace_f)
        result = WorkerFailure(failure_trace=trace_f.getvalue())

    if not os.path.exists(os.path.split(communication_file)[0]):
        # This worker is an orphan, and the parent has already cleaned up the
        # working directory. In that case we can simply exit.
        print(f"Orphaned worker {os.getpid()} exiting.")
        return

    with open(communication_file, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--communication-file", "--communication_file", type=str)
    communication_file = parser.parse_args().communication_file
    main(communication_file)
