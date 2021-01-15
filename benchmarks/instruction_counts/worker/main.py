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
import dataclasses
import enum
import os
import pickle
from typing import Any, Dict, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    # Benchmark utils are only partially strict compliant, so MyPy won't follow
    # imports using the public namespace. (Due to an exclusion rule in
    # mypy-strict.ini)
    from torch.utils.benchmark.utils.common import Measurement
    from torch.utils.benchmark.utils.timer import Language, Timer
    from torch.utils.benchmark.utils.valgrind_wrapper.timer_interface import CallgrindStats
else:
    from torch.utils.benchmark import CallgrindStats, Language, Measurement, Timer

WORKER_PATH = os.path.abspath(__file__)


# =============================================================================
# == Interfaces ===============================================================
# =============================================================================
class CostEstimate(enum.Enum):
    """Hint for how expensive a benchmark is expected to be.

    Timer supports adaptive timing for wall times, but not instruction counts.
    Generally this is desired since we want deterministic instruction counts,
    however it can be tedious to choose sensible numbers when defining a slew
    of benchmarks.
    """
    AUTO = 0
    LESS_THAN_10_US = 1
    LESS_THAN_50_US = 2
    LESS_THAN_100_US = 3
    LESS_THAN_250_US = 4
    LESS_THAN_1000_US = 5
    GIANT = 6


# While the point of this is mainly to collect instruction counts, we're going
# to have to compile C++ timers anyway (as they're used as a check before
# calling Valgrind), so we may as well grab wall times for reference. They
# are comparatively inexpensive, and also useful for enabling CostEstimate.AUTO.
MIN_RUN_TIME = 5


@dataclasses.dataclass(frozen=True)
class WorkerTimerArgs:
    """Mirrors core.api.TimerArgs

    Note that `num_threads` is narrowed from `Union[int, Tuple[int, ...]]` to
    `int`. `core.api` will assert that `WorkerTimerArgs` matches `TimerArgs`.
    """
    stmt: str
    setup: str
    global_setup: Optional[str]
    num_threads: int
    language: Language
    cost: CostEstimate

    @classmethod
    def keys(cls) -> Tuple[str, ...]:
        return tuple(f.name for f in dataclasses.fields(cls))


@dataclasses.dataclass(frozen=True)
class WorkerOutput:
    wall_time: Measurement
    instructions: CallgrindStats
    cost: CostEstimate  # Emperical cost. (If AUTO.)


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
        return {
            # Only blessed interface Enums and dataclasses need to be mapped.
            "CostEstimate": CostEstimate,
            "WorkerTimerArgs": WorkerTimerArgs,
            "WorkerOutput": WorkerOutput,
            "WorkerFailure": WorkerFailure,
        }.get(name, None) or super().find_class(module, name)

    def load_from_worker(self) -> Union[WorkerTimerArgs, WorkerOutput, WorkerFailure]:
        """Convenience method for type safe loading."""
        result = self.load()
        assert isinstance(result, (WorkerTimerArgs, WorkerOutput, WorkerFailure))
        return result
