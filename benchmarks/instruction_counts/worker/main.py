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
import os
import pickle
from typing import Any, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    # Benchmark utils are only partially strict compliant, so MyPy won't follow
    # imports using the public namespace. (Due to an exclusion rule in
    # mypy-strict.ini)
    from torch.utils.benchmark.utils.common import Measurement
    from torch.utils.benchmark.utils.timer import Language
    from torch.utils.benchmark.utils.valgrind_wrapper.timer_interface import CallgrindStats
else:
    from torch.utils.benchmark import CallgrindStats, Language, Measurement


WORKER_PATH = os.path.abspath(__file__)


# =============================================================================
# == Interfaces ===============================================================
# =============================================================================
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
    setup: Optional[str] = None
    global_setup: Optional[str] = None
    num_threads: int = 1
    language: Language = Language.PYTHON


@dataclasses.dataclass(frozen=True)
class WorkerOutput:
    wall_time: Measurement
    instructions: CallgrindStats


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
            "WorkerTimerArgs": WorkerTimerArgs,
            "WorkerOutput": WorkerOutput,
            "WorkerFailure": WorkerFailure,
        }.get(name, None) or super().find_class(module, name)

    def load_from_worker(self) -> Union[WorkerTimerArgs, WorkerOutput, WorkerFailure]:
        """Convenience method for type safe loading."""
        result = self.load()
        assert isinstance(result, (WorkerTimerArgs, WorkerOutput, WorkerFailure))
        return result
