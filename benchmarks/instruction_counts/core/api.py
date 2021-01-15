"""Key enums and structs used to handle data flow within the benchmark."""
import dataclasses
import enum
import re
import textwrap
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

from worker.main import CostEstimate, WorkerTimerArgs

if TYPE_CHECKING:
    # Benchmark utils are only partially strict compliant, so MyPy won't follow
    # imports using the public namespace. (Due to an exclusion rule in
    # mypy-strict.ini)
    from torch.utils.benchmark.utils.common import Measurement
    from torch.utils.benchmark.utils.timer import Language
    from torch.utils.benchmark.utils.valgrind_wrapper.timer_interface import CallgrindStats
else:
    from torch.utils.benchmark import CallgrindStats, Language, Measurement


class Setup(enum.Enum):
    """Defines the class of setup that a stmt requires.

    Because a GroupedTimerArgs may (and generally will) represent both Python
    and C++ code, we chunk setup into broad groups which are resolved into
    language specific strings. This also results in more compact and readable
    definitions.
    """
    NONE = 0
    TRIVIAL = 1
    GENERIC = 2
    INDEXING = 3
    MESOSCALE = 4
    AUTOGRAD = 5
    EXAMPLE_FOR_ADHOC = 6


class Mode(enum.Enum):
    # Generated from GroupedTimerArgs
    PY = "Python"
    CPP = "C++"
    PY_TS = "Python (TorchScript)"
    CPP_TS = "C++ (TorchScript)"

    # TimerArgs was explicitly provided.
    EXPLICIT_PY = "Explicit (Py)"
    EXPLICIT_CPP = "Explicit (C++)"

    @property
    def language(self) -> Language:
        py_values = (Mode.PY, Mode.PY_TS, Mode.EXPLICIT_PY)
        return Language.PYTHON if self in py_values else Language.CPP


@dataclasses.dataclass(frozen=True)
class TimerArgs:
    """Container for Timer constructor arguments.

    This dataclass serves two roles. First, it is a simple interface for
    defining benchmarks. (See GroupedTimerArgs for the advanced interface.)
    Second, it provides serialization for controlling workers. `Timer` is not
    pickleable, so instead the parent process will pass `WorkerTimerArgs`
    (which map closely to `TimerArgs`) instances to workers for processing.
    """

    # Timer constructor arguments.
    stmt: str
    setup: str
    global_setup: Optional[str] = None
    num_threads: Union[int, Tuple[int, ...]] = 1
    language: Language = Language.PYTHON

    # Unlike `adaptive_autorange`, `collect_callgrind` does not dynamically
    # adjust based on the cost of a stmt, so we must either provide a cost
    # estimate or tell workers to determine a sensible value.
    cost: CostEstimate = CostEstimate.AUTO

    def flatten(self) -> Tuple[WorkerTimerArgs, ...]:
        self_dict = dataclasses.asdict(self)
        assert tuple(self_dict.keys()) == WorkerTimerArgs.keys()
        if isinstance(self.num_threads, int):
            return WorkerTimerArgs(**self_dict),

        num_threads: Tuple[int, ...] = self_dict.pop("num_threads")
        return tuple(WorkerTimerArgs(num_threads=n, **self_dict) for n in num_threads)


@dataclasses.dataclass(frozen=True)
class GroupedTimerArgs:
    """Defines a set of related benchmarks which are semantically equivalent.

    There are four ways one might reasonably wish to run a PyTorch snippet:
      - Using the Python eager API
      - Using the C++ eager frontend
      - Running a TorchScript model eagerly from Python
      - Running a TorchScript model which has been loaded into C++

    It is useful to define them together, both for clairity when reading
    benchmark definitions and for later processing and analysis.

    We may, of course, only be interested in a subset of cases. For instance we
    may be benchmarking Python code which does not have a C++ analog, or a
    statement which is not TorchScript-able. This is supported by simply
    omitting arguments.

    In order to measure TorchScript performance, `py_stmt` must be specified
    and must be scriptable. (It will be scripted, not traced.) Secondly,
    `signature` must be specified and take the form `f(args) -> output`. e.g.

        "f(a, b, c) -> d"
        "f(x) -> None"

    This is used to build both the model and invocation. Note that the return
    is a literal variable, not a type. TorchScript will optimize away
    computation which does not have observable side effects, so some functions
    need to return a result to actually benchmark the task of interest.

    Example:
    ```
    GroupedTimerArgs(
        setup=Setup.GENERIC,  # Creates a float Tensor `x`
        py_stmt="y = x + x.t()",
        cpp_stmt="auto y = x + x.t();",

        # Optional. If present, we can make a TorchScript function as well.
        signature="f(x) -> y",
    )
    ```

    GroupedTimerArgs will ultimately be parsed down to one or more
    WorkerTimerArgs for evaluation.
    """
    py_stmt: Optional[str] = None
    cpp_stmt: Optional[str] = None
    setup: Setup = Setup.NONE
    global_setup: Optional[str] = None
    signature: Optional[str] = None
    num_threads: Union[int, Tuple[int, ...]] = 1
    cost: CostEstimate = CostEstimate.AUTO

    def __post_init__(self) -> None:
        # This is done purely to improve readability.
        if self.py_stmt is not None:
            object.__setattr__(self, "py_stmt", textwrap.dedent(self.py_stmt).strip())

        if self.cpp_stmt is not None:
            object.__setattr__(self, "cpp_stmt", textwrap.dedent(self.cpp_stmt).strip())

        if self.py_stmt is None and self.cpp_stmt is None:
            raise ValueError("You must specify at least one of `py_stmt`, `cpp_stmt`")

        # Check that signature is valid.
        self.torchscript_signature

    @property
    def torchscript_signature(self) -> Optional[Tuple[Tuple[str, ...], str]]:
        if self.signature is None:
            return None

        if self.py_stmt is None:
            # `py_stmt` populates the body of the function.
            raise ValueError("signature provided, but `py_stmt` is None.")

        match = re.search(r"^f\((.*)\) -> (.*)$", self.signature)
        if match is None:
            raise ValueError(f"Invalid signature: `{self.signature}`")

        return tuple(match.groups()[0].split(", ")), match.groups()[1].strip()
