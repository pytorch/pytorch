"""Key enums and structs used to handle data flow within the benchmark."""
import abc
import dataclasses
import enum
import textwrap
from typing import Optional, Tuple, Union, TYPE_CHECKING

from worker.main import WorkerTimerArgs

if TYPE_CHECKING:
    # Benchmark utils are only partially strict compliant, so MyPy won't follow
    # imports using the public namespace. (Due to an exclusion rule in
    # mypy-strict.ini)
    from torch.utils.benchmark.utils.timer import Language
else:
    from torch.utils.benchmark import Language


# Note:
#   WorkerTimerArgs is defined in worker.main so that the worker does not
#   depend on any files, including core.api. We mirror it with a public symbol
#   `TimerArgs` for API consistency.
TimerArgs = WorkerTimerArgs


class RuntimeMode(enum.Enum):
    EAGER = "Eager"
    JIT = "TorchScript"
    EXPLICIT = ""


class AutogradMode(enum.Enum):
    FORWARD = "Forward"
    FORWARD_BACKWARD = "Forward + Backward"
    EXPLICIT = ""


@dataclasses.dataclass(frozen=True)
class AutoLabels:
    runtime: RuntimeMode
    autograd: AutogradMode
    language: Language


@dataclasses.dataclass(frozen=True)
class GroupedSetup:
    py_setup: Optional[str] = None
    cpp_setup: Optional[str] = None
    global_setup: Optional[str] = None

    def __post_init__(self) -> None:
        # dedent all populated entries.
        for field in dataclasses.fields(self):
            assert field.type == Optional[str]
            value: Optional[str] = getattr(self, field.name)
            if value is not None:
                object.__setattr__(self, field.name, textwrap.dedent(value))


class GroupedBenchmark(abc.ABC):
    """Base class for defining groups of benchmarks.

    Implementation: `core.api_impl._GroupedBenchmarkImpl`
    Concrete interfaces:
     - `core.api_impl.GroupedStmts`     (init_from_stmts)
     - `core.api_impl.GroupedModules`   (init_from_model)

    There are a variety of dimensions along which one might wish to measure
    PyTorch performance:
      - Python, C++
      - Eager, TorchScript
      - Single threaded, multi threaded
      - Training, inference

    It is useful to define them together, both for clear and concise benchmark
    definition and more intelligent post processing and analysis.

    We may, of course, only be interested in a subset of cases. For instance we
    may be benchmarking Python code which does not have a C++ analog, or a
    statement which is not TorchScript-able.

    There are also two programming idioms in PyTorch. One is to write free form
    code (so called "NumPy with gradients"), and the other is to organize code
    using `torch.nn.Module`s. (This is how common neural network layers are
    exposed through the PyTorch API.) To support easy definition two
    initialization methods are provided:
     - `init_from_stmts`
     - `init_from_model`

    Those methods will document their unique constructor arguments, however
    most are shared and are defined here:

        setup: Defines how to initialize a benchmark in both Python and C++.
        signature:
            A string of the form:
            ```
                f(a, b, ...) -> c
            ```

            For instance, if Python setup is:
            ```
                x = torch.ones((2,), requires_grad=True)
                y = torch.ones((2,))
            ```
            and the corresponding stmt is:
            ```
                z = torch.dot(x, y)
            ```
            Then the signature is `f(x, y) -> z`. `signature` is required any
            time we need to generate part of a snippet:
             - When calling an opaque model provided by `init_from_models`
             - When `torchscript=True`
             - When `autograd=True`

            If a return value is not needed (e.g. because of in place mutation)
            then `-> None` is valid, but a non-None return must be provided if
            `autograd=True`

        torchscript:
            If True, also JIT the stmt or model and generate benchmarks which
            call the scripted version. Requires that `signature` is defined.

        autograd:
            If True, generate both forward and forward + backward benchmarks.
            Requires that `signature` is defined, and return value is not None.

        num_threads:
            Maps to the Timer arg. If a tuple of ints is provided, benchmarks
            will be generated for each value.
    """

    @staticmethod
    @abc.abstractmethod
    def init_from_stmts(
        py_stmt: Optional[str] = None,
        cpp_stmt: Optional[str] = None,

        # Generic constructor arguments
        setup: GroupedSetup = GroupedSetup(),
        signature: Optional[str] = None,
        torchscript: bool = False,
        autograd: bool = False,
        num_threads: Union[int, Tuple[int, ...]] = 1,
    ) -> "GroupedBenchmark":
        ...

    @staticmethod
    @abc.abstractmethod
    def init_from_model(
        py_model_setup: Optional[str] = None,
        cpp_model_setup: Optional[str] = None,

        # Generic constructor arguments
        setup: GroupedSetup = GroupedSetup(),
        signature: Optional[str] = None,
        torchscript: bool = False,
        autograd: bool = False,
        num_threads: Union[int, Tuple[int, ...]] = 1,
    ) -> "GroupedBenchmark":
        ...

    @abc.abstractproperty
    def ts_model_setup(self) -> Optional[str]:
        ...

    @abc.abstractmethod
    def flatten(
        self,
        model_path: Optional[str]
    ) -> Tuple[Tuple[AutoLabels, TimerArgs], ...]:
        ...
