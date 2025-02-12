"""Key enums and structs used to handle data flow within the benchmark."""

# mypy: ignore-errors

import dataclasses
import enum
import itertools as it
import re
import textwrap
from typing import Optional, TYPE_CHECKING, Union

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
    """Labels for a TimerArgs instance which are inferred during unpacking."""

    runtime: RuntimeMode
    autograd: AutogradMode
    language: Language

    @property
    def as_dict(self) -> dict[str, str]:
        """Dict representation for CI reporting."""
        return {
            "runtime": self.runtime.value,
            "autograd": self.autograd.value,
            "language": "Python" if self.language == Language.PYTHON else "C++",
        }


@dataclasses.dataclass(frozen=True)
class GroupedSetup:
    py_setup: str = ""
    cpp_setup: str = ""
    global_setup: str = ""

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            assert field.type == str
            value: str = getattr(self, field.name)
            object.__setattr__(self, field.name, textwrap.dedent(value))


@dataclasses.dataclass(frozen=True)
class GroupedBenchmark:
    """Base class for defining groups of benchmarks.

    Concrete interfaces:
     - `core.api.GroupedStmts`     (init_from_stmts)
     - `core.api.GroupedModules`   (init_from_model)
     - `core.api.GroupedVariants`  (init_from_variants)

    There are a variety of dimensions along which one might wish to measure
    PyTorch performance:
      - Python, C++
      - Eager, TorchScript
      - Single threaded, multi threaded
      - Training, inference

    It is useful to define them together, both for clear, concise benchmark
    definition and more intelligent post processing and analysis.

    There are also two programming idioms in PyTorch. One is to write free form
    code (so-called "NumPy with gradients"), and the other is to organize code
    using `torch.nn.Module`s. (This is how common neural network layers are
    exposed through the PyTorch API.) To support easy definition two simple
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

    A third method, `init_from_variants`, is provided to define several related
    benchmarks at once.
    """

    # These are the stmts which are actually executed by Timer. In the case of
    # `GroupedStmts` (init_from_stmts) they are passed through from user args.
    # In the case of `GroupedModules` (init_from_model) they are generated
    # using `signature`. (e.g. `f(x, y) -> z` generates `z = model(x, y)`)
    py_fwd_stmt: Optional[str]
    cpp_fwd_stmt: Optional[str]

    # Code block used to define a model. `init_from_stmts` will never populate
    # `cpp_model_setup`, but if TorchScript is requested it will generate
    # `py_model_setup` using `torch.jit.script`.
    py_model_setup: Optional[str]
    cpp_model_setup: Optional[str]

    # True if this benchmark used `init_from_stmts`, otherwise False.
    inferred_model_setup: bool

    # Described above
    setup: GroupedSetup
    signature_args: Optional[tuple[str, ...]]
    signature_output: Optional[str]
    torchscript: bool
    autograd: bool
    num_threads: tuple[int, ...]

    @classmethod
    def init_from_stmts(
        cls,
        py_stmt: Optional[str] = None,
        cpp_stmt: Optional[str] = None,
        # Generic constructor arguments
        setup: GroupedSetup = GroupedSetup(),
        signature: Optional[str] = None,
        torchscript: bool = False,
        autograd: bool = False,
        num_threads: Union[int, tuple[int, ...]] = 1,
    ) -> "GroupedBenchmark":
        """Create a set of benchmarks from free-form statements.

        This method of benchmark definition is analogous to Timer use, where
        we simply execute the provided stmts.
        """
        if py_stmt is not None:
            py_stmt = textwrap.dedent(py_stmt)

        if cpp_stmt is not None:
            cpp_stmt = textwrap.dedent(cpp_stmt)

        signature_args, signature_output = cls._parse_signature(signature)
        py_model_setup = (
            cls._model_from_py_stmt(
                py_stmt=py_stmt,
                signature_args=signature_args,
                signature_output=signature_output,
            )
            if torchscript
            else None
        )

        return cls(
            py_fwd_stmt=py_stmt,
            cpp_fwd_stmt=cpp_stmt,
            py_model_setup=py_model_setup,
            cpp_model_setup=None,
            inferred_model_setup=True,
            setup=setup,
            signature_args=signature_args,
            signature_output=signature_output,
            torchscript=torchscript,
            autograd=autograd,
            num_threads=(num_threads,) if isinstance(num_threads, int) else num_threads,
        )

    @classmethod
    def init_from_model(
        cls,
        py_model_setup: Optional[str] = None,
        cpp_model_setup: Optional[str] = None,
        # Generic constructor arguments
        setup: GroupedSetup = GroupedSetup(),
        signature: Optional[str] = None,
        torchscript: bool = False,
        autograd: bool = False,
        num_threads: Union[int, tuple[int, ...]] = 1,
    ) -> "GroupedBenchmark":
        """Create a set of benchmarks using torch.nn Modules.

        This method of benchmark creation takes setup code, and then calls
        a model rather than a free form block of code. As a result, there are
        two additional requirements compared to `init_from_stmts`:
          - `signature` must be provided.
          - A model (named "model") must be defined, either with `model = ...`
            or `def model(...): ...` in Python or `auto model = ...` in C++.
        """
        signature_args, signature_output = cls._parse_signature(signature)
        if signature_args is None:
            raise ValueError(
                "signature is needed when initializing from model definitions."
            )

        return cls(
            *cls._make_model_invocation(
                signature_args, signature_output, RuntimeMode.EAGER
            ),
            py_model_setup=py_model_setup,
            cpp_model_setup=cpp_model_setup,
            inferred_model_setup=False,
            setup=setup,
            signature_args=signature_args,
            signature_output=signature_output,
            torchscript=torchscript,
            autograd=autograd,
            num_threads=(num_threads,) if isinstance(num_threads, int) else num_threads,
        )

    @classmethod
    def init_from_variants(
        cls,
        py_block: str = "",
        cpp_block: str = "",
        num_threads: Union[int, tuple[int, ...]] = 1,
    ) -> dict[Union[tuple[str, ...], Optional[str]], "GroupedBenchmark"]:
        py_cases, py_setup, py_global_setup = cls._parse_variants(
            py_block, Language.PYTHON
        )
        cpp_cases, cpp_setup, cpp_global_setup = cls._parse_variants(
            cpp_block, Language.CPP
        )

        assert not py_global_setup
        setup = GroupedSetup(
            py_setup=py_setup,
            cpp_setup=cpp_setup,
            global_setup=cpp_global_setup,
        )

        # NB: The key is actually `Tuple[str, ...]`, however MyPy gets confused
        #     and we use the superset `Union[Tuple[str, ...], Optional[str]` to
        #     match the expected signature.
        variants: dict[Union[tuple[str, ...], Optional[str]], GroupedBenchmark] = {}

        seen_labels: set[str] = set()
        for label in it.chain(py_cases.keys(), cpp_cases.keys()):
            if label in seen_labels:
                continue
            seen_labels.add(label)

            py_lines = py_cases.get(label, [])
            cpp_lines = cpp_cases.get(label, [])

            n_lines = max(len(py_lines), len(cpp_lines))
            py_lines += [""] * (n_lines - len(py_lines))
            cpp_lines += [""] * (n_lines - len(cpp_lines))
            lines = [
                (py_stmt, cpp_stmt)
                for py_stmt, cpp_stmt in zip(py_lines, cpp_lines)
                if py_stmt or cpp_stmt
            ]

            for i, (py_stmt, cpp_stmt) in enumerate(lines):
                case = (f"Case: {i:>2}",) if len(lines) > 1 else ()
                variants[(label,) + case] = GroupedBenchmark.init_from_stmts(
                    py_stmt=py_stmt or None,
                    cpp_stmt=cpp_stmt or None,
                    setup=setup,
                    num_threads=num_threads,
                )

        return variants

    def __post_init__(self) -> None:
        if self.autograd and self.signature_output is None:
            raise ValueError(
                "An output variable must be specified when `autograd=True`."
            )

        if self.py_model_setup and "model" not in self.py_model_setup:
            raise ValueError(
                "`py_model_setup` appears to be missing `model` definition."
            )

        if self.cpp_model_setup and "model" not in self.cpp_model_setup:
            raise ValueError(
                "`cpp_model_setup` appears to be missing `model` definition."
            )

    # =========================================================================
    # == String manipulation methods ==========================================
    # =========================================================================

    @staticmethod
    def _parse_signature(
        signature: Optional[str],
    ) -> tuple[Optional[tuple[str, ...]], Optional[str]]:
        if signature is None:
            return None, None

        match = re.search(r"^f\((.*)\) -> (.*)$", signature)
        if match is None:
            raise ValueError(f"Invalid signature: `{signature}`")

        args: tuple[str, ...] = tuple(match.groups()[0].split(", "))
        output: str = match.groups()[1].strip()

        if "," in output:
            raise ValueError(
                f"Multiple return values are not currently allowed: `{output}`"
            )

        if output == "None":
            return args, None

        return args, output

    @staticmethod
    def _model_from_py_stmt(
        py_stmt: Optional[str],
        signature_args: Optional[tuple[str, ...]],
        signature_output: Optional[str],
    ) -> str:
        if py_stmt is None:
            raise ValueError("`py_stmt` must be defined in order to derive a model.")

        if signature_args is None:
            raise ValueError("signature is needed in order to derive a model.")

        return textwrap.dedent(
            f"""\
            def model({", ".join(signature_args)}):
            {{stmt_str}}
                return {signature_output}
        """
        ).format(stmt_str=textwrap.indent(py_stmt, " " * 4))

    @staticmethod
    def _make_model_invocation(
        signature_args: tuple[str, ...],
        signature_output: Optional[str],
        runtime: RuntimeMode,
    ) -> tuple[str, str]:
        py_prefix, cpp_prefix = "", ""
        if signature_output is not None:
            py_prefix = f"{signature_output} = "
            cpp_prefix = f"auto {signature_output} = "

        if runtime == RuntimeMode.EAGER:
            model_name = "model"
            cpp_invocation = (
                f"{cpp_prefix}{model_name}->forward({', '.join(signature_args)});"
            )

        else:
            assert runtime == RuntimeMode.JIT
            model_name = "jit_model"
            cpp_invocation = textwrap.dedent(
                f"""\
                std::vector<torch::jit::IValue> ivalue_inputs({{
                    {", ".join([f"torch::jit::IValue({a})" for a in signature_args])}
                }});
                {cpp_prefix}{model_name}.forward(ivalue_inputs);
            """
            )

        # NB:
        #   In python we invoke __call__, however C++ doesn't have an analogous
        #   method so we invoke `forward` instead. This means that Python
        #   is doing extra work (e.g. checking hooks) compared to C++; however
        #   because this is the default user experience that's acceptable.
        py_invocation = f"{py_prefix}{model_name}({', '.join(signature_args)})"

        return py_invocation, cpp_invocation

    @staticmethod
    def _parse_variants(
        block: str, language: Language
    ) -> tuple[dict[str, list[str]], str, str]:
        block = textwrap.dedent(block).strip()
        comment = "#" if language == Language.PYTHON else "//"
        label_pattern = f"{comment} @(.+)$"
        label = ""

        lines_by_label: dict[str, list[str]] = {"SETUP": [], "GLOBAL_SETUP": []}
        for line in block.splitlines(keepends=False):
            match = re.search(label_pattern, line.strip())
            if match:
                label = match.groups()[0]
                if label.replace(" ", "_").upper() in ("SETUP", "GLOBAL_SETUP"):
                    label = label.replace(" ", "_").upper()
                continue

            lines_by_label.setdefault(label, [])
            if line.startswith(comment):
                line = ""
            lines_by_label[label].append(line)

        setup = "\n".join(lines_by_label.pop("SETUP"))
        global_setup = "\n".join(lines_by_label.pop("GLOBAL_SETUP"))

        return lines_by_label, setup, global_setup


# These are the user facing APIs.
GroupedStmts = GroupedBenchmark.init_from_stmts
GroupedModules = GroupedBenchmark.init_from_model
GroupedVariants = GroupedBenchmark.init_from_variants
