import dataclasses
import itertools as it
import re
import textwrap
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

from core.api import (
    AutogradMode, AutoLabels, GroupedBenchmark, GroupedSetup, RuntimeMode,
    TimerArgs
)

if TYPE_CHECKING:
    # See core.api for an explanation.
    from torch.utils.benchmark.utils.timer import Language
else:
    from torch.utils.benchmark import Language


def _parse_signature(
    signature: Optional[str]
) -> Tuple[Optional[Tuple[str, ...]], Optional[str]]:
    if signature is None:
        return None, None

    match = re.search(r"^f\((.*)\) -> (.*)$", signature)
    if match is None:
        raise ValueError(f"Invalid signature: `{signature}`")

    args: Tuple[str, ...] = tuple(match.groups()[0].split(", "))
    output: str = match.groups()[1].strip()

    if "," in output:
        raise ValueError(f"Multiple return values are not currently allowed: `{output}`")

    if output == "None":
        return args, None

    return args, output


def _model_from_py_stmt(
    py_stmt: Optional[str],
    signature_args: Optional[Tuple[str, ...]],
    signature_output: Optional[str],
) -> str:
    if py_stmt is None:
        raise ValueError("`py_stmt` must be defined in order to derive a model.")

    if signature_args is None:
        raise ValueError("signature is needed in order to derive a model.")

    return textwrap.dedent(f"""\
        def model({', '.join(signature_args)}):
        {{stmt_str}}
            return {signature_output}
    """).format(stmt_str=textwrap.indent(py_stmt, ' ' * 4))


def _make_invocation(
    model_name: str,
    signature_args: Tuple[str, ...],
    signature_output: Optional[str],
    runtime: RuntimeMode,
) -> Tuple[str, str]:
    py_prefix, cpp_prefix = "", ""
    if signature_output is not None:
        py_prefix = f"{signature_output} = "
        cpp_prefix = f"auto {signature_output} = "

    if runtime == RuntimeMode.EAGER:
        cpp_invocation = f"{cpp_prefix}{model_name}->forward({', '.join(signature_args)});"
    else:
        assert runtime == RuntimeMode.JIT
        cpp_invocation = textwrap.dedent(f"""\
            std::vector<torch::jit::IValue> ivalue_inputs({{
                {', '.join([f'torch::jit::IValue({a})' for a in signature_args])}
            }});
            {cpp_prefix}{model_name}.forward(ivalue_inputs);
        """)

    # NB:
    #   In python we invoke __call__, however C++ doesn't have an analogous
    #   method so we invoke `forward` instead. This means that that Python
    #   is doing extra work (e.g. checking hooks) compared to C++; however
    #   because this is the default user experience that's acceptable.
    return (
        f"{py_prefix}{model_name}({', '.join(signature_args)})",
        cpp_invocation,
    )


@dataclasses.dataclass(frozen=True)
class _GroupedBenchmarkImpl(GroupedBenchmark):
    py_fwd_stmt: Optional[str]
    cpp_fwd_stmt: Optional[str]

    py_model_setup: Optional[str]
    cpp_model_setup: Optional[str]
    inferred_model_setup: bool
    setup: GroupedSetup

    signature_args: Optional[Tuple[str, ...]]
    signature_output: Optional[str]

    torchscript: bool
    autograd: bool
    num_threads: Tuple[int, ...]

    @staticmethod
    def init_from_stmts(
        py_stmt: Optional[str] = None,
        cpp_stmt: Optional[str] = None,
        setup: GroupedSetup = GroupedSetup(),
        signature: Optional[str] = None,
        torchscript: bool = False,
        autograd: bool = False,
        num_threads: Union[int, Tuple[int, ...]] = 1,
    ) -> "_GroupedBenchmarkImpl":
        if py_stmt is not None:
            py_stmt = textwrap.dedent(py_stmt)

        if cpp_stmt is not None:
            cpp_stmt = textwrap.dedent(cpp_stmt)

        signature_args, signature_output = _parse_signature(signature)
        py_model_setup = (
            _model_from_py_stmt(
                py_stmt=py_stmt,
                signature_args=signature_args,
                signature_output=signature_output
            ) if torchscript else None
        )

        return _GroupedBenchmarkImpl(
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

    @staticmethod
    def init_from_model(
        py_model_setup: Optional[str] = None,
        cpp_model_setup: Optional[str] = None,
        setup: GroupedSetup = GroupedSetup(),
        signature: Optional[str] = None,
        torchscript: bool = False,
        autograd: bool = False,
        num_threads: Union[int, Tuple[int, ...]] = 1,
    ) -> "_GroupedBenchmarkImpl":
        signature_args, signature_output = _parse_signature(signature)
        if signature_args is None:
            raise ValueError("signature is needed when initializing from model definitions.")

        return _GroupedBenchmarkImpl(
            *_make_invocation("model", signature_args, signature_output, RuntimeMode.EAGER),
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

    def __post_init__(self) -> None:
        if self.autograd and self.signature_output is None:
            raise ValueError("An output variable must be specified when `autograd=True`.")

        if not self.inferred_model_setup:
            if self.py_model_setup and not self.py_model_setup.startswith("model = "):
                raise ValueError("`py_model_setup` must have the form `model = ...`")

            if self.cpp_model_setup and not self.cpp_model_setup.startswith("auto model = "):
                raise ValueError("`cpp_model_setup` must have the form `auto model = ...`")

    @property
    def ts_model_setup(self) -> Optional[str]:
        if self.py_model_setup is None or self.torchscript is False:
            return None

        return f"{self.py_model_setup}\njit_model = torch.jit.script(model)"

    def flatten(
        self,
        model_path: Optional[str]
    ) -> Tuple[Tuple[AutoLabels, TimerArgs], ...]:
        output: List[Tuple[AutoLabels, TimerArgs]] = []
        mode_iter = it.product(
            RuntimeMode,
            AutogradMode,
            Language,
            self.num_threads,
        )
        for runtime, autograd, language, num_threads in mode_iter:
            args = self._get_timer_args(runtime, autograd, language, model_path)
            if args is None:
                continue

            stmt, setup, global_setup = args
            timer_args = TimerArgs(
                stmt=stmt,
                setup=setup,
                global_setup=global_setup,
                num_threads=num_threads,
                language=language,
            )
            output.append((AutoLabels(runtime, autograd, language), timer_args))
        return tuple(output)

    def _get_timer_args(
        self,
        runtime: RuntimeMode,
        autograd: AutogradMode,
        language: Language,
        model_path: Optional[str],
    ) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
        # Returns: stmt, setup, global_setup

        if runtime == RuntimeMode.EXPLICIT or autograd == AutogradMode.EXPLICIT:
            return None

        if runtime == RuntimeMode.JIT and not self.torchscript:
            return None

        if autograd == AutogradMode.FORWARD_BACKWARD and not self.autograd:
            return None

        stmt = self._get_stmt(runtime, autograd, language)
        if stmt is None:
            return None

        setup = self._get_setup(runtime, language, stmt, model_path)

        global_setup: Optional[str] = None
        if language == Language.CPP and runtime == RuntimeMode.JIT:
            global_setup = textwrap.dedent("""
                #include <string>
                #include <vector>
                #include <torch/script.h>
            """)

        return stmt, setup, global_setup

    def _get_stmt(
        self,
        runtime: RuntimeMode,
        autograd: AutogradMode,
        language: Language,
    ) -> Optional[str]:
        is_python = (language == Language.PYTHON)
        if runtime == RuntimeMode.EAGER:
            stmt = (self.py_fwd_stmt if is_python else self.cpp_fwd_stmt)

        else:
            assert runtime == RuntimeMode.JIT
            assert self.signature_args is not None
            py_stmt, cpp_stmt = _make_invocation(
                "jit_model", self.signature_args, self.signature_output, runtime)
            stmt = (py_stmt if is_python else cpp_stmt)

        if autograd == AutogradMode.FORWARD_BACKWARD and stmt is not None:
            assert self.signature_output is not None
            backward = (
                f"{self.signature_output}"
                f"{'.toTensor()' if runtime == RuntimeMode.JIT and language == Language.CPP else ''}"
                f".backward(){';' if language == Language.CPP else ''}"
            )
            stmt = f"{stmt}\n{backward}"
        return stmt

    def _get_setup(
        self,
        runtime: RuntimeMode,
        language: Language,
        stmt: str,
        model_path: Optional[str]
    ) -> Optional[str]:
        if language == Language.PYTHON:
            setup = self.setup.py_setup
            model_setup = self.py_model_setup
        else:
            assert language == Language.CPP
            setup = self.setup.cpp_setup
            model_setup = self.cpp_model_setup

        if runtime == RuntimeMode.EAGER:
            # If benchmark was defined using `init_from_stmts`, `setup` is
            # sufficient. If `init_from_models` was used, however, we need
            # to define `model` for the eager path to work.
            return (
                setup if self.inferred_model_setup
                else (f"{setup}\n" if setup else "") + (model_setup or "")
            )

        assert runtime == RuntimeMode.JIT
        assert model_path is not None

        # We template `"{model_path}"`, so quotes would break model loading. The
        # model path is generated within the benchmark, so this is just an
        # abundance of caution rather than something that is expected in practice.
        assert '"' not in model_path

        if language == Language.PYTHON:
            setup_template: str = textwrap.dedent(f"""
                jit_model = torch.jit.load("{model_path}")

                # Warmup `jit_model`
                for _ in range(3):
                {{stmt}}
            """)

        else:
            assert language == Language.CPP
            setup_template = textwrap.dedent(f"""
                const std::string fpath = "{model_path}";
                auto jit_model = torch::jit::load(fpath);

                // Warmup `jit_model`
                for (int i = 0; i < 3; i++) {{{{
                {{stmt}}
                }}}}
            """)
        return (
            (f"{setup}\n" if setup else "") +
            setup_template.format(stmt=textwrap.indent(stmt, ' ' * 4))
        )


GroupedStmts = _GroupedBenchmarkImpl.init_from_stmts
GroupedModules = _GroupedBenchmarkImpl.init_from_model
