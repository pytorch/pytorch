"""Logic for converting human-readable benchmarks into executable form.

This is mostly string manipulation, with just a bit of importlib magic.
"""
import importlib.abc
import importlib.util
import itertools as it
import os
import re
import textwrap
from typing import List, Optional, Tuple, TYPE_CHECKING
import uuid

import torch

if TYPE_CHECKING:
    # See the note in api.py for why this is necessary.
    from torch.utils.benchmark.utils.timer import Language
else:
    from torch.utils.benchmark import Language

from core.api import AutogradMode, AutoLabels, GroupedBenchmark, RuntimeMode, TimerArgs
from core.types import FlatDefinition, FlatIntermediateDefinition, Label
from core.utils import get_temp_dir


_ALL_MODES = tuple(it.product(
    RuntimeMode,
    AutogradMode,
    Language,
))


def _generate_torchscript_file(model_src: str, name: str) -> Optional[str]:
    """Returns the path a saved model if one can be constructed from `spec`.

    Because TorchScript requires actual source code in order to script a
    model, we can't simply `eval` an appropriate model string. Instead, we
    must write the correct source to a temporary Python file and then import
    the TorchScript model from that temporary file.

    `model_src` must contain `jit_model = ...`, which `materialize` will supply.
    """
    # Double check.
    assert "jit_model = " in model_src, f"Missing jit_model definition:\n{model_src}"

    # `torch.utils.benchmark.Timer` will automatically import torch, so we
    # need to match that convention.
    model_src = f"import torch\n{model_src}"

    model_root = os.path.join(get_temp_dir(), "TorchScript_models")
    os.makedirs(model_root, exist_ok=True)
    module_path = os.path.join(model_root, f"torchscript_{name}.py")
    artifact_path = os.path.join(model_root, f"torchscript_{name}.pt")

    if os.path.exists(module_path):
        # The uuid in `name` should protect against this, but it doesn't hurt
        # to confirm.
        raise ValueError(f"File {module_path} already exists.")

    with open(module_path, "wt") as f:
        f.write(model_src)

    # Import magic to actually load our function.
    module_spec = importlib.util.spec_from_file_location(f"torchscript__{name}", module_path)
    assert module_spec is not None
    module = importlib.util.module_from_spec(module_spec)
    loader = module_spec.loader
    assert loader is not None

    loader.exec_module(module)

    # And again, the type checker has no way of knowing that this line is valid.
    jit_model = module.jit_model  # type: ignore[attr-defined]
    assert isinstance(
        jit_model,
        (torch.jit.ScriptFunction, torch.jit.ScriptModule)
    ), f"Expected ScriptFunction or ScriptModule, got: {type(jit_model)}"
    jit_model.save(artifact_path)

    # Cleanup now that we have the actual serialized model.
    os.remove(module_path)
    return artifact_path


def _get_stmt(
    benchmark: GroupedBenchmark,
    runtime: RuntimeMode,
    autograd: AutogradMode,
    language: Language,
) -> Optional[str]:
    """Specialize a GroupedBenchmark for a particular configuration."""
    is_python = (language == Language.PYTHON)

    # During GroupedBenchmark construction, py_fwd_stmt and cpp_fwd_stmt are
    # set to the eager invocation. So in the RuntimeMode.EAGER case we can
    # simply reuse them. For the RuntimeMode.JIT case, we need to generate
    # an appropriate `jit_model(...)` invocation.
    if runtime == RuntimeMode.EAGER:
        stmts = (benchmark.py_fwd_stmt, benchmark.cpp_fwd_stmt)

    else:
        assert runtime == RuntimeMode.JIT
        assert benchmark.signature_args is not None
        stmts = GroupedBenchmark._make_model_invocation(
            benchmark.signature_args, benchmark.signature_output, RuntimeMode.JIT)

    stmt = stmts[0 if is_python else 1]

    if autograd == AutogradMode.FORWARD_BACKWARD and stmt is not None:
        assert benchmark.signature_output is not None
        backward = (
            f"{benchmark.signature_output}"

            # In C++ we have to get the Tensor out of the IValue to call `.backward()`
            f"{'.toTensor()' if runtime == RuntimeMode.JIT and language == Language.CPP else ''}"
            f".backward(){';' if language == Language.CPP else ''}"
        )
        stmt = f"{stmt}\n{backward}"
    return stmt


def _get_setup(
    benchmark: GroupedBenchmark,
    runtime: RuntimeMode,
    language: Language,
    stmt: str,
    model_path: Optional[str]
) -> str:
    """Specialize a GroupedBenchmark for a particular configuration.

    Setup requires two extra pieces of information:
      1) The benchmark stmt. This is needed to warm up the model and avoid
         measuring lazy initialization.
      2) The model path so we can load it during the benchmark.

    These are only used when `runtime == RuntimeMode.JIT`.
    """

    # By the time we get here, details about how to set up a model have already
    # been determined by GroupedBenchmark. (Or set to None if appropriate.) We
    # simply need to collect and package the code blocks.
    if language == Language.PYTHON:
        setup = benchmark.setup.py_setup
        model_setup = benchmark.py_model_setup
    else:
        assert language == Language.CPP
        setup = benchmark.setup.cpp_setup
        model_setup = benchmark.cpp_model_setup

    if runtime == RuntimeMode.EAGER:
        return "\n".join([setup, model_setup or ""])

    assert runtime == RuntimeMode.JIT
    assert model_path is not None

    # We template `"{model_path}"`, so quotes would break model loading. The
    # model path is generated within the benchmark, so this is just an
    # abundance of caution rather than something that is expected in practice.
    assert '"' not in model_path

    # `stmt` may contain newlines, so we can't use f-strings. Instead we need
    # to generate templates so that dedent works properly.
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

    model_load = setup_template.format(stmt=textwrap.indent(stmt, ' ' * 4))
    return "\n".join([setup, model_load])


def materialize(benchmarks: FlatIntermediateDefinition) -> FlatDefinition:
    """Convert a heterogeneous benchmark into an executable state.

    This entails generation of TorchScript model artifacts, splitting
    GroupedBenchmarks into multiple TimerArgs, and tagging the results with
    AutoLabels.
    """
    results: List[Tuple[Label, AutoLabels, TimerArgs]] = []

    for label, args in benchmarks.items():
        if isinstance(args, TimerArgs):
            # User provided an explicit TimerArgs, so no processing is necessary.
            auto_labels = AutoLabels(
                RuntimeMode.EXPLICIT,
                AutogradMode.EXPLICIT,
                args.language
            )
            results.append((label, auto_labels, args))

        else:
            assert isinstance(args, GroupedBenchmark)

            model_path: Optional[str] = None
            if args.py_model_setup and args.torchscript:
                model_setup = f"{args.py_model_setup}\njit_model = torch.jit.script(model)"

                # This is just for debugging. We just need a unique name for the
                # model, but embedding the label makes debugging easier.
                name: str = re.sub(r'[^a-z0-9_]', '_', '_'.join(label).lower())
                name = f"{name}_{uuid.uuid4()}"

                model_path = _generate_torchscript_file(model_setup, name=name)

            for (runtime, autograd, language), num_threads in it.product(_ALL_MODES, args.num_threads):
                if runtime == RuntimeMode.EXPLICIT or autograd == AutogradMode.EXPLICIT:
                    continue

                if runtime == RuntimeMode.JIT and not args.torchscript:
                    continue

                if autograd == AutogradMode.FORWARD_BACKWARD and not args.autograd:
                    continue

                stmt = _get_stmt(args, runtime, autograd, language)
                if stmt is None:
                    continue

                setup = _get_setup(args, runtime, language, stmt, model_path)

                global_setup: str = ""
                if language == Language.CPP and runtime == RuntimeMode.JIT:
                    global_setup = textwrap.dedent("""
                        #include <string>
                        #include <vector>
                        #include <torch/script.h>
                    """)

                autolabels = AutoLabels(runtime, autograd, language)
                timer_args = TimerArgs(
                    stmt=stmt,
                    setup=setup,
                    global_setup=global_setup,
                    num_threads=num_threads,
                    language=language,
                )

                results.append((label, autolabels, timer_args))

    return tuple(results)
