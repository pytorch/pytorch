"""Constructs TorchScript model and corresponding benchmark invocation."""
import importlib.abc
import importlib.util
import os
import textwrap
from typing import cast, Optional, Tuple, Union, TYPE_CHECKING

import torch

from core.api import GroupedTimerArgs, TimerArgs
from core.utils import get_temp_dir
from worker.main import CostEstimate


if TYPE_CHECKING:
    # See core.api for an explanation.
    from torch.utils.benchmark.utils.timer import Language
else:
    from torch.utils.benchmark import Language


# The templates cannot indent as this would only indent the first line, so
# functions invoking the template need to indent setup and stmt themselves.
_INDENTATION = 4


_MODEL_DEFINITION_TEMPLATE = """\
    import torch

    @torch.jit.script
    def f({args_str}):
    {stmt_str}
        return {return_str}
"""

def _define_model(spec: GroupedTimerArgs) -> Optional[str]:
    """Construct source for TorchScript model if sufficient information is present."""
    py_stmt = spec.py_stmt
    signature = spec.torchscript_signature
    if signature is None:
        return None

    assert py_stmt is not None  # Narrow type. GroupedTimerArgs checks this.
    torchscript_args, torchscript_return = signature

    return textwrap.dedent(_MODEL_DEFINITION_TEMPLATE).strip().format(
        args_str=', '.join(torchscript_args),
        stmt_str=textwrap.indent(py_stmt, ' ' * _INDENTATION),
        return_str=torchscript_return or ''
    )


def generate_torchscript_file(
    spec: GroupedTimerArgs,
    name: str,
) -> Optional[str]:
    """Returns the path a saved model if one can be constructed from `spec`.

    Because TorchScript requires actual source code in order to script a
    model, we can't simply `eval` an appropriate model string. Instead, we
    must write the correct source to a temporary Python file and then import
    the TorchScript model from that temporary file.
    """
    model_src: Optional[str] = _define_model(spec)
    if model_src is None:
        return None

    model_root = os.path.join(get_temp_dir(), "TorchScript_models")
    os.makedirs(model_root, exist_ok=True)
    module_path = os.path.join(model_root, f"ts_{name}.py")
    artifact_path = os.path.join(model_root, f"ts_{name}.pt")

    if os.path.exists(module_path):
        # This will only happen if two benchmarks have very similar names and
        # have labels which map to the same name. This is not expected to
        # happen in practice.
        raise ValueError(f"File {module_path} already exists.")

    with open(module_path, "wt") as f:
        f.write(model_src)

    # Import magic to actually load our function.
    module_spec = importlib.util.spec_from_file_location(f"ts_{name}", module_path)
    module = importlib.util.module_from_spec(module_spec)
    loader = module_spec.loader
    assert loader is not None

    # Module.loader has type Optional[_Loader]. Even when we assert loader is
    # not None and MyPy narrows it to type _Loader, it still can't rely on
    # methods. So we have to use a cast to tell MyPy that _Loader implements
    # importlib.abc.Loader.
    cast(importlib.abc.Loader, loader).exec_module(module)

    f = module.f  # type: ignore
    assert isinstance(f, torch.jit.ScriptFunction)
    f.save(artifact_path)

    # Cleanup now that we have the actual serialized model.
    os.remove(module_path)
    return artifact_path


def construct_model_invocation(
    model_path: str,
    arguments: Tuple[str, ...],
    setup: str,
    global_setup: Optional[str],
    language: Language,
    num_threads: Union[int, Tuple[int, ...]],
    cost: CostEstimate,
) -> TimerArgs:
    # We template `"{model_path}"`, so quotes would break model loading. The
    # model path is generated within the benchmark, so this is just an
    # abundance of caution rather than something that is expected in practice.
    assert '"' not in model_path
    if language == Language.PYTHON:
        assert global_setup is None
        setup_template: str = textwrap.dedent("""
            {base_setup}
            f = torch.jit.load("{model_path}")

            # Warmup `f`
            for _ in range(3):
            {f_invocation}
        """)

        f_invocation: str = f"f({', '.join(arguments)})"

    else:
        assert language == Language.CPP
        global_setup = textwrap.dedent("""
            #include <string>
            #include <vector>
            #include <torch/script.h>
        """) + textwrap.dedent(global_setup or "")

        setup_template = textwrap.dedent("""
            {base_setup}

            const std::string fpath = "{model_path}";
            auto f = torch::jit::load(fpath);

            // Warmup `f`
            for (int i = 0; i < 3; i++) {{
            {f_invocation}
            }}
        """)

        f_invocation = textwrap.dedent(f"""
            std::vector<torch::jit::IValue> inputs({{
                {', '.join([f'torch::jit::IValue({i})' for i in arguments])}
            }});
            f.forward(inputs);
        """)

    return TimerArgs(
        stmt=f_invocation,
        setup=setup_template.format(
            base_setup=setup,
            model_path=model_path,
            f_invocation=textwrap.indent(f_invocation, " " * _INDENTATION)),
        global_setup=global_setup,
        num_threads=num_threads,
        language=language,
        cost=cost,
    )
