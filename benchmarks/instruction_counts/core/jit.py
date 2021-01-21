"""Constructs TorchScript model artifact."""
import importlib.abc
import importlib.util
import os
from typing import cast, Optional

import torch


def generate_torchscript_file(
    model_src: str,
    name: str,
    temp_dir: str,
) -> Optional[str]:
    """Returns the path a saved model if one can be constructed from `spec`.

    Because TorchScript requires actual source code in order to script a
    model, we can't simply `eval` an appropriate model string. Instead, we
    must write the correct source to a temporary Python file and then import
    the TorchScript model from that temporary file.

    `model_src` must contain `jit_model = ...`, which
    `core.api_impl._GroupedBenchmarkImpl` will supply.
    """
    # Double check.
    assert "jit_model = " in model_src, model_src
    model_src = f"import torch\n{model_src}"

    model_root = os.path.join(temp_dir, "TorchScript_models")
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

    jit_model = module.jit_model  # type: ignore
    assert isinstance(
        jit_model,
        (torch.jit.ScriptFunction, torch.jit.ScriptModule)
    ), f"{type(jit_model)}"
    jit_model.save(artifact_path)

    # Cleanup now that we have the actual serialized model.
    os.remove(module_path)
    return artifact_path
