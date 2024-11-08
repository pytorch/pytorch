# mypy: allow-untyped-defs

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch._inductor.config
import torch.fx

if TYPE_CHECKING:
    from torch._inductor.utils import InputType


__all__ = [
    "compile",
    "list_mode_options",
    "list_options",
    "cudagraph_mark_step_begin",
]


def compile(
    gm: torch.fx.GraphModule,
    example_inputs: List["InputType"],
    options: Optional[Dict[str, Any]] = None,
):
    """
    Compile a given FX graph with TorchInductor.  This allows compiling
    FX graphs captured without using TorchDynamo.

    Args:
        gm: The FX graph to compile.
        example_inputs:  List of tensor inputs.
        options:  Optional dict of config options.  See `torch._inductor.config`.

    Returns:
        Callable with same behavior as gm but faster.
    """
    from .compile_fx import compile_fx

    return compile_fx(gm, example_inputs, config_patches=options)


def aoti_compile_and_package(
    exported_program,
    args: Tuple[Any],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    package_path: Optional[str] = None,
    inductor_configs: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Compiles the exported program with AOTInductor, and packages it into a .pt2
    artifact specified by the input package_path. To load the package, you can
    call `torch._inductor.aoti_load_package(package_path)`.

    To compile and save multiple models into a single .pt2 artifact, you can do
    the following:
    ```
    ep1 = torch.export.export(M1(), ...)
    aoti_file1 = torch._inductor.aot_compile(ep1, ...)
    ep2 = torch.export.export(M2(), ...)
    aoti_file2 = torch._inductor.aot_compile(ep2, ...)

    from torch._inductor.package import package_aoti, load_package
    package_aoti("my_package.pt2", {"model1": aoti_file1, "model2": aoti_file2})

    compiled_model1 = load_package("my_package.pt2", "model1")
    compiled_model2 = load_package("my_package.pt2", "model2")
    ```

    Args:
        exported_program: An exported program created through a call from torch.export
        args: Example positional inputs
        kwargs: Optional example keyword inputs
        package_path: Optional specified path to the generated .pt2 artifact.
        inductor_configs: Optional dictionary of configs to control inductor.

    Returns:
        Path to the generated artifact
    """
    from torch.export import ExportedProgram

    from .compile_fx import _flatten_inputs
    from .debug import aot_inductor_minifier_wrapper

    if not isinstance(exported_program, ExportedProgram):
        raise ValueError("Only ExportedProgram is supported")

    assert package_path is None or package_path.endswith(
        ".pt2"
    ), f"Expect package path to end with .pt2, got {package_path}"

    inductor_configs = inductor_configs or {}
    inductor_configs["aot_inductor.package"] = True

    if inductor_configs.get("aot_inductor.output_path"):
        raise RuntimeError(
            "Please pass in a package path to aot_inductor_compile() instead "
            "of setting the aot_inductor.output_path config."
        )

    gm = exported_program.module()

    flat_example_inputs, options = _flatten_inputs(
        gm, args, kwargs, options=inductor_configs
    )

    # a wrapper around aoti_compile_and_package_inner.
    return aot_inductor_minifier_wrapper(
        _aoti_compile_and_package_inner,
        exported_program,
        gm,
        flat_example_inputs,
        package_path=package_path,
        inductor_configs=options,
    )


def _aoti_compile_and_package_inner(
    gm: torch.fx.GraphModule,
    flat_example_inputs: Tuple[Any],
    *,
    inductor_configs: Dict[str, Any],
    load_and_run: bool = False,
    package_path: Optional[str] = None,
):
    """
    See docstring for aoti_compile_and_package.

    `inductor_configs` should contain the serialized input and output specs.

    If `load_and_run` is True, this function will load the compiled model and run it.
    This is for the minifier to check the correctness of the compiled model.
    """
    from .compile_fx import compile_fx_aot
    from .package import package_aoti

    aoti_files = compile_fx_aot(
        gm,
        flat_example_inputs,  # type: ignore[arg-type]
        config_patches=inductor_configs,
    )

    if package_path is None:
        package_path = aoti_files + ".pt2"

    res = package_aoti(package_path, aoti_files)
    assert res == package_path

    if load_and_run:
        compiled_model = aoti_load_package(package_path)
        aoti_result = compiled_model(*flat_example_inputs)
    return package_path


def aoti_load_package(path: str) -> Any:  # type: ignore[type-arg]
    """
    Loads the model from the PT2 package.

    If multiple models were packaged into the PT2, this will load the default
    model. To load a specific model, you can directly call the load API
    ```
    from torch._inductor.package import load_package

    compiled_model1 = load_package("my_package.pt2", "model1")
    compiled_model2 = load_package("my_package.pt2", "model2")
    ```

    Args:
        path: Path to the .pt2 package
    """
    from torch._inductor.package import load_package

    return load_package(path)


def aot_compile(
    gm: torch.fx.GraphModule,
    args: Tuple[Any],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Ahead-of-time compile a given FX graph with TorchInductor into a shared library.

    Args:
        gm: The FX graph to compile.
        args:  Example arguments
        kwargs: Example keyword arguments
        options:  Optional dict of config options.  See `torch._inductor.config`.

    Returns:
        Path to the generated shared library
    """
    from .compile_fx import _flatten_inputs, compile_fx_aot

    flat_example_inputs, options = _flatten_inputs(gm, args, kwargs, options=options)

    return compile_fx_aot(
        gm,
        flat_example_inputs,  # type: ignore[arg-type]
        config_patches=options,
    )


def list_mode_options(
    mode: Optional[str] = None, dynamic: Optional[bool] = None
) -> Dict[str, Any]:
    r"""Returns a dictionary describing the optimizations that each of the available
    modes passed to `torch.compile()` performs.

    Args:
        mode (str, optional): The mode to return the optimizations for.
        If None, returns optimizations for all modes
        dynamic (bool, optional): Whether dynamic shape is enabled.

    Example::
        >>> torch._inductor.list_mode_options()
    """

    mode_options: Dict[str, Dict[str, bool]] = {
        "default": {},
        # enable cudagraphs
        "reduce-overhead": {
            "triton.cudagraphs": True,
        },
        # enable max-autotune
        "max-autotune-no-cudagraphs": {
            "max_autotune": True,
            "coordinate_descent_tuning": True,
        },
        # enable max-autotune
        # enable cudagraphs
        "max-autotune": {
            "max_autotune": True,
            "triton.cudagraphs": True,
            "coordinate_descent_tuning": True,
        },
    }
    return mode_options[mode] if mode else mode_options  # type: ignore[return-value]


def list_options() -> List[str]:
    r"""Returns a dictionary describing the optimizations and debug configurations
    that are available to `torch.compile()`.

    The options are documented in `torch._inductor.config`.

    Example::

        >>> torch._inductor.list_options()
    """

    from torch._inductor import config

    current_config: Dict[str, Any] = config.get_config_copy()

    return list(current_config.keys())


def cudagraph_mark_step_begin():
    "Indicates that a new iteration of inference or training is about to begin."
    from .cudagraph_trees import mark_step_begin

    mark_step_begin()
