# mypy: allow-untyped-defs
from __future__ import annotations

import io
import logging
import os
from typing import Any, IO, Optional, TYPE_CHECKING, Union

import torch._inductor.config
import torch.fx


if TYPE_CHECKING:
    from torch._inductor.utils import InputType
    from torch.export import ExportedProgram
    from torch.types import FileLike

__all__ = [
    "compile",
    "list_mode_options",
    "list_options",
    "cudagraph_mark_step_begin",
]


log = logging.getLogger(__name__)


def compile(
    gm: torch.fx.GraphModule,
    example_inputs: list[InputType],
    options: Optional[dict[str, Any]] = None,
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
    exported_program: ExportedProgram,
    _deprecated_unused_args=None,
    _deprecated_unused_kwargs=None,
    *,
    package_path: Optional[FileLike] = None,
    inductor_configs: Optional[dict[str, Any]] = None,
) -> str:
    """
    Compiles the exported program with AOTInductor, and packages it into a .pt2
    artifact specified by the input package_path. To load the package, you can
    call ``torch._inductor.aoti_load_package(package_path)``.

    An example usage is as follows:

    .. code-block:: python

        ep = torch.export.export(M(), ...)
        aoti_file = torch._inductor.aoti_compile_and_package(
            ep, package_path="my_package.pt2"
        )
        compiled_model = torch._inductor.aoti_load_package("my_package.pt2")

    To compile and save multiple models into a single ``.pt2`` artifact, you can do
    the following:

    .. code-block:: python

        ep1 = torch.export.export(M1(), ...)
        aoti_file1 = torch._inductor.aot_compile(
            ep1, ..., options={"aot_inductor.package": True}
        )
        ep2 = torch.export.export(M2(), ...)
        aoti_file2 = torch._inductor.aot_compile(
            ep2, ..., options={"aot_inductor.package": True}
        )

        from torch._inductor.package import package_aoti, load_package

        package_aoti("my_package.pt2", {"model1": aoti_file1, "model2": aoti_file2})

        compiled_model1 = load_package("my_package.pt2", "model1")
        compiled_model2 = load_package("my_package.pt2", "model2")

    Args:
        exported_program: An exported program created through a call from torch.export
        package_path: Optional specified path to the generated .pt2 artifact.
        inductor_configs: Optional dictionary of configs to control inductor.

    Returns:
        Path to the generated artifact
    """
    from torch.export import ExportedProgram

    from .debug import aot_inductor_minifier_wrapper

    if not isinstance(exported_program, ExportedProgram):
        raise ValueError("Only ExportedProgram is supported")

    if exported_program.example_inputs is None:
        raise RuntimeError(
            "exported_program.example_inputs is required to be set in order "
            "for AOTInductor compilation."
        )

    if _deprecated_unused_args is not None or _deprecated_unused_kwargs is not None:
        log.warning(
            "You no longer need to specify args/kwargs to aoti_compile_and_package "
            "as we can get this information from exported_program.example_inputs."
        )

    assert (
        package_path is None
        or (
            isinstance(package_path, (io.IOBase, IO))
            and package_path.writable()
            and package_path.seekable()
        )
        or (
            isinstance(package_path, (str, os.PathLike))
            and os.fspath(package_path).endswith(".pt2")
        )
    ), (
        f"Expect package path to be a file ending in .pt2, is None, or is a buffer. Instead got {package_path}"
    )

    inductor_configs = inductor_configs or {}
    inductor_configs["aot_inductor.package"] = True

    if inductor_configs.get("aot_inductor.output_path"):
        raise RuntimeError(
            "Please pass in a package path to aot_inductor_compile() instead "
            "of setting the aot_inductor.output_path config."
        )

    # a wrapper around aoti_compile_and_package_inner.
    return aot_inductor_minifier_wrapper(
        _aoti_compile_and_package_inner,
        exported_program,
        package_path=package_path,
        inductor_configs=inductor_configs,
    )


def _aoti_compile_and_package_inner(
    gm: torch.nn.Module,
    # flat_example_inputs: List[Any],
    args: tuple[Any],
    kwargs: Optional[dict[str, Any]] = None,
    *,
    load_and_run: bool = False,
    check_accuracy: Optional[str] = None,
    package_path: Optional[Union[str, io.BytesIO]] = None,
    inductor_configs: Optional[dict[str, Any]] = None,
):
    """
    See docstring for aoti_compile_and_package.

    If `load_and_run` is True, this function will load the compiled model and run it.
    This is for the minifier to check the correctness of the compiled model.

    If `check_accuracy` is set, this function will check the accuracy of the compiled
    model against gm. kwargs must be None if check_accuracy is set.
    "strict_accuracy" means "we will minify any time we see anything that
     diverges", whereas "accuracy" is more conservative, and will only minify if there
     is a meaningful fp64 divergence
    """

    if check_accuracy:
        assert kwargs is None or len(kwargs) == 0, (
            "when checking for accuracy, the inputs must have been flattened and kwargs is None"
        )

    from .package import package_aoti

    assert isinstance(gm, torch.fx.GraphModule)

    kwargs = kwargs or {}

    aoti_files = aot_compile(gm, args, kwargs, options=inductor_configs)
    assert isinstance(aoti_files, list)

    if package_path is None:
        path = [
            os.path.splitext(file)[0]
            for file in aoti_files
            if os.path.splitext(file)[1] == ".so"
        ]
        if len(path) == 0:
            path = [
                os.path.splitext(file)[0]
                for file in aoti_files
                if os.path.splitext(file)[1] == ".cpp"
            ]
        package_path = path[0] + ".pt2"

    res = package_aoti(package_path, aoti_files)
    assert res == package_path

    if load_and_run or check_accuracy:
        compiled_model = aoti_load_package(package_path)
        if check_accuracy:
            from torch._dynamo.debug_utils import AccuracyError, same_two_models

            # This might look inverted but it's not.  strict_accuracy means "we will
            # minify any time we see anything that diverges", whereas accuracy is more
            # conservative, and will only minify if there is a meaningful fp64
            # divergence
            not_strict_accuracy = check_accuracy == "accuracy"
            if not same_two_models(
                gm,
                compiled_model,
                args,
                only_fwd=True,
                require_fp64=not_strict_accuracy,
                ignore_non_fp=not_strict_accuracy,
            ):
                raise AccuracyError("Bad accuracy detected")
        else:
            compiled_model(*args, **kwargs)

    return package_path


def aoti_load_package(path: FileLike, run_single_threaded: bool = False) -> Any:  # type: ignore[type-arg]
    """
    Loads the model from the PT2 package.

    If multiple models were packaged into the PT2, this will load the default
    model. To load a specific model, you can directly call the load API

    .. code-block:: python

        from torch._inductor.package import load_package

        compiled_model1 = load_package("my_package.pt2", "model1")
        compiled_model2 = load_package("my_package.pt2", "model2")

    Args:
        path: Path to the .pt2 package
        run_single_threaded (bool): Whether the model should be run without
            thread synchronization logic. This is useful to avoid conflicts with
            CUDAGraphs.
    """
    from torch._inductor.package import load_package

    return load_package(path, run_single_threaded=run_single_threaded)


def aot_compile(
    gm: torch.fx.GraphModule,
    args: tuple[Any],
    kwargs: Optional[dict[str, Any]] = None,
    *,
    options: Optional[dict[str, Any]] = None,
) -> Union[str, list[str]]:
    """
    Ahead-of-time compile a given FX graph with TorchInductor into a shared library.

    Args:
        gm: The FX graph to compile.
        args:  Example arguments
        kwargs: Example keyword arguments
        options:  Optional dict of config options.  See `torch._inductor.config`.

    Returns:
        Path to the generated shared library, or a list of files generated by
        AOTI if aot_inductor.package=True.
        TODO: make it return a list by default
    """
    from .compile_fx import _aoti_flatten_inputs, compile_fx_aot

    flat_example_inputs, options = _aoti_flatten_inputs(
        gm, args, kwargs, options=options
    )

    return compile_fx_aot(
        gm,
        flat_example_inputs,  # type: ignore[arg-type]
        config_patches=options,
    )


def list_mode_options(
    mode: Optional[str] = None, dynamic: Optional[bool] = None
) -> dict[str, Any]:
    r"""Returns a dictionary describing the optimizations that each of the available
    modes passed to `torch.compile()` performs.

    Args:
        mode (str, optional): The mode to return the optimizations for.
        If None, returns optimizations for all modes
        dynamic (bool, optional): Whether dynamic shape is enabled.

    Example::
        >>> torch._inductor.list_mode_options()
    """

    mode_options: dict[str, dict[str, bool]] = {
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
    try:
        return mode_options[mode] if mode else mode_options
    except KeyError as e:
        raise RuntimeError(
            f"Unrecognized mode={mode}, should be one of: {', '.join(mode_options.keys())}"
        ) from e


def list_options() -> list[str]:
    r"""Returns a dictionary describing the optimizations and debug configurations
    that are available to `torch.compile()`.

    The options are documented in `torch._inductor.config`.

    Example::

        >>> torch._inductor.list_options()
    """

    from torch._inductor import config

    current_config: dict[str, Any] = config.get_config_copy()

    return list(current_config.keys())


def cudagraph_mark_step_begin():
    "Indicates that a new iteration of inference or training is about to begin."
    from .cudagraph_trees import mark_step_begin

    mark_step_begin()
