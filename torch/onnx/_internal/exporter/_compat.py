"""Compatibility functions for the torch.onnx.export API."""

# mypy: allow-untyped-defs
# mypy: disable-error-code=attr-defined
from __future__ import annotations

import inspect
import logging
from typing import Any, Mapping, Sequence, TYPE_CHECKING

import onnx

import torch
import torch.export
from torch.onnx._internal.exporter import _core, _onnx_program


if TYPE_CHECKING:
    import os

logger = logging.getLogger(__name__)


def _signature(model) -> inspect.Signature:
    should_be_callable = getattr(model, "forward", model)
    if callable(should_be_callable):
        return inspect.signature(should_be_callable)
    raise ValueError("model has no forward method and is not callable")


def _from_dynamic_axes_to_dynamic_shapes(
    model,
    dynamic_axes=None,
    input_names: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    """

    dynamic_axes examples:
    (1) dynamic_axes = {"x": {0: "my_custom_axis_name_1"}, "y": {1: "my_custom_axis_name_2"}}
    (2) dynamic_axes = {"x": [0], "y": [1]}

    these will be converted to dynamic_shapes respectively:
    (1) dynamic_shapes = {"x": {0: Dim("my_custom_axis_name_1")}, "y": {1: Dim("my_custom_axis_name_2")}}
    (2) dynamic_shapes = {"x": {0: Dim("x_dim_0")}, "y": {1: Dim("y_dim_1")}}  # auto-generated dim names

    """
    # https://github.com/pytorch/pytorch/pull/128371
    # 1. The function does not need to provide dynamic_shapes to torch.export.export
    if dynamic_axes is None:
        return None

    if input_names is None:
        input_names = []

    sig = _signature(model)
    if len(input_names) > len(sig.parameters):
        raise ValueError(
            f"Number of input names ({len(input_names)}) should not be greater than "
            f"the number of model inputs ({len(sig.parameters)})"
        )
    input_names_to_model_inputs = {}
    for idx, param_name in enumerate(sig.parameters):
        if idx < len(input_names):
            input_names_to_model_inputs[input_names[idx]] = param_name
        else:
            input_names_to_model_inputs[param_name] = param_name

    # NOTE: torch.export.export does not support input names assignment,
    # so we need to map input names to model inputs to create dynamic_shapes
    # for the exported program
    dynamic_shapes_to_exported_program = {}
    for input_name, axes in dynamic_axes.items():
        # input_name can be either from inptu_names or from the model inputs
        if input_name not in input_names_to_model_inputs:
            raise ValueError(
                f"dynamix axis: {input_name} is not found in the input names: {input_names}"
            )
        model_input_name = input_names_to_model_inputs[input_name]
        if isinstance(axes, dict):
            dynamic_shapes_to_exported_program[model_input_name] = {
                k: torch.export.Dim(v) for k, v in axes.items()
            }
        elif isinstance(axes, list):
            dynamic_shapes_to_exported_program[model_input_name] = {
                k: torch.export.Dim(f"{model_input_name}_dim_{k}") for k in axes
            }
        else:
            raise TypeError(
                f"dynamic_axes value must be either a dict or a list, but got {type(axes)}"
            )
    # torch.export.export needs static dim to present in dynamic_shapes
    # for all input tensors, so we need to add them with None
    for input_name in sig.parameters:
        if input_name not in dynamic_shapes_to_exported_program:
            dynamic_shapes_to_exported_program[input_name] = None  # type: ignore[assignment]

    return dynamic_shapes_to_exported_program


def _get_torch_export_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
) -> tuple[tuple[Any, ...], dict[str, Any] | None]:
    """Obtain the arguments for torch.onnx.export from the model and the input arguments."""
    if not kwargs and args and isinstance(args[-1], dict):
        kwargs = args[-1]
        args = args[:-1]
    return args, kwargs


def _convert_version(path: str | os.PathLike, opset_version: int) -> None:
    """Convert the ONNX file to a specific version."""
    model = onnx.load(path, load_external_data=False)
    model = onnx.version_converter.convert_version(model, opset_version)
    onnx.save(model, path)


def export_compat(
    model: torch.nn.Module
    | torch.export.ExportedProgram
    | torch.jit.ScriptModule
    | torch.jit.ScriptFunction,
    args: tuple[Any, ...],
    f: str | os.PathLike | None = None,
    *,
    kwargs: dict[str, Any] | None = None,
    export_params: bool = True,
    verbose: bool | None = None,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    opset_version: int | None = None,
    dynamic_axes: Mapping[str, Mapping[int, str]]
    | Mapping[str, Sequence[int]]
    | None = None,
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = None,
    keep_initializers_as_inputs: bool = False,
    external_data: bool = True,
    report: bool = False,
    verify: bool = False,
    profile: bool = False,
    dump_exported_program: bool = False,
    artifacts_dir: str | os.PathLike = ".",
    fallback: bool = False,
    **_,
) -> _onnx_program.ONNXProgram | None:
    if isinstance(model, torch.export.ExportedProgram):
        # We the model is already exported program, so the args, kwargs, and dynamic_shapes
        # are not used
        dynamic_shapes = dynamic_shapes or {}
    else:
        args, kwargs = _get_torch_export_args(args, kwargs)
        if dynamic_shapes is None and dynamic_axes is not None:
            dynamic_shapes = _from_dynamic_axes_to_dynamic_shapes(
                model, dynamic_axes, input_names
            )

    should_convert_version = False

    try:
        onnx_program = _core.export(
            model,
            args,
            kwargs,
            registry=None,
            dynamic_shapes=dynamic_shapes,
            input_names=input_names,
            output_names=output_names,
            profile=profile,
            report=report,
            verify=verify,
            dump_exported_program=dump_exported_program,
            artifacts_dir=artifacts_dir,
            verbose=verbose,
        )

        if f is not None:
            # Always save the initializers as external data to reduce the size of the ONNX file
            onnx_program.save(
                f,
                include_initializers=export_params,
                keep_initializers_as_inputs=keep_initializers_as_inputs,
                external_data=external_data,
            )
            if (
                opset_version is not None
                and opset_version != onnx_program.model.opset_imports.get("")
            ):
                should_convert_version = True

    except Exception as e:
        if fallback:
            if verbose is not False:
                print(
                    "[torch.onnx] Falling back to legacy torch.onnx.export due "
                    f"to the following error: {e}",
                )
            torch.onnx.utils.export(
                model,  # type: ignore[arg-type]
                args,
                f,  # type: ignore[arg-type]
                kwargs=kwargs,
                export_params=export_params,
                input_names=input_names,
                output_names=output_names,
                opset_version=17,  # TODO(justinchuby): Hard coded to 17 for now
                dynamic_axes=dynamic_axes,
                keep_initializers_as_inputs=keep_initializers_as_inputs,
            )
            onnx_program = None
            if opset_version is None:
                opset_version = 18
            if opset_version != 17:
                should_convert_version = True
        else:
            raise

    if f is not None and should_convert_version:
        assert opset_version is not None
        if verbose is not False:
            print(
                f"[torch.onnx] Converting the ONNX file to opset version {opset_version}..."
            )
        _convert_version(f, opset_version)

    return onnx_program
