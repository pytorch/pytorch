"""Compatibility functions for the torch.onnx.export API."""

# mypy: allow-untyped-defs
# mypy: disable-error-code=attr-defined
from __future__ import annotations

import inspect
import logging
import re
import warnings
from typing import Any, Callable, Mapping, Sequence, TYPE_CHECKING

import torch
from torch.onnx._internal._lazy_import import onnxscript_apis, onnxscript_ir as ir
from torch.onnx._internal.exporter import _core, _onnx_program, _registration
from torch.utils import _pytree


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
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
    *,
    dynamic_axes=None,
    output_names: set[str],
    input_names: Sequence[str] | None = None,
) -> dict[str, Any | None] | None:
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

    if kwargs is None:
        kwargs = {}

    dynamic_shapes: dict[str, Any | None] = {}
    for input_name, axes in dynamic_axes.items():
        # NOTE: torch.export.Dim requires strict min and max constraints, and it
        # dpends on the traced model to provide the correct min and max values.
        # We set max to 99999 to avoid the constraints violation error with the default int64 max.
        # https://github.com/pytorch/pytorch/blob/32f585d9346e316e554c8d9bf7548af9f62141fc/test/export/test_export.py#L687
        if input_name in output_names:
            # User specified an output name as a dynamic axis, so we skip it
            continue
        if isinstance(axes, dict):
            # Dim needs to pass str.isidentifier()
            dynamic_shapes[input_name] = {
                k: torch.export.Dim(re.sub(r"[^A-Za-z_]", "", v), max=99999)
                for k, v in axes.items()
            }
        elif isinstance(axes, list):
            dynamic_shapes[input_name] = {
                k: torch.export.Dim(f"{input_name}_dim_{k}", max=99999) for k in axes
            }
        else:
            raise ValueError(
                "Unsupported dynamic_axes format. Please provide a dict or a list."
            )

    for input_name in input_names:
        if input_name not in dynamic_shapes:
            dynamic_shapes[input_name] = None

    # Order the inputs according to the signature of the model
    sig = _signature(model)
    inputs = []
    for idx, param_name in enumerate(sig.parameters):
        if idx < len(args):
            inputs.append(args[idx])
        elif param_name in kwargs:
            inputs.append(kwargs[param_name])

    # We need tree structure to represent dynamic_shapes
    _, tree_structure = _pytree.tree_flatten(inputs)
    dynamic_shapes = _pytree.tree_unflatten(dynamic_shapes.values(), tree_structure)

    return dynamic_shapes


def _get_torch_export_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
) -> tuple[tuple[Any, ...], dict[str, Any] | None]:
    """Obtain the arguments for torch.onnx.export from the model and the input arguments."""
    if not kwargs and args and isinstance(args[-1], dict):
        kwargs = args[-1]
        args = args[:-1]
    return args, kwargs


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
    custom_translation_table: dict[Callable, Callable | Sequence[Callable]]
    | None = None,
    dynamic_axes: Mapping[str, Mapping[int, str]]
    | Mapping[str, Sequence[int]]
    | None = None,
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = None,
    keep_initializers_as_inputs: bool = False,
    external_data: bool = True,
    report: bool = False,
    optimize: bool = False,
    verify: bool = False,
    profile: bool = False,
    dump_exported_program: bool = False,
    artifacts_dir: str | os.PathLike = ".",
    fallback: bool = False,
    **_,
) -> _onnx_program.ONNXProgram:
    if opset_version is None:
        # TODO(justinchuby): Change the hardcoded opset version for it to be flexible
        opset_version = 18

    if isinstance(model, torch.export.ExportedProgram):
        # We know the model is already exported program, so the args, kwargs, and dynamic_shapes
        # are not used
        dynamic_shapes = dynamic_shapes or {}
    else:
        args, kwargs = _get_torch_export_args(args, kwargs)
        if dynamic_shapes is None and dynamic_axes is not None:
            warnings.warn(
                "# 'dynamic_axes' is not recommended when dynamo=True, "
                "and may lead to 'torch._dynamo.exc.UserError: Constraints violated.' "
                "Supply the 'dynamic_shapes' argument instead if export is unsuccessful.",
                UserWarning,
            )
            try:
                dynamic_shapes = _from_dynamic_axes_to_dynamic_shapes(
                    model,
                    args,
                    kwargs,
                    dynamic_axes=dynamic_axes,
                    input_names=input_names,
                    output_names=set(output_names or ()),
                )
            except Exception as e:
                raise RuntimeError(
                    "# Failed to convert 'dynamic_axes' to 'dynamic_shapes'. "
                    "Please provide 'dynamic_shapes' directly. "
                    "Refer to the documentation for 'torch.export.export' for more information on dynamic shapes."
                ) from e

    registry = _registration.ONNXRegistry.from_torchlib()
    if custom_translation_table is not None:
        for torch_op, onnx_ops in custom_translation_table.items():
            # TODO(justinchuby): Support complex inputs with annotations
            if not isinstance(onnx_ops, Sequence):
                onnx_ops = (onnx_ops,)
            for op in reversed(onnx_ops):
                # register_op places the op in the front of all onnx variants,
                # so we reverse the list to maintain the order of the custom ops provided
                registry.register_op(torch_op, op, is_complex=False)
    try:
        onnx_program = _core.export(
            model,
            args,
            kwargs,
            registry=registry,
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

    except Exception as e:
        if fallback:
            if verbose is not False:
                print(
                    "[torch.onnx] Falling back to legacy torch.onnx.export due "
                    f"to the following error: {e}",
                )
            if f is None:
                raise TypeError("f must be provided when fallback is enabled") from e
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
            onnx_program = _onnx_program.ONNXProgram(ir.load(f), None)

            # NOTE: It it's falling back to the legacy exporter, we don't need to
            # optimize the model, so we return it here. Users can still optimize
            # the model using the optimize() if they want.
            return onnx_program
        else:
            raise

    # Converter opset version and optimize
    onnx_program.model = onnxscript_apis.convert_version(
        onnx_program.model, opset_version
    )
    if optimize:
        onnx_program.optimize()

    if f is not None:
        onnx_program.save(
            f,
            include_initializers=export_params,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            external_data=external_data,
        )

    return onnx_program
