"""Compatibility functions for the torch.onnx.export API."""

# mypy: allow-untyped-defs
# mypy: disable-error-code=attr-defined
from __future__ import annotations

import io
import logging
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TYPE_CHECKING

import torch
from torch.onnx import _constants as onnx_constants
from torch.onnx._internal._lazy_import import onnx, onnx_ir as ir, onnxscript_apis
from torch.onnx._internal.exporter import (
    _constants,
    _core,
    _dynamic_shapes,
    _exportable_module,
    _onnx_program,
    _registration,
)


if TYPE_CHECKING:
    import os

logger = logging.getLogger(__name__)


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
    opset_version: int | None = onnx_constants.ONNX_DEFAULT_OPSET,
    custom_translation_table: dict[Callable, Callable | Sequence[Callable]]
    | None = None,
    dynamic_axes: Mapping[str, Mapping[int, str]]
    | Mapping[str, Sequence[int]]
    | None = None,
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = None,
    keep_initializers_as_inputs: bool = False,
    external_data: bool = True,
    report: bool = False,
    optimize: bool = True,
    verify: bool = False,
    profile: bool = False,
    dump_exported_program: bool = False,
    artifacts_dir: str | os.PathLike = ".",
    fallback: bool = False,
    # Legacy export parameters for fallback
    legacy_export_kwargs: dict[str, Any] | None = None,
) -> _onnx_program.ONNXProgram:
    if opset_version is None:
        opset_version = onnx_constants.ONNX_DEFAULT_OPSET

    if isinstance(model, torch.nn.Module):
        if model.training:
            warnings.warn(
                "Exporting a model while it is in training mode. "
                "Please ensure that this is intended, as it may lead to "
                "different behavior during inference. "
                "Calling model.eval() before export is recommended.",
                UserWarning,
                stacklevel=3,
            )

    if isinstance(model, _exportable_module.ExportableModule):
        # Skip argument extraction if args or kwargs are provided
        if not args and not kwargs:
            args, kwargs = model.example_arguments()
            if input_names is None:
                input_names = model.input_names()
            if output_names is None:
                output_names = model.output_names()
            if dynamic_shapes is None:
                dynamic_shapes = model.dynamic_shapes()

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
                stacklevel=3,
            )
            try:
                dynamic_shapes, args, kwargs = (
                    _dynamic_shapes.from_dynamic_axes_to_dynamic_shapes(
                        model,
                        args,
                        kwargs,
                        dynamic_axes=dynamic_axes,
                        input_names=input_names,
                        output_names=set(output_names or ()),
                    )
                )
            except Exception as e:
                raise RuntimeError(
                    "# Failed to convert 'dynamic_axes' to 'dynamic_shapes'. "
                    "Please provide 'dynamic_shapes' directly. "
                    "Refer to the documentation for 'torch.export.export' for more information on dynamic shapes."
                ) from e

    dynamic_shapes_with_export_dim, need_axis_mapping = (
        _dynamic_shapes.convert_str_to_export_dim(dynamic_shapes)
    )

    if opset_version < _constants.TORCHLIB_OPSET:
        logger.warning(
            "Setting ONNX exporter to use operator set version %s because "
            "the requested opset_version %s is a lower version than we have implementations for. "
            "Automatic version conversion will be performed, which may not be successful "
            "at converting to the requested version. If version conversion is unsuccessful, "
            "the opset version of the exported model will be kept at %s. "
            "Please consider setting opset_version >=%s to leverage latest ONNX features",
            _constants.TORCHLIB_OPSET,
            opset_version,
            _constants.TORCHLIB_OPSET,
            _constants.TORCHLIB_OPSET,
        )
        registry_opset_version = _constants.TORCHLIB_OPSET
    else:
        registry_opset_version = opset_version

    registry = _registration.ONNXRegistry().from_torchlib(
        opset_version=registry_opset_version
    )
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
            dynamic_shapes=dynamic_shapes_with_export_dim,
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
            if dynamic_shapes is not None and dynamic_axes is None:
                if input_names is None:
                    raise ValueError(
                        "Failed to convert dynamic_shapes to dynamic_axes. "
                        "Either input_names or dynamic_axes must be provided "
                        "when dynamic is requested in fallback"
                    ) from e
                dynamic_axes = _dynamic_shapes.from_dynamic_shapes_to_dynamic_axes(
                    dynamic_shapes=dynamic_shapes, input_names=input_names, exception=e
                )
            # Use the legacy export kwargs prepared in __init__.py
            if legacy_export_kwargs is None:
                legacy_export_kwargs = {}

            torch.onnx.utils.export(
                model,  # type: ignore[arg-type]
                args,
                f,  # type: ignore[arg-type]
                kwargs=kwargs,
                export_params=export_params,
                input_names=input_names,
                output_names=output_names,
                opset_version=opset_version,
                dynamic_axes=dynamic_axes,
                keep_initializers_as_inputs=keep_initializers_as_inputs,
                **legacy_export_kwargs,
            )
            onnx_program = _onnx_program.ONNXProgram(ir.load(f), None)

            # NOTE: It it's falling back to the legacy exporter, we don't need to
            # optimize the model, so we return it here. Users can still optimize
            # the model using the optimize() if they want.
            return onnx_program
        else:
            raise

    if need_axis_mapping and dynamic_shapes is not None:
        onnx_program._rename_dynamic_axes(dynamic_shapes)

    # Converter opset version and optimize
    onnx_program.model = onnxscript_apis.convert_version(
        onnx_program.model, opset_version
    )
    if optimize:
        onnx_program.optimize()

    if f is not None:
        if isinstance(f, io.BytesIO):
            # For legacy export compatibility, we allow f to be a BytesIO object.
            # This is not explicitly supported but we may need to maintain the
            # behavior indefinitely.
            warnings.warn(
                "Saving ONNX model to a BytesIO object is deprecated. "
                "Please use a file path instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            onnx.save(onnx_program.model_proto, f)
        else:
            onnx_program.save(
                f,
                include_initializers=export_params,
                keep_initializers_as_inputs=keep_initializers_as_inputs,
                external_data=external_data,
            )

    return onnx_program
