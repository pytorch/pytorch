# mypy: allow-untyped-defs
from __future__ import annotations


__all__ = [
    # Modules
    "errors",
    "ops",
    # Public functions
    "export",
    "is_in_onnx_export",
    # Base error
    "OnnxExporterError",
    "ONNXProgram",
]

from typing import Any, TYPE_CHECKING

import torch
from torch._C import _onnx as _C_onnx
from torch._C._onnx import (  # Deprecated members that are excluded from __all__
    OperatorExportTypes as OperatorExportTypes,
    TensorProtoDataType as TensorProtoDataType,
    TrainingMode as TrainingMode,
)

from . import errors, ops
from ._internal.exporter._onnx_program import ONNXProgram
from ._internal.torchscript_exporter import (  # Deprecated members that are excluded from __all__
    symbolic_helper,
    symbolic_opset10,
    symbolic_opset9,
    utils,
)
from ._internal.torchscript_exporter._type_utils import (
    JitScalarType,  # Deprecated members that are excluded from __all__
)
from ._internal.torchscript_exporter.utils import (  # Deprecated members that are excluded from __all__
    register_custom_op_symbolic,
    select_model_mode_for_export,  # pyrefly: ignore  # deprecated
    unregister_custom_op_symbolic,
)
from .errors import OnnxExporterError


if TYPE_CHECKING:
    import os
    from collections.abc import Callable, Collection, Mapping, Sequence

# Set namespace for exposed private names
ONNXProgram.__module__ = "torch.onnx"
OnnxExporterError.__module__ = "torch.onnx"

# TODO(justinchuby): Remove these two properties
producer_name = "pytorch"
producer_version = _C_onnx.PRODUCER_VERSION


def export(
    model: torch.nn.Module
    | torch.export.ExportedProgram
    | torch.jit.ScriptModule
    | torch.jit.ScriptFunction,
    args: tuple[Any, ...] = (),
    f: str | os.PathLike | None = None,
    *,
    kwargs: dict[str, Any] | None = None,
    verbose: bool | None = None,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    opset_version: int | None = None,
    dynamo: bool = True,
    # Dynamo only options
    external_data: bool = True,
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = None,
    custom_translation_table: dict[Callable, Callable | Sequence[Callable]]
    | None = None,
    report: bool = False,
    optimize: bool = True,
    verify: bool = False,
    profile: bool = False,
    dump_exported_program: bool = False,
    artifacts_dir: str | os.PathLike = ".",
    fallback: bool = False,
    # BC options
    export_params: bool = True,
    keep_initializers_as_inputs: bool = False,
    dynamic_axes: Mapping[str, Mapping[int, str]]
    | Mapping[str, Sequence[int]]
    | None = None,
    # Deprecated options
    training: _C_onnx.TrainingMode = _C_onnx.TrainingMode.EVAL,
    operator_export_type: _C_onnx.OperatorExportTypes = _C_onnx.OperatorExportTypes.ONNX,
    do_constant_folding: bool = True,
    custom_opsets: Mapping[str, int] | None = None,
    export_modules_as_functions: bool | Collection[type[torch.nn.Module]] = False,
    autograd_inlining: bool = True,
) -> ONNXProgram | None:
    r"""Exports a model into ONNX format.

    Setting ``dynamo=True`` enables the new ONNX export logic
    which is based on :class:`torch.export.ExportedProgram` and a more modern
    set of translation logic. This is the recommended and default way to export models
    to ONNX.

    When ``dynamo=True``:

    The exporter tries the following strategies to get an ExportedProgram for conversion to ONNX.

    #. If the model is already an ExportedProgram, it will be used as-is.
    #. Use :func:`torch.export.export` and set ``strict=False``.
    #. Use :func:`torch.export.export` and set ``strict=True``.

    Args:
        model: The model to be exported.
        args: Example positional inputs. Any non-Tensor arguments will be hard-coded into the
            exported model; any Tensor arguments will become inputs of the exported model,
            in the order they occur in the tuple.
        f: Path to the output ONNX model file. E.g. "model.onnx". This argument is kept for
            backward compatibility. It is recommended to leave unspecified (None)
            and use the returned :class:`torch.onnx.ONNXProgram` to serialize the model
            to a file instead.
        kwargs: Optional example keyword inputs.
        verbose: Whether to enable verbose logging.
        input_names: names to assign to the input nodes of the graph, in order.
        output_names: names to assign to the output nodes of the graph, in order.
        opset_version: The version of the
            `default (ai.onnx) opset <https://github.com/onnx/onnx/blob/master/docs/Operators.md>`_
            to target. You should set ``opset_version`` according to the supported opset versions
            of the runtime backend or compiler you want to run the exported model with.
            Leave as default (``None``) to use the recommended version, or refer to
            the ONNX operators documentation for more information.
        dynamo: Whether to export the model with ``torch.export`` ExportedProgram instead of TorchScript.
        external_data: Whether to save the model weights as an external data file.
            This is required for models with large weights that exceed the ONNX file size limit (2GB).
            When False, the weights are saved in the ONNX file with the model architecture.
        dynamic_shapes: A dictionary or a tuple of dynamic shapes for the model inputs. Refer to
            :func:`torch.export.export` for more details. This is only used (and preferred) when dynamo is True.
            Note that dynamic_shapes is designed to be used when the model is exported with dynamo=True, while
            dynamic_axes is used when dynamo=False.
        custom_translation_table: A dictionary of custom decompositions for operators in the model.
            The dictionary should have the callable target in the fx Node as the key (e.g. ``torch.ops.aten.stft.default``),
            and the value should be a function that builds that graph using ONNX Script. This option
            is only valid when dynamo is True.
        report: Whether to generate a markdown report for the export process. This option
            is only valid when dynamo is True.
        optimize: Whether to optimize the exported model. This option
            is only valid when dynamo is True. Default is True.
        verify: Whether to verify the exported model using ONNX Runtime. This option
            is only valid when dynamo is True.
        profile: Whether to profile the export process. This option
            is only valid when dynamo is True.
        dump_exported_program: Whether to dump the :class:`torch.export.ExportedProgram` to a file.
            This is useful for debugging the exporter. This option is only valid when dynamo is True.
        artifacts_dir: The directory to save the debugging artifacts like the report and the serialized
            exported program. This option is only valid when dynamo is True.
        fallback: Whether to fallback to the TorchScript exporter if the dynamo exporter fails.
            This option is only valid when dynamo is True. When fallback is enabled, It is
            recommended to set dynamic_axes even when dynamic_shapes is provided.
        export_params: **When ``f`` is specified**: If false, parameters (weights) will not be exported.

            You can also leave it unspecified and use the returned :class:`torch.onnx.ONNXProgram`
            to control how initializers are treated when serializing the model.
        keep_initializers_as_inputs: **When ``f`` is specified**: If True, all the
            initializers (typically corresponding to model weights) in the
            exported graph will also be added as inputs to the graph. If False,
            then initializers are not added as inputs to the graph, and only
            the user inputs are added as inputs.

            Set this to True if you intend to supply model weights at runtime.
            Set it to False if the weights are static to allow for better optimizations
            (e.g. constant folding) by backends/runtimes.

            You can also leave it unspecified and use the returned :class:`torch.onnx.ONNXProgram`
            to control how initializers are treated when serializing the model.
        dynamic_axes:
            Prefer specifying ``dynamic_shapes`` when ``dynamo=True`` and when ``fallback``
            is not enabled.

            By default the exported model will have the shapes of all input and output tensors
            set to exactly match those given in ``args``. To specify axes of tensors as
            dynamic (i.e. known only at run-time), set ``dynamic_axes`` to a dict with schema:

            * KEY (str): an input or output name. Each name must also be provided in ``input_names`` or
                ``output_names``.
            * VALUE (dict or list): If a dict, keys are axis indices and values are axis names. If a
                list, each element is an axis index.

            For example::

                class SumModule(torch.nn.Module):
                    def forward(self, x):
                        return torch.sum(x, dim=1)


                torch.onnx.export(
                    SumModule(),
                    (torch.ones(2, 2),),
                    "onnx.pb",
                    input_names=["x"],
                    output_names=["sum"],
                )

            Produces::

                input {
                  name: "x"
                  ...
                      shape {
                        dim {
                          dim_value: 2  # axis 0
                        }
                        dim {
                          dim_value: 2  # axis 1
                ...
                output {
                  name: "sum"
                  ...
                      shape {
                        dim {
                          dim_value: 2  # axis 0
                ...

            While::

                torch.onnx.export(
                    SumModule(),
                    (torch.ones(2, 2),),
                    "onnx.pb",
                    input_names=["x"],
                    output_names=["sum"],
                    dynamic_axes={
                        # dict value: manually named axes
                        "x": {0: "my_custom_axis_name"},
                        # list value: automatic names
                        "sum": [0],
                    },
                )

            Produces::

                input {
                  name: "x"
                  ...
                      shape {
                        dim {
                          dim_param: "my_custom_axis_name"  # axis 0
                        }
                        dim {
                          dim_value: 2  # axis 1
                ...
                output {
                  name: "sum"
                  ...
                      shape {
                        dim {
                          dim_param: "sum_dynamic_axes_1"  # axis 0
                ...

        training: Deprecated option. Instead, set the training mode of the model before exporting.
        operator_export_type: Deprecated option. Only ONNX is supported.
        do_constant_folding: Deprecated option.
        custom_opsets: Deprecated option.
        export_modules_as_functions: Deprecated option.
        autograd_inlining: Deprecated option.

    Returns:
        :class:`torch.onnx.ONNXProgram` if dynamo is True, otherwise None.

    .. versionchanged:: 2.6
        *training* is now deprecated. Instead, set the training mode of the model before exporting.
        *operator_export_type* is now deprecated. Only ONNX is supported.
        *do_constant_folding* is now deprecated. It is always enabled.
        *export_modules_as_functions* is now deprecated.
        *autograd_inlining* is now deprecated.
    .. versionchanged:: 2.7
        *optimize* is now True by default.
    .. versionchanged:: 2.9
        *dynamo* is now True by default.
    """
    if dynamo is True or isinstance(model, torch.export.ExportedProgram):
        from torch.onnx._internal.exporter import _compat

        if isinstance(args, torch.Tensor):
            args = (args,)
        # Prepare legacy export parameters for potential fallback
        legacy_export_kwargs = {
            "training": training,
            "operator_export_type": operator_export_type,
            "do_constant_folding": do_constant_folding,
            "custom_opsets": custom_opsets,
            "export_modules_as_functions": export_modules_as_functions,
            "autograd_inlining": autograd_inlining,
        }

        return _compat.export_compat(
            model,
            args,
            f,
            kwargs=kwargs,
            export_params=export_params,
            verbose=verbose,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            custom_translation_table=custom_translation_table,
            dynamic_axes=dynamic_axes,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            external_data=external_data,
            dynamic_shapes=dynamic_shapes,
            report=report,
            optimize=optimize,
            verify=verify,
            profile=profile,
            dump_exported_program=dump_exported_program,
            artifacts_dir=artifacts_dir,
            fallback=fallback,
            legacy_export_kwargs=legacy_export_kwargs,
        )
    else:
        import warnings

        from ._internal.torchscript_exporter.utils import export

        warnings.warn(
            "You are using the legacy TorchScript-based ONNX export. Starting in PyTorch 2.9, "
            "the new torch.export-based ONNX exporter has become the default. "
            "Learn more about the new export logic: https://docs.pytorch.org/docs/stable/onnx_export.html. "
            "For exporting control flow: "
            "https://pytorch.org/tutorials/beginner/onnx/export_control_flow_model_to_onnx_tutorial.html",
            category=DeprecationWarning,
            stacklevel=2,
        )

        if dynamic_shapes:
            raise ValueError(
                "The exporter only supports dynamic shapes "
                "through parameter dynamic_axes when dynamo=False."
            )

        export(
            model,
            args,
            f,  # type: ignore[arg-type]
            kwargs=kwargs,
            export_params=export_params,
            verbose=verbose is True,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            training=training,
            operator_export_type=operator_export_type,
            do_constant_folding=do_constant_folding,
            custom_opsets=custom_opsets,
            export_modules_as_functions=export_modules_as_functions,
            autograd_inlining=autograd_inlining,
        )
        return None


def is_in_onnx_export() -> bool:
    """Returns whether it is in the middle of ONNX export."""
    from torch.onnx._internal.exporter import _flags
    from torch.onnx._internal.torchscript_exporter._globals import GLOBALS

    return GLOBALS.in_onnx_export or _flags._is_onnx_exporting
