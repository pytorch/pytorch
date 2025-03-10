# mypy: allow-untyped-defs
from __future__ import annotations


__all__ = [
    # Modules
    "symbolic_helper",
    "utils",
    "errors",
    # All opsets
    "symbolic_caffe2",
    "symbolic_opset7",
    "symbolic_opset8",
    "symbolic_opset9",
    "symbolic_opset10",
    "symbolic_opset11",
    "symbolic_opset12",
    "symbolic_opset13",
    "symbolic_opset14",
    "symbolic_opset15",
    "symbolic_opset16",
    "symbolic_opset17",
    "symbolic_opset18",
    "symbolic_opset19",
    "symbolic_opset20",
    # Enums
    "OperatorExportTypes",
    "TrainingMode",
    "TensorProtoDataType",
    "JitScalarType",
    # Public functions
    "export",
    "is_in_onnx_export",
    "select_model_mode_for_export",
    "register_custom_op_symbolic",
    "unregister_custom_op_symbolic",
    # Base error
    "OnnxExporterError",
    # Dynamo Exporter
    "DiagnosticOptions",
    "ExportOptions",
    "ONNXProgram",
    "ONNXRuntimeOptions",
    "OnnxRegistry",
    "dynamo_export",
    "enable_fake_mode",
    # DORT / torch.compile
    "is_onnxrt_backend_supported",
]

from typing import Any, Callable, TYPE_CHECKING
from typing_extensions import deprecated

import torch
from torch import _C
from torch._C import _onnx as _C_onnx
from torch._C._onnx import OperatorExportTypes, TensorProtoDataType, TrainingMode

from ._internal.exporter._onnx_program import ONNXProgram
from ._internal.onnxruntime import (
    is_onnxrt_backend_supported,
    OrtBackend as _OrtBackend,
    OrtBackendOptions as _OrtBackendOptions,
    OrtExecutionProvider as _OrtExecutionProvider,
)
from ._type_utils import JitScalarType
from .errors import OnnxExporterError
from .utils import (
    _run_symbolic_function,
    _run_symbolic_method,
    is_in_onnx_export,
    register_custom_op_symbolic,
    select_model_mode_for_export,
    unregister_custom_op_symbolic,
)


from . import (  # usort: skip. Keep the order instead of sorting lexicographically
    errors,
    symbolic_caffe2,
    symbolic_helper,
    symbolic_opset7,
    symbolic_opset8,
    symbolic_opset9,
    symbolic_opset10,
    symbolic_opset11,
    symbolic_opset12,
    symbolic_opset13,
    symbolic_opset14,
    symbolic_opset15,
    symbolic_opset16,
    symbolic_opset17,
    symbolic_opset18,
    symbolic_opset19,
    symbolic_opset20,
    utils,
)


from ._internal._exporter_legacy import (  # usort: skip. needs to be last to avoid circular import
    DiagnosticOptions,
    ExportOptions,
    ONNXRuntimeOptions,
    OnnxRegistry,
    enable_fake_mode,
)


if TYPE_CHECKING:
    import os
    from collections.abc import Collection, Mapping, Sequence

# Set namespace for exposed private names
DiagnosticOptions.__module__ = "torch.onnx"
ExportOptions.__module__ = "torch.onnx"
JitScalarType.__module__ = "torch.onnx"
ONNXProgram.__module__ = "torch.onnx"
ONNXRuntimeOptions.__module__ = "torch.onnx"
OnnxExporterError.__module__ = "torch.onnx"
OnnxRegistry.__module__ = "torch.onnx"
_OrtBackend.__module__ = "torch.onnx"
_OrtBackendOptions.__module__ = "torch.onnx"
_OrtExecutionProvider.__module__ = "torch.onnx"
enable_fake_mode.__module__ = "torch.onnx"
is_onnxrt_backend_supported.__module__ = "torch.onnx"

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
    export_params: bool = True,
    verbose: bool | None = None,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    opset_version: int | None = None,
    dynamic_axes: Mapping[str, Mapping[int, str]]
    | Mapping[str, Sequence[int]]
    | None = None,
    keep_initializers_as_inputs: bool = False,
    dynamo: bool = False,
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
    # Deprecated options
    training: _C_onnx.TrainingMode = _C_onnx.TrainingMode.EVAL,
    operator_export_type: _C_onnx.OperatorExportTypes = _C_onnx.OperatorExportTypes.ONNX,
    do_constant_folding: bool = True,
    custom_opsets: Mapping[str, int] | None = None,
    export_modules_as_functions: bool | Collection[type[torch.nn.Module]] = False,
    autograd_inlining: bool = True,
) -> ONNXProgram | None:
    r"""Exports a model into ONNX format.

    Args:
        model: The model to be exported.
        args: Example positional inputs. Any non-Tensor arguments will be hard-coded into the
            exported model; any Tensor arguments will become inputs of the exported model,
            in the order they occur in the tuple.
        f: Path to the output ONNX model file. E.g. "model.onnx".
        kwargs: Optional example keyword inputs.
        export_params: If false, parameters (weights) will not be exported.
        verbose: Whether to enable verbose logging.
        input_names: names to assign to the input nodes of the graph, in order.
        output_names: names to assign to the output nodes of the graph, in order.
        opset_version: The version of the
            `default (ai.onnx) opset <https://github.com/onnx/onnx/blob/master/docs/Operators.md>`_
            to target. Must be >= 7.
        dynamic_axes:

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

        keep_initializers_as_inputs: If True, all the
            initializers (typically corresponding to model weights) in the
            exported graph will also be added as inputs to the graph. If False,
            then initializers are not added as inputs to the graph, and only
            the user inputs are added as inputs.

            Set this to True if you intend to supply model weights at runtime.
            Set it to False if the weights are static to allow for better optimizations
            (e.g. constant folding) by backends/runtimes.

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

        training: Deprecated option. Instead, set the training mode of the model before exporting.
        operator_export_type: Deprecated option. Only ONNX is supported.
        do_constant_folding: Deprecated option.
        custom_opsets: Deprecated.
            A dictionary:

            * KEY (str): opset domain name
            * VALUE (int): opset version

            If a custom opset is referenced by ``model`` but not mentioned in this dictionary,
            the opset version is set to 1. Only custom opset domain name and version should be
            indicated through this argument.
        export_modules_as_functions: Deprecated option.

            Flag to enable
            exporting all ``nn.Module`` forward calls as local functions in ONNX. Or a set to indicate the
            particular types of modules to export as local functions in ONNX.
            This feature requires ``opset_version`` >= 15, otherwise the export will fail. This is because
            ``opset_version`` < 15 implies IR version < 8, which means no local function support.
            Module variables will be exported as function attributes. There are two categories of function
            attributes.

            1. Annotated attributes: class variables that have type annotations via
            `PEP 526-style <https://www.python.org/dev/peps/pep-0526/#class-and-instance-variable-annotations>`_
            will be exported as attributes.
            Annotated attributes are not used inside the subgraph of ONNX local function because
            they are not created by PyTorch JIT tracing, but they may be used by consumers
            to determine whether or not to replace the function with a particular fused kernel.

            2. Inferred attributes: variables that are used by operators inside the module. Attribute names
            will have prefix "inferred::". This is to differentiate from predefined attributes retrieved from
            python module annotations. Inferred attributes are used inside the subgraph of ONNX local function.

            * ``False`` (default): export ``nn.Module`` forward calls as fine grained nodes.
            * ``True``: export all ``nn.Module`` forward calls as local function nodes.
            * Set of type of nn.Module: export ``nn.Module`` forward calls as local function nodes,
                only if the type of the ``nn.Module`` is found in the set.
        autograd_inlining: Deprecated.
            Flag used to control whether to inline autograd functions.
            Refer to https://github.com/pytorch/pytorch/pull/74765 for more details.

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
    """
    if dynamo is True or isinstance(model, torch.export.ExportedProgram):
        from torch.onnx._internal.exporter import _compat

        if isinstance(args, torch.Tensor):
            args = (args,)
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
        )
    else:
        from torch.onnx.utils import export

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


@deprecated(
    "torch.onnx.dynamo_export is deprecated since 2.6.0. Please use torch.onnx.export(..., dynamo=True) instead."
)
def dynamo_export(
    model: torch.nn.Module | Callable | torch.export.ExportedProgram,  # type: ignore[name-defined]
    /,
    *model_args,
    export_options: ExportOptions | None = None,
    **model_kwargs,
) -> ONNXProgram:
    """Export a torch.nn.Module to an ONNX graph.

    .. deprecated:: 2.7
        Please use ``torch.onnx.export(..., dynamo=True)`` instead.

    Args:
        model: The PyTorch model to be exported to ONNX.
        model_args: Positional inputs to ``model``.
        model_kwargs: Keyword inputs to ``model``.
        export_options: Options to influence the export to ONNX.

    Returns:
        An in-memory representation of the exported ONNX model.
    """

    import warnings

    from torch.onnx import _flags
    from torch.onnx._internal.exporter import _compat
    from torch.utils import _pytree

    if isinstance(model, torch.export.ExportedProgram):
        return _compat.export_compat(
            model,  # type: ignore[arg-type]
            model_args,
            f=None,
            kwargs=model_kwargs,
            opset_version=18,
            external_data=True,
            export_params=True,
            fallback=True,
        )
    elif _flags.USE_EXPERIMENTAL_LOGIC:
        if export_options is not None:
            warnings.warn(
                "You are using an experimental ONNX export logic, which currently only supports dynamic shapes. "
                "For a more comprehensive set of export options, including advanced features, please consider using "
                "`torch.onnx.export(..., dynamo=True)`. ",
                category=DeprecationWarning,
            )

        if export_options is not None and export_options.dynamic_shapes:
            # Make all shapes dynamic if it's possible
            def _to_dynamic_shape(x):
                if isinstance(x, torch.Tensor):
                    rank = len(x.shape)
                    dynamic_shape = {}
                    for i in range(rank):
                        dynamic_shape[i] = torch.export.Dim.AUTO  # type: ignore[attr-defined]
                    return dynamic_shape
                else:
                    return None

            # model_args could be nested
            dynamic_shapes = _pytree.tree_map(
                _to_dynamic_shape,
                model_args,
            )
        else:
            dynamic_shapes = None

        return _compat.export_compat(
            model,  # type: ignore[arg-type]
            model_args,
            f=None,
            kwargs=model_kwargs,
            dynamic_shapes=dynamic_shapes,
            opset_version=18,
            external_data=True,
            export_params=True,
            fallback=True,
        )
    else:
        from torch.onnx._internal._exporter_legacy import dynamo_export

        return dynamo_export(
            model, *model_args, export_options=export_options, **model_kwargs
        )
