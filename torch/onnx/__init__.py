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
    "ExportTypes",
    "OperatorExportTypes",
    "TrainingMode",
    "TensorProtoDataType",
    "JitScalarType",
    # Public functions
    "export",
    "export_to_pretty_string",
    "is_in_onnx_export",
    "select_model_mode_for_export",
    "register_custom_op_symbolic",
    "unregister_custom_op_symbolic",
    "disable_log",
    "enable_log",
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

from typing import Any, Callable, Collection, Mapping, Sequence, TYPE_CHECKING

import torch
from torch import _C
from torch._C import _onnx as _C_onnx
from torch._C._onnx import OperatorExportTypes, TensorProtoDataType, TrainingMode

from ._exporter_states import ExportTypes
from ._internal.onnxruntime import (
    is_onnxrt_backend_supported,
    OrtBackend as _OrtBackend,
    OrtBackendOptions as _OrtBackendOptions,
    OrtExecutionProvider as _OrtExecutionProvider,
)
from ._type_utils import JitScalarType
from .errors import OnnxExporterError
from .utils import (
    _optimize_graph,
    _run_symbolic_function,
    _run_symbolic_method,
    export_to_pretty_string,
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
    ONNXProgram,
    ONNXRuntimeOptions,
    OnnxRegistry,
    enable_fake_mode,
)


if TYPE_CHECKING:
    import os

# Set namespace for exposed private names
DiagnosticOptions.__module__ = "torch.onnx"
ExportOptions.__module__ = "torch.onnx"
ExportTypes.__module__ = "torch.onnx"
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
    report: bool = False,
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
    **_: Any,  # ignored options
) -> Any | None:
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
        dynamic_shapes: A dictionary of dynamic shapes for the model inputs. Refer to
            :func:`torch.export.export` for more details. This is only used (and preferred) when dynamo is True.
            Only one parameter `dynamic_axes` or `dynamic_shapes` should be set
            at the same time.
        report: Whether to generate a markdown report for the export process.
        verify: Whether to verify the exported model using ONNX Runtime.
        profile: Whether to profile the export process.
        dump_exported_program: Whether to dump the :class:`torch.export.ExportedProgram` to a file.
            This is useful for debugging the exporter.
        artifacts_dir: The directory to save the debugging artifacts like the report and the serialized
            exported program.
        fallback: Whether to fallback to the TorchScript exporter if the dynamo exporter fails.

        training: Deprecated option. Instead, set the training mode of the model before exporting.
        operator_export_type: Deprecated option. Only ONNX is supported.
        do_constant_folding: Deprecated option. The exported graph is always optimized.
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
    """
    if dynamo is True or isinstance(model, torch.export.ExportedProgram):
        from torch.onnx._internal import exporter

        if isinstance(args, torch.Tensor):
            args = (args,)
        return exporter.export_compat(
            model,
            args,
            f,
            kwargs=kwargs,
            export_params=export_params,
            verbose=verbose,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            external_data=external_data,
            dynamic_shapes=dynamic_shapes,
            report=report,
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


def dynamo_export(
    model: torch.nn.Module | Callable | torch.export.ExportedProgram,  # type: ignore[name-defined]
    /,
    *model_args,
    export_options: ExportOptions | None = None,
    **model_kwargs,
) -> ONNXProgram | Any:
    """Export a torch.nn.Module to an ONNX graph.

    Args:
        model: The PyTorch model to be exported to ONNX.
        model_args: Positional inputs to ``model``.
        model_kwargs: Keyword inputs to ``model``.
        export_options: Options to influence the export to ONNX.

    Returns:
        An in-memory representation of the exported ONNX model.

    **Example 1 - Simplest export**
    ::

        class MyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x, bias=None):
                out = self.linear(x)
                out = out + bias
                return out


        model = MyModel()
        kwargs = {"bias": 3.0}
        args = (torch.randn(2, 2, 2),)
        onnx_program = torch.onnx.dynamo_export(model, *args, **kwargs).save(
            "my_simple_model.onnx"
        )

    **Example 2 - Exporting with dynamic shapes**
    ::

        # The previous model can be exported with dynamic shapes
        export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
        onnx_program = torch.onnx.dynamo_export(
            model, *args, **kwargs, export_options=export_options
        )
        onnx_program.save("my_dynamic_model.onnx")
    """

    # NOTE: The new exporter is experimental and is not enabled by default.
    import warnings

    from torch.onnx import _flags
    from torch.onnx._internal import exporter
    from torch.utils import _pytree

    if isinstance(model, torch.export.ExportedProgram):
        return exporter.export_compat(
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
                category=FutureWarning,
            )

        if export_options is not None and export_options.dynamic_shapes:
            # Make all shapes dynamic
            def _to_dynamic_shapes_mapper():
                arg_order = 0

                def _to_dynamic_shape(x):
                    nonlocal arg_order
                    if isinstance(x, torch.Tensor):
                        rank = len(x.shape)
                        dynamic_shape = {}
                        for i in range(rank):
                            dynamic_shape[i] = torch.export.Dim(
                                f"arg_{arg_order}_dim_{i}"
                            )
                        arg_order += 1
                        return dynamic_shape
                    else:
                        return None

                return _to_dynamic_shape

            # model_args could be nested
            dynamic_shapes = _pytree.tree_map(
                _to_dynamic_shapes_mapper(),
                model_args,
            )
        else:
            dynamic_shapes = None

        return exporter.export_compat(
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


# TODO(justinchuby): Deprecate these logging functions in favor of the new diagnostic module.

# Returns True iff ONNX logging is turned on.
is_onnx_log_enabled = _C._jit_is_onnx_log_enabled


def enable_log() -> None:
    r"""Enables ONNX logging."""
    _C._jit_set_onnx_log_enabled(True)


def disable_log() -> None:
    r"""Disables ONNX logging."""
    _C._jit_set_onnx_log_enabled(False)


"""Sets output stream for ONNX logging.

Args:
    stream_name (str, default "stdout"): Only 'stdout' and 'stderr' are supported
        as ``stream_name``.
"""
set_log_stream = _C._jit_set_onnx_log_output_stream


"""A simple logging facility for ONNX exporter.

Args:
    args: Arguments are converted to string, concatenated together with a newline
        character appended to the end, and flushed to output stream.
"""
log = _C._jit_onnx_log
