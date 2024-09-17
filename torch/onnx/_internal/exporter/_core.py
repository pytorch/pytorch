# mypy: allow-untyped-defs
# flake8: noqa: B950 We do not need flake8 as it complains line length
from __future__ import annotations

import ctypes
import datetime
import inspect
import itertools
import logging
import operator
import pathlib
import textwrap
import traceback
import typing
from typing import Any, Callable, Literal, Sequence

import onnxscript
import onnxscript.evaluator
from onnxscript import ir
from onnxscript.ir import convenience as ir_convenience

import torch
import torch.fx
from torch.export import graph_signature
from torch.onnx._internal._lazy_import import onnxscript_apis
from torch.onnx._internal.exporter import (
    _analysis,
    _building,
    _capture_strategies,
    _dispatching,
    _errors,
    _fx_passes,
    _ir_passes,
    _onnx_program,
    _registration,
    _reporting,
    _tensors,
    _verification,
)


if typing.TYPE_CHECKING:
    import os

    import numpy as np


# Define utilities to convert PyTorch data types so users do not need to specify manually
_TORCH_DTYPE_TO_ONNX: dict[torch.dtype, ir.DataType] = {
    torch.bfloat16: ir.DataType.BFLOAT16,
    torch.bool: ir.DataType.BOOL,
    torch.complex128: ir.DataType.COMPLEX128,
    torch.complex64: ir.DataType.COMPLEX64,
    torch.float16: ir.DataType.FLOAT16,
    torch.float32: ir.DataType.FLOAT,
    torch.float64: ir.DataType.DOUBLE,
    torch.float8_e4m3fn: ir.DataType.FLOAT8E4M3FN,
    torch.float8_e4m3fnuz: ir.DataType.FLOAT8E4M3FNUZ,
    torch.float8_e5m2: ir.DataType.FLOAT8E5M2,
    torch.float8_e5m2fnuz: ir.DataType.FLOAT8E5M2FNUZ,
    torch.int16: ir.DataType.INT16,
    torch.int32: ir.DataType.INT32,
    torch.int64: ir.DataType.INT64,
    torch.int8: ir.DataType.INT8,
    torch.uint8: ir.DataType.UINT8,
    torch.uint16: ir.DataType.UINT16,
    torch.uint32: ir.DataType.UINT32,
    torch.uint64: ir.DataType.UINT64,
}
_BLUE = "\033[96m"
_END = "\033[0m"

_STEP_ONE_ERROR_MESSAGE = textwrap.dedent(
    f"""\
    Failed to export the model with torch.export. {_BLUE}This is step 1/2{_END} of exporting the model to ONNX. Next steps:
    - Modify the model code for `torch.export.export` to succeed. Refer to https://pytorch.org/docs/stable/generated/exportdb/index.html for more information.
    - Debug `torch.export.export` and summit a PR to PyTorch.
    - Create an issue in the PyTorch GitHub repository against the {_BLUE}*torch.export*{_END} component and attach the full error stack as well as reproduction scripts."""
)

_STEP_TWO_ERROR_MESSAGE = textwrap.dedent(
    f"""\
    Failed to convert the exported program to an ONNX model. {_BLUE}This is step 2/2{_END} of exporting the model to ONNX. Next steps:
    - If there is a missing ONNX function, implement it and register it to the registry.
    - If there is an internal error during ONNX conversion, debug the error and summit a PR to PyTorch.
    - Save the ExportedProgram as a pt2 file and create an error report with `export(..., report=True)`. Create an issue in the PyTorch GitHub repository against the {_BLUE}*onnx*{_END} component. Attach the pt2 model and the error report."""
)

logger = logging.getLogger(__name__)


def _torch_dtype_to_onnx_dtype(dtype: torch.dtype) -> ir.DataType:
    return _TORCH_DTYPE_TO_ONNX[dtype]


class TorchTensor(ir.Tensor):
    def __init__(self, tensor: torch.Tensor, name: str | None = None):
        # Pass the tensor as the raw data to ir.Tensor's constructor
        super().__init__(
            tensor, dtype=_torch_dtype_to_onnx_dtype(tensor.dtype), name=name
        )

    def numpy(self) -> np.ndarray:
        self.raw: torch.Tensor
        if self.dtype == ir.DataType.BFLOAT16:
            return self.raw.view(torch.uint16).numpy(force=True)
        if self.dtype in {
            ir.DataType.FLOAT8E4M3FN,
            ir.DataType.FLOAT8E4M3FNUZ,
            ir.DataType.FLOAT8E5M2,
            ir.DataType.FLOAT8E5M2FNUZ,
        }:
            # TODO: Use ml_dtypes
            return self.raw.view(torch.uint8).numpy(force=True)
        return self.raw.numpy(force=True)

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        del copy  # Unused, but needed for the signature
        if dtype is None:
            return self.numpy()
        return self.numpy().__array__(dtype)

    def tobytes(self) -> bytes:
        # Implement tobytes to support native PyTorch types so we can use types like bloat16
        # Reading from memory directly is also more efficient because
        # it avoids copying to a NumPy array
        import torch._subclasses.fake_tensor

        if isinstance(self.raw, torch._subclasses.fake_tensor.FakeTensor):
            raise TypeError(
                f"Cannot take content out from the FakeTensor ('{self.name}'). Please replace the tensor "
                "with a tensor backed by real data using ONNXProgram.apply_weights() "
                "or save the model without initializers by setting include_initializers=False."
            )
        tensor = self.raw.detach().cpu().contiguous()
        return bytes(
            (ctypes.c_ubyte * tensor.element_size() * tensor.numel()).from_address(
                tensor.data_ptr()
            )
        )


# https://github.com/pytorch/pytorch/blob/ee6cb6daa173896f8ea1876266a19775aaa4f610/torch/export/graph_signature.py#L56C1-L62C19
# class InputKind(Enum):
#     USER_INPUT = auto()
#     PARAMETER = auto()
#     BUFFER = auto()
#     CONSTANT_TENSOR = auto()
#     CUSTOM_OBJ = auto()
#     TOKEN = auto()

# https://github.com/pytorch/pytorch/blob/ee6cb6daa173896f8ea1876266a19775aaa4f610/torch/export/graph_signature.py#L89C1-L96C19
# class OutputKind(Enum):
#     USER_OUTPUT = auto()
#     LOSS_OUTPUT = auto()
#     BUFFER_MUTATION = auto()
#     GRADIENT_TO_PARAMETER = auto()
#     GRADIENT_TO_USER_INPUT = auto()
#     USER_INPUT_MUTATION = auto()
#     TOKEN = auto()


def _set_shape_types(
    values: Sequence[ir.Value],
    meta_vals: Sequence[torch.Tensor],
    complex_to_float: bool = True,
) -> None:
    if not isinstance(meta_vals, Sequence):
        logger.warning(
            "Expected meta_vals to be a sequence, but got %s. There may be an internal error.",
            meta_vals,
        )
        meta_vals = (meta_vals,)
    for value, meta_val in zip(values, meta_vals):
        _set_shape_type(value, meta_val, complex_to_float=complex_to_float)


def _set_shape_type(
    value: ir.Value,
    meta_val: torch.Tensor
    | torch.SymBool
    | torch.SymInt
    | torch.SymFloat
    | tuple[torch.Tensor],
    complex_to_float: bool,
) -> None:
    # TODO: Consider using meta["tensor_meta"] for this? Would it be faster?
    if isinstance(meta_val, tuple):
        logger.warning("Setting shape and type of tensors is not supported yet")
    if isinstance(meta_val, torch.Tensor):
        # FIXME: Consider shape for complex values
        dims = []
        for dim in meta_val.shape:
            if isinstance(dim, int):
                dims.append(dim)
            else:
                dims.append(str(dim.node))
        value.dtype = _torch_dtype_to_onnx_dtype(meta_val.dtype)
        if complex_to_float:
            if meta_val.dtype == torch.complex64:
                value.dtype = ir.DataType.FLOAT
                # Add 2 as the last dimension if the tensor is complex to hold the real/imag parts
                dims.append(2)
            elif meta_val.dtype == torch.complex128:
                value.dtype = ir.DataType.DOUBLE
                # Add 2 as the last dimension if the tensor is complex to hold the real/imag parts
                dims.append(2)

        value.shape = ir.Shape(dims)
    elif isinstance(meta_val, (int, torch.SymInt)):
        # aten::sym_size output is a int, not a tensor, which stands
        # for the size of one dim. We treat it as a scalar.
        value.dtype = ir.DataType.INT64
        value.shape = ir.Shape([])
    elif isinstance(meta_val, (bool, torch.SymBool)):
        value.dtype = ir.DataType.BOOL
        value.shape = ir.Shape([])
    elif isinstance(meta_val, (float, torch.SymFloat)):
        value.dtype = ir.DataType.FLOAT
        value.shape = ir.Shape([])
    else:
        pass


def _get_qualified_module_name(cls: Any) -> str:
    if isinstance(cls, str):
        return cls
    module = cls.__module__
    if module is None or module == str.__class__.__module__:
        return cls.__name__
    return module + "." + cls.__name__


def _get_node_namespace(node: torch.fx.Node) -> tuple[str, list[str], list[str]]:
    """Get the namespace and scope of the node.

    Example::

        {
            'L__self__': ('', <class 'torchvision.models.resnet.ResNet'>),
            'L__self___avgpool': ('avgpool', <class 'torch.nn.modules.pooling.AdaptiveAvgPool2d'>)
        }

    Will yield

    namespace: ": torchvision.models.resnet.ResNet/avgpool: torch.nn.modules.pooling.AdaptiveAvgPool2d/node_name: node_target"
    class_hierarchy: ["torchvision.models.resnet.ResNet", "torch.nn.modules.pooling.AdaptiveAvgPool2d", <node_target>]
    name_scopes: ["", "avgpool", <node_name>]

    Args:
        node: The node to get the namespace and scope of.

    Returns:
        (namespace, class_hierarchy, name_scope)
    """
    nn_module_stack = node.meta.get("nn_module_stack")
    logger.debug("%s", nn_module_stack)
    if nn_module_stack is None:
        logger.warning(
            "nn_module_stack not found for node '%s'. Skip adding metadata...",
            node.name,
        )
        return f"{node.name}: {node.target}", [str(node.target)], [node.name]
    namespaces = []
    class_hierarchy = []
    name_scopes = []
    for name, nn_module in nn_module_stack.values():
        name_scopes.append(name)
        nn_module_name = _get_qualified_module_name(nn_module)
        class_hierarchy.append(nn_module_name)
        namespaces.append(f"{name}: {_get_qualified_module_name(nn_module)}")
    namespaces.append(f"{node.name}: {node.target}")
    class_hierarchy.append(str(node.target))
    name_scopes.append(node.name)

    return "/".join(namespaces), class_hierarchy, name_scopes


def _set_node_metadata(fx_node: torch.fx.Node, ir_node: ir.Node) -> None:
    """Adds namespace and other node metadata to the ONNX node."""
    namespace, class_hierarchy, name_scopes = _get_node_namespace(fx_node)
    ir_node.metadata_props["namespace"] = namespace
    ir_node.metadata_props["pkg.torch.onnx.class_hierarchy"] = repr(class_hierarchy)
    ir_node.metadata_props["pkg.torch.onnx.name_scopes"] = repr(name_scopes)
    ir_node.metadata_props["pkg.torch.onnx.fx_node"] = str(fx_node.format_node())
    ir_node.metadata_props["pkg.torch.onnx.stack_trace"] = fx_node.meta.get(
        "stack_trace", ""
    )


def _handle_getitem_node(
    node: torch.fx.Node, node_name_to_values: dict[str, ir.Value | Sequence[ir.Value]]
) -> ir.Value:
    """Handle a getitem node.

    Add the input value it is getting to the mapping, then return the value.

    There are two cases for this node:
    1. The output is a Sequence (traced), we can simply get the value from the sequence
    2. The output is produced by a SplitToSequence node, we need to get the value from the sequence value
    This function only handles the first case
    """
    assert len(node.all_input_nodes) == 1
    source = node.all_input_nodes[0]
    source_outputs = node_name_to_values[source.name]
    assert isinstance(
        source_outputs, Sequence
    ), f"Expected {source.name} to output sequence, got {node_name_to_values[source.name]}"
    index = typing.cast(int, node.args[1])
    value = source_outputs[index]
    # Save the getitem value to the values mapping to in case
    # it is one of the graph outputs
    node_name_to_values[node.name] = value
    # Rename the name of value with the getitem name.
    value.name = node.name
    return value


def _handle_call_function_node(
    graph: ir.Graph,
    node: torch.fx.Node,
    node_name_to_values: dict[str, ir.Value | Sequence[ir.Value]],
) -> None:
    """Handle a call_function node.

    Args:
        graph: The ONNX graph at construction.
        node: The FX node to translate.
        node_name_to_values: A mapping of FX node names to their produced ir.Value.
    """
    if node.target == operator.getitem:
        _handle_getitem_node(node, node_name_to_values)
    # Add op to the graph
    op = str(node.target)
    fx_inputs, attributes, input_names, output_names = _get_inputs_and_attributes(node)
    inputs: list[ir.Value | None] = []
    for i, input_ in enumerate(fx_inputs):
        if input_ is None:
            inputs.append(None)
        elif hasattr(input_, "name"):
            if isinstance(input_, torch.fx.Node) and input_.target == operator.getitem:
                actual_input = _handle_getitem_node(input_, node_name_to_values)
                inputs.append(actual_input)
            else:
                value = node_name_to_values[input_.name]
                assert not isinstance(value, Sequence)
                inputs.append(value)
        else:
            attributes[f"arg_{i}"] = input_

    outputs = [ir.Value(name=name) for name in output_names]
    if len(outputs) > 1:
        _set_shape_types(outputs, node.meta["val"], complex_to_float=False)
        node_name_to_values[node.name] = outputs
    else:
        _set_shape_type(outputs[0], node.meta["val"], complex_to_float=False)
        node_name_to_values[node.name] = outputs[0]
    ir_node = ir.Node(
        "pkg.torch.ops",
        op,
        inputs,
        attributes=ir_convenience.convert_attributes(attributes),
        outputs=outputs,
        name=node.name,
    )
    ir_node.meta["node"] = node
    ir_node.metadata_props["pkg.torch.onnx.input_names"] = repr(input_names)
    # Record the nn.Module stack for the node
    _set_node_metadata(node, ir_node)

    graph.append(ir_node)


def _convert_fx_arg_to_onnx_arg(
    arg, node_name_to_values: dict[str, ir.Value | Sequence[ir.Value]]
) -> Any:
    """Convert an FX argument to an ONNX compatible argument.

    This function
    - Converts a torch dtype to an integer
    - Converts a torch device/memory_format/layout to a string
    - Converts a torch.fx.Node to an ir.Value
    - Converts a sequence of torch.fx.Node to a sequence of ir.Value
    """
    if arg is None:
        # None arguments are not modified because when the arg is an ONNX input
        # we need to preserve the None value; when the arg is an ONNX attribute,
        # we want to drop the value.
        # The actual dropping of a None attribute value is done by OpRecorder
        return None
    if hasattr(arg, "name"):
        if isinstance(arg, torch.fx.Node) and arg.target == operator.getitem:
            source = arg.all_input_nodes[0]
            source_outputs = node_name_to_values[source.name]
            if isinstance(source_outputs, Sequence):
                # If the node is getting an input from another node, get the actual value the node is retrieving
                return _handle_getitem_node(arg, node_name_to_values)
            else:
                # `source_outputs` is a sequence(tensor()) value and we need to
                # use SequenceAt to get the value. This is handled by torchlib
                pass
        # If the input is a node, get the value from the mapping
        return node_name_to_values[arg.name]
    if isinstance(arg, (list, tuple)):
        return [_convert_fx_arg_to_onnx_arg(elem, node_name_to_values) for elem in arg]
    if isinstance(arg, (torch.device, torch.memory_format, torch.layout)):
        return str(arg)
    if isinstance(arg, torch.dtype):
        return _torch_dtype_to_onnx_dtype(arg)
    # Maybe a Python value
    return arg


def _get_onnxscript_opset(opset_version: int) -> onnxscript.values.Opset:
    return onnxscript.values.Opset("", opset_version)


def _handle_call_function_node_with_lowering(
    model: ir.Model,
    node: torch.fx.Node,
    node_name_to_values: dict[str, ir.Value | Sequence[ir.Value]],
    constant_farm: dict[Any, ir.Value],
    registry: _registration.ONNXRegistry,
    opset: onnxscript.values.Opset,
) -> None:
    if node.target == operator.getitem:
        source = node.all_input_nodes[0]
        source_outputs = node_name_to_values[source.name]
        if isinstance(source_outputs, Sequence):
            _handle_getitem_node(node, node_name_to_values)
            return
        else:
            # `source_outputs` is a sequence(tensor()) value and we need to
            # use SequenceAt to get the value. This is handled by torchlib
            pass

    # Find the matching ONNX overload for the node
    # NOTE: Create different registries for different ONNX opset versions
    # TODO: Log the message here to expose false positives
    onnx_function, message = _dispatching.dispatch(node, registry)

    if onnx_function is None:
        # TODO(justinchuby): Fall back to ATen op or do something else?
        raise _errors.DispatchError(
            f"No ONNX function found for {node.target!r}. Failure message: {message}"
        )

    # Map FX inputs to ONNX inputs and fill optional inputs.
    # torch_args and torch_kwargs are for op-level validation
    fx_args = node.args
    fx_kwargs = node.kwargs

    # Replace the input FX nodes with ONNX values
    onnx_args = [
        _convert_fx_arg_to_onnx_arg(input_, node_name_to_values) for input_ in fx_args
    ]

    onnx_kwargs = {}
    for key, value in fx_kwargs.items():
        onnx_kwargs[key] = _convert_fx_arg_to_onnx_arg(value, node_name_to_values)
        if key == "dtype" and onnx_kwargs[key] is None:
            # Set dtype to -1 if it is None
            onnx_kwargs[key] = -1

    with onnxscript.evaluator.default_as(
        tracer := _building.OpRecorder(opset, constant_farm)
    ):
        try:
            outputs = onnx_function(*onnx_args, **onnx_kwargs)
        except Exception as e:
            raise _errors.GraphConstructionError(
                f"Error when calling function '{onnx_function}' with args '{onnx_args}' and kwargs '{onnx_kwargs}'"
            ) from e

    # NOTE: Instead of using the output names from node.target._schema,
    # we always use the index if there are more than one outputs so the
    # names can be programmatically reconstructed. This is useful for
    # comparing values from the ONNX graph with those from the FX graph.
    #
    # When there are multiple outputs, the output names will be
    # node_name__0, node_name__1, etc.
    if isinstance(outputs, Sequence):
        _set_shape_types(outputs, node.meta["val"], complex_to_float=True)
        node_name_to_values[node.name] = outputs
        for i, output in enumerate(outputs):
            output.name = f"{node.name}__{i}"
    else:
        _set_shape_type(outputs, node.meta["val"], complex_to_float=True)
        node_name_to_values[node.name] = outputs
        outputs.name = node.name

    for ir_node in tracer.nodes:
        ir_node.meta["node"] = node
        # Record the nn.Module stack for the node
        _set_node_metadata(node, ir_node)

    # Add the traced nodes to the graph
    model.graph.extend(tracer.nodes)
    # Add the defined functions to the model
    for identifier, onnxscript_function in tracer.functions.items():
        if identifier in model.functions:
            continue
        # TODO: Get IR function directly when onnxscript is updated
        proto = onnxscript_function.to_function_proto()
        ir_function = ir.serde.deserialize_function(proto)
        model.functions[identifier] = ir_function
        if ir_function.domain not in model.opset_imports:
            # FIXME: Record the correct opset version of the function
            model.opset_imports[ir_function.domain] = 1


def _handle_placeholder_node(
    node: torch.fx.Node,
    node_name_to_values: dict[str, ir.Value | Sequence[ir.Value]],
    *,
    lower: str,
    opset: onnxscript.values.Opset,
) -> None:
    # Placeholder nodes are user inputs
    # We need to create a new tensor for each user input
    # and add it to the graph's inputs
    name = node.name
    input_ = _tensors.SymbolicTensor(opset, name=name)
    input_.meta["node"] = node
    _set_shape_type(input_, node.meta["val"], complex_to_float=lower != "none")
    node_name_to_values[name] = input_
    # The inputs will be added to the graph later


def _add_nodes(
    exported_program: torch.export.ExportedProgram,
    model: ir.Model,
    lower: Literal["at_conversion", "post_conversion", "none"],
    registry: _registration.ONNXRegistry,
) -> dict[str, ir.Value | Sequence[ir.Value]]:
    node_name_to_values: dict[str, ir.Value | Sequence[ir.Value]] = {}
    constant_farm: dict[Any, ir.Value] = {}
    opset = _get_onnxscript_opset(registry.opset_version)
    for node in exported_program.graph.nodes:
        logger.debug(
            "%s", (node.name, node.args, node.target, node.op, node.type, node.kwargs)
        )
        try:
            if node.op == "placeholder":
                _handle_placeholder_node(
                    node,
                    node_name_to_values,
                    lower=lower,
                    opset=opset,
                )
            elif node.op == "call_function":
                if lower == "at_conversion":
                    _handle_call_function_node_with_lowering(
                        model,
                        node,
                        node_name_to_values,
                        constant_farm,
                        registry=registry,
                        opset=opset,
                    )
                else:
                    # No lowering
                    _handle_call_function_node(model.graph, node, node_name_to_values)
        except Exception as e:
            raise _errors.ConversionError(
                f"Error when translating node {node.format_node()}. See the stack trace for more information."
            ) from e
    return node_name_to_values


def _torch_version_integer() -> int:
    return int(torch.__version__.replace(".", "").split("dev")[0])


def _get_inputs_and_attributes(
    node: torch.fx.Node,
) -> tuple[list[torch.fx.Node | None], dict[str, Any], list[str], list[str]]:
    """Find and Fill in the not provided kwargs with default values.

    Returns:
        (inputs, attributes, input_names, output_names)
    """
    if inspect.isbuiltin(node.target) or isinstance(node.target, str):
        inputs = list(node.args)
        return inputs, {}, [], [node.name]  # type: ignore[return-value]

    # The target should be an ATen operator now
    assert hasattr(
        node.target, "_schema"
    ), f"The target should be an ATen operator now, but node target {node.target} has no schema"
    node_schema: torch.FunctionSchema = node.target._schema

    # This function assumes the order of arguments in FX op is the
    # same as the order of arguments in TorchScript op.
    inputs: list[Any] = []  # type: ignore[no-redef]
    input_names: list[str] = []
    attributes: dict[str, Any] = {}

    if inspect.isbuiltin(node.target):
        inputs = list(node.args)
    else:
        for arg, schema_arg in zip(node.args, node_schema.arguments):
            if arg is None or isinstance(arg, torch.fx.Node):
                inputs.append(arg)
                input_names.append(schema_arg.name)
            elif isinstance(arg, Sequence) and all(
                elem is None or isinstance(elem, torch.fx.Node) for elem in arg
            ):
                inputs.extend(arg)
                input_names.extend([schema_arg.name] * len(arg))
            elif isinstance(arg, torch.device):
                attributes[schema_arg.name] = str(arg)
            elif isinstance(arg, torch.dtype):
                attributes[schema_arg.name] = _torch_dtype_to_onnx_dtype(arg)
            else:
                attributes[schema_arg.name] = arg
        for schema_arg in node_schema.arguments:
            if schema_arg.name not in node.kwargs:
                continue
            kwarg = node.kwargs[schema_arg.name]
            if schema_arg.name in {
                "layout",
                "device",
                "requires_grad",
                "memory_format",
                "implicit",
            } or isinstance(kwarg, torch.device):
                attr = str(kwarg)
            elif isinstance(kwarg, torch.dtype):
                attr = _torch_dtype_to_onnx_dtype(kwarg)  # type: ignore[assignment]
            else:
                attr = kwarg  # type: ignore[assignment]

            attributes[schema_arg.name] = attr

    output_names = [f"{node.name}_{output.name}" for output in node_schema.returns]

    return inputs, attributes, input_names, output_names  # type: ignore[return-value]


def _maybe_start_profiler(should_profile: bool) -> Any:
    if should_profile:
        import pyinstrument  # type: ignore[import-not-found]

        profiler = pyinstrument.Profiler(async_mode="disabled")
        profiler.start()
        return profiler
    return None


def _maybe_stop_profiler_and_get_result(profiler) -> str | None:
    if profiler is None:
        return None
    profiler.stop()
    return profiler.output_text(unicode=True)


def _format_exception(e: Exception) -> str:
    """Format the full traceback as Python would show it."""
    return "\n".join(traceback.format_exception(type(e), e, e.__traceback__))


def _summarize_exception_stack(e: BaseException) -> str:
    """Format the exception stack by showing the text of each exception."""
    causes = [e]
    while e.__cause__ is not None:
        causes.append(e.__cause__)
        e = e.__cause__
    return (
        "\n\n## Exception summary\n\n"
        + "⬆️\n".join([f"{type(e)}: {e}\n" for e in reversed(causes)])
        + "\n(Refer to the full stack trace above for more information.)"
    )


def _format_exceptions_for_all_strategies(
    results: list[_capture_strategies.Result],
) -> str:
    """Format all the exceptions from the capture strategies."""
    return "\n".join(
        [
            f"# ⚠️ Errors from strategy '{result.strategy}': -----------------------\n\n"
            f"{_format_exception(result.exception)}\n"
            for result in results
            if result.exception is not None
        ]
    )


def exported_program_to_ir(
    exported_program: torch.export.ExportedProgram,
    *,
    registry: _registration.ONNXRegistry | None = None,
    lower: Literal["at_conversion", "post_conversion", "none"] = "at_conversion",
) -> ir.Model:
    """Convert an exported program to an ONNX IR model.

    Reference:
        - ExportedProgram spec: https://pytorch.org/docs/stable/export.ir_spec.html

    Args:
        exported_program: The exported program to convert.
        lower: Whether to lower the graph to core ONNX operators.
            at_conversion: Lower whe translating the FX graph to ONNX IR.
            post_conversion: Use an IR pass to lower the graph.
            none: Do not lower the graph.
        registry: The registry of all ONNX Script decomposition.
    """
    if registry is None:
        registry = _registration.ONNXRegistry.from_torchlib()
    if lower != "none":
        exported_program = _prepare_exported_program_for_export(
            exported_program, registry=registry
        )
    return _exported_program_to_onnx_program(
        exported_program, registry=registry, lower=lower
    ).model


def _prepare_exported_program_for_export(
    exported_program: torch.export.ExportedProgram,
    *,
    registry: _registration.ONNXRegistry,
) -> torch.export.ExportedProgram:
    """Decompose and apply pre-export transformations to the exported program."""
    # Decompose the graph given the implemented torch ops in ONNX
    exported_program = _fx_passes.decompose_with_registry(exported_program, registry)

    graph_module = exported_program.graph_module
    # Include explicit type promotion nodes
    graph_module = _fx_passes.insert_type_promotion_nodes(graph_module)
    graph_module = _fx_passes.remove_assertion_nodes(graph_module)
    # TODO(justinchuby): Reassigning the graph module to save some runtime.
    # If this does not work, we need to retrace the module with torch.export
    exported_program._graph_module = graph_module
    return exported_program


def _exported_program_to_onnx_program(
    exported_program: torch.export.ExportedProgram,
    *,
    registry: _registration.ONNXRegistry,
    lower: Literal["at_conversion", "post_conversion", "none"] = "at_conversion",
) -> _onnx_program.ONNXProgram:
    """Convert an exported program to an ONNX Program.

    The exported_program field in the returned ONNXProgram is one that is after
    decompositions have been applied.

    Reference:
        - ExportedProgram spec: https://pytorch.org/docs/stable/export.ir_spec.html

    Args:
        exported_program: The exported program to convert. The exported program
            should be the one that is after decompositions have been applied.
        lower: Whether to lower the graph to core ONNX operators.
            at_conversion: Lower whe translating the FX graph to ONNX IR.
            post_conversion: Use an IR pass to lower the graph.
            none: Do not lower the graph.
        registry: The registry of all ONNX Script decomposition.
    """
    model = ir.Model(
        graph=ir.Graph(
            [],
            [],
            nodes=[],
            opset_imports={
                "": registry.opset_version,
            },
            name="main_graph",
            metadata_props={
                "pkg.torch.export.ExportedProgram.graph_signature": str(
                    exported_program.graph_signature
                ),
                "pkg.torch.export.ExportedProgram.range_constraints": str(
                    exported_program.range_constraints
                ),
            },
        ),
        ir_version=9,
        producer_name="pytorch",
        producer_version=torch.__version__,
    )

    if lower == "none":
        # Add the opset import for the torch ops
        model.opset_imports["pkg.torch.ops"] = _torch_version_integer()
    # NOTE: Function domains are added when translating nodes when lower="at_conversion"

    # 1. Add all nodes to the graph and create a dictionary of values
    values = _add_nodes(exported_program, model, lower=lower, registry=registry)

    # 2. Add user inputs and all parameters/buffers to the graph.
    # Since the node names and the tensor names are different, we need to rename
    # the nodes to match the tensor names later. For now we will just use the node names.
    user_inputs = [
        spec
        for spec in exported_program.graph_signature.input_specs
        if spec.kind == graph_signature.InputKind.USER_INPUT
    ]
    non_user_inputs = [
        spec
        for spec in exported_program.graph_signature.input_specs
        if spec.kind != graph_signature.InputKind.USER_INPUT
    ]

    for spec in itertools.chain(user_inputs, non_user_inputs):
        # Put the user inputs first and then the parameters/buffers
        if isinstance(spec.arg, graph_signature.ConstantArgument):
            logger.debug("Skipping constant argument %s", spec.arg)
            continue
        value_name = spec.arg.name
        input_kind = spec.kind
        persistent = spec.persistent
        value = values[value_name]

        assert not isinstance(
            value, Sequence
        ), f"Input '{value_name}' should not be a sequence. This is unexpected."

        value.metadata_props["pkg.torch.export.graph_signature.InputSpec.kind"] = (
            input_kind.name
        )
        value.metadata_props[
            "pkg.torch.export.graph_signature.InputSpec.persistent"
        ] = str(persistent)

        if input_kind == graph_signature.InputKind.USER_INPUT:
            # Add only user inputs to the graph
            # Subsequent passes can decide if they want to add initializers as inputs
            model.graph.inputs.append(value)
        else:
            model.graph.initializers[value_name] = value

    # 3. Add user outputs to the graph and assign metadata to all outputs
    user_outputs = [
        spec
        for spec in exported_program.graph_signature.output_specs
        if spec.kind == graph_signature.OutputKind.USER_OUTPUT
    ]
    non_user_outputs = [
        spec
        for spec in exported_program.graph_signature.output_specs
        if spec.kind != graph_signature.OutputKind.USER_OUTPUT
    ]
    for spec in itertools.chain(user_outputs, non_user_outputs):
        if isinstance(spec.arg, graph_signature.ConstantArgument):
            logger.warning("Skipping constant argument %s", spec.arg)
            continue
        value_name = spec.arg.name
        output_kind = spec.kind
        value = values[value_name]

        if not isinstance(value, (ir.Value, Sequence)):
            raise TypeError(
                f"Output '{value_name}' should be an ir.Value. Actual type is '{type(value)}': {value!r}. "
                "This may be due to an incorrect implementation of the ONNX function that produced this output."
            )

        # The output value may be a sequence, meaning the operator has multiple outputs
        _values = (value,) if not isinstance(value, Sequence) else value

        if len(_values) > 1:
            logger.warning(
                "Model output '%s' has multiple values: %s (output spec: %s). Please make sure this is expected.",
                value_name,
                _values,
                spec,
            )

        for value in _values:
            value.metadata_props["pkg.torch.export.graph_signature.OutputSpec.kind"] = (
                output_kind.name
            )
            if output_kind == graph_signature.OutputKind.USER_OUTPUT:
                model.graph.outputs.append(value)

    # 4. Rename the initializers to match the tensor names
    for name, param_name in itertools.chain(
        exported_program.graph_signature.inputs_to_parameters.items(),
        exported_program.graph_signature.inputs_to_buffers.items(),
        exported_program.graph_signature.inputs_to_lifted_tensor_constants.items(),
    ):
        initializer = model.graph.initializers.pop(name)
        initializer.name = param_name
        # Record the original name so users can search the metadata and correspond
        # with the FX graph
        initializer.metadata_props["pkg.torch.onnx.original_node_name"] = name
        model.graph.initializers[param_name] = initializer

    # 5. Add initializers to the graph
    # ExportedProgram stores parameters and buffers in state_dict,
    # but non_persistent_buffers and lifted_tensor_constants are not there
    # so we need to get them from the name_* apis.
    for name, torch_tensor in itertools.chain(
        exported_program.named_parameters(),
        exported_program.named_buffers(),
        exported_program.constants.items(),
    ):
        initializer = model.graph.initializers.get(name)  # type: ignore[assignment]
        if initializer is None:
            logger.warning("Tensor '%s' is not one of the initializers", name)
            continue
        if not isinstance(torch_tensor, torch.Tensor):
            raise NotImplementedError(
                f"Tensor '{name}' should be a torch.Tensor. Actual type is '{type(torch_tensor)}': {torch_tensor!r}. "
                "This is unexpected and not yet supported."
            )
        ir_tensor = TorchTensor(torch_tensor, name=name)
        initializer.const_value = ir_tensor
        _set_shape_type(
            initializer,
            torch_tensor,
            complex_to_float=lower != "none",
        )

    # TODO: Decide if we should keep mutated buffers as inputs/outputs

    # TODO(justinchuby): Remove the hack
    _ir_passes.add_torchlib_common_imports(model)

    return _onnx_program.ONNXProgram(model, exported_program)


def _verbose_printer(verbose: bool | None) -> Callable[..., None]:
    """Prints messages based on `verbose`."""
    if verbose is False:
        return lambda *_, **__: None
    return lambda *args, **kwargs: print("[torch.onnx]", *args, **kwargs)


def export(
    model: torch.nn.Module
    | torch.export.ExportedProgram
    | torch.fx.GraphModule
    | torch.jit.ScriptModule
    | torch.jit.ScriptFunction,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    *,
    registry: _registration.ONNXRegistry | None = None,
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = None,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    report: bool = False,
    verify: bool = False,
    profile: bool = False,
    dump_exported_program: bool = False,
    artifacts_dir: str | os.PathLike = ".",
    verbose: bool | None = None,
) -> _onnx_program.ONNXProgram:
    """Export a PyTorch model to ONNXProgram.

    Args:
        model: The model to export. This can be a PyTorch nn.Module or an ExportedProgram.
        args: The arguments to pass to the model.
        kwargs: The keyword arguments to pass to the model.
        registry: The registry of all ONNX decompositions.
        dynamic_shapes: Dynamic shapes in the graph.
        input_names: If provided, rename the inputs.
        output_names: If provided, rename the outputs.
        report: Whether to generate an error report if the export fails.
        verify: Whether to verify the ONNX model after exporting.
        profile: Whether to profile the export process. When report is True,
            the profile result will be saved in the report. Otherwise, the profile
            result will be printed.
        dump_exported_program: Whether to save the exported program to a file.
        artifacts_dir: The directory to save the exported program and error reports.
        verbose: Whether to print verbose messages. If None (default), some messages will be printed.

    Returns:
        The ONNXProgram with the exported IR graph.

    Raises:
        TorchExportError: If the export process fails with torch.export.
        ConversionError: If the ExportedProgram to ONNX translation fails.
    """
    # Set up the error reporting facilities
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    profiler = _maybe_start_profiler(profile)

    # Create the artifacts directory if it does not exist
    artifacts_dir = pathlib.Path(artifacts_dir)
    if report or profile or dump_exported_program:
        artifacts_dir.mkdir(parents=True, exist_ok=True)

    verbose_print = _verbose_printer(verbose)
    export_status = _reporting.ExportStatus()
    failed_results: list[_capture_strategies.Result] = []

    program: torch.export.ExportedProgram | None = None
    # Step 1: Export the model with torch.export.export if the model is not already an ExportedProgram
    if isinstance(model, torch.export.ExportedProgram):
        # We know the model is already exported program, so the args, kwargs, and dynamic_shapes
        # are not used.
        program = model
        export_status.torch_export = True
    else:
        # Convert an nn.Module to an ExportedProgram
        # Try everything 🐰 (all paths for getting an ExportedProgram)
        # When input is a JIT module, the last strategy will succeed so it is handled
        result: _capture_strategies.Result | None = None
        for strategy_class in _capture_strategies.CAPTURE_STRATEGIES:
            strategy = strategy_class(  # type: ignore[abstract]
                verbose=verbose is not False,  # Treat None as verbose
                dump=dump_exported_program,
                artifacts_dir=artifacts_dir,
                timestamp=timestamp,
            )
            result = strategy(model, args, kwargs, dynamic_shapes=dynamic_shapes)

            # Record the status
            if strategy_class is _capture_strategies.TorchExportStrategy:
                export_status.torch_export = result.success
            elif strategy_class is _capture_strategies.TorchExportNonStrictStrategy:
                export_status.torch_export_non_strict = result.success
            elif strategy_class is _capture_strategies.JitTraceConvertStrategy:
                export_status.torch_jit = result.success

            if result.exported_program is not None:
                program = result.exported_program
                break
            else:
                failed_results.append(result)

        assert result is not None
        if result.exported_program is None:
            # If all strategies fail, produce an error report and raise the first error
            profile_result = _maybe_stop_profiler_and_get_result(profiler)

            if report:
                report_path = artifacts_dir / _reporting.construct_report_file_name(
                    timestamp, export_status
                )

                try:
                    _reporting.create_torch_export_error_report(
                        report_path,
                        _format_exceptions_for_all_strategies(failed_results),
                        export_status=export_status,
                        profile_result=profile_result,
                    )
                except Exception as e_report:
                    verbose_print(
                        f"Failed to save error report due to an error: {e_report}"
                    )
            else:
                report_path = None

            first_error = failed_results[0].exception
            assert first_error is not None

            # NOTE: We only throw the torch.export (first) exception because we want to
            # focus on the torch.export.export error. Errors from other strategies like
            # torch.jit.trace is due to the fallback and can be confusing to users.
            # We save all errors in the error report.
            raise _errors.TorchExportError(
                _STEP_ONE_ERROR_MESSAGE
                + (
                    f"\nError report has been saved to '{report_path}'."
                    if report
                    else ""
                )
                + _summarize_exception_stack(first_error)
            ) from first_error

    assert program is not None

    if dump_exported_program:
        verbose_print("Dumping ExportedProgram because `dump_exported_program=True`...")
        program_path = artifacts_dir / f"onnx_export_{timestamp}.pt2"
        try:
            torch.export.save(program, program_path)
        except Exception as e:
            verbose_print(f"Failed to save ExportedProgram due to an error: {e}")
        else:
            verbose_print(f"ExportedProgram has been saved to '{program_path}'.")

    # Step 2: Convert the exported program to an ONNX model
    verbose_print("Translate the graph into ONNX...")

    # Step 2a: Decompose the exported program and insert type promotion nodes
    try:
        # Build the ONNX function registry
        if registry is None:
            registry = _registration.ONNXRegistry.from_torchlib()

        # Process the exported program to run decompositions and type promotions etc.
        decomposed_program = _prepare_exported_program_for_export(
            program, registry=registry
        )
    except Exception as e:
        export_status.onnx_translation = False
        verbose_print("Translate the graph into ONNX... ❌")
        profile_result = _maybe_stop_profiler_and_get_result(profiler)

        if report:
            report_path = artifacts_dir / _reporting.construct_report_file_name(
                timestamp, export_status
            )

            # Run the analysis to get the error report
            try:
                _reporting.create_onnx_export_report(
                    report_path,
                    f"{_format_exceptions_for_all_strategies(failed_results)}\n\n{_format_exception(e)}",
                    program,
                    export_status=export_status,
                    profile_result=profile_result,
                    registry=registry,
                )
            except Exception:
                logger.exception("Failed to save report due to an error.")
        else:
            report_path = None

        raise _errors.ConversionError(
            _STEP_TWO_ERROR_MESSAGE
            + (f"\nError report has been saved to '{report_path}'." if report else "")
            + _summarize_exception_stack(e)
        ) from e

    # Step 2b: Translate the decomposed program to ONNX and produce ONNXProgram
    if report or profile:
        pre_decomp_unique_ops, post_decomp_unique_ops = _analysis.compare_ops(
            program, decomposed_program
        )
    else:
        pre_decomp_unique_ops = None
        post_decomp_unique_ops = None

    try:
        # Convert the exported program to an ONNX model
        onnx_program = _exported_program_to_onnx_program(
            decomposed_program, registry=registry
        )

        # Run the ONNX passes
        if input_names:
            _ir_passes.rename_inputs(onnx_program.model, input_names)
        if output_names:
            _ir_passes.rename_outputs(onnx_program.model, output_names)

        # TODO(justinchuby): Remove the hack
        _ir_passes.add_torchlib_common_imports(onnx_program.model)

        export_status.onnx_translation = True
        verbose_print("Translate the graph into ONNX... ✅")
    except Exception as e:
        export_status.onnx_translation = False
        verbose_print("Translate the graph into ONNX... ❌")
        profile_result = _maybe_stop_profiler_and_get_result(profiler)

        if report:
            report_path = artifacts_dir / _reporting.construct_report_file_name(
                timestamp, export_status
            )

            try:
                assert pre_decomp_unique_ops is not None
                assert post_decomp_unique_ops is not None

                # Run the analysis to get the error report
                _reporting.create_onnx_export_report(
                    report_path,
                    f"{_format_exceptions_for_all_strategies(failed_results)}\n\n{_format_exception(e)}",
                    program,
                    decomp_comparison=_reporting.format_decomp_comparison(
                        pre_decomp_unique_ops, post_decomp_unique_ops
                    ),
                    export_status=export_status,
                    profile_result=profile_result,
                    registry=registry,
                )
                verbose_print(f"Export report has been saved to '{report_path}'.")
            except Exception:
                logger.exception("Failed to save report due to an error.")
        else:
            report_path = None

        raise _errors.ConversionError(
            _STEP_TWO_ERROR_MESSAGE
            + (f"\nError report has been saved to '{report_path}'." if report else "")
            + _summarize_exception_stack(e)
        ) from e

    profile_result = _maybe_stop_profiler_and_get_result(profiler)

    assert onnx_program.exported_program is not None

    if not verify:
        # Return if verification is not requested
        if report:
            try:
                assert pre_decomp_unique_ops is not None
                assert post_decomp_unique_ops is not None
                report_path = artifacts_dir / _reporting.construct_report_file_name(
                    timestamp, export_status
                )
                _reporting.create_onnx_export_report(
                    report_path,
                    "No errors"
                    if not failed_results
                    else _format_exceptions_for_all_strategies(failed_results),
                    onnx_program.exported_program,
                    decomp_comparison=_reporting.format_decomp_comparison(
                        pre_decomp_unique_ops, post_decomp_unique_ops
                    ),
                    export_status=export_status,
                    profile_result=profile_result,
                    model=onnx_program.model,
                    registry=registry,
                )
                verbose_print(f"Export report has been saved to '{report_path}'.")
            except Exception:
                logger.exception("Failed to save report due to an error.")
        elif profile and profile_result is not None:
            verbose_print("Profile result:")
            verbose_print(profile_result)
        return onnx_program

    # Step 3: (verify=True) Check the ONNX model with ONNX checker
    try:
        verbose_print("Check the ONNX model...")
        onnxscript_apis.check_model(onnx_program.model)
        export_status.onnx_checker = True
        verbose_print("Check the ONNX model... ✅")
    except Exception as e:
        export_status.onnx_checker = False
        verbose_print("Check the ONNX model... ❌")
        if report:
            try:
                assert pre_decomp_unique_ops is not None
                assert post_decomp_unique_ops is not None
                report_path = artifacts_dir / _reporting.construct_report_file_name(
                    timestamp, export_status
                )
                _reporting.create_onnx_export_report(
                    report_path,
                    f"{_format_exceptions_for_all_strategies(failed_results)}\n\n{_format_exception(e)}",
                    onnx_program.exported_program,
                    decomp_comparison=_reporting.format_decomp_comparison(
                        pre_decomp_unique_ops, post_decomp_unique_ops
                    ),
                    export_status=export_status,
                    profile_result=profile_result,
                    model=onnx_program.model,
                    registry=registry,
                )
                verbose_print(f"Export report has been saved to '{report_path}'.")
            except Exception:
                logger.exception("Failed to save report due to an error.")
        logger.warning(
            "Conversion successful but the ONNX model fails ONNX checker. "  # noqa: G004
            "Please create an issue "
            f"in the PyTorch GitHub repository against the {_BLUE}*onnx*{_END} component and "
            "attach the full error stack as well as reproduction scripts. ",
            exc_info=e,
        )
        return onnx_program

    # Step 4: (verify=True) Execute the model with ONNX Runtime
    try:
        verbose_print("Execute the model with ONNX Runtime...")
        verification_results = _verification.verify_onnx_program(onnx_program)
        verbose_print("Execute the model with ONNX Runtime... ✅")
        export_status.onnx_runtime = True
        onnx_runtime_error_message = None
    except Exception as e:
        verbose_print("Execute the model with ONNX Runtime... ❌")
        export_status.onnx_runtime = False
        onnx_runtime_error_message = _format_exception(e)
        verification_message = None

    else:
        # Step 5: (verify=True) Validate the output values
        verbose_print("Verify output accuracy...")
        export_status.output_accuracy = True
        for verification_result in verification_results:
            # TODO(justinchuby): The threshold is arbitrary right now
            if verification_result.max_abs_diff >= 5e-3:
                logger.warning(
                    "Output '%s' has a large absolute difference of %f. ",
                    verification_result.name,
                    verification_result.max_abs_diff,
                )
                export_status.output_accuracy = False
            if verification_result.max_rel_diff >= 1e-1:
                logger.warning(
                    "Output '%s' has a large relative difference of %f. ",
                    verification_result.name,
                    verification_result.max_rel_diff,
                )
                export_status.output_accuracy = False
        if export_status.output_accuracy:
            verbose_print("Verify output accuracy... ✅")
        else:
            verbose_print("Verify output accuracy... ❌")
        verification_message = _reporting.format_verification_infos(
            verification_results
        )

    if report:
        try:
            assert pre_decomp_unique_ops is not None
            assert post_decomp_unique_ops is not None

            traceback_lines = []
            if failed_results:
                traceback_lines.append(
                    _format_exceptions_for_all_strategies(failed_results)
                )
            if onnx_runtime_error_message:
                traceback_lines.append("# ⚠️ ONNX Runtime error -----------------------")
                traceback_lines.append(onnx_runtime_error_message)
            if not traceback_lines:
                traceback_lines.append("No errors")

            report_path = artifacts_dir / _reporting.construct_report_file_name(
                timestamp, export_status
            )
            _reporting.create_onnx_export_report(
                report_path,
                "\n\n".join(traceback_lines),
                onnx_program.exported_program,
                profile_result=profile_result,
                export_status=export_status,
                decomp_comparison=_reporting.format_decomp_comparison(
                    pre_decomp_unique_ops, post_decomp_unique_ops
                ),
                model=onnx_program.model,
                registry=registry,
                verification_result=verification_message,
            )
            verbose_print(f"Export report has been saved to '{report_path}'.")
        except Exception:
            logger.exception("Failed to save report due to an error.")

    # Release the inference session created during verification
    onnx_program.release()
    return onnx_program
