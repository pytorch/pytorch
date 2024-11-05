# mypy: allow-untyped-defs
from __future__ import annotations

import logging
from typing import Callable, Sequence

from onnxscript import ir

import torch
import torch.fx
from torch.onnx._internal.exporter import _registration, _schemas


logger = logging.getLogger(__name__)

# Define utilities to convert PyTorch data types so users do not need to specify manually
_TORCH_DTYPE_TO_ONNX_COMPATIBLE: dict[torch.dtype, ir.DataType] = {
    torch.bfloat16: ir.DataType.BFLOAT16,
    torch.bool: ir.DataType.BOOL,
    torch.complex128: ir.DataType.DOUBLE,
    torch.complex64: ir.DataType.FLOAT,
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
}


def _torch_dtype_to_onnx_compatible_dtype(dtype: torch.dtype) -> ir.DataType:
    return _TORCH_DTYPE_TO_ONNX_COMPATIBLE[dtype]


def _attribute_type_compatible_with_arg(
    attr: _schemas.AttributeParameter,
    value: ir.Value | int | float | bool | Sequence[int] | Sequence[float] | None,
) -> bool:
    """Check if the attribute type is compatible with the argument."""
    if isinstance(value, bool):
        return attr.type is ir.AttributeType.INT
    if isinstance(value, str):
        return attr.type is ir.AttributeType.STRING
    if isinstance(value, int):
        return attr.type in {ir.AttributeType.INT, ir.AttributeType.FLOAT}
    if isinstance(value, float):
        return attr.type is ir.AttributeType.FLOAT
    if isinstance(value, complex):
        return False
    if isinstance(value, Sequence):
        if attr.type is ir.AttributeType.INTS:
            return all(isinstance(i, int) for i in value)
        if attr.type is ir.AttributeType.FLOATS:
            return all(isinstance(i, (int, float)) for i in value)
    if isinstance(value, torch.dtype):
        return attr.type is ir.AttributeType.INT
    if isinstance(value, (torch.device, torch.memory_format, torch.layout)):
        return attr.type is ir.AttributeType.STRING
    if value is None and not attr.required:
        # An optional attribute is not supplied
        return True
    return False


def _param_type_compatible_with_arg(
    param: _schemas.Parameter,
    value: ir.TypeProtocol
    | str
    | int
    | float
    | complex
    | Sequence[int]
    | Sequence[float]
    | None,
    assigned_types: dict[str, ir.TypeProtocol],
) -> bool:
    # Handle Python types first
    if isinstance(value, bool):  # noqa: SIM102
        if param.type_constraint.allowed_types & {ir.TensorType(ir.DataType.BOOL)}:
            return True
    if isinstance(value, int) and param.type_constraint.allowed_types & {
        ir.TensorType(ir.DataType.INT4),
        ir.TensorType(ir.DataType.INT8),
        ir.TensorType(ir.DataType.INT16),
        ir.TensorType(ir.DataType.INT32),
        ir.TensorType(ir.DataType.INT64),
        # Int inputs can be casted to a float too
        ir.TensorType(ir.DataType.FLOAT8E4M3FN),
        ir.TensorType(ir.DataType.FLOAT8E4M3FNUZ),
        ir.TensorType(ir.DataType.FLOAT8E5M2),
        ir.TensorType(ir.DataType.FLOAT8E5M2FNUZ),
        ir.TensorType(ir.DataType.FLOAT16),
        ir.TensorType(ir.DataType.FLOAT),
        ir.TensorType(ir.DataType.DOUBLE),
    }:
        return True
    if isinstance(value, float) and param.type_constraint.allowed_types & {
        ir.TensorType(ir.DataType.FLOAT8E4M3FN),
        ir.TensorType(ir.DataType.FLOAT8E4M3FNUZ),
        ir.TensorType(ir.DataType.FLOAT8E5M2),
        ir.TensorType(ir.DataType.FLOAT8E5M2FNUZ),
        ir.TensorType(ir.DataType.FLOAT16),
        ir.TensorType(ir.DataType.FLOAT),
        ir.TensorType(ir.DataType.DOUBLE),
    }:
        return True
    if isinstance(value, complex) and param.type_constraint.allowed_types & {
        ir.TensorType(ir.DataType.FLOAT),
        ir.TensorType(ir.DataType.DOUBLE),
        ir.TensorType(ir.DataType.COMPLEX64),
        ir.TensorType(ir.DataType.COMPLEX128),
    }:
        return True
    if isinstance(value, str):  # noqa: SIM102
        if param.type_constraint.allowed_types & {ir.TensorType(ir.DataType.STRING)}:
            return True
    if isinstance(value, (list, tuple)):
        if param.type_constraint.allowed_types & {
            ir.TensorType(ir.DataType.INT32),
            ir.TensorType(ir.DataType.INT64),
            ir.TensorType(ir.DataType.FLOAT),
            ir.TensorType(ir.DataType.DOUBLE),
            ir.SequenceType(ir.TensorType(ir.DataType.INT32)),
            ir.SequenceType(ir.TensorType(ir.DataType.INT64)),
            ir.SequenceType(ir.TensorType(ir.DataType.FLOAT)),
            ir.SequenceType(ir.TensorType(ir.DataType.DOUBLE)),
        } and all(isinstance(i, (int)) for i in value):
            # We will just allow any fx node and trust that the overload handles it
            return True
        if param.type_constraint.allowed_types & {
            ir.TensorType(ir.DataType.FLOAT),
            ir.TensorType(ir.DataType.DOUBLE),
            ir.SequenceType(ir.TensorType(ir.DataType.FLOAT)),
            ir.SequenceType(ir.TensorType(ir.DataType.DOUBLE)),
        } and all(isinstance(i, (int, float)) for i in value):
            # We will just allow any fx node and trust that the overload handles it
            return True
    if value is None and not param.required:
        # An optional parameter is not supplied
        return True

    if not isinstance(value, ir.TypeProtocol):
        return False

    # Then check tensor types
    if param.type_constraint.name in assigned_types:
        # If a typevar is already bound, check if the value has the same type
        assigned_type = assigned_types[param.type_constraint.name]
        return assigned_type == value
    # If the typevar is not bound, bind it to the value type
    if value in param.type_constraint.allowed_types:
        # TODO: Maybe just check dtype? Being more strict here for now
        assigned_types[param.type_constraint.name] = value
        return True
    return False


def _get_type_from_tensor(
    tensor: torch.Tensor
    | torch.SymBool
    | torch.SymInt
    | torch.SymFloat
    | Sequence[torch.Tensor],
) -> ir.TypeProtocol:
    if isinstance(tensor, torch.Tensor):
        return ir.TensorType(_torch_dtype_to_onnx_compatible_dtype(tensor.dtype))
    if isinstance(tensor, torch.SymBool):
        return ir.TensorType(ir.DataType.BOOL)
    if isinstance(tensor, torch.SymInt):
        return ir.TensorType(ir.DataType.INT64)
    if isinstance(tensor, torch.SymFloat):
        return ir.TensorType(ir.DataType.FLOAT)

    # Handle sequences
    first_tensor = next((item for item in tensor if item is not None), None)
    if first_tensor is None:
        return ir.SequenceType(ir.TensorType(ir.DataType.UNDEFINED))
    return ir.SequenceType(
        ir.TensorType(_torch_dtype_to_onnx_compatible_dtype(first_tensor.dtype))
    )


def _get_first_tensor_in_node_list(
    nodes: Sequence[torch.fx.Node | None],
) -> torch.Tensor | None:
    for node in nodes:
        if (
            node is not None
            and "val" in node.meta
            and isinstance(node.meta["val"], torch.Tensor)
        ):
            return node.meta["val"]
    return None


def _get_named_fx_node_args(node: torch.fx.Node) -> dict[str, torch.fx.node.Argument]:
    assert hasattr(node.target, "_schema")
    torch_schema: torch.FunctionSchema = node.target._schema  # type: ignore[union-attr]
    node_args = {}
    for arg, schema_arg in zip(node.args, torch_schema.arguments):
        node_args[schema_arg.name] = arg

    node_args.update(node.kwargs)
    return node_args


def get_matching_overload(
    node: torch.fx.Node,
    overloads: Sequence[Callable],
) -> tuple[Callable | None, str]:
    """Get the overload that matches the node's arguments.

    Args:
        node: The node to match.
        overloads: The overloads to match against.

    Returns:
        A tuple containing the matched overload and a string describing the reason for failure or success.
    """
    if not hasattr(node.target, "_schema"):
        # FIXME(justinchuby): When the target is a builtin, we should instead
        # Match only the inputs positionally. Figure out how to do that as right
        # now we assume all inputs are named.
        return overloads[
            0
        ], "The node target does not have a schema. Return the first one."
    named_args = _get_named_fx_node_args(node)
    # FIXME: Handle when we don't know the names of the arguments
    schema_args: dict[str, torch.Argument] = {
        arg.name: arg
        for arg in node.target._schema.arguments  # type: ignore[union-attr]
    }
    failure_messages: list[str] = []
    for overload in overloads:
        assigned_types: dict[str, ir.TypeProtocol] = {}
        fail_reason = ""
        if not hasattr(overload, "signature"):
            # When an overload does not have a signature, we assume it is a custom op and should be matched
            return (
                overload,
                "The overload does not have a signature. Assuming it is a custom op and matching it.",
            )
        for param in overload.signature:
            if param.name not in schema_args and param.required:
                # We don't need to handle variadic inputs as there is none.
                # A required parameter is not supplied.
                fail_reason = "Required parameter not supplied"
                break

            # Get the argument
            if param.name in named_args:
                # Provided in Node args
                arg = named_args[param.name]
            elif (
                param.name in schema_args
                and schema_args[param.name].has_default_value()
            ):
                # Provided in schema args
                arg = schema_args[param.name].default_value
            elif param.has_default():
                # Provided in the ONNX op definition
                arg = param.default
            else:
                fail_reason = "Parameter not provided"
                break

            if isinstance(param, _schemas.Parameter):
                if isinstance(arg, torch.Tensor):
                    arg = _get_type_from_tensor(arg)  # type: ignore[assignment]
                if isinstance(arg, (list, tuple)) and any(
                    isinstance(t, torch.fx.Node) for t in arg
                ):
                    first_tensor = _get_first_tensor_in_node_list(arg)
                    assert first_tensor is not None
                    # FIXME: Handle symfloat here
                    arg = ir.SequenceType(_get_type_from_tensor(first_tensor))  # type: ignore[assignment]
                elif isinstance(arg, torch.fx.Node):
                    meta_val = arg.meta["val"]
                    arg = _get_type_from_tensor(meta_val)  # type: ignore[assignment]
                # TODO: Handle None attributes
                # FIXME: Handle symfloat etc.
                # Handle tensors and Python values
                if not _param_type_compatible_with_arg(param, arg, assigned_types):  # type: ignore[arg-type]
                    fail_reason = (
                        f"Parameter type not compatible with argument: param=`{param}`, "
                        f"assigned_types=`{assigned_types}`, arg=`{arg}`"
                    )
                    break
            elif isinstance(param, _schemas.AttributeParameter):
                if not _attribute_type_compatible_with_arg(param, arg):  # type: ignore[arg-type]
                    fail_reason = f"Attribute type not compatible with argument: param=`{param}`, arg=`{arg}`"
                    break
        if not fail_reason:
            return overload, "Successfully matched overload"
        else:
            failure_messages.append(
                f"- Failed to match overload `{overload}`: {fail_reason}"
            )
    return (
        None,
        f"All overloads did not match the node `{node.format_node()}`.\n"
        + "\n".join(failure_messages),
    )


def _arg_has_complex_dtype(arg) -> bool:
    """Check if the node has complex dtype recursively."""
    if (
        isinstance(arg, torch.fx.Node)
        and "val" in arg.meta
        and isinstance(arg.meta["val"], torch.Tensor)
        and torch.is_complex(arg.meta["val"])
    ):
        return True
    elif isinstance(arg, list):
        return any(_arg_has_complex_dtype(item) for item in arg)
    return False


def dispatch(
    node: torch.fx.Node, registry: _registration.ONNXRegistry
) -> tuple[Callable | None, str]:
    """Dispatch a node to an ONNX function based on the node's target and the ONNX registry.

    Args:
        node: The node to dispatch.
        registry: The ONNX registry to use for dispatching.

    Returns:
        A tuple containing the matched ONNX function and a string describing the reason for failure or success.
    """
    # TODO: Handle when node does not have a target
    decomp_metas = registry.get_decomps(node.target)  # type: ignore[arg-type]
    # Determine if the node has complex inputs.
    is_complex = any(_arg_has_complex_dtype(arg) for arg in node.args) or any(
        _arg_has_complex_dtype(arg) for arg in node.kwargs.values()
    )
    if is_complex:
        decomp_metas = [decomp for decomp in decomp_metas if decomp.is_complex]
        if not decomp_metas:
            return None, "No decompositions registered for the complex-valued input"
    else:
        decomp_metas = [decomp for decomp in decomp_metas if not decomp.is_complex]
        if not decomp_metas:
            return None, "No decompositions registered for the real-valued input"

    if len(decomp_metas) == 1:
        return (
            decomp_metas[0].onnx_function,
            "Fast path: Only one decomposition is defined",
        )

    overload, message = get_matching_overload(
        node, [decomp.onnx_function for decomp in decomp_metas]
    )
    return overload, message
