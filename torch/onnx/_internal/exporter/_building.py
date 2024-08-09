"""NOTES:

We need a typing module that will handling Python to ONNX type promotion for use.
For example, if we have torch.ops.aten.add(Tensor, 1.0), we need to promote 1.0
to the same type as Tensor. The same thing needs to work for
torch.ops.aten.add(1.0, Tensor) as well, which means we need a mechanism to`
"""

# mypy: allow-untyped-defs
from __future__ import annotations

import copy
import inspect
import logging
from typing import Any, Mapping, Sequence, TYPE_CHECKING, Union

import onnxscript
from onnxscript import evaluator, ir
from onnxscript.ir import convenience as ir_convenience

import torch
from torch.onnx._internal.exporter import _schemas, _tensors, errors


if TYPE_CHECKING:
    import onnx


logger = logging.getLogger(__name__)

# TODO(justinchuby): Update ValidAttributeType to ir_convenience.SupportedAttrTypes
ValidAttributeType = Union[
    ir.TensorProtocol, int, float, bool, str, Sequence[int], Sequence[float], None
]

AllowedArgType = Union[ir.Value, Sequence[ir.Value], ValidAttributeType]


# Logic for adapting inputs from general Python or PyTorch inputs to ONNX ir.Value
def _construct_named_inputs_and_attrs(
    signature: _schemas.OpSignature,
    args: Sequence[AllowedArgType],
    kwargs: Mapping[str, AllowedArgType],
) -> tuple[dict[str, AllowedArgType], dict[str, ValidAttributeType]]:
    """Construct two mappings: name to inputs and named to attributes based on the signature and args/kwargs.

    This function uses the OpSignature to determine which argument in args and kwargs corresponds to
    which parameter in the signature. ONNX node inputs are stored in named_inputs, and attributes are
    stored in named_attrs. If an _optional input_ is not provided, it is filled with None.

    Args:
        signature: The OpSignature for the node.
        args: The positional arguments for the node.
        kwargs: The keyword arguments for the node.

    Returns:
        A tuple of two mappings: named_inputs and named_attrs.

    Raises:
        ValueError: If a required parameter is not provided.
    """
    # 1. Construct the (named_inputs, named_attrs) mapping based on (args, kwargs) and the signature.
    #   a. Loop over all parameters in the signature and args together
    #   b. Depending on param.is_input, Record named_inputs[param.name] = arg or named_attrs[param.name] = arg
    #   c. Handle kwargs as well
    #   d. Fill in None if the input is not provided
    named_inputs: dict[str, Any] = {}
    named_attrs: dict[str, Any] = {}
    reversed_args_stack = list(reversed(args))
    for param in signature.params:
        if isinstance(param, _schemas.Parameter):
            # Handle inputs
            if reversed_args_stack:
                # First exhaust the positional arguments
                if param.variadic:
                    # Handle variadic arguments
                    named_inputs[param.name] = tuple(args)
                    reversed_args_stack.clear()
                else:
                    named_inputs[param.name] = reversed_args_stack.pop()
            elif param.name in kwargs:
                named_inputs[param.name] = kwargs[param.name]
            elif param.required:
                raise ValueError(
                    f"Required parameter '{param.name}' is not provided. "
                    f"Signature: {signature}. Args: {args}. Kwargs: {kwargs}."
                )
            else:
                logger.debug(
                    "Optional parameter '%s' is not provided. Added as None. Signature: %s",
                    param.name,
                    signature,
                )
                named_inputs[param.name] = None
        else:
            # Handle attributes
            attribute: ValidAttributeType | ir.Attr
            assert isinstance(
                param, _schemas.AttributeParameter
            ), f"Expected AttributeParameter, got {type(param)}"
            if reversed_args_stack:
                # First exhaust the positional arguments
                attribute = reversed_args_stack.pop()  # type: ignore[assignment]
            elif param.name in kwargs:
                attribute = kwargs[param.name]  # type: ignore[assignment]
            elif param.default is not None:
                attribute = param.default
            else:
                attribute = None

            if attribute is None:
                if param.required:
                    raise ValueError(
                        f"Required attribute '{param.name}' is not provided. "
                        f"Signature: {signature}. Args: {args}. Kwargs: {kwargs}."
                    )
                else:
                    logger.debug(
                        "Optional attribute '%s' is None. Dropped. Signature: %s",
                        param.name,
                        signature,
                    )
                    continue

            if isinstance(attribute, ir.Attr):
                # Turn the attribute from an default value into an actual parameter for the node
                attr_copied = copy.copy(attribute)
                # Make sure the name is the same as the parameter name and not the name of the default parameter
                attr_copied.name = param.name
                attribute = attr_copied

            if isinstance(attribute, int) and param.type == ir.AttributeType.FLOAT:
                # Convert the attribute to float if needed. This happens in PyTorch
                # where an attribute marked as float can be passed as an int.
                attribute = float(attribute)
            named_attrs[param.name] = attribute
    return named_inputs, named_attrs


def _resolve_parameter_dtypes(
    signature: _schemas.OpSignature, named_inputs: Mapping[str, AllowedArgType]
) -> Mapping[_schemas.TypeConstraintParam, ir.TypeProtocol]:
    """Determine which parameter takes which type.

    Handle non-tensor input corner cases and type promotion.

    Requires:
        All ir.Value in name_inputs should have type set. Their type should be
        compatible with the type_constraint of the corresponding parameter in the signature.

    Args:
        signature: The OpSignature for the node.
        named_inputs: The mapping of parameter names to their arguments.

    Returns:
        A mapping of Constraint names to ir.TypeProtocol.
    """
    #   a. Create type_binding: dict[str, ir.TypeProtocol]
    #   b. Iterate over all named_inputs
    #   b0. Find the corresponding parameter in the signature
    #   b1. If the argument is a Python constant, skip.
    #   b2. If the argument is a ir.Value, Bind {constraint: arg.type}.
    type_binding = {}
    for name, arg in named_inputs.items():
        param = signature.params_map[name]
        assert isinstance(
            param, _schemas.Parameter
        ), f"Expected Parameter, got {type(param)}"
        if isinstance(arg, (int, float, bool, str, Sequence, torch.Tensor)):
            # Skip the Python constants because we do not know what dtype they should take yet
            continue
        elif isinstance(arg, ir.Value):
            if arg.type is None:
                # Skip the ir.Value if the type is not set
                continue
            # NOTE: We assume arg.type is compatible with the type_constraint
            assert arg.type is not None, f"Expected type to be set for {arg}"
            # TODO(justinchuby): Implement type promotion logic here.
            type_binding[param.type_constraint] = arg.type
    return type_binding


def _process_python_constants_and_sequences(
    signature: _schemas.OpSignature,
    named_inputs: dict[str, AllowedArgType],
    type_binding: Mapping[_schemas.TypeConstraintParam, ir.TypeProtocol],
    constant_farm: dict[
        tuple[
            bool | int | float | str | ir.TensorProtocol | tuple[int] | tuple[float],
            ir.DataType,
        ],
        ir.Value,
    ],
    opset: onnxscript.values.Opset,
) -> dict[str, ir.Value | None]:
    """Convert Python constants to Constant nodes and list to Sequence nodes based on the dtype information.

    The added constants will be replacing values in named_inputs in place.

    Args:
        signature: The OpSignature for the node.
        named_inputs: The mapping of parameter names to their arguments.
        type_binding: A mapping of Constraint names to ir.DataType.
        constant_farm: A dictionary of {(py_value, ir.DataType): ir.Value} to store the deduplicated constants.
        opset: The Opset to use for creating Constant nodes.

    Returns:
        None
    """
    # 3. Convert Python constants to Constant nodes based on the dtype information;
    #    construct sequences
    #   a. Iterate over all parameters in the signature the second time
    #   b. If the parameter is in to_resolve_type:
    #       - If param.constraint in type_binding,
    #         Get the constant from constant_farm (deduplicated);
    #            otherwise set named_inputs[param.name] = Constant(value, dtype=type_binding[param.constraint])
    #       - Otherwise, set named_inputs[param.name] = Constant(value)
    for name, arg in named_inputs.items():
        param = signature.params_map[name]
        assert isinstance(
            param, _schemas.Parameter
        ), f"Expected Parameter, got {type(param)}"

        if isinstance(arg, ir.Value):
            # TODO(justinchuby): Cast the ir.Value here if needed
            continue
        if (
            isinstance(arg, Sequence)
            and len(arg) > 0
            and all(isinstance(val, ir.Value) for val in arg)
        ):
            # Skip the sequence of ir.Value. This is a variadic input or a Sequence input
            # NOTE: Variadic operators like Max can be called with mixed ir.Value and Python constants
            # like `Max(0, ir.Value())`
            # We need to convert the Python constants to Constant nodes
            # NOTE: Important to check that arg is not empty because we need to treat it as list[int] or list[float]
            continue
            # if param.variadic:
            #     # FXIME: Handle variadic inputs and sequence inputs differently
            #     raise NotImplementedError
            # TODO: Find a way to recursively build constants. Maybe extract the logic out.
            # FIXME: I am here

        assert isinstance(
            param, _schemas.Parameter
        ), f"Expected Parameter, got {type(param)}"

        if param.type_constraint in type_binding:
            # A known dtype is available
            dtype = type_binding[param.type_constraint].dtype
        elif len(param.type_constraint.allowed_types) == 1:
            # Only one type is allowed
            dtype = next(iter(param.type_constraint.allowed_types)).dtype
        else:
            # No dtype information available. Infer from the Python constant
            if isinstance(arg, bool):
                dtype = ir.DataType.BOOL
            elif isinstance(arg, float):
                dtype = ir.DataType.FLOAT
            elif isinstance(arg, int):
                dtype = ir.DataType.INT64
            elif isinstance(arg, str):
                dtype = ir.DataType.STRING
            elif isinstance(arg, (tuple, list)) and all(
                isinstance(val, int) for val in arg
            ):
                dtype = ir.DataType.INT64
            elif isinstance(arg, (tuple, list)) and any(
                isinstance(val, float) for val in arg
            ):
                # NOTE: if any float is present, the dtype is float
                dtype = ir.DataType.FLOAT
            elif isinstance(arg, (ir.Tensor, ir.TensorProtocol)):
                dtype = arg.dtype
            elif arg is None:
                dtype = ir.DataType.UNDEFINED
            else:
                raise TypeError(
                    f"Constant input '{arg}' of type '{type(arg)}' is not supported"
                )

        if arg is None:
            constant_value = None
        elif not isinstance(arg, (ir.Tensor, ir.TensorProtocol)):
            # Deduplicate the constants
            if isinstance(arg, (tuple, list)):
                # Make the arg hashable
                arg = tuple(arg)  # noqa: PLW2901
            constant_value = constant_farm.get((arg, dtype))
            if constant_value is None:
                constant_tensor = ir.tensor(value=arg, dtype=dtype)
                constant_value = opset.Constant(value=constant_tensor)
                constant_farm[(arg, dtype)] = constant_value
        else:
            constant_value = opset.Constant(value=arg)

        named_inputs[param.name] = constant_value
    return named_inputs  # type: ignore[return-type]


def _construct_node(
    signature: _schemas.OpSignature,
    named_inputs: Mapping[str, ir.Value | None],
    named_attrs: Mapping[str, ValidAttributeType],
    opset: onnxscript.values.Opset,
) -> ir.Node:
    """Construct the node with the inputs and attributes.

    Variadic inputs are flattened.

    Args:
        signature: The OpSignature for the node.
        named_inputs: The mapping of parameter names to their arguments. When we
            do not have the schema of an operator, we do not know the names of
            the inputs, in which case the names can be anything because they
            are not used in this function. The data structure is passed in for
            consistency with the other functions.
        named_attrs: The mapping of attribute names to their values.
    """
    inputs: list[ir.Value | None] = []
    # Flatten variadic inputs
    for value in named_inputs.values():
        if isinstance(value, Sequence):
            inputs.extend(value)
        else:
            inputs.append(value)

    # Construct and filter out None attributes
    attributes = [
        attr
        for attr in ir_convenience.convert_attributes(named_attrs)
        if attr.value is not None
    ]
    outputs = [_tensors.SymbolicTensor(opset) for _ in signature.outputs]
    return ir.Node(
        signature.domain,
        signature.name,
        inputs=inputs,
        attributes=attributes,
        outputs=outputs,
    )


class OpRecorder(evaluator.Evaluator):
    """An onnxscript Evaluator that captures the graph into torchscript."""

    def __init__(
        self, opset: onnxscript.values.Opset, constant_farm: dict[Any, ir.Value]
    ):
        self.nodes = []
        self.opset = opset
        self.functions: dict[ir.OperatorIdentifier, onnxscript.OnnxFunction] = {}
        self.constant_farm = constant_farm

    def _call_op(
        self,
        op_signature: _schemas.OpSignature,
        named_inputs: dict[str, AllowedArgType],
        named_attrs: dict[str, ValidAttributeType],
    ) -> Sequence[_tensors.SymbolicTensor]:
        """Record nodes for the given opschema and arguments.

        Args:
            op_signature: The OpSchema containing the node signature.
            named_inputs: The mapping of parameter names to their arguments.
            named_attrs: The mapping of attribute names to their values.
        """
        type_binding = _resolve_parameter_dtypes(op_signature, named_inputs)
        try:
            converted_named_inputs = _process_python_constants_and_sequences(
                op_signature, named_inputs, type_binding, self.constant_farm, self.opset
            )
        except Exception as e:
            raise errors.GraphConstructionError(
                f"Error processing Python constants for operator '{op_signature.domain}::{op_signature.name}'. "
                f"named_inputs={named_inputs}, named_attrs={named_attrs}, opset={self.opset}, op_signature={op_signature}."
            ) from e

        try:
            self.nodes.append(
                node := _construct_node(
                    op_signature, converted_named_inputs, named_attrs, self.opset
                )
            )
        except Exception as e:
            raise errors.GraphConstructionError(
                f"Error constructing node for operator '{op_signature.domain}::{op_signature.name}'. "
                f"named_inputs={named_inputs}, converted_named_inputs={converted_named_inputs}, "
                f"named_attrs={named_attrs}, opset={self.opset}, op_signature={op_signature}."
            ) from e
        return node.outputs  # type: ignore[return-value]

    def eval(
        self,
        schema: onnx.defs.OpSchema,
        args: Sequence[AllowedArgType],
        kwargs: Mapping[str, AllowedArgType],
    ) -> _tensors.SymbolicTensor | Sequence[_tensors.SymbolicTensor]:
        try:
            op_signature = _schemas.OpSignature.from_opschema(schema)
            named_inputs, named_attrs = _construct_named_inputs_and_attrs(
                op_signature, args, kwargs
            )
            # TODO(justinchuby): Handle cast
            if schema.name == "CastLike":
                assert len(named_inputs) == 2
                # Skip CastLike if the input and output types are the same
                src_input = named_inputs["input"]
                target_type = named_inputs["target_type"]

                dtypes_available = (
                    isinstance(src_input, ir.Value)
                    and isinstance(target_type, ir.Value)
                    and src_input.dtype is not None
                    and target_type.dtype is not None
                )
                if dtypes_available:
                    if src_input.dtype == target_type.dtype:  # type: ignore[union-attr]
                        # Same type. No cast needed
                        return src_input  # type: ignore[return-value]
                    else:
                        # Create a Cast node
                        return self.opset.Cast(src_input, to=target_type.dtype)  # type: ignore[union-attr,return-value]

            outputs = self._call_op(op_signature, named_inputs, named_attrs)
            if len(outputs) == 1:
                return outputs[0]
            return outputs
        except Exception as e:
            raise errors.GraphConstructionError(
                f"Error calling operator '{schema.name}' with args {args} and kwargs {kwargs}."
            ) from e

    def eval_function(  # type: ignore[override]
        self,
        function: onnxscript.OnnxFunction,
        args: Sequence[AllowedArgType],
        kwargs: Mapping[str, AllowedArgType],
    ) -> _tensors.SymbolicTensor | Sequence[_tensors.SymbolicTensor] | bool | int:
        try:
            # Special cases for handling IsScalar and Rank
            if function.name == "IsScalar":
                if len(args) != 1:
                    raise TypeError(
                        f"Expected 1 positional argument for function '{function}', got {len(args)}."
                    )
                if isinstance(args[0], _tensors.SymbolicTensor):
                    if args[0].rank is not None:
                        return args[0].rank == 0
                    else:
                        # Fall to call add_function_call
                        pass
                elif isinstance(args[0], Sequence):
                    return False
                else:
                    # Python constants are scalars
                    return True
            if function.name == "Rank":
                if len(args) != 1:
                    raise TypeError(
                        f"Expected 1 positional argument for function '{function}', got {len(args)}."
                    )
                if isinstance(args[0], _tensors.SymbolicTensor):
                    if args[0].rank is not None:
                        return args[0].rank
                    else:
                        # Fall to call add_function_call
                        pass
                elif isinstance(args[0], Sequence):
                    if all(isinstance(arg, (int, float)) for arg in args[0]):
                        return 1
                    else:
                        # Fall to call add_function_call
                        pass
                else:
                    # Python constants are scalars
                    return 0

            # NOTE: signature is written to function in the registration process
            # TODO: Upstream signature to ONNX Function
            if hasattr(function, "signature"):
                op_signature = function.signature
            else:
                op_signature = _schemas.OpSignature.from_function(
                    function, function.function_ir.domain, function.name
                )

            named_inputs, named_attrs = _construct_named_inputs_and_attrs(
                op_signature, args, kwargs
            )

            # NOTE: We need to call traceable functions after the _construct_named_inputs_and_attrs
            # call because it will filter out the unexpected kwargs for us.
            if function.traceable:
                # Trace the function call instead of adding the function as a node
                return function.function(**named_inputs, **named_attrs)

            outputs = self._call_op(op_signature, named_inputs, named_attrs)

            self.functions[(function.function_ir.domain, function.name, "")] = function
            if len(outputs) == 1:
                return outputs[0]
            return outputs
        except Exception as e:
            try:
                source_file = inspect.getsourcefile(function.function)
                _, lineno = inspect.getsourcelines(function.function)
            except Exception:
                source_file = lineno = None
            raise errors.GraphConstructionError(
                f"Error calling function '{function.name}' with args {args} and kwargs {kwargs}."
                + f" The function is defined at '{source_file}:{lineno}'."
                if source_file
                else ""
            ) from e
