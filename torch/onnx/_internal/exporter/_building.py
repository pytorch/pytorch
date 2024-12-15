"""NOTES:

We need a typing module that will handling Python to ONNX type promotion for use.
For example, if we have torch.ops.aten.add(Tensor, 1.0), we need to promote 1.0
to the same type as Tensor. The same thing needs to work for
torch.ops.aten.add(1.0, Tensor) as well, which means we need a mechanism to`
"""

# mypy: allow-untyped-defs
# mypy: disable-error-code=union-attr
from __future__ import annotations

import copy
import inspect
import logging
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING, Union

import onnxscript
from onnxscript import evaluator, ir
from onnxscript.ir import convenience as ir_convenience

import torch
from torch.onnx._internal.exporter import _errors, _schemas, _tensors


if TYPE_CHECKING:
    import onnx


logger = logging.getLogger(__name__)

ValidAttributeType = Union[
    ir.TensorProtocol, int, float, bool, str, Sequence[int], Sequence[float], None
]

AllowedArgType = Union[
    ir.Value, Sequence[Union[ir.Value, ValidAttributeType]], ValidAttributeType
]


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
                attribute = reversed_args_stack.pop()  # type: ignore[assignment, unused-ignore]
            elif param.name in kwargs:
                attribute = kwargs[param.name]  # type: ignore[assignment, unused-ignore]
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


def _determine_input_dtype(
    param: _schemas.Parameter,
    arg: AllowedArgType,
    type_binding: Mapping[_schemas.TypeConstraintParam, ir.TypeProtocol],
) -> ir.DataType:
    """Determine the dtype of the input that is a mix of Python constants and ir.Value."""
    if param.type_constraint in type_binding:
        # A known dtype is available because it was resolved
        return type_binding[param.type_constraint].dtype
    if len(param.type_constraint.allowed_types) == 1:
        # Only one type is allowed by the type constraint
        return next(iter(param.type_constraint.allowed_types)).dtype

    # No dtype information available. Infer from the Python constant or (in the Sequence case)
    # from a mix of Python constants and ir.Value
    if isinstance(arg, bool):
        return ir.DataType.BOOL
    if isinstance(arg, float):
        return ir.DataType.FLOAT
    if isinstance(arg, int):
        return ir.DataType.INT64
    if isinstance(arg, str):
        return ir.DataType.STRING
    if isinstance(arg, (ir.Tensor, ir.TensorProtocol)):
        return arg.dtype
    if isinstance(arg, complex):
        return ir.DataType.FLOAT
    if arg is None:
        return ir.DataType.UNDEFINED

    # Handle sequences
    if isinstance(arg, (tuple, list)):
        if len(arg) == 0:
            # Special case: Treat empty sequence as INT64 as they are typically used for shape
            return ir.DataType.INT64

        # Try to obtain the dtype from one of the values
        for val in arg:
            if isinstance(val, ir.Value) and val.dtype is not None:
                return val.dtype

        if any(isinstance(val, float) for val in arg):
            # If any float is present, the dtype is float
            return ir.DataType.FLOAT
        elif any(isinstance(val, int) for val in arg):
            # Otherwise if any int is present, the dtype is int
            return ir.DataType.INT64

    raise ValueError(
        f"Could not determine the dtype for the input '{param.name}'. "
        f"param={param}, arg={arg}, param_type_constraint={param.type_constraint}, "
        f"type_binding={type_binding}"
    )


def _allowed_types_are_sequence_types(allowed_types: Iterable[ir.TypeProtocol]) -> bool:
    """Check if all allowed types are Sequence types."""
    return all(isinstance(t, ir.SequenceType) for t in allowed_types)


def _get_or_create_constant(
    constant_farm: dict[
        tuple[
            bool | int | float | str | tuple[int] | tuple[float],
            ir.DataType,
        ],
        ir.Value,
    ],
    arg: bool
    | int
    | float
    | str
    | tuple[int]
    | tuple[float]
    | tuple[bool]
    | list[int]
    | list[float]
    | list[bool],
    dtype: ir.DataType,
    opset: onnxscript.values.Opset,
) -> ir.Value:
    # float representation of complex numbers
    if isinstance(arg, complex):
        # Convert the complex number to a float
        arg = (arg.real, arg.imag)

    if isinstance(arg, list):
        # Make the arg hashable
        arg = tuple(arg)  # type: ignore[assignment]

    constant_value = constant_farm.get((arg, dtype))  # type: ignore[arg-type]
    if constant_value is None:
        constant_tensor = ir.tensor(value=arg, dtype=dtype)
        constant_value = opset.Constant(value=constant_tensor)
        constant_farm[(arg, dtype)] = constant_value  # type: ignore[index]
    return constant_value


def _process_python_constants(
    signature: _schemas.OpSignature,
    named_inputs: dict[str, AllowedArgType],
    type_binding: Mapping[_schemas.TypeConstraintParam, ir.TypeProtocol],
    constant_farm: dict[
        tuple[
            bool | int | float | str | tuple[int] | tuple[float],
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
        A mapping of parameter names to Python constants converted to constant Nodes.
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
            and any(isinstance(val, ir.Value) for val in arg)
        ):
            # Skip the sequence of ir.Value. This is a variadic input or a Sequence input
            # It will be handled by _process_python_sequences
            continue
        if param.variadic:
            # Handled by _process_python_sequences
            continue
        if _allowed_types_are_sequence_types(param.type_constraint.allowed_types):
            # Handled by _process_python_sequences
            continue

        dtype = _determine_input_dtype(param, arg, type_binding)

        if arg is None:
            constant_value = None
        elif isinstance(arg, (ir.Tensor, ir.TensorProtocol)):
            constant_value = opset.Constant(value=arg)
        else:
            # Deduplicate the constants
            constant_value = _get_or_create_constant(constant_farm, arg, dtype, opset)  # type: ignore[arg-type]

        named_inputs[param.name] = constant_value
    return named_inputs  # type: ignore[return-value, unused-ignore]


def _process_python_sequences(
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
):
    """Handle three types of sequences.

    1. Variadic inputs
    2. Sequence input of ir.Value,
    3. Sequence of Python constants that contains ir.Value
    """
    for name, arg in named_inputs.items():
        param = signature.params_map[name]
        assert isinstance(
            param, _schemas.Parameter
        ), f"Expected Parameter, got {type(param)}"

        if not isinstance(arg, (tuple, list)):
            continue

        if len(arg) == 0:
            # Skip empty sequences
            continue

        # 1. Sequence input of ir.Value
        if _allowed_types_are_sequence_types(param.type_constraint.allowed_types):
            # Turn the list into a Sequence node
            # Constant op creation will be handled by the variadic case below when calling
            # the SequenceConstruct op.
            named_inputs[name] = opset.SequenceConstruct(*arg)
            continue

        # 2. Variadic inputs
        # NOTE: Variadic operators like Max can be called with mixed ir.Value and Python constants
        # like `Max(0, ir.Value())`
        # We need to convert the Python constants to Constant nodes
        if param.variadic:
            if all(isinstance(val, ir.Value) for val in arg):
                # Skip the variadic input if all values are ir.Value
                continue

            dtype = _determine_input_dtype(param, arg, type_binding)
            new_args = []
            for val in arg:
                if isinstance(val, ir.Value):
                    new_args.append(val)
                else:
                    constant_tensor = ir.tensor(value=val, dtype=dtype)
                    constant_value = opset.Constant(value=constant_tensor)
                    new_args.append(constant_value)
            named_inputs[name] = new_args
            continue
        else:
            # 3. Concat the list as a single input
            # E.g. [Value, 42] should be converted to op.Concat(Value, Constant(42))
            # when the expected input type is INT64
            # We assume this only happens for 1D cases
            if all(isinstance(val, ir.Value) for val in arg):
                named_inputs[name] = opset.Concat(*arg, axis=0)
                continue

            dtype = _determine_input_dtype(param, arg, type_binding)
            new_args = []
            for val in arg:
                if isinstance(val, ir.Value):
                    new_args.append(val)
                elif val is None:
                    # Skip None values
                    continue
                elif isinstance(val, (ir.Tensor, ir.TensorProtocol)):
                    new_args.append(opset.Constant(value=val))
                else:
                    # Turn the Python constant into 1D tensor for the constant
                    assert isinstance(
                        val, (bool, int, float)
                    ), f"Expected int or float, got {type(val)}"
                    new_args.append(
                        _get_or_create_constant(constant_farm, [val], dtype, opset)  # type: ignore[arg-type, unused-ignore]
                    )
            named_inputs[name] = opset.Concat(*new_args, axis=0)
            continue
    return named_inputs


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

    # If final inputs are None, strip them from the node inputs
    for input in reversed(inputs):
        if input is not None:
            break
        inputs.pop()

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
        version=signature.opset_version,
    )


class OpRecorder(evaluator.Evaluator):
    """An onnxscript Evaluator that captures the graph into ONNX IR."""

    def __init__(
        self, opset: onnxscript.values.Opset, constant_farm: dict[Any, ir.Value]
    ):
        self.nodes: list[ir.Node] = []
        self.opset = opset
        self.functions: dict[
            ir.OperatorIdentifier, onnxscript.OnnxFunction | ir.Function
        ] = {}
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
            converted_named_inputs = _process_python_constants(
                op_signature, named_inputs, type_binding, self.constant_farm, self.opset
            )
            converted_named_inputs = _process_python_sequences(
                op_signature,
                converted_named_inputs,  # type: ignore[arg-type, unused-ignore]
                type_binding,
                self.constant_farm,
                self.opset,
            )

        except Exception as e:
            raise _errors.GraphConstructionError(
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
            raise _errors.GraphConstructionError(
                f"Error constructing node for operator '{op_signature.domain}::{op_signature.name}'. "
                f"named_inputs={named_inputs}, converted_named_inputs={converted_named_inputs}, "
                f"named_attrs={named_attrs}, opset={self.opset}, op_signature={op_signature}."
            ) from e
        return node.outputs  # type: ignore[return-value, unused-ignore]

    def eval(
        self,
        schema: onnx.defs.OpSchema,
        args: Sequence[AllowedArgType],  # type: ignore[override, unused-ignore]
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

                if (
                    isinstance(src_input, ir.Value)
                    and isinstance(target_type, ir.Value)
                    and src_input.dtype is not None
                    and target_type.dtype is not None
                ):
                    # dtypes are available
                    if src_input.dtype == target_type.dtype:
                        # Same type. No cast needed
                        return src_input  # type: ignore[return-value, unused-ignore]
                    else:
                        # Create a Cast node
                        return self.opset.Cast(src_input, to=target_type.dtype)

            outputs = self._call_op(op_signature, named_inputs, named_attrs)
            if len(outputs) == 1:
                return outputs[0]
            return outputs
        except Exception as e:
            raise _errors.GraphConstructionError(
                f"Error calling operator '{schema.name}' with args {args} and kwargs {kwargs}."
            ) from e

    def eval_function(  # type: ignore[override, unused-ignore]
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
                    function,
                    function.function_ir.domain,
                    function.name,
                    opset_version=function.opset.version,
                )

            named_inputs, named_attrs = _construct_named_inputs_and_attrs(
                op_signature, args, kwargs
            )

            # NOTE: We need to call traceable functions after the _construct_named_inputs_and_attrs
            # call because it will filter out the unexpected kwargs for us.
            if function.traceable:
                # Trace the function call instead of adding the function as a node
                # Turn the ir.Attr objects into Python constants first
                named_attrs = {
                    name: attr.value if isinstance(attr, ir.Attr) else attr
                    for name, attr in named_attrs.items()
                }

                # Use the type binding to resolve the dtypes of the inputs, and
                # convert Python constants to Constant nodes
                type_binding = _resolve_parameter_dtypes(op_signature, named_inputs)
                try:
                    # _process_python_sequences is not here because we want to preserve python list
                    # properties for the function call
                    converted_named_inputs = _process_python_constants(
                        op_signature,
                        named_inputs,
                        type_binding,
                        self.constant_farm,
                        self.opset,
                    )

                except Exception as e:
                    raise _errors.GraphConstructionError(
                        f"Error processing Python constants for operator '{op_signature.domain}::{op_signature.name}'. "
                        f"named_inputs={named_inputs}, named_attrs={named_attrs}, opset={self.opset}, op_signature={op_signature}."
                    ) from e

                return function.function(**converted_named_inputs, **named_attrs)

            outputs = self._call_op(
                op_signature,
                named_inputs,
                named_attrs,
            )

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
            raise _errors.GraphConstructionError(
                f"Error calling function '{function.name}' with args {args} and kwargs {kwargs}."
                + f" The function is defined at '{source_file}:{lineno}'."
                if source_file
                else ""
            ) from e
