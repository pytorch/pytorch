# mypy: allow-untyped-defs
from __future__ import annotations

import collections.abc
import dataclasses
import inspect
import logging
import types
import typing
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Optional, TypeVar, Union

import onnx

import onnxscript
from onnxscript import ir


logger = logging.getLogger(__name__)


# A special value to indicate that the default value is not specified
class _Empty:
    def __repr__(self) -> str:
        return "_EMPTY_DEFAULT"


_EMPTY_DEFAULT = _Empty()

# Map from python type to corresponding ONNX AttributeProto type
_PY_TYPE_TO_ATTR_TYPE = {
    float: ir.AttributeType.FLOAT,
    int: ir.AttributeType.INT,
    str: ir.AttributeType.STRING,
    bool: ir.AttributeType.INT,
    ir.Tensor: ir.AttributeType.TENSOR,
    ir.TensorProtocol: ir.AttributeType.TENSOR,
    ir.Graph: ir.AttributeType.GRAPH,
    ir.GraphProtocol: ir.AttributeType.GRAPH,
}

# Map from python type to corresponding ONNX AttributeProto type,
# for repeated (i.e., list of) values
_LIST_TYPE_TO_ATTR_TYPE = {
    float: ir.AttributeType.FLOATS,
    int: ir.AttributeType.INTS,
    str: ir.AttributeType.STRINGS,
    bool: ir.AttributeType.INTS,
    ir.Tensor: ir.AttributeType.TENSORS,
    ir.TensorProtocol: ir.AttributeType.TENSORS,
    ir.Graph: ir.AttributeType.GRAPHS,
    ir.GraphProtocol: ir.AttributeType.GRAPHS,
}

_ALL_VALUE_TYPES = (
    {ir.TensorType(dtype) for dtype in ir.DataType}
    | {ir.SequenceType(ir.TensorType(dtype)) for dtype in ir.DataType}
    | {ir.OptionalType(ir.TensorType(dtype)) for dtype in ir.DataType}
)

# TypeAnnotationValue represents the (value of) valid type-annotations recognized
# by ONNX Script. Currently, it supports
# - float, int, str (primitive attribute types)
# - Sequence[float], Sequence[int], Sequence[str] (attribute types)
# - Tensor types
# - Sequence[Tensor] types
# - Union of above 2
# - TypeVars with above bounds
# - Above types with annotation attached
TypeAnnotationValue = Any


@dataclasses.dataclass(frozen=True)
class TypeConstraintParam:
    """Type constraint for a parameter.

    Attributes:
        name: Name of the parameter. E.g. "TFloat"
        allowed_types: Allowed types for the parameter.
    """

    name: str
    allowed_types: set[ir.TypeProtocol]
    description: str = ""

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.allowed_types)))

    def __str__(self) -> str:
        allowed_types_str = " | ".join(str(t) for t in self.allowed_types)
        return f"{self.name}={allowed_types_str}"

    @classmethod
    def any_tensor(cls, name: str, description: str = "") -> TypeConstraintParam:
        return cls(name, {ir.TensorType(dtype) for dtype in ir.DataType}, description)

    @classmethod
    def any_value(cls, name: str, description: str = "") -> TypeConstraintParam:
        return cls(name, _ALL_VALUE_TYPES, description)  # type: ignore[arg-type]


@dataclasses.dataclass(frozen=True)
class Parameter:
    """A formal parameter of an operator."""

    name: str
    type_constraint: TypeConstraintParam
    required: bool
    variadic: bool
    default: Any = _EMPTY_DEFAULT
    # TODO: Add other properties too

    def __str__(self) -> str:
        type_str = self.type_constraint.name
        if self.has_default():
            return f"{self.name}: {type_str} = {self.default}"
        return f"{self.name}: {type_str}"

    def has_default(self) -> bool:
        return self.default is not _EMPTY_DEFAULT


@dataclasses.dataclass(frozen=True)
class AttributeParameter:
    """A parameter in the function signature that represents an ONNX attribute."""

    name: str
    type: ir.AttributeType
    required: bool
    default: ir.Attr | None = None

    def __str__(self) -> str:
        type_str = self.type.name
        if self.has_default():
            return f"{self.name}: {type_str} = {self.default}"
        return f"{self.name}: {type_str}"

    def has_default(self) -> bool:
        return self.default is not None


def _get_type_from_str(
    type_str: str,
) -> ir.TensorType | ir.SequenceType | ir.OptionalType:
    """Converter a type_str from ONNX Opschema to ir.TypeProtocol.

    A type str has the form of "tensor(float)" or composite type like "seq(tensor(float))".
    """

    # TODO: Upstream this to IR

    # Split the type_str a sequence types and dtypes
    # 1. Remove the ending ")"
    striped = type_str.rstrip(")")
    # 2. Split the type_str by "("
    type_parts = striped.split("(")

    # Convert the dtype to ir.DataType
    dtype = ir.DataType[type_parts[-1].upper()]

    # Create a place holder type first
    type_: ir.TypeProtocol = ir.TensorType(ir.DataType.UNDEFINED)

    # Construct the type
    for type_part in reversed(type_parts[:-1]):
        if type_part == "tensor":
            type_ = ir.TensorType(dtype)
        elif type_part == "seq":
            type_ = ir.SequenceType(type_)
        elif type_part == "optional":
            type_ = ir.OptionalType(type_)
        else:
            raise ValueError(f"Unknown type part: '{type_part}' in type '{type_str}'")
    return type_  # type: ignore[return-value]


def _convert_formal_parameter(
    param: onnx.defs.OpSchema.FormalParameter,
    type_constraints: Mapping[str, TypeConstraintParam],
) -> Parameter:
    """Convert a formal parameter from ONNX Opschema to Parameter."""
    if param.type_str in type_constraints:
        type_constraint = type_constraints[param.type_str]
    else:
        # param.type_str can be a plain type like 'int64'.
        type_constraint = TypeConstraintParam(
            name=param.name,
            allowed_types={_get_type_from_str(param.type_str)},
        )
    return Parameter(
        name=param.name,
        type_constraint=type_constraint,
        required=param.option != onnx.defs.OpSchema.FormalParameterOption.Optional,
        variadic=param.option == onnx.defs.OpSchema.FormalParameterOption.Variadic,
    )


def _is_optional(type_: type) -> bool:
    """Returns whether a type_ is an Optional."""
    origin_type = typing.get_origin(type_)
    if origin_type is Union and type(None) in typing.get_args(type_):
        # Python < 3.10
        return True
    if origin_type is Optional:
        # Python >= 3.10
        return True
    if (
        hasattr(types, "UnionType")
        and origin_type is types.UnionType
        and type(None) in typing.get_args(type_)
    ):
        # Python >= 3.10
        return True
    return False


def _get_attr_type(type_: type) -> ir.AttributeType:
    """Obtain the type of the attribute from a Python class."""
    try:
        if type_ in _PY_TYPE_TO_ATTR_TYPE:
            return _PY_TYPE_TO_ATTR_TYPE[type_]
        origin_type = typing.get_origin(type_)
        if origin_type is None:
            return ir.AttributeType.UNDEFINED
        if origin_type in (
            collections.abc.Sequence,
            Sequence,
            list,
            list,
            tuple,
            tuple,
        ):
            inner_type = typing.get_args(type_)[0]
            if inner_type in _LIST_TYPE_TO_ATTR_TYPE:
                return _LIST_TYPE_TO_ATTR_TYPE[inner_type]
    except TypeError:
        logger.warning("TypeError when checking %s.", type_, exc_info=True)
    return ir.AttributeType.UNDEFINED


def _get_type_constraint_name(type_: TypeAnnotationValue) -> str | None:
    """Returns the name of the type constraint for a given type annotation.

    Args:
        type_: A Python type.

    Returns:
        The name of the type constraint if it is a TypeVar.
        - Prefixes the name with "Sequence_" if the type annotation is a Sequence[].
    """
    if isinstance(type_, TypeVar):
        return type_.__name__
    if _is_optional(type_):
        subtypes = typing.get_args(type_)
        for subtype in subtypes:
            if subtype is type(None):
                continue
            type_param_name = _get_type_constraint_name(subtype)
            return type_param_name if type_param_name else None
    origin_type = typing.get_origin(type_)
    if isinstance(origin_type, type) and issubclass(origin_type, Sequence):
        subtypes = typing.get_args(type_)
        type_param_name = _get_type_constraint_name(subtypes[0])
        return f"Sequence_{type_param_name}" if type_param_name else None
    return None


def _get_allowed_types_from_type_annotation(
    type_: TypeAnnotationValue,
) -> set[ir.TypeProtocol]:
    """Obtain the allowed types from a type annotation."""
    if type_ is onnxscript.onnx_types.TensorType:
        # Any tensor type
        return {ir.TensorType(dtype) for dtype in ir.DataType}

    allowed_types: set[ir.TypeProtocol]

    if isinstance(type_, TypeVar):
        allowed_types = set()
        if constraints := type_.__constraints__:
            for constraint in constraints:
                allowed_types.update(
                    _get_allowed_types_from_type_annotation(constraint)
                )
        else:
            bound = type_.__bound__
            if bound is None:
                allowed_types = _ALL_VALUE_TYPES  # type: ignore[assignment]
            else:
                allowed_types.update(_get_allowed_types_from_type_annotation(bound))
        return allowed_types
    if hasattr(type_, "dtype"):
        # A single tensor type like INT64, FLOAT, etc.
        return {ir.TensorType(ir.DataType(type_.dtype))}
    if _is_optional(type_):
        allowed_types = set()
        subtypes = typing.get_args(type_)
        for subtype in subtypes:
            if subtype is type(None):
                continue
            allowed_types.update(_get_allowed_types_from_type_annotation(subtype))
        # NOTE: We do not consider dynamic optional types like optional(float) because they are not very useful.
        return allowed_types

    origin_type = typing.get_origin(type_)
    if origin_type is Union:
        allowed_types = set()
        subtypes = typing.get_args(type_)
        for subtype in subtypes:
            assert subtype is not type(None), (
                "Union should not contain None type because it is handled by _is_optional."
            )
            allowed_types.update(_get_allowed_types_from_type_annotation(subtype))
        return allowed_types

    if isinstance(origin_type, type) and issubclass(origin_type, Sequence):
        subtypes = typing.get_args(type_)
        return {
            ir.SequenceType(t)
            for t in _get_allowed_types_from_type_annotation(subtypes[0])
        }

    # Allow everything by default
    return _ALL_VALUE_TYPES  # type: ignore[return-value]


@dataclasses.dataclass
class OpSignature:
    """Schema for an operator.

    Attributes:
        domain: Domain of the operator. E.g. "".
        name: Name of the operator. E.g. "Add".
        overload: Overload name of the operator.
        params: Input parameters. When the op is an ONNX function definition,
          the order is according to the function signature. This mean we can
          interleave ONNX inputs and ONNX attributes in the list.
        outputs: Output parameters.
    """

    domain: str
    name: str
    overload: str
    params: Sequence[Parameter | AttributeParameter]
    outputs: Sequence[Parameter]
    params_map: Mapping[str, Parameter | AttributeParameter] = dataclasses.field(
        init=False, repr=False
    )
    opset_version: int | None = None

    def __post_init__(self):
        self.params_map = {param.name: param for param in self.params}

    def get(self, name: str) -> Parameter | AttributeParameter:
        return self.params_map[name]

    def __contains__(self, name: str) -> bool:
        return name in self.params_map

    def __iter__(self) -> Iterator[Parameter | AttributeParameter]:
        return iter(self.params)

    def __str__(self) -> str:
        domain = self.domain or "''"
        # TODO: Double check the separator for overload
        overload = f"::{self.overload}" if self.overload else ""
        params = ", ".join(str(param) for param in self.params)
        outputs = ", ".join(str(param.type_constraint.name) for param in self.outputs)
        type_constraints = {}
        for param in self.params:
            if isinstance(param, Parameter):
                type_constraints[param.type_constraint.name] = param.type_constraint
        for param in self.outputs:
            type_constraints[param.type_constraint.name] = param.type_constraint
        type_constraints_str = ", ".join(
            str(type_constraint) for type_constraint in type_constraints.values()
        )
        return f"{domain}::{self.name}{overload}({params}) -> ({outputs}) where {type_constraints_str}"

    @classmethod
    def from_opschema(cls, opschema: onnx.defs.OpSchema) -> OpSignature:
        """Produce an OpSignature from an ONNX Opschema."""
        type_constraints = {
            constraint.type_param_str: TypeConstraintParam(
                name=constraint.type_param_str,
                allowed_types={
                    _get_type_from_str(type_str)
                    for type_str in constraint.allowed_type_strs
                },
                description=constraint.description,
            )
            for constraint in opschema.type_constraints
        }

        params = [
            _convert_formal_parameter(param, type_constraints)
            for param in opschema.inputs
        ]

        for param in opschema.attributes.values():
            default_attr = (
                ir.serde.deserialize_attribute(param.default_value)
                if param.default_value is not None
                else None
            )
            if default_attr is not None:
                # Set the name of the default attribute because it may have a different name from the parameter
                default_attr.name = param.name
            params.append(
                # pyrefly: ignore [bad-argument-type]
                AttributeParameter(
                    name=param.name,
                    type=ir.AttributeType(param.type),  # type: ignore[arg-type]
                    required=param.required,
                    default=default_attr,  # type: ignore[arg-type]
                )
            )

        outputs = [
            _convert_formal_parameter(param, type_constraints)
            for param in opschema.outputs
        ]

        return cls(
            domain=opschema.domain,
            name=opschema.name,
            overload="",
            params=params,
            outputs=outputs,
            opset_version=opschema.since_version,
        )

    @classmethod
    def from_function(
        cls,
        func,
        domain: str,
        name: str | None = None,
        overload: str = "",
        *,
        opset_version: int = 1,
    ) -> OpSignature:
        """Produce an OpSignature from a function using type annotation."""

        py_signature = inspect.signature(func)
        # Not using inspect.get_annotations because typing.get_type_hints seems to handle more cases
        # https://github.com/python/cpython/issues/102405
        type_hints = typing.get_type_hints(func)

        params: list[Parameter | AttributeParameter] = []
        # Create a mapping from type to a unique name
        type_constraints: dict[str, TypeConstraintParam] = {}

        for param in py_signature.parameters.values():
            if param.name not in type_hints:
                logger.debug(
                    "Missing annotation for parameter '%s' from %s. Treating as an Input.",
                    param.name,
                    py_signature,
                )
                type_constraint = TypeConstraintParam.any_value(f"T_{param.name}")
                type_constraints[param.name] = type_constraint
                params.append(
                    Parameter(
                        name=param.name,
                        type_constraint=type_constraint,
                        required=param.default is inspect.Parameter.empty,
                        # TODO: Handle variadic
                        variadic=False,
                        default=param.default
                        if param.default is not inspect.Parameter.empty
                        else _EMPTY_DEFAULT,
                    )
                )
            else:
                type_ = type_hints[param.name]
                if (attr_type := _get_attr_type(type_)) != ir.AttributeType.UNDEFINED:
                    # Construct the default attribute
                    if param.default is not inspect.Parameter.empty:
                        # TODO: Use ir_convenience instead to handle int as float
                        default = ir.Attr(param.name, attr_type, param.default)
                    else:
                        default = None
                    params.append(
                        AttributeParameter(
                            name=param.name,
                            type=attr_type,
                            required=param.default is inspect.Parameter.empty,
                            default=default,
                        )
                    )
                else:
                    # Obtain the type constraint from the type annotation

                    # 1. Get a type constraint name from the type annotation
                    # If the type annotation is a TypeVar or Optional[TypeVar], get its name
                    # Otherwise, name it T_{param.name}
                    type_constraint_name = _get_type_constraint_name(type_)
                    if type_constraint_name is None:
                        type_constraint_name = f"T_{param.name}"

                    # 2. If the type constraint param is already initialized, use it
                    if type_constraint_name in type_constraints:
                        type_constraint = type_constraints[type_constraint_name]
                    else:
                        # 3. Otherwise, create a new TypeConstraintParam
                        type_constraint = TypeConstraintParam(
                            name=type_constraint_name,
                            allowed_types=_get_allowed_types_from_type_annotation(
                                type_
                            ),
                        )
                        type_constraints[type_constraint_name] = type_constraint
                    # 4. Create Parameter
                    params.append(
                        Parameter(
                            name=param.name,
                            type_constraint=type_constraint,
                            required=param.default is inspect.Parameter.empty,
                            # TODO: Handle variadic
                            variadic=False,
                            default=param.default
                            if param.default is not inspect.Parameter.empty
                            else _EMPTY_DEFAULT,
                        )
                    )

        return_type = type_hints.get("return")

        outputs = []
        if return_type is None:
            # No returns
            pass
        else:
            if typing.get_origin(return_type) is tuple:
                # Multiple returns
                return_types = typing.get_args(return_type)
            else:
                return_types = [return_type]  # type: ignore[assignment]

            for i, return_type_i in enumerate(return_types):
                if (
                    return_param_name := _get_type_constraint_name(return_type_i)
                ) in type_constraints:
                    # pyrefly: ignore [bad-index, index-error]
                    type_constraint = type_constraints[return_param_name]
                else:
                    return_param_name = f"TReturn{i}"
                    type_constraint = TypeConstraintParam(
                        name=return_param_name,
                        allowed_types=_get_allowed_types_from_type_annotation(
                            return_type_i
                        ),
                    )
                    type_constraints[return_param_name] = type_constraint
                outputs.append(
                    Parameter(
                        # pyrefly: ignore [bad-argument-type]
                        name=return_param_name,
                        type_constraint=type_constraint,
                        required=True,
                        variadic=False,
                        default=_EMPTY_DEFAULT,
                    )
                )

        return cls(
            domain=domain,
            name=name or func.__name__,
            overload=overload,
            params=params,
            outputs=outputs,
            opset_version=opset_version,
        )
