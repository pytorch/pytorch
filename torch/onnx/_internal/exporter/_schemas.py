# mypy: allow-untyped-defs
"""Helpers for constructing ONNX operator signatures from Python functions."""

from __future__ import annotations

import collections.abc
import inspect
import logging
import types
import typing
from collections.abc import Sequence
from typing import Any, Optional, TypeVar, Union

from torch.onnx._internal._lazy_import import onnx_ir as ir, onnxscript


logger = logging.getLogger(__name__)


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
            if subtype is type(None):
                raise AssertionError(
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


def op_signature_from_function(
    func,
    domain: str,
    name: str | None = None,
    overload: str = "",
    *,
    since_version: int = 1,
) -> ir.schemas.OpSignature:
    """Produce an OpSignature from a function using type annotation."""

    py_signature = inspect.signature(func)
    # Not using inspect.get_annotations because typing.get_type_hints seems to handle more cases
    # https://github.com/python/cpython/issues/102405
    type_hints = typing.get_type_hints(func)

    params: list[ir.schemas.Parameter | ir.schemas.AttributeParameter] = []
    # Create a mapping from type to a unique name
    type_constraints: dict[str, ir.schemas.TypeConstraintParam] = {}

    for param in py_signature.parameters.values():
        if param.name not in type_hints:
            logger.debug(
                "Missing annotation for parameter '%s' from %s. Treating as an Input.",
                param.name,
                py_signature,
            )
            type_constraint = ir.schemas.TypeConstraintParam.any_value(
                f"T_{param.name}"
            )
            type_constraints[param.name] = type_constraint
            kwargs: dict[str, Any] = {}
            if param.default is not inspect.Parameter.empty:
                kwargs["default"] = param.default
            params.append(
                ir.schemas.Parameter(
                    name=param.name,
                    type_constraint=type_constraint,
                    required=param.default is inspect.Parameter.empty,
                    # TODO: Handle variadic
                    variadic=False,
                    **kwargs,
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
                    ir.schemas.AttributeParameter(
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
                    type_constraint = ir.schemas.TypeConstraintParam(
                        name=type_constraint_name,
                        allowed_types=_get_allowed_types_from_type_annotation(type_),
                    )
                    type_constraints[type_constraint_name] = type_constraint
                # 4. Create Parameter
                kwargs: dict[str, Any] = {}
                if param.default is not inspect.Parameter.empty:
                    kwargs["default"] = param.default
                params.append(
                    ir.schemas.Parameter(
                        name=param.name,
                        type_constraint=type_constraint,
                        required=param.default is inspect.Parameter.empty,
                        # TODO: Handle variadic
                        variadic=False,
                        **kwargs,
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
                # pyrefly: ignore [bad-index]
                type_constraint = type_constraints[return_param_name]
            else:
                return_param_name = f"TReturn{i}"
                type_constraint = ir.schemas.TypeConstraintParam(
                    name=return_param_name,
                    allowed_types=_get_allowed_types_from_type_annotation(
                        return_type_i
                    ),
                )
                type_constraints[return_param_name] = type_constraint
            outputs.append(
                ir.schemas.Parameter(
                    name=return_param_name,
                    type_constraint=type_constraint,
                    required=True,
                    variadic=False,
                )
            )

    return ir.schemas.OpSignature(
        domain=domain,
        name=name or func.__name__,
        overload=overload,
        params=params,
        outputs=outputs,
        since_version=since_version,
    )
