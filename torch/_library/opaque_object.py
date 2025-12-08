"""
Note [Opaque Objects]

Opaque objects are the way we allow custom operators to accept a user-defined
"black box" object as an input.

There are two kinds of opaque types: VALUE type and REFERENCE type.
The distinction determines how torch.compile handles the object.

REFERENCE TYPES (default):

Reference-typed opaque objects represent mutable stateful objects and are
treated as black boxes. In torch.compile, since torch.compile cannot optimize
the anything (including tensors) within the object, the object must be an
input to the graph.

You can register a custom class as being a reference-based opaque object class
through `register_opaque_type(MyClass)`.

VALUE TYPES:

Value-typed opaque objects represent constant values.
In torch.compile, the graph specializes on the object like how other constants
are. Therefore there are a couple of methods on the class that must be
implemented before registering it as a value-typed opaque object class:
  - __eq__: torch.compile will create guards based on the equality of this
  object, meaning that a recompilation will happen if __eq__ returns False.
  - __hash__: This must be implemented for Fake Tensor caching
  - __repr__: This must be implemented as it will be used in the FX graph's
    codegen to reconstruct the object. The string representation must be able to
    construct the object again through its __init__ method.

You can register a custom class as being a reference-based opaque object class
through `register_opaque_type(MyClass, value_type=True)`.
"""

import dataclasses
from typing import Any, NewType

import torch

from .fake_class_registry import register_fake_class


@register_fake_class("aten::OpaqueObject")
class FakeOpaqueObject:
    def __init__(self) -> None:
        pass

    @classmethod
    def __obj_unflatten__(cls, flattened_ctx: dict[str, Any]) -> None:
        raise RuntimeError(
            "FakeOpaqueObject should not be created through __obj_unflatten__ "
            "and should be special handled. Please file an issue to Github."
        )


OpaqueTypeStr = "__torch__.torch.classes.aten.OpaqueObject"

OpaqueType = NewType("OpaqueType", torch._C.ScriptObject)

# Mapping of type -> (registered string name, whether or not it is a value type)
_OPAQUE_TYPES: dict[Any, tuple[str, bool]] = {}


def get_opaque_type_name(cls: Any) -> str:
    """
    Gets the registered opaque type name for a given class.

    Args:
        cls (type): The class to get the type name for.

    Returns:
        str: The registered type name for the class.

    Raises:
        ValueError: If the class is not registered as an opaque type.
    """
    if cls not in _OPAQUE_TYPES:
        raise ValueError(
            f"Class {cls} is not registered as an opaque type. "
            f"Call register_opaque_type({cls.__name__}) first."
        )
    return _OPAQUE_TYPES[cls][0]


def register_opaque_type(cls: Any, *, value_type=False) -> None:
    """
    Registers the given type as an opaque type which allows this to be consumed
    by a custom operator.

    The type name will be automatically generated from the class's fully
    qualified name (ex. my_module.MyClass).

    Args:
        cls (type): The class to register as an opaque type.
        value_type (bool): Whether or not the opaque type is a value-type or a
            reference-type. See Note [Opaque Objects] for more details.
    """
    import torch.utils._pytree as pytree

    # Prevent registration of built-in types (int, str, list, dict, etc.) and torch.Tensor
    if cls.__module__ == "builtins" or cls is torch.Tensor:
        raise ValueError(
            f"Unable to register built-in type {cls} as an opaque type. "
            "Please wrap it in a custom class and register the custom class as opaque."
        )

    if cls in pytree.SUPPORTED_NODES:
        raise ValueError(
            f"{cls} cannot be registered as an opaque object as it has been "
            "registered as a pytree. Opaque objects must be pytree leaves."
        )

    # Generate a fully qualified name by combining module and qualname
    name = f"{cls.__module__}.{cls.__qualname__}"

    _OPAQUE_TYPES[cls] = (name, value_type)

    torch._C._register_opaque_type(name)


def is_opaque_type(cls: Any) -> bool:
    """
    Checks if the given type is an opaque type.
    """
    if isinstance(cls, str):
        return torch._C._is_opaque_type_registered(cls)

    if cls not in _OPAQUE_TYPES:
        return False

    return torch._C._is_opaque_type_registered(_OPAQUE_TYPES[cls][0])


def is_opaque_value_type(cls: Any) -> bool:
    """
    Checks if the given type is an opaque **value** type.
    See Note [Opaque Objects] for more information.
    """
    if not is_opaque_type(cls):
        return False

    if isinstance(cls, str):
        for cls_str, is_value in _OPAQUE_TYPES.values():
            if cls_str == cls:
                return is_value

    return _OPAQUE_TYPES[cls][1]


def is_opaque_reference_type(cls: Any) -> bool:
    """
    Checks if the given type is an opaque **reference** type.
    See Note [Opaque Objects] for more information.
    """
    if not is_opaque_type(cls):
        return False

    if isinstance(cls, str):
        for cls_str, is_value in _OPAQUE_TYPES.values():
            if cls_str == cls:
                return not is_value

    return not _OPAQUE_TYPES[cls][1]


def get_opaque_obj_repr(obj: Any) -> tuple[str, dict[str, type]]:
    types = {}

    def _add_type(_obj) -> str:
        obj_type = type(_obj)
        types[obj_type.__name__] = obj_type

        if obj_type.__repr__ is object.__repr__:  # type: ignore[comparison-overlap]
            raise TypeError(
                f"Value-type opaque object of type {obj_type} is "
                "expected to have to have a non-default `__repr__` "
                "implementation as we will use this to reconstruct "
                "the object in the FX codegen."
            )

        obj_repr = repr(_obj)
        try:
            import ast

            ast.parse(obj_repr, mode="eval")
        except SyntaxError as e:
            raise TypeError(
                f"Value-type opaque object of type {obj_type} does "
                "not have a valid implementation of `__repr__` "
                "which correctly reconstructs the original object. "
                f"Instead of {_obj}, it is returning {obj_repr}."
            ) from e

        return obj_repr

    # Recursively add globals for all nested opaque value types
    # that appear in the object's attributes
    def add_nested_opaque_types(_obj):
        if _obj is None:
            return

        if isinstance(_obj, (list, tuple)):
            for item in _obj:
                add_nested_opaque_types(item)
            return
        elif isinstance(_obj, dict):
            for value in _obj.values():
                add_nested_opaque_types(value)
            return
        elif is_opaque_value_type(type(_obj)):
            _add_type(_obj)

        if hasattr(_obj, "__dict__"):
            for attr_value in _obj.__dict__.values():
                add_nested_opaque_types(attr_value)
        elif hasattr(_obj, "__dataclass_fields__"):
            for field in dataclasses.fields(_obj):
                attr_value = getattr(_obj, field.name)
                add_nested_opaque_types(attr_value)

    obj_repr = _add_type(obj)
    add_nested_opaque_types(obj)
    return obj_repr, types
