from typing import Any, NewType

import torch
import torch.utils._pytree as pytree

from .fake_class_registry import register_fake_class


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Note [Opaque Objects]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Opaque objects are objects that we allow to be passed into custom operators.
#
# There are two kinds of opaque types: VALUE types and REFERENCE types.
# The distinction determines how torch.compile handles the object.
#
# REFERENCE TYPES (default):
# - Represents mutable stateful objects and are treated as black boxes.
# - In torch.compile, the object is treated as an input/output to the graph.
# - Register with: register_opaque_type(MyClass)
#
# VALUE TYPES:
# - Represents constant values that are embedded in the graph
# - When torch.compiling, the object is treated as a constant and the graph
#   specializes on the constant. Guards are based on equality (obj.__eq__),
#   meaning torch.compile will recompile if the hash of the object changes.
# - The type's __eq__, __hash__, and __repr__ must be implemented.
# - Register with: register_opaque_type(MyClass) + pytree.register_constant(MyClass)
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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

_OPAQUE_TYPES: dict[Any, str] = {}


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
    return _OPAQUE_TYPES[cls]


def register_opaque_type(cls: Any) -> None:
    """
    Registers the given type as an opaque type which allows this to be consumed
    by a custom operator.

    The type name will be automatically generated from the class's fully
    qualified name (ex. my_module.MyClass).

    Args:
        cls (type): The class to register as an opaque type.
    """
    if cls in (int, str, bool, float, torch.Tensor):
        raise RuntimeError(
            f"Unable to register built-in type {cls} as an opaque type. "
            "Please wrap it in a custom class instead."
        )

    # Generate a fully qualified name by combining module and qualname
    name = f"{cls.__module__}.{cls.__qualname__}"

    _OPAQUE_TYPES[cls] = name

    torch._C._register_opaque_type(name)


def is_opaque_type(cls: Any) -> bool:
    """
    Checks if the given type is an opaque type.
    """
    if isinstance(cls, str):
        return torch._C._is_opaque_type_registered(cls)

    if cls not in _OPAQUE_TYPES:
        return False

    return torch._C._is_opaque_type_registered(_OPAQUE_TYPES[cls])


def is_opaque_value_type(cls: Any) -> bool:
    """
    Checks if the given type is an opaque **value** type.
    See Note [Opaque Objects] for more information.
    """
    if not is_opaque_type(cls):
        return False

    if isinstance(cls, str):
        for cls_type, cls_str in _OPAQUE_TYPES.items():
            if cls_str == cls:
                cls = cls_type
                return pytree.is_constant_class(cls_type)

    return pytree.is_constant_class(cls)  # pyrefly: ignore[bad-argument-type]
