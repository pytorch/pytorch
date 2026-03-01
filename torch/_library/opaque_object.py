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
through `register_opaque_type(MyClass, typ="reference")`.

VALUE TYPES:

Value-typed opaque objects represent constant values.
In torch.compile, the graph specializes on the object like how other constants
are. Therefore there are a couple of methods on the class that must be
implemented before registering it as a value-typed opaque object class:
  - __eq__: torch.compile will create guards based on the equality of this
  object, meaning that a recompilation will happen if __eq__ returns False.
  - __hash__: This must be implemented for Fake Tensor caching
  - __fx_repr__: This must be implemented to provide an evaluable representation
    for FX graph codegen. It should return a tuple of (repr_string, dict[str, type])
    where repr_string can reconstruct the object and the dict maps names used in
    repr_string to their corresponding types.

You can register a custom class as being a reference-based opaque object class
through `register_opaque_type(MyClass, typ="value")`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, NewType
from typing_extensions import TypeIs
from weakref import WeakKeyDictionary

import torch
from torch._opaque_base import OpaqueBase, OpaqueBaseMeta  # noqa: F401

from .fake_class_registry import register_fake_class


class MemberType(Enum):
    """
    Defines how a member (attribute/property/method) of an opaque object is handled
    during torch.compile tracing.
    """

    # Reads/calls the member at trace time with the real object and bakes the result as a constant
    USE_REAL = "use_real"
    # Inlines/traces the member
    INLINED = "inlined"


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


@dataclass
class _OpaqueTypeInfo:
    class_name: str
    opaque_typ: Literal["reference", "value"]
    guard_fn: Callable[
        [Any], list[Any]
    ]  # Callable that takes the object and returns list of values to guard on
    members: dict[str, MemberType]  # Maps member name to how it should be handled
    hoist: bool


# Mapping of type -> (string name, reference/value type)
_OPAQUE_TYPES: WeakKeyDictionary[Any, _OpaqueTypeInfo] = WeakKeyDictionary()
# Mapping of class_name -> (type, reference/value type)
_OPAQUE_TYPES_BY_NAME: dict[str, _OpaqueTypeInfo] = {}


def _resolve_opaque_type_info(cls: Any) -> _OpaqueTypeInfo | None:
    if cls in _OPAQUE_TYPES:
        return _OPAQUE_TYPES[cls]
    if not isinstance(cls, type):
        return None

    # Allow subclasses too
    for parent in cls.__mro__[1:]:
        if parent in _OPAQUE_TYPES:
            return _OPAQUE_TYPES[parent]
    return None


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
    info = _resolve_opaque_type_info(cls)
    if info is None:
        raise ValueError(
            f"Class {cls} is not registered as an opaque type. "
            f"Call register_opaque_type({cls.__name__}) first."
        )
    return info.class_name


def register_opaque_type(
    cls: Any,
    *,
    typ: str,
    hoist=False,
    guard_fn: Any = None,
    members: dict[str, MemberType] | None = None,
) -> None:
    """
    Registers the given type as an opaque type which allows this to be consumed
    by a custom operator.

    The type name will be automatically generated from the class's fully
    qualified name (ex. my_module.MyClass).

    Args:
        cls (type): The class to register as an opaque type.
        typ (str): Either "reference" or "value". See Note [Opaque Objects] for
            more details.
        hoist (bool): Only applies to value types. A hoist=True value type
            object is lifted as an input to the torch.compile'd graph, instead
            of being a constant baked into the graph. This is useful to
            improve compilation times in hierarchical compilation
            (e.g., change your custom ops to use hoisted strings to avoid
            baking the string into the Dynamo/AOTAutograd/FX graphs).
            This flag does nothing for reference types.
        guard_fn (callable | None): A function that takes an instance of the opaque
            object and returns a list of values to guard on. These values will be compared
            for equality on each function call, triggering recompilation if they change.
            Only applicable for reference types.
            Example: lambda obj: [obj.x, obj.y]
        members (dict[str, MemberType] | None): Dictionary mapping member names
            (attributes, properties, or methods) to their MemberType, which controls
            how they are handled during torch.compile tracing:
            - MemberType.USE_REAL: Evaluates with the real object at compile time and
              bakes the result as a constant
            - MemberType.INLINED: Inlines the method call into the trace
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

    if not isinstance(cls, OpaqueBaseMeta):
        raise TypeError(
            f"Opaque type {cls} must subclass torch._opaque_base.OpaqueBase "
            "or 'metaclass=torch._opaque_base.OpaqueBaseMeta'. "
            "This is required so that FakeScriptObject can be registered "
            "as a virtual subclass, allowing isinstance() checks to work "
            "during torch.compile tracing. "
        )

    if typ not in ["reference", "value"]:
        raise AssertionError(
            f"Opaque type must be either 'reference' or 'value', got {typ!r}"
        )

    if typ == "value":
        if cls.__eq__ is object.__eq__:  # type: ignore[comparison-overlap]
            raise TypeError(
                f"Value-type opaque object of type {cls} is "
                "expected to have a non-default `__eq__` "
                "implementation as we will use this in torch.compile "
                "to guard on the equality of objects."
            )

        # Class with a custom `__eq__` without `__hash__` won't inherit the default
        # `__hash__` from object; see https://stackoverflow.com/a/1608907.
        if cls.__hash__ is None:  # type: ignore[comparison-overlap]
            raise TypeError(
                f"Value-type opaque object of type {cls} is "
                "expected to have a non-default `__hash__` "
                "implementation as we will use this in torch.compile "
                "for FakeTensor caching."
            )

        if not hasattr(cls, "__fx_repr__"):
            raise TypeError(
                f"Value-type opaque object of type {cls} is "
                "expected to have a `__fx_repr__` method "
                "implementation as we will use this to reconstruct "
                "the object in the FX codegen. __fx_repr__ should return "
                "a tuple of (repr_string, set_of_types)."
            )

        if guard_fn is not None:
            raise TypeError(
                "No need to specify `guard_fn` for "
                f"value-type opaque class {cls} as it will be guarded based "
                "on `__eq__`."
            )

    # Generate a fully qualified name by combining module and qualname
    name = f"{cls.__module__}.{cls.__qualname__}"

    type_info = _OpaqueTypeInfo(name, typ, guard_fn, members or {}, hoist)
    _OPAQUE_TYPES[cls] = type_info
    _OPAQUE_TYPES_BY_NAME[name] = type_info

    torch._C._register_opaque_type(name)


def is_opaque_value(value: object) -> TypeIs[OpaqueType]:
    return is_opaque_type(type(value))


def should_hoist(cls: Any) -> bool:
    info = _resolve_opaque_type_info(cls)
    if info is None:
        return False
    return info.hoist


def has_members(cls: Any) -> bool:
    info = _resolve_opaque_type_info(cls)
    if info is None:
        return False
    return len(info.members) > 0


def is_opaque_type(cls: Any) -> bool:
    """
    Checks if the given type is an opaque type.
    Also returns True for subclasses of registered opaque types.
    """
    if isinstance(cls, str):
        return torch._C._is_opaque_type_registered(cls)

    info = _resolve_opaque_type_info(cls)
    if info is None:
        return False

    return torch._C._is_opaque_type_registered(info.class_name)


def is_opaque_value_type(cls: Any) -> bool:
    """
    Checks if the given type is an opaque **value** type.
    See Note [Opaque Objects] for more information.
    """
    if not is_opaque_type(cls):
        return False

    if isinstance(cls, str):
        return _OPAQUE_TYPES_BY_NAME[cls].opaque_typ == "value"

    info = _resolve_opaque_type_info(cls)
    if info is None:
        return False
    return info.opaque_typ == "value"


def is_opaque_reference_type(cls: Any) -> bool:
    """
    Checks if the given type is an opaque **reference** type.
    See Note [Opaque Objects] for more information.
    """
    if not is_opaque_type(cls):
        return False

    if isinstance(cls, str):
        return _OPAQUE_TYPES_BY_NAME[cls].opaque_typ == "reference"

    info = _resolve_opaque_type_info(cls)
    if info is None:
        return False
    return info.opaque_typ == "reference"


def get_opaque_obj_repr(obj: Any) -> tuple[str, dict[str, type]]:
    """
    Get the FX-evaluable repr for an opaque object and collect required globals.

    Objects must implement __fx_repr__() which should return:
        (repr_string, dict_mapping_name_to_type)

    where repr_string is an evaluable string representation and
    dict_mapping_name_to_type maps the names used in repr_string to their types.

    For example, if repr_string is "Foo(bar=Bar(1))", the dict should be:
        {"Foo": Foo, "Bar": Bar}
    """
    if not hasattr(obj, "__fx_repr__"):
        raise TypeError(
            f"Value-type opaque object of type {obj} is "
            "expected to have a `__fx_repr__` method "
            "implementation as we will use this to reconstruct "
            "the object in the FX codegen. __fx_repr__ should return "
            "a tuple of (repr_string, dict[str, type])."
        )

    repr_str, globals_dict = obj.__fx_repr__()

    if not isinstance(repr_str, str):
        raise TypeError(
            f"__fx_repr__ for {type(obj).__name__} must return a string as the "
            f"first element, got {type(repr_str).__name__}"
        )

    if not isinstance(globals_dict, dict):
        raise TypeError(
            f"__fx_repr__ for {type(obj).__name__} must return a dict as the "
            f"second element, got {type(globals_dict).__name__}"
        )

    return repr_str, globals_dict


def get_opaque_obj_info(cls: Any) -> _OpaqueTypeInfo | None:
    if not is_opaque_type(cls):
        return None

    if isinstance(cls, str):
        return _OPAQUE_TYPES_BY_NAME[cls]

    return _resolve_opaque_type_info(cls)


def get_member_type(cls: Any, member_name: str) -> MemberType | None:
    """
    Get the MemberType for a specific member of an opaque object class.

    Args:
        cls: The opaque object class (or its string name)
        member_name: The name of the member to query

    Returns:
        MemberType if the member is registered, None otherwise
    """
    info = get_opaque_obj_info(cls)
    if info is None:
        return None
    return info.members.get(member_name)
