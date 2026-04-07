"""
Dynamo implementations of CPython's PyObject_* default slot algorithms.

Analogous to CPython's Objects/object.c, this module holds the general
comparison dispatch machinery that is independent of any specific type.
Per-type richcompare_impl hooks live in their respective VT files.
"""

from functools import lru_cache
from typing import TYPE_CHECKING

from torch._C._dynamo import (
    get_type_slots,
    has_slot,
    PyMappingSlots,
    PyNumberSlots,
    PySequenceSlots,
    PyTypeSlots,
)

from .. import graph_break_hints
from ..exc import raise_type_error, unimplemented
from ..utils import istype
from .base import NO_SUCH_SUBOBJ, VariableTracker
from .constant import CONSTANT_VARIABLE_FALSE, CONSTANT_VARIABLE_TRUE


if TYPE_CHECKING:
    from ..symbolic_convert import InstructionTranslator


def vt_identity_compare(
    left: VariableTracker,
    right: VariableTracker,
) -> "VariableTracker | None":
    """Try to determine Python identity (left is right) at trace time.

    Returns ConstantVariable(True/False) if determinable, else None.
    Mirrors the logic in BuiltinVariable's handle_is handler.
    """
    if left is right:
        return CONSTANT_VARIABLE_TRUE

    left_val = left.get_real_python_backed_value()
    right_val = right.get_real_python_backed_value()
    left_known = left_val is not NO_SUCH_SUBOBJ
    right_known = right_val is not NO_SUCH_SUBOBJ

    if left_known and right_known:
        return (
            CONSTANT_VARIABLE_TRUE if left_val is right_val else CONSTANT_VARIABLE_FALSE
        )

    # One side has a concrete backing object, the other doesn't — they can't
    # be the same object.
    if left_known != right_known:
        return CONSTANT_VARIABLE_FALSE

    # Mutable containers created during tracing: VT identity = Python identity.
    from .dicts import ConstDictVariable
    from .lists import ListVariable

    if isinstance(left, (ConstDictVariable, ListVariable)):
        return CONSTANT_VARIABLE_FALSE

    # Different Python types can never be the same object.
    try:
        if left.python_type() is not right.python_type():
            return CONSTANT_VARIABLE_FALSE
    except NotImplementedError:
        pass

    # Different exception types are never identical.
    from .. import variables

    if (
        istype(left, variables.ExceptionVariable)
        and istype(right, variables.ExceptionVariable)
        and left.exc_type is not right.exc_type  # type: ignore[attr-defined]
    ):
        return CONSTANT_VARIABLE_FALSE

    return None


@lru_cache(maxsize=256)
def _get_cached_slots(obj_type: type) -> tuple[int, int, int, int]:
    """Get all type slots for a type (cached)."""
    return get_type_slots(obj_type)


def type_implements_sq_length(obj_type: type) -> bool:
    """Check whether obj_type implements __len__ as sequence protocol"""
    seq_slots, _, _, _ = _get_cached_slots(obj_type)
    return has_slot(seq_slots, PySequenceSlots.SQ_LENGTH)


def type_implements_mp_length(obj_type: type) -> bool:
    """Check whether obj_type implements __len__ as mapping protocol"""
    _, map_slots, _, _ = _get_cached_slots(obj_type)
    return has_slot(map_slots, PyMappingSlots.MP_LENGTH)


def type_implements_mp_subscript(obj_type: type) -> bool:
    """Check whether obj_type has tp_as_mapping->mp_subscript."""
    _, map_slots, _, _ = _get_cached_slots(obj_type)
    return has_slot(map_slots, PyMappingSlots.MP_SUBSCRIPT)


def type_implements_sq_item(obj_type: type) -> bool:
    """Check whether obj_type has tp_as_sequence->sq_item."""
    seq_slots, _, _, _ = _get_cached_slots(obj_type)
    return has_slot(seq_slots, PySequenceSlots.SQ_ITEM)


def type_implements_nb_index(obj_type: type) -> bool:
    """CPython's _PyIndex_Check: tp_as_number->nb_index != NULL."""
    _, _, num_slots, _ = _get_cached_slots(obj_type)
    return has_slot(num_slots, PyNumberSlots.NB_INDEX)


def type_implements_tp_hash(obj_type: type) -> bool:
    """Check whether obj_type has a real tp_hash (not PyObject_HashNotImplemented)."""
    _, _, _, type_slots = _get_cached_slots(obj_type)
    return has_slot(type_slots, PyTypeSlots.TP_HASH)


def maybe_get_python_type(obj: VariableTracker) -> type:
    try:
        return obj.python_type()
    except NotImplementedError:
        unimplemented(
            gb_type="Unsupported python_type() call",
            context=f"{obj} does not implement python_type()",
            explanation="This VariableTracker does not implement python_type(), "
            "which is required for object protocol operations.",
            hints=[
                *graph_break_hints.DYNAMO_BUG,
            ],
        )


def vt_mapping_size(
    tx: "InstructionTranslator", obj: "VariableTracker"
) -> "VariableTracker":
    # ref: https://github.com/python/cpython/blob/v3.13.3/Objects/abstract.c#L2308-L2330
    T = maybe_get_python_type(obj)
    if type_implements_mp_length(T):
        return obj.mp_length(tx)

    if type_implements_sq_length(T):
        raise_type_error(tx, f"{obj.python_type_name()} is not a mapping")

    raise_type_error(tx, f"object of type {obj.python_type_name()} has no len()")


def generic_len(
    tx: "InstructionTranslator", obj: "VariableTracker"
) -> "VariableTracker":
    # ref: https://github.com/python/cpython/blob/v3.13.3/Objects/abstract.c#L53-L69
    """
    Implements PyObject_Size/PyObject_Length semantics for VariableTracker objects.
    Dispatches to sq_length (sequences) or mp_length (mappings) depending on the VT type.
    """

    T = maybe_get_python_type(obj)
    if type_implements_sq_length(T):
        return obj.sq_length(tx)
    return vt_mapping_size(tx, obj)


def _try_get_python_type(obj: VariableTracker) -> type | None:
    try:
        return obj.python_type()
    except NotImplementedError:
        return None


def vt_getitem(
    tx: "InstructionTranslator",
    obj: VariableTracker,
    key: VariableTracker,
) -> VariableTracker:
    """CPython's PyObject_GetItem — dispatch to the type's mp_subscript/sq_item.

    PyObject_GetItem: https://github.com/python/cpython/blob/62a6e898e01/Objects/abstract.c#L155-L206

    CPython checks three branches in order:
      1. tp_as_mapping->mp_subscript  (L161-166)
      2. tp_as_sequence->sq_item      (L168-181) — only if key passes _PyIndex_Check
      3. PyType_Check(o)              (L183-203) — type[int] → GenericAlias/__class_getitem__

    Branch 1 is the common path (list, tuple, dict, range all have mp_subscript).
    Branch 2 fires for types with only sq_item (e.g. deque).
    Branch 3 is handled by TypingVariable.mp_subscript_impl for typing module types
    and by BuiltinVariable for builtin types like list[int].

    """
    from ..exc import raise_observed_exception

    obj_type = _try_get_python_type(obj)
    if obj_type is not None:
        # Branch 1: mp_subscript
        if type_implements_mp_subscript(obj_type):
            return obj.mp_subscript_impl(tx, key)
        # Branch 2: sq_item (only if mp_subscript is absent)
        # CPython: abstract.c L168-181 — _PyIndex_Check(key) → PyNumber_AsSsize_t → sq_item
        if type_implements_sq_item(obj_type):
            key_type = _try_get_python_type(key)
            if key_type is not None and not type_implements_nb_index(key_type):
                raise_observed_exception(
                    TypeError,
                    tx,
                    args=[
                        f"{obj_type.__name__} indices must be integers, not {key_type.__name__}"
                    ],
                )
            key = key.nb_index_impl(tx)
            return obj.sq_item_impl(tx, key)
    # Fallback for unknown types or types without slots
    return obj.mp_subscript_impl(tx, key)
