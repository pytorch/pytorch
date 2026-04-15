"""
Dynamo implementations of CPython's PyObject_* default slot algorithms.

Analogous to CPython's Objects/object.c, this module holds the general
dispatch machinery that is independent of any specific type.
Per-type hook implementations (bool_impl, richcompare_impl, etc.)
live in their respective VT files.
"""

from functools import lru_cache
from typing import TYPE_CHECKING

from torch._C._dynamo import (
    get_type_slots,
    has_slot,
    PyMappingSlots,
    PyNumberSlots,
    PySequenceSlots,
)

from .. import graph_break_hints
from ..exc import (
    handle_observed_exception,
    ObservedTypeError,
    raise_observed_exception,
    raise_type_error,
    unimplemented,
)
from ..utils import istype
from .base import NO_SUCH_SUBOBJ, VariableTracker
from .constant import ConstantVariable


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
        return ConstantVariable.create(True)

    left_val = left.get_real_python_backed_value()
    right_val = right.get_real_python_backed_value()
    left_known = left_val is not NO_SUCH_SUBOBJ
    right_known = right_val is not NO_SUCH_SUBOBJ

    if left_known and right_known:
        return (
            ConstantVariable.create(True)
            if left_val is right_val
            else ConstantVariable.create(False)
        )

    # One side has a concrete backing object, the other doesn't — they can't
    # be the same object.
    if left_known != right_known:
        return ConstantVariable.create(False)

    # Mutable containers created during tracing: VT identity = Python identity.
    from .dicts import ConstDictVariable
    from .lists import ListVariable
    from .sets import SetVariable

    if isinstance(left, (ConstDictVariable, ListVariable, SetVariable)):
        return ConstantVariable.create(False)

    # Different Python types can never be the same object.
    try:
        if left.python_type() is not right.python_type():
            return ConstantVariable.create(False)
    except NotImplementedError:
        pass

    # Different exception types are never identical.
    from .. import variables

    if (
        istype(left, variables.ExceptionVariable)
        and istype(right, variables.ExceptionVariable)
        and left.exc_type is not right.exc_type  # type: ignore[attr-defined]
    ):
        return ConstantVariable.create(False)

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


def type_implements_nb_bool(obj_type: type) -> bool:
    """Check whether obj_type implements the nb_bool slot (i.e. has __bool__ or __len__)."""
    _, _, number_slots, _ = _get_cached_slots(obj_type)
    return has_slot(number_slots, PyNumberSlots.NB_BOOL)


def type_implements_nb_int(obj_type: type) -> bool:
    """Check whether obj_type implements the nb_int slot."""
    _, _, number_slots, _ = _get_cached_slots(obj_type)
    return has_slot(number_slots, PyNumberSlots.NB_INT)


def type_implements_nb_index(obj_type: type) -> bool:
    """Check whether obj_type implements the nb_index slot."""
    _, _, number_slots, _ = _get_cached_slots(obj_type)
    return has_slot(number_slots, PyNumberSlots.NB_INDEX)


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


def generic_bool(tx: "InstructionTranslator", obj: VariableTracker) -> VariableTracker:
    """Mirrors PyObject_IsTrue.

    https://github.com/python/cpython/blob/c09ccd9c429/Objects/object.c#L2135-L2158

    Resolution order: constants → nb_bool → mp_length/sq_length → truthy.
    """
    from .constant import ConstantVariable

    if obj.is_python_constant():
        return ConstantVariable.create(bool(obj.as_python_constant()))

    obj_type = maybe_get_python_type(obj)

    if type_implements_nb_bool(obj_type):
        result = obj.bool_impl(tx)
        if result is not None:
            return result

    try:
        length = generic_len(tx, obj)
        from .tensor import SymNodeVariable

        if isinstance(length, SymNodeVariable):
            return SymNodeVariable.create(tx, length.as_proxy() > 0)
        return ConstantVariable.create(length.as_python_constant() > 0)
    except ObservedTypeError:
        handle_observed_exception(tx)

    return ConstantVariable.create(True)


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
    TODO(follow-up): use has_slot(map_slots, PyMappingSlots.MP_SUBSCRIPT) to gate
    Branch 1 and has_slot(seq_slots, PySequenceSlots.SQ_ITEM) to gate Branch 2,
    matching CPython's dispatch order.
    TODO(follow-up): Branch 2 (sq_item) for C extension types that only have
    tp_as_sequence (e.g. deque — Modules/_collectionsmodule.c:1888).
    Branch 3 is handled by TypingVariable.mp_subscript_impl for typing module types
    and by BuiltinVariable for builtin types like list[int].

    Types that work via constant fold fallback (no dedicated mp_subscript_impl):
    TODO(follow-up): str (unicode_subscript, Objects/unicodeobject.c:13809)
    TODO(follow-up): bytes (bytes_subscript, Objects/bytesobject.c)
    """
    return obj.mp_subscript_impl(tx, key)


def generic_int(tx: "InstructionTranslator", obj: VariableTracker) -> VariableTracker:
    """Mirrors PyNumber_Long (int(x) dispatch).

    https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1520-L1632

    Resolution: nb_int → nb_index → str/bytes/bytearray parsing → TypeError.
    """
    from .constant import ConstantVariable

    # Fast path for int (sub)class instances — mirrors PyLong_Check at the
    # top of PyNumber_Long (abstract.c:1531). Avoids infinite recursion for
    # int subclasses like IntEnum whose __int__ calls int() again.
    if obj.is_python_constant() and isinstance(obj.as_python_constant(), int):
        return ConstantVariable.create(int(obj.as_python_constant()))

    obj_type = maybe_get_python_type(obj)

    if type_implements_nb_int(obj_type):
        return obj.nb_int_impl(tx)

    if type_implements_nb_index(obj_type):
        return obj.nb_index_impl(tx)

    # String/bytes/bytearray parsing fallback.
    # https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1598-L1612
    if obj.is_python_constant() and isinstance(
        obj.as_python_constant(), (str, bytes, bytearray)
    ):
        try:
            return ConstantVariable.create(int(obj.as_python_constant()))
        except ValueError as e:
            raise_observed_exception(ValueError, tx, args=[str(e)])

    raise_type_error(
        tx,
        f"int() argument must be a string, a bytes-like object "
        f"or a real number, not '{obj.python_type_name()}'",
    )
