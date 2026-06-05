"""
Dynamo implementations of CPython's PyObject_* default slot algorithms.

Analogous to CPython's Objects/object.c, this module holds the general
dispatch machinery that is independent of any specific type.
Per-type hook implementations (bool_impl, richcompare_impl, etc.)
live in their respective VT files.
"""

import abc
import collections
import enum
import sys
import types
import typing
from functools import lru_cache, partial
from typing import NoReturn, TYPE_CHECKING

import torch
from torch._C._dynamo import (
    get_type_slots,
    has_slot,
    PyMappingSlots,
    PyNumberSlots,
    PySequenceSlots,
    PyTypeSlots,
)

from .. import graph_break_hints, polyfills
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
    from ..symbolic_convert import InstructionTranslatorBase


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

    # Objects created during tracing: VT identity = Python identity.
    from .dicts import ConstDictVariable
    from .lists import ListVariable
    from .misc import TracebackVariable
    from .sets import SetVariable

    if isinstance(
        left, (ConstDictVariable, ListVariable, SetVariable, TracebackVariable)
    ):
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


def binop_type_error(
    tx: "InstructionTranslatorBase",
    v: VariableTracker,
    w: VariableTracker,
    op_symbol: str,
) -> NoReturn:
    raise_type_error(
        tx,
        f"unsupported operand type(s) for {op_symbol}: '{v.python_type_name()}' and '{w.python_type_name()}'",
    )


@lru_cache(maxsize=256)
def _get_cached_slots(obj_type: type) -> tuple[int, int, int, int]:
    """Get all type slots for a type (cached)."""
    return get_type_slots(obj_type)


def type_implements_sq_slot(obj_type: type, slot: int) -> bool:
    """Check whether obj_type implements the given sq slot."""
    seq_slots, _, _, _ = _get_cached_slots(obj_type)
    return has_slot(seq_slots, slot)


def type_implements_mp_slot(obj_type: type, slot: int) -> bool:
    """Check whether obj_type implements the given mp slot."""
    _, map_slots, _, _ = _get_cached_slots(obj_type)
    return has_slot(map_slots, slot)


# PySequenceSlots
type_implements_sq_item = partial(type_implements_sq_slot, slot=PySequenceSlots.SQ_ITEM)
type_implements_sq_length = partial(
    type_implements_sq_slot, slot=PySequenceSlots.SQ_LENGTH
)
type_implements_sq_concat = partial(
    type_implements_sq_slot, slot=PySequenceSlots.SQ_CONCAT
)
type_implements_sq_inplace_concat = partial(
    type_implements_sq_slot, slot=PySequenceSlots.SQ_INPLACE_CONCAT
)
type_implements_sq_contains = partial(
    type_implements_sq_slot, slot=PySequenceSlots.SQ_CONTAINS
)
type_implements_sq_ass_item = partial(
    type_implements_sq_slot, slot=PySequenceSlots.SQ_ASS_ITEM
)
type_implements_sq_repeat = partial(
    type_implements_sq_slot, slot=PySequenceSlots.SQ_REPEAT
)

type_implements_sq_inplace_repeat = partial(
    type_implements_sq_slot, slot=PySequenceSlots.SQ_INPLACE_REPEAT
)

# PyMappingSlots
type_implements_mp_length = partial(
    type_implements_mp_slot, slot=PyMappingSlots.MP_LENGTH
)
type_implements_mp_subscript = partial(
    type_implements_mp_slot, slot=PyMappingSlots.MP_SUBSCRIPT
)
type_implements_mp_ass_subscript = partial(
    type_implements_mp_slot, slot=PyMappingSlots.MP_ASS_SUBSCRIPT
)
type_implements_mp_length = partial(
    type_implements_mp_slot, slot=PyMappingSlots.MP_LENGTH
)


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


def type_implements_nb_float(obj_type: type) -> bool:
    """Check whether obj_type implements the nb_float slot."""
    _, _, number_slots, _ = _get_cached_slots(obj_type)
    return has_slot(number_slots, PyNumberSlots.NB_FLOAT)


def type_implements_nb_negative(obj_type: type) -> bool:
    """Check whether obj_type implements the nb_negative slot."""
    _, _, number_slots, _ = _get_cached_slots(obj_type)
    return has_slot(number_slots, PyNumberSlots.NB_NEGATIVE)


def type_implements_nb_positive(obj_type: type) -> bool:
    """Check whether obj_type implements the nb_positive slot."""
    _, _, number_slots, _ = _get_cached_slots(obj_type)
    return has_slot(number_slots, PyNumberSlots.NB_POSITIVE)


def type_implements_nb_absolute(obj_type: type) -> bool:
    """Check whether obj_type implements the nb_absolute slot."""
    _, _, number_slots, _ = _get_cached_slots(obj_type)
    return has_slot(number_slots, PyNumberSlots.NB_ABSOLUTE)


def type_implements_tp_iter(obj_type: type) -> bool:
    _, _, _, type_slot = _get_cached_slots(obj_type)
    return has_slot(type_slot, PyTypeSlots.TP_ITER)


def type_implements_tp_iternext(obj_type: type) -> bool:
    _, _, _, type_slot = _get_cached_slots(obj_type)
    return has_slot(type_slot, PyTypeSlots.TP_ITERNEXT)


def type_implements_tp_repr(obj_type: type) -> bool:
    """Check whether obj_type implements the tp_repr slot."""
    _, _, _, type_slot = _get_cached_slots(obj_type)
    return has_slot(type_slot, PyTypeSlots.TP_REPR)


def type_implements_nb_slot(obj_type: type, slot: int) -> bool:
    """Check whether obj_type implements the nb slot."""
    _, _, number_slots, _ = _get_cached_slots(obj_type)
    return has_slot(number_slots, slot)


def pyiter_check(obj_type: type) -> bool:
    # ref: https://github.com/python/cpython/blob/3.13/Objects/abstract.c#L2891-L2897
    # CPython checks if tp_iternext != _PyObject_NextNotImplemented
    # Dynamo only sets the bit if __next__ is actually defined
    return type_implements_tp_iternext(obj_type)


def pysequence_check(obj_type: type) -> bool:
    """Implements PySequence_Check semantics for VariableTracker objects."""
    # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1714-L1721
    if issubclass(obj_type, dict):
        return False
    return type_implements_sq_item(obj_type)


def pyindex_check(obj_type: type) -> bool:
    """Implements _PyIndex_Check semantics for VariableTracker objects."""
    # ref: https://github.com/python/cpython/blob/3.13/Include/internal/pycore_abstract.h#L11-L17
    return type_implements_nb_index(obj_type)


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


def validate_sequence_index(
    tx: "InstructionTranslatorBase",
    key: VariableTracker,
    container_name: str,
) -> VariableTracker:
    """_PyIndex_Check → nb_index path used by list/tuple/range/str/bytes subscript.

    ref: https://github.com/python/cpython/blob/v3.13.3/Include/internal/pycore_abstract.h (_PyIndex_Check)
    """
    key_type = maybe_get_python_type(key)
    if key_type not in (int, bool, slice):
        if not type_implements_nb_index(key_type):
            raise_observed_exception(
                TypeError,
                tx,
                args=[
                    f"{container_name} indices must be integers or slices, not {key.python_type_name()}"
                ],
            )
        key = key.nb_index_impl(tx)
    return key


def vt_mapping_size(
    tx: "InstructionTranslatorBase", obj: "VariableTracker"
) -> "VariableTracker":
    # ref: https://github.com/python/cpython/blob/v3.13.3/Objects/abstract.c#L2308-L2330
    T = maybe_get_python_type(obj)
    if type_implements_mp_length(T):
        return obj.mp_length(tx)

    if type_implements_sq_length(T):
        raise_type_error(tx, f"{obj.python_type_name()} is not a mapping")

    raise_type_error(tx, f"object of type {obj.python_type_name()} has no len()")


def generic_len(
    tx: "InstructionTranslatorBase", obj: "VariableTracker"
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


def generic_bool(
    tx: "InstructionTranslatorBase", obj: VariableTracker
) -> VariableTracker:
    """Mirrors PyObject_IsTrue.

    https://github.com/python/cpython/blob/c09ccd9c429/Objects/object.c#L2135-L2158

    Resolution order: constants → nb_bool → mp_length/sq_length → truthy.
    """
    from .constant import ConstantVariable

    if obj.is_python_constant():
        try:
            return ConstantVariable.create(bool(obj.as_python_constant()))
        except Exception as e:
            raise_observed_exception(type(e), tx, args=[str(e)])

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
        length_val = length.as_python_constant()
        if length_val < 0:
            raise_observed_exception(
                ValueError, tx, args=["__len__() should return >= 0"]
            )
        return ConstantVariable.create(length_val > 0)
    except ObservedTypeError:
        handle_observed_exception(tx)

    return ConstantVariable.create(True)


_repr_running: set[int] = set()


def generic_repr(
    tx: "InstructionTranslatorBase", obj: VariableTracker
) -> VariableTracker:
    """Mirrors PyObject_Repr with Py_ReprEnter/Py_ReprLeave cycle detection.

    https://github.com/python/cpython/blob/v3.13.3/Objects/object.c#L745-L778

    Resolution order: tp_repr -> TypeError if the result is not str.
    """
    obj_type = maybe_get_python_type(obj)

    if type_implements_tp_repr(obj_type):
        obj_id = id(obj)
        if obj_id in _repr_running:
            sentinel = {list: "[...]", dict: "{...}", collections.deque: "[...]"}
            return ConstantVariable.create(sentinel.get(obj_type, "..."))
        _repr_running.add(obj_id)
        try:
            result = obj.repr_impl(tx)
        finally:
            _repr_running.discard(obj_id)
        result_type = maybe_get_python_type(result)
        if not issubclass(result_type, str):
            raise_type_error(
                tx,
                f"__repr__ returned non-string (type {result_type.__name__})",
            )
        return result

    raise_type_error(tx, f"object of type '{obj.python_type_name()}' has no repr")


def vt_getitem(
    tx: "InstructionTranslatorBase",
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
    Branch 3 delegates to mp_subscript_impl for type objects (__class_getitem__).
    """
    obj_type = maybe_get_python_type(obj)
    # Branch 1: mp_subscript
    if type_implements_mp_subscript(obj_type):
        return obj.mp_subscript_impl(tx, key)
    # Branch 2: sq_item (only if mp_subscript is absent)
    # CPython: abstract.c L168-181 — _PyIndex_Check(key) → PyNumber_AsSsize_t
    #          → PySequence_GetItem (wraps negative, calls sq_item)
    if type_implements_sq_item(obj_type):
        key_type = maybe_get_python_type(key)
        if type_implements_nb_index(key_type):
            key = key.nb_index_impl(tx)
            return vt_sequence_getitem(tx, obj, key)
        raise_type_error(
            tx,
            f"{obj_type.__name__} indices must be integers, not {key_type.__name__}",
        )
    # Branch 3: PyType_Check → __class_getitem__ (abstract.c L183-203)
    # In 3.10+ type.__getitem__ sets mp_subscript so this is normally caught
    # by Branch 1, but we check explicitly for safety.
    if issubclass(obj_type, type):
        return obj.mp_subscript_impl(tx, key)
    # CPython: abstract.c L205
    raise_type_error(tx, f"'{obj_type.__name__}' object is not subscriptable")


def vt_sequence_getitem(
    tx: "InstructionTranslatorBase",
    obj: VariableTracker,
    index: VariableTracker,
) -> VariableTracker:
    """CPython's PySequence_GetItem — always sq_item, never mp_subscript.

    ref: https://github.com/python/cpython/blob/v3.13.3/Objects/abstract.c#L1874-L1902

    Called by PyObject_GetItem branch 2, reversed() fallback, and the old
    iteration protocol.  Wraps negative indices via sq_length before
    dispatching to sq_item.
    """
    obj_type = maybe_get_python_type(obj)

    if type_implements_sq_item(obj_type):
        # Negative index wrapping (abstract.c L2175-2183)
        if isinstance(index, ConstantVariable):
            index_val = index.as_python_constant()
            if isinstance(index_val, int) and index_val < 0:
                if type_implements_sq_length(obj_type):
                    length = obj.sq_length(tx)
                    index = ConstantVariable.create(
                        index_val + length.as_python_constant()
                    )
        return obj.sq_item_impl(tx, index)

    if type_implements_mp_subscript(obj_type):
        raise_type_error(tx, f"'{obj.python_type_name()}' is not a sequence")

    raise_type_error(tx, f"'{obj.python_type_name()}' object does not support indexing")


def vt_sequence_setitem(
    tx: "InstructionTranslatorBase",
    s: VariableTracker,
    i: VariableTracker,
    o: VariableTracker,
) -> VariableTracker:
    # ref: https://github.com/python/cpython/blob/3.13/Objects/abstract.c#L1926-L1957 (PySequence_SetItem)
    s_type = maybe_get_python_type(s)
    if type_implements_sq_ass_item(s_type):
        # Negative index wrapping (abstract.c L1944-1952)
        if isinstance(i, ConstantVariable):
            index_val = i.as_python_constant()
            if isinstance(index_val, int) and index_val < 0:
                if type_implements_sq_length(s_type):
                    length = s.sq_length(tx)
                    i = ConstantVariable.create(index_val + length.as_python_constant())
        return s.sq_ass_item_impl(tx, i, o)

    if type_implements_mp_ass_subscript(s_type):
        raise_type_error(tx, f"'{s.python_type_name()}' is not a sequence")

    raise_type_error(
        tx, f"'{s.python_type_name()}' object does not support item assignment"
    )


def generic_setitem(
    tx: "InstructionTranslatorBase",
    o: VariableTracker,
    key: VariableTracker,
    value: VariableTracker,
) -> VariableTracker:
    # ref: https://github.com/python/cpython/blob/3.13/Objects/abstract.c#L222-L254
    o_type = maybe_get_python_type(o)
    if type_implements_mp_ass_subscript(o_type):
        return o.mp_ass_subscript_impl(tx, key, value)

    if type_implements_sq_ass_item(o_type):
        key_type = maybe_get_python_type(key)
        if pyindex_check(key_type):
            key = key.nb_index_impl(tx)
            return vt_sequence_setitem(tx, o, key, value)
        raise_type_error(
            tx, f"sequence index must be integer, not '{key.python_type_name()}'"
        )
    raise_type_error(
        tx, f"'{o.python_type_name()}' object does not support item assignment"
    )


def sequence_delitem(
    tx: "InstructionTranslatorBase",
    s: VariableTracker,
    i: VariableTracker,
) -> VariableTracker:
    # ref: https://github.com/python/cpython/blob/3.13/Objects/abstract.c#L1959-L1990

    s_type = maybe_get_python_type(s)
    if type_implements_sq_ass_item(s_type):
        if isinstance(i, ConstantVariable):
            idx = i.as_python_constant()
            if idx < 0:
                if type_implements_sq_length(s_type):
                    length = s.sq_length(tx)
                    i = vt_add(tx, i, length)
        return s.sq_ass_item_impl(tx, i, None)

    if type_implements_mp_ass_subscript(s_type):
        raise_type_error(tx, f"'{s.python_type_name()}' is not a sequence")

    raise_type_error(
        tx, f"'{s.python_type_name()}' object does not support item deletion"
    )


def generic_delitem(
    tx: "InstructionTranslatorBase",
    o: VariableTracker,
    key: VariableTracker,
) -> VariableTracker:
    # ref: https://github.com/python/cpython/blob/3.13/Objects/abstract.c#L256-L288

    o_type = maybe_get_python_type(o)
    if type_implements_mp_ass_subscript(o_type):
        return o.mp_ass_subscript_impl(tx, key, None)

    key_type = maybe_get_python_type(key)
    if pyindex_check(key_type):
        key_value = key.nb_index_impl(tx)
        return sequence_delitem(tx, o, key_value)
    elif type_implements_sq_ass_item(o_type):
        raise_type_error(
            tx, f"sequence index must be integer, not {key.python_type_name()}"
        )

    raise_type_error(tx, f"'{o.python_type_name()}' does not support item deletion")


def generic_int(
    tx: "InstructionTranslatorBase", obj: VariableTracker
) -> VariableTracker:
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


def generic_float(
    tx: "InstructionTranslatorBase", obj: VariableTracker
) -> VariableTracker:
    """Mirrors PyNumber_Float (float(x) dispatch).

    https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1635-L1692

    Resolution: nb_float → nb_index → str parsing → TypeError.
    """
    from .constant import ConstantVariable

    # Fast path: if the value is already a float constant, return it directly.
    # Mirrors PyFloat_CheckExact fast path at the top of PyNumber_Float
    # (abstract.c:1641-1643).
    if obj.is_python_constant() and isinstance(obj.as_python_constant(), float):
        return ConstantVariable.create(float(obj.as_python_constant()))

    obj_type = maybe_get_python_type(obj)

    if type_implements_nb_float(obj_type):
        return obj.nb_float_impl(tx)

    # https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1674-L1685
    if type_implements_nb_index(obj_type):
        return obj.nb_index_impl(tx)

    # PyFloat_FromString fallback — handles str and bytes.
    # https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1691
    if obj.is_python_constant() and isinstance(obj.as_python_constant(), (str, bytes)):
        try:
            return ConstantVariable.create(float(obj.as_python_constant()))
        except ValueError as e:
            raise_observed_exception(ValueError, tx, args=[str(e)])

    raise_type_error(
        tx,
        f"float() argument must be a string or a real number, "
        f"not '{obj.python_type_name()}'",
    )


def generic_iternext(
    tx: "InstructionTranslatorBase", obj: VariableTracker
) -> "VariableTracker":
    """
    Implements PyIter_Next / tp_iternext semantics for VariableTracker objects.

    Calls obj.tp_iternext_impl(tx) if the object is an iterator, otherwise raises
    TypeError. StopIteration propagation is left to the caller (mirrors
    CPython's iternext contract where NULL return signals exhaustion).
    """
    # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L2865

    T = maybe_get_python_type(obj)
    if not type_implements_tp_iternext(T):
        raise_type_error(tx, f"expected an iterator, got '{obj.python_type_name()}'")

    return obj.tp_iternext_impl(tx)


def generic_neg(
    tx: "InstructionTranslatorBase", obj: VariableTracker
) -> VariableTracker:
    """Mirrors PyNumber_Negative.

    https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1375-L1392

    Algorithm:
    1. If type has nb_negative slot, call obj.nb_negative_impl(tx)
    2. Otherwise, raise TypeError
    """
    obj_type = maybe_get_python_type(obj)

    if type_implements_nb_negative(obj_type):
        return obj.nb_negative_impl(tx)

    raise_type_error(
        tx,
        f"bad operand type for unary -: '{obj.python_type_name()}'",
    )


def generic_pos(
    tx: "InstructionTranslatorBase", obj: VariableTracker
) -> VariableTracker:
    """Mirrors PyNumber_Positive.

    https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1375-L1393

    Algorithm:
    1. If type has nb_positive slot, call obj.nb_positive_impl(tx)
    2. Otherwise, raise TypeError
    """
    obj_type = maybe_get_python_type(obj)

    if type_implements_nb_positive(obj_type):
        return obj.nb_positive_impl(tx)

    raise_type_error(
        tx,
        f"bad operand type for unary +: '{obj.python_type_name()}'",
    )


def generic_abs(
    tx: "InstructionTranslatorBase", obj: VariableTracker
) -> VariableTracker:
    """Mirrors PyNumber_Absolute.

    https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1375-L1395

    Algorithm:
    1. If type has nb_absolute slot, call obj.nb_absolute_impl(tx)
    2. Otherwise, raise TypeError
    """
    obj_type = maybe_get_python_type(obj)

    if type_implements_nb_absolute(obj_type):
        return obj.nb_absolute_impl(tx)

    raise_type_error(
        tx,
        f"bad operand type for abs(): '{obj.python_type_name()}'",
    )


def vt_is_iterable(obj: VariableTracker) -> bool:
    """Check if the object supports iteration (i.e. has tp_iter or sequence protocol)."""
    T = maybe_get_python_type(obj)
    return type_implements_tp_iter(T) or pysequence_check(T)


def generic_getiter(
    tx: "InstructionTranslatorBase", obj: VariableTracker
) -> "VariableTracker":
    """
    Implements PyObject_GetIter semantics for VariableTracker objects.
    Routes to obj.tp_iter_impl(tx), the tp_iter slot on the object's type.
    """

    # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L2847-L2870
    # The algorithm for PyObject_GetIter works as follows: Steps:
    # 1. If the object has tp_iter slot, call it and return the result. The
    #    return object must be an iterator (it must have a tp_iternext slot)
    # 2. If the object implements the sequence protocol - implements __getitem__
    #    then create a sequence iterator for the object and return it
    # 3. Otherwise, raise a TypeError

    T = maybe_get_python_type(obj)
    if type_implements_tp_iter(T):
        res = obj.tp_iter_impl(tx)
        res_T = maybe_get_python_type(res)
        if not pyiter_check(res_T):
            raise_type_error(
                tx,
                f"iter() returned non-iterator of type '{res.python_type_name()}'",
            )
        return res
    elif pysequence_check(T):
        from .functions import UserFunctionVariable

        return UserFunctionVariable(polyfills.builtins.sequence_iterator).call_function(
            tx, [obj], {}
        )
    else:
        raise_type_error(tx, f"'{obj.python_type_name()}' object is not iterable")


# ---------------------------------------------------------------------------
# Binary-op dispatch (CPython's abstract.c: binary_op1 / binary_op)
# https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L927 (binary_op1)
# ---------------------------------------------------------------------------

NB_SLOT_MAPPING = {
    "nb_lshift": PyNumberSlots.NB_LSHIFT,
    "nb_inplace_lshift": PyNumberSlots.NB_INPLACE_LSHIFT,
    "nb_inplace_rshift": PyNumberSlots.NB_INPLACE_RSHIFT,
    "nb_rshift": PyNumberSlots.NB_RSHIFT,
    "nb_or": PyNumberSlots.NB_OR,
    "nb_inplace_or": PyNumberSlots.NB_INPLACE_OR,
    "nb_subtract": PyNumberSlots.NB_SUBTRACT,
    "nb_inplace_subtract": PyNumberSlots.NB_INPLACE_SUBTRACT,
    "nb_add": PyNumberSlots.NB_ADD,
    "nb_inplace_add": PyNumberSlots.NB_INPLACE_ADD,
    "nb_multiply": PyNumberSlots.NB_MULTIPLY,
    "nb_inplace_multiply": PyNumberSlots.NB_INPLACE_MULTIPLY,
}


def is_nb_not_implemented(result: VariableTracker) -> bool:
    return result.is_constant_match(NotImplemented)


def is_python_subtype(w: VariableTracker, v: VariableTracker) -> bool:
    """Check if w's underlying Python type is a proper subtype of v's."""
    try:
        return issubclass(w.python_type(), v.python_type())
    except NotImplementedError:
        return False


#   Calling scheme used for binary operations:
#
#   Order operations are tried until either a valid result or error:
#     w.op(v,w)[*], v.op(v,w), w.op(v,w)
#
#   [*] only when Py_TYPE(v) != Py_TYPE(w) && Py_TYPE(w) is a subclass of
#       Py_TYPE(v)


def binary_op1(
    tx: "InstructionTranslatorBase",
    v: VariableTracker,
    w: VariableTracker,
    op_slot: str,
) -> VariableTracker:
    """CPython's binary_op1: try v's slot, then w's slot with subclass priority.

    Each VT that participates provides a ``<op_slot>_impl(self, tx, other,
    reverse)`` method. ``reverse=False`` means "self is left operand" (forward,
    e.g. ``__or__``), ``reverse=True`` means "self is right operand" (reverse,
    e.g. ``__ror__``). For built-in types the flag is ignored because their
    slots check both operands symmetrically.

    CPython splits the check into two steps: ``Py_TYPE(v)->tp_as_number !=
    NULL`` (does the type have a number-protocol struct), then read the
    slot pointer (which may itself be NULL).  We collapse both into a
    single :func:`type_implements_nb_slot` query — its bit is set only
    when the specific slot is non-NULL, which already implies
    ``tp_as_number`` is non-NULL.  Treating a missing slot as "no impl"
    is what keeps the base ``VariableTracker.nb_*_impl`` graph-break
    out of the path for types that genuinely lack the slot in C.

    https://github.com/python/cpython/blob/3.13/Objects/abstract.c#L926-L977
    """
    impl_attr = f"{op_slot}_impl"
    nb_slot_bit = NB_SLOT_MAPPING[op_slot]

    v_type = maybe_get_python_type(v)
    w_type = maybe_get_python_type(w)

    v_slot = (
        getattr(type(v), impl_attr, None)
        if type_implements_nb_slot(v_type, nb_slot_bit)
        else None
    )
    w_slot = (
        getattr(type(w), impl_attr, None)
        if type_implements_nb_slot(w_type, nb_slot_bit)
        else None
    )

    # CPython skips slotw if Py_TYPE(w) == Py_TYPE(v) (one C type, one slot
    # function).  In Dynamo two VT subclasses can share a Python type — e.g.
    # ConstantVariable(3) and SymNodeVariable both report ``int`` — yet have
    # different ``nb_*_impl`` methods.  Comparing the slots themselves
    # captures both "literally the same function" (CPython's check) and
    # "different VT subclasses sharing a python_type", so we drop the type
    # equality check.
    if v_slot is w_slot:
        w_slot = None

    if v_slot is not None:
        # Subclass priority: if w's Python type is a proper subtype of v's
        # Python type and overrides the slot, try w first (CPython abstract.c:952-960).
        if w_slot is not None and is_python_subtype(w, v):
            # CPython ALWAYS calls the slot with (v, w), even for reverse slots.
            # Since w_slot is a method call, we use reverse=True to indicate w
            # is the right operand, matching CPython's semantics.
            result = w_slot(w, tx, v, True)
            if not is_nb_not_implemented(result):
                return result
            w_slot = None
        result = v_slot(v, tx, w, False)
        if not is_nb_not_implemented(result):
            return result
    if w_slot is not None:
        result = w_slot(w, tx, v, True)
        if not is_nb_not_implemented(result):
            return result
    return ConstantVariable.create(NotImplemented)


def binary_op(
    tx: "InstructionTranslatorBase",
    v: VariableTracker,
    w: VariableTracker,
    op_slot: str,
    op_symbol: str,
) -> VariableTracker:
    """CPython's binary_op: binary_op1 + TypeError fallback.
    https://github.com/python/cpython/blob/3.13/Objects/abstract.c#L997-L1020
    """

    result = binary_op1(tx, v, w, op_slot)
    if is_nb_not_implemented(result):
        binop_type_error(tx, v, w, op_symbol)
    return result


#  Binary in-place operators
#
#    The in-place operators are defined to fall back to the 'normal', non
#    in-place operations, if the in-place methods are not in place.
#
#    - If the left hand object has the appropriate struct members, and they are
#      filled, call the appropriate function and return the result.  No coercion
#      is done on the arguments; the left-hand object is the one the operation
#      is performed on, and it's up to the function to deal with the right-hand
#      object.
#
#    - Otherwise, in-place modification is not supported. Handle it exactly as a
#      non in-place operation of the same kind.


def binary_iop1(
    tx: "InstructionTranslatorBase",
    v: VariableTracker,
    w: VariableTracker,
    iop_slot: str,
    op_slot: str,
) -> VariableTracker:
    v_type = maybe_get_python_type(v)

    if type_implements_nb_slot(v_type, NB_SLOT_MAPPING[iop_slot]):
        impl_attr = f"{iop_slot}_impl"
        slot = getattr(type(v), impl_attr)
        result = slot(v, tx, w)
        if not is_nb_not_implemented(result):
            return result

    return binary_op1(tx, v, w, op_slot)


def binary_iop(
    tx: "InstructionTranslatorBase",
    v: VariableTracker,
    w: VariableTracker,
    iop_slot: str,
    op_slot: str,
    op_symbol: str,
) -> VariableTracker:
    """CPython's binary_iop: try inplace slot, fallback to binary_op1.

    Combines binary_iop1 + TypeError fallback from binary_iop.
    https://github.com/python/cpython/blob/3.13/Objects/abstract.c#L1229-L1270 (binary_iop1, binary_iop)
    """
    result = binary_iop1(tx, v, w, iop_slot, op_slot)
    if is_nb_not_implemented(result):
        binop_type_error(tx, v, w, op_symbol)
    return result


# add / inplace add needs special handling
def vt_add(
    tx: "InstructionTranslatorBase",
    v: VariableTracker,
    w: VariableTracker,
) -> VariableTracker:
    """Implements addition via nb_add / nb_inplace_add with binary_op dispatch."""
    # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1138-L1155
    result = binary_op1(tx, v, w, "nb_add")
    if not is_nb_not_implemented(result):
        return result

    T = maybe_get_python_type(v)
    if type_implements_sq_concat(T):
        return v.sq_concat_impl(tx, w)
    binop_type_error(tx, v, w, "+")


def vt_inplace_add(
    tx: "InstructionTranslatorBase",
    v: VariableTracker,
    w: VariableTracker,
) -> VariableTracker:
    """Implements in-place addition via nb_inplace_add with binary_iop dispatch."""
    # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1307-L1328
    result = binary_iop1(tx, v, w, "nb_inplace_add", "nb_add")
    if is_nb_not_implemented(result):
        obj_type = maybe_get_python_type(v)
        if type_implements_sq_inplace_concat(obj_type):
            return v.sq_inplace_concat_impl(tx, w)
        elif type_implements_sq_concat(obj_type):
            return v.sq_concat_impl(tx, w)
        else:
            binop_type_error(tx, v, w, "+=")
    return result


# ---------------------------------------------------------------------------
# Multiplication: PyNumber_Multiply / PyNumber_InPlaceMultiply
#
# Multiplication is special because numeric types implement nb_multiply but
# sequence types (list, tuple, str, bytes, bytearray) implement only sq_repeat.
# CPython's PyNumber_Multiply tries nb_multiply first (via binary_op1) and on
# NotImplemented falls back to sq_repeat on either operand.
#
# https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1156-L1193
# ---------------------------------------------------------------------------


def sequence_repeat(
    tx: "InstructionTranslatorBase",
    seq: VariableTracker,
    n: VariableTracker,
) -> VariableTracker:
    """Mirrors CPython's sequence_repeat helper.

    Validates that ``n`` is index-like, converts it to an int, and dispatches
    to ``seq.sq_repeat_impl(tx, count)``.

    https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1156-L1174
    """
    n_type = maybe_get_python_type(n)
    if not type_implements_nb_index(n_type):
        raise_type_error(
            tx,
            f"can't multiply sequence by non-int of type '{n.python_type_name()}'",
        )
    count = n.nb_index_impl(tx)
    validate_sequence_repeat_count(tx, count)
    return seq.sq_repeat_impl(tx, count)


def sequence_inplace_repeat(
    tx: "InstructionTranslatorBase",
    seq: VariableTracker,
    n: VariableTracker,
) -> VariableTracker:
    """sequence_repeat using sq_inplace_repeat.

    The validation step is identical to ``sequence_repeat``; only the
    target slot differs.
    """
    n_type = maybe_get_python_type(n)
    if not type_implements_nb_index(n_type):
        raise_type_error(
            tx,
            f"can't multiply sequence by non-int of type '{n.python_type_name()}'",
        )
    count = n.nb_index_impl(tx)
    validate_sequence_repeat_count(tx, count)
    return seq.sq_inplace_repeat_impl(tx, count)


def validate_sequence_repeat_count(
    tx: "InstructionTranslatorBase",
    count: VariableTracker,
) -> None:
    n = count.as_python_constant()
    if n < -sys.maxsize - 1 or n > sys.maxsize:
        raise_observed_exception(
            OverflowError,
            tx,
            args=["cannot fit 'int' into an index-sized integer"],
        )


def generic_multiply(
    tx: "InstructionTranslatorBase",
    v: VariableTracker,
    w: VariableTracker,
) -> VariableTracker:
    """Mirrors CPython's PyNumber_Multiply.

    Try nb_multiply via binary_op1; on NotImplemented fall back to sq_repeat
    on either operand.  TypeError if no path works.

    https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1176-L1193
    """
    result = binary_op1(tx, v, w, "nb_multiply")
    if not is_nb_not_implemented(result):
        return result

    v_type = maybe_get_python_type(v)
    w_type = maybe_get_python_type(w)
    if type_implements_sq_repeat(v_type):
        return sequence_repeat(tx, v, w)
    if type_implements_sq_repeat(w_type):
        return sequence_repeat(tx, w, v)

    raise_type_error(
        tx,
        f"unsupported operand type(s) for *: "
        f"'{v.python_type_name()}' and '{w.python_type_name()}'",
    )


def generic_inplace_multiply(
    tx: "InstructionTranslatorBase",
    v: VariableTracker,
    w: VariableTracker,
) -> VariableTracker:
    """Mirrors CPython's PyNumber_InPlaceMultiply.

    Try nb_inplace_multiply / nb_multiply via binary_iop1; on NotImplemented
    fall back to sq_inplace_repeat (preferred), then sq_repeat.

    https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1330-L1357
    """
    result = binary_iop1(tx, v, w, "nb_inplace_multiply", "nb_multiply")
    if not is_nb_not_implemented(result):
        return result

    v_type = maybe_get_python_type(v)
    w_type = maybe_get_python_type(w)
    if type_implements_sq_inplace_repeat(v_type):
        return sequence_inplace_repeat(tx, v, w)
    if type_implements_sq_repeat(v_type):
        return sequence_repeat(tx, v, w)
    # Cannot mutate w in-place — abstract.c L1348-1352 explicitly avoids
    # sq_inplace_repeat on the right-hand operand.
    if type_implements_sq_repeat(w_type):
        return sequence_repeat(tx, w, v)

    raise_type_error(
        tx,
        f"unsupported operand type(s) for *=: "
        f"'{v.python_type_name()}' and '{w.python_type_name()}'",
    )


# ---------------------------------------------------------------------------
# Type-object slot wrappers for ``__mul__`` / ``__rmul__`` / ``__imul__``
#
# These mirror the *type object*'s installation of ``__mul__`` etc. as a
# method, not the operator dispatch.  In CPython the user-visible
# ``int.__mul__`` and ``list.__mul__`` are different functions, each
# generated at type-construction time from ``Objects/typeobject.c``'s
# ``slotdefs[]`` table:
#
#   slotdefs[]:
#     BINSLOT(__mul__, nb_multiply, slot_nb_multiply, "*")    [L10323]
#     SQSLOT (__mul__, sq_repeat,   NULL, wrap_indexargfunc)  [L10419]
#     RBINSLOT(__rmul__, nb_multiply, slot_nb_multiply, "*")  [L10325]
#     SQSLOT (__rmul__, sq_repeat,   NULL, wrap_indexargfunc) [L10421]
#     IBSLOT (__imul__, nb_inplace_multiply, ...)             [L10364]
#     SQSLOT (__imul__, sq_inplace_repeat, NULL, ...)         [L10434]
#
# https://github.com/python/cpython/blob/v3.13.13/Objects/typeobject.c#L10244 (slotdefs[])
# https://github.com/python/cpython/blob/v3.13.13/Objects/typeobject.c#L10412-L10416 (rationale comment)
#
# ``add_operators`` walks ``slotdefs[]`` and inserts a method into the
# type's ``__dict__`` for whichever slot the type fills.  As the comment
# at L10412 notes, types fill at most one of ``nb_multiply`` /
# ``sq_repeat`` per op; nb_* takes priority because its slotdef appears
# first.  ``slot_wrapper_mul``/``slot_wrapper_imul`` reproduce that
# selection so that ``call_method`` routing for direct dunder calls
# (``[1, 2].__mul__(3)``) reaches the correct slot.
#
# Distinct from ``generic_multiply`` / ``generic_inplace_multiply``, which
# mirror the operator-level algorithm in ``Objects/abstract.c``
# (``PyNumber_Multiply``) — cross-operand subclass priority and the
# ``sq_repeat`` fallback when ``nb_multiply`` returns ``NotImplemented``.
# ---------------------------------------------------------------------------


def slot_wrapper_mul(
    tx: "InstructionTranslatorBase",
    self: VariableTracker,
    other: VariableTracker,
    reverse: bool = False,
) -> VariableTracker:
    """``self.__mul__(other)`` / ``self.__rmul__(other)`` slot wrapper."""
    self_type = maybe_get_python_type(self)
    if type_implements_nb_slot(self_type, PyNumberSlots.NB_MULTIPLY):
        return self.nb_multiply_impl(tx, other, reverse=reverse)
    if type_implements_sq_repeat(self_type):
        # SQSLOT for __mul__ and __rmul__ both use ``wrap_indexargfunc`` —
        # the wrapper ignores the reverse flag because sq_repeat takes
        # ``(seq, count)`` regardless of which side ``self`` is on.
        return sequence_repeat(tx, self, other)
    raise_type_error(
        tx,
        f"unsupported operand type(s) for *: "
        f"'{self.python_type_name()}' and '{other.python_type_name()}'",
    )


def slot_wrapper_imul(
    tx: "InstructionTranslatorBase",
    self: VariableTracker,
    other: VariableTracker,
) -> VariableTracker:
    """``self.__imul__(other)`` slot wrapper.

    When neither ``nb_inplace_multiply`` nor ``sq_inplace_repeat`` is
    installed, the slotdef machinery doesn't generate ``__imul__`` at all
    — but the operator-level fallback in ``slot_nb_inplace_multiply``
    (typeobject.c) does try the non-inplace slot.  We mirror that here so
    method lookups don't graph-break on (e.g.) tuple, even though tuple
    has no ``__imul__`` attribute in standard CPython.
    """
    self_type = maybe_get_python_type(self)
    if type_implements_nb_slot(self_type, PyNumberSlots.NB_INPLACE_MULTIPLY):
        return self.nb_inplace_multiply_impl(tx, other)
    if type_implements_sq_inplace_repeat(self_type):
        return sequence_inplace_repeat(tx, self, other)
    return slot_wrapper_mul(tx, self, other)


# ---------------------------------------------------------------------------
# tp_richcompare -- comparison dispatch
#
# CPython comparison architecture (Objects/object.c, Objects/typeobject.c):
#
#   a == b  (COMPARE_OP bytecode)
#     -> PyObject_RichCompare(a, b, Py_EQ)
#       -> do_richcompare(a, b, Py_EQ)            # the 4-step algorithm
#         -> type(a)->tp_richcompare(a, b, Py_EQ)  # per-type slot
#
#   a.__eq__(b)  (attribute access)
#     -> type(a)->tp_getattro(a, "__eq__")          # descriptor protocol
#     -> returns wrapper bound to tp_richcompare
#     -> calling wrapper invokes tp_richcompare(a, b, Py_EQ) directly
#
# do_richcompare algorithm (Objects/object.c#L901-L955):
#   1. Subclass priority: if type(b) is a proper subclass of type(a),
#      try type(b)->tp_richcompare(b, a, swapped_op) first
#   2. Forward: type(a)->tp_richcompare(a, b, op)
#   3. Reflected: type(b)->tp_richcompare(b, a, swapped_op)
#   4. Fallback: identity for eq/ne, TypeError for ordering
#
# Dynamo implementation:
#
#   richcompare_impl(self, tx, other, op) -- per-VT slot, analogous to
#     tp_richcompare.  Returns ConstantVariable(NotImplemented) when the
#     type does not handle the comparison.
#
#   generic_richcompare(tx, lhs, rhs, op) -- analogous to do_richcompare.
#     Implements the 4-step algorithm directly using richcompare_impl
#     slots.  If a user comparison method graph-breaks, the Unsupported
#     exception propagates to COMPARE_OP (which has
#     @break_graph_if_unsupported) and runs the comparison eagerly.
#     UDOV.richcompare_impl disables nested graph breaks on the resolved
#     funcvar so the InliningInstructionTranslator does not try to split
#     the inlined user method mid-function.
#
# Two entry points converge on richcompare_impl:
#
#   COMPARE_OP (a == b):
#     -> BuiltinVariable dispatch -> generic_richcompare
#       -> richcompare_impl (4-step: subclass priority, forward, reflected, fallback)
#
#   call_method("__eq__") (a.__eq__(b) in user code):
#     -> base.py call_method -> richcompare_impl directly
#
# The call_method path calls richcompare_impl directly (not
# generic_richcompare) to match CPython semantics: a.__eq__(b) invokes
# the type's tp_richcompare slot without do_richcompare's reflected-
# operand protocol, and may return NotImplemented.
# ---------------------------------------------------------------------------


is_richcompare_not_implemented = is_nb_not_implemented


def object_richcompare(
    self: VariableTracker,
    tx: "InstructionTranslatorBase",
    other: VariableTracker,
    op: str,
) -> VariableTracker:
    """object's tp_richcompare.

    https://github.com/python/cpython/blob/e76aa128fe/Objects/typeobject.c#L6263-L6305
    - __eq__: identity check, else NotImplemented
    - __ne__: delegates to tp_richcompare(self, other, Py_EQ) and inverts
    - ordering
    """
    if op == "__eq__":
        identity = vt_identity_compare(self, other)
        if identity is not None and identity.as_python_constant():
            return ConstantVariable.create(True)
        return ConstantVariable.create(NotImplemented)
    elif op == "__ne__":
        # https://github.com/python/cpython/blob/e76aa128fe/Objects/typeobject.c#L6279-L6298
        # Safe to call as_python_constant(): only identity-based types use
        # object_richcompare, so eq_result is always True or NotImplemented.
        eq_result = self.richcompare_impl(tx, other, "__eq__")
        if is_richcompare_not_implemented(eq_result):
            return eq_result
        return ConstantVariable.create(not eq_result.as_python_constant())
    else:
        return ConstantVariable.create(NotImplemented)


def python_constant_richcompare_impl(
    self: VariableTracker,
    tx: "InstructionTranslatorBase",
    other: VariableTracker,
    op: str,
) -> VariableTracker:
    """Constant-fold comparison for types with as_python_constant()."""
    if not self.is_python_constant() or not other.is_python_constant():
        return ConstantVariable.create(NotImplemented)
    self_val = self.as_python_constant()
    other_val = other.as_python_constant()
    try:
        result = getattr(type(self_val), op)(self_val, other_val)
    except TypeError as e:
        raise_observed_exception(TypeError, tx, args=list(e.args))
    return ConstantVariable.create(result)


# _Py_SwappedOp: https://github.com/python/cpython/blob/e76aa128fe/Objects/object.c#L987
_REFLECTED_OP: dict[str, str] = {
    "__lt__": "__gt__",
    "__gt__": "__lt__",
    "__le__": "__ge__",
    "__ge__": "__le__",
    "__eq__": "__eq__",
    "__ne__": "__ne__",
}

_OP_STR: dict[str, str] = {
    "__lt__": "<",
    "__le__": "<=",
    "__eq__": "==",
    "__ne__": "!=",
    "__gt__": ">",
    "__ge__": ">=",
}


def generic_richcompare(
    tx: "InstructionTranslatorBase",
    v: VariableTracker,
    w: VariableTracker,
    op: str,
) -> VariableTracker:
    """Dynamo's do_richcompare.

    https://github.com/python/cpython/blob/e76aa128fe/Objects/object.c#L994-L1039

    Implements the 4-step algorithm directly using richcompare_impl slots.
    Graph breaks inside user comparison methods propagate to COMPARE_OP
    (which runs eagerly) because UDOV.richcompare_impl disables nested
    graph breaks on the resolved funcvar.
    """
    reflected = _REFLECTED_OP[op]

    try:
        v_type = v.python_type()
    except NotImplementedError:
        v_type = None
    try:
        w_type = w.python_type()
    except NotImplementedError:
        w_type = None

    checked_reverse = False

    # Step 1: subclass priority
    if (
        v_type is not None
        and w_type is not None
        and v_type is not w_type
        and issubclass(w_type, v_type)
    ):
        checked_reverse = True
        result = w.richcompare_impl(tx, v, reflected)
        if not is_richcompare_not_implemented(result):
            return result

    # Step 2: forward
    result = v.richcompare_impl(tx, w, op)
    if not is_richcompare_not_implemented(result):
        return result

    # Step 3: reflected (if not already tried)
    if not checked_reverse:
        result = w.richcompare_impl(tx, v, reflected)
        if not is_richcompare_not_implemented(result):
            return result

    # Step 4: fallback
    if op in ("__eq__", "__ne__"):
        identity = vt_identity_compare(v, w)
        if identity is not None:
            if op == "__ne__":
                return ConstantVariable.create(not identity.as_python_constant())
            return identity
        unimplemented(
            gb_type="richcompare identity fallback undetermined",
            context=f"generic_richcompare({v}, {w}, {op})",
            explanation="Cannot determine object identity for comparison fallback.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )
    else:
        raise_type_error(
            tx,
            f"'{_OP_STR[op]}' not supported between instances of "
            f"'{v.python_type_name()}' and '{w.python_type_name()}'",
        )


def generic_richcompare_bool(
    tx: "InstructionTranslatorBase",
    v: VariableTracker,
    w: VariableTracker,
    op: str,
) -> VariableTracker:
    """Dynamo's PyObject_RichCompareBool for eq/ne.

    https://github.com/python/cpython/blob/e76aa128fe/Objects/object.c#L1046-L1080

    Like generic_richcompare, but with an identity shortcut first: if v
    and w are the same Python object, eq is True and ne is False. This
    matters for NaN (nan is nan -> True, but nan == nan -> False). Used by
    container comparisons (list_richcompare, tuplerichcompare) which call
    PyObject_RichCompareBool per element in CPython.
    """
    if op not in ("__eq__", "__ne__"):
        raise AssertionError(f"generic_richcompare_bool only supports eq/ne, got {op}")
    identity = vt_identity_compare(v, w)
    if identity is not None and identity.as_python_constant():
        return ConstantVariable.create(op == "__eq__")
    return generic_richcompare(tx, v, w, op)


def generic_hash_impl(
    tx: "InstructionTranslatorBase", obj: VariableTracker
) -> tuple[int, bool]:
    """Internal API: compute hash as (value, is_fake).

    Dispatches to the VT's hash_impl.  Called by generic_hash (which wraps
    the result in a VT), container hash_impls (which propagate is_fake),
    and HashableTracker (which just needs the int).
    """
    return obj.hash_impl(tx)


def generic_hash(
    tx: "InstructionTranslatorBase", obj: VariableTracker
) -> VariableTracker:
    """User-facing API: mirrors PyObject_Hash, returns a VariableTracker.

    https://github.com/python/cpython/blob/e76aa128fe/Objects/object.c#L1101-L1115

    Wraps the result in ConstantVariable or FakeIdVariable depending on
    whether the hash depends on a sourceless object's identity.
    """
    from .constant import ConstantVariable, FakeIdVariable, FakeValueKind

    h, is_fake = generic_hash_impl(tx, obj)
    if is_fake:
        return FakeIdVariable(h, kind=FakeValueKind.HASH)
    return ConstantVariable.create(h)


def generic_contains(
    tx: "InstructionTranslatorBase", obj: "VariableTracker", item: "VariableTracker"
) -> "VariableTracker":
    """
    Implements PySequence_Contains semantics for VariableTracker objects.

    If the object has sq_contains (i.e., __contains__), calls obj.sq_contains(tx, item).
    Otherwise falls back to iterating over obj and comparing each element.
    """
    # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L2272-L2283
    T = maybe_get_python_type(obj)
    if type_implements_sq_contains(T):
        return obj.sq_contains(tx, item)
    else:
        # iter fallback handles both __iter__ and __getitem__ sequence protocol cases
        it = generic_getiter(tx, obj)
        return VariableTracker.build(
            tx, polyfills.impl_CONTAINS_OP_fallback
        ).call_function(tx, [item, it], {})


# Metaclasses whose __subclasscheck__ Dynamo can't trace but whose
# behavior we're willing to observe at trace time via Python's issubclass.
# Each entry trades fidelity to the metaclass's side effects (e.g. ABC's
# subclass cache mutation) for coverage of the common case.
_CONSTANT_FOLD_SUBCLASSCHECK_METACLASSES: tuple[type, ...] = (
    abc.ABCMeta,
    torch._C._TensorMeta,  # actually just type.__subclasscheck__, but easier to list it here
    enum.EnumMeta,
)


def generic_issubclass(
    tx: "InstructionTranslatorBase",
    derived: VariableTracker,
    cls: VariableTracker,
) -> VariableTracker:
    """Mirrors CPython's PyObject_IsSubclass / object_issubclass.

    https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L2766-L2823

    This only attempts to replicate object_issubclass, otherwise we delegate to cpython
    """
    derived_py = derived.get_real_python_backed_value()
    cls_py = cls.get_real_python_backed_value()
    if derived_py is NO_SUCH_SUBOBJ or cls_py is NO_SUCH_SUBOBJ:
        unimplemented(
            gb_type="issubclass() with unsupported arguments",
            context=f"issubclass({derived}, {cls})",
            explanation="Arguments to issubclass() must be backed by python values.",
            hints=[
                "Make sure your arguments are types.",
                *graph_break_hints.USER_ERROR,
                *graph_break_hints.SUPPORTABLE,
            ],
        )
    cls_type = maybe_get_python_type(cls)

    # Step 1: PyType_CheckExact fast path — abstract.c L2772
    if cls_type is type:
        try:
            return ConstantVariable.create(
                issubclass(
                    derived_py,  # pyrefly: ignore [bad-argument-type]
                    cls_py,  # pyrefly: ignore [invalid-argument]
                )
            )
        except TypeError as e:
            raise_observed_exception(TypeError, tx, args=list(e.args))

    # Step 2: PEP 604 Union (e.g. ``int | str``) — abstract.c L2779-2781.
    union_types = {types.UnionType}
    if sys.version_info < (3, 14):
        union_types.add(
            typing._UnionGenericAlias  # pyrefly: ignore [missing-attribute]
        )
    if cls_type in union_types:
        # TODO can trace this once TypingVariable is removed
        args = typing.get_args(cls_py)
        cls = VariableTracker.build(tx, args)

    # Step 3: tuple of classes — abstract.c L2783-2799.  Check for
    # TupleVariable instead of tuple to make the type checker happy.
    from .lists import TupleVariable

    if isinstance(cls, TupleVariable):
        for item in cls.items:
            r = generic_issubclass(tx, derived, item)
            if isinstance(r, ConstantVariable) and r.value:
                return ConstantVariable.create(True)
        return ConstantVariable.create(False)

    # Allowlist short-circuit for Step 4: constant-fold via Python's
    # issubclass for metaclasses whose ``__subclasscheck__`` Dynamo can't
    # trace (see _CONSTANT_FOLD_SUBCLASSCHECK_METACLASSES).  Note that ABCMeta
    # is problematic in particular since it caches registered subclasses.
    # Ideally this should be traced or guarded
    if isinstance(cls_py, type) and issubclass(
        type(cls_py), _CONSTANT_FOLD_SUBCLASSCHECK_METACLASSES
    ):
        try:
            return ConstantVariable.create(
                issubclass(
                    derived_py,  # pyrefly: ignore [bad-argument-type]
                    cls_py,
                )
            )
        except TypeError as e:
            raise_observed_exception(TypeError, tx, args=list(e.args))

    # TypeError gate, mirroring abstract.c L2822 ``recursive_issubclass``:
    # CPython reaches that fallback when ``_PyObject_LookupSpecial`` for
    # ``__subclasscheck__`` returns NULL, and its first action is
    # ``check_class(cls, ...)`` which raises this TypeError.  We check
    # eagerly because Dynamo's ``call_method`` below would graph-break
    # rather than cleanly signal "no such method".
    if not isinstance(cls_py, type):
        raise_type_error(
            tx,
            "issubclass() arg 2 must be a class, a tuple of classes, or a union",
        )

    # Step 4: general case — call ``__subclasscheck__`` on cls's metaclass
    # (abstract.c L2801-2815).  Runs user code on a custom metaclass.
    result = cls.call_method(tx, "__subclasscheck__", [derived], {})

    # Coerce to bool (PyObject_IsTrue, abstract.c L2812).
    return generic_bool(tx, result)


def virtual_iterator_next(
    tx: "InstructionTranslatorBase",
    iter_: VariableTracker,
    null_or_idx: VariableTracker,
) -> tuple[VariableTracker, VariableTracker]:
    """
    Mirrors _PyForIter_VirtualIteratorNext.

    When ``null_or_idx`` is a tagged int (3.15+ virtual-iter path), dispatch
    to the iterable's ``_tp_iteritem`` slot via ``tp_iteritem_impl`` and
    return ``(value, next_index)``.  Otherwise fall back to the standard
    iterator protocol (``tp_iternext``) and return ``(value, NULL)``.

    Iterator exhaustion is signaled by ``ObservedUserStopIteration``
    propagating out of the slot impl, matching the rest of Dynamo's
    iterator protocol.

    https://github.com/python/cpython/blob/f31a89bb901067dd105b00cfa90523cf7ffdbbdd/Python/ceval.c#L3733
    """
    from .misc import NullVariable

    if (
        not isinstance(null_or_idx, NullVariable)
        and maybe_get_python_type(null_or_idx) is int
    ):
        return iter_.tp_iteritem_impl(tx, null_or_idx)
    next_ = generic_iternext(tx, iter_)
    return next_, NullVariable()
