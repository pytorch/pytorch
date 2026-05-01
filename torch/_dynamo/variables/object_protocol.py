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
from .functions import UserFunctionVariable


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


def type_implements_sq_item(obj_type: type) -> bool:
    """Check whether obj_type implements __getitem__ as sequence protocol"""
    seq_slots, _, _, _ = _get_cached_slots(obj_type)
    return has_slot(seq_slots, PySequenceSlots.SQ_ITEM)


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


def type_implements_nb_float(obj_type: type) -> bool:
    """Check whether obj_type implements the nb_float slot."""
    _, _, number_slots, _ = _get_cached_slots(obj_type)
    return has_slot(number_slots, PyNumberSlots.NB_FLOAT)


def type_implements_mp_subscript(obj_type: type) -> bool:
    """Check whether obj_type has tp_as_mapping->mp_subscript."""
    _, map_slots, _, _ = _get_cached_slots(obj_type)
    return has_slot(map_slots, PyMappingSlots.MP_SUBSCRIPT)


def type_implements_tp_iter(obj_type: type) -> bool:
    _, _, _, type_slot = _get_cached_slots(obj_type)
    return has_slot(type_slot, PyTypeSlots.TP_ITER)


def type_implements_tp_iternext(obj_type: type) -> bool:
    _, _, _, type_slot = _get_cached_slots(obj_type)
    return has_slot(type_slot, PyTypeSlots.TP_ITERNEXT)


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
    tx: "InstructionTranslator",
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
    tx: "InstructionTranslator",
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


def generic_float(tx: "InstructionTranslator", obj: VariableTracker) -> VariableTracker:
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
    tx: "InstructionTranslator", obj: VariableTracker
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


def generic_getiter(
    tx: "InstructionTranslator", obj: VariableTracker
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
                f"{obj.python_type_name()}.__iter__() returned non-iterator {res.python_type_name()}",
            )
        return res
    elif pysequence_check(T):
        return UserFunctionVariable(polyfills.builtins.sequence_iterator).call_function(
            tx, [obj], {}
        )
    else:
        raise_type_error(tx, f"'{obj.python_type_name()}' object is not iterable")
