"""
Constant variable tracking in Dynamo.

This module is fundamental to Dynamo's ability to track and propagate constant
values during compilation, ensuring proper handling of Python literals and
maintaining type safety through the compilation process.
"""

from __future__ import annotations

import enum
import operator
import types
from collections.abc import Callable
from typing import Any, cast, Literal, overload, Self, TYPE_CHECKING
from typing_extensions import TypeIs

import torch
from torch._dynamo.source import GetItemSource

from .. import graph_break_hints, variables
from ..exc import raise_observed_exception, unimplemented
from ..utils import common_constant_types, np, unpack_iterable
from .base import (
    GetSet,
    Member,
    Method,
    readonly_setter,
    SourceLocation,
    ValueMutationNew,
    VariableTracker,
)


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase

    from .functions import UserFunctionVariable


def _constant_method(
    descriptor: types.MethodDescriptorType,
) -> Callable[..., VariableTracker | None]:
    def handler(
        self: ConstantVariable,
        tx: InstructionTranslatorBase,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker | None:
        if not all(a.is_python_constant() for a in (*args, *kwargs.values())):
            return None
        try:
            res = descriptor(
                self.value,
                *[a.as_python_constant() for a in args],
                **{k: v.as_python_constant() for k, v in kwargs.items()},
            )
        except Exception as e:
            raise_observed_exception(type(e), tx, args=list(e.args))
        # str.split and friends return a fresh list the caller may mutate.
        mutation_type = ValueMutationNew() if isinstance(res, list) else None
        return ConstantVariable.create(res, mutation_type=mutation_type)

    return handler


def _constant_getter(
    descriptor: types.GetSetDescriptorType | types.MemberDescriptorType,
) -> Callable[..., VariableTracker]:
    def getter(
        self: ConstantVariable, tx: InstructionTranslatorBase
    ) -> VariableTracker:
        return ConstantVariable.create(descriptor.__get__(self.value))

    return getter


ConstantTables = tuple[dict[str, Method], dict[str, GetSet], dict[str, Member]]


def constant_type_tables(ty: type) -> ConstantTables:
    """tp_methods / tp_getset / tp_members of a builtin constant type, read from
    the descriptors in its own __dict__. Base types' entries are inherited
    through the VT class MRO, like CPython's per-type tp_dict."""
    methods: dict[str, Method] = {}
    getset: dict[str, GetSet] = {}
    members: dict[str, Member] = {}
    for name, d in ty.__dict__.items():
        if isinstance(d, types.MethodDescriptorType):
            methods[name] = Method(_constant_method(d))
        elif isinstance(d, types.GetSetDescriptorType):
            getset[name] = GetSet(_constant_getter(d), readonly_setter)
        elif isinstance(d, types.MemberDescriptorType):
            members[name] = Member(_constant_getter(d), readonly_setter)
    return methods, getset, members


SlotImpl = Callable[..., VariableTracker]


class ConstantVariable(VariableTracker):
    """
    Variable tracker for Python literals and basic immutable types, with automatic
    routing support for collection types (lists, tuples, sets, etc.).

    The create() method intelligently constructs appropriate variable types for
    nested collections.
    """

    # The builtin literal types have their own subclasses below; the base class
    # holds the other common_constant_types (torch.dtype, NotImplemented, ...).
    _cpython_type = None

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        # Route ConstantVariable(v) to the subclass modeling type(v), so every
        # construction path (create(), clone(), direct calls) agrees.
        if cls is ConstantVariable and (args or "value" in kwargs):
            value = args[0] if args else kwargs["value"]
            if np is not None and isinstance(value, np.number):
                value = value.item()
            cls = _CONSTANT_VT_FOR_TYPE.get(type(value), ConstantVariable)
        return super().__new__(cls)

    @staticmethod
    def is_constant_vt(vt: VariableTracker) -> TypeIs[ConstantVariable]:
        """Whether vt is ConstantVariable or one of its per-type subclasses
        (IntConstantVariable, ...). False for user subclass VTs such as
        UserDefinedConstantVariable."""
        return type(vt) in _CONSTANT_VT_CLASSES

    # Builders for the per-type subclasses' slot impls. Each subclass declares
    # only the slots its type defines, each running that type's own C slot on
    # the constant values. A slot reached through a base type's descriptor, e.g.
    # int.__and__(True, True), therefore gets the base type's result (1, not True).
    @staticmethod
    def _fold(
        tx: InstructionTranslatorBase, slot: Any, *args: VariableTracker
    ) -> VariableTracker:
        try:
            res = slot(*(a.as_python_constant() for a in args))
        except Exception as e:
            raise_observed_exception(type(e), tx, args=list(e.args))
        return ConstantVariable.create(res)

    @staticmethod
    def _unary_slot(slot: Any) -> SlotImpl:
        return lambda self, tx: ConstantVariable._fold(tx, slot, self)

    @staticmethod
    def _binary_slot(ty: type, name: str) -> SlotImpl:
        # A non-constant operand is left to its own (reflected) slot.
        fwd, rev = getattr(ty, f"__{name}__"), getattr(ty, f"__r{name}__")

        def impl(self, tx, other, reverse=False):  # type: ignore[no-untyped-def]
            if not other.is_python_constant():
                return ConstantVariable.create(NotImplemented)
            return ConstantVariable._fold(tx, rev if reverse else fwd, self, other)

        return impl

    @staticmethod
    def _power_slot(ty: type) -> SlotImpl:
        def impl(self, tx, other, z, reverse=False):  # type: ignore[no-untyped-def]
            operands = [other] if z is None else [other, z]
            if not all(a.is_python_constant() for a in operands):
                return ConstantVariable.create(NotImplemented)
            slot = ty.__rpow__ if reverse else ty.__pow__
            return ConstantVariable._fold(tx, slot, self, *operands)

        return impl

    @staticmethod
    def _richcompare_slot(ty: type) -> SlotImpl:
        def impl(self, tx, other, op):  # type: ignore[no-untyped-def]
            if not other.is_python_constant():
                return ConstantVariable.create(NotImplemented)
            return ConstantVariable._fold(tx, getattr(ty, op), self, other)

        return impl

    @staticmethod
    def _iter_slot(ty: type) -> SlotImpl:
        # Iterates with ty's own tp_iter, not self.unpack_var_sequence, which a
        # user subclass VT overrides with the full iteration protocol.
        def impl(self, tx):  # type: ignore[no-untyped-def]
            from .lists import ListIteratorVariable

            items: list[VariableTracker] = [
                ConstantVariable.create(c) for c in ty.__iter__(self.value)
            ]
            return ListIteratorVariable(items, mutation_type=ValueMutationNew())

        return impl

    @staticmethod
    def _sequence_slot(slot: Any, fallback: SlotImpl) -> SlotImpl:
        def impl(self, tx, arg):  # type: ignore[no-untyped-def]
            if not arg.is_python_constant():
                return fallback(self, tx, arg)
            return ConstantVariable._fold(tx, slot, self, arg)

        return impl

    @overload
    @staticmethod
    def create(value: None) -> ConstantVariable: ...

    @overload
    @staticmethod
    def create(value: bool) -> ConstantVariable: ...

    @overload
    @staticmethod
    def create(value: Any, **kwargs: Any) -> VariableTracker: ...

    @staticmethod
    def create(value: Any, **kwargs: Any) -> VariableTracker:
        """
        Create a `ConstantVariable` based on the given value, and supports
        automatic routing for collection types like `tuple` (in which case we'd
        create `ConstantVariable` for the leaf items).

        NOTE: the caller must install the proper guards if needed; most often
        the guard will be `CONSTANT_MATCH`.
        """
        # Return pre-allocated singleton constants when there are
        # no extra kwargs (source, etc.) that would differentiate the instance.
        if not kwargs:
            match value:
                case None:
                    return CONSTANT_VARIABLE_NONE
                case True:
                    return CONSTANT_VARIABLE_TRUE
                case False:
                    return CONSTANT_VARIABLE_FALSE
            if value is NotImplemented:
                return CONSTANT_VARIABLE_NOT_IMPLEMENTED

        source = kwargs.get("source")

        # Routing for supported collection literals.
        if isinstance(value, set):
            items = [ConstantVariable.create(x) for x in value]
            return variables.SetVariable(items, **kwargs)  # type: ignore[arg-type]
        elif isinstance(value, frozenset):
            items = [ConstantVariable.create(x) for x in value]
            return variables.FrozensetVariable(items, **kwargs)  # type: ignore[arg-type]
        elif isinstance(value, slice):
            slice_args = (value.start, value.stop, value.step)
            slice_args_vars: list[VariableTracker] = [
                ConstantVariable.create(arg) for arg in slice_args
            ]
            return variables.SliceVariable(slice_args_vars, **kwargs)
        elif isinstance(value, (list, tuple)):
            items = []
            for i, x in enumerate(value):
                item_source = GetItemSource(source, i) if source else None
                items.append(
                    ConstantVariable.create(
                        x,
                        source=item_source,
                    )
                )
            return variables.BaseListVariable.cls_for(type(value))(items, **kwargs)

        return ConstantVariable(value, **kwargs)

    def __init__(self, value: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not ConstantVariable.is_base_literal(value):
            raise AssertionError(
                f"Cannot construct `ConstantVariable` for value of type {type(value)}.\n"
                "\n"
                "This failure likely due to PyTorch-internal use of `ConstantVariable` on\n"
                "non-literal python values, please try using `VariableTracker.build` instead. If\n"
                "you believe it's a necessary and legitimate use case (the value is immutable and\n"
                "can't easily be represented with another `VariableTracker` class), please add\n"
                "its type to `common_constant_types`."
            )
        if np is not None and isinstance(value, np.number):
            self.value = value.item()
        else:
            self.value = value

    def as_proxy(self) -> Any:
        return self.value

    def __repr__(self) -> str:
        return f"ConstantVariable({type(self.value).__name__}: {repr(self.value)})"

    def as_python_constant(self) -> Any:
        return self.value

    def is_python_constant(self) -> Literal[True]:
        return True

    def tp_repr_impl(self, tx: InstructionTranslatorBase) -> VariableTracker:
        return ConstantVariable.create(repr(self.value))

    def is_symnode_like(self) -> bool:
        return isinstance(self.value, (int, bool))

    def is_constant_match(self, *values: Any) -> bool:
        return self.value in values

    def is_constant_none(self) -> bool:
        return self.value is None

    @staticmethod
    def is_base_literal(obj: object) -> bool:
        return type(obj) in common_constant_types

    @staticmethod
    def is_literal(obj: object, cache: dict[int, object] | None = None) -> bool:
        if cache is None:
            cache = {}
        if id(obj) in cache:
            # no-op if there is a cyclical reference
            return True
        if type(obj) in (list, tuple, set, frozenset, torch.Size):
            cache[id(obj)] = obj
            return all(ConstantVariable.is_literal(x, cache) for x in obj)  # type: ignore[attr-defined]
        return ConstantVariable.is_base_literal(obj)

    def tp_hash_impl(self, tx: InstructionTranslatorBase) -> tuple[int, bool]:
        """Dynamo tracing rule for long_hash, float_hash, unicode_hash, etc."""
        from torch.fx.experimental.proxy_tensor import _coor_enabled

        from .user_defined import _CONSTANT_BASE_TYPES

        if isinstance(self.value, torch.device) and _coor_enabled():
            # Drop the index so an explicit cuda:N lands in the same bucket as a
            # rank-relative CurrentDeviceVariable; see its tp_hash_impl.
            return hash(torch.device(self.value.type)), False

        value_type = cast(Any, type(self.value))
        mro = value_type.__mro__
        base = next((c for c in mro if c in _CONSTANT_BASE_TYPES), value_type)
        return base.__hash__(self.value), False

    def tp_richcompare_impl(
        self, tx: InstructionTranslatorBase, other: VariableTracker, op: str
    ) -> VariableTracker:
        from .object_protocol import python_constant_richcompare_impl

        return python_constant_richcompare_impl(self, tx, other, op)

    def tp_str_impl(self, tx: InstructionTranslatorBase) -> VariableTracker:
        return ConstantVariable.create(str(self.value))

    def const_getattr(
        self, tx: InstructionTranslatorBase, name: str
    ) -> VariableTracker:
        if not hasattr(self.value, name):
            raise_observed_exception(AttributeError, tx, args=[name])
        member = getattr(self.value, name)
        if callable(member):
            raise NotImplementedError
        return member

    def call_tree_map(
        self,
        tx: InstructionTranslatorBase,
        tree_map_fn: UserFunctionVariable,
        map_fn: VariableTracker,
        rest: list[VariableTracker],
        tree_map_kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if self.value is None:
            none_is_leaf_var = tree_map_kwargs.get("none_is_leaf")
            if none_is_leaf_var is not None:
                try:
                    none_is_leaf = bool(none_is_leaf_var.as_python_constant())
                except NotImplementedError:
                    return self._tree_map_fallback(
                        tx,
                        tree_map_fn,
                        map_fn,
                        rest,
                        tree_map_kwargs,
                    )
            else:
                tree_map_module = getattr(
                    getattr(tree_map_fn, "fn", None), "__module__", ""
                )
                # torch.utils._pytree and torch.utils._cxx_pytree treat None as a leaf
                # by default, while optree keeps it as an internal node unless
                # none_is_leaf=True is provided.
                none_is_leaf = not tree_map_module.startswith("optree")
            if none_is_leaf:
                return map_fn.call_function(tx, [self, *rest], {})
            else:
                for other in rest:
                    if not other.is_constant_none():
                        return self._tree_map_fallback(
                            tx,
                            tree_map_fn,
                            map_fn,
                            rest,
                            tree_map_kwargs,
                        )
                return self.clone()
        if isinstance(self.value, (int, float, bool, complex, str, bytes, torch.dtype)):
            return map_fn.call_function(tx, [self, *rest], {})
        return super().call_tree_map(
            tx,
            tree_map_fn,
            map_fn,
            rest,
            tree_map_kwargs,
        )

    def reconstruct_pycode(self, codegen) -> str:
        return repr(self.value)

    def get_id(self, tx: InstructionTranslatorBase) -> int | None:
        # Singletons have guaranteed stable identity across the process lifetime.
        if (
            self.value is None
            or self.value is True
            or self.value is False
            or self.value is NotImplemented
        ):
            return id(self.value)
        # Sourceful constants resolve via source like any other sourceful VT.
        # Sourceless non-singleton constants (e.g. literal 42 in compiled code)
        # get FakeIdVariable — CPython interning of small ints/strings is an
        # implementation detail users shouldn't rely on.
        return super().get_id(tx)

    def with_source_location(self, source_location: SourceLocation) -> VariableTracker:
        if (
            self is CONSTANT_VARIABLE_NONE
            or self is CONSTANT_VARIABLE_TRUE
            or self is CONSTANT_VARIABLE_FALSE
            or self is CONSTANT_VARIABLE_NOT_IMPLEMENTED
        ):
            return ConstantVariable.create(self.value, source_location=source_location)
        return super().with_source_location(source_location)

    def get_real_python_backed_value(self) -> object:
        return self.value


class IntConstantVariable(ConstantVariable):
    # PyLong_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c#L6585
    _cpython_type = int
    tp_methods, tp_getset, tp_members = constant_type_tables(int)

    tp_repr_impl = ConstantVariable._unary_slot(int.__repr__)
    tp_richcompare_impl = ConstantVariable._richcompare_slot(int)
    nb_bool_impl = ConstantVariable._unary_slot(int.__bool__)
    nb_index_impl = ConstantVariable._unary_slot(int.__index__)
    nb_int_impl = ConstantVariable._unary_slot(int.__int__)
    nb_float_impl = ConstantVariable._unary_slot(int.__float__)
    nb_negative_impl = ConstantVariable._unary_slot(int.__neg__)
    nb_positive_impl = ConstantVariable._unary_slot(int.__pos__)
    nb_absolute_impl = ConstantVariable._unary_slot(int.__abs__)
    nb_invert_impl = ConstantVariable._unary_slot(int.__invert__)
    nb_add_impl = ConstantVariable._binary_slot(int, "add")
    nb_subtract_impl = ConstantVariable._binary_slot(int, "sub")
    nb_multiply_impl = ConstantVariable._binary_slot(int, "mul")
    nb_floor_divide_impl = ConstantVariable._binary_slot(int, "floordiv")
    nb_true_divide_impl = ConstantVariable._binary_slot(int, "truediv")
    nb_remainder_impl = ConstantVariable._binary_slot(int, "mod")
    nb_divmod_impl = ConstantVariable._binary_slot(int, "divmod")
    nb_lshift_impl = ConstantVariable._binary_slot(int, "lshift")
    nb_rshift_impl = ConstantVariable._binary_slot(int, "rshift")
    nb_and_impl = ConstantVariable._binary_slot(int, "and")
    nb_or_impl = ConstantVariable._binary_slot(int, "or")
    nb_xor_impl = ConstantVariable._binary_slot(int, "xor")
    nb_power_impl = ConstantVariable._power_slot(int)

    def nb_power_z_impl(
        self,
        tx: InstructionTranslatorBase,
        v: VariableTracker,
        w: VariableTracker,
    ) -> VariableTracker:
        # pow(v, w, self): ternary_op tries the modulus's slot last.
        if not v.is_python_constant() or not w.is_python_constant():
            return ConstantVariable.create(NotImplemented)
        return ConstantVariable._fold(tx, int.__pow__, v, w, self)


class BoolConstantVariable(IntConstantVariable):
    # PyBool_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/boolobject.c#L171
    # tp_base is PyLong_Type; bool_repr and bool_as_number override int's.
    _cpython_type = bool
    tp_methods, tp_getset, tp_members = constant_type_tables(bool)

    tp_repr_impl = ConstantVariable._unary_slot(bool.__repr__)
    nb_invert_impl = ConstantVariable._unary_slot(bool.__invert__)
    nb_and_impl = ConstantVariable._binary_slot(bool, "and")
    nb_or_impl = ConstantVariable._binary_slot(bool, "or")
    nb_xor_impl = ConstantVariable._binary_slot(bool, "xor")


class FloatConstantVariable(ConstantVariable):
    # PyFloat_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/floatobject.c#L1880
    _cpython_type = float
    tp_methods, tp_getset, tp_members = constant_type_tables(float)

    tp_repr_impl = ConstantVariable._unary_slot(float.__repr__)
    tp_richcompare_impl = ConstantVariable._richcompare_slot(float)
    nb_bool_impl = ConstantVariable._unary_slot(float.__bool__)
    nb_int_impl = ConstantVariable._unary_slot(float.__int__)
    nb_float_impl = ConstantVariable._unary_slot(float.__float__)
    nb_negative_impl = ConstantVariable._unary_slot(float.__neg__)
    nb_positive_impl = ConstantVariable._unary_slot(float.__pos__)
    nb_absolute_impl = ConstantVariable._unary_slot(float.__abs__)
    nb_add_impl = ConstantVariable._binary_slot(float, "add")
    nb_subtract_impl = ConstantVariable._binary_slot(float, "sub")
    nb_multiply_impl = ConstantVariable._binary_slot(float, "mul")
    nb_floor_divide_impl = ConstantVariable._binary_slot(float, "floordiv")
    nb_true_divide_impl = ConstantVariable._binary_slot(float, "truediv")
    nb_remainder_impl = ConstantVariable._binary_slot(float, "mod")
    nb_divmod_impl = ConstantVariable._binary_slot(float, "divmod")
    nb_power_impl = ConstantVariable._power_slot(float)


class ComplexConstantVariable(ConstantVariable):
    # PyComplex_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/complexobject.c#L1099
    _cpython_type = complex
    tp_methods, tp_getset, tp_members = constant_type_tables(complex)

    tp_repr_impl = ConstantVariable._unary_slot(complex.__repr__)
    tp_richcompare_impl = ConstantVariable._richcompare_slot(complex)
    nb_bool_impl = ConstantVariable._unary_slot(complex.__bool__)
    nb_negative_impl = ConstantVariable._unary_slot(complex.__neg__)
    nb_positive_impl = ConstantVariable._unary_slot(complex.__pos__)
    nb_absolute_impl = ConstantVariable._unary_slot(complex.__abs__)
    nb_add_impl = ConstantVariable._binary_slot(complex, "add")
    nb_subtract_impl = ConstantVariable._binary_slot(complex, "sub")
    nb_multiply_impl = ConstantVariable._binary_slot(complex, "mul")
    nb_true_divide_impl = ConstantVariable._binary_slot(complex, "truediv")
    nb_power_impl = ConstantVariable._power_slot(complex)


class StrConstantVariable(ConstantVariable):
    # PyUnicode_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/unicodeobject.c#L14931
    _cpython_type = str
    _tp_methods, tp_getset, tp_members = constant_type_tables(str)

    tp_repr_impl = ConstantVariable._unary_slot(str.__repr__)
    tp_richcompare_impl = ConstantVariable._richcompare_slot(str)
    nb_remainder_impl = ConstantVariable._binary_slot(str, "mod")
    mp_subscript_impl = ConstantVariable._sequence_slot(
        str.__getitem__, VariableTracker.mp_subscript_impl
    )
    sq_item_impl = ConstantVariable._sequence_slot(
        str.__getitem__, VariableTracker.sq_item_impl
    )
    sq_length_impl = ConstantVariable._unary_slot(str.__len__)
    mp_length_impl = ConstantVariable._unary_slot(str.__len__)
    sq_contains_impl = ConstantVariable._sequence_slot(
        str.__contains__, VariableTracker.sq_contains_impl
    )
    tp_iter_impl = ConstantVariable._iter_slot(str)
    sq_concat_impl = ConstantVariable._sequence_slot(
        str.__add__, VariableTracker.sq_concat_impl
    )
    sq_repeat_impl = ConstantVariable._sequence_slot(
        str.__mul__, VariableTracker.sq_repeat_impl
    )

    def unpack_var_sequence(
        self, tx: InstructionTranslatorBase | None
    ) -> list[VariableTracker]:
        return [ConstantVariable.create(c) for c in str.__iter__(self.value)]

    def join_method(
        self,
        tx: InstructionTranslatorBase,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker | None:
        # str.join consumes any iterable, including lazy ones (generators, map)
        # that the table's constant fold declines; unpack it, then fold str.join.
        items = unpack_iterable(tx, args[0])
        if not all(item.is_python_constant() for item in items):
            return None
        return ConstantVariable._fold(
            tx, str.join, self, variables.TupleVariable(items)
        )

    def format_method(
        self,
        tx: InstructionTranslatorBase,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # str.format with non-constant arguments (e.g. a tensor) builds a
        # StringFormatVariable through the str.format builtin.
        from .builder import SourcelessBuilder

        fmt = SourcelessBuilder.create_internal_builtin(str.format)
        return fmt.call_function(tx, [self, *args], kwargs)

    tp_methods = {
        **_tp_methods,
        "join": Method(join_method),
        "format": Method(format_method),
    }


class BytesConstantVariable(ConstantVariable):
    # PyBytes_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/bytesobject.c#L3017
    _cpython_type = bytes
    tp_methods, tp_getset, tp_members = constant_type_tables(bytes)

    tp_repr_impl = ConstantVariable._unary_slot(bytes.__repr__)
    tp_richcompare_impl = ConstantVariable._richcompare_slot(bytes)
    nb_remainder_impl = ConstantVariable._binary_slot(bytes, "mod")
    mp_subscript_impl = ConstantVariable._sequence_slot(
        bytes.__getitem__, VariableTracker.mp_subscript_impl
    )
    sq_item_impl = ConstantVariable._sequence_slot(
        bytes.__getitem__, VariableTracker.sq_item_impl
    )
    sq_length_impl = ConstantVariable._unary_slot(bytes.__len__)
    mp_length_impl = ConstantVariable._unary_slot(bytes.__len__)
    sq_contains_impl = ConstantVariable._sequence_slot(
        bytes.__contains__, VariableTracker.sq_contains_impl
    )
    tp_iter_impl = ConstantVariable._iter_slot(bytes)
    sq_concat_impl = ConstantVariable._sequence_slot(
        bytes.__add__, VariableTracker.sq_concat_impl
    )
    sq_repeat_impl = ConstantVariable._sequence_slot(
        bytes.__mul__, VariableTracker.sq_repeat_impl
    )

    def unpack_var_sequence(
        self, tx: InstructionTranslatorBase | None
    ) -> list[VariableTracker]:
        return [ConstantVariable.create(c) for c in bytes.__iter__(self.value)]


class NoneConstantVariable(ConstantVariable):
    # _PyNone_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/object.c#L2022
    _cpython_type = type(None)
    tp_methods, tp_getset, tp_members = constant_type_tables(type(None))

    tp_repr_impl = ConstantVariable._unary_slot(type(None).__repr__)
    nb_bool_impl = ConstantVariable._unary_slot(type(None).__bool__)


class EllipsisConstantVariable(ConstantVariable):
    # PyEllipsis_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/sliceobject.c#L77
    _cpython_type = type(...)
    tp_methods, tp_getset, tp_members = constant_type_tables(type(...))

    tp_repr_impl = ConstantVariable._unary_slot(type(...).__repr__)


_CONSTANT_VT_FOR_TYPE: dict[type, type[ConstantVariable]] = {
    int: IntConstantVariable,
    bool: BoolConstantVariable,
    float: FloatConstantVariable,
    complex: ComplexConstantVariable,
    str: StrConstantVariable,
    bytes: BytesConstantVariable,
    type(None): NoneConstantVariable,
    type(...): EllipsisConstantVariable,
}
_CONSTANT_VT_CLASSES = frozenset((ConstantVariable, *_CONSTANT_VT_FOR_TYPE.values()))


# These interned singletons back ConstantVariable.create itself.
CONSTANT_VARIABLE_NONE = ConstantVariable(None)  # noqa: RVT001
CONSTANT_VARIABLE_TRUE = ConstantVariable(True)  # noqa: RVT001
CONSTANT_VARIABLE_FALSE = ConstantVariable(False)  # noqa: RVT001
CONSTANT_VARIABLE_NOT_IMPLEMENTED = ConstantVariable(NotImplemented)  # noqa: RVT001


class FakeValueKind(enum.Enum):
    ID = "id"
    HASH = "hash"


class FakeIdVariable(VariableTracker):
    """A compile-time-only id or hash value that can be used as a dict key but
    cannot be reconstructed across graph breaks.

    When dynamo evaluates ``id(x)`` or ``hash(x)`` on a variable tracker that
    has no corresponding runtime object, we mint a fake integer.  The ``kind``
    field tracks which builtin produced the value so that same-kind comparisons
    (e.g. ``id(a) != id(b)``) can be resolved at compile time while cross-kind
    comparisons graph-break.
    """

    # PyLong_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c#L6585
    _cpython_type = int

    _nonvar_fields = {
        "kind",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self, value: int, *, kind: FakeValueKind = FakeValueKind.ID, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.value = value
        self.kind = kind

    @staticmethod
    def _create(value: int, kind: FakeValueKind = FakeValueKind.ID) -> FakeIdVariable:
        from .builder import SourcelessBuilder

        return SourcelessBuilder.create_internal_fake_id(value, kind)

    def as_python_constant(self) -> int:
        return self.value

    def is_python_constant(self) -> bool:
        return False

    def python_type(self) -> type:
        return int

    def tp_hash_impl(self, tx: InstructionTranslatorBase) -> tuple[int, bool]:
        return hash(self.value), True

    def tp_repr_impl(self, tx: InstructionTranslatorBase) -> VariableTracker:
        # Mirrors int.__repr__: the value is an int, so str()/repr() yield its
        # decimal string. The distinct-but-compile-time-only identity carried by
        # the fake id is preserved in the resulting string, matching how
        # FakeIdVariable already resolves same-kind id()/hash() comparisons.
        return ConstantVariable.create(repr(self.value))

    def nb_bool_impl(self, tx: InstructionTranslatorBase) -> VariableTracker:
        # Mirrors long_bool. The fake value is only meaningful at compile time,
        # but its truthiness is, like tp_repr_impl, a plain function of it.
        return ConstantVariable.create(bool(self.value))

    def tp_richcompare_impl(
        self, tx: InstructionTranslatorBase, other: VariableTracker, op: str
    ) -> VariableTracker:
        if (
            isinstance(other, FakeIdVariable)
            and self.kind == other.kind
            and op in ("__eq__", "__ne__")
        ):
            result = (
                (self.value == other.value)
                if op == "__eq__"
                else (self.value != other.value)
            )
            return ConstantVariable.create(result)
        unimplemented(
            gb_type="Comparison on compile-time-only id or hash value",
            context=f"FakeIdVariable({self.value}) {op} {type(other).__name__}",
            explanation="Cannot compare a compile-time-only id() or hash() "
            "value. The comparison will run eagerly.",
            hints=[
                "Avoid comparing id() or hash() of objects created inside "
                "the compiled region against other values.",
                *graph_break_hints.SUPPORTABLE,
            ],
        )

    # Arithmetic on a fake id/hash (e.g. ``id(self) & 0x7fffffff`` inside a
    # custom __hash__) stays compile-time-only: CPython computes int op int,
    # but the operand is a fake address so the result must not be baked into
    # the graph. We mirror int's nb_* slots and return another fake int.
    def _operand_int(self, other: VariableTracker) -> int | None:
        if isinstance(other, FakeIdVariable):
            return other.value
        if other.is_python_constant():
            val = other.as_python_constant()
            if isinstance(val, int):
                return val
        return None

    def _nb_binary_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        op: Any,
        reverse: bool,
    ) -> VariableTracker:
        other_val = self._operand_int(other)
        if other_val is None:
            return ConstantVariable.create(NotImplemented)
        lhs, rhs = (other_val, self.value) if reverse else (self.value, other_val)
        try:
            result = op(lhs, rhs)
        except (TypeError, ValueError, OverflowError, ZeroDivisionError) as e:
            raise_observed_exception(type(e), tx, args=list(e.args))
        return self._create(result)

    def nb_add_impl(self, tx, other, reverse=False):  # type: ignore[no-untyped-def]
        return self._nb_binary_impl(tx, other, operator.add, reverse)

    def nb_subtract_impl(self, tx, other, reverse=False):  # type: ignore[no-untyped-def]
        return self._nb_binary_impl(tx, other, operator.sub, reverse)

    def nb_multiply_impl(self, tx, other, reverse=False):  # type: ignore[no-untyped-def]
        return self._nb_binary_impl(tx, other, operator.mul, reverse)

    def nb_floor_divide_impl(self, tx, other, reverse=False):  # type: ignore[no-untyped-def]
        return self._nb_binary_impl(tx, other, operator.floordiv, reverse)

    def nb_remainder_impl(self, tx, other, reverse=False):  # type: ignore[no-untyped-def]
        return self._nb_binary_impl(tx, other, operator.mod, reverse)

    def nb_and_impl(self, tx, other, reverse=False):  # type: ignore[no-untyped-def]
        return self._nb_binary_impl(tx, other, operator.and_, reverse)

    def nb_or_impl(self, tx, other, reverse=False):  # type: ignore[no-untyped-def]
        return self._nb_binary_impl(tx, other, operator.or_, reverse)

    def nb_xor_impl(self, tx, other, reverse=False):  # type: ignore[no-untyped-def]
        return self._nb_binary_impl(tx, other, operator.xor, reverse)

    def nb_lshift_impl(self, tx, other, reverse=False):  # type: ignore[no-untyped-def]
        return self._nb_binary_impl(tx, other, operator.lshift, reverse)

    def nb_rshift_impl(self, tx, other, reverse=False):  # type: ignore[no-untyped-def]
        return self._nb_binary_impl(tx, other, operator.rshift, reverse)

    def nb_negative_impl(self, tx):  # type: ignore[no-untyped-def]
        return self._create(-self.value)

    def nb_positive_impl(self, tx):  # type: ignore[no-untyped-def]
        return self._create(+self.value)

    def nb_absolute_impl(self, tx):  # type: ignore[no-untyped-def]
        return self._create(abs(self.value))

    def nb_invert_impl(self, tx):  # type: ignore[no-untyped-def]
        return self._create(~self.value)

    def nb_int_impl(self, tx):  # type: ignore[no-untyped-def]
        return self._create(int(self.value), kind=self.kind)

    def nb_index_impl(self, tx):  # type: ignore[no-untyped-def]
        return self._create(self.value, kind=self.kind)

    def reconstruct(self, codegen: Any) -> None:
        unimplemented(
            gb_type="Reconstruction of FakeIdVariable",
            context=str(self.value),
            explanation=(
                "A fake id produced by id() on a compile-time container "
                "cannot be reconstructed across a graph break."
            ),
            hints=[
                "Avoid using id() on containers in code that may graph-break.",
            ],
        )
