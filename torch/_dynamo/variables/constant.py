"""
Constant variable tracking in Dynamo.

This module is fundamental to Dynamo's ability to track and propagate constant
values during compilation, ensuring proper handling of Python literals and
maintaining type safety through the compilation process.
"""

from __future__ import annotations

import enum
import math
import operator
from collections.abc import Iterable
from typing import Any, Literal, overload, TYPE_CHECKING
from typing_extensions import override

import torch
from torch._dynamo.bytecode_transformation import create_call_function
from torch._dynamo.source import AttrSource, GetItemSource

from .. import graph_break_hints, variables
from ..exc import raise_observed_exception, unimplemented
from ..utils import (
    common_constant_types,
    istype,
    np,
    raise_args_mismatch,
    unpack_iterable,
)
from .base import ValueMutationNew, VariableTracker


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase

    from .functions import UserFunctionVariable


class ConstantVariable(VariableTracker):
    """
    Variable tracker for Python literals and basic immutable types, with automatic
    routing support for collection types (lists, tuples, sets, etc.).

    The create() method intelligently constructs appropriate variable types for
    nested collections.
    """

    # PyLong_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c#L6585
    # PyFloat_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/floatobject.c#L1880
    # PyBool_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/boolobject.c#L171
    # PyUnicode_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/unicodeobject.c#L14931
    # PyBytes_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/bytesobject.c#L3017
    # PyComplex_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/complexobject.c#L1099
    # _PyNone_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/object.c#L2022
    _cpython_type = (int, float, str, bytes, bool, type(None), complex, type(...))

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
        # Return pre-allocated sentinels for None/True/False when there are
        # no extra kwargs (source, etc.) that would differentiate the instance.
        if not kwargs:
            match value:
                case None:
                    return CONSTANT_VARIABLE_NONE
                case True:
                    return CONSTANT_VARIABLE_TRUE
                case False:
                    return CONSTANT_VARIABLE_FALSE

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

    def __init__(self, value: Any, **kwargs: Any) -> None:
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

    def repr_impl(self, tx: InstructionTranslatorBase) -> VariableTracker:
        return ConstantVariable.create(repr(self.value))

    def is_symnode_like(self) -> bool:
        return isinstance(self.value, (int, bool))

    def is_constant_match(self, *values: Any) -> bool:
        return self.value in values

    def is_constant_none(self) -> bool:
        return self.value is None

    @property
    def items(self) -> list[VariableTracker]:
        """
        Need this when adding a BaseListVariable and a ConstantVariable together.
        Happens in detectron2.
        """
        try:
            return [ConstantVariable.create(x) for x in self.value]
        except TypeError as e:
            raise NotImplementedError from e

    def getitem_const(
        self, tx: InstructionTranslatorBase, arg: VariableTracker
    ) -> VariableTracker:
        if isinstance(self.value, (str, bytes)):
            from .object_protocol import validate_sequence_index

            container_name = "string" if isinstance(self.value, str) else "bytes"
            arg = validate_sequence_index(tx, arg, container_name)
        return ConstantVariable.create(
            self.value[arg.as_python_constant()],
        )

    def sq_item_impl(
        self, tx: InstructionTranslatorBase, key: VariableTracker
    ) -> VariableTracker:
        # unicode_getitem: https://github.com/python/cpython/blob/62a6e898e01/Objects/unicodeobject.c#L13777
        # bytes_item: https://github.com/python/cpython/blob/62a6e898e01/Objects/bytesobject.c#L319
        # CPython's sq_item takes Py_ssize_t (already int from vt_getitem's
        # nb_index_impl).  Unlike mp_subscript, sq_item never handles slices.
        index = key.as_python_constant()
        try:
            return ConstantVariable.create(self.value[index])
        except IndexError as e:
            raise_observed_exception(IndexError, tx, args=list(e.args))

    def tp_iteritem_impl(
        self, tx: InstructionTranslatorBase, index: VariableTracker
    ) -> tuple[VariableTracker, VariableTracker]:
        # unicode_iteritem: https://github.com/python/cpython/blob/f31a89bb9010/Objects/unicodeobject.c#L13994
        # bytes_iteritem:   https://github.com/python/cpython/blob/f31a89bb9010/Objects/bytesobject.c#L3210
        if not isinstance(self.value, (str, bytes, list, tuple)):
            return super().tp_iteritem_impl(tx, index)
        i = index.as_python_constant()
        if i < 0:
            raise AssertionError(f"Invalid index {i}")
        if i >= len(self.value):
            raise_observed_exception(IndexError, tx)
        return ConstantVariable.create(self.value[i]), ConstantVariable.create(i + 1)

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

    def unpack_var_sequence(
        self, tx: InstructionTranslatorBase | None
    ) -> list[VariableTracker]:
        try:
            return [ConstantVariable.create(x) for x in self.as_python_constant()]
        except TypeError as e:
            raise NotImplementedError from e

    def hash_impl(self, tx: InstructionTranslatorBase) -> tuple[int, bool]:
        """Dynamo tracing rule for long_hash, float_hash, unicode_hash, etc."""
        return hash(self.value), False

    def richcompare_impl(
        self, tx: InstructionTranslatorBase, other: VariableTracker, op: str
    ) -> VariableTracker:
        from .object_protocol import python_constant_richcompare_impl

        return python_constant_richcompare_impl(self, tx, other, op)

    def len_impl(self, tx: InstructionTranslatorBase) -> VariableTracker:
        """Generic len for any constant value (sequence or mapping)."""
        try:
            return ConstantVariable.create(len(self.value))
        except TypeError as e:
            raise_observed_exception(type(e), tx, args=list(e.args))

    def sq_length(self, tx: InstructionTranslatorBase) -> VariableTracker:
        """Sequence length - delegates to len_impl for constants."""
        return self.len_impl(tx)

    def mp_length(self, tx: InstructionTranslatorBase) -> VariableTracker:
        """Mapping length - delegates to len_impl for constants."""
        return self.len_impl(tx)

    def const_getattr(
        self, tx: InstructionTranslatorBase, name: str
    ) -> VariableTracker:
        if not hasattr(self.value, name):
            raise_observed_exception(AttributeError, tx, args=[name])
        member = getattr(self.value, name)
        if callable(member):
            raise NotImplementedError
        return member

    def sq_contains(self, tx: InstructionTranslatorBase, item: VariableTracker):
        """Sequence contains for constants."""
        if item.is_python_constant():
            search = item.as_python_constant()
            try:
                result = search in self.value
                return ConstantVariable.create(result)
            except TypeError as e:
                raise_observed_exception(
                    type(e),
                    tx,
                    args=list(e.args),
                )
        return super().sq_contains(tx, item)

    def tp_iter_impl(self, tx: InstructionTranslatorBase) -> VariableTracker:
        from .lists import ListIteratorVariable

        if isinstance(self.value, Iterable):
            try:
                return ListIteratorVariable(
                    [ConstantVariable.create(c) for c in self.value],
                    mutation_type=ValueMutationNew(),
                )
            except NotImplementedError:
                pass
        return super().tp_iter_impl(tx)

    def call_method(
        self,
        tx: InstructionTranslatorBase,
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from .tensor import SymNodeVariable

        if name == "format" and istype(self.value, str):
            return variables.BuiltinVariable(str.format).call_function(
                tx,
                [self, *args],
                kwargs,
            )
        elif name == "join" and istype(self.value, str):
            if kwargs or len(args) != 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            arg_unpacked = unpack_iterable(tx, args[0])
            try:
                arg_const = [x.as_python_constant() for x in arg_unpacked]
                return ConstantVariable.create(self.value.join(arg_const))
            except NotImplementedError:
                return super().call_method(tx, name, args, kwargs)

        if any(isinstance(x, SymNodeVariable) for x in args):
            # Promote to SymNodeVariable for operations involving dynamic shapes.
            return variables.SymNodeVariable.create(
                tx, self.as_proxy(), self.value
            ).call_method(tx, name, args, kwargs)

        try:
            const_args = [a.as_python_constant() for a in args]
            const_kwargs = {k: v.as_python_constant() for k, v in kwargs.items()}
        except NotImplementedError:
            return super().call_method(tx, name, args, kwargs)

        if name == "__iter__":
            return self.tp_iter_impl(tx)

        if isinstance(self.value, str) and name in str.__dict__:
            method = getattr(self.value, name)
            try:
                return ConstantVariable.create(method(*const_args, **const_kwargs))
            except Exception as e:
                raise_observed_exception(type(e), tx)
        elif isinstance(self.value, (float, int)) and hasattr(self.value, name):
            if not (args or kwargs):
                try:
                    return ConstantVariable.create(getattr(self.value, name)())
                except (OverflowError, ValueError) as exc:
                    raise_observed_exception(
                        type(exc),
                        tx,
                        args=list(exc.args),
                    )
            if (
                hasattr(operator, name)
                and len(args) == 1
                and args[0].is_python_constant()
            ):
                add_target = const_args[0]
                op = getattr(operator, name)
                if isinstance(
                    add_target, (torch.SymBool, torch.SymFloat, torch.SymInt)
                ):
                    # Addition between a non sym and sym makes a sym
                    proxy = tx.output.create_proxy(
                        "call_function", op, (self.value, add_target), {}
                    )
                    return SymNodeVariable.create(tx, proxy, add_target)
                else:
                    try:
                        return ConstantVariable.create(op(self.value, add_target))
                    except Exception as e:
                        raise_observed_exception(type(e), tx, args=list(e.args))
        elif isinstance(self.value, bytes) and name == "decode":
            method = getattr(self.value, name)
            return ConstantVariable.create(method(*const_args, **const_kwargs))
        elif type(self.value) is complex and name in complex.__dict__:
            method = getattr(self.value, name)
            try:
                return ConstantVariable.create(method(*const_args, **const_kwargs))
            except Exception as e:
                raise_observed_exception(type(e), tx)

        if name == "__round__" and len(args) == 1 and args[0].is_python_constant():
            try:
                return ConstantVariable.create(
                    round(self.value, args[0].as_python_constant())
                )
            except Exception as e:
                raise_observed_exception(type(e), tx, args=list(e.args))
        return super().call_method(tx, name, args, kwargs)

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

    @override
    def call_obj_hasattr(
        self, tx: InstructionTranslatorBase, name: str
    ) -> ConstantVariable:
        result = hasattr(self.value, name)
        return variables.ConstantVariable.create(result)

    def is_python_equal(self, other: object) -> bool:
        from .tensor import SymNodeVariable

        if isinstance(other, SymNodeVariable):
            return self.as_python_constant() == other.evaluate_expr()
        return (
            isinstance(other, VariableTracker)
            and self.as_python_constant() == other.as_python_constant()
        )

    def get_id(self, tx: InstructionTranslatorBase) -> int | None:
        # Singletons have guaranteed stable identity across the process lifetime.
        if self.value is None or self.value is True or self.value is False:
            return id(self.value)
        # Sourceful constants resolve via source like any other sourceful VT.
        # Sourceless non-singleton constants (e.g. literal 42 in compiled code)
        # get FakeIdVariable — CPython interning of small ints/strings is an
        # implementation detail users shouldn't rely on.
        return super().get_id(tx)

    def get_real_python_backed_value(self) -> object:
        return self.value

    def nb_index_impl(
        self,
        tx: Any,
    ) -> VariableTracker:
        # CPython: int and bool define nb_index (returns self for int,
        # int(self) for bool). All other constant types do not.
        from .object_protocol import type_implements_nb_index

        if type_implements_nb_index(type(self.value)):
            return ConstantVariable.create(operator.index(self.value))
        return super().nb_index_impl(tx)

    def nb_int_impl(
        self,
        tx: Any,
    ) -> VariableTracker:
        # CPython: int defines nb_int (long_long, returns copy).
        # bool inherits nb_int from int via slot inheritance.
        # float defines nb_int (truncates toward zero via PyLong_FromDouble).
        return ConstantVariable.create(int(self.value))

    def nb_float_impl(
        self,
        tx: Any,
    ) -> VariableTracker:
        # CPython: float defines nb_float (float_float, returns copy).
        # int defines nb_float (long_float, converts to float).
        # bool inherits nb_float from int via slot inheritance.
        return ConstantVariable.create(float(self.value))

    def _nb_binary_impl(
        self,
        tx: Any,
        other: VariableTracker,
        op: Any,
        type_check: Any,
        reverse: bool,
    ) -> VariableTracker:
        # Shared body for ConstantVariable's binary nb_* slots. Mirrors the
        # CPython contract: if the type doesn't implement the slot, return
        # NotImplemented so dispatch can try the reverse slot; otherwise run
        # ``op(v, w)`` and re-raise arithmetic exceptions as observed.
        if not type_check(type(self.value)):
            return ConstantVariable.create(NotImplemented)
        if not other.is_python_constant():
            return ConstantVariable.create(NotImplemented)
        self_, other_ = (other, self) if reverse else (self, other)
        v, w = self_.as_python_constant(), other_.as_python_constant()
        try:
            result = op(v, w)
        except (TypeError, ValueError, OverflowError) as e:
            raise_observed_exception(type(e), tx, args=list(e.args))
        if result is NotImplemented:
            return ConstantVariable.create(NotImplemented)
        return VariableTracker.build(tx, result)

    def nb_lshift_impl(
        self,
        tx: Any,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # CPython: only int defines nb_lshift; bool inherits via slot inheritance.
        # https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c#L5489 (long_lshift)
        from .object_protocol import type_implements_nb_lshift

        return self._nb_binary_impl(
            tx, other, operator.lshift, type_implements_nb_lshift, reverse
        )

    def nb_rshift_impl(
        self,
        tx: Any,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # CPython: only int defines nb_rshift; bool inherits via slot inheritance.
        # https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c#L5526 (long_rshift)
        from .object_protocol import type_implements_nb_rshift

        return self._nb_binary_impl(
            tx, other, operator.rshift, type_implements_nb_rshift, reverse
        )

    def nb_or_impl(
        self,
        tx: Any,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # CPython: int, frozenset, and type all define nb_or.
        # https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c#L5606 (long_or)
        # https://github.com/python/cpython/blob/v3.13.0/Objects/setobject.c#L1319 (set_or)
        # https://github.com/python/cpython/blob/v3.13.0/Objects/typeobject.c#L6028-L6030 (type_as_number.nb_or)
        # bool inherits int's nb_or via slot inheritance.
        from .object_protocol import type_implements_nb_or

        return self._nb_binary_impl(
            tx, other, operator.or_, type_implements_nb_or, reverse
        )

    def nb_subtract_impl(
        self,
        tx: Any,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # CPython: int, float, and complex define nb_subtract; bool inherits int's.
        # https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c#L3819-L3824 (long_sub_method)
        # https://github.com/python/cpython/blob/v3.13.0/Objects/floatobject.c#L598-L606 (float_sub)
        # https://github.com/python/cpython/blob/v3.13.0/Objects/complexobject.c#L494-L503 (COMPLEX_BINOP(sub, diff))
        from .object_protocol import type_implements_nb_subtract

        return self._nb_binary_impl(
            tx, other, operator.sub, type_implements_nb_subtract, reverse
        )

    def nb_multiply_impl(
        self,
        tx: InstructionTranslatorBase,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # int, float, and complex all define nb_multiply (bool inherits int's slot).
        # https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c#L4242-L4260 (long_mul)
        # https://github.com/python/cpython/blob/v3.13.0/Objects/floatobject.c#L608-L616 (float_mul)
        # https://github.com/python/cpython/blob/v3.13.0/Objects/complexobject.c#L506 (complex_mul)
        # str/bytes/bytearray do NOT have nb_multiply — they go through sq_repeat,
        # so this method should not see them as ``self``.
        from .object_protocol import type_implements_nb_multiply

        if not other.is_python_constant():
            return ConstantVariable.create(NotImplemented)
        other_val = other.as_python_constant()
        # CPython's nb_multiply (e.g. long_mul, float_mul) returns NotImplemented
        # whenever the other operand isn't numeric — sequence repetition is
        # then handled by PyNumber_Multiply's sq_repeat fallback. We mirror that
        # here because ``operator.mul`` performs the full protocol (including
        # sq_repeat) and would incorrectly short-circuit the dispatch.
        if not isinstance(other_val, (int, float, complex)):
            return ConstantVariable.create(NotImplemented)
        return self._nb_binary_impl(
            tx, other, operator.mul, type_implements_nb_multiply, reverse
        )

    def sq_repeat_impl(
        self,
        tx: InstructionTranslatorBase,
        count: VariableTracker,
    ) -> VariableTracker:
        # Only str / bytes are reachable via ConstantVariable since list, tuple,
        # bytearray have their own VTs.  ``count`` was already validated as an
        # index by sequence_repeat -> nb_index_impl.
        # https://github.com/python/cpython/blob/v3.13.0/Objects/unicodeobject.c#L12371 (unicode_repeat)
        # https://github.com/python/cpython/blob/v3.13.0/Objects/bytesobject.c#L1448 (bytes_repeat)
        if not isinstance(self.value, (str, bytes)):
            raise AssertionError("Expected str or bytes in sq_repeat_impl")
        n = count.as_python_constant()
        try:
            return ConstantVariable.create(self.value * n)
        except (MemoryError, OverflowError) as e:
            raise_observed_exception(type(e), tx, args=list(e.args))

    def nb_and_impl(
        self,
        tx: Any,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # CPython: int, frozenset, and type all define nb_and.
        # https://github.com/python/cpython/blob/3.13/Objects/longobject.c#L5574 (long_and)
        # https://github.com/python/cpython/blob/3.13/Objects/setobject.c#L1506-L1518 (set_and)
        # bool inherits int's nb_and via slot inheritance.
        from .object_protocol import type_implements_nb_and

        return self._nb_binary_impl(
            tx, other, operator.and_, type_implements_nb_and, reverse
        )

    def nb_xor_impl(
        self,
        tx: Any,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # CPython: int and frozenset define nb_xor.
        # https://github.com/python/cpython/blob/3.13/Objects/longobject.c#L5587 (long_xor)
        # https://github.com/python/cpython/blob/3.13/Objects/setobject.c#L1984-L1990 (set_xor)
        # bool inherits int's nb_xor via slot inheritance.
        from .object_protocol import type_implements_nb_xor

        return self._nb_binary_impl(
            tx, other, operator.xor, type_implements_nb_xor, reverse
        )

    def nb_negative_impl(
        self,
        tx: Any,
    ) -> VariableTracker:
        # int: https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c#L5179-L5189
        # float: https://github.com/python/cpython/blob/v3.13.0/Objects/floatobject.c#L839-L849
        # complex: https://github.com/python/cpython/blob/v3.13.0/Objects/complexobject.c#L569-L575
        # bool inherits nb_negative from int via slot inheritance.
        return ConstantVariable.create(-self.value)

    def nb_positive_impl(
        self,
        tx: Any,
    ) -> VariableTracker:
        # int: https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c#L5619 (long_long)
        # float: https://github.com/python/cpython/blob/v3.13.0/Objects/floatobject.c#L1114 (float_float)
        # complex: https://github.com/python/cpython/blob/v3.13.0/Objects/complexobject.c#L578 (complex_pos)
        # bool inherits nb_positive from int via slot inheritance.
        return ConstantVariable.create(+self.value)

    def nb_add_impl(
        self,
        tx: Any,
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # CPython: int, float, and complex define nb_add; bool inherits int's.
        # https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c#L3800 (long_add)
        # https://github.com/python/cpython/blob/v3.13.0/Objects/floatobject.c#L559 (float_add)
        # https://github.com/python/cpython/blob/v3.13.0/Objects/complexobject.c#L720 (COMPLEX_BINOP(add, sum))
        from .object_protocol import type_implements_nb_add

        return self._nb_binary_impl(
            tx, other, operator.add, type_implements_nb_add, reverse
        )

    def nb_absolute_impl(
        self,
        tx: Any,
    ) -> VariableTracker:
        # int: https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c#L5184-L5190
        # float: https://github.com/python/cpython/blob/v3.13.0/Objects/floatobject.c#L847-L850
        # complex: https://github.com/python/cpython/blob/v3.13.0/Objects/complexobject.c#L588-L600
        #   _Py_c_abs can set errno=ERANGE on overflow, which complex_abs
        #   converts to OverflowError("absolute value too large").
        # bool inherits nb_absolute from int via slot inheritance.
        try:
            return ConstantVariable.create(abs(self.value))
        except OverflowError as e:
            raise_observed_exception(OverflowError, tx, args=list(e.args))


class NumpyScalarVariable(ConstantVariable):
    """
    Represents a NumPy scalar specialized by value.

    Tensor operations should see the equivalent Python scalar, while Python-level
    type checks and reconstruction still observe the original NumPy scalar.
    """

    _nonvar_fields = {
        "numpy_value",
        "numpy_type",
        *ConstantVariable._nonvar_fields,
    }

    def __init__(
        self,
        value: Any,
        *,
        numpy_value: Any | None = None,
        **kwargs: Any,
    ) -> None:
        if np is None:
            raise AssertionError("numpy must be available for NumpyScalarVariable")

        if numpy_value is None:
            numpy_value = value
            value = value.item()

        if not isinstance(numpy_value, np.generic):
            raise AssertionError(
                f"Expected np.generic, got {type(numpy_value).__name__}"
            )
        if not self.can_use_python_scalar_proxy(numpy_value):
            raise AssertionError(
                f"Unsupported NumPy scalar type {type(numpy_value).__name__}"
            )

        super().__init__(value, **kwargs)
        self.numpy_value = numpy_value
        self.numpy_type = type(numpy_value)

    @staticmethod
    def can_use_python_scalar_proxy(value: Any) -> bool:
        if np is None:
            return False
        if not isinstance(value, (np.floating, np.signedinteger)):
            return False
        return ConstantVariable.is_base_literal(value.item())

    @staticmethod
    def maybe_create_supported(value: Any, context: str) -> VariableTracker | None:
        if np is None or not isinstance(value, np.generic):
            return None
        if NumpyScalarVariable.can_use_python_scalar_proxy(value):
            return NumpyScalarVariable(value)
        unimplemented(
            gb_type="numpy_scalar_comparison_result",
            context=context,
            explanation=(
                "Dynamo cannot safely lower this NumPy scalar comparison "
                "result as a Python scalar."
            ),
            hints=[
                "Convert the NumPy scalar to a Python scalar outside the compiled region.",
            ],
        )

    def __repr__(self) -> str:
        return f"NumpyScalarVariable({self.numpy_type.__name__}: {self.numpy_value!r})"

    def as_proxy(self) -> Any:
        return self.value

    def as_python_constant(self) -> Any:
        return self.numpy_value

    def python_type(self) -> type:
        return self.numpy_type

    @override
    def call_obj_hasattr(
        self, tx: InstructionTranslatorBase, name: str
    ) -> ConstantVariable:
        return ConstantVariable.create(hasattr(self.numpy_value, name))

    def hash_impl(self, tx: InstructionTranslatorBase) -> tuple[int, bool]:
        if (
            np is not None
            and isinstance(self.numpy_value, np.floating)
            and np.isnan(self.numpy_value)
        ):
            unimplemented(
                gb_type="numpy_scalar_nan_hash",
                context=self.numpy_type.__name__,
                explanation=(
                    "Dynamo cannot safely specialize hash() of a NumPy NaN "
                    "scalar because NaN hashes are object-specific."
                ),
                hints=[
                    "Avoid calling hash() on NumPy NaN scalars inside compiled code.",
                ],
            )
        return hash(self.numpy_value), False

    def richcompare_impl(
        self, tx: InstructionTranslatorBase, other: VariableTracker, op: str
    ) -> VariableTracker:
        if not other.is_python_constant():
            return ConstantVariable.create(NotImplemented)
        try:
            result = getattr(type(self.numpy_value), op)(
                self.numpy_value, other.as_python_constant()
            )
            numpy_result = self.maybe_create_supported(
                result, f"{self.numpy_type.__name__}.{op}"
            )
            if numpy_result is not None:
                return numpy_result
            return VariableTracker.build(tx, result)
        except TypeError as exc:
            raise_observed_exception(type(exc), tx, args=list(exc.args))

    def var_getattr(self, tx: InstructionTranslatorBase, name: str) -> VariableTracker:
        member = self.const_getattr(tx, name)
        source = self.source and AttrSource(self.source, name)

        if (
            np is not None
            and isinstance(member, np.generic)
            and self.can_use_python_scalar_proxy(member)
        ):
            return NumpyScalarVariable(member, source=source)

        if not ConstantVariable.is_literal(member):
            raise NotImplementedError
        return ConstantVariable.create(member, source=source)

    def const_getattr(
        self, tx: InstructionTranslatorBase, name: str
    ) -> VariableTracker:
        if not hasattr(self.numpy_value, name):
            raise_observed_exception(AttributeError, tx, args=[name])
        member = getattr(self.numpy_value, name)
        if callable(member):
            raise NotImplementedError
        return member

    def call_method(
        self,
        tx: InstructionTranslatorBase,
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if (
            name in ("__lt__", "__le__", "__eq__", "__ne__", "__gt__", "__ge__")
            and len(args) == 1
            and not kwargs
        ):
            return self.richcompare_impl(tx, args[0], name)

        if name == "item" and not args and not kwargs:
            return ConstantVariable.create(self.numpy_value.item())
        return super().call_method(tx, name, args, kwargs)

    def reconstruct(self, codegen: Any) -> None:
        codegen.add_push_null(
            lambda: codegen.load_import_from("numpy", self.numpy_type.__name__)
        )
        codegen.append_output(codegen.create_load_const(self.value))
        codegen.extend_output(create_call_function(1, False))

    def reconstruct_pycode(self, codegen: Any) -> str:
        value = (
            "float('nan')"
            if isinstance(self.value, float) and math.isnan(self.value)
            else repr(self.value)
        )
        return f"__import__('numpy').{self.numpy_type.__name__}({value})"


CONSTANT_VARIABLE_NONE = ConstantVariable(None)
CONSTANT_VARIABLE_TRUE = ConstantVariable(True)
CONSTANT_VARIABLE_FALSE = ConstantVariable(False)


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

    def as_python_constant(self) -> int:
        return self.value

    def is_python_constant(self) -> bool:
        return False

    def python_type(self) -> type:
        return int

    def hash_impl(self, tx: Any) -> tuple[int, bool]:
        return hash(self.value), True

    def richcompare_impl(
        self, tx: Any, other: VariableTracker, op: str
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

    def is_python_equal(self, other: object) -> bool:
        if isinstance(other, (FakeIdVariable, ConstantVariable)):
            return self.value == other.as_python_constant()
        return False

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
