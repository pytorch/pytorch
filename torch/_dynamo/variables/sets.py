"""
Set-related variable tracking classes for PyTorch Dynamo.

This module implements variable tracking for different types of set-like objects:
- Regular Python sets (set)
- Frozen sets (frozenset)
- Ordered sets (torch.utils._ordered_set.OrderedSet)
- Dictionary key sets (dict_keys views used as sets)

These classes are responsible for tracking set operations during graph compilation,
maintaining proper guards for set mutations and element existence checks.

The implementation uses a special HashableTracker wrapper to handle set elements
while preserving proper aliasing semantics. Sets are modeled internally as
dictionaries with None values.
"""

import functools
import operator
import types
from collections.abc import Iterable, Iterator
from typing import Any, TYPE_CHECKING

from torch.utils._ordered_set import OrderedSet

from .. import variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import raise_observed_exception, raise_type_error
from ..guards import GuardBuilder, install_guard
from ..source import is_constant_source, is_from_local_source
from ..utils import (
    _item_debug_repr,
    cmp_name_to_op_mapping,
    istype,
    lazily_unpack,
    raise_args_mismatch,
    set_methods,
    tracked_repr,
    unpack_iterable,
)
from .base import Member, Method, readonly_setter, ValueMutationNew, VariableTracker
from .constant import ConstantVariable
from .hashable import HashableTracker, is_hashable


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase


# [Adding a new supported class within the keys of SetVariable]
# see steps outlined for ConstDictVariable


def pyanyset_check(obj: VariableTracker) -> bool:
    return issubclass(obj.python_type(), (set, frozenset))


def pyset_check(obj: VariableTracker) -> bool:
    # ref: https://github.com/python/cpython/blob/v3.13.0/Include/setobject.h#L36-L38
    return issubclass(obj.python_type(), set)


def set_copy(obj: VariableTracker) -> VariableTracker:
    """Mirrors CPython's internal `set_copy` (Objects/setobject.c).

    Always allocates a fresh set/frozenset with a shallow-copied items dict.
    Distinct from the user-visible `.copy()` method, which preserves identity
    for exact frozenset (`frozenset_copy`).  Use this for binary-op scratch
    storage so mutations don't bleed into the input.
    """
    base = obj._base_vt if isinstance(obj, variables.UserDefinedSetVariable) else obj
    if base is None:
        raise AssertionError("_base_vt must not be None")
    return base.clone(
        items=base.items.copy(),  # type: ignore[missing-attribute]
        mutation_type=ValueMutationNew(),
        source=None,
    )


class BaseSetVariable(VariableTracker):
    """Shared logic for the set-family VTs (set, frozenset, dict_keys, OrderedSet).

    In CPython these are siblings, not a subclass chain: frozenset is not a
    subclass of set, and neither is dict_keys (see #192874). This base holds
    every operation that is valid on an immutable set-like collection.
    SetVariable (mutable Python ``set``) adds the mutating operations on top.
    FrozensetVariable, DictKeySetVariable and OrderedSetVariable inherit only
    this base, so a traced frozenset, dict_keys or OrderedSet is not matched by
    isinstance checks against SetVariable.

    Only ``isdisjoint`` is registered in ``tp_methods`` here: that is the whole
    named surface of ``collections.abc.Set``, and of ``dict_keys``. The other
    read-only methods (``union``, ``copy``, ...) are implemented here but
    registered by SetVariable, FrozensetVariable and OrderedSetVariable, the
    same way CPython's ``set_methods`` and ``frozenset_methods`` tables share
    one implementation.
    """

    CONTAINS_GUARD = GuardBuilder.SET_CONTAINS
    NOT_CONTAINS_GUARD = GuardBuilder.SET_NOT_CONTAINS

    def __init__(
        self,
        items: Iterable[VariableTracker | HashableTracker],
        **kwargs: Any,
    ) -> None:
        # .clone() passes these arguments in kwargs but they're recreated below
        if "original_items" in kwargs:
            kwargs.pop("original_items")
        if "should_reconstruct_all" in kwargs:
            kwargs.pop("should_reconstruct_all")

        super().__init__(**kwargs)

        # Items can be either VariableTrackers or HashableTrackers (from set ops).
        # For VariableTrackers, realize them to ensure aliasing guards are installed
        # when the same object appears multiple times.
        hashable_items = []
        for item in items:
            if isinstance(item, HashableTracker):
                # Already a HashableTracker from a set operation
                hashable_items.append(item)
            else:
                # VariableTracker - realize to install guards, then wrap
                # pyrefly: ignore [bad-argument-type]
                hashable_items.append(HashableTracker(item.realize()))
        # Internal representation as dict allows for simple integration with
        # OrderedSet, notably polyfills. Using set moves complexity to OrderedSet
        self.items = dict.fromkeys(hashable_items, BaseSetVariable._default_value())
        self.should_reconstruct_all = (
            not is_from_local_source(self.source) if self.source else True
        )
        self.original_items = dict.fromkeys(
            hashable_items, BaseSetVariable._default_value()
        )

    @property
    def set_items(self) -> set["HashableTracker"]:
        return set(self.items.keys())

    @staticmethod
    def _default_value() -> VariableTracker:
        # Variable to fill in the keys of the dictionary
        return ConstantVariable.create(None)

    def as_proxy(self) -> Any:
        return {k.vt.as_proxy() for k in self.set_items}

    def is_python_constant(self) -> bool:
        # Avoid the base implementation, which probes as_python_constant() and
        # thus rebuilds a real set, re-hashing the elements (wrong for elements
        # with a side-effecting __hash__).  Check element constness directly.
        return all(k.vt.is_python_constant() for k in self.set_items)

    def __contains__(self, vt: VariableTracker) -> bool:
        if not isinstance(vt, VariableTracker):
            raise AssertionError(f"Expected VariableTracker, got {type(vt)}")
        # Use is_hashable as a side-effect-free pre-check.  We can't catch
        # ObservedTypeError from HashableTracker because it modifies
        # tx.exn_vt_stack as a side effect.
        if not is_hashable(vt):
            return False
        key = HashableTracker(vt)
        return key in self.items

    def has_new_items(self) -> bool:
        return self.should_reconstruct_all or any(
            # pyrefly: ignore [bad-argument-type]
            self.is_new_item(self.original_items.get(key.vt), value)
            for key, value in self.items.items()
        )

    def is_new_item(
        self, value: VariableTracker | None, other: VariableTracker
    ) -> bool:
        if value and value.is_realized() and other.is_realized():
            return id(value.realize()) != id(other.realize())
        return id(value) != id(other)

    def unpack_var_sequence(
        self, tx: "InstructionTranslatorBase"
    ) -> list[VariableTracker]:
        return [x.vt for x in self.items]

    def clone(self, **kwargs: Any) -> VariableTracker:
        from torch._dynamo.variables.base import AttributeMutationNew, ValueMutationNew

        if isinstance(
            kwargs.get("mutation_type"), (ValueMutationNew, AttributeMutationNew)
        ):
            kwargs["source"] = None
        return super().clone(**kwargs)

    def call_obj_hasattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> ConstantVariable:
        return VariableTracker.build(tx, hasattr(self.python_type(), name))

    def install_set_contains_guard(
        self, tx: "InstructionTranslatorBase", args: list[VariableTracker]
    ) -> None:
        if not self.source:
            return

        if tx.output.side_effects.is_modified(self):
            return

        contains = args[0] in self
        if args[0].source is None and args[0].is_python_constant():
            guard_fn = (
                type(self).CONTAINS_GUARD if contains else type(self).NOT_CONTAINS_GUARD
            )
            install_guard(
                self.make_guard(
                    functools.partial(
                        guard_fn,
                        key=args[0].as_python_constant(),
                    )
                )
            )

    def _fast_set_method(
        self,
        tx: "InstructionTranslatorBase",
        fn: Any,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        try:
            res = fn(
                *[x.as_python_constant() for x in [self, *args]],
                **{k: v.as_python_constant() for k, v in kwargs.items()},
            )
        except Exception as exc:
            raise_observed_exception(type(exc), tx, args=list(exc.args))
        return VariableTracker.build(tx, res)

    def _iter_operand_keys(
        self, tx: "InstructionTranslatorBase", other: VariableTracker
    ) -> "Iterator[HashableTracker]":
        # Lazily yield HashableTracker keys for a set-operation operand, one
        # element at a time so callers that short-circuit (isdisjoint) observe
        # the same generator side effects as CPython. A set or dict operand's
        # keys are reused directly (no re-hashing), mirroring CPython's
        # set_update_internal fast path for set/frozenset/dict operands.
        # HashableTracker raises ObservedTypeError for an unhashable element,
        # matching CPython's hash-at-insert behavior.
        from .dicts import ConstDictVariable

        if isinstance(other, (BaseSetVariable, ConstDictVariable)):
            yield from other.items.keys()
            return

        yield from (HashableTracker(item) for item in lazily_unpack(tx, other))

    def _operand_keys(
        self, tx: "InstructionTranslatorBase", other: VariableTracker
    ) -> list[HashableTracker]:
        # Eager variant: unpack_iterable takes the fast unpack_var_sequence
        # path for builtin iterables instead of the per-element iterator
        # protocol.
        from .dicts import ConstDictVariable

        if isinstance(other, (BaseSetVariable, ConstDictVariable)):
            return list(other.items.keys())
        return [HashableTracker(x) for x in unpack_iterable(tx, other)]

    def _new_set(self, items: "Iterable[HashableTracker]") -> "BaseSetVariable":
        # Build a fresh set of the same concrete type (set / frozenset /
        # OrderedSet). list() preserves insertion order, which matters for
        # OrderedSet.
        return type(self)(list(items), mutation_type=ValueMutationNew())

    def sq_contains_impl(
        self, tx: "InstructionTranslatorBase", item: VariableTracker
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/setobject.c#L2131-L2149
        if not is_hashable(item):
            # Mirror CPython's set_contains: if hashing fails with TypeError due to
            # an unhashable set, coerce the key to frozenset and retry.
            # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/setobject.c#L2151-L2159
            if not pyset_check(item):
                raise_type_error(tx, f"unhashable type: '{item.python_type_name()}'")
            # CPython NOTE:
            # Note that 'key' could be a set() or frozenset() object.  Unlike most
            # container types, set allows membership testing with a set key, even
            # though it is not hashable.
            item = FrozensetVariable(item.items)  # type: ignore[missing-attribute]
        self.install_set_contains_guard(tx, [item])
        contains = item in self
        return VariableTracker.build(tx, contains)

    def _try_fast_set_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> "VariableTracker | None":
        from ..utils import check_constant_args
        from .dicts import ConstDictVariable

        # Constant, exact-builtin fast path: materialize and call the real
        # method. Set/dict operands take the slow VT path below so their keys
        # are not re-hashed (CPython's do-not-rehash-dict-keys fast path);
        # OrderedSet is excluded because its methods are pure Python, so there
        # is no C fast path to mirror.
        if (
            not any(isinstance(a, (BaseSetVariable, ConstDictVariable)) for a in args)
            and check_constant_args(args, kwargs)
            and self.python_type() in (set, frozenset)
        ):
            py_type = self.python_type()
            return self._fast_set_method(tx, getattr(py_type, name), args, kwargs)
        return None

    def isdisjoint(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        fast = self._try_fast_set_method(tx, "isdisjoint", args, kwargs)
        if fast is not None:
            return fast
        for key in self._iter_operand_keys(tx, args[0]):
            if key in self.items:
                return ConstantVariable.create(False)
        return ConstantVariable.create(True)

    def intersection(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        fast = self._try_fast_set_method(tx, "intersection", args, kwargs)
        if fast is not None:
            return fast
        out_items = dict(self.items)
        for other in args:
            other_keys = set(self._operand_keys(tx, other))
            out_items = {k: v for k, v in out_items.items() if k in other_keys}
        return self._new_set(out_items)

    def union(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        fast = self._try_fast_set_method(tx, "union", args, kwargs)
        if fast is not None:
            return fast
        out_items = dict(self.items)
        for other in args:
            for key in self._operand_keys(tx, other):
                out_items.setdefault(key, BaseSetVariable._default_value())
        return self._new_set(out_items)

    def difference(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        fast = self._try_fast_set_method(tx, "difference", args, kwargs)
        if fast is not None:
            return fast
        out_items = dict(self.items)
        for other in args:
            for key in self._operand_keys(tx, other):
                out_items.pop(key, None)
        return self._new_set(out_items)

    def symmetric_difference(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        fast = self._try_fast_set_method(tx, "symmetric_difference", args, kwargs)
        if fast is not None:
            return fast
        other = dict.fromkeys(
            self._operand_keys(tx, args[0]), BaseSetVariable._default_value()
        )
        out_items = {k: v for k, v in self.items.items() if k not in other}
        out_items.update({k: v for k, v in other.items() if k not in self.items})
        return self._new_set(out_items)

    def _ordering_test(self, tx, args, op):
        from .builder import SourcelessBuilder

        other = args[0].realize()
        if not istype(other, SetVariable):
            other = SourcelessBuilder.create(tx, set).call_function(tx, [other], {})
        return SourcelessBuilder.create(tx, op).call_function(tx, [self, other], {})

    def issubset(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        return self._ordering_test(tx, args, operator.le)

    def issuperset(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        return self._ordering_test(tx, args, operator.ge)

    def copy(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        return set_copy(self)

    def getitem_const(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker:
        raise RuntimeError("Illegal to getitem on a set")

    def tp_iter_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        from .iter import SetIterator

        if self.source and not is_constant_source(self.source):
            tx.output.guard_on_key_order.add(self.source)
        return SetIterator(self.items)

    def nb_or_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/3.13/Objects/setobject.c#L1318-L1338
        self_, other_ = (other, self) if reverse else (self, other)

        if not pyanyset_check(self_) or not pyanyset_check(other_):
            return ConstantVariable.create(NotImplemented)

        result = set_copy(self_)
        if self_ is other_:
            return result
        result.items.update(other_.items)  # type: ignore[missing-attribute]
        return result

    def nb_subtract_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/setobject.c#L1801-L1812
        self_, other_ = (other, self) if reverse else (self, other)

        if not pyanyset_check(self_) or not pyanyset_check(other_):
            return ConstantVariable.create(NotImplemented)

        result = set_copy(self_)
        for k in list(other_.items.keys()):  # type: ignore[missing-attribute]
            result.items.pop(k, None)  # type: ignore[missing-attribute]
        return result

    def nb_and_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/3.13/Objects/setobject.c#L1506-L1518 (set_and)
        self_, other_ = (other, self) if reverse else (self, other)

        if not pyanyset_check(self_) or not pyanyset_check(other_):
            return ConstantVariable.create(NotImplemented)

        return self_.call_method(tx, "intersection", [other_], {})

    def nb_xor_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/3.13/Objects/setobject.c#L1984-L1990 (set_xor)
        self_, other_ = (other, self) if reverse else (self, other)

        if not pyanyset_check(self_) or not pyanyset_check(other_):
            return ConstantVariable.create(NotImplemented)

        return self_.call_method(tx, "symmetric_difference", [other_], {})

    def sq_length_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        return VariableTracker.build(tx, len(self.set_items))

    def tp_richcompare_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        op: str,
    ) -> VariableTracker:
        """set_richcompare: subset/superset comparisons for all 6 ops.

        https://github.com/python/cpython/blob/e76aa128fe/Objects/setobject.c#L2097
        CPython uses PyAnySet_Check: only accepts set/frozenset (not dict views).
        """
        if not isinstance(other, BaseSetVariable):
            try:
                other_type = other.python_type()
            except NotImplementedError:
                return ConstantVariable.create(NotImplemented)
            if not issubclass(other_type, (set, frozenset)):
                return ConstantVariable.create(NotImplemented)

        return self._richcompare_items(
            tx,
            self.set_items,
            other.set_items,  # type: ignore[attr-defined]
            op,
        )

    def _richcompare_items(
        self,
        tx: "InstructionTranslatorBase",
        self_items: set["HashableTracker"],
        other_items: set["HashableTracker"],
        op: str,
    ) -> VariableTracker:
        # Accessing the item sets directly is correct: CPython's set_richcompare
        # operates on the internal C struct (PySet_GET_SIZE, set_next,
        # set_contains_entry) -- it never calls __len__ or __contains__.
        # https://github.com/python/cpython/blob/e76aa128fe/Objects/setobject.c#L2093-L2130
        if op == "__eq__":
            # len check + issubset: same length and subset implies equality.
            if len(self_items) != len(other_items):
                return ConstantVariable.create(False)
            return VariableTracker.build(tx, self_items <= other_items)
        elif op == "__ne__":
            if len(self_items) != len(other_items):
                return ConstantVariable.create(True)
            return VariableTracker.build(tx, not (self_items <= other_items))
        else:
            return VariableTracker.build(
                tx,
                cmp_name_to_op_mapping[op](self_items, other_items),
            )

    tp_methods = {
        "isdisjoint": Method(isdisjoint),
    }


class SetVariable(BaseSetVariable):
    """Represents a Python set during symbolic execution."""

    # PySet_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/setobject.c#L2436
    _cpython_type = set

    def debug_repr(self) -> str:
        if not self.items:
            return "set()"
        else:
            items: list[str] = []
            for v in self.items:
                vt = v.vt if isinstance(v, HashableTracker) else v
                val_str = _item_debug_repr(vt)
                items.append(val_str)
            return "{" + ", ".join(items) + "}"

    def python_type(self) -> type:
        return set

    def as_python_constant(self) -> Any:
        return {k.vt.as_python_constant() for k in self.set_items}

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> "VariableTracker":
        # https://github.com/python/cpython/blob/3.13/Objects/setobject.c#L763-L822
        if not self.items:
            return VariableTracker.build(tx, f"{self.python_type_name()}()")
        items = ", ".join(tracked_repr(tx, item.vt) for item in self.set_items)
        return VariableTracker.build(tx, "{" + items + "}")

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.foreach([x.vt for x in self.set_items])
        codegen.append_output(create_instruction("BUILD_SET", arg=len(self.set_items)))

    def is_hashable(self) -> bool:
        return False

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        from ..exc import raise_type_error

        raise_type_error(tx, f"unhashable type: '{self.python_type_name()}'")

    def add(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # Convert add to __setitem__ with None value
        tx.output.side_effects.mutation(self)
        self.items[HashableTracker(args[0])] = BaseSetVariable._default_value()
        return ConstantVariable.create(None)

    def pop(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # Choose an item at random and pop it
        try:
            result: VariableTracker = self.set_items.pop().vt  # type: ignore[assignment]
        except KeyError as e:
            raise_observed_exception(KeyError, tx, args=list(e.args))
        self.should_reconstruct_all = True
        tx.output.side_effects.mutation(self)
        self.items.pop(HashableTracker(result))
        return result

    def intersection_update(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        kept = dict(self.items)
        for other in args:
            other_keys = set(self._operand_keys(tx, other))
            kept = {k: v for k, v in kept.items() if k in other_keys}
        tx.output.side_effects.mutation(self)
        self.should_reconstruct_all = True
        self.items.clear()
        self.items.update(kept)
        return ConstantVariable.create(None)

    def difference_update(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        tx.output.side_effects.mutation(self)
        self.should_reconstruct_all = True
        for other in args:
            for key in self._operand_keys(tx, other):
                self.items.pop(key, None)
        return ConstantVariable.create(None)

    def symmetric_difference_update(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        other = dict.fromkeys(
            self._operand_keys(tx, args[0]), BaseSetVariable._default_value()
        )
        new_items = {k: v for k, v in self.items.items() if k not in other}
        new_items.update({k: v for k, v in other.items() if k not in self.items})
        tx.output.side_effects.mutation(self)
        self.should_reconstruct_all = True
        self.items.clear()
        self.items.update(new_items)
        return ConstantVariable.create(None)

    def update(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker | None:
        if not self.is_mutable():
            return None
        tx.output.side_effects.mutation(self)
        for other in args:
            for key in self._operand_keys(tx, other):
                self.items.setdefault(key, BaseSetVariable._default_value())
        return ConstantVariable.create(None)

    def remove(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if not self.sq_contains_impl(tx, args[0]).as_python_constant():
            raise_observed_exception(KeyError, tx, args=[args[0]])
        self.should_reconstruct_all = True
        tx.output.side_effects.mutation(self)
        # sq_contains validated/normalized args[0]; a set key was coerced to
        # a frozenset, so pop that same normalized key.
        key = args[0] if is_hashable(args[0]) else FrozensetVariable(args[0].items)  # type: ignore[missing-attribute]
        self.items.pop(HashableTracker(key))
        return ConstantVariable.create(None)

    def discard(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if self.sq_contains_impl(tx, args[0]).as_python_constant():
            self.should_reconstruct_all = True
            tx.output.side_effects.mutation(self)
            # sq_contains validated/normalized args[0]; a set key was coerced
            # to a frozenset, so pop that same normalized key.
            key = (
                args[0] if is_hashable(args[0]) else FrozensetVariable(args[0].items)  # type: ignore[missing-attribute]
            )
            self.items.pop(HashableTracker(key))
        return ConstantVariable.create(None)

    def clear(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        self.should_reconstruct_all = True
        tx.output.side_effects.mutation(self)
        self.items.clear()
        return ConstantVariable.create(None)

    def tp_init_impl(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from .builder import SourcelessBuilder

        temp_set_vt = SourcelessBuilder.create(tx, set).call_set(tx, *args, **kwargs)
        tx.output.side_effects.mutation(self)
        self.items.clear()
        self.items.update(temp_set_vt.items)  # type: ignore[attr-defined]
        return ConstantVariable.create(None)

    def nb_inplace_or_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/3.13/Objects/setobject.c#L1340-L1350
        if not pyanyset_check(other):
            return ConstantVariable.create(NotImplemented)

        tx.output.side_effects.mutation(self)
        self.items.update(other.items)  # type: ignore[missing-attribute]
        return self

    def nb_inplace_subtract_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/setobject.c#L1814-L1828
        if not pyanyset_check(other):
            return ConstantVariable.create(NotImplemented)

        tx.output.side_effects.mutation(self)
        for k in list(other.items.keys()):  # type: ignore[missing-attribute]
            self.items.pop(k, None)
        return self

    def nb_inplace_and_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/3.13/Objects/setobject.c#L1520-L1536 (set_iand)
        if not pyanyset_check(other):
            return ConstantVariable.create(NotImplemented)

        self.call_method(tx, "intersection_update", [other], {})
        return self

    def nb_inplace_xor_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/3.13/Objects/setobject.c#L1992-L2004 (set_ixor)
        if not pyanyset_check(other):
            return ConstantVariable.create(NotImplemented)

        self.call_method(tx, "symmetric_difference_update", [other], {})
        return self

    tp_methods = {
        "intersection": Method(BaseSetVariable.intersection),
        "union": Method(BaseSetVariable.union),
        "difference": Method(BaseSetVariable.difference),
        "symmetric_difference": Method(BaseSetVariable.symmetric_difference),
        "issubset": Method(BaseSetVariable.issubset),
        "issuperset": Method(BaseSetVariable.issuperset),
        "copy": Method(BaseSetVariable.copy),
        "add": Method(add),
        "pop": Method(pop),
        "intersection_update": Method(intersection_update),
        "difference_update": Method(difference_update),
        "symmetric_difference_update": Method(symmetric_difference_update),
        "update": Method(update),
        "remove": Method(remove),
        "discard": Method(discard),
        "clear": Method(clear),
    }


class OrderedSetClassVariable(VariableTracker):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def as_python_constant(self) -> type[OrderedSet[Any]]:
        return OrderedSet

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "__new__":
            if len(args) != 2 or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "OrderedSet.__new__ only accepts one arg"
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )

            return variables.OrderedSetVariable([], mutation_type=ValueMutationNew())

        # OrderedSet has methods set lacks (_from_dict, __reversed__, ...) and
        # the receiver may be missing altogether (OrderedSet.union()); neither
        # is ours to handle, so fall through instead of raising internally.
        resolved_fn = getattr(set, name, None)
        if (
            resolved_fn in set_methods
            and args
            and isinstance(args[0], variables.OrderedSetVariable)
        ):
            return args[0].call_method(tx, name, args[1:], kwargs)

        fn = getattr(OrderedSet, name, None)
        if isinstance(fn, types.FunctionType):
            # Any other receiver (a plain set, a frozenset, a subclass instance,
            # or none at all): run OrderedSet's own Python method, which is what
            # CPython does, so the result or the exception is the eager one.
            from .builder import SourcelessBuilder

            return SourcelessBuilder.create(tx, fn).call_function(tx, args, kwargs)

        return super().call_method(tx, name, args, kwargs)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> "OrderedSetVariable":
        if not args and set(kwargs) == {"iterable"}:
            # OrderedSet(iterable=...): the parameter's name in __init__.
            args = [kwargs["iterable"]]
        elif len(args) > 1 or kwargs:
            raise_args_mismatch(
                tx,
                "OrderedSet",
                "OrderedSet only accepts one arg"
                f"{len(args)} args and {len(kwargs)} kwargs",
            )

        if len(args) == 0 or args[0].is_constant_none():
            # OrderedSet(iterable=None): the documented empty constructor.
            # pyrefly: ignore [implicit-any]
            items = []
        else:
            items = unpack_iterable(tx, args[0])
        return variables.OrderedSetVariable(items, mutation_type=ValueMutationNew())


class OrderedSetVariable(BaseSetVariable):
    """``torch.utils._ordered_set.OrderedSet``, an insertion-ordered set.

    OrderedSet is a pure-Python ``collections.abc.MutableSet``, not a subclass
    of ``set`` (see #192874), so this is a sibling of SetVariable under
    BaseSetVariable rather than a subclass of it. Its named-method surface is
    the same as ``set``'s, mutators included, so the ``set`` implementations
    are registered in ``tp_methods`` below. What OrderedSet does differently
    is overridden here: insertion order is observable (iteration, ``repr``,
    ``pop`` and reconstruction all follow it), and the in-place operators are
    ``MutableSet``'s, which accept any iterable and mutate in place.
    """

    _cpython_type = OrderedSet

    def method_flags_type(self) -> type:
        # OrderedSet is pure-Python (no C ml_flags); its named methods mirror
        # set's arities, so derive MethodFlags from set to enforce them.
        return set

    def _get_internal_dict(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # OrderedSet is backed by a dict (self._dict). Expose it so inlined
        # OrderedSet methods (e.g. __contains__ -> `elem in self._dict`) trace
        # natively instead of graph-breaking on an unmodeled attribute.
        from .dicts import ConstDictVariable

        return ConstDictVariable(self.items, mutation_type=ValueMutationNew())  # type: ignore[bad-argument-type]

    tp_members = {"_dict": Member(_get_internal_dict, readonly_setter)}

    # A passed-in OrderedSet is guarded through its ``_dict`` (DICT_KEYS_MATCH
    # and key order on ``AttrSource(source, "_dict")``, installed by the
    # builder), which already pins membership and iteration order. The per-key
    # SET_CONTAINS guard and the key-order registration BaseSetVariable adds
    # for the object's own source are both wrong here: SET_CONTAINS evaluates
    # ``set.__contains__`` on an OrderedSet, and the guard manager expects a
    # dict at a key-order source.
    def install_set_contains_guard(
        self, tx: "InstructionTranslatorBase", args: list[VariableTracker]
    ) -> None:
        pass

    def tp_iter_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        from .iter import SetIterator

        return SetIterator(self.items)

    def debug_repr(self) -> str:
        if not self.items:
            return "OrderedSet([])"
        else:
            items: list[str] = []
            for k in self.items:
                key_str = _item_debug_repr(k.vt)
                items.append(key_str)
            return "OrderedSet([" + ", ".join(items) + "])"

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> "VariableTracker":
        # self.items is insertion-ordered; set_items is a set and would not be.
        items = ", ".join(tracked_repr(tx, item.vt) for item in self.items)
        return VariableTracker.build(tx, f"{self.python_type_name()}([{items}])")

    def as_python_constant(self) -> OrderedSet[Any]:
        return OrderedSet([k.vt.as_python_constant() for k in self.items])

    def python_type(self) -> type[OrderedSet[Any]]:
        return OrderedSet

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(
            lambda: codegen.load_import_from("torch.utils._ordered_set", "OrderedSet")
        )
        codegen.foreach([x.vt for x in self.items])
        codegen.append_output(create_instruction("BUILD_LIST", arg=len(self.items)))
        codegen.extend_output(create_call_function(1, False))

    def is_hashable(self) -> bool:
        return False

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        raise_type_error(tx, f"unhashable type: '{self.python_type_name()}'")

    def sq_contains_impl(
        self, tx: "InstructionTranslatorBase", item: VariableTracker
    ) -> VariableTracker:
        # OrderedSet.__contains__ is ``elem in self._dict``: an unhashable key
        # raises TypeError. set_contains's coerce-a-set-key-to-frozenset
        # special case is set's, not OrderedSet's. remove and discard go
        # through here first, so they raise the same way.
        if not is_hashable(item):
            raise_type_error(tx, f"unhashable type: '{item.python_type_name()}'")
        return VariableTracker.build(tx, item in self)

    def reversed_(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # OrderedSet.__reversed__ is reversed(self._dict): reverse insertion
        # order. Key order of a passed-in OrderedSet is already guarded.
        keys: list[VariableTracker] = [k.vt for k in reversed(list(self.items))]
        return variables.ListIteratorVariable(keys, mutation_type=ValueMutationNew())

    def pop(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # OrderedSet.pop is dict.popitem: the most recently inserted element,
        # not the arbitrary one set.pop returns.
        if not self.items:
            raise_observed_exception(KeyError, tx, args=["pop from an empty set"])
        self.should_reconstruct_all = True
        tx.output.side_effects.mutation(self)
        key, _ = self.items.popitem()
        return key.vt

    # ``reverse`` is deliberately ignored by the |, & and ^ slots: in
    # collections.abc.Set ``__ror__``, ``__rand__`` and ``__rxor__`` are the
    # forward methods, so ``some_set | ordered_set`` runs ``ordered_set.__or__``
    # and the OrderedSet's elements come first either way. Only ``-`` has a
    # distinct ``__rsub__``, handled below.
    def nb_or_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # OrderedSet does not inherit from Python set, so BaseSetVariable.nb_or_impl
        # won't work due to the PyAnySet_Check
        return super().call_method(tx, "union", [other], {})

    def nb_and_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # Set.__and__ (and __rand__, the same function) yields in the order of
        # the *other* operand: ``self._from_iterable(v for v in other if v in
        # self)``. OrderedSet.__and__ first swaps the operands when the other
        # is an OrderedSet longer than self, so that case comes out in self's
        # order. intersection() differs: it copies self and discards, so it
        # keeps self's order.
        other_keys = self._operand_keys(tx, other)
        if isinstance(other, OrderedSetVariable) and len(self.items) < len(other_keys):
            keys: list[HashableTracker] = [k for k in self.items if k in other.items]
        else:
            keys = [k for k in other_keys if k in self.items]
        return OrderedSetVariable(keys, mutation_type=ValueMutationNew())

    def nb_xor_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # OrderedSet does not inherit from Python set, so BaseSetVariable.nb_xor_impl
        # won't work due to the PyAnySet_Check
        return super().call_method(tx, "symmetric_difference", [other], {})

    def nb_subtract_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        if reverse:
            # Set.__rsub__: ``self._from_iterable(v for v in other if v not in
            # self)``. The left operand can be any iterable, and the result is
            # an OrderedSet, not the left operand's type.
            keys = [k for k in self._operand_keys(tx, other) if k not in self.items]
            return OrderedSetVariable(keys, mutation_type=ValueMutationNew())
        return self.call_method(tx, "difference", [other], {})

    # The in-place slots are MutableSet's, not set's: they accept any iterable
    # and mutate in place. set's slots return NotImplemented for a non-set
    # operand, which would fall back to the binary operator and rebind the
    # name to a new object.
    def nb_inplace_or_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker
    ) -> VariableTracker:
        self.call_method(tx, "update", [other], {})
        return self

    def nb_inplace_and_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker
    ) -> VariableTracker:
        self.call_method(tx, "intersection_update", [other], {})
        return self

    def nb_inplace_xor_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker
    ) -> VariableTracker:
        self.call_method(tx, "symmetric_difference_update", [other], {})
        return self

    def nb_inplace_subtract_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker
    ) -> VariableTracker:
        self.call_method(tx, "difference_update", [other], {})
        return self

    tp_methods = {
        "intersection": Method(BaseSetVariable.intersection),
        "union": Method(BaseSetVariable.union),
        "difference": Method(BaseSetVariable.difference),
        "symmetric_difference": Method(BaseSetVariable.symmetric_difference),
        "issubset": Method(BaseSetVariable.issubset),
        "issuperset": Method(BaseSetVariable.issuperset),
        "copy": Method(BaseSetVariable.copy),
        "add": Method(SetVariable.add),
        "pop": Method(pop),
        "intersection_update": Method(SetVariable.intersection_update),
        "difference_update": Method(SetVariable.difference_update),
        "symmetric_difference_update": Method(SetVariable.symmetric_difference_update),
        "update": Method(SetVariable.update),
        "remove": Method(SetVariable.remove),
        "discard": Method(SetVariable.discard),
        "clear": Method(SetVariable.clear),
        "__reversed__": Method(reversed_),
    }


class FrozensetVariable(BaseSetVariable):
    # PyFrozenSet_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/setobject.c#L2526
    _cpython_type = frozenset

    def debug_repr(self) -> str:
        if not self.items:
            return "frozenset()"
        else:
            items: list[str] = []
            for k in self.items:
                key_str = _item_debug_repr(k.vt)
                items.append(key_str)
            return "frozenset({" + ", ".join(items) + "})"

    def python_type(self) -> type:
        return frozenset

    def as_python_constant(self) -> Any:
        return frozenset({k.vt.as_python_constant() for k in self.set_items})

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> "VariableTracker":
        # https://github.com/python/cpython/blob/3.13/Objects/setobject.c#L763-L822
        if not self.items:
            return VariableTracker.build(tx, f"{self.python_type_name()}()")
        items = ", ".join(tracked_repr(tx, item.vt) for item in self.set_items)
        return VariableTracker.build(tx, f"{self.python_type_name()}({{{items}}})")

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(
            lambda: codegen.extend_output(
                [
                    codegen.create_load_global("frozenset"),
                ]
            )
        )
        codegen.foreach([x.vt for x in self.set_items])
        codegen.extend_output(
            [
                create_instruction("BUILD_LIST", arg=len(self.set_items)),
                *create_call_function(1, False),
            ]
        )

    def copy(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if type(self) is FrozensetVariable:
            return self
        return BaseSetVariable.copy(self, tx, args, kwargs)

    def difference(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        r = BaseSetVariable.difference(self, tx, args, kwargs)
        return FrozensetVariable(r.items)  # type: ignore[attr-defined]

    def intersection(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        r = BaseSetVariable.intersection(self, tx, args, kwargs)
        return FrozensetVariable(r.items)  # type: ignore[attr-defined]

    def symmetric_difference(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        r = BaseSetVariable.symmetric_difference(self, tx, args, kwargs)
        return FrozensetVariable(r.items)  # type: ignore[attr-defined]

    tp_methods = {
        "copy": Method(copy),
        "difference": Method(difference),
        "intersection": Method(intersection),
        "symmetric_difference": Method(symmetric_difference),
        "union": Method(BaseSetVariable.union),
        "issubset": Method(BaseSetVariable.issubset),
        "issuperset": Method(BaseSetVariable.issuperset),
    }

    def tp_init_impl(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # frozenset is immutable. Calling __init__ again shouldn't have any effect.
        return ConstantVariable.create(None)

    def is_hashable(self) -> bool:
        return True

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        # frozenset is hashable, unlike set (SetVariable.hash_impl raises TypeError).
        # CPython frozenset_hash: https://github.com/python/cpython/blob/e76aa128fe/Objects/setobject.c#L769
        from .hashable import RawHash
        from .object_protocol import generic_hash_impl

        if self.is_python_constant():
            return hash(self.as_python_constant()), False
        is_fake = False
        raw_hashes = []
        for item in self.set_items:
            h, fake = generic_hash_impl(tx, item.vt)
            is_fake = is_fake or fake
            raw_hashes.append(RawHash(h))
        return hash(frozenset(raw_hashes)), is_fake


class DictKeySetVariable(BaseSetVariable):
    """A ``dict_keys`` view passed into the compiled region.

    ``dict_keys`` is a ``collections.abc.Set``, not a ``set``: it has no named
    set method other than ``isdisjoint``, and its number slots (``dictviews_or``
    and friends) accept any iterable rather than requiring another set.
    ref: https://github.com/python/cpython/blob/v3.13.0/Objects/dictobject.c#L4667-L4790
    """

    def debug_repr(self) -> str:
        if not self.items:
            return "dict_keys([])"
        else:
            items: list[str] = []
            for k in self.items:
                key_str = _item_debug_repr(k.vt)
                items.append(key_str)
            return "dict_keys([" + ", ".join(items) + "])"

    def install_set_contains_guard(
        self, tx: "InstructionTranslatorBase", args: list[VariableTracker]
    ) -> None:
        # Already EQUALS_MATCH guarded
        pass

    @property
    def set_items(self) -> Any:
        return self.items

    def python_type(self) -> type:
        from ..utils import dict_keys

        return dict_keys

    def as_python_constant(self) -> Any:
        return dict.fromkeys(
            {k.vt.as_python_constant() for k in self.set_items}, None
        ).keys()

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> "VariableTracker":
        # dictview_repr in Objects/dictobject.c: the type name around a list repr.
        items = ", ".join(tracked_repr(tx, k.vt) for k in self.items)
        return VariableTracker.build(tx, f"dict_keys([{items}])")

    def is_hashable(self) -> bool:
        return False

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        raise_type_error(tx, f"unhashable type: '{self.python_type_name()}'")

    def _dictview_binop(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        name: str,
        reverse: bool,
    ) -> VariableTracker:
        # dictviews_or/_and/_sub/_xor build a set from the left operand and
        # update it with the right, so either side may be any iterable and the
        # result is always a plain set.
        left, right = (other, self) if reverse else (self, other)
        return self._as_set_variable(tx, left).call_method(tx, name, [right], {})

    @staticmethod
    def _as_set_variable(
        tx: "InstructionTranslatorBase", vt: VariableTracker
    ) -> "SetVariable":
        from .dicts import ConstDictVariable

        if isinstance(vt, (BaseSetVariable, ConstDictVariable)):
            items: list[Any] = list(vt.items.keys())
        else:
            items = unpack_iterable(tx, vt)
        return SetVariable(items, mutation_type=ValueMutationNew())

    def nb_or_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        return self._dictview_binop(tx, other, "union", reverse)

    def nb_and_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        return self._dictview_binop(tx, other, "intersection", reverse)

    def nb_subtract_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        return self._dictview_binop(tx, other, "difference", reverse)

    def nb_xor_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        return self._dictview_binop(tx, other, "symmetric_difference", reverse)

    def tp_richcompare_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        op: str,
    ) -> VariableTracker:
        """dictview_richcompare, for the operands this VT can read directly.

        Anything else returns NotImplemented so the reflected operation runs,
        which is what already handles a `DictKeysVariable` on the right.
        https://github.com/python/cpython/blob/v3.13.0/Objects/dictobject.c#L4623
        """
        if not isinstance(other, BaseSetVariable):
            return ConstantVariable.create(NotImplemented)

        return self._richcompare_items(
            tx, set(self.items.keys()), set(other.items.keys()), op
        )
