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
from collections.abc import Iterable, Sequence
from typing import Any, Literal, TYPE_CHECKING

from torch.utils._ordered_set import OrderedSet

from .. import polyfills, variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import raise_observed_exception
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, is_constant_source, is_from_local_source
from ..utils import cmp_name_to_op_mapping, istype, raise_args_mismatch
from .base import ValueMutationNew, VariableTracker
from .constant import CONSTANT_VARIABLE_FALSE, CONSTANT_VARIABLE_NONE, ConstantVariable
from .hashable import HashableTracker, is_hashable, raise_unhashable


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslator
    from torch._dynamo.variables.builtin import BuiltinVariable


# [Adding a new supported class within the keys of SetVariable]
# see steps outlined for ConstDictVariable


class SetVariable(VariableTracker):
    """Represents a Python set during symbolic execution."""

    # PySet_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/setobject.c#L2436
    _cpython_type = set

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
        self.items = dict.fromkeys(hashable_items, SetVariable._default_value())
        self.should_reconstruct_all = (
            not is_from_local_source(self.source) if self.source else True
        )
        self.original_items = dict.fromkeys(
            hashable_items, SetVariable._default_value()
        )

    def debug_repr(self) -> str:
        if not self.items:
            return "set()"
        else:
            items: list[str] = []
            for v in self.items:
                vt = v.vt if isinstance(v, HashableTracker) else v
                val_str = repr(vt.value) if hasattr(vt, "value") else vt.debug_repr()
                items.append(val_str)
            return "{" + ",".join(items) + "}"

    @property
    def set_items(self) -> set["HashableTracker"]:
        return set(self.items.keys())

    @staticmethod
    def _default_value() -> VariableTracker:
        # Variable to fill in the keys of the dictionary
        return CONSTANT_VARIABLE_NONE

    def as_proxy(self) -> Any:
        return {k.vt.as_proxy() for k in self.set_items}

    def python_type(self) -> type:
        return set

    def as_python_constant(self) -> Any:
        return {k.vt.as_python_constant() for k in self.set_items}

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.foreach([x.vt for x in self.set_items])
        codegen.append_output(create_instruction("BUILD_SET", arg=len(self.set_items)))

    def __contains__(self, vt: VariableTracker) -> bool:
        assert isinstance(vt, VariableTracker)
        if not is_hashable(vt):
            return False
        key = HashableTracker(vt)
        return key in self.items and not isinstance(
            self.items[key], variables.DeletedVariable
        )

    def len(self) -> int:
        return sum(
            not isinstance(x, variables.DeletedVariable) for x in self.items.values()
        )

    def has_new_items(self) -> bool:
        return self.should_reconstruct_all or any(
            self.is_new_item(self.original_items.get(key.vt), value)
            for key, value in self.items.items()
        )

    def is_new_item(
        self, value: VariableTracker | None, other: VariableTracker
    ) -> bool:
        if value and value.is_realized() and other.is_realized():
            return id(value.realize()) != id(other.realize())
        return id(value) != id(other)

    def unpack_var_sequence(self, tx: "InstructionTranslator") -> list[VariableTracker]:
        return [x.vt for x in self.items]

    def clone(self, **kwargs: Any) -> VariableTracker:
        return super().clone(**kwargs)

    def is_python_hashable(self) -> bool:
        return False

    def var_getattr(self, tx: "InstructionTranslator", name: str):
        if name == "__class__":
            return VariableTracker.build(tx, self.python_type())
        return super().var_getattr(tx, name)

    def call_obj_hasattr(
        self, tx: "InstructionTranslator", name: str
    ) -> ConstantVariable:
        return VariableTracker.build(tx, hasattr(set, name))

    def install_set_contains_guard(
        self, tx: "InstructionTranslator", args: list[VariableTracker]
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
        tx: "InstructionTranslator",
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

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from ..utils import check_constant_args
        from .builder import SourcelessBuilder

        if (
            name
            in (
                "isdisjoint",
                "union",
                "intersection",
                "difference",
                "symmetric_difference",
            )
            and check_constant_args(args, kwargs)
            and self.python_type() is set
        ):
            py_type = self.python_type()
            return self._fast_set_method(tx, getattr(py_type, name), args, kwargs)

        # Lazy imports to avoid circular dependencies
        from .dicts import DictItemsVariable, DictKeysVariable

        if name == "__init__":
            temp_set_vt = SourcelessBuilder.create(tx, set).call_set(
                tx, *args, **kwargs
            )
            tx.output.side_effects.mutation(self)
            self.items.clear()
            self.items.update(temp_set_vt.items)  # type: ignore[attr-defined]
            return CONSTANT_VARIABLE_NONE
        elif name == "add":
            if kwargs or len(args) != 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            # Convert add to __setitem__ with None value
            if not is_hashable(args[0]):
                raise_unhashable(args[0], tx)
            tx.output.side_effects.mutation(self)
            self.items[HashableTracker(args[0])] = SetVariable._default_value()
            return CONSTANT_VARIABLE_NONE
        elif name == "pop":
            if kwargs or args:
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            # Choose an item at random and pop it
            try:
                result: VariableTracker = self.set_items.pop().vt  # type: ignore[assignment]
            except KeyError as e:
                raise_observed_exception(KeyError, tx, args=list(e.args))
            self.should_reconstruct_all = True
            tx.output.side_effects.mutation(self)
            self.items.pop(HashableTracker(result))
            return result
        elif name == "isdisjoint":
            if kwargs or len(args) != 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            return SourcelessBuilder.create(tx, polyfills.set_isdisjoint).call_function(
                tx, [self, args[0]], {}
            )
        elif name == "intersection":
            if kwargs:
                raise_args_mismatch(tx, name, "0 kwargs", f"{len(kwargs)} kwargs")
            return SourcelessBuilder.create(
                tx, polyfills.set_intersection
            ).call_function(
                tx,
                [self, *args],
                {"cls": self.python_type_var()},
            )
        elif name == "intersection_update":
            if kwargs:
                raise_args_mismatch(tx, name, "0 kwargs", f"{len(kwargs)} kwargs")
            return SourcelessBuilder.create(
                tx, polyfills.set_intersection_update
            ).call_function(tx, [self, *args], {})
        elif name == "union":
            if kwargs:
                raise_args_mismatch(tx, name, "0 kwargs", f"{len(kwargs)} kwargs")
            return SourcelessBuilder.create(tx, polyfills.set_union).call_function(
                tx,
                [self, *args],
                {"cls": self.python_type_var()},
            )
        elif name == "difference":
            if kwargs:
                raise_args_mismatch(
                    tx, name, f"Expect: 0 kwargs, Actual: {len(kwargs)} kwargs"
                )
            return SourcelessBuilder.create(tx, polyfills.set_difference).call_function(
                tx,
                [self, *args],
                {"cls": self.python_type_var()},
            )
        elif name == "difference_update":
            if kwargs:
                raise_args_mismatch(tx, name, "0 kwargs", f"{len(kwargs)} kwargs")
            return SourcelessBuilder.create(
                tx, polyfills.set_difference_update
            ).call_function(tx, [self, *args], {})
        elif name == "symmetric_difference":
            if kwargs or len(args) != 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            return SourcelessBuilder.create(
                tx, polyfills.set_symmetric_difference
            ).call_function(
                tx,
                [self, *args],
                {"cls": self.python_type_var()},
            )
        elif name == "symmetric_difference_update":
            if kwargs or len(args) != 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            return SourcelessBuilder.create(
                tx, polyfills.set_symmetric_difference_update
            ).call_function(tx, [self, *args], {})
        elif name == "update" and self.is_mutable():
            if kwargs:
                raise_args_mismatch(tx, name, "0 kwargs", f"{len(kwargs)} kwargs")
            return SourcelessBuilder.create(tx, polyfills.set_update).call_function(
                tx, [self, *args], {}
            )
        elif name == "remove":
            if kwargs or len(args) != 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            if args[0] not in self:
                raise_observed_exception(KeyError, tx, args=args)
            self.should_reconstruct_all = True
            tx.output.side_effects.mutation(self)
            self.items.pop(HashableTracker(args[0]))
            return CONSTANT_VARIABLE_NONE
        elif name == "discard":
            if kwargs or len(args) != 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            if args[0] in self:
                self.should_reconstruct_all = True
                tx.output.side_effects.mutation(self)
                self.items.pop(HashableTracker(args[0]))
            return CONSTANT_VARIABLE_NONE
        elif name in ("issubset", "issuperset"):
            if len(args) != 1:
                raise_args_mismatch(tx, name, "1 args", f"{len(args)} args")

            op = {
                "issubset": operator.le,
                "issuperset": operator.ge,
            }
            other = args[0].realize()
            if not istype(other, SetVariable):
                other = SourcelessBuilder.create(tx, set).call_function(tx, [other], {})
            return SourcelessBuilder.create(tx, op.get(name)).call_function(
                tx, [self, other], {}
            )
        elif name in ("__and__", "__or__", "__xor__", "__sub__"):
            m = {
                "__and__": "intersection",
                "__or__": "union",
                "__xor__": "symmetric_difference",
                "__sub__": "difference",
            }.get(name)
            if not isinstance(
                args[0],
                (
                    SetVariable,
                    variables.UserDefinedSetVariable,
                    DictItemsVariable,
                    DictKeysVariable,
                ),
            ):
                raise_observed_exception(
                    TypeError,
                    tx,
                    args=[
                        f"unsupported operand type(s) for {name}: '{self.python_type_name()}' and '{args[0].python_type_name()}'"
                    ],
                )
            assert m is not None
            return self.call_method(tx, m, args, kwargs)
        elif name in ("__rand__", "__ror__", "__rxor__", "__rsub__"):
            m = {
                "__rand__": "__and__",
                "__ror__": "__or__",
                "__rxor__": "__xor__",
                "__rsub__": "__sub__",
            }.get(name)
            if not isinstance(
                args[0],
                (
                    SetVariable,
                    variables.UserDefinedSetVariable,
                    DictItemsVariable,
                    DictKeysVariable,
                ),
            ):
                raise_observed_exception(
                    TypeError,
                    tx,
                    args=[
                        f"unsupported operand type(s) for {name}: '{args[0].python_type_name()}' and '{self.python_type_name()}'"
                    ],
                )
            assert m is not None
            return args[0].call_method(tx, m, [self], kwargs)
        elif name in ("__iand__", "__ior__", "__ixor__", "__isub__"):
            if not isinstance(
                args[0],
                (
                    SetVariable,
                    variables.UserDefinedSetVariable,
                    DictItemsVariable,
                    DictKeysVariable,
                ),
            ):
                raise_observed_exception(
                    TypeError,
                    tx,
                    args=[
                        f"unsupported operand type(s) for {name}: '{self.python_type_name()}' and '{args[0].python_type_name()}'"
                    ],
                )
            m = {
                "__iand__": "intersection_update",
                "__ior__": "update",
                "__ixor__": "symmetric_difference_update",
                "__isub__": "difference_update",
            }.get(name)
            assert m is not None
            self.call_method(tx, m, args, kwargs)
            return self
        elif name == "__eq__":
            if not isinstance(
                args[0],
                (
                    SetVariable,
                    variables.UserDefinedSetVariable,
                    DictItemsVariable,
                    DictKeysVariable,
                ),
            ):
                return CONSTANT_VARIABLE_FALSE
            r = self.call_method(tx, "symmetric_difference", args, kwargs)
            return VariableTracker.build(tx, len(r.set_items) == 0)  # type: ignore[attr-defined]
        elif name == "__ne__":
            eq_result = self.call_method(tx, "__eq__", args, kwargs)
            return VariableTracker.build(tx, not eq_result.value)  # type: ignore[attr-defined]
        elif name in cmp_name_to_op_mapping:
            if not isinstance(
                args[0],
                (
                    SetVariable,
                    variables.UserDefinedSetVariable,
                    DictItemsVariable,
                    DictKeysVariable,
                ),
            ):
                return VariableTracker.build(tx, NotImplemented)
            return VariableTracker.build(
                tx,
                cmp_name_to_op_mapping[name](self.set_items, args[0].set_items),  # type: ignore[attr-defined]
            )
        elif name == "__contains__":
            if not len(args):
                raise_args_mismatch(
                    tx,
                    name,
                    "more than 1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            if not (args and is_hashable(args[0])):
                raise_unhashable(args[0], tx)
            self.install_set_contains_guard(tx, args)
            contains = args[0] in self
            return VariableTracker.build(tx, contains)
        elif name == "__len__":
            if args or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            return VariableTracker.build(tx, len(self.items))
        elif name == "copy":
            if args or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            return self.clone(
                items=self.items.copy(), mutation_type=ValueMutationNew(), source=None
            )
        elif name == "clear":
            if args or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            self.should_reconstruct_all = True
            tx.output.side_effects.mutation(self)
            self.items.clear()
            return CONSTANT_VARIABLE_NONE
        elif name == "__iter__":
            from .lists import ListIteratorVariable

            if self.source and not is_constant_source(self.source):
                tx.output.guard_on_key_order.add(self.source)
            return ListIteratorVariable(
                self.unpack_var_sequence(tx), mutation_type=ValueMutationNew()
            )
        return super().call_method(tx, name, args, kwargs)

    def python_type_var(self) -> "BuiltinVariable":
        return variables.BuiltinVariable(set)

    def getitem_const(
        self, tx: "InstructionTranslator", arg: VariableTracker
    ) -> VariableTracker:
        raise RuntimeError("Illegal to getitem on a set")

    def sq_length(self, tx: "InstructionTranslator") -> VariableTracker:
        return VariableTracker.build(tx, len(self.set_items))


class OrderedSetClassVariable(VariableTracker):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def as_python_constant(self) -> type[OrderedSet[Any]]:
        return OrderedSet

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        if name == "__new__":
            from .misc import GetAttrVariable

            if self.source:
                attr_source = AttrSource(self.source, name)
            else:
                attr_source = None
            return GetAttrVariable(
                self, name, py_type=type(getattr(OrderedSet, name)), source=attr_source
            )
        else:
            return super().var_getattr(tx, name)

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from .builtin import set_methods

        if name == "__new__":
            if len(args) != 2 or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "OrderedSet.__new__ only accepts one arg"
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )

            return variables.OrderedSetVariable([], mutation_type=ValueMutationNew())

        resolved_fn = getattr(set, name)
        if resolved_fn in set_methods and isinstance(args[0], variables.SetVariable):
            return args[0].call_method(tx, name, args[1:], kwargs)

        return super().call_method(tx, name, args, kwargs)

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> "OrderedSetVariable":
        if len(args) > 1 or kwargs:
            raise_args_mismatch(
                tx,
                "OrderedSet",
                "OrderedSet only accepts one arg"
                f"{len(args)} args and {len(kwargs)} kwargs",
            )

        if len(args) == 0:
            # pyrefly: ignore [implicit-any]
            items = []
        else:
            items = args[0].force_unpack_var_sequence(tx)
        return variables.OrderedSetVariable(items, mutation_type=ValueMutationNew())


class OrderedSetVariable(SetVariable):
    def debug_repr(self) -> str:
        if not self.items:
            return "OrderedSet([])"
        else:
            items: list[str] = []
            for k in self.items:
                key_str = (
                    repr(k.vt.value) if hasattr(k.vt, "value") else k.vt.debug_repr()
                )
                items.append(key_str)
            return "OrderedSet([" + ",".join(items) + "])"

    def as_python_constant(self) -> OrderedSet[Any]:
        return OrderedSet([k.vt.as_python_constant() for k in self.set_items])

    def python_type(self) -> type[OrderedSet[Any]]:
        return OrderedSet

    # pyrefly: ignore[bad-override]
    def python_type_var(self) -> OrderedSetClassVariable:
        return OrderedSetClassVariable()

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(
            lambda: codegen.load_import_from("torch.utils._ordered_set", "OrderedSet")
        )
        codegen.foreach([x.vt for x in self.set_items])
        codegen.append_output(create_instruction("BUILD_LIST", arg=len(self.set_items)))
        codegen.extend_output(create_call_function(1, False))


class FrozensetVariable(SetVariable):
    # PyFrozenSet_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/setobject.c#L2526
    _cpython_type = frozenset

    def debug_repr(self) -> str:
        if not self.items:
            return "frozenset()"
        else:
            items: list[str] = []
            for k in self.items:
                key_str = (
                    repr(k.vt.value) if hasattr(k.vt, "value") else k.vt.debug_repr()
                )
                items.append(key_str)
            return "{" + ",".join(items) + "}"

    @property
    def set_items(self) -> set["HashableTracker"]:
        return set(self.items.keys())

    def python_type(self) -> type:
        return frozenset

    def python_type_var(self) -> "BuiltinVariable":
        return variables.BuiltinVariable(frozenset)

    def as_python_constant(self) -> Any:
        return frozenset({k.vt.as_python_constant() for k in self.set_items})

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

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name in ["add", "pop", "update", "remove", "discard", "clear"]:
            raise RuntimeError(f"Illegal call_method {name} on a frozenset")
        elif name == "__init__":
            # frozenset is immutable. Calling __init__ again shouldn't have any effect
            return CONSTANT_VARIABLE_NONE
        elif name in (
            "copy",
            "difference",
            "intersection",
            "symmetric_difference",
        ):
            r = super().call_method(tx, name, args, kwargs)
            return FrozensetVariable(r.items)  # type: ignore[attr-defined]
        return super().call_method(tx, name, args, kwargs)

    def is_python_hashable(self) -> Literal[True]:
        """
        Frozensets are immutable and hashable in Python.
        """
        return True

    def get_python_hash(self) -> int:
        return hash(self.as_python_constant())

    def is_python_equal(self, other: object) -> bool:
        return (
            isinstance(other, VariableTracker)
            and self.as_python_constant() == other.as_python_constant()
        )


class DictKeySetVariable(SetVariable):
    def debug_repr(self) -> str:
        if not self.items:
            return "dict_keys([])"
        else:
            items: list[str] = []
            for k in self.items:
                key_str = (
                    repr(k.vt.value) if hasattr(k.vt, "value") else k.vt.debug_repr()
                )
                items.append(key_str)
            return "dict_keys([" + ",".join(items) + "])"

    def install_set_contains_guard(
        self, tx: "InstructionTranslator", args: list[VariableTracker]
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

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name in ["add", "pop", "update", "remove", "discard", "clear"]:
            raise RuntimeError(f"Illegal call_method {name} on a dict_keys")
        return super().call_method(tx, name, args, kwargs)
