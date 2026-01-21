"""
Constant and enum variable tracking in Dynamo.

This module is fundamental to Dynamo's ability to track and propagate constant
values during compilation, ensuring proper handling of Python literals and
maintaining type safety through the compilation process.
"""

import enum
import operator
from collections.abc import Sequence
from typing import Any, Literal, Optional, overload, TYPE_CHECKING, Union
from typing_extensions import override

import torch
from torch._dynamo.source import AttrSource, GetItemSource

from .. import graph_break_hints, variables
from ..exc import raise_observed_exception, unimplemented
from ..utils import (
    cmp_name_to_op_mapping,
    common_constant_types,
    istype,
    np,
    raise_args_mismatch,
    raise_on_overridden_hash,
)
from .base import ValueMutationNew, VariableTracker


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator

    from .functions import UserFunctionVariable


class ConstantVariable(VariableTracker):
    """
    Variable tracker for Python literals and basic immutable types, with automatic
    routing support for collection types (lists, tuples, sets, etc.).

    The create() method intelligently constructs appropriate variable types for
    nested collections.
    """

    @overload
    @staticmethod
    def create(value: bool) -> "ConstantVariable": ...

    # TODO: Refactor to make these return ConstantVariable
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
            slice_args_vars = tuple(ConstantVariable.create(arg) for arg in slice_args)
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
        assert ConstantVariable.is_base_literal(value), f"""
Cannot construct `ConstantVariable` for value of type {type(value)}.

This failure likely due to PyTorch-internal use of `ConstantVariable` on
non-literal python values, please try using `VariableTracker.build` instead. If
you believe it's a necessary and legitimate use case (the value is immutable and
can't easily be represented with another `VariableTracker` class), please add
its type to `common_constant_types`.
"""
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
        return self.unpack_var_sequence(tx=None)

    def getitem_const(
        self, tx: "InstructionTranslator", arg: VariableTracker
    ) -> VariableTracker:
        return ConstantVariable.create(
            self.value[arg.as_python_constant()],
        )

    @staticmethod
    def is_base_literal(obj: object) -> bool:
        return type(obj) in common_constant_types

    @staticmethod
    def is_literal(obj: object) -> bool:
        if type(obj) in (list, tuple, set, frozenset, torch.Size):
            return all(ConstantVariable.is_literal(x) for x in obj)  # type: ignore[attr-defined]
        return ConstantVariable.is_base_literal(obj)

    def unpack_var_sequence(
        self, tx: Optional["InstructionTranslator"]
    ) -> list[VariableTracker]:
        try:
            return [ConstantVariable.create(x) for x in self.as_python_constant()]
        except TypeError as e:
            raise NotImplementedError from e

    def const_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        if not hasattr(self.value, name):
            raise_observed_exception(AttributeError, tx, args=[name])
        member = getattr(self.value, name)
        if callable(member):
            raise NotImplementedError
        return member

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from .tensor import SymNodeVariable

        if name == "format" and istype(self.value, str):
            return variables.BuiltinVariable(str.format).call_function(
                tx, [self, *args], kwargs
            )
        elif name == "join" and istype(self.value, str):
            if kwargs or len(args) != 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            arg_unpacked = args[0].force_unpack_var_sequence(tx)
            try:
                arg_const = [x.as_python_constant() for x in arg_unpacked]
                return ConstantVariable.create(self.value.join(arg_const))
            except NotImplementedError:
                return super().call_method(tx, name, args, kwargs)
        elif name == "__iter__" and istype(self.value, str):
            # this could be some generic iterator to avoid the circular import,
            # but ListIterator does what we want
            from .lists import ListIteratorVariable

            return ListIteratorVariable(
                self.unpack_var_sequence(tx), mutation_type=ValueMutationNew()
            )

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
                        args=list(map(ConstantVariable.create, exc.args)),
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
                        raise_observed_exception(
                            type(e), tx, args=list(map(ConstantVariable.create, e.args))
                        )
        elif isinstance(self.value, bytes) and name == "decode":
            method = getattr(self.value, name)
            return ConstantVariable.create(method(*const_args, **const_kwargs))
        elif type(self.value) is complex and name in complex.__dict__:
            method = getattr(self.value, name)
            try:
                return ConstantVariable.create(method(*const_args, **const_kwargs))
            except Exception as e:
                raise_observed_exception(type(e), tx)

        if name == "__len__" and not (args or kwargs):
            try:
                # pyrefly: ignore [bad-argument-type]
                return ConstantVariable.create(len(self.value))
            except TypeError as e:
                raise_observed_exception(type(e), tx, args=list(e.args))
        elif name == "__round__" and len(args) == 1 and args[0].is_python_constant():
            try:
                return ConstantVariable.create(
                    # pyrefly: ignore [no-matching-overload]
                    round(self.value, args[0].as_python_constant())
                )
            except Exception as e:
                raise_observed_exception(
                    type(e), tx, args=list(map(ConstantVariable.create, e.args))
                )
        elif name == "__contains__" and len(args) == 1 and args[0].is_python_constant():
            assert not kwargs
            search = args[0].as_python_constant()
            try:
                # pyrefly: ignore [not-iterable, unsupported-operation]
                result = search in self.value
                return ConstantVariable.create(result)
            except TypeError as e:
                raise_observed_exception(
                    type(e), tx, args=list(map(ConstantVariable.create, e.args))
                )
        return super().call_method(tx, name, args, kwargs)

    def call_tree_map(
        self,
        tx: "InstructionTranslator",
        tree_map_fn: "UserFunctionVariable",
        map_fn: VariableTracker,
        rest: Sequence[VariableTracker],
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

    @override
    def call_obj_hasattr(
        self, tx: "InstructionTranslator", name: str
    ) -> "ConstantVariable":
        result = hasattr(self.value, name)
        return variables.ConstantVariable.create(result)

    def is_python_hashable(self) -> Literal[True]:
        return True

    def get_python_hash(self) -> int:
        return hash(self.value)

    def is_python_equal(self, other: object) -> bool:
        # Could be an EnumVariable as well
        from .tensor import SymNodeVariable

        if isinstance(other, SymNodeVariable):
            return self.as_python_constant() == other.evaluate_expr()
        return (
            isinstance(other, VariableTracker)
            and self.as_python_constant() == other.as_python_constant()
        )


class EnumVariable(VariableTracker):
    """VariableTracker for enum.Enum and enum.IntEnum instances

    Provides specialized handling for Python enum types, supporting
    both standard Enum and IntEnum with proper value tracking and comparison.
    """

    def __init__(self, value: Union[enum.Enum, enum.IntEnum], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.value = value

    @classmethod
    def create(
        cls, cls_type: Any, value_vt: VariableTracker, options: Any
    ) -> "EnumVariable":
        if value_vt.is_python_constant():
            for member in list(cls_type):
                if member.value == value_vt.as_python_constant():
                    return cls(member, **options)
        unimplemented(
            gb_type="Failed to construct Enum variable",
            context=f"value: {value_vt}, allowed enum values: {list(cls_type)}",
            explanation="Attempted to construct an Enum value that is non-constant (e.g. int, string) "
            "or is not an acceptable value for the Enum. "
            f"Acceptable values for Enum `{cls_type}`: {list(cls_type)}.",
            hints=[*graph_break_hints.USER_ERROR, *graph_break_hints.SUPPORTABLE],
        )

    def as_proxy(self) -> Union[enum.Enum, int]:
        if isinstance(self.value, int):
            return int(self.value)  # convert IntEnum to a normal int
        return self.value

    def __repr__(self) -> str:
        return f"EnumVariable({type(self.value)})"

    def as_python_constant(self) -> Union[enum.Enum, enum.IntEnum]:
        return self.value

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        if not hasattr(self.value, name):
            raise NotImplementedError
        if name in cmp_name_to_op_mapping:
            return variables.GetAttrVariable(self, name)
        member = getattr(self.value, name)
        source = self.source and AttrSource(self.source, name)
        return VariableTracker.build(tx, member, source=source)

    def is_python_hashable(self) -> Literal[True]:
        raise_on_overridden_hash(self.value, self)
        return True

    def get_python_hash(self) -> int:
        return hash(self.as_python_constant())

    def is_python_equal(self, other: object) -> bool:
        return (
            isinstance(other, VariableTracker)
            and self.as_python_constant() == other.as_python_constant()
        )
