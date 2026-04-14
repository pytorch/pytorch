"""
Constant variable tracking in Dynamo.

This module is fundamental to Dynamo's ability to track and propagate constant
values during compilation, ensuring proper handling of Python literals and
maintaining type safety through the compilation process.
"""

from __future__ import annotations

import operator
from typing import Any, Literal, overload, TYPE_CHECKING
from typing_extensions import override

import torch
from torch._dynamo.source import GetItemSource

from .. import variables
from ..exc import raise_observed_exception, unimplemented
from ..utils import common_constant_types, istype, np, raise_args_mismatch
from .base import ValueMutationNew, VariableTracker


if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch._dynamo.symbolic_convert import InstructionTranslator

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
        self, tx: InstructionTranslator, arg: VariableTracker
    ) -> VariableTracker:
        return ConstantVariable.create(
            self.value[arg.as_python_constant()],
        )

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
        self, tx: InstructionTranslator | None
    ) -> list[VariableTracker]:
        try:
            return [ConstantVariable.create(x) for x in self.as_python_constant()]
        except TypeError as e:
            raise NotImplementedError from e

    def len_impl(self, tx: InstructionTranslator) -> VariableTracker:
        """Generic len for any constant value (sequence or mapping)."""
        try:
            return ConstantVariable.create(len(self.value))
        except TypeError as e:
            raise_observed_exception(type(e), tx, args=list(e.args))

    def sq_length(self, tx: InstructionTranslator) -> VariableTracker:
        """Sequence length - delegates to len_impl for constants."""
        return self.len_impl(tx)

    def mp_length(self, tx: InstructionTranslator) -> VariableTracker:
        """Mapping length - delegates to len_impl for constants."""
        return self.len_impl(tx)

    def const_getattr(self, tx: InstructionTranslator, name: str) -> VariableTracker:
        if not hasattr(self.value, name):
            raise_observed_exception(AttributeError, tx, args=[name])
        member = getattr(self.value, name)
        if callable(member):
            raise NotImplementedError
        return member

    def call_method(
        self,
        tx: InstructionTranslator,
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
        elif name == "__contains__" and len(args) == 1 and args[0].is_python_constant():
            assert not kwargs
            search = args[0].as_python_constant()
            try:
                result = search in self.value
                return ConstantVariable.create(result)
            except TypeError as e:
                raise_observed_exception(type(e), tx, args=list(e.args))
        return super().call_method(tx, name, args, kwargs)

    def call_tree_map(
        self,
        tx: InstructionTranslator,
        tree_map_fn: UserFunctionVariable,
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
        self, tx: InstructionTranslator, name: str
    ) -> ConstantVariable:
        result = hasattr(self.value, name)
        return variables.ConstantVariable.create(result)

    def is_python_hashable(self) -> Literal[True]:
        return True

    def get_python_hash(self) -> int:
        return hash(self.value)

    def is_python_equal(self, other: object) -> bool:
        from .tensor import SymNodeVariable

        if isinstance(other, SymNodeVariable):
            return self.as_python_constant() == other.evaluate_expr()
        return (
            isinstance(other, VariableTracker)
            and self.as_python_constant() == other.as_python_constant()
        )

    def get_real_python_backed_value(self) -> object:
        return self.value

    def nb_index_impl(
        self,
        tx: Any,
    ) -> VariableTracker:
        # CPython: int and bool define nb_index (returns self for int,
        # int(self) for bool). All other constant types do not.
        if isinstance(self.value, (int, bool)):
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


CONSTANT_VARIABLE_NONE = ConstantVariable(None)
CONSTANT_VARIABLE_TRUE = ConstantVariable(True)
CONSTANT_VARIABLE_FALSE = ConstantVariable(False)


class FakeIdVariable(VariableTracker):
    """A compile-time-only id value that can be used as a dict key but cannot
    be reconstructed across graph breaks.

    When dynamo evaluates ``id(x)`` on a variable tracker that has no
    corresponding runtime object (e.g. a ``ConstDictVariable`` created during
    tracing), we mint a fake integer id.  This variable holds that id and
    supports the minimal interface needed to participate as a dict key
    (hashing and equality).  It intentionally blocks reconstruction so that a
    graph break does not silently bake a stale id into the resumed bytecode.
    """

    # PyLong_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/longobject.c#L6585
    _cpython_type = int

    def __init__(self, value: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.value = value

    def as_python_constant(self) -> int:
        return self.value

    def is_python_constant(self) -> bool:
        return False

    def python_type(self) -> type:
        return int

    def is_python_hashable(self) -> bool:
        return True

    def get_python_hash(self) -> int:
        return hash(self.value)

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
