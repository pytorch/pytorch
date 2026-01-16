"""
Variable tracking implementations for list-like data structures in Dynamo.

This module provides specialized variable tracking for various collection types:
- Lists and list subclasses (including torch.nn.ModuleList, ParameterList)
- Tuples and named tuples
- Ranges and slices
- Collections.deque
- torch.Size with special proxy handling

The implementations support both mutable and immutable collections, iteration,
and common sequence operations. Each collection type has a dedicated Variable
class that handles its unique behaviors while integrating with Dynamo's
variable tracking system.
"""

import collections
import operator
import sys
from collections.abc import Sequence
from typing import Any, Literal, Optional, TYPE_CHECKING

import torch
import torch.fx

from .. import graph_break_hints, polyfills, variables
from ..bytecode_transformation import (
    create_build_tuple,
    create_call_function,
    create_instruction,
    create_rot_n,
)
from ..exc import raise_observed_exception, unimplemented
from ..source import AttrSource, NamedTupleFieldsSource
from ..utils import (
    cmp_name_to_op_mapping,
    cmp_name_to_op_str_mapping,
    get_fake_value,
    guard_if_dyn,
    iter_contains,
    namedtuple_fields,
    odict_values,
    raise_args_mismatch,
    range_iterator,
    set_example_value,
)
from .base import ValueMutationNew, VariableTracker
from .constant import ConstantVariable
from .functions import UserFunctionVariable
from .iter import IteratorVariable
from .user_defined import UserDefinedTupleVariable


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslator


class BaseListVariable(VariableTracker):
    @staticmethod
    def cls_for_instance(obj: Any) -> type["BaseListVariable"]:
        return BaseListVariable.cls_for(type(obj))

    @staticmethod
    def cls_for(obj: Any) -> type:
        return {
            iter: ListIteratorVariable,
            list: ListVariable,
            slice: SliceVariable,
            torch.Size: SizeVariable,
            tuple: TupleVariable,
            odict_values: ListVariable,
            torch.nn.ParameterList: ListVariable,
            torch.nn.ModuleList: ListVariable,
            collections.deque: DequeVariable,
        }[obj]

    def __init__(
        self,
        items: list[VariableTracker],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        assert isinstance(items, list)
        assert all(isinstance(x, VariableTracker) for x in items)
        self.items: list[VariableTracker] = items

    def _as_proxy(self) -> list[Any]:
        return [x.as_proxy() for x in self.items]

    def modified(
        self, items: list[VariableTracker], **kwargs: Any
    ) -> "BaseListVariable":
        return type(self)(items, **kwargs)

    @property
    def value(self) -> Any:
        return self.as_python_constant()

    def debug_repr_helper(self, prefix: str, suffix: str) -> str:
        return prefix + ", ".join(i.debug_repr() for i in self.items) + suffix

    def as_python_constant(self) -> Any:
        return self.python_type()([x.as_python_constant() for x in self.items])

    def as_proxy(self) -> Any:
        assert self.python_type() is not SizeVariable
        return self.python_type()(self._as_proxy())

    def getitem_const(
        self, tx: "InstructionTranslator", arg: VariableTracker
    ) -> VariableTracker:
        from .tensor import SymNodeVariable

        if isinstance(arg, SymNodeVariable):
            index = arg.sym_num
        else:
            index = arg.as_python_constant()

        if isinstance(index, slice):
            if index.step == 0:
                msg = ConstantVariable.create("slice step cannot be zero")
                raise_observed_exception(ValueError, tx, args=[msg])
            # Set source to None because slicing a list gives a new local
            return self.clone(
                items=self.items[index],
                source=None,
                mutation_type=ValueMutationNew() if self.mutation_type else None,
            )
        else:
            assert isinstance(index, (int, torch.SymInt))
            try:
                return self.items[index]
            except IndexError:
                raise_observed_exception(
                    IndexError, tx, args=["list index out of range"]
                )

    def unpack_var_sequence(self, tx: "InstructionTranslator") -> list[VariableTracker]:
        return list(self.items)

    def call_tree_map_branch(
        self,
        tx: "InstructionTranslator",
        tree_map_fn: UserFunctionVariable,
        map_fn: VariableTracker,
        rest: Sequence[VariableTracker],
        tree_map_kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if not isinstance(self, (ListVariable, TupleVariable)):
            return self._tree_map_fallback(
                tx, tree_map_fn, map_fn, rest, tree_map_kwargs
            )

        other_lists: list[BaseListVariable] = []
        for candidate in rest:
            if (
                not isinstance(candidate, BaseListVariable)
                or len(candidate.items) != len(self.items)
                or self.python_type() != candidate.python_type()
            ):
                return self._tree_map_fallback(
                    tx, tree_map_fn, map_fn, rest, tree_map_kwargs
                )
            other_lists.append(candidate)

        new_items: list[VariableTracker] = []
        for idx, item in enumerate(self.items):
            sibling_leaves = [candidate.items[idx] for candidate in other_lists]
            new_items.append(
                item.call_tree_map(
                    tx,
                    tree_map_fn,
                    map_fn,
                    sibling_leaves,
                    tree_map_kwargs,
                )
            )

        return self.clone(
            items=new_items,
            source=None,
            mutation_type=ValueMutationNew(),
        )

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "__getitem__":
            if kwargs or len(args) != 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )

            if args[0].is_tensor():
                value = get_fake_value(args[0].as_proxy().node, tx)
                if value.constant is not None and value.constant.numel() == 1:
                    value = variables.ConstantVariable.create(value.constant.item())
                else:
                    unimplemented(
                        gb_type="Indexing list with non-scalar tensor",
                        context=f"call_method {self} {name} {args} {kwargs}",
                        explanation=(
                            "Attempted to index list-like object with tensor with > 1 element."
                        ),
                        hints=[*graph_break_hints.USER_ERROR],
                    )
            else:
                value = args[0]

            if value.python_type() not in (int, slice):
                msg = f"indices must be integers or slices, not {value.python_type()}"
                raise_observed_exception(TypeError, tx, args=[ConstantVariable(msg)])

            return self.getitem_const(tx, value)
        elif name == "__contains__":
            if kwargs or len(args) != 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            return iter_contains(self.unpack_var_sequence(tx), args[0], tx)
        elif name == "index":
            if not len(args):
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )

            return tx.inline_user_function_return(
                VariableTracker.build(tx, polyfills.index),
                [self] + list(args),
                kwargs,
            )
        elif name == "count":
            if len(args) != 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            return VariableTracker.build(tx, operator.countOf).call_function(
                tx,
                [self, args[0]],
                kwargs,
            )
        elif name in ("__add__", "__iadd__"):
            if kwargs or len(args) != 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )

            if type(self) is not type(args[0]):
                tp_name = self.python_type_name()
                other = args[0].python_type_name()
                msg_vt = ConstantVariable.create(
                    f'can only concatenate {tp_name} (not "{other}") to {tp_name}'
                )
                raise_observed_exception(TypeError, tx, args=[msg_vt])

            if name == "__add__":
                return type(self)(self.items + args[0].items, source=self.source)  # type: ignore[attr-defined]
            else:
                self.items += args[0].items  # type: ignore[attr-defined]
                return self
        elif name in ("__mul__", "__imul__"):
            if kwargs or len(args) != 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )

            if not (args[0].is_python_constant() and args[0].python_type() is int):
                msg_vt = ConstantVariable.create(
                    f"can't multiply sequence by non-int type of '{args[0].python_type_name()}'"
                )
                raise_observed_exception(TypeError, tx, args=[msg_vt])

            val = args[0].as_python_constant()

            if name == "__mul__":
                return type(self)(self.items * val, source=self.source)
            else:
                self.items *= val
                return self
        elif name in cmp_name_to_op_mapping:
            if len(args) != 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )

            left = self
            right = args[0]
            # TODO this type check logic mirrors the following
            # https://github.com/python/cpython/blob/a1c52d1265c65bcf0d9edf87e143843ad54f9b8f/Objects/object.c#L991-L1007
            # But we should probably move it up the stack to so that we don't
            # need to duplicate it for different VTs.
            if not isinstance(left, BaseListVariable) or not isinstance(
                right, BaseListVariable
            ):
                if name == "__eq__":
                    return variables.BuiltinVariable(operator.is_).call_function(
                        tx, (left, right), {}
                    )
                elif name == "__ne__":
                    return variables.BuiltinVariable(operator.is_not).call_function(
                        tx, (left, right), {}
                    )
                else:
                    op_str = cmp_name_to_op_str_mapping[name]
                    left_ty = left.python_type_name()
                    right_ty = right.python_type_name()
                    msg = f"{op_str} not supported between instances of '{left_ty}' and '{right_ty}'"
                    raise_observed_exception(TypeError, tx, args=[msg])

            return variables.UserFunctionVariable(polyfills.list_cmp).call_function(
                tx,
                [variables.BuiltinVariable(cmp_name_to_op_mapping[name]), left, right],
                {},
            )
        elif name == "__iter__":
            return ListIteratorVariable(self.items, mutation_type=ValueMutationNew())

        return super().call_method(tx, name, args, kwargs)


class RangeVariable(BaseListVariable):
    def __init__(self, items: Sequence[VariableTracker], **kwargs: Any) -> None:
        items_to_map = items
        start = variables.ConstantVariable.create(0)
        stop = None
        step = variables.ConstantVariable.create(1)

        if len(items_to_map) == 1:
            (stop,) = items_to_map
        elif len(items_to_map) == 2:
            start, stop = items_to_map
        elif len(items_to_map) == 3:
            start, stop, step = items_to_map
        else:
            raise AssertionError

        def maybe_as_int(x: VariableTracker) -> VariableTracker:
            return (
                ConstantVariable.create(int(x.as_python_constant()))
                if x.is_python_constant()
                else x
            )

        # cast each argument to an integer
        start = maybe_as_int(start)
        step = maybe_as_int(step)
        stop = maybe_as_int(stop)

        assert stop is not None
        super().__init__([start, stop, step], **kwargs)

    def debug_repr(self) -> str:
        return self.debug_repr_helper("range(", ")")

    def python_type(self) -> type:
        return range

    def start(self) -> Any:
        return self.items[0].as_python_constant()

    def stop(self) -> Any:
        return self.items[1].as_python_constant()

    def step(self) -> Any:
        return self.items[2].as_python_constant()

    def range_length(self) -> int:
        lo = self.start()
        hi = self.stop()
        step = self.step()

        assert step != 0
        if step > 0 and lo < hi:
            return 1 + (hi - 1 - lo) // step
        elif step < 0 and lo > hi:
            return 1 + (lo - 1 - hi) // (0 - step)
        else:
            return 0

    def _get_slice_indices(self, length: int, slice: slice) -> list[int]:
        step_is_negative = 0

        if slice.step is None:
            step = 1
            step_is_negative = False
        else:
            step = slice.step
            step_is_negative = slice.step < 0

        # Find lower and upper bounds for start and stop.
        if step_is_negative:
            lower = -1
            upper = length + lower
        else:
            lower = 0
            upper = length

        # Compute start
        if slice.start is None:
            start = upper if step_is_negative else lower
        else:
            start = slice.start

        if start < 0:
            start += length
            if start < lower:
                start = lower
        else:
            if start > upper:
                start = upper

        # Compute stop.
        if slice.stop is None:
            stop = lower if step_is_negative else upper

        else:
            stop = slice.stop

            if stop < 0:
                stop += length
                if stop < lower:
                    stop = lower
            else:
                if stop > upper:
                    stop = upper

        return [start, stop, step]

    def apply_index(self, tx: "InstructionTranslator", index: int) -> VariableTracker:
        length = self.range_length()
        if index < 0:
            index = length + index

        if index < 0 or index >= length:
            raise_observed_exception(
                IndexError,
                tx,
                args=[ConstantVariable("range object index out of range")],
            )

        return variables.ConstantVariable.create(self.start() + (index * self.step()))

    def apply_slice(self, slice: slice) -> "RangeVariable":
        (slice_start, slice_stop, slice_step) = self._get_slice_indices(
            self.range_length(), slice
        )

        def compute_item(index: int) -> int:
            return self.start() + (index * self.step())

        sub_step = self.step() * slice_step
        sub_start = compute_item(slice_start)
        sub_stop = compute_item(slice_stop)

        result = RangeVariable(
            [
                variables.ConstantVariable.create(x)
                for x in [sub_start, sub_stop, sub_step]
            ],
            mutation_type=ValueMutationNew() if self.mutation_type else None,
        )
        return result

    def as_python_constant(self) -> range:
        return range(*[x.as_python_constant() for x in self.items])

    def getitem_const(
        self, tx: "InstructionTranslator", arg: VariableTracker
    ) -> VariableTracker:
        # implementations mimics https://github.com/python/cpython/blob/main/Objects/rangeobject.c
        index = arg.as_python_constant()

        if isinstance(index, slice):
            return self.apply_slice(index)
        elif isinstance(index, int):
            return self.apply_index(tx, index)
        else:
            msg = ConstantVariable("range indices must be integers or slices")
            raise_observed_exception(TypeError, tx, args=[msg])

    def as_proxy(self) -> range:
        return self.python_type()(*self._as_proxy())

    def unpack_var_sequence(
        self, tx: Optional["InstructionTranslator"] = None
    ) -> list[VariableTracker]:
        return [variables.ConstantVariable.create(x) for x in self.as_python_constant()]

    def reconstruct(self, codegen: "PyCodegen") -> None:
        assert "range" not in codegen.tx.f_globals
        codegen.add_push_null(
            lambda: codegen.append_output(codegen.create_load_python_module(range))  # type: ignore[arg-type]
        )
        codegen.foreach(self.items)
        codegen.extend_output(create_call_function(3, False))

    def call_obj_hasattr(
        self, tx: "InstructionTranslator", name: str
    ) -> ConstantVariable:
        if self.python_type() is range:
            return variables.ConstantVariable.create(name in range.__dict__)
        return super().call_obj_hasattr(tx, name)

    def range_equals(self, other: "RangeVariable") -> bool:
        r0, r1 = self, other
        if (
            self.range_length() != r1.range_length()
            or self.range_length() == 0
            or r0.start() != r1.start()
        ):
            return False

        if self.range_length() == 1:
            return True

        return r0.step() == r1.step()

    def range_count(self, x: VariableTracker) -> int:
        # Based on CPython
        # https://github.com/guilhermeleobas/cpython/blob/baefaa6cba1d69efd2f930cdc56bca682c54b139/Objects/rangeobject.c#L442-L486
        x = x.as_python_constant()
        if type(x) not in (bool, int, float):
            return 0

        start, stop, step = self.start(), self.stop(), self.step()

        if step == 0:
            return 0

        in_range = (start <= x < stop) if step > 0 else (stop < x <= start)

        if in_range:
            re = ((x - start) % step) == 0
            return int(re)
        return 0

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "__iter__":
            if not all(var.is_python_constant() for var in self.items):
                # Can't represent a `range_iterator` without well defined bounds
                return variables.misc.DelayGraphBreakVariable(
                    msg="Cannot create range_iterator: bounds (start, stop, step) must be fully defined as concrete constants.",
                )
            return RangeIteratorVariable(
                self.start(), self.stop(), self.step(), self.range_length()
            )
        elif name == "__len__":
            length = self.range_length()
            if length > sys.maxsize:
                raise_observed_exception(OverflowError, tx)
            return ConstantVariable.create(self.range_length())
        elif name in ("count", "__contains__"):
            return ConstantVariable(self.range_count(*args))
        elif name == "__getitem__":
            return self.getitem_const(tx, *args)
        elif name in cmp_name_to_op_mapping:
            other = args[0]
            pt = other.python_type()
            if name not in ("__eq__", "__ne__"):
                # ranges are only comparable to other ranges
                msg = f"{name} not supported between instances of 'range' and '{pt}'"
                raise_observed_exception(
                    TypeError,
                    tx,
                    args=[ConstantVariable.create(msg)],
                )

            if pt is not range:
                return ConstantVariable.create(NotImplemented)

            if isinstance(other, RangeVariable):
                cmp = self.range_equals(other)
            else:
                cmp = False

            # Two ranges are equal if they produce the same sequence of values
            if name == "__eq__":
                return ConstantVariable(cmp)
            else:
                return ConstantVariable(not cmp)
        return super().call_method(tx, name, args, kwargs)

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        fields = ["start", "stop", "step"]
        if name in fields:
            return self.items[fields.index(name)]
        return super().var_getattr(tx, name)

    def is_python_hashable(self) -> Literal[True]:
        return True

    def get_python_hash(self) -> int:
        l = self.range_length()
        start = self.start()
        step = self.step()
        return hash((l, start, step))

    def is_python_equal(self, other: object) -> bool:
        if not isinstance(other, variables.RangeVariable):
            return False

        return (
            self.start() == other.start()
            and self.step() == other.step()
            and self.stop() == other.stop()
        )


class CommonListMethodsVariable(BaseListVariable):
    """
    Implement methods common to List and other List-like things
    """

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from .tensor import SymNodeVariable

        if name == "append" and self.is_mutable():
            if kwargs or len(args) != 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            (arg,) = args
            tx.output.side_effects.mutation(self)
            self.items.append(arg)
            return ConstantVariable.create(None)
        elif name == "extend" and self.is_mutable():
            if kwargs or len(args) != 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )

            if not args[0].has_force_unpack_var_sequence(tx):
                msg = ConstantVariable.create(f"{type(args[0])} object is not iterable")
                raise_observed_exception(TypeError, tx, args=[msg])

            (arg,) = args
            arg.force_apply_to_var_sequence(
                tx, lambda item: self.call_method(tx, "append", [item], {})
            )
            return ConstantVariable.create(None)
        elif name == "insert" and self.is_mutable():
            if kwargs or len(args) != 2:
                raise_args_mismatch(
                    tx,
                    name,
                    "2 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            idx, value = args
            if isinstance(idx, SymNodeVariable):
                const_idx = idx.evaluate_expr()
            else:
                const_idx = idx.as_python_constant()
            tx.output.side_effects.mutation(self)
            # type: ignore[arg-type]
            self.items.insert(const_idx, value)
            return ConstantVariable.create(None)
        elif name == "pop" and self.is_mutable():
            if kwargs or len(args) > 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "at most 1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )

            if len(self.items) == 0:
                msg = ConstantVariable.create("pop from empty list")
                raise_observed_exception(IndexError, tx, args=[msg])

            if len(args):
                idx = args[0].as_python_constant()
                if idx > len(self.items):
                    msg = ConstantVariable.create("pop index out of range")
                    raise_observed_exception(IndexError, tx, args=[msg])
            tx.output.side_effects.mutation(self)
            return self.items.pop(*[a.as_python_constant() for a in args])
        elif name == "clear" and self.is_mutable():
            if args or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            tx.output.side_effects.mutation(self)
            self.items.clear()
            return ConstantVariable.create(None)
        elif name == "__setitem__" and self.is_mutable() and args:
            # Realize args[0] to get the concrete type for proper type checking
            key = args[0].realize()
            if not (
                key.is_python_constant()
                or isinstance(key, SymNodeVariable)
                or (
                    isinstance(key, SliceVariable)
                    and all(
                        s.is_python_constant() or isinstance(s, SymNodeVariable)
                        for s in key.items
                    )
                )
            ):
                return super().call_method(tx, name, args, kwargs)
            if kwargs:
                raise_args_mismatch(tx, name, "0 kwargs", f"{len(kwargs)} kwargs")
            value = args[1]
            tx.output.side_effects.mutation(self)
            if isinstance(key, SymNodeVariable):
                # pyrefly: ignore[unsupported-operation]
                self.items[key.evaluate_expr()] = value
            elif isinstance(key, SliceVariable):
                if key.is_python_constant():
                    self.items[key.as_python_constant()] = list(value.items)  # type: ignore[attr-defined]
                else:
                    items_slice = slice(
                        *[
                            (
                                s.evaluate_expr()
                                if isinstance(s, SymNodeVariable)
                                else s.as_python_constant()
                            )
                            for s in key.items
                        ]
                    )
                    self.items[items_slice] = list(value.items)  # type: ignore[attr-defined]
            else:
                self.items[key.as_python_constant()] = value
            return ConstantVariable.create(None)
        elif name == "__delitem__" and self.is_mutable():
            if kwargs or len(args) != 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )

            tx.output.side_effects.mutation(self)
            if args[0].is_python_constant() and isinstance(
                args[0].as_python_constant(), (int, slice)
            ):
                if isinstance(args[0], SymNodeVariable):
                    idx = args[0].evaluate_expr()
                else:
                    idx = args[0].as_python_constant()

                try:
                    self.items.__delitem__(idx)  # type: ignore[arg-type]

                except (IndexError, ValueError) as exc:
                    raise_observed_exception(
                        type(exc),
                        tx,
                        args=list(map(ConstantVariable.create, exc.args)),
                    )
            else:
                msg = ConstantVariable.create(
                    f"list indices must be integers or slices, not {args[0].python_type_name()}"
                )
                raise_observed_exception(TypeError, tx, args=[msg])
            return ConstantVariable.create(None)
        elif name == "copy":
            # List copy() doesn't have args and kwargs
            if args or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            items_lst: list[VariableTracker] = list(self.items)
            return self.modified(items_lst, mutation_type=ValueMutationNew())
        elif name == "reverse" and self.is_mutable():
            if args or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            self.items.reverse()
            tx.output.side_effects.mutation(self)
            return ConstantVariable.create(None)
        elif name == "remove" and self.is_mutable():
            if kwargs or len(args) != 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )

            idx = self.call_method(tx, "index", args, kwargs)
            self.call_method(tx, "pop", [idx], {})
            return ConstantVariable.create(None)
        else:
            return super().call_method(tx, name, args, kwargs)


class ListVariable(CommonListMethodsVariable):
    def python_type(self) -> type:
        return list

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(length={len(self.items)})"

    def debug_repr(self) -> str:
        return self.debug_repr_helper("[", "]")

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.foreach(self.items)
        codegen.append_output(create_instruction("BUILD_LIST", arg=len(self.items)))

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "__setitem__" and self.is_mutable():
            if kwargs or len(args) != 2:
                raise_args_mismatch(
                    tx,
                    name,
                    "2 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            key, value = args

            if not key.is_python_constant():
                # probably will graph-break
                super().call_method(tx, name, args, kwargs)

            tx.output.side_effects.mutation(self)
            if isinstance(key, SliceVariable):
                if not value.has_force_unpack_var_sequence(tx):
                    msg = ConstantVariable.create("can only assign an iterable")
                    raise_observed_exception(TypeError, tx, args=[msg])

                key_as_const = key.as_python_constant()
                if key_as_const.step == 0:
                    msg = ConstantVariable.create("slice step cannot be zero")
                    raise_observed_exception(ValueError, tx, args=[msg])

                value_unpack = value.force_unpack_var_sequence(tx)
                try:
                    self.items[key_as_const] = value_unpack
                except Exception as exc:
                    raise_observed_exception(
                        type(exc),
                        tx,
                        args=list(map(ConstantVariable.create, exc.args)),
                    )
            else:
                # Use guard_if_dyn to handle SymNodeVariable and LazyVariableTracker
                # that may realize to SymNodeVariable
                key = guard_if_dyn(key)

                try:
                    # pyrefly: ignore[unsupported-operation]
                    self.items[key] = value
                except (IndexError, TypeError) as e:
                    raise_observed_exception(
                        type(e), tx, args=list(map(ConstantVariable.create, e.args))
                    )
            return ConstantVariable.create(None)

        if name == "sort" and self.is_mutable():
            if len(args) != 0:
                raise_args_mismatch(tx, name, "0 args", f"{len(args)} args")
            key_fn_var = kwargs.pop("key", ConstantVariable.create(None))
            reverse = kwargs.pop(
                "reverse", ConstantVariable.create(False)
            ).as_python_constant()
            if len(kwargs) != 0:
                raise_args_mismatch(tx, name, "0 kwargs", f"{len(kwargs)} kwargs")

            if key_fn_var.is_constant_none():
                keys = self.items.copy()
            else:
                keys = [key_fn_var.call_function(tx, [x], {}) for x in self.items]

            if not all(k.is_python_constant() for k in keys):
                first_non_constant_key = None
                for k in keys:
                    if not k.is_python_constant():
                        first_non_constant_key = k
                assert first_non_constant_key is not None

                try:
                    python_type = str(first_non_constant_key.python_type())
                except NotImplementedError:
                    python_type = "unknown"

                unimplemented(
                    gb_type="sort with non-constant keys",
                    context=str(first_non_constant_key),
                    explanation=(
                        f"Cannot perform sort with non-constant key. "
                        f"First non-constant key type: {python_type}. "
                        f"Most notably, we cannot sort with Tensor or SymInt keys, but we can "
                        f"sort ints."
                    ),
                    hints=["Use something else as the key."],
                )

            try:
                tx.output.side_effects.mutation(self)
                sorted_items_with_keys = sorted(
                    (
                        (
                            x,
                            k.as_python_constant(),
                            -i if reverse else i,  # extra key to ensure stable sort
                        )
                        for i, (k, x) in enumerate(zip(keys, self.items))
                    ),
                    key=operator.itemgetter(1, 2),
                    reverse=reverse,
                )
                self.items[:] = [x for x, *_ in sorted_items_with_keys]
            except Exception as e:
                raise_observed_exception(type(e), tx, args=list(e.args))
            return ConstantVariable.create(None)

        if name == "__init__" and self.is_mutable():
            if kwargs:
                raise_args_mismatch(tx, name, "0 kwargs", f"{len(kwargs)} kwargs")
            if len(args) == 0:
                return ConstantVariable.create(None)
            elif len(args) == 1 and args[0].has_force_unpack_var_sequence(tx):
                (arg,) = args
                tx.output.side_effects.mutation(self)
                self.items[:] = arg.force_unpack_var_sequence(tx)
                return ConstantVariable.create(None)

        return super().call_method(tx, name, args, kwargs)

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        if name == "__class__":
            source = AttrSource(self.source, name) if self.source else None
            class_type = self.python_type()
            if class_type is list:
                return variables.BuiltinVariable(class_type, source=source)
            else:
                return variables.UserDefinedClassVariable(class_type, source=source)
        return super().var_getattr(tx, name)

    def call_obj_hasattr(
        self, tx: "InstructionTranslator", name: str
    ) -> ConstantVariable:
        if self.python_type() is not list:
            return super().call_obj_hasattr(tx, name)
        return variables.ConstantVariable.create(hasattr([], name))

    def is_python_hashable(self) -> bool:
        return False


class DequeVariable(CommonListMethodsVariable):
    def __init__(
        self,
        items: list[VariableTracker],
        maxlen: Optional[VariableTracker] = None,
        **kwargs: Any,
    ) -> None:
        if maxlen is None:
            maxlen = ConstantVariable.create(None)
        assert maxlen.is_python_constant(), (
            f"maxlen must be a constant, got: {maxlen.debug_repr()}"
        )
        self.maxlen = maxlen
        items = list(items)
        if self.maxlen.as_python_constant() is not None:
            items = items[-maxlen.as_python_constant() :]
        super().__init__(items, **kwargs)

    def python_type(self) -> type:
        return collections.deque

    def debug_repr(self) -> str:
        if self.maxlen.as_python_constant() is None:
            return self.debug_repr_helper(
                "deque([", "], maxlen=" + self.maxlen.debug_repr() + ")"
            )
        return self.debug_repr_helper("deque([", "])")

    def as_python_constant(self) -> collections.deque[Any]:
        return self.python_type()(
            [x.as_python_constant() for x in self.items],
            maxlen=self.maxlen.as_python_constant(),
        )

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(
            lambda: codegen.append_output(
                codegen.create_load_python_module(collections.deque)  # type: ignore[arg-type]
            )
        )
        codegen.foreach(self.items)
        codegen.extend_output([create_instruction("BUILD_LIST", arg=len(self.items))])
        codegen(self.maxlen)
        codegen.extend_output(codegen.create_call_function_kw(2, ("maxlen",), False))

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        if name == "maxlen":
            return self.maxlen
        return super().var_getattr(tx, name)

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if (
            name == "__setitem__"
            and self.is_mutable()
            and args
            and args[0].is_python_constant()
        ):
            if kwargs or len(args) != 2:
                raise_args_mismatch(
                    tx,
                    name,
                    "2 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            key, value = args
            assert key.is_python_constant()
            assert isinstance(key.as_python_constant(), int)
            tx.output.side_effects.mutation(self)
            self.items[key.as_python_constant()] = value
            return ConstantVariable.create(None)

        maxlen = self.maxlen.as_python_constant()
        if maxlen is not None:
            slice_within_maxlen = slice(-maxlen, None)
        else:
            slice_within_maxlen = None

        if (
            name == "extendleft"
            and self.is_mutable()
            and len(args) > 0
            and args[0].has_force_unpack_var_sequence(tx)
        ):
            if kwargs or len(args) != 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            # NOTE this is inefficient, but the alternative is to represent self.items
            # as a deque, which is a more intrusive change.
            args[0].force_apply_to_var_sequence(
                tx, lambda item: self.call_method(tx, "appendleft", [item], {})
            )
            slice_within_maxlen = slice(None, maxlen)
            result = ConstantVariable.create(None)
        elif name == "popleft" and self.is_mutable():
            if kwargs or len(args) > 0:
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            tx.output.side_effects.mutation(self)
            result, *self.items[:] = self.items
        elif name == "appendleft" and len(args) > 0 and self.is_mutable():
            if kwargs or len(args) != 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            tx.output.side_effects.mutation(self)
            self.items[:] = [args[0], *self.items]
            slice_within_maxlen = slice(None, maxlen)
            result = ConstantVariable.create(None)
        elif name == "insert" and len(args) > 0 and self.is_mutable():
            if kwargs or len(args) != 2:
                raise_args_mismatch(
                    tx,
                    name,
                    "2 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            if maxlen is not None and len(self.items) == maxlen:
                raise_observed_exception(
                    IndexError, tx, args=["deque already at its maximum size"]
                )
            result = super().call_method(tx, name, args, kwargs)
        else:
            result = super().call_method(tx, name, args, kwargs)

        if (
            slice_within_maxlen is not None
            and maxlen is not None
            and len(self.items) > maxlen
        ):
            self.items[:] = self.items[slice_within_maxlen]
        return result

    def call_obj_hasattr(
        self, tx: "InstructionTranslator", name: str
    ) -> ConstantVariable:
        if self.python_type() is collections.deque:
            return variables.ConstantVariable.create(name in collections.deque.__dict__)
        return super().call_obj_hasattr(tx, name)


class TupleVariable(BaseListVariable):
    def python_type(self) -> type[tuple]:  # type: ignore[type-arg]
        return tuple

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(length={len(self.items)})"

    def debug_repr(self) -> str:
        return self.debug_repr_helper("(", ")")

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.foreach(self.items)
        codegen.append_output(create_build_tuple(len(self.items)))

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        if name == "__class__":
            source = AttrSource(self.source, name) if self.source else None
            class_type = self.python_type()
            if class_type is tuple:
                return variables.BuiltinVariable(class_type, source=source)
            else:
                return variables.UserDefinedClassVariable(class_type, source=source)
        return super().var_getattr(tx, name)

    def call_obj_hasattr(
        self, tx: "InstructionTranslator", name: str
    ) -> ConstantVariable:
        if self.python_type() is not tuple:
            return super().call_obj_hasattr(tx, name)
        return variables.ConstantVariable.create(hasattr((), name))

    def is_python_hashable(self) -> bool:
        return all(item.is_python_hashable() for item in self.items)

    def get_python_hash(self) -> int:
        items = tuple(x.get_python_hash() for x in self.items)
        return hash(items)

    def is_python_equal(self, other: object) -> bool:
        return isinstance(other, variables.TupleVariable) and all(
            a.is_python_equal(b) for (a, b) in zip(self.items, other.items)
        )


class SizeVariable(TupleVariable):
    """torch.Size(...)"""

    _nonvar_fields = {
        "proxy",
        *TupleVariable._nonvar_fields,
    }

    def __init__(
        self,
        items: list[VariableTracker],
        proxy: Optional[torch.fx.Proxy] = None,
        **kwargs: Any,
    ) -> None:
        self.proxy = proxy
        super().__init__(items, **kwargs)

    def debug_repr(self) -> str:
        return self.debug_repr_helper("torch.Size([", "])")

    def python_type(self) -> type:
        return torch.Size

    def as_proxy(self) -> Any:
        if self.proxy is not None:
            return self.proxy

        # torch.Size needs special handling.  Normally, we pun a list-like
        # container to directly contain Proxy/Node objects from FX, and FX
        # knows to look inside containers (via map_aggregate).  But torch.Size
        # is weird; although it subclasses from tuple, it doesn't allow
        # members which aren't int-like (rejecting Proxy and Node).  This
        # means we can't use the normal representation trick
        # torch.Size([proxy0, proxy1]).  I looked into seeing if I could
        # relax torch.Size in PyTorch proper, but if torch.Size constructor
        # sees a type that it doesn't recognize, it will try to call
        # __index__() on it, so there is no BC way to actually change this
        # behavior (though it occurs to me that I could have just added a
        # YOLO no checking alternate constructor.)
        #
        # To work around this problem, I represent a torch.Size proxy as
        # a straight up proxy, that would have been constructed by taking
        # the constituent proxies as arguments.  This trick can be generally
        # used for any construct that we need a proxy for but we can't
        # directly represent as an aggregate; I don't see very many examples
        # of this in torchdynamo though!

        # Look for a proxy.  If there are none, do the legacy behavior
        tracer = None
        proxies = self._as_proxy()
        for proxy in proxies:
            if isinstance(proxy, torch.fx.Proxy):
                tracer = proxy.tracer
                break

        if tracer is None:
            return torch.Size(proxies)

        proxy = tracer.create_proxy("call_function", torch.Size, (proxies,), {})
        set_example_value(
            proxy.node,
            torch.Size(
                [
                    p.node.meta["example_value"] if not isinstance(p, int) else p
                    for p in proxies
                ]
            ),
        )
        return proxy

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(lambda: codegen.load_import_from("torch", "Size"))
        codegen.foreach(self.items)
        build_torch_size = [
            create_build_tuple(len(self.items)),
        ] + create_call_function(1, False)
        codegen.extend_output(build_torch_size)

    def unpack_var_sequence(self, tx: "InstructionTranslator") -> list[VariableTracker]:
        return list(self.items)

    def numel(self, tx: "InstructionTranslator") -> VariableTracker:
        from .builtin import BuiltinVariable
        from .tensor import SymNodeVariable

        const_result = 1
        sym_sizes = []

        for v in self.items:
            if v.is_python_constant():
                const_result *= v.as_python_constant()
            else:
                assert isinstance(v, SymNodeVariable), type(v)
                # Delay proxy calls  until we know it will be necessary
                sym_sizes.append(v)

        result = ConstantVariable.create(const_result)
        if sym_sizes and const_result == 1:
            # Skip multiplying by 1
            result, *sym_sizes = sym_sizes

        if not sym_sizes or const_result == 0:
            return result

        mul = BuiltinVariable(operator.mul)
        for v in sym_sizes:
            result = mul.call_function(tx, [result, v], {})
        return result

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "__getitem__":
            if kwargs or len(args) != 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            out = self.get_item_dyn(tx, args[0])
            return out
        elif name == "numel":
            if args or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            return self.numel(tx)

        return super().call_method(tx, name, args, kwargs)

    def get_item_dyn(
        self, tx: "InstructionTranslator", arg: VariableTracker
    ) -> VariableTracker:
        from .tensor import SymNodeVariable, TensorVariable

        if isinstance(arg, SymNodeVariable):
            index = arg.sym_num
        elif isinstance(arg, TensorVariable):
            value = get_fake_value(arg.as_proxy().node, tx)
            if value.constant is None or value.constant.numel() != 1:
                unimplemented(
                    gb_type="Indexing torch.Size with non-scalar tensor",
                    context=f"get_item_dyn {self} {arg}",
                    explanation=(
                        "Attempted to index torch.Size with a tensor that is not a scalar constant."
                    ),
                    hints=[*graph_break_hints.USER_ERROR],
                )
            index = value.constant.item()
        else:
            index = arg.as_python_constant()

        if isinstance(index, slice):
            return SizeVariable(self.items[index])
        else:
            assert isinstance(index, (int, torch.SymInt))
            return self.items[index]

    def call_obj_hasattr(
        self, tx: "InstructionTranslator", name: str
    ) -> ConstantVariable:
        return variables.ConstantVariable.create(hasattr(torch.Size, name))


class NamedTupleVariable(UserDefinedTupleVariable):
    _nonvar_fields = {
        "tuple_cls",
        "dynamic_attributes",
        *UserDefinedTupleVariable._nonvar_fields,
    }

    def __init__(
        self,
        items: list[VariableTracker],
        tuple_cls: type[tuple],
        dynamic_attributes: Optional[dict[str, VariableTracker]] = None,
        tuple_vt: Optional[TupleVariable] = None,
        **kwargs: Any,
    ) -> None:
        if tuple_vt is None:
            assert getattr(kwargs, "source", None) is None
            tuple_vt = variables.TupleVariable(
                items, mutation_type=kwargs.get("mutation_type", ValueMutationNew())
            )

        if tuple_cls.__module__ == "torch.return_types":
            # Structseq: single iterable argument
            dummy_value = tuple_cls(items)
        else:
            # Namedtuple: positional arguments
            dummy_value = tuple_cls(*items)  # type: ignore[arg-type]

        super().__init__(
            value=dummy_value,
            tuple_vt=tuple_vt,
            init_args=items,
            **kwargs,
        )
        self.tuple_cls = tuple_cls
        if len(self.tuple_cls.__mro__) < 3:
            raise ValueError("NamedTuple should inherit from Tuple and Object.")
        self.dynamic_attributes = dynamic_attributes if dynamic_attributes else {}

    @property
    def items(self) -> list[VariableTracker]:
        return self._tuple_vt.items

    def is_namedtuple(self) -> bool:
        return isinstance(getattr(self.tuple_cls, "_fields", None), tuple) and callable(
            getattr(self.tuple_cls, "_make", None)
        )

    def is_structseq(self) -> bool:
        return not self.is_namedtuple()

    def fields(self) -> tuple[str, ...]:
        return namedtuple_fields(self.tuple_cls)

    def as_python_constant(self) -> Any:
        if self.is_structseq():
            # StructSequenceType(iterable)
            result = self.python_type()([x.as_python_constant() for x in self.items])
        else:
            # NamedTupleType(*iterable)
            result = self.python_type()(*[x.as_python_constant() for x in self.items])

        # Apply dynamic attributes if any were set
        if self.dynamic_attributes:
            for attr_name, attr_value in self.dynamic_attributes.items():
                # Convert VariableTracker to Python constant if needed
                if hasattr(attr_value, "as_python_constant"):
                    python_value = attr_value.as_python_constant()
                else:
                    raise NotImplementedError(
                        "Can not convert dynamic attribute without python constant value to python constant."
                    )
                setattr(result, attr_name, python_value)

        return result

    def as_proxy(self) -> Any:
        if self.is_structseq():
            return self.python_type()([x.as_proxy() for x in self._tuple_vt.items])
        return self.python_type()(*[x.as_proxy() for x in self._tuple_vt.items])

    def reconstruct(self, codegen: "PyCodegen") -> None:
        if self.is_structseq():
            create_fn = self.tuple_cls
        else:
            create_fn = self.tuple_cls._make  # type: ignore[attr-defined]

        codegen.add_push_null(
            lambda: codegen.append_output(
                codegen.create_load_const_unchecked(create_fn)
            )
        )
        codegen.foreach(self._tuple_vt.items)
        codegen.extend_output(
            [
                create_build_tuple(len(self._tuple_vt.items)),
            ]
            + create_call_function(1, False)
        )

        # Apply initial dynamic attributes after construction (if any)
        # Runtime dynamic attributes are tracked via side effects system
        for name, value in self.dynamic_attributes.items():
            codegen.dup_top()
            codegen(value)
            codegen.extend_output(create_rot_n(2))
            codegen.store_attr(name)

    def _is_method_overridden(self, method_name: str) -> bool:
        if len(self.tuple_cls.__mro__) < 3:
            raise ValueError("NamedTuple should inherit from Tuple and Object.")
        if getattr(self.tuple_cls, method_name, None) == getattr(
            self.tuple_cls.__mro__[-3], method_name, None
        ):
            return False
        return True

    def is_python_equal(self, other: Any) -> bool:
        if isinstance(other, UserDefinedTupleVariable):
            return super().is_python_equal(other)
        elif isinstance(other, TupleVariable):
            return all(a.is_python_equal(b) for (a, b) in zip(self.items, other.items))
        return False

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if self._is_method_overridden(name):
            # Fall back to UserDefinedTupleVariable
            return super().call_method(tx, name, args, kwargs)
        elif name == "__eq__":
            if len(args) != 1 or kwargs:
                raise ValueError("Improper arguments for method.")
            return ConstantVariable(self.is_python_equal(args[0]))
        elif name == "__ne__":
            if len(args) != 1 or kwargs:
                raise ValueError("Improper arguments for method.")
            return ConstantVariable(not self.is_python_equal(args[0]))
        elif name == "__setattr__":
            if kwargs or len(args) != 2:
                raise_args_mismatch(
                    tx,
                    name,
                    "2 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            attr_var, value = args
            attr = attr_var.as_python_constant()

            if (
                # structseq is immutable
                self.is_structseq()
                # namedtuple directly created by `collections.namedtuple` is immutable
                or self.tuple_cls.__bases__ == (tuple,)
                or attr in self.fields()
            ):
                raise_observed_exception(AttributeError, tx)

            result = self.method_setattr_standard(tx, attr_var, value)
            # Also update self.dynamic_attributes
            self.dynamic_attributes[attr] = value
            return result

        return super().call_method(tx, name, args, kwargs)

    def python_type(self) -> type:
        return self.tuple_cls

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> "VariableTracker":
        if name == "_fields":
            source = NamedTupleFieldsSource(self.source) if self.source else None
            return VariableTracker.build(tx, self.fields(), source=source)

        if name in self.dynamic_attributes:
            return self.dynamic_attributes[name]

        fields = self.fields()
        if name in fields:
            field_index = fields.index(name)
            return self._tuple_vt.items[field_index]

        return super().var_getattr(tx, name)


class SliceVariable(VariableTracker):
    def __init__(
        self,
        items: Sequence[VariableTracker],
        tx: Optional["InstructionTranslator"] = None,
        **kwargs: Any,
    ) -> None:
        items_to_map = items
        start, stop, step = [variables.ConstantVariable.create(None)] * 3

        if len(items_to_map) == 1:
            (stop,) = items_to_map
        elif len(items_to_map) == 2:
            start, stop = items_to_map
        elif len(items_to_map) == 3:
            start, stop, step = items_to_map
        else:
            raise AssertionError

        # Convert TensorVariable to SymIntVariable by calling .item()
        # This decomposes a[:t] to u=t.item(); a[:u] at the dynamo level
        if start.is_tensor():
            assert tx is not None, (
                "tx is required when slice indices are TensorVariables"
            )
            start = start.call_method(tx, "item", [], {})
        if stop.is_tensor():
            assert tx is not None, (
                "tx is required when slice indices are TensorVariables"
            )
            stop = stop.call_method(tx, "item", [], {})
        if step.is_tensor():
            assert tx is not None, (
                "tx is required when slice indices are TensorVariables"
            )
            step = step.call_method(tx, "item", [], {})

        self.items = (start, stop, step)

        super().__init__(**kwargs)

    def debug_repr(self) -> str:
        return "slice(" + ", ".join(i.debug_repr() for i in self.items) + ")"

    def as_proxy(self) -> slice:
        return slice(*[x.as_proxy() for x in self.items])

    def python_type(self) -> type:
        return slice

    def as_python_constant(self) -> slice:
        return slice(*[guard_if_dyn(x) for x in self.items])

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.foreach(self.items)
        codegen.append_output(create_instruction("BUILD_SLICE", arg=len(self.items)))

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        if name in cmp_name_to_op_mapping:
            return variables.GetAttrVariable(self, name)
        fields = ["start", "stop", "step"]
        if name not in fields:
            unimplemented(
                gb_type="Unsupported attribute for slice() object",
                context=f"var_getattr {self} {name}",
                explanation=f"Expected attribute to be one of {','.join(fields)} "
                f"but got {name}",
                hints=[*graph_break_hints.USER_ERROR],
            )
        return self.items[fields.index(name)]


class ListIteratorVariable(IteratorVariable):
    _nonvar_fields = {
        "index",
        *IteratorVariable._nonvar_fields,
    }

    def __init__(
        self, items: list[VariableTracker], index: int = 0, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        assert isinstance(items, list)
        # Removing this check as it slows things down too much
        # https://github.com/pytorch/pytorch/pull/87533#issuecomment-1287574492

        # assert all(isinstance(x, VariableTracker) for x in items)
        self.items = items
        self.index = index
        self.is_exhausted = False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(length={len(self.items)}, index={repr(self.index)})"

    def next_variable(self, tx: "InstructionTranslator") -> VariableTracker:
        assert self.is_mutable()
        old_index = self.index
        if old_index >= len(self.items) or self.is_exhausted:
            self.is_exhausted = True
            raise_observed_exception(StopIteration, tx)

        tx.output.side_effects.mutation(self)
        self.index += 1
        return self.items[old_index]

    def call_obj_hasattr(
        self, tx: "InstructionTranslator", name: str
    ) -> ConstantVariable:
        return variables.ConstantVariable.create(hasattr(iter([]), name))

    def python_type(self) -> type:
        return type(iter([]))

    def as_python_constant(self) -> Any:
        if self.index > 0:
            raise NotImplementedError
        return iter([x.as_python_constant() for x in self.items])

    def has_unpack_var_sequence(self, tx: "InstructionTranslator") -> bool:
        return True

    def unpack_var_sequence(self, tx: "InstructionTranslator") -> list[VariableTracker]:
        if self.is_exhausted:
            return []
        self.is_exhausted = True
        return list(self.items[self.index :])

    def force_unpack_var_sequence(
        self, tx: "InstructionTranslator"
    ) -> list[VariableTracker]:
        return self.unpack_var_sequence(tx)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        if not self.is_exhausted:
            remaining_items = self.items[self.index :]
        else:
            remaining_items = []
        codegen.foreach(remaining_items)
        codegen.extend_output(
            [
                create_build_tuple(len(remaining_items)),
                create_instruction("GET_ITER"),
            ]
        )


class TupleIteratorVariable(ListIteratorVariable):
    pass


class RangeIteratorVariable(IteratorVariable):
    # only needed for isinstance(..., range_iterator) to work
    _nonvar_fields = {
        "iter_obj",
    }

    def __init__(
        self, start: int, stop: int, step: int, len_: int, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.start = start
        self.stop = stop
        self.step = step
        self.len = len_

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "__next__":
            return self.next_variable(tx)
        elif name == "__iter__":
            return self
        return super().call_method(tx, name, args, kwargs)

    def call_obj_hasattr(
        self, tx: "InstructionTranslator", name: str
    ) -> ConstantVariable:
        if self.python_type() is range_iterator:
            ri = iter(range(0))
            return ConstantVariable(hasattr(ri, name))
        return super().call_obj_hasattr(tx, name)

    def next_variable(self, tx: "InstructionTranslator") -> VariableTracker:
        if self.len <= 0:
            raise_observed_exception(StopIteration, tx)

        self.len -= 1
        current = self.start
        self.start += self.step
        return ConstantVariable.create(current)

    def python_type(self) -> type:
        return range_iterator

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(
            lambda: codegen.append_output(codegen.create_load_python_module(range))  # type: ignore[arg-type]
        )
        codegen.append_output(codegen.create_load_const(self.start))
        codegen.append_output(codegen.create_load_const(self.stop))
        codegen.append_output(codegen.create_load_const(self.step))
        codegen.extend_output(create_call_function(3, False))
        codegen.append_output(create_instruction("GET_ITER"))
