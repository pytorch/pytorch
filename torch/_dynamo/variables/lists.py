# mypy: ignore-errors

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
import inspect
import operator
from typing import Optional, TYPE_CHECKING

import torch
import torch.fx

from .. import graph_break_hints, polyfills, variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import raise_observed_exception, unimplemented_v2
from ..source import AttrSource
from ..utils import (
    cmp_name_to_op_mapping,
    cmp_name_to_op_str_mapping,
    get_fake_value,
    guard_if_dyn,
    iter_contains,
    Lit,
    namedtuple_fields,
    odict_values,
    raise_args_mismatch,
    set_example_value,
)
from .base import ValueMutationNew, VariableTracker
from .constant import ConstantVariable
from .functions import UserFunctionVariable, UserMethodVariable
from .iter import IteratorVariable


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslator


class BaseListVariable(VariableTracker):
    @staticmethod
    def cls_for_instance(obj):
        return BaseListVariable.cls_for(type(obj))

    @staticmethod
    def cls_for(obj):
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
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        assert isinstance(items, list)
        assert all(isinstance(x, VariableTracker) for x in items)
        self.items: list[VariableTracker] = items

    def _as_proxy(self):
        return [x.as_proxy() for x in self.items]

    def modified(self, items, **kwargs):
        return type(self)(items, **kwargs)

    @property
    def value(self):
        return self.as_python_constant()

    def debug_repr_helper(self, prefix, suffix):
        return prefix + ", ".join(i.debug_repr() for i in self.items) + suffix

    def as_python_constant(self):
        return self.python_type()([x.as_python_constant() for x in self.items])

    def as_proxy(self):
        assert self.python_type() is not SizeVariable
        return self.python_type()(self._as_proxy())

    def getitem_const(self, tx: "InstructionTranslator", arg: VariableTracker):
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

    def unpack_var_sequence(self, tx):
        return list(self.items)

    def call_method(
        self,
        tx,
        name,
        args: list["VariableTracker"],
        kwargs: dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        if name == "__getitem__":
            from .tensor import TensorVariable

            if len(args) != 1:
                msg = ConstantVariable.create(
                    f"{name} takes exactly one argument ({len(args)} given)"
                )
                raise_observed_exception(TypeError, tx, args=[msg])

            assert not kwargs and len(args) == 1
            if isinstance(args[0], TensorVariable):
                value = get_fake_value(args[0].as_proxy().node, tx)
                if value.constant is not None and value.constant.numel() == 1:
                    value = variables.ConstantVariable.create(value.constant.item())
                else:
                    unimplemented_v2(
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
            if len(args) != 1 or kwargs:
                raise_args_mismatch(tx, name)
            return iter_contains(self.unpack_var_sequence(tx), args[0], tx)
        elif name == "index":
            if not len(args):
                raise_args_mismatch(tx, name)

            return tx.inline_user_function_return(
                VariableTracker.build(tx, polyfills.index),
                [self] + list(args),
                kwargs,
            )
        elif name == "count":
            if len(args) != 1:
                raise_args_mismatch(tx, name)
            return VariableTracker.build(tx, operator.countOf).call_function(
                tx,
                [self, args[0]],
                kwargs,
            )
        elif name in ("__add__", "__iadd__"):
            if kwargs or len(args) != 1:
                raise_args_mismatch(tx, name)

            if type(self) != type(args[0]):
                tp_name = self.python_type_name()
                other = args[0].python_type_name()
                msg = ConstantVariable.create(
                    f'can only concatenate {tp_name} (not "{other}") to {tp_name}'
                )
                raise_observed_exception(TypeError, tx, args=[msg])

            if name == "__add__":
                return type(self)(self.items + args[0].items, source=self.source)
            else:
                self.items += args[0].items
                return self
        elif name in ("__mul__", "__imul__"):
            if kwargs or len(args) != 1:
                raise_args_mismatch(tx, name)

            if not (args[0].is_python_constant() and args[0].python_type() is int):
                msg = ConstantVariable.create(
                    f"can't multiply sequence by non-int type of '{args[0].python_type_name()}'"
                )
                raise_observed_exception(TypeError, tx, args=[msg])

            val = args[0].as_python_constant()

            if name == "__mul__":
                return type(self)(self.items * val, source=self.source)
            else:
                self.items *= val
                return self
        elif name in cmp_name_to_op_mapping:
            if len(args) != 1:
                raise_args_mismatch(tx, name)

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

        return super().call_method(tx, name, args, kwargs)


class RangeVariable(BaseListVariable):
    def __init__(self, items, **kwargs) -> None:
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

        assert stop is not None
        super().__init__([start, stop, step], **kwargs)

    def debug_repr(self):
        return self.debug_repr_helper("range(", ")")

    def python_type(self):
        return range

    def start(self):
        return self.items[0].as_python_constant()

    def stop(self):
        return self.items[1].as_python_constant()

    def step(self):
        return self.items[2].as_python_constant()

    def range_length(self):
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

    def _get_slice_indices(self, length, slice):
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

    def apply_index(self, index):
        length = self.range_length()
        if index < 0:
            index = length + index

        if index < 0 or index >= length:
            raise IndexError(f"index {index} is out of range")

        return variables.ConstantVariable.create(self.start() + (index * self.step()))

    def apply_slice(self, slice):
        (slice_start, slice_stop, slice_step) = self._get_slice_indices(
            self.range_length(), slice
        )

        def compute_item(index):
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

    def as_python_constant(self):
        return range(*[x.as_python_constant() for x in self.items])

    def getitem_const(self, tx: "InstructionTranslator", arg: VariableTracker):
        # implementations mimics https://github.com/python/cpython/blob/main/Objects/rangeobject.c
        index = arg.as_python_constant()

        if isinstance(index, slice):
            return self.apply_slice(index)
        else:
            return self.apply_index(index)

    def as_proxy(self):
        return self.python_type()(*self._as_proxy())

    def unpack_var_sequence(self, tx=None):
        return [variables.ConstantVariable.create(x) for x in self.as_python_constant()]

    def reconstruct(self, codegen: "PyCodegen") -> None:
        assert "range" not in codegen.tx.f_globals
        codegen.add_push_null(
            lambda: codegen.append_output(codegen.create_load_python_module(range))
        )
        codegen.foreach(self.items)
        codegen.extend_output(create_call_function(3, False))

    def var_getattr(self, tx: "InstructionTranslator", name):
        fields = ["start", "stop", "step"]
        if name not in fields:
            unimplemented_v2(
                gb_type="Unsupported attribute for range() object",
                context=f"var_getattr {self} {name}",
                explanation=f"Expected attribute to be one of {','.join(fields)} "
                f"but got {name}",
                hints=[*graph_break_hints.USER_ERROR],
            )
        return self.items[fields.index(name)]


class CommonListMethodsVariable(BaseListVariable):
    """
    Implement methods common to List and other List-like things
    """

    def call_method(
        self,
        tx,
        name,
        args: list["VariableTracker"],
        kwargs: dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        from .tensor import SymNodeVariable

        if name == "append" and self.is_mutable():
            assert not kwargs
            if len(args) != 1:
                raise_args_mismatch(tx, name)
            (arg,) = args
            tx.output.side_effects.mutation(self)
            self.items.append(arg)
            return ConstantVariable.create(None)
        elif name == "extend" and self.is_mutable():
            if len(args) != 1 or kwargs:
                raise_args_mismatch(tx, name)

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
                raise_args_mismatch(tx, name)
            idx, value = args
            if isinstance(idx, SymNodeVariable):
                const_idx = idx.evaluate_expr()
            else:
                const_idx = idx.as_python_constant()
            tx.output.side_effects.mutation(self)
            self.items.insert(const_idx, value)
            return ConstantVariable.create(None)
        elif name == "pop" and self.is_mutable():
            assert not kwargs
            if kwargs or len(args) > 1:
                raise_args_mismatch(tx, name)

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
                raise_observed_exception(TypeError, tx)
            tx.output.side_effects.mutation(self)
            self.items.clear()
            return ConstantVariable.create(None)
        elif (
            name == "__setitem__"
            and self.is_mutable()
            and args
            and (
                args[0].is_python_constant()
                or isinstance(args[0], SymNodeVariable)
                or (
                    isinstance(args[0], SliceVariable)
                    and all(
                        s.is_python_constant() or isinstance(s, SymNodeVariable)
                        for s in args[0].items
                    )
                )
            )
        ):
            assert not kwargs
            key, value = args
            tx.output.side_effects.mutation(self)
            if isinstance(key, SymNodeVariable):
                self.items[key.evaluate_expr()] = value
            elif isinstance(key, SliceVariable):
                if key.is_python_constant():
                    self.items[key.as_python_constant()] = list(value.items)
                else:
                    items = slice(
                        *[
                            s.evaluate_expr()
                            if isinstance(s, SymNodeVariable)
                            else s.as_python_constant()
                            for s in key.items
                        ]
                    )
                    self.items[items] = list(value.items)
            else:
                self.items[key.as_python_constant()] = value
            return ConstantVariable.create(None)
        elif name == "__delitem__" and self.is_mutable():
            if kwargs or len(args) != 1:
                raise_args_mismatch(tx, name)

            tx.output.side_effects.mutation(self)
            if args[0].is_python_constant() and isinstance(
                args[0].as_python_constant(), (int, slice)
            ):
                if isinstance(args[0], SymNodeVariable):
                    idx = args[0].evaluate_expr()
                else:
                    idx = args[0].as_python_constant()

                try:
                    self.items.__delitem__(idx)
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
                raise_args_mismatch(tx, name)
            items = list(self.items)
            return self.modified(items, mutation_type=ValueMutationNew())
        elif name == "reverse" and self.is_mutable():
            if args or kwargs:
                raise_args_mismatch(tx, name)
            self.items.reverse()
            tx.output.side_effects.mutation(self)
            return ConstantVariable.create(None)
        elif name == "remove" and self.is_mutable():
            if len(args) != 1 or kwargs:
                raise_args_mismatch(tx, name)

            idx = self.call_method(tx, "index", args, kwargs)
            self.call_method(tx, "pop", [idx], {})
            return ConstantVariable.create(None)
        else:
            return super().call_method(tx, name, args, kwargs)


class ListVariable(CommonListMethodsVariable):
    def python_type(self):
        return list

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(length={len(self.items)})"

    def debug_repr(self):
        return self.debug_repr_helper("[", "]")

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.foreach(self.items)
        codegen.append_output(create_instruction("BUILD_LIST", arg=len(self.items)))

    def call_method(
        self,
        tx,
        name,
        args: list["VariableTracker"],
        kwargs: dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        from .tensor import SymNodeVariable

        if name == "__setitem__" and self.is_mutable():
            if kwargs or len(args) != 2:
                raise_args_mismatch(tx, name)
            key, value = args

            if not key.is_python_constant():
                # probably will graph-break
                super().call_method(tx, name, args, kwargs)

            tx.output.side_effects.mutation(self)
            if isinstance(key, SliceVariable):
                if not value.has_force_unpack_var_sequence(tx):
                    msg = ConstantVariable.create("can only assign an iterable")
                    raise_observed_exception(TypeError, tx, args=[msg])

                key = key.as_python_constant()
                if key.step == 0:
                    msg = ConstantVariable.create("slice step cannot be zero")
                    raise_observed_exception(ValueError, tx, args=[msg])

                value = value.force_unpack_var_sequence(tx)
                try:
                    self.items[key] = value
                except Exception as exc:
                    raise_observed_exception(
                        type(exc),
                        tx,
                        args=list(map(ConstantVariable.create, exc.args)),
                    )
            else:
                if isinstance(key, SymNodeVariable):
                    key = key.evaluate_expr()
                else:
                    key = key.as_python_constant()

                try:
                    self.items[key] = value
                except (IndexError, TypeError) as e:
                    raise_observed_exception(
                        type(e), tx, args=list(map(ConstantVariable.create, e.args))
                    )
            return ConstantVariable.create(None)

        if name == "sort" and self.is_mutable():
            assert len(args) == 0
            key_fn_var = kwargs.pop("key", ConstantVariable.create(None))
            reverse = kwargs.pop(
                "reverse", ConstantVariable.create(False)
            ).as_python_constant()
            assert len(kwargs) == 0

            if (
                key_fn_var.is_python_constant()
                and key_fn_var.as_python_constant() is None
            ):
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
                    python_type = first_non_constant_key.python_type()
                except NotImplementedError:
                    python_type = "unknown"

                unimplemented_v2(
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
            return ConstantVariable.create(None)

        if name == "__init__" and self.is_mutable():
            assert not kwargs
            if len(args) == 0:
                return ConstantVariable.create(None)
            elif len(args) == 1 and args[0].has_force_unpack_var_sequence(tx):
                (arg,) = args
                tx.output.side_effects.mutation(self)
                self.items[:] = arg.force_unpack_var_sequence(tx)
                return ConstantVariable.create(None)

        return super().call_method(tx, name, args, kwargs)

    def var_getattr(self, tx, name):
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
    ) -> "VariableTracker":
        if self.python_type() is not list:
            return super().call_obj_hasattr(tx, name)
        return variables.ConstantVariable.create(hasattr([], name))


class DequeVariable(CommonListMethodsVariable):
    def __init__(self, items, maxlen=None, **kwargs) -> None:
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

    def python_type(self):
        return collections.deque

    def debug_repr(self):
        if self.maxlen.as_python_constant() is None:
            return self.debug_repr_helper(
                "deque([", "], maxlen=" + self.maxlen.debug_repr() + ")"
            )
        return self.debug_repr_helper("deque([", "])")

    def as_python_constant(self):
        return self.python_type()(
            [x.as_python_constant() for x in self.items],
            maxlen=self.maxlen.as_python_constant(),
        )

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(
            lambda: codegen.append_output(
                codegen.create_load_python_module(collections.deque)
            )
        )
        codegen.foreach(self.items)
        codegen.extend_output([create_instruction("BUILD_LIST", arg=len(self.items))])
        codegen(self.maxlen)
        codegen.extend_output(codegen.create_call_function_kw(2, ("maxlen",), False))

    def var_getattr(self, tx: "InstructionTranslator", name):
        if name == "maxlen":
            return self.maxlen
        return super().var_getattr(tx, name)

    def call_method(
        self,
        tx,
        name,
        args: list["VariableTracker"],
        kwargs: dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        if (
            name == "__setitem__"
            and self.is_mutable()
            and args
            and args[0].is_python_constant()
        ):
            assert len(args) == 2
            assert not kwargs
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
            assert len(args) == 1
            assert not kwargs
            # NOTE this is inefficient, but the alternative is to represent self.items
            # as a deque, which is a more intrusive change.
            args[0].force_apply_to_var_sequence(
                tx, lambda item: self.call_method(tx, "appendleft", [item], {})
            )
            slice_within_maxlen = slice(None, maxlen)
            result = ConstantVariable.create(None)
        elif name == "popleft" and self.is_mutable():
            assert not args
            assert not kwargs
            tx.output.side_effects.mutation(self)
            result, *self.items[:] = self.items
        elif name == "appendleft" and len(args) > 0 and self.is_mutable():
            assert len(args) == 1
            assert not kwargs
            tx.output.side_effects.mutation(self)
            self.items[:] = [args[0], *self.items]
            slice_within_maxlen = slice(None, maxlen)
            result = ConstantVariable.create(None)
        elif name == "insert" and len(args) > 0 and self.is_mutable():
            assert len(args) == 2
            assert not kwargs
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


class TupleVariable(BaseListVariable):
    def python_type(self):
        return tuple

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(length={len(self.items)})"

    def debug_repr(self):
        return self.debug_repr_helper("(", ")")

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.foreach(self.items)
        codegen.append_output(create_instruction("BUILD_TUPLE", arg=len(self.items)))

    def call_method(
        self,
        tx,
        name,
        args: list["VariableTracker"],
        kwargs: dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        return super().call_method(tx, name, args, kwargs)

    def var_getattr(self, tx, name):
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
    ) -> "VariableTracker":
        if self.python_type() is not tuple:
            return super().call_obj_hasattr(tx, name)
        return variables.ConstantVariable.create(hasattr((), name))


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
        **kwargs,
    ) -> None:
        self.proxy = proxy
        super().__init__(items, **kwargs)

    def debug_repr(self):
        return self.debug_repr_helper("torch.Size([", "])")

    def python_type(self):
        return torch.Size

    def as_proxy(self):
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
            create_instruction("BUILD_TUPLE", arg=len(self.items)),
        ] + create_call_function(1, False)
        codegen.extend_output(build_torch_size)

    def unpack_var_sequence(self, tx):
        return list(self.items)

    def numel(self, tx):
        from .builtin import BuiltinVariable
        from .tensor import SymNodeVariable

        const_result = 1
        sym_sizes = []

        for v in self.items:
            if isinstance(v, ConstantVariable):
                const_result *= v.value
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
        tx,
        name,
        args: list["VariableTracker"],
        kwargs: dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        if name == "__getitem__":
            assert not kwargs and len(args) == 1
            out = self.get_item_dyn(tx, args[0])
            return out
        elif name == "numel":
            assert not args and not kwargs
            return self.numel(tx)

        return super().call_method(tx, name, args, kwargs)

    def get_item_dyn(self, tx: "InstructionTranslator", arg: VariableTracker):
        from .tensor import SymNodeVariable

        if isinstance(arg, SymNodeVariable):
            index = arg.sym_num
        else:
            index = arg.as_python_constant()

        if isinstance(index, slice):
            return SizeVariable(self.items[index])
        else:
            assert isinstance(index, (int, torch.SymInt))
            return self.items[index]

    def call_obj_hasattr(
        self, tx: "InstructionTranslator", name: str
    ) -> "VariableTracker":
        return variables.ConstantVariable.create(hasattr(torch.Size, name))


class NamedTupleVariable(TupleVariable):
    _nonvar_fields = {
        "tuple_cls",
        "dynamic_attributes",
        *TupleVariable._nonvar_fields,
    }

    def __init__(self, items, tuple_cls, dynamic_attributes=None, **kwargs) -> None:
        super().__init__(items, **kwargs)
        self.tuple_cls = tuple_cls
        self.dynamic_attributes = {} if not dynamic_attributes else dynamic_attributes

    def is_namedtuple(self):
        return isinstance(getattr(self.tuple_cls, "_fields", None), tuple) and callable(
            getattr(self.tuple_cls, "_make", None)
        )

    def is_structseq(self):
        return not self.is_namedtuple()

    def fields(self):
        return namedtuple_fields(self.tuple_cls)

    def debug_repr(self):
        if self.is_structseq():
            # StructSequenceType(iterable)
            return repr(self.tuple_cls([Lit(x.debug_repr()) for x in self.items]))
        # NamedTupleType(*iterable)
        return repr(self.tuple_cls(*(Lit(x.debug_repr()) for x in self.items)))

    def python_type(self):
        return self.tuple_cls

    def as_python_constant(self):
        if self.is_structseq():
            # StructSequenceType(iterable)
            return self.python_type()([x.as_python_constant() for x in self.items])
        # NamedTupleType(*iterable)
        return self.python_type()(*[x.as_python_constant() for x in self.items])

    def as_proxy(self):
        assert self.python_type() is not SizeVariable
        if self.is_structseq():
            # StructSequenceType(iterable)
            return self.python_type()(self._as_proxy())
        # NamedTupleType(*iterable)
        return self.python_type()(*self._as_proxy())

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # Constructors:
        #   StructSequenceType(iterable)
        #   NamedTupleType(*iterable)
        #   NamedTupleType._make(iterable)
        create_fn = self.tuple_cls if self.is_structseq() else self.tuple_cls._make
        codegen.add_push_null(
            lambda: codegen.append_output(
                codegen.create_load_const_unchecked(create_fn)
            )
        )
        codegen.foreach(self.items)
        codegen.extend_output(
            [
                create_instruction("BUILD_TUPLE", arg=len(self.items)),
            ]
            + create_call_function(1, False)
        )

    def call_method(
        self,
        tx,
        name,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "__setattr__":
            assert len(args) == 2
            assert len(kwargs) == 0
            attr, value = args
            attr = attr.as_python_constant()
            if (
                # structseq is immutable
                self.is_structseq()
                # namedtuple directly created by `collections.namedtuple` is immutable
                or self.tuple_cls.__bases__ == (tuple,)
                # fields are immutable
                or attr in self.fields()
            ):
                raise_observed_exception(AttributeError, tx)
            # Subclass of namedtuple type can have dynamic attributes
            tx.output.side_effects.mutation(self)
            self.dynamic_attributes[attr] = value
            return ConstantVariable.create(None)
        return super().call_method(tx, name, args, kwargs)

    def var_getattr(self, tx: "InstructionTranslator", name):
        def check_and_create_method():
            method = inspect.getattr_static(self.tuple_cls, name, None)
            if isinstance(method, classmethod):
                # We need the unbounded cls method to avoid the inline __self__
                return UserMethodVariable(
                    method.__func__,
                    variables.UserDefinedClassVariable(self.tuple_cls),
                )
            elif isinstance(method, staticmethod):
                return UserFunctionVariable(method.__func__)
            elif inspect.isfunction(method):
                return UserMethodVariable(method, self)
            else:
                return None

        if name in self.dynamic_attributes:
            return self.dynamic_attributes[name]

        fields = self.fields()
        if name not in fields:
            method = check_and_create_method()
            if not method:
                return super().var_getattr(tx, name)
            return method
        return self.items[fields.index(name)]

    def call_obj_hasattr(
        self, tx: "InstructionTranslator", name: str
    ) -> "VariableTracker":
        return variables.ConstantVariable.create(
            name in self.dynamic_attributes or hasattr(self.tuple_cls, name)
        )


class SliceVariable(VariableTracker):
    def __init__(self, items, **kwargs) -> None:
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

        if isinstance(start, variables.TensorVariable) or isinstance(
            stop, variables.TensorVariable
        ):
            unimplemented_v2(
                gb_type="Dynamic slicing with Tensor arguments",
                context=f"SliceVariable start: {start}, stop: {stop}, step: {step}",
                explanation="Creating slices with Tensor arguments is not supported. "
                "e.g. `l[:x]`, where `x` is a 1-element tensor.",
                hints=[
                    *graph_break_hints.SUPPORTABLE,
                ],
            )
        self.items = (start, stop, step)

        super().__init__(**kwargs)

    def debug_repr(self):
        return self.debug_repr_helper("slice(", ")")

    def as_proxy(self):
        return slice(*[x.as_proxy() for x in self.items])

    def python_type(self):
        return slice

    def as_python_constant(self):
        return slice(*[guard_if_dyn(x) for x in self.items])

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.foreach(self.items)
        codegen.append_output(create_instruction("BUILD_SLICE", arg=len(self.items)))

    def var_getattr(self, tx: "InstructionTranslator", name):
        if name in cmp_name_to_op_mapping:
            return variables.GetAttrVariable(self, name)
        fields = ["start", "stop", "step"]
        if name not in fields:
            unimplemented_v2(
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

    def __init__(self, items, index: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        assert isinstance(items, list)
        # Removing this check as it slows things down too much
        # https://github.com/pytorch/pytorch/pull/87533#issuecomment-1287574492

        # assert all(isinstance(x, VariableTracker) for x in items)
        self.items = items
        self.index = index

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(length={len(self.items)}, index={repr(self.index)})"

    def next_variable(self, tx):
        assert self.is_mutable()
        old_index = self.index
        if old_index >= len(self.items):
            raise_observed_exception(StopIteration, tx)

        tx.output.side_effects.mutation(self)
        self.index += 1
        return self.items[old_index]

    def call_obj_hasattr(self, tx, name):
        return variables.ConstantVariable.create(hasattr(iter([]), name))

    def python_type(self):
        return type(iter([]))

    def as_python_constant(self):
        if self.index > 0:
            raise NotImplementedError
        return iter([x.as_python_constant() for x in self.items])

    def has_unpack_var_sequence(self, tx):
        return True

    def unpack_var_sequence(self, tx):
        r = list(self.items[self.index :])
        self.index = len(self.items)
        return r

    def force_unpack_var_sequence(self, tx) -> list[VariableTracker]:
        return self.unpack_var_sequence(tx)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        remaining_items = self.items[self.index :]
        codegen.foreach(remaining_items)
        codegen.extend_output(
            [
                create_instruction("BUILD_TUPLE", arg=len(remaining_items)),
                create_instruction("GET_ITER"),
            ]
        )


class TupleIteratorVariable(ListIteratorVariable):
    pass
