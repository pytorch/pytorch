# mypy: ignore-errors

"""
This module provides iterator-related variable tracking functionality for Dynamo.
It implements variable classes for handling Python iterators and itertools functions
during symbolic execution and tracing.

The module includes:
- Base iterator variable classes for tracking iterator state
- Implementations of built-in iterators (zip, map, filter)
- Support for itertools functions (product, accumulate, combinations, etc.)
- Mutation tracking and reconstruction capabilities for iterator operations

These classes integrate with Dynamo's variable tracking system to enable proper
handling of iterator operations during code transformation and optimization.
"""

import itertools
import operator
import sys
from typing import Optional, TYPE_CHECKING, Union

from .. import graph_break_hints, polyfills, variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import (
    handle_observed_exception,
    ObservedUserStopIteration,
    raise_observed_exception,
    unimplemented_v2,
    UserError,
)
from .base import ValueMutationNew, VariableTracker
from .constant import ConstantVariable


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslator


MAX_ITERATOR_LIMIT = 100 * 1024  # 100k


class ItertoolsVariable(VariableTracker):
    def __init__(self, value, **kwargs) -> None:
        super().__init__(**kwargs)
        self.value = value

    def __repr__(self) -> str:
        return f"ItertoolsVariable({self.value})"

    def as_python_constant(self):
        return self.value

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        # See also: module `torch._dynamo.polyfills.itertools`

        if (
            self.value is itertools.product
            and not kwargs
            and all(arg.has_unpack_var_sequence(tx) for arg in args)
        ):
            seqs = [arg.unpack_var_sequence(tx) for arg in args]
            items = [
                variables.TupleVariable(list(item)) for item in itertools.product(*seqs)
            ]
            return variables.ListIteratorVariable(
                items, mutation_type=ValueMutationNew()
            )
        elif self.value is itertools.accumulate:
            from .builtin import BuiltinVariable

            if any(key not in ["initial", "func"] for key in kwargs.keys()):
                unimplemented_v2(
                    gb_type="Unsupported kwargs for itertools.accumulate",
                    context=f"call_function {self} {args} {kwargs}",
                    explanation=f"Expected kwargs: 'initial', 'func', but got "
                    f"{','.join(set(kwargs.keys()) - {'initial', 'func'})}",
                    hints=[*graph_break_hints.USER_ERROR],
                )

            if len(args) in [1, 2] and args[0].has_unpack_var_sequence(tx):
                seq = args[0].unpack_var_sequence(tx)

                if "func" in kwargs and len(args) == 1:
                    func = kwargs["func"].call_function
                elif len(args) == 2:
                    func = args[1].call_function
                elif len(args) == 1:
                    # Default to operator.add
                    func = BuiltinVariable(operator.add).call_function
                else:
                    unimplemented_v2(
                        gb_type="Unsupported `func` in itertools.accumulate",
                        context=f"call_function {self} {args} {kwargs}",
                        explanation="Dynamo does not know how to get the "
                        "function to use for itertools.accumulate. "
                        "itertools.accumulate expects the `func` as the second "
                        "argument or as a keyword argument.",
                        hints=[*graph_break_hints.USER_ERROR],
                    )
            else:
                unimplemented_v2(
                    gb_type="Unsupported arguments for itertools.accumulate",
                    context=f"call_function {self} {args} {kwargs}",
                    explanation="Dynamo does not know how to trace "
                    f"itertools.accumulate with args: {args} and kwargs: {kwargs}. "
                    "itertools.accumulate expects an iterable, an optional "
                    "binary function for accumulation, and an optional initial "
                    "value to set the starting state.",
                    hints=[
                        "Make sure the arguments to itertools.accumulate are correct.",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )

            items = []
            acc = kwargs.get("initial")
            if acc is not None:
                items.append(acc)
            for item in seq:
                if acc is None:
                    acc = item
                else:
                    try:
                        acc = func(tx, [acc, item], {})
                    except Exception as e:
                        unimplemented_v2(
                            gb_type="Unexpected failure during itertools.accumulate() iteration",
                            context=f"call_function {self} {args} {kwargs}",
                            explanation="Unexpected failure in invoking function during accumulate. "
                            f"Failed running func {func}({item}{acc})",
                            hints=[*graph_break_hints.DIFFICULT],
                            from_exc=e,
                        )
                items.append(acc)

            return variables.ListIteratorVariable(
                items, mutation_type=ValueMutationNew()
            )
        elif (
            self.value is itertools.combinations
            and not kwargs
            and len(args) == 2
            and args[0].has_unpack_var_sequence(tx)
            and args[1].is_python_constant()
        ):
            iterable = args[0].unpack_var_sequence(tx)
            r = args[1].as_python_constant()

            items = []
            for item in itertools.combinations(iterable, r):
                items.append(variables.TupleVariable(list(item)))
            return variables.ListIteratorVariable(
                items, mutation_type=ValueMutationNew()
            )
        elif self.value is itertools.groupby:
            if any(kw != "key" for kw in kwargs.keys()):
                unimplemented_v2(
                    gb_type="Unsupported kwargs for itertools.groupby",
                    context=f"call_function {self} {args} {kwargs}",
                    explanation=f"Expected kwargs: 'key', but got "
                    f"{','.join(set(kwargs.keys()) - {'key'})}",
                    hints=[*graph_break_hints.USER_ERROR],
                )

            def retrieve_const_key(key):
                if isinstance(key, variables.SymNodeVariable):
                    return key.evaluate_expr()
                elif isinstance(key, variables.ConstantVariable):
                    return key.as_python_constant()
                else:
                    unimplemented_v2(
                        gb_type="Unsupported key type for itertools.groupby",
                        context=f"call_function {self} {args} {kwargs}",
                        explanation="Dynamo does not know how to trace "
                        f"itertools.groupby with key type: {str(type(key))}. "
                        "We only support grouping keys that are constants (int, float, str, etc.)",
                        hints=[*graph_break_hints.SUPPORTABLE],
                    )

            if len(args) == 1 and args[0].has_unpack_var_sequence(tx):
                seq = args[0].unpack_var_sequence(tx)
            else:
                unimplemented_v2(
                    gb_type="Unsupported arguments for itertools.groupby",
                    context=f"call_function {self} {args} {kwargs}",
                    explanation="Dynamo does not know how to trace "
                    f"itertools.groupby with args: {args} and kwargs: {kwargs}. "
                    "itertools.groupby expects an iterable to group and an "
                    "optional key function to determine groupings.",
                    hints=[
                        "Make sure the arguments to itertools.groupby are correct.",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )

            if "key" in kwargs:

                def keyfunc(x):
                    return retrieve_const_key(
                        kwargs.get("key").call_function(tx, [x], {})
                    )

            else:

                def keyfunc(x):
                    return retrieve_const_key(x)

            result = []
            try:
                for k, v in itertools.groupby(seq, key=keyfunc):
                    result.append(
                        variables.TupleVariable(
                            [
                                variables.ConstantVariable.create(k)
                                if variables.ConstantVariable.is_literal(k)
                                else k,
                                variables.ListIteratorVariable(
                                    list(v), mutation_type=ValueMutationNew()
                                ),
                            ],
                            mutation_type=ValueMutationNew(),
                        )
                    )
            except Exception as e:
                unimplemented_v2(
                    gb_type="Unexpected failure during itertools.groupby() iteration",
                    context=f"call_function {self} {args} {kwargs}",
                    explanation="Unexpected failure in invoking function during groupby",
                    hints=[*graph_break_hints.SUPPORTABLE],
                    from_exc=e,
                )
            return variables.ListIteratorVariable(
                result, mutation_type=ValueMutationNew()
            )
        elif self.value is itertools.repeat:
            if len(args) < 2:
                return variables.RepeatIteratorVariable(
                    *args, mutation_type=ValueMutationNew()
                )

            return tx.inline_user_function_return(
                VariableTracker.build(tx, polyfills.repeat), args, kwargs
            )
        elif self.value is itertools.count:
            return variables.CountIteratorVariable(
                *args, mutation_type=ValueMutationNew()
            )
        elif self.value is itertools.cycle:
            return variables.CycleIteratorVariable(
                *args, mutation_type=ValueMutationNew()
            )
        else:
            return super().call_function(tx, args, kwargs)


class IteratorVariable(VariableTracker):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def next_variable(self, tx):
        unimplemented_v2(
            gb_type="Unsupported next() call",
            context=f"next({self})",
            explanation="This abstract method must be implemented",
            hints=[*graph_break_hints.DYNAMO_BUG],
        )

    # NOTE: only call when unpacking this iterator safely done eagerly!
    # Normally, iterators are accessed lazily.
    # Example of safe eager unpacking: list(map(f, seq))
    # Example of unsafe eager unpacking: list(islice(map(f, seq), 5))
    def force_unpack_var_sequence(self, tx) -> list[VariableTracker]:
        result = []
        self.force_apply_to_var_sequence(tx, result.append)
        return result

    def force_apply_to_var_sequence(self, tx, fn) -> None:
        while True:
            try:
                fn(self.next_variable(tx))
            except ObservedUserStopIteration:
                handle_observed_exception(tx)
                break

    # don't call force_unpack_var_sequence since it can mutate
    # IteratorVariable state!
    def has_force_unpack_var_sequence(self, tx) -> bool:
        return True


class RepeatIteratorVariable(IteratorVariable):
    def __init__(self, item: VariableTracker, **kwargs) -> None:
        super().__init__(**kwargs)
        self.item = item

    # Repeat needs no mutation, clone self
    def next_variable(self, tx):
        return self.item

    def reconstruct(self, codegen: "PyCodegen"):
        codegen.add_push_null(
            lambda: codegen.extend_output(
                [
                    codegen.create_load_python_module(itertools),
                    codegen.create_load_attr("repeat"),
                ]
            )
        )
        codegen(self.item)
        codegen.extend_output(create_call_function(1, False))


class CountIteratorVariable(IteratorVariable):
    def __init__(self, item: int = 0, step: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        if not isinstance(item, VariableTracker):
            item = ConstantVariable.create(item)
        if not isinstance(step, VariableTracker):
            step = ConstantVariable.create(step)
        self.item = item
        self.step = step

    def next_variable(self, tx):
        assert self.is_mutable()
        old_item = self.item
        tx.output.side_effects.mutation(self)
        self.item = self.item.call_method(tx, "__add__", [self.step], {})
        return old_item

    def reconstruct(self, codegen: "PyCodegen"):
        codegen.add_push_null(
            lambda: codegen.extend_output(
                [
                    codegen.create_load_python_module(itertools),
                    codegen.create_load_attr("count"),
                ]
            )
        )
        codegen(self.item)
        codegen(self.step)
        codegen.extend_output(create_call_function(2, False))


class CycleIteratorVariable(IteratorVariable):
    def __init__(
        self,
        iterator: IteratorVariable,
        saved: Optional[list[VariableTracker]] = None,
        saved_index: int = 0,
        item: Optional[VariableTracker] = None,
        **kwargs,
    ) -> None:
        if saved is None:
            saved = []
        super().__init__(**kwargs)
        self.iterator = iterator
        self.saved = saved
        self.saved_index = saved_index
        self.item = item

    def next_variable(self, tx):
        assert self.is_mutable()

        if self.iterator is not None:
            try:
                new_item = self.iterator.next_variable(tx)
                if len(self.saved) > MAX_ITERATOR_LIMIT:
                    unimplemented_v2(
                        gb_type="input iterator to itertools.cycle has too many items",
                        context=f"next({self})",
                        explanation=f"Has reached internal Dynamo max iterator limit: {MAX_ITERATOR_LIMIT}",
                        hints=[],
                    )
                tx.output.side_effects.mutation(self)
                self.saved.append(new_item)
                self.item = new_item
                if self.item is None:
                    return self.next_variable(tx)
                return self.item
            except ObservedUserStopIteration:
                handle_observed_exception(tx)
                self.iterator = None
                return self.next_variable(tx)
        elif len(self.saved) > 0:
            tx.output.side_effects.mutation(self)
            self.saved_index = (self.saved_index + 1) % len(self.saved)
            return self.item
        else:
            raise_observed_exception(StopIteration, tx)


class ZipVariable(IteratorVariable):
    """
    Represents zip(*iterables)
    """

    _nonvar_fields = {
        "index",
        "strict",
        *IteratorVariable._nonvar_fields,
    }

    def __init__(
        self,
        iterables: list[Union[list[VariableTracker], VariableTracker]],
        strict: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        assert isinstance(iterables, list)
        # can be list[Variable] or VariableTracker (with next_variable implemented)
        self.iterables = iterables
        self.index = 0
        self.strict = strict

    def python_type(self):
        return zip

    def has_unpack_var_sequence(self, tx) -> bool:
        return all(
            isinstance(it, list) or it.has_unpack_var_sequence(tx)
            for it in self.iterables
        )

    def unpack_var_sequence(self, tx) -> list["VariableTracker"]:
        assert self.has_unpack_var_sequence(tx)
        iterables = []
        for it in self.iterables:
            if isinstance(it, list):
                iterables.append(it[self.index :])
            else:
                iterables.append(it.unpack_var_sequence(tx))
        kwargs = {"strict": self.strict} if self.strict else {}
        zipped = zip(*iterables, **kwargs)
        return [variables.TupleVariable(list(var)) for var in zipped]

    def next_variable(self, tx):
        assert self.is_mutable()
        old_index = self.index
        args = []

        def get_item(it):
            if isinstance(it, list):
                if old_index >= len(it):
                    raise_observed_exception(StopIteration, tx)
                return it[old_index]
            else:
                return it.next_variable(tx)

        try:
            for idx, it in enumerate(self.iterables):
                args.append(get_item(it))
        except ObservedUserStopIteration:
            if self.strict:
                if idx == 0:
                    # all other iterables should be exhausted
                    for it in self.iterables:
                        try:
                            get_item(it)
                        except ObservedUserStopIteration:
                            handle_observed_exception(tx)
                            continue
                        # no ObservedUserStopIteration - fall through to UserError
                        break
                    else:
                        # all iterables exhausted, raise original error
                        raise
                handle_observed_exception(tx)
                raise UserError(
                    ValueError,
                    "zip() has one argument of len differing from others",
                ) from None
            raise

        tx.output.side_effects.mutation(self)
        self.index += 1
        return variables.TupleVariable(args)

    def reconstruct_items(self, codegen: "PyCodegen"):
        for it in self.iterables:
            if isinstance(it, list):
                remaining_items = it[self.index :]
                codegen.foreach(remaining_items)
                codegen.append_output(
                    create_instruction("BUILD_TUPLE", arg=len(remaining_items))
                )
            else:
                codegen(it)

    def reconstruct(self, codegen: "PyCodegen"):
        codegen.add_push_null(
            lambda: codegen.load_import_from("builtins", "zip"), call_function_ex=True
        )
        self.reconstruct_items(codegen)
        codegen.append_output(
            create_instruction("BUILD_TUPLE", arg=len(self.iterables))
        )
        if sys.version_info >= (3, 10):
            codegen.extend_output(
                [
                    codegen.create_load_const("strict"),
                    codegen.create_load_const(self.strict),
                    create_instruction("BUILD_MAP", arg=1),
                    create_instruction("CALL_FUNCTION_EX", arg=1),
                ]
            )
        else:
            codegen.append_output(create_instruction("CALL_FUNCTION_EX", arg=0))


class MapVariable(ZipVariable):
    """
    Represents map(fn, *iterables)
    """

    def __init__(
        self,
        fn: VariableTracker,
        iterables: list[Union[list[VariableTracker], VariableTracker]],
        **kwargs,
    ) -> None:
        super().__init__(iterables, **kwargs)
        self.fn = fn

    def python_type(self):
        return map

    def has_unpack_var_sequence(self, tx) -> bool:
        return False

    def next_variable(self, tx):
        args = super().next_variable(tx)
        return self.fn.call_function(tx, args.items, {})

    def reconstruct(self, codegen: "PyCodegen"):
        codegen.add_push_null(
            lambda: codegen.load_import_from("builtins", "map"), call_function_ex=True
        )
        codegen(self.fn)
        self.reconstruct_items(codegen)
        codegen.extend_output(
            [
                create_instruction("BUILD_TUPLE", arg=len(self.iterables) + 1),
                create_instruction("CALL_FUNCTION_EX", arg=0),
            ]
        )


class FilterVariable(IteratorVariable):
    """
    Represents filter(fn, iterable)
    """

    _nonvar_fields = {
        "index",
        *IteratorVariable._nonvar_fields,
    }

    def __init__(
        self,
        fn: VariableTracker,
        iterable: Union[list[VariableTracker], VariableTracker],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.fn = fn
        self.iterable = iterable
        self.index = 0

    def python_type(self):
        return filter

    def has_unpack_var_sequence(self, tx) -> bool:
        return isinstance(self.iterable, list) or self.iterable.has_unpack_var_sequence(
            tx
        )

    def unpack_var_sequence(self, tx) -> list["VariableTracker"]:
        assert self.has_unpack_var_sequence(tx)
        it = None
        if isinstance(self.iterable, list):
            it = self.iterable[self.index :]
        else:
            it = self.iterable.unpack_var_sequence(tx)
        filtered = self.fn.call_function(tx, it, {})
        return [variables.TupleVariable([filtered])]

    def next_variable(self, tx):
        def _next():
            old_index = self.index
            if isinstance(self.iterable, list):
                if old_index >= len(self.iterable):
                    raise_observed_exception(StopIteration, tx)
                return self.iterable[old_index]
            else:
                return self.iterable.next_variable(tx)

        # A do-while loop to find elements that make fn return true
        while True:
            item = _next()
            self.index += 1
            res = self.fn.call_function(tx, [item], {})
            pred_res = variables.UserFunctionVariable(
                polyfills.predicate
            ).call_function(tx, [res], {})
            if pred_res.as_python_constant():
                return item

    def reconstruct_items(self, codegen: "PyCodegen"):
        if isinstance(self.iterable, list):
            remaining_items = self.iterable[self.index :]
            codegen.foreach(remaining_items)
            codegen.append_output(
                create_instruction("BUILD_TUPLE", arg=len(remaining_items))
            )
        else:
            codegen(self.iterable)

    def reconstruct(self, codegen: "PyCodegen"):
        codegen.add_push_null(lambda: codegen.load_import_from("builtins", "filter"))
        codegen(self.fn)
        self.reconstruct_items(codegen)
        codegen.extend_output(create_call_function(2, False))
