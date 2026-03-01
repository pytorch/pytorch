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
import sys
from collections.abc import Callable, Sequence
from typing import Any, TYPE_CHECKING

from .. import graph_break_hints, polyfills, variables
from ..bytecode_transformation import (
    create_build_tuple,
    create_call_function,
    create_call_function_ex,
    create_instruction,
)
from ..exc import (
    handle_observed_exception,
    ObservedUserStopIteration,
    raise_observed_exception,
    unimplemented,
    UserError,
)
from .base import ValueMutationNew, VariableTracker
from .constant import ConstantVariable


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslator


MAX_ITERATOR_LIMIT = 100 * 1024  # 100k


class ItertoolsVariable(VariableTracker):
    def __init__(self, value: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.value = value

    def __repr__(self) -> str:
        return f"ItertoolsVariable({self.value})"

    def as_python_constant(self) -> Any:
        return self.value

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence["VariableTracker"],
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        # See also: module `torch._dynamo.polyfills.itertools`

        if self.value is itertools.product:
            if any(kw != "repeat" for kw in kwargs):
                unimplemented(
                    gb_type="Unsupported kwargs for itertools.product",
                    context=f"call_function {self} {args} {kwargs}",
                    explanation=f"Expected kwargs: 'repeat', but got "
                    f"{','.join(set(kwargs.keys()) - {'repeat'})}",
                    hints=[*graph_break_hints.USER_ERROR],
                )

            if "repeat" in kwargs:
                r = kwargs["repeat"].as_python_constant()
            else:
                r = 1
            seqs = [arg.force_unpack_var_sequence(tx) for arg in args]
            items = [
                variables.TupleVariable(list(item))
                for item in itertools.product(*seqs, repeat=r)
            ]
            return variables.ListIteratorVariable(
                items,  # type: ignore[arg-type]
                mutation_type=ValueMutationNew(),
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
                items,  # type: ignore[arg-type]
                mutation_type=ValueMutationNew(),
            )
        elif self.value is itertools.groupby:
            if any(kw != "key" for kw in kwargs):
                unimplemented(
                    gb_type="Unsupported kwargs for itertools.groupby",
                    context=f"call_function {self} {args} {kwargs}",
                    explanation=f"Expected kwargs: 'key', but got "
                    f"{','.join(set(kwargs.keys()) - {'key'})}",
                    hints=[*graph_break_hints.USER_ERROR],
                )

            def retrieve_const_key(key: VariableTracker) -> Any:
                if isinstance(key, variables.SymNodeVariable):
                    return key.evaluate_expr()
                elif key.is_python_constant():
                    return key.as_python_constant()
                else:
                    unimplemented(
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
                unimplemented(
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

                def keyfunc(x: VariableTracker) -> Any:
                    return retrieve_const_key(
                        kwargs.get("key").call_function(tx, [x], {})  # type: ignore[union-attr]
                    )

            else:

                def keyfunc(x: VariableTracker) -> Any:
                    return retrieve_const_key(x)

            result = []
            try:
                for k, v in itertools.groupby(seq, key=keyfunc):
                    result.append(
                        variables.TupleVariable(
                            [
                                (
                                    variables.ConstantVariable.create(k)
                                    if variables.ConstantVariable.is_literal(k)
                                    else k
                                ),
                                variables.ListIteratorVariable(
                                    list(v), mutation_type=ValueMutationNew()
                                ),
                            ],
                            mutation_type=ValueMutationNew(),
                        )
                    )
            except Exception as e:
                unimplemented(
                    gb_type="Unexpected failure during itertools.groupby() iteration",
                    context=f"call_function {self} {args} {kwargs}",
                    explanation="Unexpected failure in invoking function during groupby",
                    hints=[*graph_break_hints.SUPPORTABLE],
                    from_exc=e,
                )
            return variables.ListIteratorVariable(
                result,  # type: ignore[arg-type]
                mutation_type=ValueMutationNew(),
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
        elif (
            self.value is itertools.permutations
            and (len(args) == 1 or (len(args) == 2 and args[1].is_python_constant()))
            and not kwargs
        ):
            if len(args) == 2:
                r = args[1].as_python_constant()
            else:
                r = None
            items = [
                variables.TupleVariable(list(item))
                for item in itertools.permutations(
                    args[0].force_unpack_var_sequence(tx), r
                )
            ]
            return variables.ListIteratorVariable(
                items,  # type: ignore[arg-type]
                mutation_type=ValueMutationNew(),
            )
        else:
            return super().call_function(tx, args, kwargs)


class IteratorVariable(VariableTracker):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def next_variable(self, tx: "InstructionTranslator") -> VariableTracker:
        unimplemented(
            gb_type="Unimplemented next() call",
            context=f"next({self})",
            explanation="This abstract method must be implemented",
            hints=[*graph_break_hints.DYNAMO_BUG],
        )

    # NOTE: only call when unpacking this iterator safely done eagerly!
    # Normally, iterators are accessed lazily.
    # Example of safe eager unpacking: list(map(f, seq))
    # Example of unsafe eager unpacking: list(islice(map(f, seq), 5))
    def force_unpack_var_sequence(
        self, tx: "InstructionTranslator"
    ) -> list[VariableTracker]:
        result: list[VariableTracker] = []
        self.force_apply_to_var_sequence(tx, result.append)
        return result

    def force_apply_to_var_sequence(
        self, tx: "InstructionTranslator", fn: Callable[[Any], Any]
    ) -> None:
        while True:
            try:
                fn(self.next_variable(tx))
            except ObservedUserStopIteration:
                handle_observed_exception(tx)
                break

    # don't call force_unpack_var_sequence since it can mutate
    # IteratorVariable state!
    def has_force_unpack_var_sequence(self, tx: "InstructionTranslator") -> bool:
        return True

    def call_obj_hasattr(
        self, tx: "InstructionTranslator", name: str
    ) -> "ConstantVariable":
        if name == "__iter__" or name == "__next__":
            return variables.CONSTANT_VARIABLE_TRUE
        return super().call_obj_hasattr(tx, name)

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "__iter__":
            return self
        elif name == "__next__":
            return self.next_variable(tx)
        return super().call_method(tx, name, args, kwargs)


class ObjectIteratorVariable(IteratorVariable):
    """
    VariableTracker for iter(obj) that implements the iterator protocol (i.e.,
    has a `__next__` method).

    We use this class to track the state of the iterator and handle the case
    when the iterator is exhausted:

    Example usage:
        > b = iter(obj)
        > list(b)  # exhaust the iterator
        > list(b)  # empty list
    """

    def __init__(self, obj: VariableTracker, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.obj = obj
        self.generator_exhausted = False

    def next_variable(self, tx: "InstructionTranslator") -> VariableTracker:
        if self.generator_exhausted:
            raise_observed_exception(StopIteration, tx)

        try:
            return self.obj.next_variable(tx)
        except ObservedUserStopIteration:
            # Do not rely on the object to always return StopIteration once it
            # is exhausted.
            self.generator_exhausted = True
            raise


class RepeatIteratorVariable(IteratorVariable):
    def __init__(self, item: VariableTracker, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.item = item

    # Repeat needs no mutation, clone self
    def next_variable(self, tx: "InstructionTranslator") -> VariableTracker:
        return self.item

    def reconstruct(self, codegen: "PyCodegen") -> None:
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
    def __init__(
        self,
        item: int | VariableTracker = 0,
        step: int | VariableTracker = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(item, VariableTracker):
            item = ConstantVariable.create(item)
        if not isinstance(step, VariableTracker):
            step = ConstantVariable.create(step)
        self.item = item
        self.step = step

    def next_variable(self, tx: "InstructionTranslator") -> VariableTracker:
        assert self.is_mutable()
        old_item = self.item
        tx.output.side_effects.mutation(self)
        self.item = self.item.call_method(tx, "__add__", [self.step], {})
        return old_item

    def reconstruct(self, codegen: "PyCodegen") -> None:
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
        iterables: list[VariableTracker],
        strict: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        assert isinstance(iterables, list)
        # can be list[Variable] or VariableTracker (with next_variable implemented)
        self.iterables = iterables
        self.index = 0
        self.strict = strict

    def python_type(self) -> type[zip]:  # type: ignore[type-arg]
        return zip

    def has_unpack_var_sequence(self, tx: "InstructionTranslator") -> bool:
        return all(
            isinstance(it, list) or it.has_unpack_var_sequence(tx)
            for it in self.iterables
        )

    def unpack_var_sequence(
        self, tx: "InstructionTranslator"
    ) -> list["VariableTracker"]:
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

    def next_variable(self, tx: "InstructionTranslator") -> VariableTracker:
        assert self.is_mutable()

        if len(self.iterables) == 0:
            raise_observed_exception(StopIteration, tx)

        old_index = self.index
        args = []

        def get_item(
            it: list[VariableTracker] | VariableTracker,
        ) -> VariableTracker:
            if isinstance(it, list):
                if old_index >= len(it):
                    raise_observed_exception(StopIteration, tx)
                return it[old_index]
            else:
                return it.next_variable(tx)

        idx: int | None = None
        try:
            for idx, it in enumerate(self.iterables):  # noqa:B007
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
                    ValueError,  # type: ignore[arg-type]
                    "zip() has one argument of len differing from others",
                ) from None
            raise

        tx.output.side_effects.mutation(self)
        self.index += 1
        return variables.TupleVariable(args)

    def reconstruct_items(self, codegen: "PyCodegen") -> None:
        for it in self.iterables:
            if isinstance(it, list):
                remaining_items = it[self.index :]
                codegen.foreach(remaining_items)
                codegen.append_output(create_build_tuple(len(remaining_items)))
            else:
                codegen(it)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(
            lambda: codegen.load_import_from("builtins", "zip"), call_function_ex=True
        )
        self.reconstruct_items(codegen)
        codegen.append_output(create_build_tuple(len(self.iterables)))
        codegen.extend_output(
            [
                codegen.create_load_const("strict"),
                codegen.create_load_const(self.strict),
                create_instruction("BUILD_MAP", arg=1),
                *create_call_function_ex(True, False),
            ]
        )


class MapVariable(ZipVariable):
    """
    Represents map(fn, *iterables)
    """

    def __init__(
        self,
        fn: VariableTracker,
        iterables: list[VariableTracker],
        **kwargs: Any,
    ) -> None:
        super().__init__(iterables, **kwargs)
        self.fn = fn

    def python_type(self) -> type:
        return map

    def has_unpack_var_sequence(self, tx: "InstructionTranslator") -> bool:
        return False

    def next_variable(self, tx: "InstructionTranslator") -> VariableTracker:
        args = super().next_variable(tx)
        return self.fn.call_function(tx, args.items, {})  # type: ignore[attr-defined]

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(
            lambda: codegen.load_import_from("builtins", "map"), call_function_ex=True
        )
        codegen(self.fn)
        self.reconstruct_items(codegen)
        codegen.append_output(create_build_tuple(len(self.iterables) + 1))
        if self.strict:
            assert sys.version_info >= (3, 14), (
                "Unexpected bug: map(strict=True) requires Python 3.14+"
            )
            codegen.extend_output(
                [
                    codegen.create_load_const("strict"),
                    codegen.create_load_const(self.strict),
                    create_instruction("BUILD_MAP", arg=1),
                    *create_call_function_ex(True, False),
                ]
            )
        else:
            codegen.extend_output(create_call_function_ex(False, False))


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
        iterable: list[VariableTracker],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.fn = fn
        self.iterable = iterable
        self.index = 0

    def python_type(self) -> type:
        return filter

    def has_unpack_var_sequence(self, tx: "InstructionTranslator") -> bool:
        return isinstance(self.iterable, list) or self.iterable.has_unpack_var_sequence(
            tx
        )

    def unpack_var_sequence(
        self, tx: "InstructionTranslator"
    ) -> list["VariableTracker"]:
        assert self.has_unpack_var_sequence(tx)
        it = None
        if isinstance(self.iterable, list):
            it = self.iterable[self.index :]
        else:
            it = self.iterable.unpack_var_sequence(tx)
        filtered = self.fn.call_function(tx, it, {})
        return [variables.TupleVariable([filtered])]

    def next_variable(self, tx: "InstructionTranslator") -> VariableTracker:
        def _next() -> VariableTracker:
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
            if self.fn.is_constant_none():
                res = item
            else:
                res = self.fn.call_function(tx, [item], {})
            pred_res = variables.UserFunctionVariable(
                polyfills.predicate  # type: ignore[arg-type]
            ).call_function(tx, [res], {})
            if pred_res.as_python_constant():
                return item

    def reconstruct_items(self, codegen: "PyCodegen") -> None:
        if isinstance(self.iterable, list):
            remaining_items = self.iterable[self.index :]
            codegen.foreach(remaining_items)
            codegen.append_output(create_build_tuple(len(remaining_items)))
        else:
            codegen(self.iterable)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(lambda: codegen.load_import_from("builtins", "filter"))
        codegen(self.fn)
        self.reconstruct_items(codegen)
        codegen.extend_output(create_call_function(2, False))
