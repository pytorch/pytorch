# mypy: ignore-errors

MAX_CYCLE = 3000

import itertools
import operator
import sys

from typing import Dict, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator

from .. import polyfill, variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import (
    handle_observed_user_stop_iteration,
    ObservedUserStopIteration,
    raise_observed_user_stop_iteration,
    unimplemented,
    UserError,
)

from .base import MutableLocal, VariableTracker
from .constant import ConstantVariable


class ItertoolsVariable(VariableTracker):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def __repr__(self):
        return f"ItertoolsVariable({self.value})"

    def python_type(self):
        return type(self.value)

    def as_python_constant(self):
        return self.value

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if (
            self.value is itertools.product
            and not kwargs
            and all(arg.has_unpack_var_sequence(tx) for arg in args)
        ):
            seqs = [arg.unpack_var_sequence(tx) for arg in args]
            items = []
            for item in itertools.product(*seqs):
                items.append(variables.TupleVariable(list(item)))
            return variables.ListIteratorVariable(items, mutable_local=MutableLocal())
        elif (
            self.value is itertools.chain
            and not kwargs
            and all(arg.has_unpack_var_sequence(tx) for arg in args)
        ):
            # TODO support itertools.chain with arbitrary iterables
            seqs = [arg.unpack_var_sequence(tx) for arg in args]
            items = list(itertools.chain.from_iterable(seqs))
            return variables.ListIteratorVariable(items, mutable_local=MutableLocal())
        elif self.value is itertools.accumulate:
            from .builtin import BuiltinVariable

            if any(key not in ["initial", "func"] for key in kwargs.keys()):
                unimplemented(
                    "Unsupported kwargs for itertools.accumulate: "
                    f"{','.join(set(kwargs.keys()) - {'initial', 'func'})}"
                )

            acc = kwargs.get("initial")

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
                    unimplemented(
                        "itertools.accumulate can only accept one of: `func` kwarg, pos 2 arg"
                    )
            else:
                unimplemented("Unsupported arguments for itertools.accumulate")

            items = []
            if acc is not None:
                items.append(acc)
            for item in seq:
                if acc is None:
                    acc = item
                else:
                    try:
                        acc = func(tx, [acc, item], {})
                    except Exception as e:
                        unimplemented(
                            f"Unexpected failure in invoking function during accumulate. Failed running func {func}({item}{acc})",
                            from_exc=e,
                        )
                items.append(acc)

            return variables.ListIteratorVariable(items, mutable_local=MutableLocal())
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
            return variables.ListIteratorVariable(items, mutable_local=MutableLocal())
        elif self.value is itertools.groupby:
            if any(kw != "key" for kw in kwargs.keys()):
                unimplemented(
                    "Unsupported kwargs for itertools.groupby: "
                    f"{','.join(set(kwargs.keys()) - {'key'})}"
                )

            def retrieve_const_key(key):
                if isinstance(key, variables.SymNodeVariable):
                    return key.evaluate_expr()
                elif isinstance(key, variables.ConstantVariable):
                    return key.as_python_constant()
                else:
                    unimplemented(
                        "Unsupported key type for itertools.groupby: " + str(type(key))
                    )

            if len(args) == 1 and args[0].has_unpack_var_sequence(tx):
                seq = args[0].unpack_var_sequence(tx)
                keyfunc = (
                    (
                        lambda x: (
                            retrieve_const_key(
                                kwargs.get("key").call_function(tx, [x], {})
                            )
                        )
                    )
                    if "key" in kwargs
                    else None
                )
            else:
                unimplemented("Unsupported arguments for itertools.groupby")

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
                                    list(v), mutable_local=MutableLocal()
                                ),
                            ],
                            mutable_local=MutableLocal(),
                        )
                    )
            except Exception as e:
                unimplemented(
                    "Unexpected failure when calling itertools.groupby",
                    from_exc=e,
                )
            return variables.ListIteratorVariable(result, mutable_local=MutableLocal())
        elif self.value is itertools.repeat:
            if len(args) < 2:
                return variables.RepeatIteratorVariable(
                    *args, mutable_local=MutableLocal()
                )

            from .builder import SourcelessBuilder

            return tx.inline_user_function_return(
                SourcelessBuilder.create(tx, polyfill.repeat), args, kwargs
            )
        elif self.value is itertools.count:
            return variables.CountIteratorVariable(*args, mutable_local=MutableLocal())
        elif self.value is itertools.cycle:
            return variables.CycleIteratorVariable(*args, mutable_local=MutableLocal())
        elif self.value is itertools.dropwhile:
            return variables.UserFunctionVariable(polyfill.dropwhile).call_function(
                tx, args, kwargs
            )
        elif self.value is itertools.zip_longest:
            return variables.UserFunctionVariable(polyfill.zip_longest).call_function(
                tx, args, kwargs
            )
        else:
            return super().call_function(tx, args, kwargs)


class IteratorVariable(VariableTracker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def next_variable(self, tx):
        unimplemented("abstract method, must implement")

    # NOTE: only call when unpacking this iterator safely done eagerly!
    # Normally, iterators are accessed lazily.
    # Example of safe eager unpacking: list(map(f, seq))
    # Example of unsafe eager unpacking: list(islice(map(f, seq), 5))
    def force_unpack_var_sequence(self, tx) -> List[VariableTracker]:
        result = []
        while True:
            try:
                result.append(self.next_variable(tx))
            except ObservedUserStopIteration:
                handle_observed_user_stop_iteration(tx)
                break
        return result

    # don't call force_unpack_var_sequence since it can mutate
    # IteratorVariable state!
    def has_force_unpack_var_sequence(self, tx) -> bool:
        return True


class RepeatIteratorVariable(IteratorVariable):
    def __init__(self, item: VariableTracker, **kwargs):
        super().__init__(**kwargs)
        self.item = item

    # Repeat needs no mutation, clone self
    def next_variable(self, tx):
        return self.item

    def reconstruct(self, codegen):
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
    def __init__(self, item: int = 0, step: int = 1, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(item, VariableTracker):
            item = ConstantVariable.create(item)
        if not isinstance(step, VariableTracker):
            step = ConstantVariable.create(step)
        self.item = item
        self.step = step

    def next_variable(self, tx):
        assert self.mutable_local
        old_item = self.item
        tx.output.side_effects.mutation(self)
        self.item = self.item.call_method(tx, "__add__", [self.step], {})
        return old_item

    def reconstruct(self, codegen):
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
        saved: List[VariableTracker] = None,
        saved_index: int = 0,
        item: Optional[VariableTracker] = None,
        **kwargs,
    ):
        if saved is None:
            saved = []
        super().__init__(**kwargs)
        self.iterator = iterator
        self.saved = saved
        self.saved_index = saved_index
        self.item = item

    def next_variable(self, tx):
        assert self.mutable_local

        if self.iterator is not None:
            try:
                new_item = self.iterator.next_variable(tx)
                if len(self.saved) > MAX_CYCLE:
                    unimplemented(
                        "input iterator to itertools.cycle has too many items"
                    )
                tx.output.side_effects.mutation(self)
                self.saved.append(new_item)
                self.item = new_item
                if self.item is None:
                    return self.next_variable(tx)
                return self.item
            except ObservedUserStopIteration:
                handle_observed_user_stop_iteration(tx)
                self.iterator = None
                return self.next_variable(tx)
        elif len(self.saved) > 0:
            tx.output.side_effects.mutation(self)
            self.saved_index = (self.saved_index + 1) % len(self.saved)
            return self.item
        else:
            raise_observed_user_stop_iteration(self, tx)


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
        iterables: List[Union[List[VariableTracker], VariableTracker]],
        strict: bool = False,
        **kwargs,
    ):
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

    def unpack_var_sequence(self, tx) -> List["VariableTracker"]:
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
        assert self.mutable_local
        old_index = self.index
        args = []

        def get_item(it):
            if isinstance(it, list):
                if old_index >= len(it):
                    raise_observed_user_stop_iteration(self, tx)
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
                            handle_observed_user_stop_iteration(tx)
                            continue
                        # no ObservedUserStopIteration - fall through to UserError
                        break
                    else:
                        # all iterables exhausted, raise original error
                        raise
                handle_observed_user_stop_iteration(tx)
                raise UserError(
                    ValueError,
                    "zip() has one argument of len differing from others",
                ) from None
            raise

        tx.output.side_effects.mutation(self)
        self.index += 1
        return variables.TupleVariable(args)

    def reconstruct_items(self, codegen):
        for it in self.iterables:
            if isinstance(it, list):
                remaining_items = it[self.index :]
                codegen.foreach(remaining_items)
                codegen.append_output(
                    create_instruction("BUILD_TUPLE", arg=len(remaining_items))
                )
            else:
                codegen(it)

    def reconstruct(self, codegen):
        codegen.add_push_null(lambda: codegen.load_import_from("builtins", "zip"))
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
        iterables: List[Union[List[VariableTracker], VariableTracker]],
        **kwargs,
    ):
        super().__init__(iterables, **kwargs)
        self.fn = fn

    def python_type(self):
        return map

    def has_unpack_var_sequence(self, tx) -> bool:
        return False

    def next_variable(self, tx):
        args = super().next_variable(tx)
        return self.fn.call_function(tx, args.items, {})

    def reconstruct(self, codegen):
        codegen.add_push_null(lambda: codegen.load_import_from("builtins", "map"))
        codegen(self.fn)
        self.reconstruct_items(codegen)
        codegen.extend_output(
            [
                create_instruction("BUILD_TUPLE", arg=len(self.iterables) + 1),
                create_instruction("CALL_FUNCTION_EX", arg=0),
            ]
        )


class EnumerateVariable(ZipVariable):
    def __init__(
        self,
        iterable: Union[List[VariableTracker], VariableTracker],
        start: int = 0,
        **kwargs,
    ):
        super().__init__(
            [CountIteratorVariable(start, mutable_local=MutableLocal()), iterable],
            **kwargs,
        )

    def reconstruct(self, codegen):
        codegen.add_push_null(lambda: codegen.load_import_from("builtins", "enumerate"))
        codegen(self.iterables[1])
        assert isinstance(self.iterables[0], CountIteratorVariable)
        codegen(self.iterables[0].item)
        codegen.extend_output(codegen.create_call_function_kw(2, ("start",), False))
