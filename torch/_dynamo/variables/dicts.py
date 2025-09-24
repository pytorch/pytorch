# mypy: ignore-errors

"""
Dictionary-related variable tracking classes for PyTorch Dynamo.

This module implements variable tracking for different types of dictionary-like objects:
- Regular Python dictionaries (dict)
- Ordered dictionaries (collections.OrderedDict)
- Default dictionaries (collections.defaultdict)
- Dictionary views (keys and values)
- Sets and frozensets (implemented internally using dictionaries)

These classes are responsible for tracking dictionary operations during graph compilation,
maintaining proper guards for dictionary mutations and key existence checks. They handle
dictionary creation, modification, key/value access, and view operations while ensuring
correct behavior in the compiled code through appropriate guard installation.

The implementation uses a special _HashableTracker wrapper to handle dictionary keys
while preserving proper aliasing semantics. Sets are implemented as dictionaries with
None values for efficiency and code reuse.
"""

import collections
import functools
import inspect
import operator
import types
from collections.abc import Hashable as py_Hashable
from typing import Optional, TYPE_CHECKING

from torch._subclasses.fake_tensor import is_fake

from .. import graph_break_hints, polyfills, variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import raise_observed_exception, unimplemented_v2
from ..guards import GuardBuilder, install_guard
from ..source import is_from_local_source
from ..utils import (
    cmp_name_to_op_mapping,
    dict_items,
    dict_keys,
    dict_values,
    istype,
    raise_args_mismatch,
    specialize_symnode,
)
from .base import ValueMutationNew, VariableTracker
from .constant import ConstantVariable


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslator


# [Adding a new supported class within the keys of ConstDictVarialble]
# - Add its tracker type to is_hashable
# - (perhaps) Define how it is compared in _HashableTracker._eq_impl


def was_instancecheck_override(obj):
    return type(obj).__dict__.get("__instancecheck__", False)


def raise_unhashable(arg, tx=None):
    if tx is None:
        from torch._dynamo.symbolic_convert import InstructionTranslator

        tx = InstructionTranslator.current_tx()
    raise_observed_exception(
        TypeError,
        tx,
        args=[ConstantVariable(f"unhashable type: {type(arg.realize())}")],
    )


def is_hashable(x):
    # NB - performing isinstance check on a LazVT realizes the VT, accidentally
    # inserting the guard. To avoid this, lazyVT `is_hashable` methods looks at
    # the underlying value without realizing the VT. Consider updating the
    # lazyVT `is_hashable` method if you see unnecessary guarding for a key VT.
    if (
        isinstance(x, variables.LazyVariableTracker)
        and not x.is_realized()
        and x.is_hashable()
    ):
        return True

    if isinstance(x, variables.TensorVariable):
        # Tensors are hashable if they have an example_value (a fake tensor)
        # Most VT's should have one.
        # It'd be nice if at some point we could assert that they all have one
        return x.as_proxy().node.meta.get("example_value") is not None
    elif isinstance(x, variables.TupleVariable):
        return all(is_hashable(e) for e in x.items)
    elif isinstance(x, variables.FrozenDataClassVariable):
        return all(is_hashable(e) for e in x.fields.values())
    elif (
        isinstance(x, variables.UserDefinedObjectVariable)
        and not was_instancecheck_override(x.value)
        and inspect.getattr_static(x.value, "__hash__") is int.__hash__
        and isinstance(x.value, int)
    ):
        return isinstance(x.value, py_Hashable)
    else:
        return isinstance(
            x,
            (
                variables.BuiltinVariable,
                variables.SymNodeVariable,
                variables.ConstantVariable,
                variables.EnumVariable,
                variables.FrozensetVariable,
                variables.UserDefinedClassVariable,
                variables.UserFunctionVariable,
                variables.SkipFunctionVariable,
                variables.misc.NumpyVariable,
                variables.NNModuleVariable,
                variables.UnspecializedNNModuleVariable,
                variables.MethodWrapperVariable,
                variables.TorchInGraphFunctionVariable,
                variables.TypingVariable,
                variables.FunctoolsPartialVariable,
                variables.WeakRefVariable,
                variables.TorchHigherOrderOperatorVariable,
            ),
        )


class ConstDictVariable(VariableTracker):
    CONTAINS_GUARD = GuardBuilder.DICT_CONTAINS

    _nonvar_fields = {
        "user_cls",
        *VariableTracker._nonvar_fields,
    }

    class _HashableTracker:
        """
        Auxiliary opaque internal class that wraps a VariableTracker and makes it hashable
        This should not be seen or touched by anything outside of ConstDictVariable and its children
        Note that it's also fine to put VTs into dictionaries and sets, but doing so does not take into account aliasing
        """

        def __init__(self, vt) -> None:
            # We specialize SymNodes
            vt = specialize_symnode(vt)
            # TODO Temporarily remove to figure out what keys are we breaking on
            # and add proper support for them
            if not is_hashable(vt):
                raise_unhashable(vt)
            self.vt = vt

        @property
        def underlying_value(self):
            if (
                isinstance(self.vt, variables.LazyVariableTracker)
                and not self.vt.is_realized()
                and self.vt.is_hashable()
            ):
                return self.vt.original_value()
            if isinstance(self.vt, variables.TensorVariable):
                x = self.vt.as_proxy().node.meta["example_value"]
            elif isinstance(self.vt, variables.TupleVariable):
                Hashable = ConstDictVariable._HashableTracker
                x = tuple(Hashable(e).underlying_value for e in self.vt.items)
            elif isinstance(self.vt, variables.NNModuleVariable):
                return self.vt.value
            elif isinstance(self.vt, variables.UnspecializedNNModuleVariable):
                return self.vt.value
            elif isinstance(self.vt, variables.UserFunctionVariable):
                return self.vt.get_function()
            elif isinstance(self.vt, variables.WeakRefVariable):
                # Access the underlying value inside the referent_vt for the key representation
                Hashable = ConstDictVariable._HashableTracker
                return Hashable(self.vt.referent_vt).underlying_value
            elif isinstance(self.vt, variables.FrozenDataClassVariable):
                Hashable = ConstDictVariable._HashableTracker
                fields_values = {
                    k: Hashable(v).underlying_value for k, v in self.vt.fields.items()
                }
                return variables.FrozenDataClassVariable.HashWrapper(
                    self.vt.python_type(), fields_values
                )
            elif isinstance(self.vt, variables.UserDefinedObjectVariable):
                # The re module in Python 3.13+ has a dictionary (_cache2) with
                # an object as key (`class _ZeroSentinel(int): ...`):
                # python test/dynamo/test_unittest.py CPythonTestLongMessage.test_baseAssertEqual
                return self.vt.value
            else:
                x = self.vt.as_python_constant()
            return x

        def __hash__(self):
            return hash(self.underlying_value)

        @staticmethod
        def _eq_impl(a, b):
            # TODO: Put this in utils and share it between variables/builtin.py and here
            if type(a) != type(b):
                return False
            elif isinstance(a, tuple):
                Hashable = ConstDictVariable._HashableTracker
                return len(a) == len(b) and all(
                    Hashable._eq_impl(u, v) for u, v in zip(a, b)
                )
            elif is_fake(a):
                return a is b
            else:
                return a == b

        def __eq__(self, other: "ConstDictVariable._HashableTracker") -> bool:
            Hashable = ConstDictVariable._HashableTracker
            assert isinstance(other, Hashable) or ConstantVariable.is_literal(other), (
                type(other)
            )
            if isinstance(other, Hashable):
                return Hashable._eq_impl(self.underlying_value, other.underlying_value)

            # constant
            return Hashable._eq_impl(self.underlying_value, other)

    def __init__(
        self,
        items: dict[VariableTracker, VariableTracker],
        user_cls=dict,
        **kwargs,
    ) -> None:
        # .clone() pass these arguments in kwargs but they're recreated a few
        # lines below
        if "original_items" in kwargs:
            kwargs.pop("original_items")
        if "should_reconstruct_all" in kwargs:
            kwargs.pop("should_reconstruct_all")

        super().__init__(**kwargs)

        Hashable = ConstDictVariable._HashableTracker

        # Keys will just be HashableTrackers when cloning, in any other case they'll be VariableTrackers
        assert all(
            isinstance(x, (VariableTracker, Hashable))
            and isinstance(v, VariableTracker)
            for x, v in items.items()
        )

        def make_hashable(key):
            return key if isinstance(key, Hashable) else Hashable(key)

        dict_cls = self._get_dict_cls_from_user_cls(user_cls)
        self.items = dict_cls({make_hashable(x): v for x, v in items.items()})
        # need to reconstruct everything if the dictionary is an intermediate value
        # or if a pop/delitem was executed
        self.should_reconstruct_all = not is_from_local_source(self.source)
        self.original_items = items.copy()
        self.user_cls = user_cls

    def _get_dict_cls_from_user_cls(self, user_cls):
        accepted_dict_types = (dict, collections.OrderedDict, collections.defaultdict)

        # avoid executing user code if user_cls is a dict subclass
        if user_cls in accepted_dict_types:
            dict_cls = user_cls
        else:
            # <Subclass, ..., dict, object>
            dict_cls = next(
                base for base in user_cls.__mro__ if base in accepted_dict_types
            )
        assert dict_cls in accepted_dict_types, dict_cls

        # Use a dict instead as the call "defaultdict({make_hashable(x): v ..})"
        # would fail as defaultdict expects a callable as first argument
        if dict_cls is collections.defaultdict:
            dict_cls = dict
        return dict_cls

    def as_proxy(self):
        return {k.vt.as_proxy(): v.as_proxy() for k, v in self.items.items()}

    def debug_repr(self):
        return (
            "{"
            + ", ".join(
                f"{k.vt.debug_repr()}: {v.debug_repr()}" for k, v in self.items.items()
            )
            + "}"
        )

    def as_python_constant(self):
        return {
            k.vt.as_python_constant(): v.as_python_constant()
            for k, v in self.items.items()
        }

    def keys_as_python_constant(self):
        self.install_dict_keys_match_guard()
        return {k.vt.as_python_constant(): v for k, v in self.items.items()}

    def python_type(self):
        return self.user_cls

    def __contains__(self, vt) -> bool:
        assert isinstance(vt, VariableTracker)
        Hashable = ConstDictVariable._HashableTracker
        return (
            is_hashable(vt)
            and Hashable(vt) in self.items
            and not isinstance(self.items[Hashable(vt)], variables.DeletedVariable)
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

    def is_new_item(self, value, other):
        # compare the id of the realized values if both values are not lazy VTs
        if value and value.is_realized() and other.is_realized():
            return id(value.realize()) != id(other.realize())
        return id(value) != id(other)

    def reconstruct_kvs_into_new_dict(self, codegen):
        # Build a dictionary that contains the keys and values.
        num_args = 0
        for key, value in self.items.items():
            # We can safely call realize() here as it won't introduce any new guards
            item = self.original_items.get(key.vt)
            if self.is_new_item(item, value) or self.should_reconstruct_all:
                codegen(key.vt)
                codegen(value)
                num_args += 1
        codegen.append_output(create_instruction("BUILD_MAP", arg=num_args))

    def reconstruct(self, codegen: "PyCodegen"):
        if self.user_cls is collections.OrderedDict:
            # emit `OrderedDict(constructed_dict)`
            codegen.add_push_null(
                lambda: codegen.extend_output(
                    [
                        codegen.create_load_python_module(collections),
                        codegen.create_load_attr("OrderedDict"),
                    ]
                )
            )
            self.reconstruct_kvs_into_new_dict(codegen)
            codegen.extend_output(create_call_function(1, False))
        else:
            self.reconstruct_kvs_into_new_dict(codegen)

    def getitem_const_raise_exception_if_absent(
        self, tx: "InstructionTranslator", arg: VariableTracker
    ):
        key = ConstDictVariable._HashableTracker(arg)
        if key not in self.items:
            raise_observed_exception(KeyError, tx)
        return self.items[key]

    def getitem_const(self, tx: "InstructionTranslator", arg: VariableTracker):
        key = ConstDictVariable._HashableTracker(arg)
        if key not in self.items:
            msg = f"Dictionary key {arg.value} not found during tracing"
            unimplemented_v2(
                gb_type="key not found in dict",
                context=f"Key {arg.value}",
                explanation=msg,
                hints=[
                    "Check if the key exists in the dictionary before accessing it.",
                    *graph_break_hints.USER_ERROR,
                ],
            )
        return self.items[key]

    def maybe_getitem_const(self, arg: VariableTracker):
        key = ConstDictVariable._HashableTracker(arg)
        if key not in self.items:
            return None
        return self.items[key]

    def realize_key_vt(self, arg: VariableTracker):
        # Realize the LazyVT on a particular index
        assert arg in self
        key = ConstDictVariable._HashableTracker(arg)
        index = tuple(self.items.keys()).index(key)
        original_key_vt = tuple(self.original_items.keys())[index]
        if isinstance(original_key_vt, variables.LazyVariableTracker):
            original_key_vt.realize()

    def install_dict_keys_match_guard(self):
        if self.source:
            install_guard(self.make_guard(GuardBuilder.DICT_KEYS_MATCH))

    def install_dict_contains_guard(self, tx, args):
        # Key guarding - These are the cases to consider
        # 1) The dict has been mutated. In this case, we would have already
        # inserted a DICT_KEYS_MATCH guard, so we can skip.
        #
        # 2) args[0].source is None. This happens for const keys. Here, we
        # have to insert the DICT_CONTAINS guard.
        #
        # 3) args[0].source is not None. This can happen for non-const VTs.
        #   3a) contains=True. In this case, we can access the lazyVT from
        #   original_items and selectively realize it.
        #   3b) contains=False. There is no easy way to selectively apply this
        #   DICT_NOT_CONTAINS guard because our guard are represented via trees.
        #   Be conservative and add DICT_KEYS_MATCH guard.
        from . import ConstantVariable

        if not self.source:
            return

        if tx.output.side_effects.is_modified(self):
            return

        contains = args[0] in self
        if args[0].source is None and isinstance(args[0], ConstantVariable):
            install_guard(
                self.make_guard(
                    functools.partial(
                        type(self).CONTAINS_GUARD,
                        key=args[0].value,
                        invert=not contains,
                    )
                )
            )
        elif args[0].source:
            if contains:
                self.realize_key_vt(args[0])
            else:
                self.install_dict_keys_match_guard()

    def call_method(
        self,
        tx,
        name,
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        # NB - Both key and value are LazyVariableTrackers in the beginning. So,
        # we have to insert guards when a dict method is accessed. For this to
        # be simple, we are conservative and overguard. We skip guard only for
        # get/__getitem__ because the key guard will be inserted by the
        # corresponding value VT. For __contains__, we add a DICT_CONTAINS
        # guard. But for all the other methods, we insert the DICT_KEYS_MATCH
        # guard to be conservative.
        from . import BuiltinVariable, ConstantVariable

        Hashable = ConstDictVariable._HashableTracker

        arg_hashable = args and is_hashable(args[0])

        if name == "__init__":
            temp_dict_vt = variables.BuiltinVariable(dict).call_dict(
                tx, *args, **kwargs
            )
            tx.output.side_effects.mutation(self)
            self.items.update(temp_dict_vt.items)
            return ConstantVariable.create(None)
        elif name == "__getitem__":
            # Key guarding - Nothing to do. LazyVT for value will take care.
            if len(args) != 1:
                raise_args_mismatch(tx, name)
            return self.getitem_const_raise_exception_if_absent(tx, args[0])
        elif name == "items":
            if args or kwargs:
                raise_args_mismatch(tx, name)
            self.install_dict_keys_match_guard()
            if self.source:
                tx.output.guard_on_key_order.add(self.source)
            return DictItemsVariable(self)
        elif name == "keys":
            if len(args):
                raise_args_mismatch(tx, name)
            self.install_dict_keys_match_guard()
            if self.source:
                tx.output.guard_on_key_order.add(self.source)
            return DictKeysVariable(self)
        elif name == "values":
            if args or kwargs:
                raise_args_mismatch(tx, name)
            self.install_dict_keys_match_guard()
            if self.source:
                tx.output.guard_on_key_order.add(self.source)
            if args or kwargs:
                raise_observed_exception(TypeError, tx)
            return DictValuesVariable(self)
        elif name == "copy":
            self.install_dict_keys_match_guard()
            if args or kwargs:
                raise_args_mismatch(tx, name)
            return self.clone(
                items=self.items.copy(), mutation_type=ValueMutationNew(), source=None
            )
        elif name == "__len__":
            if args or kwargs:
                raise_args_mismatch(tx, name)
            self.install_dict_keys_match_guard()
            return ConstantVariable.create(len(self.items))
        elif name == "__setitem__" and self.is_mutable():
            if not arg_hashable:
                raise_unhashable(args[0])

            self.install_dict_keys_match_guard()
            assert not kwargs and len(args) == 2
            tx.output.side_effects.mutation(self)
            self.items[Hashable(args[0])] = args[1]
            return ConstantVariable.create(None)
        elif name == "__delitem__" and arg_hashable and self.is_mutable():
            self.install_dict_keys_match_guard()
            self.should_reconstruct_all = True
            tx.output.side_effects.mutation(self)
            self.items.__delitem__(Hashable(args[0]))
            return ConstantVariable.create(None)
        elif name == "get":
            if len(args) not in (1, 2):
                raise_args_mismatch(tx, name)

            if not arg_hashable:
                raise_unhashable(args[0])

            if args[0] not in self:
                self.install_dict_contains_guard(tx, args)
                if len(args) == 1:
                    # if default is not given, return None
                    return ConstantVariable.create(None)
                return args[1]
            # Key guarding - Nothing to do.
            return self.getitem_const(tx, args[0])
        elif name == "pop" and self.is_mutable():
            if len(args) not in (1, 2):
                raise_args_mismatch(tx, name)

            if not arg_hashable:
                raise_unhashable(args[0])

            if args[0] not in self:
                # missing item, return the default value. Install no DICT_CONTAINS guard.
                self.install_dict_contains_guard(tx, args)
                if len(args) == 1:
                    # if default is not given, raise KeyError
                    raise_observed_exception(KeyError, tx)
                return args[1]

            self.should_reconstruct_all = True
            tx.output.side_effects.mutation(self)
            return self.items.pop(Hashable(args[0]))
        elif name == "popitem" and self.is_mutable():
            if (
                issubclass(self.user_cls, dict)
                and not issubclass(self.user_cls, collections.OrderedDict)
                and len(args)
            ):
                raise_args_mismatch(tx, name)

            if not self.items:
                msg = ConstantVariable.create("popitem(): dictionary is empty")
                raise_observed_exception(KeyError, tx, args=[msg])

            if self.user_cls is collections.OrderedDict and (
                len(args) == 1 or "last" in kwargs
            ):
                if len(args) == 1 and isinstance(args[0], ConstantVariable):
                    last = args[0].value
                elif (v := kwargs.get("last")) and isinstance(v, ConstantVariable):
                    last = v.value
                else:
                    raise_args_mismatch(tx, name)
                k, v = self.items.popitem(last=last)
            else:
                k, v = self.items.popitem()

            self.should_reconstruct_all = True
            tx.output.side_effects.mutation(self)

            return variables.TupleVariable([k.vt, v])
        elif name == "clear":
            if args or kwargs:
                raise_args_mismatch(tx, name)
            self.should_reconstruct_all = True
            tx.output.side_effects.mutation(self)
            self.items.clear()
            return ConstantVariable.create(None)
        elif name == "update" and self.is_mutable():
            # In general, this call looks like `a.update(b, x=1, y=2, ...)`.
            # Either `b` or the kwargs is omittable, but not both.
            self.install_dict_keys_match_guard()
            has_arg = len(args) == 1
            has_kwargs = len(kwargs) > 0
            if has_arg or has_kwargs:
                tx.output.side_effects.mutation(self)
                if has_arg:
                    if isinstance(args[0], ConstDictVariable):
                        # NB - Guard on all the keys of the other dict to ensure
                        # correctness.
                        args[0].install_dict_keys_match_guard()
                        dict_vt = args[0]
                    else:
                        dict_vt = BuiltinVariable.call_custom_dict(tx, dict, args[0])
                    self.items.update(dict_vt.items)
                if has_kwargs:
                    # Handle kwargs
                    kwargs = {
                        Hashable(ConstantVariable.create(k)): v
                        for k, v in kwargs.items()
                    }
                    self.items.update(kwargs)
                return ConstantVariable.create(None)
            else:
                return super().call_method(tx, name, args, kwargs)
        elif name == "__contains__":
            if not len(args):
                raise_args_mismatch(tx, name)

            if not arg_hashable:
                raise_unhashable(args[0])

            self.install_dict_contains_guard(tx, args)
            contains = args[0] in self
            return ConstantVariable.create(contains)
        elif name == "setdefault" and self.is_mutable():
            if len(args) not in (1, 2):
                raise_args_mismatch(tx, name)

            if not arg_hashable:
                raise_unhashable(args[0])

            self.install_dict_keys_match_guard()
            assert not kwargs
            assert len(args) <= 2
            value = self.maybe_getitem_const(args[0])
            if value is not None:
                return value
            else:
                if len(args) == 1:
                    x = ConstantVariable.create(None)
                else:
                    x = args[1]
                tx.output.side_effects.mutation(self)
                self.items[Hashable(args[0])] = x
                return x
        elif name == "move_to_end":
            self.install_dict_keys_match_guard()
            tx.output.side_effects.mutation(self)
            if args[0] not in self:
                raise_observed_exception(KeyError, tx)

            last = True
            if len(args) == 2 and isinstance(args[1], ConstantVariable):
                last = args[1].value

            if (
                kwargs
                and "last" in kwargs
                and isinstance(kwargs["last"], ConstantVariable)
            ):
                last = kwargs.get("last").value

            key = Hashable(args[0])
            self.items.move_to_end(key, last=last)
            return ConstantVariable.create(None)
        elif name == "__eq__" and istype(
            self, ConstDictVariable
        ):  # don't let Set use this function
            if len(args) != 1:
                raise_args_mismatch(tx, name)

            return variables.UserFunctionVariable(polyfills.dict___eq__).call_function(
                tx, [self, args[0]], {}
            )
        elif name == "__ne__":
            return ConstantVariable.create(
                not self.call_method(tx, "__eq__", args, kwargs).value
            )
        elif name == "__or__":
            assert len(args) == 1
            other = args[0]

            # Method resolution for binops works as follow (using __or__ as example):
            # (1) dict.__or__(dict) => dict
            # (2) dict.__or__(subclass): return NotImplemented
            # (3) Check if subclass implements __ror__ => forward the call
            # to subclass.__ror__(dict)

            # Let's not forward the call to __ror__ yet because __ror__ can be
            # implemented in C (i.e. OrderedDict subclass) which Dynamo cannot
            # trace
            # if istype(other, variables.UserDefinedDictVariable):
            #     if other.call_obj_hasattr(tx, "__ror__").value:
            #         return other.call_method(tx, "__ror__", [self], kwargs)

            # The three dict types Dynamo can handle are dict, OrderedDict and
            # defaultdict.

            # TODO(guilhermeleobas): this check should be on builtin.py::call_or_
            if not istype(
                other, (ConstDictVariable, variables.UserDefinedDictVariable)
            ):
                msg = (
                    f"unsupported operand type(s) for |: '{self.python_type().__name__}'"
                    f"and '{other.python_type().__name__}'"
                )
                raise_observed_exception(TypeError, tx, args=[msg])

            # OrderedDict overloads __ror__
            ts = {self.user_cls, other.user_cls}
            user_cls = (
                collections.OrderedDict
                if any(issubclass(t, collections.OrderedDict) for t in ts)
                else dict
            )

            self.install_dict_keys_match_guard()
            new_dict_vt = self.clone(
                items=self.items.copy(),
                mutation_type=ValueMutationNew(),
                source=None,
                user_cls=user_cls,
            )

            # NB - Guard on all the keys of the other dict to ensure
            # correctness.
            args[0].install_dict_keys_match_guard()
            new_dict_vt.items.update(args[0].items)
            return new_dict_vt
        elif name == "__ior__":
            self.call_method(tx, "update", args, kwargs)
            return self
        else:
            return super().call_method(tx, name, args, kwargs)

    def unpack_var_sequence(self, tx):
        self.install_dict_keys_match_guard()
        return [x.vt for x in self.items.keys()]

    def call_obj_hasattr(self, tx, name):
        # dict not allow setting arbitrary attributes. To check for hasattr, we can just check the __dict__ of the dict.
        # OrderedDict though requires side effects tracking because it supports arbitrary setattr.
        if self.user_cls is dict:
            if name in self.user_cls.__dict__:
                return ConstantVariable.create(True)
            return ConstantVariable.create(False)

        msg = f"hasattr on {self.user_cls} is not supported"
        unimplemented_v2(
            gb_type="unsupported hasattr operation",
            context=f"Class {self.user_cls}",
            explanation=msg,
            hints=[
                "Consider using a regular dictionary instead",
                *graph_break_hints.SUPPORTABLE,
            ],
        )

    def clone(self, **kwargs):
        self.install_dict_keys_match_guard()
        return super().clone(**kwargs)


class MappingProxyVariable(VariableTracker):
    # proxies to the original dict_vt
    def __init__(self, dv_dict: ConstDictVariable, **kwargs) -> None:
        super().__init__(**kwargs)
        assert isinstance(dv_dict, ConstDictVariable)
        self.dv_dict = dv_dict

    def python_type(self):
        return types.MappingProxyType

    def unpack_var_sequence(self, tx):
        return self.dv_dict.unpack_var_sequence(tx)

    def reconstruct(self, codegen: "PyCodegen"):
        # load types.MappingProxyType
        if self.source:
            msg = (
                f"Preexisting MappingProxyVariable (source: {self.source}) cannot be reconstructed "
                "because the connection to the original dict will be lost."
            )
            unimplemented_v2(
                gb_type="mapping proxy cannot be reconstructed",
                context=f"Source: {self.source}",
                explanation=msg,
                hints=[
                    "Use a mapping proxy constructed in the same `torch.compile` region.",
                    *graph_break_hints.SUPPORTABLE,
                ],
            )
        codegen.add_push_null(
            lambda: codegen.extend_output(
                [
                    codegen.create_load_python_module(types),
                    codegen.create_load_attr("MappingProxyType"),
                ]
            )
        )
        codegen(self.dv_dict)
        codegen.extend_output(create_call_function(1, False))

    def call_method(
        self,
        tx,
        name,
        args: list["VariableTracker"],
        kwargs: dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        if self.source and tx.output.side_effects.has_existing_dict_mutation():
            msg = (
                "A dict has been modified while we have an existing mappingproxy object. "
                "A mapping proxy object, as the name suggest, proxies a mapping "
                "object (usually a dict). If the original dict object mutates, it "
                "is reflected in the proxy object as well. For an existing proxy "
                "object, we do not know the original dict it points to. Therefore, "
                "for correctness we graph break when there is dict mutation and we "
                "are trying to access a proxy object."
            )

            unimplemented_v2(
                gb_type="mapping proxy affected by dictionary mutation",
                context=f"Source: {self.source}, Dict mutation detected",
                explanation=msg,
                hints=[
                    "Avoid modifying dictionaries that might be referenced by mapping proxy objects",
                    "Or avoid using the mapping proxy objects after modifying its underlying dictionary",
                ],
            )
        return self.dv_dict.call_method(tx, name, args, kwargs)


class NNModuleHooksDictVariable(ConstDictVariable):
    # Special class to avoid adding any guards on the nn module hook ids.
    def install_dict_keys_match_guard(self):
        pass

    def install_dict_contains_guard(self, tx, args):
        pass


class DefaultDictVariable(ConstDictVariable):
    def __init__(self, items, user_cls, default_factory=None, **kwargs) -> None:
        super().__init__(items, user_cls, **kwargs)
        assert user_cls is collections.defaultdict
        self.default_factory = default_factory

    def is_python_constant(self):
        # Return false for unsupported defaults. This ensures that a bad handler
        # path is not taken in BuiltinVariable for getitem.
        if self.default_factory not in [list, tuple, dict] and not self.items:
            return False
        return super().is_python_constant()

    def debug_repr(self):
        return (
            f"defaultdict({self.default_factory.debug_repr()}, {super().debug_repr()})"
        )

    @staticmethod
    def is_supported_arg(arg):
        if isinstance(arg, variables.BuiltinVariable):
            return arg.fn in (list, tuple, dict, set)
        else:
            return isinstance(arg, variables.functions.BaseUserFunctionVariable)

    def call_method(
        self,
        tx,
        name,
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if name == "__getitem__":
            assert len(args) == 1

            if args[0] in self:
                return self.getitem_const(tx, args[0])
            else:
                if self.default_factory is None:
                    raise KeyError(f"{args[0]}")
                else:
                    default_var = self.default_factory.call_function(tx, [], {})
                    super().call_method(
                        tx, "__setitem__", (args[0], default_var), kwargs
                    )
                    return default_var
        else:
            return super().call_method(tx, name, args, kwargs)

    def reconstruct(self, codegen):
        # emit `defaultdict(default_factory, new_dict)`
        codegen.add_push_null(
            lambda: codegen.extend_output(
                [
                    codegen.create_load_python_module(collections),
                    codegen.create_load_attr("defaultdict"),
                ]
            )
        )
        codegen(self.default_factory)
        self.reconstruct_kvs_into_new_dict(codegen)
        codegen.extend_output(create_call_function(2, False))


# TODO: Implementing this via inheritance rather than composition is a
# footgun, because self method calls in dict will route back to the set
# implementation, which is almost assuredly wrong
class SetVariable(ConstDictVariable):
    """We model a sets as dictionary with None values"""

    CONTAINS_GUARD = GuardBuilder.SET_CONTAINS

    def __init__(
        self,
        items: list[VariableTracker],
        **kwargs,
    ) -> None:
        items = dict.fromkeys(items, SetVariable._default_value())
        super().__init__(items, **kwargs)

    def debug_repr(self):
        if not self.items:
            return "set()"
        else:
            return "{" + ",".join(k.vt.debug_repr() for k in self.items.keys()) + "}"

    @property
    def set_items(self):
        return set(self.items.keys())

    @staticmethod
    def _default_value():
        # Variable to fill in he keys of the dictionary
        return ConstantVariable.create(None)

    def as_proxy(self):
        return {k.vt.as_proxy() for k in self.set_items}

    def python_type(self):
        return set

    def as_python_constant(self):
        return {k.vt.as_python_constant() for k in self.set_items}

    def reconstruct(self, codegen: "PyCodegen"):
        codegen.foreach([x.vt for x in self.set_items])
        codegen.append_output(create_instruction("BUILD_SET", arg=len(self.set_items)))

    def _fast_set_method(self, tx, fn, args, kwargs):
        try:
            res = fn(
                *[x.as_python_constant() for x in [self, *args]],
                **{k: v.as_python_constant() for k, v in kwargs.items()},
            )
        except Exception as exc:
            raise_observed_exception(
                type(exc), tx, args=list(map(ConstantVariable.create, exc.args))
            )
        return VariableTracker.build(tx, res)

    def call_method(
        self,
        tx,
        name,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> "VariableTracker":
        # We forward the calls to the dictionary model
        from ..utils import check_constant_args

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

        if name == "__init__":
            temp_set_vt = variables.BuiltinVariable(set).call_set(tx, *args, *kwargs)
            tx.output.side_effects.mutation(self)
            self.items.clear()
            self.items.update(temp_set_vt.items)
            return ConstantVariable.create(None)
        elif name == "add":
            assert not kwargs
            if len(args) != 1:
                raise_args_mismatch(tx, name)
            name = "__setitem__"
            args = (args[0], SetVariable._default_value())
        elif name == "pop":
            assert not kwargs
            assert not args
            # Choose an item at random and pop it via the Dict.pop method
            try:
                result = self.set_items.pop().vt
            except KeyError as e:
                raise_observed_exception(
                    KeyError, tx, args=list(map(ConstantVariable.create, e.args))
                )
            super().call_method(tx, name, (result,), kwargs)
            return result
        elif name == "isdisjoint":
            if len(args) != 1:
                raise_args_mismatch(tx, name)
            assert not kwargs
            return variables.UserFunctionVariable(
                polyfills.set_isdisjoint
            ).call_function(tx, [self, args[0]], {})
        elif name == "intersection":
            assert not kwargs
            return variables.UserFunctionVariable(
                polyfills.set_intersection
            ).call_function(tx, [self, *args], {})
        elif name == "intersection_update":
            assert not kwargs
            return variables.UserFunctionVariable(
                polyfills.set_intersection_update
            ).call_function(tx, [self, *args], {})
        elif name == "union":
            assert not kwargs
            return variables.UserFunctionVariable(polyfills.set_union).call_function(
                tx, [self, *args], {}
            )
        elif name == "difference":
            assert not kwargs
            return variables.UserFunctionVariable(
                polyfills.set_difference
            ).call_function(tx, [self, *args], {})
        elif name == "difference_update":
            assert not kwargs
            return variables.UserFunctionVariable(
                polyfills.set_difference_update
            ).call_function(tx, [self, *args], {})
        elif name == "symmetric_difference":
            if len(args) != 1:
                raise_args_mismatch(tx, name)
            assert not kwargs
            return variables.UserFunctionVariable(
                polyfills.set_symmetric_difference
            ).call_function(tx, [self, *args], {})
        elif name == "symmetric_difference_update":
            if len(args) != 1:
                raise_args_mismatch(tx, name)
            assert not kwargs
            return variables.UserFunctionVariable(
                polyfills.set_symmetric_difference_update
            ).call_function(tx, [self, *args], {})
        elif name == "update" and self.is_mutable():
            assert not kwargs
            return variables.UserFunctionVariable(polyfills.set_update).call_function(
                tx, [self, *args], {}
            )
        elif name == "remove":
            assert not kwargs
            assert len(args) == 1
            if args[0] not in self:
                raise_observed_exception(KeyError, tx, args=args)
            return super().call_method(tx, "pop", args, kwargs)
        elif name == "discard":
            assert not kwargs
            assert len(args) == 1
            if args[0] in self:
                return super().call_method(tx, "pop", args, kwargs)
            else:
                return ConstantVariable.create(value=None)
        elif name in ("issubset", "issuperset"):
            if len(args) != 1:
                raise_args_mismatch(tx, name)

            op = {
                "issubset": operator.le,
                "issuperset": operator.ge,
            }
            other = args[0].realize()
            if not istype(other, SetVariable):
                other = variables.BuiltinVariable(set).call_function(tx, [other], {})
            return variables.BuiltinVariable(op.get(name)).call_function(
                tx, [self, other], {}
            )
        elif name in ("__and__", "__or__", "__xor__", "__sub__"):
            m = {
                "__and__": "intersection",
                "__or__": "union",
                "__xor__": "symmetric_difference",
                "__sub__": "difference",
            }.get(name)
            if not isinstance(args[0], (SetVariable, variables.UserDefinedSetVariable)):
                msg = ConstantVariable.create(
                    f"unsupported operand type(s) for {name}: '{self.python_type_name()}' and '{args[0].python_type_name()}'"
                )
                raise_observed_exception(TypeError, tx, args=[msg])
            return self.call_method(tx, m, args, kwargs)
        elif name in ("__iand__", "__ior__", "__ixor__", "__isub__"):
            if not isinstance(args[0], (SetVariable, variables.UserDefinedSetVariable)):
                msg = ConstantVariable.create(
                    f"unsupported operand type(s) for {name}: '{self.python_type_name()}' and '{args[0].python_type_name()}'"
                )
                raise_observed_exception(TypeError, tx, args=[msg])
            m = {
                "__iand__": "intersection_update",
                "__ior__": "update",
                "__ixor__": "symmetric_difference_update",
                "__isub__": "difference_update",
            }.get(name)
            self.call_method(tx, m, args, kwargs)
            return self
        elif name == "__eq__":
            if not isinstance(args[0], (SetVariable, variables.UserDefinedSetVariable)):
                return ConstantVariable.create(False)
            r = self.call_method(tx, "symmetric_difference", args, kwargs)
            return ConstantVariable.create(len(r.set_items) == 0)
        elif name in cmp_name_to_op_mapping:
            if not isinstance(args[0], (SetVariable, variables.UserDefinedSetVariable)):
                return ConstantVariable.create(NotImplemented)
            return ConstantVariable.create(
                cmp_name_to_op_mapping[name](self.set_items, args[0].set_items)
            )
        return super().call_method(tx, name, args, kwargs)

    def getitem_const(self, tx: "InstructionTranslator", arg: VariableTracker):
        raise RuntimeError("Illegal to getitem on a set")

    def install_dict_keys_match_guard(self):
        # Already EQUALS_MATCH guarded
        pass

    def install_dict_contains_guard(self, tx, args):
        super().install_dict_contains_guard(tx, args)


class FrozensetVariable(SetVariable):
    def __init__(
        self,
        items: list[VariableTracker],
        **kwargs,
    ) -> None:
        super().__init__(items, **kwargs)

    def debug_repr(self):
        if not self.items:
            return "frozenset()"
        else:
            return "{" + ",".join(k.vt.debug_repr() for k in self.items.keys()) + "}"

    @property
    def set_items(self):
        return self.items.keys()

    def python_type(self):
        return frozenset

    def as_python_constant(self):
        return frozenset({k.vt.as_python_constant() for k in self.set_items})

    def reconstruct(self, codegen: "PyCodegen"):
        codegen.foreach([x.vt for x in self.set_items])
        codegen.add_push_null(
            lambda: codegen.extend_output(
                [
                    codegen.create_load_global("frozenset"),
                ]
            )
        )
        codegen.extend_output(create_call_function(0, False))

    def call_method(
        self,
        tx,
        name,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> "VariableTracker":
        if name in ["add", "pop", "update", "remove", "discard", "clear"]:
            raise RuntimeError(f"Illegal call_method {name} on a frozenset")
        elif name == "__init__":
            # frozenset is immutable. Calling __init__ again shouldn't have any effect
            # In[1]: s = frozenset([1, 2])
            #
            # In[2]: s.__init__([3, 4])
            #
            # In[3]: s
            # frozenset({1, 2})
            return ConstantVariable.create(None)
        elif name in (
            "copy",
            "difference",
            "intersection",
            "symmetric_difference",
        ):
            r = super().call_method(tx, name, args, kwargs)
            return FrozensetVariable(r.items)
        return super().call_method(tx, name, args, kwargs)


class DictKeySetVariable(SetVariable):
    def __init__(
        self,
        items: list[VariableTracker],
        **kwargs,
    ) -> None:
        super().__init__(items, **kwargs)

    def debug_repr(self):
        if not self.items:
            return "dict_keys([])"
        else:
            return (
                "dict_keys(["
                + ",".join(k.vt.debug_repr() for k in self.items.keys())
                + "])"
            )

    def install_dict_keys_match_guard(self):
        # Already EQUALS_MATCH guarded
        pass

    def install_dict_contains_guard(self, tx, args):
        # Already EQUALS_MATCH guarded
        pass

    @property
    def set_items(self):
        return self.items

    def python_type(self):
        return dict_keys

    def as_python_constant(self):
        return dict.fromkeys(
            {k.vt.as_python_constant() for k in self.set_items}, None
        ).keys()

    def call_method(
        self,
        tx,
        name,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> "VariableTracker":
        if name in ["add", "pop", "update", "remove", "discard", "clear"]:
            raise RuntimeError(f"Illegal call_method {name} on a dict_keys")
        return super().call_method(tx, name, args, kwargs)


class DictViewVariable(VariableTracker):
    """
    Models _PyDictViewObject

    This is an "abstract" class. Subclasses will override kv and the items method
    """

    kv: Optional[str] = None

    def __init__(self, dv_dict: ConstDictVariable, **kwargs) -> None:
        super().__init__(**kwargs)
        assert self.kv in ("keys", "values", "items")
        assert isinstance(dv_dict, ConstDictVariable)
        self.dv_dict = dv_dict

    @property
    def view_items(self):
        return getattr(self.dv_dict.items, self.kv)()

    @property
    def view_items_vt(self):
        # Returns an iterable of the unpacked items
        # Implement in the subclasses
        raise NotImplementedError

    def unpack_var_sequence(self, tx):
        return self.view_items_vt

    def reconstruct(self, codegen: "PyCodegen"):
        codegen(self.dv_dict)
        codegen.load_method(self.kv)
        codegen.call_method(0)

    def call_obj_hasattr(self, tx, name):
        if name in self.python_type().__dict__:
            return ConstantVariable.create(True)
        return ConstantVariable.create(False)

    def call_method(
        self,
        tx,
        name,
        args: list["VariableTracker"],
        kwargs: dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        if name == "__len__":
            return self.dv_dict.call_method(tx, name, args, kwargs)
        return super().call_method(tx, name, args, kwargs)


class DictKeysVariable(DictViewVariable):
    kv = "keys"

    @property
    def set_items(self):
        return set(self.view_items)

    @property
    def view_items_vt(self):
        # Returns an iterable of the unpacked items
        return [x.vt for x in self.view_items]

    def python_type(self):
        return dict_keys

    def call_method(
        self,
        tx,
        name,
        args: list["VariableTracker"],
        kwargs: dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        if name == "__contains__":
            return self.dv_dict.call_method(tx, name, args, kwargs)
        elif name in (
            "__and__",
            "__iand__",
            "__or__",
            "__ior__",
            "__sub__",
            "__isub__",
            "__xor__",
            "__ixor__",
        ):
            # These methods always returns a set
            m = getattr(self.set_items, name)
            r = m(args[0].set_items)
            return SetVariable(r)
        if name in cmp_name_to_op_mapping:
            if not isinstance(args[0], (SetVariable, DictKeysVariable)):
                return ConstantVariable.create(NotImplemented)
            return ConstantVariable.create(
                cmp_name_to_op_mapping[name](self.set_items, args[0].set_items)
            )
        return super().call_method(tx, name, args, kwargs)


class DictValuesVariable(DictViewVariable):
    # DictValuesVariable is an iterable but cannot be compared.
    kv = "values"

    @property
    def view_items_vt(self):
        return list(self.view_items)

    def python_type(self):
        return dict_values


class DictItemsVariable(DictViewVariable):
    kv = "items"

    @property
    def view_items_vt(self):
        # Returns an iterable of the unpacked items
        return [variables.TupleVariable([k.vt, v]) for k, v in self.view_items]

    def python_type(self):
        return dict_items
