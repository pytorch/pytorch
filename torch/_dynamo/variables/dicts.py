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
import types
from typing import Optional, TYPE_CHECKING

from torch._subclasses.fake_tensor import is_fake

from .. import polyfills, variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import raise_observed_exception, unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import is_from_local_source
from ..utils import cmp_name_to_op_mapping, dict_keys, dict_values, specialize_symnode
from .base import ValueMutationNew, VariableTracker
from .constant import ConstantVariable


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator


# [Adding a new supported class within the keys of ConstDictVarialble]
# - Add its tracker type to is_hashable
# - (perhaps) Define how it is compared in _HashableTracker._eq_impl


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
    else:
        return isinstance(
            x,
            (
                variables.BuiltinVariable,
                variables.SymNodeVariable,
                variables.ConstantVariable,
                variables.EnumVariable,
                variables.user_defined.UserDefinedClassVariable,
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
            ),
        )


class ConstDictVariable(VariableTracker):
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
            # TODO Temorarily remove to figure out what keys are we breaking on
            # and add proper support for them
            if not is_hashable(vt):
                unimplemented(f"Dict key of type {type(vt)}. Key: {vt}")
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
            assert isinstance(other, Hashable) or ConstantVariable.is_literal(
                other
            ), type(other)
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

        self.items = {make_hashable(x): v for x, v in items.items()}
        # need to reconstruct everything if the dictionary is an intermediate value
        # or if a pop/delitem was executed
        self.should_reconstruct_all = not is_from_local_source(self.source)
        self.original_items = items.copy()
        self.user_cls = user_cls

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

    def len(self):
        return len(
            [
                x
                for x in self.items.values()
                if not isinstance(x, variables.DeletedVariable)
            ]
        )

    def has_new_items(self):
        if self.should_reconstruct_all:
            return True
        return any(
            self.is_new_item(self.original_items.get(key.vt), value)
            for key, value in self.items.items()
        )

    def is_new_item(self, value, other):
        # compare the id of the realized values if both values are not lazy VTs
        if value and value.is_realized() and other.is_realized():
            return id(value.realize()) != id(other.realize())
        return id(value) != id(other)

    def reconstruct(self, codegen):
        # instructions to load collections.OrderedDict if necessary
        if self.user_cls is collections.OrderedDict:
            codegen.add_push_null(
                lambda: codegen.extend_output(
                    [
                        codegen.create_load_python_module(collections),
                        codegen.create_load_attr("OrderedDict"),
                    ]
                )
            )
        # instructions to build the dict keys and values
        num_args = 0
        for key, value in self.items.items():
            # We can safely call realize() here as it won't introduce any new guards
            item = self.original_items.get(key.vt)
            if self.is_new_item(item, value) or self.should_reconstruct_all:
                codegen(key.vt)
                codegen(value)
                num_args += 1

        # BUILD_MAP and calling collections.OrderedDict if necessary
        if self.user_cls is collections.OrderedDict:
            codegen.extend_output(
                [
                    create_instruction("BUILD_MAP", arg=num_args),
                    *create_call_function(1, False),
                ]
            )
        # BUILD_MAP only if user_cls is dict
        else:
            codegen.append_output(create_instruction("BUILD_MAP", arg=num_args))

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
            unimplemented(f"dict KeyError: {arg.value}")
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
                        GuardBuilder.DICT_CONTAINS,
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
        from . import BuiltinVariable, ConstantVariable, TupleVariable

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
            assert len(args) == 1
            return self.getitem_const_raise_exception_if_absent(tx, args[0])
        elif name == "items":
            assert not (args or kwargs)
            self.install_dict_keys_match_guard()
            if self.source:
                tx.output.guard_on_key_order.add(self.source.name())
            return TupleVariable(
                [TupleVariable([k.vt, v]) for k, v in self.items.items()]
            )
        elif name == "keys":
            self.install_dict_keys_match_guard()
            if self.source:
                tx.output.guard_on_key_order.add(self.source.name())
            assert not (args or kwargs)
            return DictKeysVariable(self)
        elif name == "values":
            self.install_dict_keys_match_guard()
            if self.source:
                tx.output.guard_on_key_order.add(self.source.name())
            assert not (args or kwargs)
            return DictValuesVariable(self)
        elif name == "copy":
            self.install_dict_keys_match_guard()
            assert not (args or kwargs)
            return self.clone(
                items=self.items.copy(), mutation_type=ValueMutationNew(), source=None
            )
        elif name == "__len__":
            assert not (args or kwargs)
            self.install_dict_keys_match_guard()
            return ConstantVariable.create(len(self.items))
        elif name == "__setitem__" and arg_hashable and self.is_mutable():
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
        elif name in ("pop", "get") and len(args) in (1, 2) and args[0] not in self:
            # missing item, return the default value. Install no DICT_CONTAINS guard.
            self.install_dict_contains_guard(tx, args)
            if len(args) == 1:
                if name == "pop":
                    raise_observed_exception(KeyError, tx)
                return ConstantVariable(None)
            else:
                return args[1]
        elif name == "pop" and arg_hashable and self.is_mutable():
            self.should_reconstruct_all = True
            tx.output.side_effects.mutation(self)
            return self.items.pop(Hashable(args[0]))
        elif name == "clear":
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
        elif name in ("get", "__getattr__") and args[0] in self:
            # Key guarding - Nothing to do.
            return self.getitem_const(tx, args[0])
        elif name == "__contains__" and len(args) == 1:
            self.install_dict_contains_guard(tx, args)
            contains = args[0] in self
            return ConstantVariable.create(contains)
        elif name == "setdefault" and arg_hashable and self.is_mutable():
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
            assert not kwargs and len(args) == 1
            tx.output.side_effects.mutation(self)
            key = Hashable(args[0])
            val = self.items[key]
            self.items.pop(key)
            self.items[key] = val
            return ConstantVariable.create(None)
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
        unimplemented(f"hasattr on {self.user_cls} is not supported")

    def clone(self, **kwargs):
        self.install_dict_keys_match_guard()
        return super().clone(**kwargs)


class MappingProxyVariable(VariableTracker):
    # proxies to the original dict_vt
    def __init__(self, dv_dict: ConstDictVariable, **kwargs) -> None:
        super().__init__(**kwargs)
        assert isinstance(dv_dict, ConstDictVariable)
        self.dv_dict = dv_dict

    def unpack_var_sequence(self, tx):
        return self.dv_dict.unpack_var_sequence(tx)

    def reconstruct(self, codegen):
        # load types.MappingProxyType
        if self.source:
            unimplemented(
                "Can't reconstruct an existing mapping variable because"
                " the connection to the original dict will be lost"
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
            unimplemented(
                "A dict has been modified while we have an existing mappingproxy object. "
                "A mapping proxy object, as the name suggest, proxies a mapping "
                "object (usually a dict). If the original dict object mutates, it "
                "is reflected in the proxy object as well. For an existing proxy "
                "object, we do not know the original dict it points to. Therefore, "
                "for correctness we graph break when there is dict mutation and we "
                "are trying to access a proxy object."
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


# TODO: Implementing this via inheritance rather than composition is a
# footgun, because self method calls in dict will route back to the set
# implementation, which is almost assuredly wrong
class SetVariable(ConstDictVariable):
    """We model a sets as dictonary with None values"""

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
        # Variable to fill in he keys of the dictinary
        return ConstantVariable.create(None)

    def as_proxy(self):
        return {k.vt.as_proxy() for k in self.set_items}

    def python_type(self):
        return set

    def as_python_constant(self):
        return {k.vt.as_python_constant() for k in self.set_items}

    def reconstruct(self, codegen):
        codegen.foreach([x.vt for x in self.set_items])
        codegen.append_output(create_instruction("BUILD_SET", arg=len(self.set_items)))

    def call_method(
        self,
        tx,
        name,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> "VariableTracker":
        # We foward the calls to the dictionary model
        if name == "add":
            assert not kwargs
            assert len(args) == 1
            name = "__setitem__"
            args = (args[0], SetVariable._default_value())
        elif name == "pop":
            assert not kwargs
            assert not args
            # Choose an item at random and pop it via the Dict.pop method
            result = self.set_items.pop().vt
            super().call_method(tx, name, (result,), kwargs)
            return result
        elif name == "isdisjoint":
            assert not kwargs
            assert len(args) == 1
            return variables.UserFunctionVariable(
                polyfills.set_isdisjoint
            ).call_function(tx, [self, args[0]], {})
        elif name == "intersection":
            assert not kwargs
            assert len(args) == 1
            return variables.UserFunctionVariable(
                polyfills.set_intersection
            ).call_function(tx, [self, args[0]], {})
        elif name == "union":
            assert not kwargs
            assert len(args) == 1
            return variables.UserFunctionVariable(polyfills.set_union).call_function(
                tx, [self, args[0]], {}
            )
        elif name == "difference":
            assert not kwargs
            assert len(args) == 1
            return variables.UserFunctionVariable(
                polyfills.set_difference
            ).call_function(tx, [self, args[0]], {})
        elif name == "update" and len(args) == 1 and self.is_mutable():
            assert not kwargs
            assert len(args) == 1
            return variables.UserFunctionVariable(polyfills.set_update).call_function(
                tx, [self, args[0]], {}
            )
        elif name == "remove":
            assert not kwargs
            assert len(args) == 1
            if args[0] not in self:
                unimplemented("key does not exist")
            return super().call_method(tx, "pop", args, kwargs)
        elif name == "discard":
            assert not kwargs
            assert len(args) == 1
            if args[0] in self:
                return super().call_method(tx, "pop", args, kwargs)
            else:
                return ConstantVariable.create(value=None)
        return super().call_method(tx, name, args, kwargs)

    def getitem_const(self, tx: "InstructionTranslator", arg: VariableTracker):
        raise RuntimeError("Illegal to getitem on a set")

    def install_dict_keys_match_guard(self):
        # Already EQUALS_MATCH guarded
        pass

    def install_dict_contains_guard(self, tx, args):
        # Already EQUALS_MATCH guarded
        pass


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
        return {k.vt.as_python_constant() for k in self.set_items}

    def reconstruct(self, codegen):
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
        assert self.kv in ("keys", "values")
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
        def unwrap(x):
            return x.vt if self.kv == "keys" else x

        return [unwrap(x) for x in self.view_items]

    def reconstruct(self, codegen):
        codegen(self.dv_dict)
        codegen.load_method(self.kv)
        codegen.call_method(0)

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
