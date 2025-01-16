# mypy: ignore-errors

import collections
import functools
import sys
from typing import Dict, List, Optional, TYPE_CHECKING

from torch._subclasses.fake_tensor import is_fake

from .. import polyfills, variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import raise_observed_exception, unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import DictGetItemSource, is_from_local_source
from ..utils import dict_keys, dict_values, specialize_symnode
from .base import ValueMutationNew, VariableTracker
from .constant import ConstantVariable


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator


# [Adding a new supported class within the keys of ConstDictVarialble]
# - Add its tracker type to is_hashable
# - (perhaps) Define how it is compared in _HashableTracker._eq_impl


def is_hashable(x):
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
        items: Dict[VariableTracker, VariableTracker],
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

    def as_proxy(self, tx=None):
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

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
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
            assert len(args) == 1
            return self.getitem_const_raise_exception_if_absent(tx, args[0])
        elif name == "items":
            assert not (args or kwargs)
            if self.source:
                tx.output.guard_on_key_order.add(self.source.name())
            return TupleVariable(
                [TupleVariable([k.vt, v]) for k, v in self.items.items()]
            )
        elif name == "keys":
            if self.source:
                tx.output.guard_on_key_order.add(self.source.name())
            assert not (args or kwargs)
            return DictKeysVariable(self)
        elif name == "values":
            if self.source:
                tx.output.guard_on_key_order.add(self.source.name())
            assert not (args or kwargs)
            return DictValuesVariable(self)
        elif name == "copy":
            assert not (args or kwargs)
            return self.clone(
                items=self.items.copy(), mutation_type=ValueMutationNew(), source=None
            )
        elif name == "__len__":
            assert not (args or kwargs)
            return ConstantVariable.create(len(self.items))
        elif name == "__setitem__" and arg_hashable and self.is_mutable():
            assert not kwargs and len(args) == 2
            tx.output.side_effects.mutation(self)
            self.items[Hashable(args[0])] = args[1]
            return ConstantVariable.create(None)
        elif name == "__delitem__" and arg_hashable and self.is_mutable():
            self.should_reconstruct_all = True
            tx.output.side_effects.mutation(self)
            self.items.__delitem__(Hashable(args[0]))
            return ConstantVariable.create(None)
        elif name in ("pop", "get") and len(args) in (1, 2) and args[0] not in self:
            # missing item, return the default value
            if len(args) == 1:
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
            has_arg = len(args) == 1
            has_kwargs = len(kwargs) > 0
            if has_arg or has_kwargs:
                tx.output.side_effects.mutation(self)
                if has_arg:
                    if isinstance(args[0], ConstDictVariable):
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
            return self.getitem_const(tx, args[0])
        elif name == "__contains__" and len(args) == 1:
            return ConstantVariable.create(args[0] in self)
        elif name == "setdefault" and arg_hashable and self.is_mutable():
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
        else:
            return super().call_method(tx, name, args, kwargs)

    def unpack_var_sequence(self, tx):
        return [x.vt for x in self.items.keys()]

    def call_hasattr(self, tx, name):
        # dict not allow setting arbitrary attributes. To check for hasattr, we can just check the __dict__ of the dict.
        # OrderedDict though requires side effects tracking because it supports arbitrary setattr.
        if self.user_cls is dict:
            if name in self.user_cls.__dict__:
                return ConstantVariable.create(True)
            return ConstantVariable.create(False)
        unimplemented(f"hasattr on {self.user_cls} is not supported")


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
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
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
        items: List[VariableTracker],
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

    def as_proxy(self, tx=None):
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
        args: List[VariableTracker],
        kwargs: Dict[str, VariableTracker],
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


class FrozensetVariable(SetVariable):
    def __init__(
        self,
        items: List[VariableTracker],
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
        args: List[VariableTracker],
        kwargs: Dict[str, VariableTracker],
    ) -> "VariableTracker":
        if name in ["add", "pop", "update", "remove", "discard", "clear"]:
            raise RuntimeError(f"Illegal call_method {name} on a frozenset")
        return super().call_method(tx, name, args, kwargs)


class DictKeySetVariable(SetVariable):
    def __init__(
        self,
        items: List[VariableTracker],
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
        unimplemented("DictKeySetVariable.as_python_constant")

    def call_method(
        self,
        tx,
        name,
        args: List[VariableTracker],
        kwargs: Dict[str, VariableTracker],
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
        args: List["VariableTracker"],
        kwargs: Dict[str, "VariableTracker"],
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
        args: List["VariableTracker"],
        kwargs: Dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        if name == "__contains__":
            return self.dv_dict.call_method(tx, name, args, kwargs)
        return super().call_method(tx, name, args, kwargs)


class DictValuesVariable(DictViewVariable):
    # DictValuesVariable is an iterable but cannot be compared.
    kv = "values"

    @property
    def view_items_vt(self):
        return list(self.view_items)

    def python_type(self):
        return dict_values


class PythonSysModulesVariable(VariableTracker):
    """Special case for sys.modules.

    Without this we will guard on the exact set of modules imported in the
    lifetime of the python program.
    """

    def python_type(self):
        return dict

    def reconstruct(self, codegen):
        codegen.add_push_null(
            lambda: codegen.extend_output(
                [
                    codegen.create_load_python_module(sys),
                    codegen.create_load_attr("modules"),
                ]
            )
        )

    def call_method(
        self,
        tx: "InstructionTranslator",
        name,
        args: List[VariableTracker],
        kwargs: Dict[str, VariableTracker],
    ):
        if name == "__getitem__":
            return self.call_getitem(tx, *args, **kwargs)
        elif name == "get":
            return self.call_get(tx, *args, **kwargs)
        elif name == "__contains__":
            return self.call_contains(tx, *args, **kwargs)
        unimplemented(f"sys.modules.{name}(*{args}, **{kwargs})")

    def _contains_helper(self, tx: "InstructionTranslator", key: VariableTracker):
        k = key.as_python_constant()
        has_key = k in sys.modules
        install_guard(
            self.make_guard(
                functools.partial(GuardBuilder.DICT_CONTAINS, key=k, invert=not has_key)
            )
        )
        return k, has_key

    def call_contains(self, tx: "InstructionTranslator", key: VariableTracker):
        _k, has_key = self._contains_helper(tx, key)
        return ConstantVariable.create(value=has_key)

    def call_get(
        self,
        tx: "InstructionTranslator",
        key: VariableTracker,
        default: Optional[VariableTracker] = None,
    ):
        k, has_key = self._contains_helper(tx, key)

        if has_key:
            source = self.source and DictGetItemSource(self.source, k)
            return VariableTracker.build(tx, sys.modules[k], source)

        if default is not None:
            return default

        return ConstantVariable.create(value=None)

    def call_getitem(self, tx: "InstructionTranslator", key: VariableTracker):
        k, _has_key = self._contains_helper(tx, key)
        source = self.source and DictGetItemSource(self.source, k)
        return VariableTracker.build(tx, sys.modules[k], source)
