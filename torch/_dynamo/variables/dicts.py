import collections
import dataclasses
import enum
import functools
import inspect
import sys
from typing import Any, Dict, List, Optional

import torch

from torch._subclasses.fake_tensor import is_fake

from .. import variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..eval_frame import skip_code

from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard, make_dupe_guard
from ..source import AttrSource, GetItemSource
from ..utils import istype, iter_contains, specialize_symnode
from .base import MutableLocal, VariableTracker
from .constant import ConstantVariable

# Note: [Adding a new supported class the keys of ConstDictVarialble]
# You'll need to add it to:
# - `is_hashable_python_var` in this file
# - `is_hashable` in this file
# - `const_repr` in util.py, and perhaps modify DICT_KEYS in guards.py


def is_hashable_python_var(x):
    from torch import Tensor

    # Note: Keep me in sync with is_hashable!
    # Even better, we should have a map of functions connecting the two
    from ..trace_rules import is_builtin_callable, is_numpy

    return (
        ConstantVariable.is_literal(x)
        or isinstance(x, (Tensor, enum.Enum, type, torch.nn.Module))
        or is_builtin_callable(x)
        or (isinstance(x, tuple) and all(is_hashable_python_var(e) for e in x))
        or is_numpy(x)
    )


def is_hashable(x):
    # Keep me in sync with is_hashable_python_var!
    # Even better, we should have a map of functions connecting the two
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
                variables.misc.SkipFilesVariable,
                variables.misc.NumpyVariable,
                variables.NNModuleVariable,
            ),
        )


class ConstDictVariable(VariableTracker):
    class _HashableTracker:
        """
        Auxiliary opaque internal class that wraps a VariableTracker and makes it hashable
        This should not be seen or touched by anything outside of ConstDictVariable and its children
        Note that it's also fine to put VTs into dictionaries and sets, but doing so does not take into account aliasing
        """

        def __init__(self, vt):
            # We specialize SymNodes
            vt = specialize_symnode(vt)
            assert is_hashable(vt), type(vt)
            self.vt = vt

        @property
        def underlying_value(self):
            if isinstance(self.vt, variables.TensorVariable):
                x = self.vt.as_proxy().node.meta["example_value"]
            elif isinstance(self.vt, variables.TupleVariable):
                Hashable = ConstDictVariable._HashableTracker
                x = tuple(Hashable(e).underlying_value for e in self.vt.items)
            elif isinstance(self.vt, variables.NNModuleVariable):
                return self.vt.module
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
            assert isinstance(other, Hashable)
            return Hashable._eq_impl(self.underlying_value, other.underlying_value)

    def __init__(
        self, items: Dict[VariableTracker, VariableTracker], user_cls=dict, **kwargs
    ):
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
        self.user_cls = user_cls

    def as_proxy(self):
        return {k.vt.as_proxy(): v.as_proxy() for k, v in self.items.items()}

    def as_python_constant(self):
        return {
            k.vt.as_python_constant(): v.as_python_constant()
            for k, v in self.items.items()
        }

    def keys_as_python_constant(self):
        return {k.vt.as_python_constant(): v for k, v in self.items.items()}

    def python_type(self):
        return self.user_cls

    def __contains__(self, vt):
        assert isinstance(vt, VariableTracker)
        Hashable = ConstDictVariable._HashableTracker
        return is_hashable(vt) and Hashable(vt) in self.items

    def reconstruct(self, codegen):
        # instructions to load collections.OrderedDict if necessary
        if self.user_cls is collections.OrderedDict:
            codegen.extend_output(
                [
                    codegen.create_load_python_module(collections, True),
                    codegen.create_load_attr("OrderedDict"),
                ]
            )
        # instructions to build the dict keys and values
        for key, value in self.items.items():
            codegen(key.vt)
            codegen(value)
        # BUILD_MAP and calling collections.OrderedDict if necessary
        if self.user_cls is collections.OrderedDict:
            return [
                create_instruction("BUILD_MAP", arg=len(self.items)),
                *create_call_function(1, False),
            ]
        # BUILD_MAP only if user_cls is dict
        else:
            return [create_instruction("BUILD_MAP", arg=len(self.items))]

    @staticmethod
    def _wrap_keys_python_var(d):
        """Wrap the keys of a dictionary with python objs as keys into Hashable objects"""
        assert all(is_hashable_python_var(k) for k in d.keys())
        Hashable = ConstDictVariable._HashableTracker
        from .builder import SourcelessBuilder

        build = SourcelessBuilder()
        return {Hashable(build(k)): v for k, v in d.items()}

    def getitem_const(self, arg: VariableTracker):
        key = ConstDictVariable._HashableTracker(arg)
        return self.items[key]

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from . import (
            BuiltinVariable,
            ConstantVariable,
            ListIteratorVariable,
            ListVariable,
            TupleVariable,
        )

        Hashable = ConstDictVariable._HashableTracker

        arg_hashable = args and is_hashable(args[0])

        if name == "__getitem__":
            assert len(args) == 1
            return self.getitem_const(args[0])

        elif name == "items":
            assert not (args or kwargs)
            return TupleVariable(
                [TupleVariable([k.vt, v]) for k, v in self.items.items()]
            )
        elif name == "keys":
            assert not (args or kwargs)
            return SetVariable(
                [k.vt for k in self.items.keys()],
                mutable_local=MutableLocal(),
            )
        elif name == "values":
            assert not (args or kwargs)
            return TupleVariable(list(self.items.values()))
        elif name == "copy":
            assert not (args or kwargs)
            return self.clone(items=self.items.copy(), mutable_local=MutableLocal())
        elif name == "__len__":
            assert not (args or kwargs)
            return ConstantVariable.create(len(self.items))
        elif name == "__setitem__" and arg_hashable and self.mutable_local:
            assert not kwargs and len(args) == 2
            tx.output.side_effects.mutation(self)
            self.items[Hashable(args[0])] = args[1]
            return ConstantVariable.create(None)
        elif name in ("pop", "get") and len(args) in (1, 2) and args[0] not in self:
            # missing item, return the default value
            if len(args) == 1:
                return ConstantVariable(None)
            else:
                return args[1]
        elif name == "pop" and arg_hashable and self.mutable_local:
            tx.output.side_effects.mutation(self)
            return self.items.pop(Hashable(args[0]))
        elif (
            name == "update"
            and len(args) == 1
            and isinstance(
                args[0],
                (
                    ConstDictVariable,
                    ListVariable,
                    TupleVariable,
                    ListIteratorVariable,
                ),
            )
            and self.mutable_local
        ):
            tx.output.side_effects.mutation(self)
            if isinstance(args[0], ConstDictVariable):
                dict_vt = args[0]
            else:
                dict_vt = BuiltinVariable.call_custom_dict(tx, dict, args[0])
            self.items.update(dict_vt.items)
            # all keys in kwargs are valid (`str`s)
            kwargs = ConstDictVariable._wrap_keys_python_var(kwargs)
            self.items.update(kwargs)
            return ConstantVariable.create(None)
        elif name in ("get", "__getattr__") and args[0] in self:
            return self.getitem_const(args[0])
        elif name == "__contains__" and len(args) == 1:
            return ConstantVariable.create(args[0] in self)
        else:
            return super().call_method(tx, name, args, kwargs)

    def unpack_var_sequence(self, tx):
        return [x.vt for x in self.items.keys()]


class DefaultDictVariable(ConstDictVariable):
    def __init__(self, items, user_cls, default_factory=None, **kwargs):
        super().__init__(items, user_cls, **kwargs)
        assert user_cls is collections.defaultdict
        self.default_factory = default_factory

    def is_python_constant(self):
        # Return false for unsupported defaults. This ensures that a bad handler
        # path is not taken in BuiltinVariable for getitem.
        if self.default_factory not in [list, tuple, dict] and not self.items:
            return False
        return super().is_python_constant()

    @staticmethod
    def is_supported_arg(arg):
        if isinstance(arg, variables.BuiltinVariable):
            return arg.fn in [list, tuple, dict]
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
                return self.getitem_const(args[0])
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


class SetVariable(VariableTracker):
    @dataclasses.dataclass
    class SetElement:
        vt: VariableTracker
        underlying_value: Any

        def __hash__(self) -> int:
            return hash(self.underlying_value)

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, SetVariable.SetElement):
                return False
            if isinstance(self.vt, variables.TensorVariable):
                return self.underlying_value is other.underlying_value
            else:
                return self.underlying_value == other.underlying_value

    def __init__(
        self,
        tx,
        items: List[VariableTracker],
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Note - Set is still backed by a list, because we want set behavior over the contents,
        assert isinstance(items, list)
        assert all(isinstance(x, VariableTracker) for x in items)

        self.items = []
        self._add(items)

    def as_proxy(self):
        return [x.as_proxy() for x in self.items]

    def python_type(self):
        return set

    def reconstruct(self, codegen):
        codegen.load_import_from("builtins", "set")
        codegen.foreach(self.items)
        return [
            create_instruction("BUILD_SET", arg=len(self.items))
        ] + create_call_function(1, True)

    # Note - this is only used for producing a set
    def _as_set_element(self, vt):
        from .base import VariableTracker
        from .misc import MethodWrapperVariable
        from .tensor import TensorVariable

        assert isinstance(vt, VariableTracker)

        if isinstance(vt, TensorVariable):
            fake_tensor = vt.as_proxy().node.meta.get("example_value")
            if fake_tensor is None:
                unimplemented(
                    "Cannot check Tensor object identity without its fake value"
                )
            return SetVariable.SetElement(vt, fake_tensor)
        if isinstance(vt, ConstantVariable):
            return SetVariable.SetElement(vt, vt.value)
        if isinstance(vt, MethodWrapperVariable):
            return SetVariable.SetElement(vt, vt.as_python_constant())
        if isinstance(vt, variables.UserDefinedObjectVariable):
            return SetVariable.SetElement(vt, vt.value)
        if isinstance(vt, variables.NNModuleVariable):
            return SetVariable.SetElement(vt, vt.module)

        unimplemented(f"Sets with {type(vt)} NYI")

    @property
    def _underlying_items(self):
        underlying_items = set()
        for current_item in self.items:
            assert (
                current_item not in underlying_items
            ), "Items modeling set invariant violated"
            underlying_items.add(self._as_set_element(current_item))
        return underlying_items

    def _add(self, item):
        underlying_items = self._underlying_items

        if isinstance(item, (list, set)):
            items_to_add = item
        else:
            items_to_add = [item]

        for item_to_add in items_to_add:
            set_element = self._as_set_element(item_to_add)
            if set_element not in underlying_items:
                underlying_items.add(set_element)
                self.items.append(set_element.vt)
            else:
                for e in underlying_items:
                    if hash(set_element) == hash(e):
                        alias_guard = make_dupe_guard(
                            e.vt.source, set_element.vt.source
                        )
                        if alias_guard:
                            install_guard(e.vt.source.make_guard(alias_guard))

    def call_method(
        self,
        tx,
        name,
        args: List[VariableTracker],
        kwargs: Dict[str, VariableTracker],
    ) -> "VariableTracker":
        # Somewhat duplicative of CommonListMethodsVariable - but better than to violate substitution
        # principles and end up with things like direct item access attempts on a set, or
        # getitem sources.
        if name == "add" and args and self.mutable_local:
            assert not kwargs
            item = args[0]
            tx.output.side_effects.mutation(self)
            self._add(item)
            return ConstantVariable.create(None)
        elif name == "pop" and self.mutable_local:
            assert not kwargs
            assert not args
            tx.output.side_effects.mutation(self)
            return self.items.pop()
        elif name == "__len__":
            return ConstantVariable.create(len(self.items))
        elif name == "__contains__":
            assert len(args) == 1
            assert not kwargs
            return iter_contains(self.items, args[0], tx, check_tensor_identity=True)
        else:
            return super().call_method(tx, name, args, kwargs)

    def getitem_const(self, arg: VariableTracker):
        raise RuntimeError("Illegal to getitem on a set")

    def as_python_constant(self):
        return self.python_type()([x.as_python_constant() for x in self.items])

    def unpack_var_sequence(self, tx):
        return list(self.items)


def _is_matching_transformers_cls(cls) -> bool:
    mod = sys.modules.get("transformers.file_utils")
    return mod is not None and issubclass(cls, mod.ModelOutput)


def _is_matching_diffusers_cls(cls) -> bool:
    mod = sys.modules.get("diffusers.utils")
    return mod is not None and issubclass(cls, mod.BaseOutput)


def _call_hasattr_customobj(self, tx, name: str) -> "VariableTracker":
    """Shared method between DataClassVariable and CustomizedDictVariable where items are attrs"""
    if name in self.items or hasattr(self.user_cls, name):
        return ConstantVariable(True)
    elif istype(self.mutable_local, MutableLocal) and self.source is None:
        # Something created locally can't have any extra fields on it
        return ConstantVariable(False)
    elif self.mutable_local is None and self.source:
        # Maybe add a guard
        try:
            example = tx.output.root_tx.get_example_value(self.source)
            install_guard(
                AttrSource(self.source, name).make_guard(GuardBuilder.HASATTR)
            )
            return ConstantVariable(hasattr(example, name))
        except KeyError:
            pass
    unimplemented(
        f"hasattr({self.__class__.__name__}, {name}) {self.mutable_local} {self.source}"
    )


class DataClassVariable(ConstDictVariable):
    """
    This is a bit of a hack to deal with
    transformers.file_utils.ModelOutput() from huggingface.

    ModelOutput causes trouble because it a a mix of a dataclass and a
    OrderedDict and it calls super() methods implemented in C.
    """

    # ModelOutput() excludes None, though generic datclasses don't
    include_none = False

    @staticmethod
    @functools.lru_cache(None)
    def _patch_once():
        try:
            from transformers.file_utils import ModelOutput

            for obj in ModelOutput.__dict__.values():
                if callable(obj):
                    skip_code(obj.__code__)
        except ImportError:
            pass

        try:
            from diffusers.utils import BaseOutput

            for obj in BaseOutput.__dict__.values():
                if callable(obj):
                    skip_code(obj.__code__)
        except ImportError:
            pass

    @staticmethod
    def is_matching_cls(cls):
        return _is_matching_transformers_cls(cls) or _is_matching_diffusers_cls(cls)

    @classmethod
    def is_matching_object(cls, obj):
        return cls.is_matching_cls(type(obj))

    @classmethod
    def create(cls, user_cls, args, kwargs, options):
        DataClassVariable._patch_once()

        skip_code(user_cls.__init__.__code__)
        keys = [f.name for f in dataclasses.fields(user_cls)]
        bound = inspect.signature(user_cls).bind(*args, **kwargs)
        bound.apply_defaults()
        assert set(bound.arguments.keys()) == set(keys)
        items = {}
        for key in keys:
            val = bound.arguments[key]
            key = ConstantVariable.create(key)
            if isinstance(val, VariableTracker):
                items[key] = val
            else:
                if cls.include_none:
                    assert variables.ConstantVariable.is_literal(val)
                    items[key] = variables.ConstantVariable.create(val)
                else:
                    assert val is None, f"unexpected {val}"

        if len(items) == 1 and not isinstance(items[keys[0]], variables.TensorVariable):
            unimplemented("DataClassVariable iterator constructor")
            # TODO(jansel): implement unpacking logic in ModelOutput.__post_init__

        return cls(items, user_cls, **options)

    @classmethod
    def wrap(cls, builder, obj):
        user_cls = type(obj)
        keys = [f.name for f in dataclasses.fields(user_cls)]

        excluded = []
        items = {}
        for key in keys:
            # __init__ function of a dataclass might not have yet defined the key
            if hasattr(obj, key):
                val = getattr(obj, key)
                var = builder.__class__(
                    tx=builder.tx, source=AttrSource(builder.source, key)
                )(val)
                if val is not None or cls.include_none:
                    key = ConstantVariable.create(key)
                    items[key] = var
                else:
                    excluded.append(var)
        return cls(items, user_cls)

    def __init__(self, items, user_cls, **options):
        super().__init__(items, user_cls, **options)
        assert self.is_matching_cls(user_cls)

    def as_proxy(self):
        raise NotImplementedError()

    def reconstruct(self, codegen):
        codegen.extend_output([codegen._create_load_const(self.user_cls)])
        # All the keys are just wrapped strings
        d = self.keys_as_python_constant()
        codegen.foreach(d.values())
        keys = tuple(d.keys())
        return codegen.create_call_function_kw(len(keys), keys, True)

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if name == "__getitem__":
            assert not kwargs and len(args) == 1
            val = args[0]
            if val.python_type() == str:
                return self.getitem_const(val)
            else:
                return self.call_method(tx, "to_tuple", [], {}).call_method(
                    tx, "__getitem__", args, kwargs
                )
        elif name == "to_tuple":
            assert not (args or kwargs)
            return variables.TupleVariable(list(self.items.values()))
        elif name == "__setattr__":
            name = "__setitem__"
        return super().call_method(tx, name, args, kwargs)

    def var_getattr(self, tx, name: str) -> "VariableTracker":
        name_vt = ConstantVariable.create(name)
        if name_vt in self:
            return self.call_method(tx, "__getitem__", [name_vt], {})
        elif not self.include_none:
            defaults = {f.name: f.default for f in dataclasses.fields(self.user_cls)}
            if name in defaults:
                assert variables.ConstantVariable.is_literal(defaults[name])
                return variables.ConstantVariable.create(defaults[name])
        super().var_getattr(tx, name)

    call_hasattr = _call_hasattr_customobj


class CustomizedDictVariable(ConstDictVariable):
    @staticmethod
    def is_matching_cls(cls):
        # True if using default OrderedDict.__init__ and did not implement __post_init__
        if (
            issubclass(cls, collections.OrderedDict)
            and cls.__init__ is collections.OrderedDict.__init__
            and not hasattr(cls, "__post_init__")
        ):
            return True
        # hack for HF usecase:
        #   assume dataclass annotation for ModelOutput subclass
        #   assume self.create is AA to ModelOutput.__post_init__
        return _is_matching_transformers_cls(cls) or _is_matching_diffusers_cls(cls)

    @classmethod
    def is_matching_object(cls, obj):
        return cls.is_matching_cls(type(obj))

    # called from user_defined.py
    # when is_matching_cls(cls) is true
    @classmethod
    def create(cls, user_cls, args, kwargs, options):
        # avoid tracing when returning ModelOutput from forward func
        for attr_name in ("__init__", "__post_init__", "__setattr__", "__setitem__"):
            if hasattr(user_cls, attr_name):
                fn = getattr(user_cls, attr_name)
                assert callable(fn), f"expect callable attr {attr_name}"
                if hasattr(fn, "__code__"):
                    skip_code(fn.__code__)

        if dataclasses.is_dataclass(user_cls):
            # @dataclass CustomDict(a=1, b=2)
            bound = inspect.signature(user_cls).bind(*args, **kwargs)
            bound.apply_defaults()

            def make_var(x):
                if isinstance(x, VariableTracker):
                    return x
                elif ConstantVariable.is_literal(x):
                    return ConstantVariable.create(x)
                else:
                    unimplemented(
                        "expect VariableTracker or ConstantVariable.is_literal"
                    )

            items = {
                ConstantVariable.create(k): make_var(v)
                for k, v in bound.arguments.items()
            }
        elif not args:
            # CustomDict(a=1, b=2) in the general (non-dataclass) case.
            items = {ConstantVariable.create(k): v for k, v in kwargs.items()}
        elif len(args) == 1 and isinstance(args[0], ConstDictVariable) and not kwargs:
            # CustomDict({'a': 1, 'b': 2})
            items = args[0].items
        else:
            unimplemented("custom dict init with args/kwargs unimplemented")

        return cls(items, user_cls, **options)

    # called from builder.py
    @classmethod
    def wrap(cls, builder, obj):
        raise NotImplementedError()

    def __init__(self, items, user_cls, **options):
        super().__init__(items, user_cls, **options)
        assert self.is_matching_cls(user_cls)

    def as_proxy(self):
        raise NotImplementedError()

    # 'RETURN_VALUE triggered compile'
    # called from torch/_dynamo/codegen.py
    def reconstruct(self, codegen):
        codegen.extend_output([codegen._create_load_const(self.user_cls)])
        # All the keys are just wrapped strings
        d = self.keys_as_python_constant()
        codegen.foreach(d.values())
        keys = tuple(d.keys())
        return codegen.create_call_function_kw(len(keys), keys, True)

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        fn = getattr(self.user_cls, name)
        source = None if self.source is None else AttrSource(self.source, name)

        if hasattr(fn, "__objclass__") and fn.__objclass__ in (
            dict,
            collections.OrderedDict,
        ):
            # for python dict method without overridden
            return super().call_method(tx, name, args, kwargs)
        elif name in ("__getitem__", "to_tuple", "__setitem__", "__setattr__"):
            # for user overridden method
            return tx.inline_user_function_return(
                variables.UserFunctionVariable(fn, source=source),
                [self] + list(args),
                kwargs,
            )

        unimplemented("custom dict: call_method unimplemented name=%s", name)

    def var_getattr(self, tx, name: str) -> "VariableTracker":
        name_vt = ConstantVariable.create(name)
        if name_vt in self:
            return self.call_method(tx, "__getitem__", [name_vt], {})
        super().var_getattr(tx, name)

    call_hasattr = _call_hasattr_customobj


@functools.lru_cache(None)
def _install_PretrainedConfig_patch():
    import transformers

    # We need to monkeypatch transformers here, sadly.
    # TODO(voz): Upstream to transformers lib

    def _dynamo_overriden_transformers_eq(self, other):
        if not hasattr(other, "__dict__"):
            return False
        return self.__dict__ == other.__dict__

    transformers.configuration_utils.PretrainedConfig.__eq__ = (
        _dynamo_overriden_transformers_eq
    )


class HFPretrainedConfigVariable(VariableTracker):
    """
    Hack for HuggingFace PretrainedConfig
    """

    @staticmethod
    def is_matching_cls(cls):
        mod = sys.modules.get("transformers.configuration_utils")
        is_match = mod is not None and issubclass(cls, mod.PretrainedConfig)

        # Lazily install monkeypatch the first time we see it in dynamo
        if is_match:
            _install_PretrainedConfig_patch()
        return is_match

    @classmethod
    def is_matching_object(cls, obj):
        return cls.is_matching_cls(type(obj))

    def __init__(self, obj, **kwargs):
        super().__init__(**kwargs)
        self.obj = obj
        assert self.is_matching_cls(type(obj))

    def var_getattr(self, tx, name: str) -> "VariableTracker":
        from . import ConstantVariable

        return ConstantVariable.create(getattr(self.obj, name))

    def call_hasattr(self, tx, name: str) -> "VariableTracker":
        return variables.ConstantVariable.create(hasattr(self.obj, name))


class PythonSysModulesVariable(VariableTracker):
    """Special case for sys.modules.

    Without this we will guard on the exact set of modules imported in the
    lifetime of the python program.
    """

    def python_type(self):
        return dict

    @staticmethod
    def reconstruct(self, codegen):
        codegen.extend_output(
            [
                codegen.create_load_python_module(sys, True),
                codegen.create_load_attr("modules"),
            ]
        )

    def call_method(
        self, tx, name, args: List[VariableTracker], kwargs: Dict[str, VariableTracker]
    ):
        from .builder import VariableBuilder

        if name == "__getitem__":
            return self.call_getitem(tx, *args, **kwargs)
        elif name == "get":
            return self.call_get(tx, *args, **kwargs)
        elif name == "__contains__":
            return self.call_contains(tx, *args, **kwargs)

        # Fallback to dict implementation
        real_dict = VariableBuilder(tx, self.source)(sys.modules)
        return real_dict.call_method(tx, name, args, kwargs)

    def _contains_helper(self, tx, key: VariableTracker):
        k = key.as_python_constant()
        has_key = k in sys.modules
        install_guard(
            self.make_guard(
                functools.partial(GuardBuilder.DICT_CONTAINS, key=k, invert=not has_key)
            )
        )
        return k, has_key

    def call_contains(self, tx, key: VariableTracker):
        k, has_key = self._contains_helper(tx, key)
        return ConstantVariable.create(value=has_key)

    def call_get(
        self, tx, key: VariableTracker, default: Optional[VariableTracker] = None
    ):
        from .builder import VariableBuilder

        k, has_key = self._contains_helper(tx, key)

        if has_key:
            return VariableBuilder(
                tx,
                GetItemSource(self.source, k),
            )(sys.modules[k])

        if default is not None:
            return default

        return ConstantVariable.create(value=None)

    def call_getitem(self, tx, key: VariableTracker):
        from .builder import VariableBuilder

        k, has_key = self._contains_helper(tx, key)
        return VariableBuilder(
            tx,
            GetItemSource(self.source, k),
        )(sys.modules[k])
