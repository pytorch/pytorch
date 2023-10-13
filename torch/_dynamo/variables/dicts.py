import collections
import dataclasses
import functools
import inspect
from typing import Dict, List

from .. import variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..eval_frame import skip_code

from ..exc import unimplemented
from ..source import AttrSource
from ..utils import HashableTracker, is_hashable
from .base import MutableLocal, VariableTracker
from .constant import ConstantVariable


class ConstDictVariable(VariableTracker):
    def __init__(self, items, user_cls, recursively_contains=None, **kwargs):
        super().__init__(recursively_contains=recursively_contains, **kwargs)

        assert all(
            isinstance(x, HashableTracker) and isinstance(v, VariableTracker)
            for x, v in items.items()
        )
        self.guards.update(VariableTracker.propagate(items.values())["guards"])
        self.items = items
        self.user_cls = user_cls

    def as_proxy(self):
        return {k.vt.as_proxy(): v.as_proxy() for k, v in self.items.items()}

    def as_python_constant(self):
        return {
            k.vt.as_python_constant(): v.as_python_constant()
            for k, v in self.items.items()
        }

    def python_type(self):
        return self.user_cls

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

    def getitem_const(self, arg: VariableTracker):
        key = HashableTracker(arg)
        return self.items[key].add_options(self, arg)

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from . import ConstantVariable, TupleVariable

        options = VariableTracker.propagate(self, args, kwargs.values())
        val = self.items

        if name == "__getitem__":
            return self.getitem_const(args[0])

        elif name == "items":
            assert not (args or kwargs)
            items = [TupleVariable([k.vt, v], **options) for k, v in self.items.items()]
            return TupleVariable(items, **options)
        elif name == "keys":
            assert not (args or kwargs)
            return SetVariable(val.keys(), mutable_local=MutableLocal())
        elif name == "values":
            assert not (args or kwargs)
            return TupleVariable(list(val.values()), **options)
        elif name == "__len__":
            assert not (args or kwargs)
            return ConstantVariable.create(len(self.items), **options)
        elif (
            name == "__setitem__"
            and args
            and is_hashable(args[0])
            and self.mutable_local
        ):
            assert not kwargs and len(args) == 2
            k = HashableTracker(args[0])

            newval = collections.OrderedDict(val)
            newval[k] = args[1]

            new_rec_contains = self.recursively_contains.union(
                args[1].recursively_contains
            )
            if args[1].mutable_local is not None:
                new_rec_contains.add(args[1].mutable_local)

            return tx.replace_all(
                self,
                self.modifed(newval, new_rec_contains, **options),
            )
        elif (
            name in ("pop", "get")
            and args
            and is_hashable(args[0])
            and HashableTracker(args[0]) not in self.items
            and len(args) == 2
        ):
            # missing item, return the default value
            return args[1].add_options(options)
        elif name == "pop" and args and is_hashable(args[0]) and self.mutable_local:
            newval = collections.OrderedDict(val)
            result = newval.pop(HashableTracker(args[0]))
            tx.replace_all(self, self.modifed(newval, None, **options))
            return result.add_options(options)
        elif (
            name == "update"
            and args
            and isinstance(args[0], ConstDictVariable)
            and self.mutable_local
        ):
            newval = collections.OrderedDict(val)
            newval.update(args[0].items)
            new_rec_contains = self.recursively_contains.union(
                args[0].recursively_contains
            )
            result = self.modifed(
                newval, recursively_contains=new_rec_contains, **options
            )
            return tx.replace_all(self, result)
        elif (
            name in ("get", "__getattr__")
            and args
            and is_hashable(args[0])
            and HashableTracker(args[0]) in self.items
        ):
            result = self.items[HashableTracker(args[0])]
            return result.add_options(options)
        elif name == "__contains__" and args:
            return ConstantVariable.create(
                is_hashable(args[0]) and HashableTracker(args[0]) in self.items,
                **options,
            )
        else:
            return super().call_method(tx, name, args, kwargs)

    def modifed(self, items, recursively_contains, **options):
        """a copy of self with different items"""
        return self.clone(
            items=items, recursively_contains=recursively_contains, **options
        )

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
        options = VariableTracker.propagate(self, args, kwargs.values())

        if name == "__getitem__":
            k = HashableTracker(args[0])

            if k in self.items:
                return self.getitem_const(args[0])
            else:
                if self.default_factory is None:
                    raise KeyError(f"{k}")
                else:
                    new_val = collections.OrderedDict(self.items)
                    default_var = self.default_factory.call_function(tx, [], {})
                    new_val[k] = default_var
                    new_rec_contains = self.recursively_contains.union(
                        default_var.recursively_contains
                    )
                    if default_var.mutable_local is not None:
                        new_rec_contains.add(default_var.mutable_local)
                    tx.replace_all(
                        self, self.modifed(new_val, new_rec_contains, **options)
                    )
                    return default_var
        else:
            return super().call_method(tx, name, args, kwargs)


class SetVariable(ConstDictVariable):
    """We model a sets as dictonary with None values"""

    def __init__(
        self,
        items,
        recursively_contains=None,
        regen_guards=True,
        **kwargs,
    ):
        if "user_cls" in kwargs:
            assert kwargs["user_cls"] is dict
        else:
            kwargs = dict(user_cls=dict, **kwargs)

        items = dict.fromkeys(items, SetVariable._default_value())
        super().__init__(items, recursively_contains=recursively_contains, **kwargs)

    @property
    def set_items(self):
        return set(self.items.keys())

    @staticmethod
    def _default_value():
        # Variable to fill in he keys of the dictinary
        return ConstantVariable.create(None)

    def as_proxy(self):
        return {k.vt.as_proxy() for k in self.items}

    def python_type(self):
        return set

    def as_python_constant(self):
        return {k.vt.as_python_constant() for k in self.items}

    def reconstruct(self, codegen):
        codegen.foreach([x.vt for x in self.items])
        return [create_instruction("BUILD_SET", arg=len(self.items))]

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
        return super().call_method(tx, name, args, kwargs)

    def getitem_const(self, arg: VariableTracker):
        raise RuntimeError("Illegal to getitem on a set")


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
        from transformers.file_utils import ModelOutput

        for obj in ModelOutput.__dict__.values():
            if callable(obj):
                skip_code(obj.__code__)

    @staticmethod
    def is_matching_cls(cls):
        try:
            from transformers.file_utils import ModelOutput

            return issubclass(cls, ModelOutput)
        except ImportError:
            return False

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
        items = collections.OrderedDict()
        for key in keys:
            val = bound.arguments[key]
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
        items = collections.OrderedDict()
        for key in keys:
            # __init__ function of a dataclass might not have yet defined the key
            if hasattr(obj, key):
                val = getattr(obj, key)
                var = builder.__class__(
                    tx=builder.tx, source=AttrSource(builder.source, key)
                )(val)
                if val is not None or cls.include_none:
                    items[key] = var
                else:
                    excluded.append(var)
        return cls(
            items, user_cls, **VariableTracker.propagate(excluded, items.values())
        )

    def __init__(self, items, user_cls, **options):
        super().__init__(items, user_cls, **options)
        assert self.is_matching_cls(user_cls)

    def as_proxy(self):
        raise NotImplementedError()

    def reconstruct(self, codegen):
        codegen.extend_output([codegen._create_load_const(self.user_cls)])
        keys = tuple(self.items.keys())
        for key in keys:
            codegen(self.items[key])
        return codegen.create_call_function_kw(len(keys), keys, True)

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        options = VariableTracker.propagate(self, args, kwargs.values())
        if name == "__getitem__":
            assert not kwargs and len(args) == 1
            index = args[0].as_python_constant()
            if isinstance(index, str):
                return self.items[index].add_options(options)
            else:
                return (
                    self.call_method(tx, "to_tuple", [], {})
                    .call_method(tx, "__getitem__", args, kwargs)
                    .add_options(options)
                )
        elif name == "to_tuple":
            assert not (args or kwargs)
            return variables.TupleVariable(list(self.items.values()), **options)
        elif name == "__setattr__":
            name = "__setitem__"
        return super().call_method(tx, name, args, kwargs)

    def var_getattr(self, tx, name: str) -> "VariableTracker":
        if name in self.items:
            return self.call_method(
                tx, "__getitem__", [variables.ConstantVariable.create(name)], {}
            )
        elif not self.include_none:
            defaults = {f.name: f.default for f in dataclasses.fields(self.user_cls)}
            if name in defaults:
                assert variables.ConstantVariable.is_literal(defaults[name])
                return variables.ConstantVariable.create(defaults[name]).add_options(
                    self
                )
        super().var_getattr(tx, name)


class CustomizedDictVariable(ConstDictVariable):
    @staticmethod
    def is_matching_cls(cls):
        try:
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
            # for non-HF usecase:
            #   check __module__ string to avoid costy HF import
            if cls.__module__ != "transformers.modeling_outputs":
                return False
            from transformers.file_utils import ModelOutput

            return issubclass(cls, ModelOutput)
        except ImportError:
            return False

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

        if not args and not kwargs:
            # CustomDict() init with empty arguments
            raw_items = collections.OrderedDict()
        elif dataclasses.is_dataclass(user_cls):
            # @dataclass CustomDict(a=1, b=2)
            bound = inspect.signature(user_cls).bind(*args, **kwargs)
            bound.apply_defaults()
            raw_items = bound.arguments
        elif len(args) == 1 and isinstance(args[0], ConstDictVariable) and not kwargs:
            # CustomDict({'a': 1, 'b': 2})
            raw_items = args[0].items
        else:
            unimplemented("custome dict init with args/kwargs unimplemented")

        def make_key(k):
            return HashableTracker(variables.ConstantVariable.create(k))

        items = collections.OrderedDict()
        for key in raw_items.keys():
            val = raw_items[key]
            key = make_key(key)
            if isinstance(val, VariableTracker):
                items[key] = val
            elif variables.ConstantVariable.is_literal(val):
                items[key] = variables.ConstantVariable.create(val)
            else:
                unimplemented("expect VariableTracker or ConstantVariable.is_literal")

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
        keys = tuple(self.items.keys())
        for key in keys:
            codegen(self.items[key])
        return codegen.create_call_function_kw(len(keys), keys, True)

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        options = VariableTracker.propagate(self, args, kwargs.values())
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
                variables.UserFunctionVariable(fn, source=source, **options),
                [self] + list(args),
                kwargs,
            )

        unimplemented("custom dict: call_method unimplemented name=%s", name)

    def var_getattr(self, tx, name: str) -> "VariableTracker":
        if name in self.items:
            return self.call_method(
                tx, "__getitem__", [variables.ConstantVariable.create(name)], {}
            )
        super().var_getattr(tx, name)


class HFPretrainedConfigVariable(VariableTracker):
    """
    Hack for HuggingFace PretrainedConfig
    """

    @staticmethod
    def is_matching_cls(cls):
        try:
            from transformers.configuration_utils import PretrainedConfig

            return issubclass(cls, PretrainedConfig)
        except ImportError:
            return False

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
        return variables.ConstantVariable.create(hasattr(self.obj, name)).add_options(
            self
        )
