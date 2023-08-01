import collections
import dataclasses
import functools
import inspect
from typing import Any, Dict, List, Tuple

import torch
import torch.fx
from torch.fx import _pytree as fx_pytree

from torch.utils import _pytree as pytree

from .. import variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..eval_frame import skip_code
from ..exc import unimplemented
from ..source import AttrSource, GlobalWeakRefSource
from ..utils import global_key_name, istensor
from .base import MutableLocal, VariableTracker
from .constant import ConstantVariable
from .tensor import TensorVariable


class ConstDictVariable(VariableTracker):
    def __init__(self, items, user_cls, recursively_contains=None, **kwargs):
        super().__init__(recursively_contains=recursively_contains, **kwargs)

        self.guards.update(VariableTracker.propagate(items.values())["guards"])
        self.items = items
        self.user_cls = user_cls

    def as_proxy(self):
        return {k: v.as_proxy() for k, v in self.items.items()}

    def as_python_constant(self):
        return {k: v.as_python_constant() for k, v in self.items.items()}

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
        for key in self.items.keys():
            if istensor(key):
                codegen.append_output(
                    codegen.create_load_global(global_key_name(key), True, add=True)
                )
                codegen.extend_output(create_call_function(0, False))
            else:
                codegen.append_output(codegen.create_load_const(key))
            codegen(self.items[key])
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
        return self.items[ConstDictVariable.get_key(arg)].add_options(self, arg)

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
            return TupleVariable(
                [
                    TupleVariable(
                        [
                            ConstDictVariable._key_to_var(
                                tx,
                                k,
                                **options,
                            ),
                            v,
                        ],
                        **options,
                    )
                    for k, v in val.items()
                ],
                **options,
            )
        elif name == "keys":
            assert not (args or kwargs)
            return TupleVariable(
                [
                    ConstDictVariable._key_to_var(
                        tx,
                        k,
                        **options,
                    )
                    for k in val.keys()
                ],
                **options,
            )

        elif name == "values":
            assert not (args or kwargs)
            return TupleVariable(list(val.values()), **options)
        elif name == "__len__":
            assert not (args or kwargs)
            return ConstantVariable(len(self.items), **options)
        elif (
            name == "__setitem__"
            and args
            and ConstDictVariable.is_valid_key(args[0])
            and self.mutable_local
        ):
            assert not kwargs and len(args) == 2
            k = ConstDictVariable.get_key(args[0])

            if istensor(k):
                tx.store_dict_key(global_key_name(k), k)
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
            and ConstDictVariable.is_valid_key(args[0])
            and ConstDictVariable.get_key(args[0]) not in self.items
            and len(args) == 2
        ):
            # missing item, return the default value
            return args[1].add_options(options)
        elif (
            name == "pop"
            and args
            and ConstDictVariable.is_valid_key(args[0])
            and self.mutable_local
        ):
            newval = collections.OrderedDict(val)
            result = newval.pop(ConstDictVariable.get_key(args[0]))
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
            and ConstDictVariable.is_valid_key(args[0])
            and ConstDictVariable.get_key(args[0]) in self.items
        ):
            result = self.items[ConstDictVariable.get_key(args[0])]
            return result.add_options(options)
        elif (
            name == "__contains__" and args and ConstDictVariable.is_valid_key(args[0])
        ):
            return ConstantVariable(
                ConstDictVariable.get_key(args[0]) in self.items, **options
            )
        else:
            return super().call_method(tx, name, args, kwargs)

    def modifed(self, items, recursively_contains, **options):
        """a copy of self with different items"""
        return self.clone(
            items=items, recursively_contains=recursively_contains, **options
        )

    def unpack_var_sequence(self, tx):
        options = VariableTracker.propagate([self])
        val = self.items
        result = [ConstDictVariable._key_to_var(tx, k, **options) for k in val.keys()]
        return result

    @classmethod
    def get_key(cls, arg: VariableTracker):
        if isinstance(arg, TensorVariable) and arg.specialized_value is not None:
            return arg.specialized_value
        else:
            return arg.as_python_constant()

    @classmethod
    def is_valid_key(cls, key):
        return (
            key.is_python_constant()
            or isinstance(key, TensorVariable)
            and key.specialized_value is not None
            or isinstance(key, ConstantVariable)
            and key.python_type() is torch.dtype
        )

    @classmethod
    def _key_to_var(cls, tx, key, **options):
        from .builder import VariableBuilder

        if istensor(key):
            return VariableBuilder(tx, GlobalWeakRefSource(global_key_name(key)))(key)
        else:
            assert ConstantVariable.is_literal(key)
            return ConstantVariable(key, **options)


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
            k = ConstDictVariable.get_key(args[0])

            if k in self.items:
                return self.getitem_const(args[0])
            else:
                if self.default_factory is None:
                    raise KeyError(f"{k}")
                else:
                    if istensor(k):
                        tx.store_dict_key(global_key_name(k), k)
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
                    items[key] = variables.ConstantVariable(val)
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
                tx, "__getitem__", [variables.ConstantVariable(name)], {}
            )
        elif not self.include_none:
            defaults = {f.name: f.default for f in dataclasses.fields(self.user_cls)}
            if name in defaults:
                assert variables.ConstantVariable.is_literal(defaults[name])
                return variables.ConstantVariable(defaults[name]).add_options(self)
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

        return ConstantVariable(getattr(self.obj, name))

    def call_hasattr(self, tx, name: str) -> "VariableTracker":
        return variables.ConstantVariable(hasattr(self.obj, name)).add_options(self)


def _dictvariable_flatten(d: ConstDictVariable) -> Tuple[List[Any], pytree.Context]:
    if d.python_type() is not dict:
        # Note - ConstDictVariable can contain different kinds of dicts.
        # However, flattening for those must differ and so cannot share the same registration as even if we
        # consult the underlying python_type() to guide our flattening, that data will need to be propagated
        # to unflatten. We do not have a good mechanism of doing this today, so we find it easier to treat this
        # as unimplemented for now.

        # TODO - Add support for flattening a ConstDictVariable with any underlying user_cls
        unimplemented(f"Unsupported flattening of {d.python_type()}")
    return list(d.items.values()), list(d.items.keys())


def _dictvariable_unflatten(
    values: List[Any], context: pytree.Context
) -> ConstDictVariable:
    assert all(isinstance(x, VariableTracker) for x in values)

    # Guard propagation happens in the ConstDictVariable constructor
    return ConstDictVariable(
        dict(zip(context, values)), user_cls=dict, mutable_local=MutableLocal()
    )


def _register_dynamo_dict_to_tree_spec():
    pytree._register_pytree_node(
        ConstDictVariable,
        _dictvariable_flatten,
        _dictvariable_unflatten,
        pytree._dict_to_str,
        pytree._maybe_str_to_dict,
    )

    fx_pytree.register_pytree_flatten_spec(
        ConstDictVariable,
        _dictvariable_flatten,
    )
