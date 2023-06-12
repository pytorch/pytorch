import collections
import contextlib
import functools
import importlib
import inspect
import itertools
import random
import types
from typing import Dict, List

import torch.nn

from .. import variables
from ..allowed_functions import is_allowed
from ..exc import unimplemented
from ..guards import GuardBuilder
from ..source import AttrSource, ODictGetItemSource, RandomValueSource
from ..utils import (
    all_hook_names,
    check_constant_args,
    get_custom_getattr,
    is_namedtuple_cls,
    istype,
    namedtuple_fields,
    object_has_getattribute,
)
from .base import MutableLocal, VariableTracker
from .ctx_manager import GenericContextWrappingVariable, NullContextVariable


class UserDefinedVariable(VariableTracker):
    pass


class UserDefinedClassVariable(UserDefinedVariable):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def as_python_constant(self):
        return self.value

    def python_type(self):
        return type(self.value)

    def var_getattr(self, tx, name: str) -> "VariableTracker":
        from . import ConstantVariable
        from .builder import VariableBuilder

        options = VariableTracker.propagate(self)
        source = AttrSource(self.source, name) if self.source is not None else None
        try:
            obj = inspect.getattr_static(self.value, name)
        except AttributeError:
            obj = None
        if isinstance(obj, staticmethod):
            return variables.UserFunctionVariable(
                obj.__get__(self.value), source=source, **options
            )
        elif isinstance(obj, classmethod):
            return variables.UserMethodVariable(
                obj.__func__, self, source=source, **options
            )

        if name in getattr(self.value, "__dict__", {}) or ConstantVariable.is_literal(
            obj
        ):
            if source:
                return VariableBuilder(tx, source)(obj).add_options(options)
            elif ConstantVariable.is_literal(obj):
                return ConstantVariable(obj, **options)

        return super().var_getattr(tx, name)

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if (
            name == "__subclasses__"
            and len(args) == 0
            and not kwargs
            and "__subclasses__" not in self.value.__dict__
        ):
            options = VariableTracker.propagate(self, args, kwargs.values())
            options["mutable_local"] = MutableLocal()
            subs_as_vars: List[VariableTracker] = list()
            for sub in self.value.__subclasses__():
                source = AttrSource(tx.import_source(sub.__module__), sub.__name__)
                subs_as_vars.append(
                    variables.UserDefinedClassVariable(sub, source=source)
                )

            return variables.ListVariable(subs_as_vars, **options)

        return super().call_method(tx, name, args, kwargs)

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        from ..side_effects import SideEffects

        options = VariableTracker.propagate(self, args, kwargs.values())

        if self.value is contextlib.nullcontext:
            return NullContextVariable(**options)
        elif (
            issubclass(type(self.value), type)
            and hasattr(self.value, "__enter__")
            and hasattr(self.value, "__exit__")
            and check_constant_args(args, kwargs)
            and len(kwargs) == 0  # TODO(ybliang): support kwargs
        ):
            unwrapped_args = [x.as_python_constant() for x in args]
            return GenericContextWrappingVariable(
                unwrapped_args, cm_obj=self.value(*unwrapped_args), **options
            )
        elif is_namedtuple_cls(self.value):
            fields = namedtuple_fields(self.value)
            items = list(args)
            items.extend([None] * (len(fields) - len(items)))
            for name, value in kwargs.items():
                assert name in fields
                items[fields.index(name)] = value
            assert all(x is not None for x in items)
            return variables.NamedTupleVariable(
                items, self.value, **VariableTracker.propagate(self, items)
            )
        elif (
            inspect.getattr_static(self.value, "__new__", None) in (object.__new__,)
            and SideEffects.cls_supports_mutation_side_effects(self.value)
            and self.source
        ):
            var = tx.output.side_effects.track_object_new(
                self.source,
                self.value,
                variables.UnspecializedNNModuleVariable
                if issubclass(self.value, torch.nn.Module)
                else UserDefinedObjectVariable,
                options,
            )
            if (
                inspect.getattr_static(self.value, "__init__", None)
                is torch.nn.Module.__init__
            ):
                tx.output.side_effects.store_attr(
                    var, "__call_nn_module_init", variables.ConstantVariable(True)
                )
                return var
            else:
                return var.add_options(var.call_method(tx, "__init__", args, kwargs))
        elif variables.DataClassVariable.is_matching_cls(self.value):
            options["mutable_local"] = MutableLocal()
            return variables.DataClassVariable.create(self.value, args, kwargs, options)

        return super().call_function(tx, args, kwargs)

    def const_getattr(self, tx, name):
        if name == "__name__":
            return self.value.__name__
        return super().const_getattr(tx, name)


class UserDefinedObjectVariable(UserDefinedVariable):
    """
    Mostly objects of defined type.  Catch-all for something where we only know the type.
    """

    def __init__(self, value, value_type=None, **kwargs):
        super().__init__(**kwargs)
        self.value = value
        self.value_type = value_type or type(value)
        assert type(value) is self.value_type

    def __str__(self):
        inner = self.value_type.__name__
        if inner in [
            "builtin_function_or_method",
            "getset_descriptor",
            "method_descriptor",
            "method",
        ]:
            inner = str(getattr(self.value, "__name__", None))
        return f"{self.__class__.__name__}({inner})"

    def python_type(self):
        return self.value_type

    @staticmethod
    @functools.lru_cache(None)
    def _supported_random_functions():
        fns = {
            random.random,
            random.randint,
            random.randrange,
            random.uniform,
        }
        return fns

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from . import ConstantVariable, TupleVariable, UserMethodVariable

        options = VariableTracker.propagate(self, args, kwargs.values())

        if name not in getattr(self.value, "__dict__", {}):
            try:
                method = inspect.getattr_static(type(self.value), name)
            except AttributeError:
                method = None
            if method is object.__init__:
                return ConstantVariable(None, **options)

            if method is collections.OrderedDict.keys and self.source:
                # subclass of OrderedDict
                assert not (args or kwargs)
                keys = list(self.value.keys())
                assert all(map(ConstantVariable.is_literal, keys))
                return TupleVariable(
                    [ConstantVariable(k, **options) for k in keys], **options
                ).add_guard(self.source.make_guard(GuardBuilder.ODICT_KEYS))

            if (
                method is collections.OrderedDict.__contains__
                and len(args) == 1
                and isinstance(args[0], ConstantVariable)
                and inspect.getattr_static(type(self.value), "keys")
                is collections.OrderedDict.keys
            ):
                assert not kwargs
                return ConstantVariable(
                    args[0].as_python_constant() in self.value, **options
                ).add_guard(self.source.make_guard(GuardBuilder.ODICT_KEYS))

            if (
                method is collections.OrderedDict.items
                and isinstance(self.value, collections.OrderedDict)
                and self.source
            ):
                assert not (args or kwargs)
                items = []
                keys = self.call_method(tx, "keys", [], {})
                options = VariableTracker.propagate(self, args, kwargs.values(), keys)
                for key in keys.unpack_var_sequence(tx):
                    items.append(
                        TupleVariable(
                            [key, self.odict_getitem(tx, key)],
                            **options,
                        )
                    )
                return TupleVariable(items, **options)

            if method is collections.OrderedDict.__getitem__ and len(args) == 1:
                assert not kwargs
                return self.odict_getitem(tx, args[0])

            # check for methods implemented in C++
            if isinstance(method, types.FunctionType):
                source = (
                    None
                    if self.source is None
                    else AttrSource(AttrSource(self.source, "__class__"), name)
                )
                # TODO(jansel): add a guard to check for monkey patching?
                return UserMethodVariable(
                    method, self, source=source, **options
                ).call_function(tx, args, kwargs)

        return super().call_method(tx, name, args, kwargs)

    def is_supported_random(self):
        try:
            return self.value in self._supported_random_functions()
        except TypeError:
            # TypeError: unhashable type
            return False

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        from .builder import VariableBuilder

        if (
            self.is_supported_random()
            and all(k.is_python_constant() for k in args)
            and all(v.is_python_constant() for v in kwargs.values())
        ):
            args = [x.as_python_constant() for x in args]
            kwargs = {k: v.as_python_constant() for k, v in kwargs.items()}
            random_call_index = len(tx.random_calls)
            example_value = self.value(*args, **kwargs)
            source = RandomValueSource(random_call_index)
            tx.random_calls.append((self.value, args, kwargs))
            return VariableBuilder(tx, source).wrap_unspecialized_primitive(
                example_value
            )
        elif istype(self.value, types.MethodType):
            func = self.value.__func__
            obj = self.value.__self__
            if (
                func is torch.utils._contextlib._DecoratorContextManager.clone
                and is_allowed(obj.__class__)
                and not (args or kwargs)
            ):
                return variables.TorchVariable(obj.__class__).call_function(
                    tx, args, kwargs
                )
        elif (
            istype(self.value, functools.partial)
            and is_allowed(self.value.func)
            and all(
                variables.ConstantVariable.is_literal(v)
                for v in itertools.chain(self.value.args, self.value.keywords.values())
            )
        ):
            options = VariableTracker.propagate(self, args, kwargs.values())
            options.setdefault("guards", set())
            if self.source:
                options["guards"].add(
                    AttrSource(self.source, "func").make_guard(GuardBuilder.ID_MATCH)
                )
                options["guards"].add(
                    AttrSource(self.source, "args").make_guard(
                        GuardBuilder.CONSTANT_MATCH
                    )
                )
                options["guards"].add(
                    AttrSource(self.source, "keywords").make_guard(
                        GuardBuilder.CONSTANT_MATCH
                    )
                )

            partial_args = [variables.ConstantVariable(v) for v in self.value.args]
            partial_args.extend(args)
            partial_kwargs = {
                k: variables.ConstantVariable(v) for k, v in self.value.keywords.items()
            }
            partial_kwargs.update(kwargs)
            return variables.TorchVariable(self.value.func, **options).call_function(
                tx, partial_args, partial_kwargs
            )
        elif callable(self.value):
            self.add_guard(self.source.make_guard(GuardBuilder.FUNCTION_MATCH))
            return self.call_method(tx, "__call__", args, kwargs)

        return super().call_function(tx, args, kwargs)

    def _check_for_getattribute(self):
        if object_has_getattribute(self.value):
            unimplemented("UserDefinedObjectVariable with custom __getattribute__")

    def _check_for_getattr(self):
        return get_custom_getattr(self.value)

    def _getattr_static(self, name):
        if (
            isinstance(self.value, torch.nn.Module)
            or "__slots__" in self.value.__class__.__dict__
        ):
            # getattr_static doesn't work on these
            subobj = getattr(self.value, name)
        else:
            subobj = inspect.getattr_static(self.value, name)
        return subobj

    def var_getattr(self, tx, name):
        from . import ConstantVariable
        from .builder import VariableBuilder

        options = VariableTracker.propagate(self)
        value = self.value
        source = AttrSource(self.source, name) if self.source else None
        self._check_for_getattribute()
        getattr_fn = self._check_for_getattr()

        try:
            subobj = self._getattr_static(name)
        except AttributeError:
            subobj = None
            if isinstance(getattr_fn, types.FunctionType):
                return variables.UserMethodVariable(
                    getattr_fn, self, source=source, **options
                ).call_function(tx, [ConstantVariable(name)], {})
            elif getattr_fn is not None:
                unimplemented("UserDefined with non-function __getattr__")

        if isinstance(subobj, property):
            return variables.UserMethodVariable(
                subobj.fget, self, source=source, **options
            ).call_function(tx, [], {})
        elif isinstance(subobj, staticmethod):
            return variables.UserFunctionVariable(
                subobj.__get__(self.value), source=source, **options
            )
        elif isinstance(subobj, classmethod):
            return variables.UserMethodVariable(
                subobj.__func__, self, source=source, **options
            )
        elif isinstance(subobj, types.FunctionType):
            # Check `__dict__` to bypass the function descriptor protocol to
            # accurately check for static method
            is_staticmethod = name in type(self.value).__dict__ and isinstance(
                type(self.value).__dict__[name], staticmethod
            )
            if is_staticmethod:
                # Use `UserFunctionVariable` to avoid doubly passing in `self`
                # as an argument, which happens if using `UserMethodVariable`
                return variables.UserFunctionVariable(
                    subobj, name, source=source, **options
                )
            return variables.UserMethodVariable(subobj, self, source=source, **options)

        if (
            name in getattr(value, "__dict__", {})
            or ConstantVariable.is_literal(subobj)
            or isinstance(
                subobj,
                (
                    torch.Tensor,
                    torch.nn.Module,
                ),
            )
        ):
            if source:
                return VariableBuilder(tx, source)(subobj).add_options(options)
            elif ConstantVariable.is_literal(subobj):
                return ConstantVariable(subobj, **options)

        if (
            name not in getattr(value, "__dict__", {})
            and type(value).__module__.startswith("torch.")
            and "torch.optim" not in type(value).__module__
            and not callable(value)
        ):
            if not source:
                assert getattr(
                    importlib.import_module(type(value).__module__),
                    type(value).__name__,
                ) is type(value)
                source = AttrSource(
                    AttrSource(
                        tx.import_source(type(value).__module__), type(value).__name__
                    ),
                    name,
                )

            return VariableBuilder(tx, source)(subobj).add_options(options)
        options["source"] = source
        if isinstance(
            subobj,
            (
                torch.distributions.constraints._Interval,
                torch.distributions.constraints._Real,
                torch.distributions.constraints.Constraint,
            ),
        ):
            return UserDefinedObjectVariable(subobj, **options)
        elif isinstance(self.value, torch.nn.Module) and name in all_hook_names:
            assert isinstance(subobj, collections.OrderedDict)
            if not subobj:
                return variables.ConstDictVariable(
                    subobj, collections.OrderedDict, **options
                )

        if name == "__class__":
            return UserDefinedClassVariable(type(self.value), **options)

        return variables.GetAttrVariable(self, name, **options)

    def call_hasattr(self, tx, name: str) -> "VariableTracker":
        if tx.output.side_effects.is_attribute_mutation(self):
            try:
                result = tx.output.side_effects.load_attr(self, name, deleted_ok=True)
                return variables.ConstantVariable(
                    not isinstance(result, variables.DeletedVariable)
                ).add_options(self, result)
            except KeyError:
                pass
        if not self.source:
            unimplemented("hasattr no source")
        options = VariableTracker.propagate(self)
        options["guards"].add(
            AttrSource(self.source, name).make_guard(GuardBuilder.HASATTR)
        )
        if self._check_for_getattribute() or self._check_for_getattr():
            unimplemented("hasattr with custom __getattr__")

        try:
            self._getattr_static(name)
            return variables.ConstantVariable(True, **options)
        except AttributeError:
            return variables.ConstantVariable(False, **options)

    def odict_getitem(self, tx, key):
        from .builder import VariableBuilder

        return VariableBuilder(
            tx,
            ODictGetItemSource(self.source, key.as_python_constant()),
        )(
            collections.OrderedDict.__getitem__(self.value, key.as_python_constant())
        ).add_options(
            key, self
        )


class ProcessGroupVariable(UserDefinedObjectVariable):
    """
    We don't want a ProcessGroup object to end up in our output graph.

    But it's common for dynamo to intercept a PG that is then used to get info like
    rank() or world_size(), as well as passed to utility functions in distributed_c10d
    which desugar it into plain types like a ranklist and tag.

    For convenience and proper guarding, we construct a variable type.

    TODO: make it possible to use ProcessGroupVariable as input to simple functions
          like _expand_group without dynamo complaining about making a proxy for it.
          It is not a tensor-like type, and we don't want a proxy- but dynamo assumes
          torch library functions are dealing with tensor-like types and would have proxies
          for their args.
    TODO: should we make this inherit VT instead of UDOV? Do we want any of the default behaviors
          or just graph-break whenever one of our special cases is not hit?
    """

    def __init__(self, value, **kwargs):
        super().__init__(value, **kwargs)

    def as_python_constant(self):
        return self.value

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if name == "rank":
            return variables.ConstantVariable(self.value.rank())
        if name == "size":
            return variables.ConstantVariable(self.value.size())

        # TODO should this just raise unimplemented?
        return super().call_method(tx, name, args, kwargs)

    def var_getattr(self, tx, name):
        if name in ["rank", "size"]:
            return variables.LambdaVariable(
                lambda *args, **kwargs: self.call_method(tx, name, args, kwargs)
            ).add_options(self)
        # TODO should this just raise unimplemented?
        return super().var_getattr(tx, name)

    @staticmethod
    def is_process_group(value):
        # we can't rely on importing/accessing torch distributed, it is not always built.
        if torch.distributed.is_available():
            from torch._C._distributed_c10d import ProcessGroup

            return istype(value, ProcessGroup)
        return False
