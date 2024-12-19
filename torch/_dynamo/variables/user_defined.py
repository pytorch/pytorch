# mypy: ignore-errors

import collections
import contextlib
import dataclasses
import enum
import functools
import inspect
import itertools
import random
import sys
import threading
import types
import warnings
import weakref
from typing import Dict, Generic, List, TYPE_CHECKING
from typing_extensions import is_typeddict

import torch._dynamo.config
import torch.nn
from torch._guards import TracingContext
from torch.utils._python_dispatch import is_traceable_wrapper_subclass_type

from .. import polyfills, variables
from ..bytecode_transformation import create_call_function
from ..create_parameter_op import do_not_convert_to_tracable_parameter
from ..exc import (
    handle_observed_exception,
    ObservedAttributeError,
    raise_observed_exception,
    unimplemented,
)
from ..guards import GuardBuilder, install_guard
from ..source import (
    AttrSource,
    GetItemSource,
    ODictGetItemSource,
    RandomValueSource,
    UnspecializedParamBufferSource,
)
from ..utils import (
    build_checkpoint_variable,
    build_invoke_subgraph_variable,
    check_constant_args,
    get_custom_getattr,
    has_torch_function,
    is_frozen_dataclass,
    is_invoke_subgraph,
    is_namedtuple_cls,
    is_utils_checkpoint,
    is_wrapper_or_member_descriptor,
    istype,
    namedtuple_fields,
    object_has_getattribute,
    proxy_args_kwargs,
    tensortype_to_dtype,
    unpatched_nn_module_getattr,
)
from .base import AttributeMutationExisting, ValueMutationNew, VariableTracker
from .dicts import DefaultDictVariable


try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    from torch.utils._cxx_pytree import PyTreeSpec
except ImportError:
    PyTreeSpec = type(None)


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator


def is_standard_setattr(val):
    return val in (object.__setattr__,)


def is_forbidden_context_manager(ctx):
    f_ctxs = []

    try:
        from _pytest.python_api import RaisesContext
        from _pytest.recwarn import WarningsChecker

        f_ctxs.append(RaisesContext)
        f_ctxs.append(WarningsChecker)
    except ImportError:
        pass

    if m := sys.modules.get("torch.testing._internal.jit_utils"):
        f_ctxs.append(m._AssertRaisesRegexWithHighlightContext)

    return ctx in f_ctxs


class UserDefinedVariable(VariableTracker):
    pass


class UserDefinedClassVariable(UserDefinedVariable):
    def __init__(self, value, **kwargs) -> None:
        super().__init__(**kwargs)
        self.value = value

    def as_python_constant(self):
        return self.value

    def as_proxy(self):
        return self.value

    def __repr__(self) -> str:
        return f"UserDefinedClassVariable({self.value})"

    @staticmethod
    @functools.lru_cache(None)
    def _constant_fold_classes():
        return {
            torch.device,
            torch.finfo,
            torch.iinfo,
            torch.Size,
        }

    @staticmethod
    @functools.lru_cache(None)
    def _in_graph_classes():
        _in_graph_class_list = {
            torch.Tensor,
            torch.cuda.Stream,
            torch.cuda.Event,
        }
        if hasattr(torch, "hpu"):
            _in_graph_class_list.update(
                {
                    torch.hpu.Stream,
                    torch.hpu.Event,
                }
            )

        return set(tensortype_to_dtype.keys()) | _in_graph_class_list

    def can_constant_fold_through(self):
        return self.value in self._constant_fold_classes()

    def has_key_in_generic_dict(self, tx: "InstructionTranslator", key):
        if tx.output.side_effects.has_pending_mutation_of_attr(self, key):
            mutated_attr = tx.output.side_effects.load_attr(self, key, deleted_ok=True)
            return not isinstance(mutated_attr, variables.DeletedVariable)

        return key in self.value.__dict__

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> "VariableTracker":
        from . import ConstantVariable, EnumVariable

        source = AttrSource(self.source, name) if self.source is not None else None

        if name == "__name__":
            return ConstantVariable.create(self.value.__name__)
        elif name == "__qualname__":
            return ConstantVariable.create(self.value.__qualname__)
        elif name == "__dict__":
            options = {"source": source}
            return variables.GetAttrVariable(self, name, **options)

        # Special handling of collections.OrderedDict.fromkeys()
        # Wrap it as GetAttrVariable(collections.OrderedDict, "fromkeys") to make it consistent with
        # collections.defaultdict, and both will be handled at UserDefinedClassVariable.call_method().
        # Otherwise, it would be wrapped as UserDefinedObjectVariable(collections.OrderedDict.fromkeys),
        # and we need duplicate code to handle both cases.
        if (
            self.value in {collections.OrderedDict, collections.defaultdict}
            and name == "fromkeys"
        ):
            return super().var_getattr(tx, name)

        try:
            obj = inspect.getattr_static(self.value, name)
        except AttributeError:
            obj = None

        if isinstance(obj, staticmethod):
            return VariableTracker.build(tx, obj.__get__(self.value), source)
        elif isinstance(obj, classmethod):
            if isinstance(obj.__func__, property):
                return variables.UserFunctionVariable(obj.__func__.fget).call_function(
                    tx, [self], {}
                )
            return variables.UserMethodVariable(obj.__func__, self, source=source)
        elif isinstance(obj, types.ClassMethodDescriptorType):
            # e.g.: inspect.getattr_static(dict, "fromkeys")
            #       inspect.getattr_static(itertools.chain, "from_iterable")
            func = obj.__get__(None, self.value)
            return VariableTracker.build(tx, func, source)
        elif source:
            # __mro__ is a member in < 3.12, an attribute in >= 3.12
            if inspect.ismemberdescriptor(obj) or (
                sys.version_info >= (3, 12) and name == "__mro__"
            ):
                return VariableTracker.build(tx, obj.__get__(self.value), source)

        if ConstantVariable.is_literal(obj):
            return ConstantVariable.create(obj)
        elif isinstance(obj, enum.Enum):
            return EnumVariable(obj)
        elif name in getattr(self.value, "__dict__", {}) or (
            self.value.__module__.startswith("torch.")
            or self.value.__module__ == "torch"
        ):
            if source:
                return VariableTracker.build(tx, obj, source)

        if (
            source
            and not inspect.ismethoddescriptor(obj)
            and not is_wrapper_or_member_descriptor(obj)
        ):
            return VariableTracker.build(tx, obj, source)

        return super().var_getattr(tx, name)

    def _call_cross_entropy_loss(self, tx: "InstructionTranslator", args, kwargs):
        """
        functional: input, target, weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean',
        label_smoothing=0.0

        non functional ctor: weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean',
        label_smoothing=0.0

        non functional loss call: input, target, optional_output
        """
        from . import ConstantVariable

        def normalize_args(
            weight=ConstantVariable.create(None),
            size_average=ConstantVariable.create(None),
            ignore_index=ConstantVariable.create(-100),
            reduce=ConstantVariable.create(None),
            reduction=ConstantVariable.create("mean"),
            label_smoothing=ConstantVariable.create(0.0),
        ):
            return (
                weight,
                size_average,
                ignore_index,
                reduce,
                reduction,
                label_smoothing,
            )

        (
            weight,
            size_average,
            ignore_index,
            reduce_arg,
            reduction,
            label_smoothing,
        ) = normalize_args(*args, **kwargs)

        def fake_cross_entropy_loss(input, target):
            from .builder import wrap_fx_proxy

            return wrap_fx_proxy(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    torch.nn.functional.cross_entropy,
                    *proxy_args_kwargs(
                        [
                            input,
                            target,
                            weight,
                            size_average,
                            ignore_index,
                            reduce_arg,
                            reduction,
                            label_smoothing,
                        ],
                        {},
                    ),
                ),
            )

        return variables.LambdaVariable(fake_cross_entropy_loss)

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
            options = {"mutation_type": ValueMutationNew()}
            subs_as_vars: List[VariableTracker] = []
            for sub in self.value.__subclasses__():
                source = AttrSource(tx.import_source(sub.__module__), sub.__name__)
                subs_as_vars.append(
                    variables.UserDefinedClassVariable(sub, source=source)
                )

            return variables.ListVariable(subs_as_vars, **options)
        elif (
            self.value in {collections.OrderedDict, collections.defaultdict}
            and name == "fromkeys"
        ):
            from .builtin import BuiltinVariable

            return BuiltinVariable.call_custom_dict_fromkeys(
                tx, self.value, *args, **kwargs
            )
        elif name == "__eq__" and len(args) == 1 and hasattr(args[0], "value"):
            return variables.ConstantVariable(self.value == args[0].value)
        elif name == "__ne__" and len(args) == 1 and hasattr(args[0], "value"):
            return variables.ConstantVariable(self.value != args[0].value)

        return super().call_method(tx, name, args, kwargs)

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from ..side_effects import SideEffects
        from .builder import wrap_fx_proxy
        from .builtin import BuiltinVariable

        constant_args = check_constant_args(args, kwargs)

        if self.can_constant_fold_through() and constant_args:
            # constant fold
            return variables.ConstantVariable.create(
                self.as_python_constant()(
                    *[x.as_python_constant() for x in args],
                    **{k: v.as_python_constant() for k, v in kwargs.items()},
                ),
            )
        elif self.value is torch.nn.CrossEntropyLoss:
            return self._call_cross_entropy_loss(tx, args, kwargs)
        elif self.value is contextlib.nullcontext:
            # import here to avoid circular dependency
            from .ctx_manager import NullContextVariable

            return NullContextVariable()
        elif self.value is collections.OrderedDict:
            return BuiltinVariable.call_custom_dict(
                tx, collections.OrderedDict, *args, **kwargs
            )
        elif (
            self.value is collections.defaultdict
            and len(args) <= 1
            and DefaultDictVariable.is_supported_arg(args[0])
        ):
            return DefaultDictVariable(
                {},
                collections.defaultdict,
                args[0],
                mutation_type=ValueMutationNew(),
            )
        elif is_typeddict(self.value):
            if self.value.__optional_keys__:
                unimplemented("TypedDict with optional keys not supported")
            return variables.BuiltinVariable(dict).call_dict(tx, *args, **kwargs)
        elif self.value is collections.deque:
            maxlen = variables.ConstantVariable.create(None)
            if not kwargs:
                if len(args) == 0:
                    items = []
                elif len(args) == 1 and args[0].has_force_unpack_var_sequence(tx):
                    items = args[0].force_unpack_var_sequence(tx)
                elif len(args) == 2 and args[0].has_force_unpack_var_sequence(tx):
                    items = args[0].force_unpack_var_sequence(tx)
                    maxlen = args[1]
                else:
                    unimplemented("deque() with more than 2 arg not supported")
            elif tuple(kwargs) == ("maxlen",):
                maxlen = kwargs["maxlen"]
                if len(args) == 0:
                    items = []
                if len(args) == 1 and args[0].has_force_unpack_var_sequence(tx):
                    items = args[0].force_unpack_var_sequence(tx)
                else:
                    unimplemented("deque() with more than 1 arg not supported")
            else:
                unimplemented("deque() with invalid kwargs not supported")
            return variables.lists.DequeVariable(
                items, maxlen=maxlen, mutation_type=ValueMutationNew()
            )
        elif self.value is weakref.ref:
            return variables.WeakRefVariable(args[0])
        elif self.value is functools.partial:
            if not args:
                unimplemented("functools.partial malformed")
            # The first arg, a callable (the ctor below will assert on types)
            fn = args[0]
            rest_args = args[1:]
            # guards for the produced FunctoolsPartialVariable are installed in FunctoolsPartialVariable ctor from the
            # args and keywords
            return variables.functions.FunctoolsPartialVariable(
                fn, args=rest_args, keywords=kwargs
            )
        elif self.value is warnings.catch_warnings and not args:
            return variables.CatchWarningsCtxManagerVariable.create(tx, kwargs)
        elif self.value is torch.cuda.device and not kwargs and len(args) == 1:
            assert args[0].is_python_constant()
            return variables.CUDADeviceVariable.create(tx, args[0].as_python_constant())
        elif (
            issubclass(type(self.value), type)
            and hasattr(
                self.value, "__enter__"
            )  # TODO(voz): These can invoke user code!
            and hasattr(
                self.value, "__exit__"
            )  # TODO(voz): These can invoke user code!
            and self.is_standard_new()
            and SideEffects.cls_supports_mutation_side_effects(self.value)
            and self.source
            and not is_forbidden_context_manager(self.value)
        ):
            from torch.overrides import TorchFunctionMode

            from .ctx_manager import GenericContextWrappingVariable
            from .torch_function import TorchFunctionModeVariable

            if issubclass(
                self.value, TorchFunctionMode
            ) and TorchFunctionModeVariable.is_supported_torch_function_mode(
                self.value
            ):
                var_cls = TorchFunctionModeVariable
            else:
                var_cls = GenericContextWrappingVariable

            cm_obj = tx.output.side_effects.track_object_new(
                self.source, self.value, var_cls, {}
            )
            cm_obj.call_method(tx, "__init__", args, kwargs)
            return cm_obj
        elif is_namedtuple_cls(self.value):
            fields = namedtuple_fields(self.value)
            # check if this a quasi-namedtuple or a real one
            if self.value.__module__ == "torch.return_types":
                assert len(args) == 1
                assert not kwargs
                items = args[0].force_unpack_var_sequence(tx)
            else:
                field_defaults = self.value._field_defaults

                items = list(args)
                items.extend([None] * (len(fields) - len(items)))

                var_tracker_kwargs = {}
                for field_name, var_tracker in zip(fields, items):
                    if var_tracker is None:
                        if field_name in kwargs:
                            field_var = kwargs[field_name]
                        else:
                            assert field_name in field_defaults
                            field_var = VariableTracker.build(
                                tx, field_defaults[field_name]
                            )
                        var_tracker_kwargs[field_name] = field_var

                for name, value in var_tracker_kwargs.items():
                    assert name in fields
                    items[fields.index(name)] = value

                assert all(x is not None for x in items)

            return variables.NamedTupleVariable(items, self.value)
        elif is_frozen_dataclass(self.value) and self.is_standard_new():
            fields = dataclasses.fields(self.value)
            items = list(args)
            items.extend([None] * (len(fields) - len(items)))

            default_kwargs = {}
            for field, var_tracker in zip(fields, items):
                if var_tracker is None:
                    if field.name in kwargs:
                        var_tracker = kwargs[field.name]
                    else:
                        if not field.init:
                            continue

                        if field.default is not dataclasses.MISSING:
                            var_tracker = VariableTracker.build(tx, field.default)
                        elif field.default_factory is not dataclasses.MISSING:
                            factory_fn = VariableTracker.build(
                                tx, field.default_factory
                            )
                            var_tracker = factory_fn.call_function(tx, [], {})
                        else:
                            # if we are subclass, the constructor could possibly
                            # be missing args
                            continue

                    default_kwargs[field.name] = var_tracker
            kwargs.update(default_kwargs)

            var = tx.output.side_effects.track_object_new_from_user_defined_class(self)
            var.call_method(tx, "__init__", args, kwargs)
            return var
        elif (
            self.is_standard_new()
            and SideEffects.cls_supports_mutation_side_effects(self.value)
            and self.source
        ):
            var = tx.output.side_effects.track_object_new_from_user_defined_class(self)
            with do_not_convert_to_tracable_parameter():
                var.call_method(tx, "__init__", args, kwargs)
                return var
        elif variables.CustomizedDictVariable.is_matching_cls(self.value):
            options = {"mutation_type": ValueMutationNew()}
            return variables.CustomizedDictVariable.create(
                self.value, args, kwargs, options
            )
        elif (
            variables.RestrictedListSubclassVariable.is_matching_cls(self.value)
            and self.source
        ):
            return variables.RestrictedListSubclassVariable(
                variables.BuiltinVariable(list).call_function(tx, args, kwargs).items,
                user_cls=self.value,
                user_cls_source=self.source,
                mutation_type=ValueMutationNew(),
            )
        elif (
            self.value in self._in_graph_classes()
            or is_traceable_wrapper_subclass_type(self.value)
        ):
            # torch.LongTensor cannot accept a list of FakeTensors.
            # So we stack the list of FakeTensors instead.
            if (
                np
                and self.value in tensortype_to_dtype
                and len(args) == 1
                and isinstance(args[0], variables.ListVariable)
                and len(args[0].items) > 1
                and all(isinstance(x, variables.TensorVariable) for x in args[0].items)
            ):
                # Stack FakeTensor
                stacked = wrap_fx_proxy(
                    tx=tx,
                    proxy=tx.output.create_proxy(
                        "call_function",
                        torch.stack,
                        *proxy_args_kwargs(args, kwargs),
                    ),
                )
                args = [stacked]

            tensor_variable = wrap_fx_proxy(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    self.value,
                    *proxy_args_kwargs(args, kwargs),
                ),
            )

            return tensor_variable
        elif issubclass(self.value, enum.Enum) and len(args) == 1 and not kwargs:
            options = {"mutation_type": ValueMutationNew()}
            return variables.EnumVariable.create(self.value, args[0], options)
        elif self.value is random.Random:
            if len(args) == 1 and isinstance(args[0], variables.ConstantVariable):
                seed = args[0].value
            else:
                seed = None
            random_object = random.Random(seed)
            return RandomVariable(random_object)
        elif (
            not self.is_standard_new()
            and SideEffects.cls_supports_mutation_side_effects(self.value)
            and self.source
        ):
            return tx.inline_user_function_return(
                VariableTracker.build(
                    tx, polyfills.instantiate_user_defined_class_object
                ),
                [self, *args],
                kwargs,
            )
        elif (
            inspect.getattr_static(self.value, "__new__") is dict.__new__
            and inspect.getattr_static(self.value, "__getattribute__")
            is dict.__getattribute__
            and self.source
        ):
            var = tx.output.side_effects.track_object_new_from_user_defined_class(self)
            var.call_method(tx, "__init__", args, kwargs)
            return var

        return super().call_function(tx, args, kwargs)

    def is_standard_new(self):
        """Check for __new__ being overridden"""
        new_fn = inspect.getattr_static(self.value, "__new__", None)
        if isinstance(new_fn, staticmethod):
            new_fn = new_fn.__func__
        return new_fn in (object.__new__, Generic.__new__)

    def call_hasattr(self, tx: "InstructionTranslator", name: str) -> "VariableTracker":
        if self.source:
            source = AttrSource(self.source, name)
            install_guard(source.make_guard(GuardBuilder.HASATTR))
            return variables.ConstantVariable(hasattr(self.value, name))
        return super().call_hasattr(tx, name)

    def const_getattr(self, tx: "InstructionTranslator", name):
        if name == "__name__":
            return self.value.__name__
        return super().const_getattr(tx, name)


class NO_SUCH_SUBOBJ:
    pass


def call_random_fn(tx, fn, args, kwargs):
    from .builder import VariableBuilder

    args = [x.as_python_constant() for x in args]
    kwargs = {k: v.as_python_constant() for k, v in kwargs.items()}
    random_call_index = len(tx.output.random_calls)
    example_value = fn(*args, **kwargs)
    source = RandomValueSource(random_call_index)
    tx.output.random_calls.append((fn, args, kwargs))
    # TODO: arguably, this should route to wrap_symint/wrap_symfloat
    # (currently hypothetical), but I'm not going to poke my hand in
    # this nest for now
    return VariableBuilder(tx, source).wrap_unspecialized_primitive(example_value)


class UserDefinedObjectVariable(UserDefinedVariable):
    """
    Mostly objects of defined type.  Catch-all for something where we only know the type.
    """

    _nonvar_fields = {"value", "value_type", *UserDefinedVariable._nonvar_fields}

    def __init__(self, value, value_type=None, cls_source=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.value = value
        self.value_type = value_type or type(value)
        assert type(value) is self.value_type
        # This is used with __new__, when the new object is sourceless but the user class can be sourceful.
        self.cls_source = cls_source

    def __str__(self) -> str:
        inner = self.value_type.__name__
        if inner in [
            "builtin_function_or_method",
            "getset_descriptor",
            "method_descriptor",
            "method",
        ]:
            inner = str(getattr(self.value, "__name__", None))
        return f"{self.__class__.__name__}({inner})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value_type.__name__})"

    def python_type(self):
        return self.value_type

    def guard_as_python_constant(self):
        if self.source:
            install_guard(self.source.make_guard(GuardBuilder.ID_MATCH))
            return self.value
        return super().guard_as_python_constant()

    def torch_function_check(self):
        assert has_torch_function(
            self
        ), f"calling torch function on object without __torch_function__ {self}"

    def get_torch_fn(self, tx):
        self.torch_function_check()
        from .torch_function import build_torch_function_fn

        return build_torch_function_fn(tx, self.value, self.source)

    def call_torch_function(self, tx: "InstructionTranslator", fn, types, args, kwargs):
        self.torch_function_check()

        from .torch_function import _get_subclass_type_var, call_torch_function

        return call_torch_function(
            tx,
            _get_subclass_type_var(tx, self),
            self.get_torch_fn(tx),
            fn,
            types,
            args,
            kwargs,
        )

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

    def _maybe_get_baseclass_method(self, name):
        if name not in getattr(self.value, "__dict__", {}):
            try:
                return inspect.getattr_static(type(self.value), name)
            except AttributeError:
                pass
        return None

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
            TupleVariable,
            UserMethodVariable,
        )

        method = self._maybe_get_baseclass_method(name)
        if method is not None:
            if method is object.__init__:
                return ConstantVariable.create(None)

            if is_standard_setattr(method) or isinstance(self.value, threading.local):
                return self.method_setattr_standard(tx, *args, **kwargs)

            # [NOTE] OrderedDict, dict subtypes must always have source
            # We cannot instantiate such subtypes in-graph due to builtin __new__
            if method is collections.OrderedDict.keys:
                # subclass of OrderedDict
                assert not (args or kwargs)
                assert self.source  # OrderedDict, dict subtypes must always have source
                keys = list(self.value.keys())
                assert all(map(ConstantVariable.is_literal, keys))
                install_guard(self.source.make_guard(GuardBuilder.DICT_CONST_KEYS))
                tx.output.guard_on_key_order.add(self.source.name())
                return TupleVariable([ConstantVariable.create(k) for k in keys])

            if (
                method in (collections.OrderedDict.__contains__, dict.__contains__)
                and len(args) == 1
                and isinstance(args[0], (ConstantVariable, BuiltinVariable))
                and inspect.getattr_static(type(self.value), "keys")
                in (collections.OrderedDict.keys, dict.keys)
            ):
                assert not kwargs
                assert self.source  # OrderedDict, dict subtypes must always have source

                # TODO(anijain2305) - Why do we need to guard on all keys?
                install_guard(self.source.make_guard(GuardBuilder.DICT_CONST_KEYS))
                return ConstantVariable.create(
                    args[0].as_python_constant() in self.value
                )

            if method is collections.OrderedDict.items and isinstance(
                self.value, collections.OrderedDict
            ):
                assert self.source  # OrderedDict, dict subtypes must always have source
                assert not (args or kwargs)
                keys = self.call_method(tx, "keys", [], {})
                items = [
                    TupleVariable(
                        [key, self.odict_getitem(tx, key)],
                    )
                    for key in keys.force_unpack_var_sequence(tx)
                ]
                tx.output.guard_on_key_order.add(self.source.name())
                return TupleVariable(items)

            if method is collections.OrderedDict.__getitem__ and len(args) == 1:
                assert not kwargs
                assert self.source  # OrderedDict, dict subtypes must always have source
                return self.odict_getitem(tx, args[0])

            if len(args) == 1 and not kwargs:
                if method is object.__eq__:
                    func_var = VariableTracker.build(tx, polyfills.object_eq)
                    return func_var.call_function(tx, [self, *args], kwargs)

                if method is object.__ne__:
                    func_var = VariableTracker.build(tx, polyfills.object_ne)
                    return func_var.call_function(tx, [self, *args], kwargs)

            # check for methods implemented in C++
            if isinstance(method, types.FunctionType):
                source = (
                    None
                    if self.source is None
                    else AttrSource(AttrSource(self.source, "__class__"), name)
                )
                # TODO(jansel): add a guard to check for monkey patching?
                from ..mutation_guard import unpatched_nn_module_init

                if method is torch.nn.Module.__init__:
                    method = unpatched_nn_module_init
                return UserMethodVariable(method, self, source=source).call_function(
                    tx, args, kwargs
                )

            if method is list.__len__ and self.source and not (args or kwargs):
                install_guard(self.source.make_guard(GuardBuilder.SEQUENCE_LENGTH))
                return ConstantVariable(len(self.value))

        return super().call_method(tx, name, args, kwargs)

    def method_setattr_standard(self, tx: "InstructionTranslator", name, value):
        try:
            name = name.as_python_constant()
        except NotImplementedError:
            unimplemented(f"non-const setattr name: {name}")
        if not tx.output.side_effects.is_attribute_mutation(self):
            unimplemented(f"setattr({self}, {name}, ...)")

        tx.output.side_effects.store_attr(self, name, value)
        return variables.ConstantVariable(None)

    def needs_slow_setattr(self):
        return not is_standard_setattr(
            inspect.getattr_static(self.value, "__setattr__", None)
        ) and not isinstance(self.value, threading.local)

    def unpack_var_sequence(self, tx):
        if (
            self.source
            and self._maybe_get_baseclass_method("__iter__") is list.__iter__
            and self._maybe_get_baseclass_method("__len__") is list.__len__
            and self._maybe_get_baseclass_method("__getitem__") is list.__getitem__
        ):
            install_guard(self.source.make_guard(GuardBuilder.SEQUENCE_LENGTH))
            return [
                variables.LazyVariableTracker.create(
                    self.value[k],
                    source=GetItemSource(self.source, k),
                )
                for k in range(len(self.value))
            ]
        return super().unpack_var_sequence(tx)

    def next_variable(self, tx):
        return self.call_method(tx, "__next__", [], {})

    def is_supported_random(self):
        try:
            return self.value in self._supported_random_functions()
        except TypeError:
            # TypeError: unhashable type
            return False

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from .. import trace_rules

        if (
            self.is_supported_random()
            and all(k.is_python_constant() for k in args)
            and all(v.is_python_constant() for v in kwargs.values())
        ):
            return call_random_fn(tx, self.value, args, kwargs)
        elif istype(self.value, types.MethodType):
            func = self.value.__func__
            obj = self.value.__self__
            if (
                func is torch.utils._contextlib._DecoratorContextManager.clone
                and variables.TorchCtxManagerClassVariable.is_matching_cls(
                    obj.__class__
                )
                and not (args or kwargs)
            ):
                return variables.TorchCtxManagerClassVariable(
                    obj.__class__
                ).call_function(tx, args, kwargs)

            if (
                func is torch.autograd.grad_mode.inference_mode.clone
                and obj.__class__ is torch.autograd.grad_mode.inference_mode
            ):
                # simulate the inference_mode.clone implementation
                var = variables.ConstantVariable(obj.mode)
                return variables.TorchCtxManagerClassVariable(
                    obj.__class__
                ).call_function(tx, [var], kwargs)

            if self.source is None:
                unimplemented(
                    "Sourceless UserDefinedObjectVariable method not supported"
                )
            func_src = AttrSource(self.source, "__func__")
            func_var = VariableTracker.build(tx, func, func_src)
            obj_src = AttrSource(self.source, "__self__")
            obj_var = VariableTracker.build(tx, obj, obj_src)
            return func_var.call_function(tx, [obj_var] + args, kwargs)
        elif (
            istype(self.value, functools.partial)
            and trace_rules.lookup(self.value.func)
            == variables.TorchInGraphFunctionVariable
            and all(
                variables.ConstantVariable.is_literal(v)
                for v in itertools.chain(self.value.args, self.value.keywords.values())
            )
        ):
            if self.source:
                install_guard(
                    AttrSource(self.source, "func").make_guard(GuardBuilder.ID_MATCH),
                    AttrSource(self.source, "args").make_guard(
                        GuardBuilder.CONSTANT_MATCH
                    ),
                    AttrSource(self.source, "keywords").make_guard(
                        GuardBuilder.CONSTANT_MATCH
                    ),
                )

            partial_args = [
                variables.ConstantVariable.create(v) for v in self.value.args
            ]
            partial_args.extend(args)
            partial_kwargs = {
                k: variables.ConstantVariable.create(v)
                for k, v in self.value.keywords.items()
            }
            partial_kwargs.update(kwargs)

            # TODO(dynamo-team) - Consider calling VariableBuilder directly here
            if is_utils_checkpoint(self.value.func):
                return build_checkpoint_variable().call_function(
                    tx, partial_args, partial_kwargs
                )
            elif is_invoke_subgraph(self.value.func):
                return build_invoke_subgraph_variable().call_function(
                    tx, partial_args, partial_kwargs
                )
            return variables.TorchInGraphFunctionVariable(
                self.value.func
            ).call_function(tx, partial_args, partial_kwargs)
        elif callable(self.value):
            if self.source:
                install_guard(self.source.make_guard(GuardBuilder.FUNCTION_MATCH))
            return self.call_method(tx, "__call__", args, kwargs)

        return super().call_function(tx, args, kwargs)

    def _check_for_getattribute(self):
        if object_has_getattribute(self.value):
            unimplemented("UserDefinedObjectVariable with custom __getattribute__")

    def _check_for_getattr(self):
        return get_custom_getattr(self.value)

    def _is_c_defined_property(self, subobj):
        if not isinstance(subobj, property):
            return False

        # pybind def_readwrite is implemented via PyCFunction. At the python level, it is visible as a property whose
        # fget is an instancemethod wrapper - https://docs.python.org/3/c-api/method.html#c.PyInstanceMethod_Check

        # If we have a PyCFunction, we make an assumption that there is no side effect.
        return isinstance(
            subobj.fget, types.BuiltinFunctionType
        ) or torch._C._dynamo.utils.is_instancemethod(subobj.fget)

    def _getattr_static(self, name):
        subobj = inspect.getattr_static(self.value, name, NO_SUCH_SUBOBJ)
        import _collections

        # In some cases, we have to do dynamic lookup because getattr_static is not enough. For example, threading.local
        # has side-effect free __getattribute__ and the attribute is not visible without a dynamic lookup.
        if (
            subobj is NO_SUCH_SUBOBJ  # e.g., threading.local
            or isinstance(
                subobj, _collections._tuplegetter
            )  # namedtuple fields are represented by _tuplegetter
            or (
                inspect.ismemberdescriptor(subobj) and name in self.value.__slots__
            )  # handle memberdecriptor and slots
            or self._is_c_defined_property(subobj)
        ):
            # Call __getattribute__, we have already checked that this is not overridden and side-effect free. We don't
            # want to call getattr because it can be user-overridden.
            subobj = self.value.__getattribute__(name)

        return subobj

    def has_key_in_generic_dict(self, tx: "InstructionTranslator", key):
        self._check_for_getattribute()
        if tx.output.side_effects.has_pending_mutation_of_attr(self, key):
            mutated_attr = tx.output.side_effects.load_attr(self, key, deleted_ok=True)
            return not isinstance(mutated_attr, variables.DeletedVariable)

        return key in self.value.__dict__

    def get_source_by_walking_mro(self, name):
        assert self.cls_source is not None

        for idx, klass in enumerate(type(self.value).__mro__):
            if name in klass.__dict__:
                mro_source = AttrSource(self.cls_source, "__mro__")
                klass_source = GetItemSource(mro_source, idx)
                dict_source = AttrSource(klass_source, "__dict__")
                return GetItemSource(dict_source, name)

        unimplemented(f"Could not find {name} in {type(self.value).__mro__}")

    def var_getattr(self, tx: "InstructionTranslator", name):
        from .. import trace_rules
        from . import ConstantVariable

        source = AttrSource(self.source, name) if self.source else None
        self._check_for_getattribute()

        if tx.output.side_effects.has_pending_mutation_of_attr(self, name):
            result = tx.output.side_effects.load_attr(self, name, deleted_ok=True)
            if isinstance(result, variables.DeletedVariable):
                raise_observed_exception(AttributeError, tx)
            return result

        if name == "__dict__":
            options = {"source": source}
            return variables.GetAttrVariable(self, name, **options)

        # TODO(anijain2305) - Investigate if we need specialization for more
        # dunder attrs. inspect.getattr_static does not return correct value for
        # them.
        if name == "__class__":
            cls_source = source
            if cls_source is None:
                cls_source = self.cls_source
            options = {"source": cls_source}
            return UserDefinedClassVariable(type(self.value), **options)

        try:
            subobj = self._getattr_static(name)
        except AttributeError:
            subobj = NO_SUCH_SUBOBJ
            getattr_fn = self._check_for_getattr()
            if isinstance(getattr_fn, types.FunctionType):
                # Dynamo is going to trace the __getattr__ function with
                # args=name. Set the source accordingly.
                if (
                    getattr_fn is unpatched_nn_module_getattr
                    and isinstance(self, variables.UnspecializedNNModuleVariable)
                    # prevent against overwriting of params/buffers/submodules
                    and istype(self.value._parameters, dict)
                    and istype(self.value._buffers, dict)
                    and istype(self.value._modules, dict)
                ):
                    # Manually trace out the nn module __getattr__ to avoid large compilation latency.
                    out = self.manually_trace_nn_module_getattr(tx, name)
                else:
                    new_source = None
                    if self.source:
                        new_source = AttrSource(self.source, "__getattr__")
                    out = variables.UserMethodVariable(
                        getattr_fn, self, source=new_source
                    ).call_function(tx, [ConstantVariable.create(name)], {})

                if self.source and getattr_fn is torch.nn.Module.__getattr__:
                    if isinstance(
                        out,
                        (
                            variables.UnspecializedNNModuleVariable,
                            variables.NNModuleVariable,
                        ),
                    ):
                        # nn_module_stack source is BC surface area. Ensure that
                        # mod._modules["linear"] is reflected as mod.linear for
                        # nn_module_stack.
                        out.set_nn_module_stack_source(
                            AttrSource(self.get_nn_module_stack_source(), name)
                        )
                return out

            elif getattr_fn is not None:
                unimplemented("UserDefined with non-function __getattr__")

        if isinstance(subobj, property):
            if self.source:
                # Read the class attribute to reach the property
                source = AttrSource(AttrSource(self.source, "__class__"), name)
                # Get the getter function
                source = AttrSource(source, "fget")
            return variables.UserMethodVariable(
                subobj.fget, self, source=source
            ).call_function(tx, [], {})
        elif isinstance(subobj, staticmethod):
            func = subobj.__get__(self.value)
            if source is not None:
                return trace_rules.lookup(func).create_with_source(func, source=source)
            else:
                return trace_rules.lookup(func)(func)
        elif isinstance(subobj, classmethod):
            return variables.UserMethodVariable(
                subobj.__func__, self.var_getattr(tx, "__class__"), source=source
            )
        elif isinstance(subobj, types.ClassMethodDescriptorType):
            # e.g.: inspect.getattr_static({}, "fromkeys")
            func = subobj.__get__(self.value, None)
            return VariableTracker.build(tx, func, source)
        elif inspect.ismethoddescriptor(subobj) and not is_wrapper_or_member_descriptor(
            subobj.__get__
        ):
            # Attribute has a __get__ method. Create a user defined object vt
            # for the subobj, and then trace the __get__ method.
            descriptor_source = None
            descriptor_get_source = None
            if self.cls_source:
                # To access the method descriptor from the udf object w/o using
                # inspect.getattr_static, we can look into the class mro
                descriptor_source = self.get_source_by_walking_mro(name)
                descriptor_get_source = AttrSource(descriptor_source, "__get__")
                descriptor_var = VariableTracker.build(tx, subobj, descriptor_source)
            else:
                # Sourceless Builder does not support user defined objects
                descriptor_var = UserDefinedObjectVariable(subobj)

            # The arguments of the __get__ function are (self, instance, owner)
            # self - descriptor_var
            # instance - instance of the class, represented by self here
            # owner - class object
            owner_var = UserDefinedClassVariable(type(self.value))
            return variables.UserMethodVariable(
                subobj.__get__.__func__, descriptor_var, source=descriptor_get_source
            ).call_function(tx, [self, owner_var], {})
        elif isinstance(subobj, types.FunctionType) or (
            isinstance(subobj, types.MethodType)
            and isinstance(self.value, torch.nn.Module)
        ):
            # Since we get subobj via self._getattr_static, which may not trigger dynamic lookup.
            # Static lookup can't tell us it's a method or function correctly,
            # so we trigger dynamic lookup here to get the correct type.
            dynamic_subobj = getattr(self.value, name)

            while dynamic_subobj is subobj and hasattr(subobj, "_torchdynamo_inline"):
                subobj = subobj._torchdynamo_inline
                dynamic_subobj = subobj
                source = AttrSource(source, "_torchdynamo_inline") if source else None

            if isinstance(subobj, types.MethodType):
                if dynamic_subobj.__self__ is not self.value:
                    if not isinstance(dynamic_subobj.__func__, types.FunctionType):
                        unimplemented(
                            f"Found a method whose __func__ is not of FunctionType - {dynamic_subobj}"
                        )

                    from .builder import SourcelessUserDefinedObjectBuilder

                    # This means that we are calling a method of some other object here.
                    object_vt = SourcelessUserDefinedObjectBuilder.create(
                        tx, dynamic_subobj.__self__
                    )
                    return variables.UserMethodVariable(
                        dynamic_subobj.__func__, object_vt
                    )
                func = subobj.__func__
            else:
                assert isinstance(subobj, types.FunctionType)
                func = subobj

            if inspect.ismethod(dynamic_subobj):
                return variables.UserMethodVariable(func, self, source=source)
            elif inspect.isfunction(dynamic_subobj):
                if is_utils_checkpoint(func):
                    return build_checkpoint_variable(source=source)
                elif source is not None:
                    return trace_rules.lookup(func).create_with_source(
                        func, source=source
                    )
                else:
                    return trace_rules.lookup(func)(func)

        if (
            # wrap the source only if inline_inbuilt_nn_modules is set or fsdp modules. This is a temporary solution to
            # keep Dynamo behavior compatible with no inlining, as there will be some delay to turn on the flag in
            # fbcode.
            (
                torch._dynamo.config.inline_inbuilt_nn_modules
                or isinstance(self, variables.FSDPManagedNNModuleVariable)
            )
            and source
            and isinstance(self, variables.UnspecializedNNModuleVariable)
            # export has some awkwardness around specialized and unspecialized modules. Skip wrapping source for export
            # usecase for now.
            and not tx.output.export
        ):
            # Recalculate source for params/buffers
            if name in ("_buffers", "_parameters"):
                source = UnspecializedParamBufferSource(self.source, name)
            source = self._wrap_source(source)

        if subobj is not NO_SUCH_SUBOBJ:
            if is_wrapper_or_member_descriptor(subobj):
                options = {"source": source}
                return variables.GetAttrVariable(self, name, **options)
            if source:
                return variables.LazyVariableTracker.create(subobj, source)
            else:
                # Check if the subobj is accessible from the class itself. If the class source is known, we can create a
                # sourceful variable tracker.
                if self.cls_source is not None:
                    subobj_from_class = inspect.getattr_static(
                        self.value.__class__, name, NO_SUCH_SUBOBJ
                    )
                    if subobj_from_class is subobj:
                        src_from_class = AttrSource(self.cls_source, name)
                        return variables.LazyVariableTracker.create(
                            subobj_from_class, src_from_class
                        )

                return VariableTracker.build(tx, subobj)

        # Earlier we were returning GetAttrVariable but its incorrect. In absence of attr, Python raises AttributeError.
        raise_observed_exception(AttributeError, tx)

    def call_hasattr(self, tx: "InstructionTranslator", name: str) -> "VariableTracker":
        if self._check_for_getattribute():
            unimplemented("hasattr with custom __getattribute__")

        if self.source:
            install_guard(
                AttrSource(self.source, name).make_guard(GuardBuilder.HASATTR)
            )

        try:
            var_vt = self.var_getattr(tx, name)
            return variables.ConstantVariable.create(
                not isinstance(var_vt, variables.DeletedVariable)
            )
        except ObservedAttributeError:
            handle_observed_exception(tx)
            return variables.ConstantVariable.create(False)

    def odict_getitem(self, tx: "InstructionTranslator", key):
        from .dicts import is_hashable

        # TODO this should probably be merged with the dict handling

        index = (
            key.source
            if is_hashable(key) and key.source is not None
            else key.as_python_constant()
        )

        return VariableTracker.build(
            tx,
            collections.OrderedDict.__getitem__(self.value, key.as_python_constant()),
            self.source and ODictGetItemSource(self.source, index),
        )


class FrozenDataClassVariable(UserDefinedObjectVariable):
    @staticmethod
    def create(tx, value, source):
        from dataclasses import fields

        assert is_frozen_dataclass(value)

        field_map = {}
        for field in fields(value):
            if hasattr(value, field.name):
                field_map[field.name] = VariableTracker.build(
                    tx,
                    getattr(value, field.name),
                    source and AttrSource(source, field.name),
                )

        return FrozenDataClassVariable(value, fields=field_map, source=source)

    def __init__(self, value, fields=None, **kwargs) -> None:
        super().__init__(value, **kwargs)
        if fields is None:
            fields = {}
        self.fields = fields

    def as_proxy(self):
        from dataclasses import fields

        args = []
        kwargs = {}
        for field in fields(self.value):
            proxy = self.fields[field.name].as_proxy()
            if hasattr(field, "kw_only") and field.kw_only:
                kwargs[field.name] = proxy
            else:
                args.append(proxy)

        return self.python_type()(*args, **kwargs)

    # NB: This is called during __init__ for a frozen dataclass
    # use this to accumulate the most up-to-date field values
    def method_setattr_standard(self, tx: "InstructionTranslator", name, value):
        self.fields[name.as_python_constant()] = value
        return super().method_setattr_standard(tx, name, value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value_type.__name__})"


class SourcelessGraphModuleVariable(UserDefinedObjectVariable):
    def __init__(
        self,
        value,
        **kwargs,
    ) -> None:
        super().__init__(value, **kwargs)

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        fn_variable = variables.UserFunctionVariable(self.value.forward.__func__)
        args = [self] + args
        return tx.inline_user_function_return(
            fn_variable,
            args,
            kwargs,
        )


class KeyedJaggedTensorVariable(UserDefinedObjectVariable):
    @staticmethod
    def is_matching_object(obj):
        mod = sys.modules.get("torchrec.sparse.jagged_tensor")
        return mod is not None and type(obj) is mod.KeyedJaggedTensor

    def __init__(self, value, **kwargs) -> None:
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

        assert type(value) is KeyedJaggedTensor
        super().__init__(value, **kwargs)

    def var_getattr(self, tx: "InstructionTranslator", name):
        if (
            torch._dynamo.config.force_unspec_int_unbacked_size_like_on_torchrec_kjt
            and self.source is not None
            and name in ("_length_per_key", "_offset_per_key")
        ):
            with TracingContext.patch(force_unspec_int_unbacked_size_like=True):
                return super().var_getattr(tx, name)
        return super().var_getattr(tx, name)


class RemovableHandleClass:
    # Dummy class to pass to python_type of RemovableHandleVariable
    # Useful for isinstance check on hooks
    pass


class RemovableHandleVariable(VariableTracker):
    REMOVED = -1

    def __init__(
        self,
        mutation_type=None,
        # index of the registration in the side_effects owned register_hook/handle list, used during removal.
        idx=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.mutation_type = mutation_type
        self.idx = idx

    def call_method(self, tx: "InstructionTranslator", method_name, args, kwargs):
        if method_name == "remove":
            if self.idx != self.REMOVED:
                tx.output.side_effects.remove_hook(self.idx)
                self.idx = self.REMOVED
            return variables.ConstantVariable.create(None)
        super().call_method(tx, method_name, args, kwargs)

    def reconstruct(self, codegen):
        if self.idx == self.REMOVED:
            # Hook has already been removed, return a dummy handle
            codegen.add_push_null(
                lambda: codegen.load_import_from(
                    "torch._dynamo.utils", "invalid_removeable_handle"
                )
            )
            codegen.extend_output(create_call_function(0, False))
            return
        # unreachable due to codegen.add_cache() when the hook is installed
        super().reconstruct(codegen)

    def python_type(self):
        return RemovableHandleClass


class UserDefinedDictVariable(UserDefinedObjectVariable):
    _nonvar_fields = UserDefinedObjectVariable._nonvar_fields

    def __init__(self, value, dict_vt=None, **kwargs):
        super().__init__(value, **kwargs)
        self._dict_vt = dict_vt
        if self._dict_vt is None:
            assert (
                self.source is None
            ), "dict_vt must be constructed when source is present"
            self._dict_vt = variables.ConstDictVariable(
                {}, mutation_type=ValueMutationNew()
            )
        # Move this to _dynamo/utils. Include OrderedDict methods as well.
        self._dict_methods = [
            method
            for method in itertools.chain(
                dict.__dict__.values(), collections.OrderedDict.__dict__.values()
            )
            if callable(method)
        ]

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        method = self._maybe_get_baseclass_method(name)
        if method in self._dict_methods:
            return self._dict_vt.call_method(tx, name, args, kwargs)
        return super().call_method(tx, name, args, kwargs)


class MutableMappingVariable(UserDefinedObjectVariable):
    _nonvar_fields = UserDefinedObjectVariable._nonvar_fields

    def __init__(self, value, **kwargs):
        super().__init__(value, **kwargs)
        self.generic_dict_vt = variables.ConstDictVariable({})
        self.mutation_type = AttributeMutationExisting()

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> "VariableTracker":
        # A common pattern in the init code of MutableMapping objects is to
        # update the __dict__ attribute. To prevent graph break, we directly
        # return a ConstDictVariable for the __dict__attr.
        #
        # However, users can try to add a new attribute to the class using the
        # __dict__ attribute. To catch this, we save the ConstDictVariable for
        # the __dict__ and then lookup into this vt for each attr lookup.
        if name == "get" and type(self.value).get in (
            collections.abc.Mapping.get,
            dict.get,
        ):
            return variables.UserMethodVariable(polyfills.mapping_get, self)
        elif name == "__dict__" and self.source:
            self.generic_dict_vt = variables.LazyVariableTracker.create(
                self.value.__dict__, AttrSource(self.source, "__dict__")
            )
            return self.generic_dict_vt
        elif out := self.generic_dict_vt.maybe_getitem_const(
            variables.ConstantVariable(name)
        ):
            return out
        else:
            return super().var_getattr(tx, name)


class RandomVariable(UserDefinedObjectVariable):
    pass
