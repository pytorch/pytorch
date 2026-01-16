"""
This module contains variable classes for handling user-defined objects in Dynamo's tracing system.

The key classes are:
- UserDefinedVariable: Base class for representing custom Python objects
- UserDefinedClassVariable: Handles Python class objects/types
- UserDefinedObjectVariable: Fallback class for instance objects, with support for method calls,
  attribute access, and other Python object behaviors.
- Specialized subclasses for common patterns:
  - UserDefinedDictVariable: For dict subclasses
  - UserDefinedSetVariable: For set subclasses
  - UserDefinedTupleVariable: For tuple subclasses
  - UserDefinedExceptionObjectVariable: For exception subclasses
  - FrozenDataClassVariable: Special handling of frozen dataclasses
  - MutableMappingVariable: For collections.abc.MutableMapping subclasses

Dynamo specializes to VariableTracker subclasses like FrozenDataClassVariable if available; if no
subclass qualifies, it falls back to UserDefinedObjectVariable.

These classes help Dynamo track and handle arbitrary Python objects during tracing,
maintaining proper semantics while enabling optimizations where possible.
"""

import _collections  # type: ignore[import-not-found]
import builtins
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
import traceback
import types
import warnings
import weakref
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, TYPE_CHECKING, Union
from typing_extensions import is_typeddict

import torch._dynamo.config
import torch.nn
from torch._guards import Source, TracingContext
from torch.utils._python_dispatch import is_traceable_wrapper_subclass_type

from .. import graph_break_hints, polyfills, variables
from ..bytecode_transformation import create_call_function
from ..create_parameter_op import do_not_convert_to_tracable_parameter
from ..exc import (
    handle_observed_exception,
    ObservedAttributeError,
    ObservedKeyError,
    ObservedTypeError,
    ObservedUserStopIteration,
    raise_observed_exception,
    unimplemented,
)
from ..graph_bytecode_inputs import get_external_object_by_index
from ..guards import GuardBuilder, install_guard
from ..source import (
    AttrSource,
    CallFunctionNoArgsSource,
    DataclassFieldsSource,
    DictGetItemSource,
    GetItemSource,
    RandomValueSource,
    TypeDictSource,
    TypeMROSource,
    TypeSource,
    UnspecializedParamBufferSource,
)
from ..utils import (
    check_constant_args,
    cmp_name_to_op_mapping,
    dict_methods,
    frozenset_methods,
    get_custom_getattr,
    has_torch_function,
    is_frozen_dataclass,
    is_lru_cache_wrapped_function,
    is_namedtuple_cls,
    is_wrapper_or_member_descriptor,
    istype,
    list_methods,
    namedtuple_fields,
    object_has_getattribute,
    proxy_args_kwargs,
    raise_args_mismatch,
    raise_on_overridden_hash,
    set_methods,
    tensortype_to_dtype,
    tuple_methods,
    unpatched_nn_module_getattr,
)
from .base import MutationType, raise_type_error_exc, ValueMutationNew, VariableTracker
from .dicts import ConstDictVariable, DefaultDictVariable, SetVariable


try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]

try:
    from torch.utils._cxx_pytree import PyTreeSpec
except ImportError:
    PyTreeSpec = type(None)  # type: ignore[misc, assignment]


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.side_effects import SideEffects
    from torch._dynamo.symbolic_convert import InstructionTranslator
    from torch._dynamo.variables.constant import ConstantVariable

    from .lists import ListVariable, TupleVariable


def is_standard_setattr(val: object) -> bool:
    return val in (object.__setattr__, BaseException.__setattr__)


def is_standard_delattr(val: object) -> bool:
    return val in (object.__delattr__, BaseException.__delattr__)


def is_forbidden_context_manager(ctx: object) -> bool:
    f_ctxs: list[Any] = []

    try:
        from _pytest.python_api import RaisesContext  # type: ignore[attr-defined]
        from _pytest.recwarn import WarningsChecker  # type: ignore[attr-defined]

        f_ctxs.append(RaisesContext)
        f_ctxs.append(WarningsChecker)
    except ImportError:
        pass

    if m := sys.modules.get("torch.testing._internal.jit_utils"):
        f_ctxs.append(m._AssertRaisesRegexWithHighlightContext)

    return ctx in f_ctxs


def is_cython_function(obj: object) -> bool:
    return (
        callable(obj)
        and hasattr(type(obj), "__name__")
        and type(obj).__name__ == "cython_function_or_method"
    )


class UserDefinedVariable(VariableTracker):
    value: object


class UserDefinedClassVariable(UserDefinedVariable):
    # pyrefly: ignore[bad-override]
    value: type[object]

    def __init__(self, value: type[object], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.value = value
        # Used when we materialize class.__dict__ to a MappingProxyObject. In
        # this case, we don't want to allow mutation in the class because there
        # is no way to reflect it in the created MappingProxyVariable.
        self.ban_mutation = False

    def as_python_constant(self) -> type[object]:
        return self.value

    def as_proxy(self) -> object:
        return self.value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value})"

    @staticmethod
    @functools.cache
    def _constant_fold_classes() -> set[type[object]]:
        return {
            torch.device,
            torch.finfo,
            torch.iinfo,
            torch.Size,
        }

    @staticmethod
    @functools.cache
    def _in_graph_classes() -> set[type[object]]:
        _in_graph_class_list = {
            torch.Tensor,
            torch.cuda.FloatTensor,  # type: ignore[attr-defined]
            torch.cuda.DoubleTensor,  # type: ignore[attr-defined]
            torch.cuda.HalfTensor,  # type: ignore[attr-defined]
            torch.cuda.BFloat16Tensor,  # type: ignore[attr-defined]
            torch.cuda.ByteTensor,  # type: ignore[attr-defined]
            torch.cuda.CharTensor,  # type: ignore[attr-defined]
            torch.cuda.IntTensor,  # type: ignore[attr-defined]
            torch.cuda.ShortTensor,  # type: ignore[attr-defined]
            torch.cuda.LongTensor,  # type: ignore[attr-defined]
            torch.Stream,
            torch.Event,
            torch.cuda.Stream,
            torch.cuda.Event,
            torch.xpu.Stream,
            torch.xpu.Event,
        }
        if hasattr(torch, "hpu"):
            _in_graph_class_list.update(
                {
                    torch.hpu.Stream,
                    torch.hpu.Event,
                }
            )

        return set(tensortype_to_dtype.keys()) | _in_graph_class_list

    @staticmethod
    @functools.cache
    def supported_c_new_functions() -> set[Any]:
        exceptions: set[Any] = {
            getattr(builtins, name).__new__
            for name in dir(builtins)
            if isinstance(getattr(builtins, name), type)
            and issubclass(getattr(builtins, name), BaseException)
        }
        c_new_fns: set[Any] = {
            object.__new__,
            dict.__new__,
            set.__new__,
            frozenset.__new__,
            tuple.__new__,
            list.__new__,
        }
        return c_new_fns.union(exceptions)

    @staticmethod
    def is_supported_new_method(value: object) -> bool:
        # TODO(anijain2305) - Extend this to support objects with default tp_new
        # functions.
        return value in UserDefinedClassVariable.supported_c_new_functions()

    def can_constant_fold_through(self) -> bool:
        return self.value in self._constant_fold_classes()

    def has_key_in_generic_dict(self, tx: "InstructionTranslator", key: str) -> bool:
        if tx.output.side_effects.has_pending_mutation_of_attr(self, key):
            mutated_attr = tx.output.side_effects.load_attr(self, key, deleted_ok=True)
            return not isinstance(mutated_attr, variables.DeletedVariable)

        return key in self.value.__dict__

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        from . import ConstantVariable, EnumVariable

        source = AttrSource(self.source, name) if self.source is not None else None

        if name == "__name__":
            return ConstantVariable.create(self.value.__name__)
        elif name == "__qualname__":
            return ConstantVariable.create(self.value.__qualname__)
        elif name == "__dict__":
            options = {"source": source}
            return variables.GetAttrVariable(self, name, None, **options)
        elif name == "__mro__":
            attr_source = self.source and TypeMROSource(self.source)
            return VariableTracker.build(tx, self.value.__mro__, attr_source)

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

        obj = None
        try:
            obj = inspect.getattr_static(self.value, name)
        except AttributeError:
            if type(self.value) is type:
                raise_observed_exception(
                    AttributeError,
                    tx,
                    args=[
                        f"type object '{self.value.__name__}' has no attribute '{name}'"
                    ],
                )

        if name == "__new__" and UserDefinedClassVariable.is_supported_new_method(obj):
            return super().var_getattr(tx, name)

        if name in cmp_name_to_op_mapping and not isinstance(obj, types.FunctionType):
            return variables.GetAttrVariable(self, name, None, source=source)

        if isinstance(obj, staticmethod):
            return VariableTracker.build(tx, obj.__get__(self.value), source)
        elif isinstance(obj, classmethod):
            if isinstance(obj.__func__, property):
                fget_vt = VariableTracker.build(tx, obj.__func__.fget)
                return fget_vt.call_function(tx, [self], {})
            return variables.UserMethodVariable(obj.__func__, self, source=source)
        elif isinstance(obj, types.ClassMethodDescriptorType):
            # e.g.: inspect.getattr_static(dict, "fromkeys")
            #       inspect.getattr_static(itertools.chain, "from_iterable")
            func = obj.__get__(None, self.value)
            return VariableTracker.build(tx, func, source)
        elif source:
            if inspect.ismemberdescriptor(obj):
                return VariableTracker.build(tx, obj.__get__(self.value), source)

        if ConstantVariable.is_literal(obj):
            return ConstantVariable.create(obj)
        elif isinstance(obj, enum.Enum):
            return EnumVariable(obj)
        elif self.value is collections.OrderedDict:
            return variables.GetAttrVariable(self, name)
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

    def _call_cross_entropy_loss(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        """
        functional: input, target, weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean',
        label_smoothing=0.0

        non functional ctor: weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean',
        label_smoothing=0.0

        non functional loss call: input, target, optional_output
        """
        from . import ConstantVariable

        def normalize_args(
            weight: VariableTracker = ConstantVariable.create(None),
            size_average: VariableTracker = ConstantVariable.create(None),
            ignore_index: VariableTracker = ConstantVariable.create(-100),
            reduce: VariableTracker = ConstantVariable.create(None),
            reduction: VariableTracker = ConstantVariable.create("mean"),
            label_smoothing: VariableTracker = ConstantVariable.create(0.0),
        ) -> tuple[VariableTracker, ...]:
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

        def fake_cross_entropy_loss(
            input: VariableTracker, target: VariableTracker
        ) -> VariableTracker:
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
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if (
            name == "__subclasses__"
            and len(args) == 0
            and not kwargs
            and "__subclasses__" not in self.value.__dict__
        ):
            source = self.source
            if self.source:
                source = AttrSource(self.source, "__subclasses__")
                source = CallFunctionNoArgsSource(source)
            return VariableTracker.build(tx, self.value.__subclasses__(), source)
        elif (
            self.value in {collections.OrderedDict, collections.defaultdict}
            and name == "fromkeys"
        ):
            return variables.BuiltinVariable.call_custom_dict_fromkeys(
                tx, self.value, *args, **kwargs
            )
        elif self.value is collections.OrderedDict and name == "move_to_end":
            return args[0].call_method(tx, name, [*args[1:]], kwargs)
        elif name == "__eq__" and len(args) == 1 and hasattr(args[0], "value"):
            return variables.ConstantVariable(self.value == args[0].value)
        elif name == "__ne__" and len(args) == 1 and hasattr(args[0], "value"):
            return variables.ConstantVariable(self.value != args[0].value)
        elif issubclass(self.value, dict) and name != "__new__":
            # __new__ is handled below
            return variables.BuiltinVariable(dict).call_method(tx, name, args, kwargs)
        elif issubclass(self.value, (set, frozenset)) and name != "__new__":
            # __new__ is handled below
            return variables.BuiltinVariable(set).call_method(tx, name, args, kwargs)
        elif (
            name == "__new__"
            and self.value is collections.OrderedDict
            and isinstance(args[0], UserDefinedClassVariable)
            and args[0].value is collections.OrderedDict
        ):
            if kwargs and len(args) != 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            return ConstDictVariable(
                {}, collections.OrderedDict, mutation_type=ValueMutationNew()
            )
        elif (
            len(args) == 1
            and isinstance(args[0], variables.GenericContextWrappingVariable)
            and name == "__enter__"
        ):
            return args[0].enter(tx)
        elif name == "__new__" and UserDefinedClassVariable.is_supported_new_method(
            self.value.__new__
        ):
            return tx.output.side_effects.track_new_user_defined_object(
                self,
                args[0],
                args[1:],
            )
        elif name == "__setattr__" and self.ban_mutation:
            unimplemented(
                gb_type="Class attribute mutation when the __dict__ was already materialized",
                context=str(self.value),
                explanation="Dyanmo does not support tracing mutations on a class when its __dict__ is materialized",
                hints=graph_break_hints.SUPPORTABLE,
            )
        return super().call_method(tx, name, args, kwargs)

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from ..side_effects import SideEffects
        from .builder import wrap_fx_proxy
        from .ctx_manager import GenericContextWrappingVariable

        constant_args = check_constant_args(args, kwargs)

        if self.can_constant_fold_through() and constant_args:
            # constant fold
            return variables.ConstantVariable.create(
                self.as_python_constant()(  # type: ignore[operator]
                    *[x.as_python_constant() for x in args],
                    **{k: v.as_python_constant() for k, v in kwargs.items()},
                ),
            )
        elif self.value is torch.nn.CrossEntropyLoss:
            return self._call_cross_entropy_loss(tx, args, kwargs)
        elif self.value is contextlib.nullcontext:
            # import here to avoid circular dependency
            from .ctx_manager import NullContextVariable

            return NullContextVariable(*args, **kwargs)
        elif self.value is collections.OrderedDict:
            return tx.inline_user_function_return(
                VariableTracker.build(tx, polyfills.construct_dict),
                [self, *args],
                kwargs,
            )
        elif self.value is collections.defaultdict:
            if len(args) == 0:
                default_factory = variables.ConstantVariable.create(None)
            else:
                default_factory, *args = args
            dict_vt = variables.BuiltinVariable.call_custom_dict(
                tx, dict, *args, **kwargs
            )
            return DefaultDictVariable(
                dict_vt.items,  # type: ignore[attr-defined]
                collections.defaultdict,
                default_factory,
                mutation_type=ValueMutationNew(),
            )
        elif is_typeddict(self.value):
            if self.value.__optional_keys__:  # type: ignore[attr-defined]
                unimplemented(
                    gb_type="TypedDict with optional keys",
                    context=str(self.value),
                    explanation="Dyanmo does not support tracing TypedDict with optional keys",
                    hints=[
                        "Avoid using TypedDict with optional keys",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )
            return variables.BuiltinVariable(dict).call_dict(tx, *args, **kwargs)
        elif self.value is collections.deque:
            maxlen = variables.ConstantVariable.create(None)

            def deque_signature(
                iterable: Iterable[Any] | None = None, maxlen: int | None = None
            ) -> Any:
                pass

            bound_args = None
            try:
                bound_args = inspect.signature(deque_signature).bind(*args, **kwargs)
            except TypeError as e:
                unimplemented(
                    gb_type="collections.deque() with bad arguments",
                    context=f"args={args}, kwargs={kwargs}",
                    explanation="Detected call to collections.deque() with bad arguments.",
                    hints=[
                        "Fix the call to collections.deque().",
                        *graph_break_hints.USER_ERROR,
                    ],
                    from_exc=e,
                )
            assert bound_args is not None
            if "iterable" in bound_args.arguments:
                if not bound_args.arguments["iterable"].has_force_unpack_var_sequence(
                    tx
                ):
                    unimplemented(
                        gb_type="collections.deque() with bad iterable argument",
                        context=f"args={args}, kwargs={kwargs}",
                        explanation="Call to collections.deque() has an iterable argument that Dynamo cannot "
                        "convert to a list.",
                        hints=[
                            "Use a simpler sequence type that Dynamo can convert to a list "
                            "(e.g. list, tuple, list iterator, etc.)",
                            *graph_break_hints.USER_ERROR,
                        ],
                    )
                items = bound_args.arguments["iterable"].force_unpack_var_sequence(tx)
            else:
                items = []

            if "maxlen" in bound_args.arguments:
                maxlen = bound_args.arguments["maxlen"]

            return variables.lists.DequeVariable(
                items, maxlen=maxlen, mutation_type=ValueMutationNew()
            )
        elif (
            # https://github.com/python/cpython/blob/33efd7178e269cbd04233856261fd0aabbf35447/Lib/contextlib.py#L475-L477
            self.value is types.MethodType
            and len(args) == 2
            and isinstance(args[0], variables.UserFunctionVariable)
            and isinstance(args[1], GenericContextWrappingVariable)
            and args[0].get_name() in ("__enter__", "__exit__")
        ):
            cm_obj = args[1].cm_obj
            fn = getattr(cm_obj, args[0].get_name()).__func__
            return variables.UserMethodVariable(fn, args[1], source=self.source)
        elif self.value is weakref.ref:
            if len(args) > 1:
                callback = args[1]
            else:
                callback = variables.ConstantVariable.create(None)
            return variables.WeakRefVariable(args[0], callback)
        elif self.value is functools.partial:
            if not args:
                unimplemented(
                    gb_type="missing args to functools.partial",
                    context="",
                    explanation="functools.partial requires at least one argument",
                    hints=[
                        "Fix the functools.partial call.",
                        *graph_break_hints.USER_ERROR,
                    ],
                )
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
            if not args[0].is_python_constant():
                raise_type_error_exc(
                    tx, "torch.cuda.device() requires a constant argument"
                )
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
            from . import TorchCtxManagerClassVariable
            from .functions import (
                BaseUserFunctionVariable,
                FunctionDecoratedByContextlibContextManagerVariable,
            )

            # graph break on any contextlib.* that it is not contextlib.contextmanager
            # Some of the APIs below are not supported because they rely on features
            # that Dynamo doesn't play well today (i.e. contextlib.suppress)
            if self.value in (
                contextlib._AsyncGeneratorContextManager,
                contextlib.closing,
                contextlib.redirect_stdout,
                contextlib.redirect_stderr,
                contextlib.AsyncExitStack,
            ):
                # We are not changing the behavior of Dynamo as these function were
                # already ignored on trace_rules.py before #136033 landed
                unimplemented(
                    gb_type="unsupported contextlib.* API",
                    context=f"{self.value}",
                    explanation=f"{self.value} not supported. This may be due to its use of "
                    "context-specific operations that are not supported in "
                    "Dynamo yet (i.e. Exception handling)",
                    hints=[
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )
            arg_new = args
            if self.value is contextlib._GeneratorContextManager and isinstance(
                args[0], (BaseUserFunctionVariable, TorchCtxManagerClassVariable)
            ):
                if not torch._dynamo.config.enable_trace_contextlib:
                    unimplemented(
                        gb_type="attempted to trace contextlib.contextmanager",
                        context=f"args={args}",
                        explanation="Tracing contextlib.contextmanager is disabled.",
                        hints=[
                            "Set torch._dynamo.config.enable_trace_contextlib = True",
                        ],
                    )

                # Special treatments for certain context managers created via
                # contextlib, because
                # 1. we (pytorch) own their impls
                # 2. it's tedious to trace through them, so we effectively
                #    "allow in graph" them without sacrificing soundness.
                #
                # We would typically reach here via either
                # 1. the instance construction in `with ctx_manager(...):`:
                #    https://github.com/python/cpython/blob/3.12/Lib/contextlib.py#L301
                # 2. calling a function decorated with a context manager:
                #    https://github.com/python/cpython/blob/3.12/Lib/contextlib.py#L122
                #
                # So we basically trace through the surface part of the
                # contextlib code, and then special case the shared remaining
                # logic (the actual context manager instance construction and
                # usage later on).
                if isinstance(args[0], TorchCtxManagerClassVariable):
                    fn_var = args[0]
                    args_list = args[1].items  # type: ignore[union-attr,attr-defined]
                    kwargs_dict = args[2].keys_as_python_constant()  # type: ignore[union-attr,attr-defined]
                    return fn_var.call_function(tx, args_list, kwargs_dict)

                # Wrap UserFunctionVariable in FunctionDecoratedByContextlibContextManagerVariable
                # if the function is annotated with @contextlib.contextmanager
                # This shouldn't be necessary once generator functions are fully
                # supported in dynamo
                # pyrefly: ignore[unsupported-operation]
                arg_new = [
                    FunctionDecoratedByContextlibContextManagerVariable(
                        args[0], source=args[0].source
                    )
                ] + args[1:]

            cm_obj = tx.output.side_effects.track_new_user_defined_object(
                variables.BuiltinVariable(object),
                self,
                arg_new,  # type: ignore[arg-type]
            )
            cm_obj.call_method(tx, "__init__", arg_new, kwargs)  # type: ignore[arg-type]
            return cm_obj
        elif is_namedtuple_cls(self.value):
            fields = namedtuple_fields(self.value)  # type: ignore[arg-type]
            # check if this a quasi-namedtuple or a real one
            if self.value.__module__ == "torch.return_types":
                if kwargs or len(args) != 1:
                    raise_args_mismatch(
                        tx,
                        "torch.return_types",
                        "1 args and 0 kwargs",
                        f"{len(args)} args and {len(kwargs)} kwargs",
                    )
                items = args[0].force_unpack_var_sequence(tx)
            else:
                field_defaults = self.value._field_defaults  # type: ignore[attr-defined]

                items = list(args)
                # pyrefly: ignore[bad-argument-type]
                items.extend([None] * (len(fields) - len(items)))

                var_tracker_kwargs: dict[str, VariableTracker] = {}
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
                    items[fields.index(name)] = value  # type: ignore[call-overload]

                assert all(x is not None for x in items)

            # Modify mutability of namedtuple for sourcelesss instantiations.
            from .base import AttributeMutationNew
            from .lists import NamedTupleVariable

            return NamedTupleVariable(
                items,
                self.value,  # type: ignore[arg-type]
                mutation_type=AttributeMutationNew(),
            )
        elif self.value is torch.Size:
            # This simulates `THPSize_pynew`, the C impl for `Size.__new__`.
            from .lists import SizeVariable

            tup = variables.BuiltinVariable(tuple).call_function(tx, args, kwargs)
            return SizeVariable(tup.items)  # type: ignore[missing-attribute]
        elif is_frozen_dataclass(self.value) and self.is_standard_new():
            fields = dataclasses.fields(self.value)  # type: ignore[arg-type]
            assert self.source is not None
            fields_source = DataclassFieldsSource(self.source)
            items = list(args)
            items.extend([None] * (len(fields) - len(items)))  # type: ignore[arg-type]

            default_kwargs = {}
            for ind, field, var_tracker in zip(itertools.count(), fields, items):
                if var_tracker is None:
                    if field.name in kwargs:
                        var_tracker = kwargs[field.name]
                    else:
                        if not field.init:
                            continue

                        if field.default is not dataclasses.MISSING:
                            var_tracker = VariableTracker.build(
                                tx,
                                field.default,
                                source=AttrSource(
                                    GetItemSource(fields_source, ind), "default"
                                ),
                            )
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

            var = tx.output.side_effects.track_new_user_defined_object(
                variables.BuiltinVariable(object),
                self,
                args,  # type: ignore[arg-type]
            )
            var.call_method(tx, "__init__", args, kwargs)  # type: ignore[arg-type]
            return var
        elif (
            self.value in self._in_graph_classes()
            or is_traceable_wrapper_subclass_type(self.value)
        ):
            # torch.LongTensor cannot accept a list of FakeTensors.
            # So we stack the list of FakeTensors instead.
            from .lists import ListVariable

            if (
                np
                and self.value in tensortype_to_dtype
                and len(args) == 1
                and isinstance(args[0], ListVariable)
                and len(args[0].items) > 1
                and all(x.is_tensor() for x in args[0].items)
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

            if issubclass(self.value, torch.Stream):
                from .constant import ConstantVariable
                from .lists import TupleVariable

                var_kwargs = ConstDictVariable(
                    {ConstantVariable(k): v for k, v in kwargs.items()}
                )
                var_args = TupleVariable(list(args))
                stream = self.value(
                    *(var_args.as_python_constant()),
                    **(var_kwargs.as_python_constant()),
                )
                from ..graph_bytecode_inputs import register_graph_created_object
                from .streams import StreamVariable

                ind = register_graph_created_object(
                    stream,
                    StreamVariable.make_construct_in_graph_stream_fn(
                        var_args, var_kwargs
                    ),
                )
                tensor_variable = wrap_fx_proxy(
                    tx=tx,
                    proxy=tx.output.create_proxy(
                        "call_function", get_external_object_by_index, (ind,), {}
                    ),
                )
            elif issubclass(self.value, torch.Event):
                from .constant import ConstantVariable
                from .lists import TupleVariable

                # Register newly created event for reconstruction
                var_kwargs = ConstDictVariable(
                    {ConstantVariable(k): v for k, v in kwargs.items()}
                )
                var_args = TupleVariable(list(args))
                event = self.value(
                    *(var_args.as_python_constant()),
                    **(var_kwargs.as_python_constant()),
                )
                from ..graph_bytecode_inputs import register_graph_created_object
                from .streams import EventVariable

                ind = register_graph_created_object(
                    event,
                    EventVariable.make_construct_in_graph_event_fn(
                        var_args, var_kwargs
                    ),
                )
                tensor_variable = wrap_fx_proxy(
                    tx=tx,
                    proxy=tx.output.create_proxy(
                        "call_function", get_external_object_by_index, (ind,), {}
                    ),
                )
            else:
                tensor_variable = wrap_fx_proxy(
                    tx=tx,
                    proxy=tx.output.create_proxy(
                        "call_function",
                        self.value,
                        *proxy_args_kwargs(args, kwargs),
                    ),
                )

            return tensor_variable
        elif self.value is random.Random:
            if len(args) == 1 and args[0].is_python_constant():
                seed = args[0].as_python_constant()
            else:
                seed = None
            random_object = random.Random(seed)
            return RandomVariable(random_object)
        elif (
            self.value is types.MappingProxyType
            and len(args) == 1
            and isinstance(args[0], ConstDictVariable)
        ):
            # types.MappingProxyType is a read-only proxy of the dict. If the
            # original dict changes, the changes are reflected in proxy as well.
            return variables.MappingProxyVariable(args[0])
        elif SideEffects.cls_supports_mutation_side_effects(self.value) and self.source:
            with do_not_convert_to_tracable_parameter():
                return tx.inline_user_function_return(
                    VariableTracker.build(
                        tx, polyfills.instantiate_user_defined_class_object
                    ),
                    [self, *args],
                    kwargs,
                )
        return super().call_function(tx, args, kwargs)

    def is_standard_new(self) -> bool:
        """Check for __new__ being overridden"""
        new_fn = inspect.getattr_static(self.value, "__new__", None)
        if isinstance(new_fn, staticmethod):
            new_fn = new_fn.__func__
        return new_fn is object.__new__

    def call_obj_hasattr(
        self, tx: "InstructionTranslator", name: str
    ) -> "ConstantVariable":
        if self.source:
            source = AttrSource(self.source, name)
            install_guard(source.make_guard(GuardBuilder.HASATTR))
            return variables.ConstantVariable(hasattr(self.value, name))
        return super().call_obj_hasattr(tx, name)

    def const_getattr(self, tx: "InstructionTranslator", name: str) -> Any:
        if name == "__name__":
            return self.value.__name__
        return super().const_getattr(tx, name)

    def is_python_hashable(self) -> Literal[True]:
        return True

    def get_python_hash(self) -> int:
        return hash(self.value)

    def is_python_equal(self, other: object) -> bool:
        return (
            isinstance(other, variables.UserDefinedClassVariable)
            and self.value is other.value
        )


class UserDefinedExceptionClassVariable(UserDefinedClassVariable):
    @property
    def fn(self) -> type[object]:
        return self.value


class NO_SUCH_SUBOBJ:
    pass


class RemovableHandleClass:
    # Dummy class to pass to python_type of
    # RemovableHandleVariable
    # Useful for isinstance check on hooks
    pass


def call_random_fn(
    tx: "InstructionTranslator",
    fn: Callable[..., Any],
    args: Sequence[VariableTracker],
    kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    from .builder import VariableBuilder

    args = [x.as_python_constant() for x in args]
    kwargs = {k: v.as_python_constant() for k, v in kwargs.items()}
    random_call_index = len(tx.output.random_calls)
    example_value = fn(*args, **kwargs)
    source = RandomValueSource(random_call_index)
    tx.output.random_calls.append((fn, args, kwargs))  # type: ignore[arg-type]
    # TODO: arguably, this should route to wrap_symint/wrap_symfloat
    # (currently hypothetical), but I'm not going to poke my hand in
    # this nest for now
    return VariableBuilder(tx, source).wrap_unspecialized_primitive(example_value)


class UserDefinedObjectVariable(UserDefinedVariable):
    """
    Mostly objects of defined type.  Catch-all for something where we only know the type.
    """

    _nonvar_fields = {
        "value",
        "value_type",
        "attrs_directly_modifed_on_dict",
        *UserDefinedVariable._nonvar_fields,
    }

    def __init__(
        self,
        value: object,
        *,
        value_type: type | None = None,
        cls_source: TypeSource | None = None,
        base_cls_vt: VariableTracker | None = None,
        init_args: Sequence[VariableTracker] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.value = value
        self.value_type = value_type or type(value)
        assert type(value) is self.value_type
        # This is used with __new__, when the new object is sourceless but the user class can be sourceful.
        self.cls_source = cls_source
        if cls_source is None and self.source is not None:
            self.cls_source = TypeSource(self.source)

        # These attributes are used to reconstruct the user defined object. The
        # pseudo code looks like this. Builtin C __new__ do not support kwargs,
        # so init_args is sufficient.
        #   obj = base_cls.__new__(user_cls, *args)
        self.base_cls_vt = base_cls_vt
        self.init_args = init_args

        # This records names of the attributes that were modified via instance
        # `__dict__` directly, rather than the normal setattr path.
        #
        # TODO consider emulating `obj.__dict__` as a `ConstDictVariable` to get
        # rid of these workarounds here and in `GetAttrVariable`.
        self.attrs_directly_modifed_on_dict: set[str] = set()

        # Cache inspect.getattr_static outputs for the same name. This is fine
        # because if there is a mutation for the name, we use side-effects infra
        # to early return the mutated value.
        self._looked_up_attrs: dict[str, object] = {}

        # This is to avoid getattr_static calls to look up the subobj from the self.value.__class__
        self._subobj_from_class: dict[str, object] = {}

        import torch.utils._pytree as pytree

        self.is_pytree_constant_class = pytree.is_constant_class(self.value_type)
        if pytree.is_constant_class(self.value_type) and self.source:
            install_guard(self.source.make_guard(GuardBuilder.EQUALS_MATCH))

        self._object_has_getattribute = object_has_getattribute(self.value)

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

    def is_underlying_vt_modified(self, side_effects: "SideEffects") -> bool:
        return False

    def python_type(self) -> type:
        return self.value_type  # type: ignore[return-value]

    def as_python_constant(self) -> object:
        if self.is_pytree_constant_class and self.source:
            # NOTE pytree constants created in the torch.compile region will
            # NOT be guarded (even though they have a source set)
            return self.value
            # TODO else try reconstructing the object by, e.g., leveraging side
            # effects and `as_python_constant`.
        return super().as_python_constant()

    def guard_as_python_constant(self) -> object:
        if self.source:
            install_guard(self.source.make_guard(GuardBuilder.ID_MATCH))
            return self.value
        return super().guard_as_python_constant()

    def torch_function_check(self) -> None:
        assert has_torch_function(self), (
            f"calling torch function on object without __torch_function__ {self}"
        )

    def get_torch_fn(self, tx: "InstructionTranslator") -> VariableTracker:
        self.torch_function_check()
        from .torch_function import get_torch_function_fn

        return get_torch_function_fn(tx, self)

    def call_torch_function(
        self,
        tx: "InstructionTranslator",
        fn: VariableTracker,
        types: "TupleVariable",
        args: Sequence[Any],
        kwargs: dict[str, Any],
    ) -> VariableTracker:
        self.torch_function_check()

        from .torch_function import call_torch_function

        return call_torch_function(
            tx,
            self.get_torch_fn(tx),
            fn,
            types,
            args,
            kwargs,
        )

    @staticmethod
    @functools.cache
    def _supported_random_functions() -> set[Any]:
        fns = {
            random.random,
            random.randint,
            random.randrange,
            random.uniform,
        }
        return fns

    def _maybe_get_baseclass_method(self, name: str) -> set[types.FunctionType] | None:
        if name not in getattr(self.value, "__dict__", {}):
            try:
                return inspect.getattr_static(type(self.value), name)
            except AttributeError:
                pass
        return None

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[Any],
        kwargs: dict[str, Any],
    ) -> VariableTracker:
        from . import ConstantVariable, UserMethodVariable

        method = self._maybe_get_baseclass_method(name)
        if method is not None:
            if method is object.__init__:
                return ConstantVariable.create(None)

            if is_standard_setattr(method) or isinstance(self.value, threading.local):
                return self.method_setattr_standard(tx, *args, **kwargs)

            if is_standard_delattr(method):
                return self.method_setattr_standard(
                    tx, args[0], variables.DeletedVariable()
                )

            if method is object.__eq__ and len(args) == 1 and not kwargs:
                other = args[0]
                if not isinstance(other, UserDefinedObjectVariable):
                    return variables.ConstantVariable.create(NotImplemented)

                # TODO(anijain2305) - Identity checking should already be a part
                # of the cmp_eq  polyfill function.
                return ConstantVariable.create(self.value is other.value)

            if torch._dynamo.config.enable_faithful_generator_behavior and isinstance(
                self.value, types.GeneratorType
            ):
                unimplemented(
                    gb_type="call_method on generator",
                    context=f"object={self.value}, method={name}, args={args}, kwargs={kwargs}",
                    explanation="Detected a method call to a user-defined generator object. "
                    "This is not fully supported.",
                    hints=[
                        "Set `torch._dynamo.config.enable_faithful_generator_behavior = False`. Note that this "
                        "may cause silent incorrectness, since we will eagerly unpack generators instead of lazily "
                        "evaluating them.",
                    ],
                )

            # check for methods implemented in C++
            if isinstance(method, types.FunctionType):
                source = self.source
                source_fn = None
                if source:
                    source_fn = self.get_source_by_walking_mro(name)
                # TODO(jansel): add a guard to check for monkey patching?
                from ..mutation_guard import unpatched_nn_module_init

                if method is torch.nn.Module.__init__:
                    method = unpatched_nn_module_init
                return UserMethodVariable(
                    method, self, source_fn=source_fn, source=source
                ).call_function(tx, args, kwargs)  # type: ignore[arg-type]

            if method is list.__len__ and self.source and not (args or kwargs):
                install_guard(self.source.make_guard(GuardBuilder.SEQUENCE_LENGTH))
                return ConstantVariable(len(self.value))  # type: ignore[arg-type]

        return super().call_method(tx, name, args, kwargs)

    def method_setattr_standard(
        self,
        tx: "InstructionTranslator",
        name: VariableTracker,
        value: VariableTracker,
        directly_update_dict: bool = False,
    ) -> VariableTracker:
        name_str = ""
        try:
            name_str = name.as_python_constant()
        except NotImplementedError:
            unimplemented(
                gb_type="non-const setattr name on user-defined object",
                context=f"object={self}, name={name}, value={value}",
                explanation="Detected a call to `setattr` of a user-defined object with a non-constant name.",
                hints=["Ensure that the name is a string."],
            )
        assert tx.output.side_effects.is_attribute_mutation(self), (
            "Attempted setattr on a user-defined object that does not have "
            "an AttributeMutation mutation_type"
        )

        if directly_update_dict:
            self.attrs_directly_modifed_on_dict.add(name_str)
        else:
            tmp = self.try_get_descritor_and_setter_py_func(name_str)
            if tmp:
                descriptor, setter = tmp
                # Emulate
                # https://github.com/python/cpython/blob/3.11/Objects/object.c#L1371-L1452
                desc_source = None
                func_source = None
                if self.cls_source:
                    desc_source = self.get_source_by_walking_mro(name_str)
                    # use `type(...)` to ignore instance attrs.
                    func_source = AttrSource(TypeSource(desc_source), "__set__")
                desc_var = VariableTracker.build(tx, descriptor, desc_source)
                func_var = VariableTracker.build(tx, setter, func_source)
                args = [desc_var, self, value]
                return func_var.call_function(tx, args, {})
            # NOTE: else we assume the descriptor (if any) has a
            # side-effect-free `__set__` as far as Dynamo tracing is concerned.

        # Emulate the standard setattr on instance dict.
        tx.output.side_effects.store_attr(self, name_str, value)
        return variables.ConstantVariable(None)

    def needs_slow_setattr(self) -> bool:
        return not is_standard_setattr(
            inspect.getattr_static(self.value, "__setattr__", None)
        ) and not isinstance(self.value, threading.local)

    def unpack_var_sequence(self, tx: "InstructionTranslator") -> list[VariableTracker]:
        if (
            self.source
            and self._maybe_get_baseclass_method("__iter__") is list.__iter__
            and self._maybe_get_baseclass_method("__len__") is list.__len__
            and self._maybe_get_baseclass_method("__getitem__") is list.__getitem__
        ):
            install_guard(self.source.make_guard(GuardBuilder.SEQUENCE_LENGTH))
            return [
                variables.LazyVariableTracker.create(
                    self.value[k],  # type: ignore[index]
                    source=GetItemSource(self.source, k),
                )
                for k in range(len(self.value))  # type: ignore[arg-type]
            ]
        return super().unpack_var_sequence(tx)

    def has_force_unpack_var_sequence(self, tx: "InstructionTranslator") -> bool:
        try:
            variables.BuiltinVariable(iter).call_function(tx, [self], {})
            return True
        except ObservedTypeError:
            handle_observed_exception(tx)
            return False

    def force_unpack_var_sequence(
        self, tx: "InstructionTranslator"
    ) -> list[VariableTracker]:
        result = []
        iter_ = variables.BuiltinVariable(iter).call_function(tx, [self], {})

        while True:
            try:
                r = iter_.next_variable(tx)
                result.append(r)
            except ObservedUserStopIteration:
                handle_observed_exception(tx)
                break
        return result

    def next_variable(self, tx: "InstructionTranslator") -> VariableTracker:
        return self.call_method(tx, "__next__", [], {})

    def is_supported_random(self) -> bool:
        try:
            return self.value in self._supported_random_functions()
        except TypeError:
            # TypeError: unhashable type
            return False

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if (
            self.is_supported_random()
            and all(k.is_python_constant() for k in args)
            and all(v.is_python_constant() for v in kwargs.values())
        ):
            return call_random_fn(tx, self.value, args, kwargs)  # type: ignore[arg-type]
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
                var = variables.ConstantVariable(obj.mode)  # type: ignore[attr-defined]
                return variables.TorchCtxManagerClassVariable(
                    obj.__class__
                ).call_function(tx, [var], kwargs)

            if self.source is None:
                unimplemented(
                    gb_type="attempted to call sourceless user-defined object as a method",
                    context=f"object={self.value}, function={func}, args={args}, kwargs={kwargs}",
                    explanation="Dynamo does not support this.",
                    hints=[
                        f"Ensure the user-defined object {self.value} is constructed outside the compiled region.",
                    ],
                )
            assert self.source is not None
            func_src = AttrSource(self.source, "__func__")
            func_var = VariableTracker.build(tx, func, func_src)
            obj_src = AttrSource(self.source, "__self__")
            obj_var = VariableTracker.build(tx, obj, obj_src)
            return func_var.call_function(tx, [obj_var] + args, kwargs)  # type: ignore[arg-type]
        elif callable(self.value):
            if self.source:
                assert self.cls_source is not None
                source_attr = AttrSource(self.cls_source, "__call__")
                install_guard(source_attr.make_guard(GuardBuilder.CLOSURE_MATCH))
            return self.call_method(tx, "__call__", args, kwargs)  # type: ignore[arg-type]

        return super().call_function(tx, args, kwargs)

    def _check_for_getattr(self) -> object:
        return get_custom_getattr(self.value)

    def _is_c_defined_property(self, subobj: object) -> bool:
        if not isinstance(subobj, property):
            return False

        # pybind def_readwrite is implemented via PyCFunction. At the python level, it is visible as a property whose
        # fget is an instancemethod wrapper - https://docs.python.org/3/c-api/method.html#c.PyInstanceMethod_Check

        # If we have a PyCFunction, we make an assumption that there is no side effect.
        return isinstance(
            subobj.fget, types.BuiltinFunctionType
        ) or torch._C._dynamo.utils.is_instancemethod(subobj.fget)  # type: ignore[attr-defined]

    def _getattr_static(self, name: str) -> object:
        if name in self._looked_up_attrs:
            return self._looked_up_attrs[name]

        subobj = inspect.getattr_static(self.value, name, NO_SUCH_SUBOBJ)

        # In some cases, we have to do dynamic lookup because getattr_static is not enough. For example, threading.local
        # has side-effect free __getattribute__ and the attribute is not visible without a dynamic lookup.
        # NOTE we assume the following descriptors are side-effect-free as far
        # as Dynamo tracing is concerned.
        if not self._object_has_getattribute and (
            subobj is NO_SUCH_SUBOBJ  # e.g., threading.local
            or inspect.ismemberdescriptor(subobj)  # e.g., __slots__
            or inspect.isgetsetdescriptor(subobj)  # e.g., __dict__
            or self._is_c_defined_property(subobj)
        ):
            # Call __getattribute__, we have already checked that this is not overridden and side-effect free. We don't
            # want to call getattr because it can be user-overridden.
            subobj = type(self.value).__getattribute__(self.value, name)
        elif self._object_has_getattribute and subobj is NO_SUCH_SUBOBJ:
            # If the object has an overridden getattribute method, Dynamo has
            # already tried tracing it, and encountered an AttributeError. We
            # call getattr_static only when the __getattribute__ tracing fails
            # (check var_getattr impl). So, it is safe here to raise the
            # AttributeError.
            raise AttributeError

        self._looked_up_attrs[name] = subobj
        return subobj

    def should_skip_descriptor_setter(self, attr_name: str) -> bool:
        # Check if `attr_name` corresponds to a descriptor.
        descriptor = inspect.getattr_static(type(self.value), attr_name, None)
        setter = inspect.getattr_static(type(descriptor), "__set__", None)
        if setter:
            # Skip if `__set__` was traceable (no need to redo the side effect).
            if inspect.isfunction(setter):
                return True
            # For untraceable `__set__` we should still skip if the attribute
            # was mutated via instance `__dict__`.
            elif attr_name in self.attrs_directly_modifed_on_dict:
                return True
        return False

    def try_get_descritor_and_setter_py_func(
        self, attr_name: str
    ) -> tuple[object, object] | None:
        descriptor = inspect.getattr_static(type(self.value), attr_name, None)
        setter = inspect.getattr_static(type(descriptor), "__set__", None)
        if inspect.isfunction(setter):
            return (descriptor, setter)
        return None

    def has_key_in_generic_dict(self, tx: "InstructionTranslator", key: str) -> bool:
        if tx.output.side_effects.has_pending_mutation_of_attr(self, key):
            mutated_attr = tx.output.side_effects.load_attr(self, key, deleted_ok=True)
            return not isinstance(mutated_attr, variables.DeletedVariable)

        return key in self.value.__dict__

    def get_source_by_walking_mro(self, name: str) -> DictGetItemSource:
        assert self.cls_source is not None

        for idx, klass in enumerate(type(self.value).__mro__):
            if name in klass.__dict__:
                if idx != 0:
                    mro_source = TypeMROSource(self.cls_source)
                    klass_source: Source = GetItemSource(mro_source, idx)
                else:
                    klass_source = self.cls_source
                dict_source = TypeDictSource(klass_source)
                out_source = DictGetItemSource(dict_source, name)

                for absent_idx in range(1, idx):
                    # Insert a guard that the name is not present in the mro hierarchy
                    mro_source = TypeMROSource(self.cls_source)
                    klass_source = GetItemSource(mro_source, absent_idx)
                    dict_source = TypeDictSource(klass_source)
                    install_guard(
                        dict_source.make_guard(
                            functools.partial(
                                GuardBuilder.DICT_CONTAINS, key=name, invert=True
                            )
                        )
                    )
                # Insert a guard that the name is not present in the object __dict__
                if (
                    self.source
                    and hasattr(self.value, "__dict__")
                    and name not in self.value.__dict__
                ):
                    install_guard(
                        self.source.make_guard(
                            functools.partial(
                                GuardBuilder.NOT_PRESENT_IN_GENERIC_DICT, attr=name
                            )
                        )
                    )
                return out_source

        unimplemented(
            gb_type="could not find name in object's mro",
            context=f"name={name}, object type={type(self.value)}, mro={type(self.value).__mro__}",
            explanation=f"Could not find name `{name}` in mro {type(self.value).__mro__}",
            hints=[
                f"Ensure the name `{name}` is defined somewhere in {self.value}'s type hierarchy.",
                *graph_break_hints.USER_ERROR,
            ],
        )

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        from . import ConstantVariable

        source: Source | None = AttrSource(self.source, name) if self.source else None

        if self._object_has_getattribute:
            getattribute_fn = inspect.getattr_static(
                type(self.value), "__getattribute__"
            )
            new_source: AttrSource | None = (
                AttrSource(self.source, "__getattribute__") if self.source else None
            )

            try:
                return variables.UserMethodVariable(
                    getattribute_fn,
                    self,
                    # pyrefly: ignore[unbound-name]
                    source=new_source,
                ).call_function(tx, [ConstantVariable.create(name)], {})
            except ObservedAttributeError:
                # Pass through to __getattr__ if __getattribute__ fails
                handle_observed_exception(tx)

        if tx.output.side_effects.has_pending_mutation_of_attr(self, name):
            result = tx.output.side_effects.load_attr(self, name, deleted_ok=True)
            if isinstance(result, variables.DeletedVariable):
                raise_observed_exception(
                    AttributeError,
                    tx,
                    args=[
                        f"'{type(self.value).__name__}' object has no attribute '{name}'"
                    ],
                )
            return result

        if name == "__dict__":
            options_dict = {"source": source}
            return variables.GetAttrVariable(self, name, None, **options_dict)

        # TODO(anijain2305) - Investigate if we need specialization for more
        # dunder attrs. inspect.getattr_static does not return correct value for
        # them.
        if name == "__class__":
            cls_source: Source | None = source
            if source is None:
                cls_source = self.cls_source
            else:
                cls_source = source
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
                    and istype(self.value._parameters, dict)  # type: ignore[attr-defined]
                    and istype(self.value._buffers, dict)  # type: ignore[attr-defined]
                    and istype(self.value._modules, dict)  # type: ignore[attr-defined]
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
                        out.set_nn_module_stack_source(  # type: ignore[attr-defined]
                            AttrSource(self.get_nn_module_stack_source(), name)  # type: ignore[attr-defined]
                        )
                return out

            elif getattr_fn is not None:
                unimplemented(
                    gb_type="User-defined object with non-function __getattr__",
                    context=f"object={self.value}, name={name}, getattr_fn={getattr_fn}",
                    explanation=f"Found a non-function __getattr__ {getattr_fn} from a user-defined object {self.value} "
                    f" when attempting to getattr `{name}`",
                    hints=[
                        "Ensure the object's __getattr__ is a function type.",
                    ],
                )

        from ..mutation_guard import unpatched_nn_module_init

        if subobj is torch.nn.Module.__init__:
            subobj = unpatched_nn_module_init

        # Check if its already saved, avoids inspect getattr_static call
        if name in self._subobj_from_class:
            subobj_from_class = self._subobj_from_class[name]
        else:
            subobj_from_class = inspect.getattr_static(
                self.value.__class__, name, NO_SUCH_SUBOBJ
            )
            self._subobj_from_class[name] = subobj_from_class

        is_accessible_from_type_mro = (
            subobj_from_class is subobj
            and self.cls_source is not None
            and self.source is not None
            and hasattr(self.value, "__dict__")
            and name not in self.value.__dict__
        )

        if isinstance(subobj, property):
            if self.source:
                # Read the class attribute to reach the property
                source = AttrSource(self.get_source_by_walking_mro(name), "fget")
            fget_vt = VariableTracker.build(tx, subobj.fget, source=source)
            return fget_vt.call_function(tx, [self], {})
        elif isinstance(subobj, _collections._tuplegetter):
            # namedtuple fields are represented by _tuplegetter, and here we
            # emulate its `__get__`, which is implemented in C.
            _, (idx, _) = subobj.__reduce__()
            # Don't go through the `__getitem__` method anymore, see
            # https://github.com/python/cpython/blob/470941782f74288823b445120f6383914b659f23/Modules/_collectionsmodule.c#L2690
            assert isinstance(self, UserDefinedTupleVariable)
            return self._tuple_vt.items[idx]  # type: ignore[union-attr]
        elif isinstance(subobj, staticmethod):
            # Safe because `staticmethod.__get__` basically won't trigger user
            # code and just returns the underlying `__func__`:
            # https://github.com/python/cpython/blob/3.11/Objects/funcobject.c#L1088-L1100
            if is_accessible_from_type_mro:
                # Accessing from __dict__ does not resolve the descriptor, it
                # returns a staticmethod object, so access the __func__
                # attribute to get to the actual function.
                source = AttrSource(self.get_source_by_walking_mro(name), "__func__")
            func = subobj.__get__(self.value)
            return VariableTracker.build(tx, func, source)
        elif isinstance(subobj, classmethod):
            source_fn = None
            if is_accessible_from_type_mro:
                # Accessing from __dict__ does not resolve the descriptor, it
                # returns a classmethod object, so access the __func__
                # attribute to get to the actual function.
                source_fn = AttrSource(self.get_source_by_walking_mro(name), "__func__")  # type: ignore[assignment]
            return variables.UserMethodVariable(
                subobj.__func__,
                self.var_getattr(tx, "__class__"),
                source_fn=source_fn,
                source=source,
            )
        elif isinstance(subobj, types.ClassMethodDescriptorType):
            # e.g.: inspect.getattr_static({}, "fromkeys")
            func = subobj.__get__(self.value, None)
            return VariableTracker.build(tx, func, source)
        elif is_lru_cache_wrapped_function(subobj):
            # getattr_static returns the lru_wrapped function, and we cannot
            # extract the underlying method from the wrapped function. To handle
            # it, manually create a wrapped user method vt.
            return variables.WrapperUserMethodVariable(
                subobj, "__wrapped__", self, source=source
            )
        elif inspect.getattr_static(
            type(subobj), "__get__", NO_SUCH_SUBOBJ
        ) is not NO_SUCH_SUBOBJ and not is_wrapper_or_member_descriptor(
            type(subobj).__get__  # type: ignore[attr-defined]
        ):
            # Emulate https://github.com/python/cpython/blob/3.11/Objects/object.c#L1271-L1285
            #
            # Attribute has a __get__ method. Create a user defined object vt
            # for the subobj, and then trace the __get__ method.
            descriptor_source = None
            descriptor_get_source = None
            if self.cls_source:
                # To access the method descriptor from the udf object w/o using
                # inspect.getattr_static, we can look into the class mro
                descriptor_source = self.get_source_by_walking_mro(name)
                descriptor_get_source = AttrSource(
                    TypeSource(descriptor_source), "__get__"
                )
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
                subobj.__get__.__func__,  # type: ignore[attr-defined]
                descriptor_var,
                source=descriptor_get_source,
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
                            gb_type="User-defined object method with non-function __func__",
                            context=f"object={self.value}, name={name}, method={dynamic_subobj}, "
                            f"method.__self__={dynamic_subobj.__self__}, method.__func__={dynamic_subobj.__func__}",
                            explanation=f"Method {dynamic_subobj} (name={name}) of user-defined object {self.value} has a "
                            f"__func__ ({dynamic_subobj.__func__}) that is not a function type.",
                            hints=[
                                "Ensure that the method's __func__ is a function type.",
                            ],
                        )

                    # Use the __self__ attribute of the method to find the
                    # source of the new self object.
                    self_source = None
                    if source is not None:
                        self_source = AttrSource(source, "__self__")
                    object_vt = VariableTracker.build(
                        tx, dynamic_subobj.__self__, self_source
                    )

                    return variables.UserMethodVariable(
                        dynamic_subobj.__func__,
                        object_vt,
                    )
                func = subobj.__func__
            else:
                assert isinstance(subobj, types.FunctionType)
                func = subobj

            if inspect.ismethod(dynamic_subobj):
                var_source = None
                if is_accessible_from_type_mro:
                    var_source = self.get_source_by_walking_mro(name)
                return variables.UserMethodVariable(
                    func, self, source_fn=var_source, source=source
                )
            elif inspect.isfunction(dynamic_subobj):
                return VariableTracker.build(tx, func, source)

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
            and (not tx.output.export or torch._dynamo.config.install_free_tensors)
        ):
            # Recalculate source for params/buffers
            if name in ("_buffers", "_parameters"):
                assert self.source is not None
                source = UnspecializedParamBufferSource(self.source, name)
            source = self._wrap_source(source)

        if subobj is not NO_SUCH_SUBOBJ:
            if (
                is_wrapper_or_member_descriptor(subobj)
                or torch._C._dynamo.utils.is_instancemethod(subobj)  # type: ignore[attr-defined]
                or is_cython_function(subobj)
            ):
                options = {"source": source}
                return variables.GetAttrVariable(self, name, None, **options)
            if source:
                if is_accessible_from_type_mro:
                    source = self.get_source_by_walking_mro(name)

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
        raise_observed_exception(
            AttributeError,
            tx,
            args=[f"'{type(self.value).__name__}' object has no attribute '{name}'"],
        )

    def call_obj_hasattr(
        self, tx: "InstructionTranslator", name: str
    ) -> "ConstantVariable":
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

    def is_python_hashable(self) -> bool:
        raise_on_overridden_hash(self.value, self)
        return True

    def get_python_hash(self) -> int:
        # default hash
        return hash(self.value)

    def is_python_equal(self, other: object) -> bool:
        # id check
        if not isinstance(other, UserDefinedVariable):
            return False
        return self.value is other.value

    def call_tree_map_branch(
        self,
        tx: "InstructionTranslator",
        tree_map_fn: "variables.functions.UserFunctionVariable",
        map_fn: "VariableTracker",
        rest: "collections.abc.Sequence[VariableTracker]",
        tree_map_kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        """Emulate tree_map behavior for user-defined objects.

        In pytree, a type is a leaf if it is NOT in SUPPORTED_NODES.
        User-defined objects (that are not registered with register_pytree_node)
        are always treated as leaves. This works for both torch.utils._pytree
        and optree implementations.
        """
        # Determine which tree_map implementation is being used
        tree_map_module = getattr(getattr(tree_map_fn, "fn", None), "__module__", "")
        is_optree = tree_map_module.startswith("optree")

        if is_optree:
            # Check optree's registry - need to handle namespaces
            # In optree, types can be registered globally (type in registry)
            # or with a namespace ((namespace, type) in registry)
            try:
                import optree
                from optree.registry import _NODETYPE_REGISTRY

                # Check if registered globally
                # Namedtuples and structseqs are implicitly pytree nodes
                is_registered = (
                    self.value_type in _NODETYPE_REGISTRY
                    or optree.is_namedtuple_class(self.value_type)
                    or optree.is_structseq_class(self.value_type)
                )

                # Also check if registered with a namespace that's being used
                if not is_registered:
                    namespace_var = tree_map_kwargs.get("namespace")
                    if namespace_var is not None:
                        try:
                            namespace = namespace_var.as_python_constant()
                            # Check for namespaced registration
                            is_registered = (
                                namespace,
                                self.value_type,
                            ) in _NODETYPE_REGISTRY
                        except NotImplementedError:
                            # Can't determine namespace at compile time, fall back
                            return self._tree_map_fallback(
                                tx,
                                tree_map_fn,
                                map_fn,
                                rest,
                                tree_map_kwargs,
                            )
            except ImportError:
                # Can't import optree registry, fall back to tracing
                import logging

                log = logging.getLogger(__name__)
                log.warning(
                    "Failed to import optree.registry._NODETYPE_REGISTRY, "
                    "falling back to tracing for tree_map"
                )
                return self._tree_map_fallback(
                    tx,
                    tree_map_fn,
                    map_fn,
                    rest,
                    tree_map_kwargs,
                )
        else:
            # Check pytorch's pytree registry
            import torch.utils._pytree as pytree

            # Namedtuples and structseqs are implicitly pytree nodes
            is_registered = (
                self.value_type in pytree.SUPPORTED_NODES
                or pytree.is_namedtuple_class(self.value_type)
                or pytree.is_structseq_class(self.value_type)
            )

        # If not registered, it's a leaf and we should apply the map_fn directly
        if not is_registered:
            return map_fn.call_function(tx, [self, *rest], {})

        # The type is registered in pytree - we need to fall back to tracing
        # the actual tree_map implementation since we don't have the flattening
        # logic implemented here
        return self._tree_map_fallback(
            tx,
            tree_map_fn,
            map_fn,
            rest,
            tree_map_kwargs,
        )


class FrozenDataClassVariable(UserDefinedObjectVariable):
    @staticmethod
    def create(
        tx: "InstructionTranslator", value: object, source: Source
    ) -> "FrozenDataClassVariable":
        from dataclasses import fields

        assert is_frozen_dataclass(value)

        field_map = {}
        for field in fields(value):  # type: ignore[arg-type]
            if hasattr(value, field.name):
                field_map[field.name] = VariableTracker.build(
                    tx,
                    getattr(value, field.name),
                    source and AttrSource(source, field.name),
                )

        return FrozenDataClassVariable(value, fields=field_map, source=source)

    def __init__(
        self, value: object, fields: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        super().__init__(value, **kwargs)
        if fields is None:
            fields = {}
        self.fields = fields

    def as_python_constant(self) -> object:
        # NOTE: this is an intentionally limited version of
        # `as_python_constant` for `nonstrict_trace` implementation.
        from dataclasses import fields

        import torch.utils._pytree as pytree

        if not istype(
            self.value, (pytree.TreeSpec, pytree.LeafSpec, pytree.ConstantNode)
        ):
            # TODO loosen this restriction and fix `as_proxy`.
            raise NotImplementedError(
                "currently can't reconstruct arbitrary frozen dataclass instances"
            )

        # LeafSpec is deprecated, use treespec_leaf() instead
        if istype(self.value, pytree.LeafSpec):
            return pytree.treespec_leaf()

        args = []
        kwargs = {}
        for field in fields(self.value):  # type: ignore[arg-type]
            if field.init:
                data = self.fields[field.name].as_python_constant()
                if getattr(field, "kw_only", False):
                    kwargs[field.name] = data
                else:
                    args.append(data)

        # This is safe because we know the TreeSpec classes constructors don't
        # have external side effects.
        ctor = self.python_type()
        return ctor(*args, **kwargs)

    def as_proxy(self) -> object:
        from dataclasses import fields

        args = []
        kwargs = {}
        for field in fields(self.value):  # type: ignore[arg-type]
            proxy = self.fields[field.name].as_proxy()
            if hasattr(field, "kw_only") and field.kw_only:
                kwargs[field.name] = proxy
            else:
                args.append(proxy)

        # TODO this isn't really safe, because
        # 1. it could invoke a user defined `__post_init__`.
        # 2. it could invoke a user defined `__init__` if the class _subclasses_
        #    a frozen dataclass.
        # Either of the above could end up mutating external state.
        ctor = self.python_type()
        return ctor(*args, **kwargs)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        from dataclasses import fields

        # Handle specific pytree classes
        import torch.utils._pytree as pytree

        if isinstance(self.value, pytree.TreeSpec) and self.value.is_leaf():
            # Create a new LeafSpec instance by calling the constructor
            codegen.add_push_null(
                lambda: codegen.load_import_from("torch.utils._pytree", "LeafSpec")
            )
            codegen.extend_output(create_call_function(0, False))
            return

        # For general frozen dataclasses, reconstruct by calling the constructor
        # with the field values as arguments
        dataclass_cls = self.python_type()

        if hasattr(dataclass_cls, "__post_init__"):
            unimplemented(
                gb_type="Frozen dataclass with __post_init__",
                context=f"dataclass={dataclass_cls.__name__}",
                explanation="Cannot reconstruct frozen dataclass with __post_init__ method, "
                "as it may have side effects that would be incorrectly replayed.",
                hints=[
                    "Remove the __post_init__ method from the frozen dataclass.",
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

        # Collect positional and keyword-only arguments
        pos_args = []
        kw_args = []
        for field in fields(dataclass_cls):
            if not field.init:
                continue
            field_vt = self.fields.get(field.name)
            if field_vt is None:
                unimplemented(
                    gb_type="Frozen dataclass with missing field",
                    context=f"dataclass={dataclass_cls.__name__}, field={field.name}",
                    explanation=f"Cannot reconstruct frozen dataclass: field '{field.name}' "
                    "was not tracked during tracing.",
                    hints=[*graph_break_hints.SUPPORTABLE],
                )
            if getattr(field, "kw_only", False):
                kw_args.append((field.name, field_vt))
            else:
                pos_args.append(field_vt)

        # Load the dataclass constructor
        codegen.add_push_null(
            lambda: codegen.append_output(
                codegen.create_load_const_unchecked(dataclass_cls)
            )
        )
        # Reconstruct all arguments
        for arg_vt in pos_args:
            codegen(arg_vt)
        for _, arg_vt in kw_args:
            codegen(arg_vt)
        # Call the constructor
        total_args = len(pos_args) + len(kw_args)
        if kw_args:
            kw_names = tuple(name for name, _ in kw_args)
            codegen.extend_output(
                codegen.create_call_function_kw(total_args, kw_names, push_null=False)
            )
        else:
            codegen.extend_output(create_call_function(total_args, False))

    # NB: This is called during __init__ for a frozen dataclass
    # use this to accumulate the most up-to-date field values
    def method_setattr_standard(
        self,
        tx: "InstructionTranslator",
        name: VariableTracker,
        value: VariableTracker,
        directly_update_dict: bool = False,
    ) -> VariableTracker:
        self.fields[name.as_python_constant()] = value
        return super().method_setattr_standard(tx, name, value, directly_update_dict)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value_type.__name__})"

    def is_python_hashable(self) -> Literal[True]:
        # TODO - Check corner cases like eq=False, hash=False etc
        return True

    def get_python_hash(self) -> int:
        return hash(tuple(arg.get_python_hash() for arg in self.fields.values()))

    def is_python_equal(self, other: object) -> bool:
        if not isinstance(other, FrozenDataClassVariable):
            return False
        is_class_same = self.python_type() is other.python_type()
        is_field_name_same = self.fields.keys() == other.fields.keys()
        is_field_value_same = all(
            value_a.is_python_equal(value_b)
            for value_a, value_b in zip(self.fields.values(), other.fields.values())
        )
        return is_class_same and is_field_name_same and is_field_value_same


class SourcelessGraphModuleVariable(UserDefinedObjectVariable):
    def __init__(
        self,
        value: object,
        **kwargs: Any,
    ) -> None:
        super().__init__(value, **kwargs)

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        fn_variable = VariableTracker.build(tx, self.value.forward.__func__)  # type: ignore[attr-defined]
        args = [self] + args
        return tx.inline_user_function_return(
            fn_variable,
            args,
            kwargs,
        )


class UserDefinedExceptionObjectVariable(UserDefinedObjectVariable):
    def __init__(self, value: object, **kwargs: Any) -> None:
        super().__init__(value, **kwargs)
        self.exc_vt = variables.ExceptionVariable(self.value_type, ())

    @property
    def fn(self) -> Callable[..., object]:
        return self.value_type

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if (
            name == "__init__"
            and (method := self._maybe_get_baseclass_method(name))
            and inspect.ismethoddescriptor(method)
            and len(kwargs) == 0
        ):
            self.exc_vt.args = tuple(args)
            # pyrefly: ignore[missing-attribute]
            self.value.args = args
            return variables.ConstantVariable(None)
        elif (
            name == "__setattr__"
            and len(args) == 2
            and args[0].is_constant_match(
                "__cause__", "__context__", "__suppress_context__", "__traceback__"
            )
        ):
            self.exc_vt.call_setattr(tx, args[0], args[1])
        elif name == "with_traceback":
            return self.exc_vt.call_method(tx, name, args, kwargs)
        return super().call_method(tx, name, args, kwargs)

    @property
    def __context__(self) -> "ConstantVariable":
        # type: ignore[return-value]
        return self.exc_vt.__context__

    @property
    def args(self) -> tuple[VariableTracker, ...]:
        return self.exc_vt.args

    def set_context(self, context: "variables.ExceptionVariable") -> None:
        return self.exc_vt.set_context(context)

    @property
    def exc_type(self) -> type[BaseException]:
        return self.exc_vt.exc_type

    @property
    def python_stack(self) -> traceback.StackSummary | None:
        return self.exc_vt.python_stack

    @python_stack.setter
    def python_stack(self, value: traceback.StackSummary) -> None:
        self.exc_vt.python_stack = value


class KeyedJaggedTensorVariable(UserDefinedObjectVariable):
    @staticmethod
    def is_matching_object(obj: object) -> bool:
        mod = sys.modules.get("torchrec.sparse.jagged_tensor")
        return mod is not None and type(obj) is mod.KeyedJaggedTensor

    def __init__(self, value: object, **kwargs: Any) -> None:
        from torchrec.sparse.jagged_tensor import (  # type: ignore[import-not-found]
            KeyedJaggedTensor,
        )

        assert type(value) is KeyedJaggedTensor
        super().__init__(value, **kwargs)

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        if (
            torch._dynamo.config.force_unspec_int_unbacked_size_like_on_torchrec_kjt
            and self.source is not None
            and name in ("_length_per_key", "_offset_per_key")
        ):
            with TracingContext.patch(force_unspec_int_unbacked_size_like=True):
                return super().var_getattr(tx, name)
        return super().var_getattr(tx, name)


class IntWrapperVariable(UserDefinedObjectVariable):
    # Dummy class to check if the object is an IntWrapper, and turn it into a
    # symint
    @staticmethod
    def is_matching_object(obj: object) -> bool:
        mod = sys.modules.get("torch.export.dynamic_shapes")
        return mod is not None and type(obj) is mod._IntWrapper


class RemovableHandleVariable(VariableTracker):
    REMOVED = -1

    def __init__(
        self,
        mutation_type: MutationType | None = None,
        # index of the registration in the side_effects owned register_hook/handle list, used during removal.
        idx: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.mutation_type = mutation_type
        self.idx = idx

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "remove":
            if self.idx != self.REMOVED:
                assert self.idx is not None
                tx.output.side_effects.remove_hook(self.idx)
                self.idx = self.REMOVED
            return variables.ConstantVariable.create(None)
        return super().call_method(tx, name, args, kwargs)

    def reconstruct(self, codegen: "PyCodegen") -> None:
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

    def python_type(self) -> type[object]:
        return RemovableHandleClass


class UserDefinedDictVariable(UserDefinedObjectVariable):
    """
    Represents user defined objects that are subclasses of dict/OrderedDict.

    Internally, it uses a ConstDictVariable to represent the dict part of the
    variable tracker. For everything else, it falls back to
    UserDefinedObjectVariable.
    """

    def __init__(
        self,
        value: object,
        dict_vt: ConstDictVariable | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(value, **kwargs)
        if dict_vt is None:
            assert self.source is None, (
                "dict_vt must be constructed by builder.py when source is present"
            )
            self._dict_vt = ConstDictVariable(
                {}, type(value), mutation_type=ValueMutationNew()
            )
        else:
            self._dict_vt = dict_vt
        self._dict_methods = dict_methods

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        method = self._maybe_get_baseclass_method(name)
        if method in self._dict_methods:
            # Dict subclasses can override __missing__ to provide fallback
            # behavior instead of raising a KeyError. This is used, for example,
            # by collections.Counter.
            try:
                return self._dict_vt.call_method(tx, name, args, kwargs)
            except ObservedKeyError:
                if (
                    name == "__getitem__"
                    and issubclass(self.python_type(), dict)
                    and self._maybe_get_baseclass_method("__missing__")
                ):
                    return self.call_method(tx, "__missing__", args, kwargs)
                else:
                    raise
        return super().call_method(tx, name, args, kwargs)

    def unpack_var_sequence(self, tx: "InstructionTranslator") -> list[VariableTracker]:
        if type(self.value).__iter__ in (  # type: ignore[attr-defined]
            dict.__iter__,
            collections.OrderedDict.__iter__,
        ):
            return self._dict_vt.unpack_var_sequence(tx)
        raise NotImplementedError

    def is_underlying_vt_modified(self, side_effects: "SideEffects") -> bool:
        return side_effects.is_modified(self._dict_vt)

    @property
    def user_cls(self) -> type[object]:
        return self._dict_vt.user_cls

    @property
    def items(self) -> dict[VariableTracker, VariableTracker]:
        return self._dict_vt.items

    def install_dict_keys_match_guard(self) -> None:
        return self._dict_vt.install_dict_keys_match_guard()

    def install_dict_contains_guard(
        self, tx: "InstructionTranslator", args: list[VariableTracker]
    ) -> None:
        return self._dict_vt.install_dict_contains_guard(tx, args)

    def is_python_hashable(self) -> Literal[False]:
        raise_on_overridden_hash(self.value, self)
        return False


class UserDefinedSetVariable(UserDefinedObjectVariable):
    """
    Represents user defined objects that are subclasses of set.

    Internally, it uses a SetVariable to represent the set part of the
    variable tracker. For everything else, it falls back to
    UserDefinedObjectVariable.
    """

    def __init__(
        self, value: object, set_vt: SetVariable | None = None, **kwargs: Any
    ) -> None:
        tx = kwargs.pop("tx", None)
        super().__init__(value, **kwargs)

        python_type = set if isinstance(value, set) else frozenset
        self._set_methods = set_methods if python_type is set else frozenset_methods

        if set_vt is None:
            assert self.source is None, (
                "set_vt must be constructed by builder.py when source is present"
            )
            if python_type is set:
                # set is initialized later
                self._set_vt = variables.SetVariable(
                    set(),
                    mutation_type=ValueMutationNew(),
                )
            else:
                init_args = kwargs.get("init_args", {})
                if tx is None:
                    tx = torch._dynamo.symbolic_convert.InstructionTranslator.current_tx()
                self._set_vt = variables.BuiltinVariable(python_type).call_function(  # type: ignore[assignment]
                    tx, init_args, {}
                )
        else:
            self._set_vt = set_vt

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        method = self._maybe_get_baseclass_method(name)
        if method in self._set_methods:
            return self._set_vt.call_method(tx, name, args, kwargs)
        return super().call_method(tx, name, args, kwargs)

    def as_python_constant(self) -> object:
        return self._set_vt.as_python_constant()

    def unpack_var_sequence(self, tx: "InstructionTranslator") -> list[VariableTracker]:
        if inspect.getattr_static(self.value, "__iter__") in (
            set.__iter__,
            frozenset.__iter__,
        ):
            return self._set_vt.unpack_var_sequence(tx)
        raise NotImplementedError

    @property
    def set_items(self) -> set[Any]:
        return self._set_vt.set_items

    @property
    def items(self) -> list[VariableTracker]:
        return self._set_vt.items

    def is_underlying_vt_modified(self, side_effects: "SideEffects") -> bool:
        return side_effects.is_modified(self._set_vt)

    def install_dict_keys_match_guard(self) -> None:
        return self._set_vt.install_dict_keys_match_guard()

    def install_dict_contains_guard(
        self, tx: "InstructionTranslator", args: list[VariableTracker]
    ) -> None:
        return self._set_vt.install_dict_contains_guard(tx, args)

    def is_python_hashable(self) -> bool:
        raise_on_overridden_hash(self.value, self)
        return self._set_vt.is_python_hashable()

    def get_python_hash(self) -> int:
        return self._set_vt.get_python_hash()

    def is_python_equal(self, other: object) -> bool:
        return isinstance(
            other, UserDefinedSetVariable
        ) and self._set_vt.is_python_equal(other._set_vt)


class UserDefinedListVariable(UserDefinedObjectVariable):
    """
    Represents user defined objects that are subclasses of lists.

    Internally, it uses a ListVariable to represent the list part of the
    variable tracker. For everything else, it falls back to
    UserDefinedObjectVariable.
    """

    def __init__(
        self, value: object, list_vt: Union["ListVariable", None] = None, **kwargs: Any
    ) -> None:
        from .lists import ListVariable

        super().__init__(value, **kwargs)
        if list_vt is None:
            assert self.source is None, (
                "list_vt must be constructed by builder.py when source is present"
            )
            self._list_vt = ListVariable([], mutation_type=ValueMutationNew())
        else:
            self._list_vt = list_vt

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        assert self._list_vt is not None
        method = self._maybe_get_baseclass_method(name)
        if method in list_methods:
            return self._list_vt.call_method(tx, name, args, kwargs)
        return super().call_method(tx, name, args, kwargs)

    def unpack_var_sequence(self, tx: "InstructionTranslator") -> list[VariableTracker]:
        assert self._list_vt is not None
        if type(self.value).__iter__ is list.__iter__:  # type: ignore[attr-defined]
            return self._list_vt.unpack_var_sequence(tx)
        raise NotImplementedError

    def is_underlying_vt_modified(self, side_effects: "SideEffects") -> bool:
        return side_effects.is_modified(self._list_vt)

    def is_python_hashable(self) -> Literal[False]:
        raise_on_overridden_hash(self.value, self)
        return False


class UserDefinedTupleVariable(UserDefinedObjectVariable):
    """
    Represents user defined objects that are subclasses of tuple.

    Internally, it uses a TupleVariable to represent the tuple part of the
    variable tracker. For everything else, it falls back to
    UserDefinedObjectVariable.
    """

    _nonvar_fields = UserDefinedObjectVariable._nonvar_fields

    def __init__(self, value, tuple_vt=None, init_args=None, **kwargs):  # type: ignore[all]
        from .lists import TupleVariable

        tx = kwargs.pop("tx", None)
        super().__init__(value, init_args=init_args, **kwargs)
        if tuple_vt is None:
            assert self.source is None, (
                "tuple_vt must be constructed by builder.py when source is present"
            )
            assert init_args, "init_args must be provided when tuple_vt is None"
            # Emulate `tuple.__new__`
            # https://github.com/python/cpython/blob/3.11/Objects/tupleobject.c#L697-L710
            #
            # TODO this duplicates the logic in `BuiltinVariable(tuple)`
            if tx is None:
                from torch._dynamo.symbolic_convert import InstructionTranslator

                tx = InstructionTranslator.current_tx()
            elems = init_args[0].force_unpack_var_sequence(tx)
            self._tuple_vt = TupleVariable(elems, mutation_type=ValueMutationNew())
        else:
            self._tuple_vt = tuple_vt

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        assert self._tuple_vt is not None
        method = self._maybe_get_baseclass_method(name)
        if method in tuple_methods:
            return self._tuple_vt.call_method(tx, name, args, kwargs)
        return super().call_method(tx, name, args, kwargs)

    def unpack_var_sequence(self, tx: "InstructionTranslator") -> list[VariableTracker]:
        assert self._tuple_vt is not None
        if type(self.value).__iter__ is tuple.__iter__:  # type: ignore[attr-defined]
            return self._tuple_vt.unpack_var_sequence(tx)
        raise NotImplementedError

    def is_python_hashable(self) -> bool:
        raise_on_overridden_hash(self.value, self)
        return self._tuple_vt.is_python_hashable()

    def get_python_hash(self) -> int:
        return self._tuple_vt.get_python_hash()

    def is_python_equal(self, other: object) -> bool:
        return isinstance(
            other, UserDefinedTupleVariable
        ) and self._tuple_vt.is_python_equal(other._tuple_vt)


class MutableMappingVariable(UserDefinedObjectVariable):
    def __init__(self, value: object, **kwargs: Any) -> None:
        super().__init__(value, **kwargs)
        self.generic_dict_vt = ConstDictVariable({})

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        # A common pattern in the init code of MutableMapping objects is to
        # update the __dict__ attribute. To prevent graph break, we directly
        # return a ConstDictVariable for the __dict__attr.
        #
        # However, users can try to add a new attribute to the class using the
        # __dict__ attribute. To catch this, we save the ConstDictVariable for
        # the __dict__ and then lookup into this vt for each attr lookup.
        if name == "get" and type(self.value).get in (  # type: ignore[attr-defined]
            collections.abc.Mapping.get,
            dict.get,
        ):
            return variables.UserMethodVariable(polyfills.mapping_get, self)
        elif name == "__dict__" and self.source:
            self.generic_dict_vt = variables.LazyVariableTracker.create(  # type: ignore[assignment]
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
