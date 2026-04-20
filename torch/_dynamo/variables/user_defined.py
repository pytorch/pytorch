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
from torch.utils._pytree import GetAttrKey, is_structseq_class

from .. import config, graph_break_hints, polyfills, variables
from ..bytecode_transformation import create_call_function
from ..create_parameter_op import do_not_convert_to_tracable_parameter
from ..exc import (
    handle_observed_exception,
    ObservedAttributeError,
    ObservedKeyError,
    ObservedTypeError,
    ObservedUserStopIteration,
    raise_observed_exception,
    raise_type_error,
    unimplemented,
)
from ..graph_bytecode_inputs import get_external_object_by_index
from ..guards import GuardBuilder, install_guard
from ..source import (
    AttrSource,
    CallFunctionNoArgsSource,
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
from .base import MutationType, NO_SUCH_SUBOBJ, ValueMutationNew, VariableTracker
from .dicts import ConstDictVariable
from .hashable import HashableTracker
from .sets import SetVariable


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

    from .dicts import DunderDictVariable
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


def is_pydantic_dataclass_cls(value: object) -> bool:
    return (
        inspect.isclass(value)
        and dataclasses.is_dataclass(value)
        and "__is_pydantic_dataclass__" in getattr(value, "__dict__", {})
    )


# Types whose instances are data descriptors (have __get__ + (__set__ or __delete__)).
# CPython invokes data descriptors found on the type MRO *before* checking
# the instance __dict__.  This set is used by is_data_descriptor for a fast
# O(1) check before falling back to the generic hasattr probe.
KNOWN_DATA_DESCRIPTOR_TYPES: frozenset[type] = frozenset(
    {
        property,
        _collections._tuplegetter,
        types.MemberDescriptorType,
        types.GetSetDescriptorType,
    }
)


def is_data_descriptor(obj: object) -> bool:
    """Return True if *obj* is a data descriptor (has __get__ and (__set__ or __delete__))."""
    tp = type(obj)
    if tp in KNOWN_DATA_DESCRIPTOR_TYPES:
        return True
    return hasattr(tp, "__get__") and (
        hasattr(tp, "__set__") or hasattr(tp, "__delete__")
    )


class UserDefinedVariable(VariableTracker):
    value: object

    def _maybe_get_baseclass_method(self, name: str) -> Any:
        """Get method from the base class if not overridden in value's __dict__."""
        if name not in getattr(self.value, "__dict__", {}):
            try:
                return inspect.getattr_static(type(self.value), name)
            except AttributeError:
                pass
        return None


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
            int.__new__,
            float.__new__,
            str.__new__,
        }
        return c_new_fns.union(exceptions)

    @staticmethod
    def is_supported_new_method(value: object) -> bool:
        if value in UserDefinedClassVariable.supported_c_new_functions():
            return True
        # Structseq types each define their own C tp_new.
        owner = getattr(value, "__self__", None)
        return isinstance(owner, type) and is_structseq_class(owner)

    def can_constant_fold_through(self) -> bool:
        if self.value in self._constant_fold_classes():
            return True
        # Enum class calls (e.g., Color(1)) are value lookups that return
        # existing singleton members, so they can always be constant-folded.
        return isinstance(self.value, type) and issubclass(self.value, enum.Enum)

    def lookup_cls_mro_attr(self, name: str) -> object:
        """Walk cls.__mro__ only (not the metaclass chain) to find *name*."""
        for base in self.value.__mro__:
            if name in base.__dict__:
                return base.__dict__[name]
        return NO_SUCH_SUBOBJ

    def lookup_metaclass_attr(self, name: str) -> object:
        """Walk type(cls).__mro__ (the metaclass chain) to find *name*."""
        for base in type(self.value).__mro__:
            if name in base.__dict__:
                return base.__dict__[name]
        return NO_SUCH_SUBOBJ

    def bool_impl(
        self,
        tx: "InstructionTranslator",
    ) -> "VariableTracker":
        from .constant import ConstantVariable

        # bool() on a class consults the metaclass __bool__.
        # If the metaclass is the default `type`, all classes are truthy.
        metaclass = type(self.value)
        if hasattr(metaclass, "__bool__") and metaclass is not type:
            return self.call_method(tx, "__bool__", [], {})
        return ConstantVariable.create(True)

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        source = AttrSource(self.source, name) if self.source is not None else None

        # --- Dynamo-specific pre-checks ---

        # Wrap OrderedDict/defaultdict.fromkeys as GetAttrVariable so it's
        # handled uniformly in call_method().
        if (
            self.value in {collections.OrderedDict, collections.defaultdict}
            and name == "fromkeys"
        ):
            return super().var_getattr(tx, name)

        # Custom metaclasses that override __getattribute__ replace the entire
        # lookup algorithm; bail out for those. Standard metaclasses (ABCMeta,
        # EnumType, etc.) that don't override __getattribute__ use
        # type.__getattribute__ which is the algorithm we implement below.
        metacls = type(self.value)
        if metacls is not type and "__getattribute__" in metacls.__dict__:
            unimplemented(
                gb_type="Custom metaclass with __getattribute__",
                context=f"type({self.value}) = {metacls}",
                explanation="Dynamo does not trace attribute access on classes whose "
                "metaclass overrides __getattribute__",
                hints=graph_break_hints.SUPPORTABLE,
            )

        # ---- CPython type_getattro algorithm ----
        # https://github.com/python/cpython/blob/3.13/Objects/typeobject.c#L5417-L5505
        # 1. meta_attr = lookup name in type(cls).__mro__  (metaclass chain)
        # 2. if meta_attr is a DATA descriptor → invoke
        # 3. cls_attr = lookup name in cls.__mro__  (class chain)
        # 4. if cls_attr has __get__ → invoke cls_attr.__get__(None, cls)
        # 5. if cls_attr exists (plain) → return as-is
        # 6. if meta_attr is a non-data descriptor or plain → return
        # 7. raise AttributeError

        # Step 1-2: Metaclass data descriptors.
        # For type(cls) is type, these are C-level getset/member descriptors
        # for __dict__, __mro__, __name__, __qualname__, __doc__, etc.
        meta_attr = self.lookup_metaclass_attr(name)
        if meta_attr is not NO_SUCH_SUBOBJ and is_data_descriptor(meta_attr):
            return self.resolve_meta_data_descriptor(tx, name, meta_attr, source)

        # Step 3-5: Class MRO lookup.
        cls_attr = self.lookup_cls_mro_attr(name)
        if cls_attr is not NO_SUCH_SUBOBJ:
            if hasattr(type(cls_attr), "__get__"):
                # Step 4: Descriptor — invoke __get__(None, cls).
                return self.resolve_cls_descriptor(tx, name, cls_attr, source)
            # Step 5: Plain attribute.
            return self.resolve_cls_plain_attr(tx, name, cls_attr, source)

        # Step 6: Metaclass non-data descriptor or plain attr.
        # These are non-data descriptors on the metaclass (e.g. type.__call__,
        # type.__subclasses__, type.mro).  We use GetAttrVariable to defer to
        # runtime rather than VariableTracker.build, because build would create
        # a variable for the raw C-level descriptor which then fails when
        # called (e.g. type.__subclasses__ is a method_descriptor that dynamo
        # can't trace).  GetAttrVariable defers the access and lets
        # call_method handle it.
        if meta_attr is not NO_SUCH_SUBOBJ:
            return variables.GetAttrVariable(self, name, type(meta_attr), source=source)

        # __getattr__ on metaclass (not part of type_getattro proper —
        # CPython handles this via slot_tp_getattr_hook).
        metacls = type(self.value)
        if metacls is not type:
            meta_getattr = self.lookup_metaclass_attr("__getattr__")
            if meta_getattr is not NO_SUCH_SUBOBJ and isinstance(
                meta_getattr, types.FunctionType
            ):
                return variables.UserMethodVariable(meta_getattr, self).call_function(
                    tx, [variables.ConstantVariable.create(name)], {}
                )

        # Step 7: AttributeError.
        raise_observed_exception(
            AttributeError,
            tx,
            args=[f"type object '{self.value.__name__}' has no attribute '{name}'"],
        )

    def resolve_meta_data_descriptor(
        self,
        tx: "InstructionTranslator",
        name: str,
        meta_attr: object,
        source: Source | None,
    ) -> VariableTracker:
        """Handle data descriptors from the metaclass MRO (type.__dict__ slots)."""
        if name == "__dict__":
            return VariableTracker.build(
                tx,
                self.value.__dict__,
                source=self.source and AttrSource(self.source, "__dict__"),
            )
        if name == "__mro__":
            attr_source = self.source and TypeMROSource(self.source)
            return VariableTracker.build(tx, self.value.__mro__, attr_source)
        # __name__, __qualname__, __doc__, __module__, __bases__,
        # __abstractmethods__, etc. — all C-level getset descriptors on type.
        resolved = type.__getattribute__(self.value, name)
        if source:
            return VariableTracker.build(tx, resolved, source)
        from . import ConstantVariable

        if ConstantVariable.is_literal(resolved):
            return VariableTracker.build(tx, resolved)
        return variables.GetAttrVariable(self, name, type(resolved), source=source)

    def resolve_cls_descriptor(
        self,
        tx: "InstructionTranslator",
        name: str,
        cls_attr: object,
        source: Source | None,
    ) -> VariableTracker:
        """Handle descriptors found in cls.__mro__."""
        if isinstance(cls_attr, staticmethod):
            return VariableTracker.build(tx, cls_attr.__get__(self.value), source)

        if isinstance(cls_attr, classmethod):
            if isinstance(cls_attr.__func__, property):
                fget_vt = VariableTracker.build(tx, cls_attr.__func__.fget)
                return fget_vt.call_function(tx, [self], {})
            return variables.UserMethodVariable(cls_attr.__func__, self, source=source)

        if isinstance(cls_attr, types.ClassMethodDescriptorType):
            func = cls_attr.__get__(None, self.value)
            return VariableTracker.build(tx, func, source)

        # property and _tuplegetter accessed on the class return the
        # descriptor itself (descriptor.__get__(None, cls) is descriptor).
        # Build directly — no need to invoke __get__.
        if isinstance(cls_attr, (property, _collections._tuplegetter)):
            if source:
                return VariableTracker.build(tx, cls_attr, source)
            return UserDefinedObjectVariable(cls_attr)

        # Comparison dunders inherited from object — defer to runtime.
        if name in cmp_name_to_op_mapping and not isinstance(
            cls_attr, types.FunctionType
        ):
            return variables.GetAttrVariable(
                self, name, py_type=type(cls_attr), source=source
            )

        # User-defined descriptor with Python __get__.
        # For torch-internal classes or attributes in the class's own __dict__,
        # defer descriptor invocation to runtime via VariableTracker.build to
        # avoid compile-time side effects (e.g. deprecation warnings from
        # _ClassPropertyDescriptor on torch.FloatStorage.dtype).
        get_fn = inspect.getattr_static(type(cls_attr), "__get__", None)
        if isinstance(get_fn, types.FunctionType):
            if source and (
                name in getattr(self.value, "__dict__", {})
                or self.value.__module__.startswith("torch.")
                or self.value.__module__ == "torch"
            ):
                return VariableTracker.build(tx, cls_attr, source)
            return self.invoke_cls_descriptor_get(tx, name, cls_attr, source)

        # C-level descriptors (WrapperDescriptor, MethodDescriptor, etc.)
        # Build directly when the attribute lives in the class's own __dict__
        # or the class belongs to torch (needed for e.g. torch.Tensor.dim).
        # OrderedDict's C-level methods are handled at runtime.
        if inspect.ismethoddescriptor(cls_attr) or is_wrapper_or_member_descriptor(
            cls_attr
        ):
            if (
                source
                and self.value is not collections.OrderedDict
                and (
                    name in getattr(self.value, "__dict__", {})
                    or self.value.__module__.startswith("torch.")
                    or self.value.__module__ == "torch"
                )
            ):
                return VariableTracker.build(tx, cls_attr, source)
            return variables.GetAttrVariable(self, name, type(cls_attr), source=source)

        # Everything else: FunctionType, etc.
        return VariableTracker.build(tx, cls_attr, source)

    def resolve_cls_plain_attr(
        self,
        tx: "InstructionTranslator",
        name: str,
        cls_attr: object,
        source: Source | None,
    ) -> VariableTracker:
        """Handle non-descriptor attributes from cls.__mro__."""
        if name == "__new__" and UserDefinedClassVariable.is_supported_new_method(
            cls_attr
        ):
            return super().var_getattr(tx, name)
        if self.value is collections.OrderedDict:
            return variables.GetAttrVariable(self, name, py_type=type(cls_attr))
        return VariableTracker.build(tx, cls_attr, source)

    def invoke_cls_descriptor_get(
        self,
        tx: "InstructionTranslator",
        name: str,
        descriptor: object,
        source: Source | None,
    ) -> VariableTracker:
        """Trace a class-MRO descriptor's __get__(None, cls) call."""
        from .constant import ConstantVariable

        descriptor_source = None
        descriptor_get_source = None
        if self.source:
            descriptor_source = AttrSource(self.source, name)
            descriptor_get_source = AttrSource(TypeSource(descriptor_source), "__get__")
            descriptor_var = VariableTracker.build(tx, descriptor, descriptor_source)
        else:
            descriptor_var = UserDefinedObjectVariable(descriptor)

        none_var = ConstantVariable.create(None)
        return variables.UserMethodVariable(
            descriptor.__get__.__func__,  # type: ignore[union-attr]
            descriptor_var,
            source=descriptor_get_source,
        ).call_function(tx, [none_var, self], {})

    def len_impl(self, tx: "InstructionTranslator") -> VariableTracker:
        m = self._maybe_get_baseclass_method("__len__")
        if m:
            source = self.source and AttrSource(self.source, "__len__")
            return variables.UserMethodVariable(
                m, self, source_fn=source
            ).call_function(tx, [], {})
        raise_type_error(tx, f"object of type {self.python_type_name()} has no length")

    def sq_length(self, tx: "InstructionTranslator") -> VariableTracker:
        return self.len_impl(tx)

    def mp_length(self, tx: "InstructionTranslator") -> VariableTracker:
        return self.len_impl(tx)

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
        from .builder import SourcelessBuilder

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
            return variables.DictBuiltinVariable.call_custom_dict_fromkeys(
                tx, self.value, *args, **kwargs
            )
        elif self.value is collections.OrderedDict and name == "move_to_end":
            return args[0].call_method(tx, name, [*args[1:]], kwargs)
        elif name == "__len__" and len(args) == 1 and not kwargs:
            from .object_protocol import generic_len

            return generic_len(tx, args[0])
        elif name == "__eq__" and len(args) == 1 and hasattr(args[0], "value"):
            return VariableTracker.build(tx, self.value == args[0].value)
        elif name == "__ne__" and len(args) == 1 and hasattr(args[0], "value"):
            return VariableTracker.build(tx, self.value != args[0].value)
        elif issubclass(self.value, dict) and name != "__new__":
            # __new__ is handled below
            return SourcelessBuilder.create(tx, dict).call_method(
                tx, name, args, kwargs
            )
        elif issubclass(self.value, (set, frozenset)) and name != "__new__":
            # __new__ is handled below
            return SourcelessBuilder.create(tx, set).call_method(tx, name, args, kwargs)
        elif (
            len(args) == 1
            and isinstance(args[0], variables.GenericContextWrappingVariable)
            and name == "__enter__"
        ):
            return args[0].enter(tx)
        elif name == "__new__" and UserDefinedClassVariable.is_supported_new_method(
            self.value.__new__
        ):
            # Some C-level tp_new functions (dict.__new__, set.__new__) ignore
            # extra args — only the type arg matters.  Pass init_args=[] for
            # those so reconstruction emits base_cls.__new__(cls) without
            # unreconstructable args (e.g. generators).  Other tp_new functions
            # (tuple.__new__, BaseException.__new__) use the extra args.
            new_fn = self.value.__new__
            if new_fn in (dict.__new__, set.__new__):
                init_args: list[VariableTracker] = []
            else:
                init_args = list(args[1:])
            return tx.output.side_effects.track_new_user_defined_object(
                self,
                args[0],
                init_args,
            )
        elif name == "__setattr__" and self.ban_mutation:
            unimplemented(
                gb_type="Class attribute mutation when the __dict__ was already materialized",
                context=str(self.value),
                explanation="Dyanmo does not support tracing mutations on a class when its __dict__ is materialized",
                hints=graph_break_hints.SUPPORTABLE,
            )

        # Dispatch dunder methods defined on the metaclass (e.g., EnumType.__contains__).
        # In Python, `x in Color` calls `type(Color).__contains__(Color, x)`.
        metaclass = type(self.value)
        if metaclass is not type:
            # Look up the method on the metaclass MRO, not the class MRO
            for klass in metaclass.__mro__:
                if name in klass.__dict__:
                    method = klass.__dict__[name]
                    if isinstance(method, types.FunctionType):
                        source = self.source and AttrSource(self.source, name)
                        return variables.UserMethodVariable(
                            method, self, source=source
                        ).call_function(tx, args, kwargs)
                    break

        return super().call_method(tx, name, args, kwargs)

    def unpack_var_sequence(
        self, tx: "InstructionTranslator"
    ) -> list["VariableTracker"]:
        if isinstance(self.value, type) and issubclass(self.value, enum.Enum):
            return [VariableTracker.build(tx, item) for item in self.value]
        raise NotImplementedError

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from ..side_effects import SideEffects
        from .builder import SourcelessBuilder, wrap_fx_proxy
        from .ctx_manager import GenericContextWrappingVariable

        constant_args = check_constant_args(args, kwargs)

        if torch.distributed.is_available() and self.value is torch.distributed.P2POp:
            if not config.enable_p2p_compilation:
                unimplemented(
                    gb_type="P2P compilation disabled for P2POp construction",
                    context="torch.distributed.P2POp",
                    explanation="P2P compilation is disabled.",
                    hints=[
                        "Set TORCHDYNAMO_ENABLE_P2P_COMPILATION=1 to enable.",
                    ],
                )
            var = tx.output.side_effects.track_new_user_defined_object(
                SourcelessBuilder.create(tx, object),
                self,
                [],
            )
            var.call_method(tx, "__init__", list(args), kwargs)  # type: ignore[arg-type]
            return var

        if self.can_constant_fold_through() and constant_args:
            # constant fold
            return VariableTracker.build(
                tx,
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
        elif self.value is collections.defaultdict:
            # defaultdict construction — use track_new_user_defined_object
            # which creates DefaultDictVariable. __init__ handler extracts
            # default_factory and populates items.
            from .builder import SourcelessBuilder

            result = tx.output.side_effects.track_new_user_defined_object(
                SourcelessBuilder.create(tx, dict),
                self,
                [],
            )
            result.call_method(tx, "__init__", list(args), kwargs)
            return result
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
            return variables.DictBuiltinVariable.call_custom_dict(
                tx, dict, *args, **kwargs
            )
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
                # pyrefly: ignore [implicit-any]
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
                raise_type_error(tx, "torch.cuda.device() requires a constant argument")
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

            return tx.inline_user_function_return(
                VariableTracker.build(
                    tx, polyfills.instantiate_user_defined_class_object
                ),
                [self, *arg_new],
                kwargs,
            )
        elif is_namedtuple_cls(self.value):
            if is_structseq_class(self.value):
                if kwargs or len(args) != 1:
                    raise_args_mismatch(
                        tx,
                        "torch.return_types",
                        "1 args and 0 kwargs",
                        f"{len(args)} args and {len(kwargs)} kwargs",
                    )
                # Structseq tp_new is a C function, so we can't trace into
                # it like namedtuples. Use track_new_user_defined_object
                # directly with self as both base_cls_vt and cls_vt.
                return tx.output.side_effects.track_new_user_defined_object(
                    self,
                    self,
                    list(args),
                )
            else:
                # Namedtuple __new__ is a Python function that calls
                # tuple.__new__(cls, (field_values,)). Let Dynamo trace
                # into it so default values and kwargs are handled by
                # the generated __new__ itself.
                return tx.inline_user_function_return(
                    VariableTracker.build(
                        tx, polyfills.instantiate_user_defined_class_object
                    ),
                    [self, *args],
                    kwargs,
                )
        elif self.value is torch.Size:
            # This simulates `THPSize_pynew`, the C impl for `Size.__new__`.
            from .lists import SizeVariable

            tup = SourcelessBuilder.create(tx, tuple).call_function(tx, args, kwargs)
            return SizeVariable(tup.items)  # type: ignore[missing-attribute]
        elif is_pydantic_dataclass_cls(self.value):
            # Pydantic populates dataclass fields through an external validator,
            # so tracing through the constructor misses the instance mutations.
            unimplemented(
                gb_type="Pydantic dataclass constructor",
                context=f"{self.value}",
                explanation="Dynamo graph breaks on pydantic dataclass constructors "
                "because validation mutates the instance outside traced bytecode.",
                hints=graph_break_hints.SUPPORTABLE,
            )
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
                from .lists import TupleVariable

                var_kwargs = ConstDictVariable(
                    {VariableTracker.build(tx, k): v for k, v in kwargs.items()}
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
                from .lists import TupleVariable

                # Register newly created event for reconstruction
                var_kwargs = ConstDictVariable(
                    {VariableTracker.build(tx, k): v for k, v in kwargs.items()}
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
        elif self.value is types.MappingProxyType and len(args) == 1:
            # types.MappingProxyType is a read-only proxy of the dict. If the
            # original dict changes, the changes are reflected in proxy as well.
            dict_arg = args[0]
            if isinstance(dict_arg, variables.UserDefinedDictVariable):
                dict_arg = dict_arg._base_vt
            if isinstance(dict_arg, ConstDictVariable):
                return variables.MappingProxyVariable(dict_arg)
        elif SideEffects.cls_supports_mutation_side_effects(self.value) and self.source:
            with do_not_convert_to_tracable_parameter():
                result = tx.inline_user_function_return(
                    VariableTracker.build(
                        tx, polyfills.instantiate_user_defined_class_object
                    ),
                    [self, *args],
                    kwargs,
                )
                return result

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
            install_guard(
                self.source.make_guard(
                    functools.partial(GuardBuilder.HASATTR, attr=name)
                )
            )
        return VariableTracker.build(tx, hasattr(self.value, name))

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

    def get_real_python_backed_value(self) -> object:
        return self.value


class UserDefinedExceptionClassVariable(UserDefinedClassVariable):
    @property
    def fn(self) -> type[object]:
        return self.value

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from .builder import SourcelessBuilder

        if self.source is None:
            # NB: If source is added via side effects, create the exception
            # object through side_effects as well. See FrozenDataClass creation
            var = tx.output.side_effects.track_new_user_defined_object(
                SourcelessBuilder.create(tx, BaseException),
                self,
                list(args),
            )
            var.call_method(tx, "__init__", list(args), dict(kwargs))
            return var
        return super().call_function(tx, args, kwargs)


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
    # NB: it is probably not important for the example_value to be exactly correct,
    # we just need the right type
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

    # VT representing the base built-in type's data for subclassed built-in types
    # (e.g., ConstDictVariable for dict subclasses, ListVariable for list subclasses).
    # None for plain user-defined objects that don't subclass a built-in container.
    _base_vt: VariableTracker | None = None

    # Set of base class methods that can be delegated to _base_vt.
    # Used to check whether a method is overridden before delegating.
    _base_methods: set[Any] | None = None

    _nonvar_fields = {
        "value",
        "value_type",
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

        # This records the attributes that were modified via instance
        # `__dict__` directly, rather than the normal setattr path.
        self.dict_vt: DunderDictVariable | None = None

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

    def get_dict_vt(self, tx: "InstructionTranslator") -> "DunderDictVariable":
        if self.dict_vt is None:
            self.dict_vt = variables.DunderDictVariable.create(tx, self)
        return self.dict_vt

    def is_base_vt_modified(self, side_effects: "SideEffects") -> bool:
        if self._base_vt is not None:
            return side_effects.is_modified(self._base_vt)
        return False

    def python_type(self) -> type:
        return self.value_type  # type: ignore[return-value]

    def get_real_python_backed_value(self) -> object:
        return self.value

    def as_python_constant(self) -> object:
        if isinstance(
            self.value,
            (enum.Enum, torch.DispatchKey, torch._C._functorch.TransformType),
        ):
            return self.value

        if self.is_pytree_constant_class and self.source:
            # NOTE pytree constants created in the torch.compile region will
            # NOT be guarded (even though they have a source set)
            return self.value
            # TODO else try reconstructing the object by, e.g., leveraging side
            # effects and `as_python_constant`.

        # Special case for _MaskModWrapper during legacy export: Dynamo creates
        # objects via __new__ without calling __init__, so self.value.fn is unset.
        # Reconstruct from the tracked side-effect attribute instead.
        from torch.nn.attention.flex_attention import _MaskModWrapper

        if isinstance(self.value, _MaskModWrapper):
            from torch._dynamo.symbolic_convert import InstructionTranslator

            tx = InstructionTranslator.current_tx()
            if tx is not None and tx.export:
                fn_vt = tx.output.side_effects.load_attr(self, "fn", deleted_ok=True)
                if fn_vt is not None:
                    # Let as_python_constant() raise the proper exception
                    # (e.g., ClosureConversionError for non-constant closures)
                    fn = fn_vt.as_python_constant()
                    return _MaskModWrapper(fn)

        return super().as_python_constant()

    def as_proxy(self) -> object:
        if isinstance(self.value, enum.Enum):
            if isinstance(self.value, int):
                return int(self.value)
            return self.value
        return super().as_proxy()

    def guard_as_python_constant(self) -> object:
        if self.source:
            install_guard(self.source.make_guard(GuardBuilder.ID_MATCH))
            return self.value
        return super().guard_as_python_constant()

    def bool_impl(
        self,
        tx: "InstructionTranslator",
    ) -> "VariableTracker | None":
        # Mirrors slot_nb_bool:
        # https://github.com/python/cpython/blob/c09ccd9c429/Objects/typeobject.c#L9408-L9458
        if self._maybe_get_baseclass_method("__bool__"):
            result = self.call_method(tx, "__bool__", [], {})
            if result.is_python_constant():
                result_value = result.as_python_constant()
                if not isinstance(result_value, bool):
                    raise_observed_exception(
                        TypeError,
                        tx,
                        args=[
                            f"__bool__ should return bool, returned {type(result_value).__name__}"
                        ],
                    )
            return result
        return None

    def nb_index_impl(
        self,
        tx: "InstructionTranslator",
    ) -> VariableTracker:
        # CPython: PyNumber_Index checks tp_as_number->nb_index.
        # For user-defined types, __index__ in tp_dict means nb_index is set.
        type_attr = inspect.getattr_static(type(self.value), "__index__", None)
        if type_attr is None:
            return super().nb_index_impl(tx)
        source = self.source and self.get_source_by_walking_mro(tx, "__index__")
        method_var = self.resolve_type_attr(tx, "__index__", type_attr, source)
        result = method_var.call_function(tx, [], {})
        # CPython validates that __index__ returns an int.
        # https://github.com/python/cpython/blob/c09ccd9c429/Objects/abstract.c#L1433-L1438
        if result.is_python_constant() and not isinstance(
            result.as_python_constant(), int
        ):
            raise_observed_exception(
                TypeError,
                tx,
                args=[
                    f"__index__ returned non-int (type {type(result.as_python_constant()).__name__})"
                ],
            )
        return result

    def nb_int_impl(
        self,
        tx: "InstructionTranslator",
    ) -> VariableTracker:
        # CPython: slot_nb_int calls __int__(), PyNumber_Long validates the return type.
        # https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1538-L1550
        source = self.source and self.get_source_by_walking_mro(tx, "__int__")
        method_var = self.resolve_type_attr(
            tx,
            "__int__",
            inspect.getattr_static(type(self.value), "__int__"),
            source,
        )
        result = method_var.call_function(tx, [], {})
        if not issubclass(result.python_type(), int):
            raise_observed_exception(
                TypeError,
                tx,
                args=[
                    f"__int__ returned non-int (type {result.python_type().__name__})"
                ],
            )
        return result

    def nb_float_impl(
        self,
        tx: "InstructionTranslator",
    ) -> VariableTracker:
        # CPython: slot_nb_float calls __float__(), PyNumber_Float validates the return type.
        # https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#L1647-L1658
        source = self.source and self.get_source_by_walking_mro(tx, "__float__")
        method_var = self.resolve_type_attr(
            tx,
            "__float__",
            inspect.getattr_static(type(self.value), "__float__"),
            source,
        )
        result = method_var.call_function(tx, [], {})
        if not issubclass(result.python_type(), float):
            raise_observed_exception(
                TypeError,
                tx,
                args=[
                    f"__float__ returned non-float (type {result.python_type().__name__})"
                ],
            )
        return result

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

    def mp_subscript_impl(
        self,
        tx: "InstructionTranslator",
        key: VariableTracker,
    ) -> VariableTracker:
        # PyObject_GetItem: https://github.com/python/cpython/blob/62a6e898e01/Objects/abstract.c#L155-L206
        method = self._maybe_get_baseclass_method("__getitem__")
        if (
            self._base_vt is not None
            and self._base_methods is not None
            and method in self._base_methods
        ):
            return self._base_vt.mp_subscript_impl(tx, key)
        if isinstance(method, types.FunctionType):
            source_fn = self.source and self.get_source_by_walking_mro(
                tx, "__getitem__"
            )
            return variables.UserMethodVariable(
                method, self, source_fn=source_fn, source=self.source
            ).call_function(tx, [key], {})
        return super().mp_subscript_impl(tx, key)

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[Any],
        kwargs: dict[str, Any],
    ) -> VariableTracker:
        from .. import trace_rules
        from . import UserMethodVariable
        from .constant import ConstantVariable

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
                    return VariableTracker.build(tx, NotImplemented)

                # TODO(anijain2305) - Identity checking should already be a part
                # of the cmp_eq  polyfill function.
                return VariableTracker.build(tx, self.value is other.value)

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

            # torch.Generator methods like manual_seed(), get_state(), etc.
            # are stateful RNG operations that cannot be soundly traced.
            if (
                isinstance(self.value, torch._C.Generator)
                and name in trace_rules._GENERATOR_METHODS_THAT_GRAPH_BREAK
            ):
                unimplemented(
                    gb_type="torch.Generator method",
                    context=f"torch.Generator.{name}",
                    explanation=f"torch.Generator.{name}() is a stateful RNG "
                    "operation that cannot be soundly traced in the FX graph.",
                    hints=[*graph_break_hints.FUNDAMENTAL],
                )

            # Delegate to _base_vt for non-overridden base-class methods
            if (
                self._base_vt is not None
                and self._base_methods is not None
                and method in self._base_methods
            ):
                return self._base_vt.call_method(tx, name, args, kwargs)

            # check for methods implemented in C++
            if isinstance(method, types.FunctionType):
                source = self.source
                source_fn = None
                if source:
                    source_fn = self.get_source_by_walking_mro(tx, name)
                # TODO(jansel): add a guard to check for monkey patching?
                from ..mutation_guard import unpatched_nn_module_init

                if method is torch.nn.Module.__init__:
                    method = unpatched_nn_module_init
                return UserMethodVariable(
                    method, self, source_fn=source_fn, source=source
                ).call_function(tx, args, kwargs)  # type: ignore[arg-type]

            if method is list.__len__ and self.source and not (args or kwargs):
                install_guard(self.source.make_guard(GuardBuilder.SEQUENCE_LENGTH))
                return VariableTracker.build(tx, len(self.value))  # type: ignore[arg-type]

        return super().call_method(tx, name, args, kwargs)

    def len_impl(self, tx: "InstructionTranslator") -> VariableTracker:
        method = self._maybe_get_baseclass_method("__len__")
        if method is not None:
            type_attr = self.lookup_class_mro_attr("__len__")
            source = self.source and self.get_source_by_walking_mro(tx, "__len__")
            method_var = self.resolve_type_attr(tx, "__len__", type_attr, source)
            if not isinstance(method_var, variables.GetAttrVariable):
                return method_var.call_function(tx, [], {})

        unimplemented(
            gb_type="Cannot trace user-defined __len__",
            context=f"{self.python_type_name()}.__len__()",
            explanation=(
                f"Dynamo cannot trace len() on {self.python_type_name()} because the __len__ "
                "method is either not traceable (e.g., defined in C or built-in) or returns a "
                "non-constant value."
            ),
            hints=[
                *graph_break_hints.SUPPORTABLE,
            ],
        )

    def sq_length(self, tx: "InstructionTranslator") -> VariableTracker:
        if (
            self._base_vt is not None
            and self._base_methods is not None
            and self._maybe_get_baseclass_method("__len__") in self._base_methods
        ):
            return self._base_vt.sq_length(tx)
        return self.len_impl(tx)

    def mp_length(self, tx: "InstructionTranslator") -> VariableTracker:
        if (
            self._base_vt is not None
            and self._base_methods is not None
            and self._maybe_get_baseclass_method("__len__") in self._base_methods
        ):
            return self._base_vt.mp_length(tx)
        return self.len_impl(tx)

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

        if (
            torch.distributed.is_available()
            and type(self.value) is torch.distributed.P2POp
            and (
                tx.output.side_effects.has_pending_mutation_of_attr(self, name_str)
                or name_str in self.value.__dict__
            )
        ):
            unimplemented(
                gb_type="P2POp mutation",
                context=f"object={self}, name={name}, value={value}",
                explanation="Dynamo does not support mutating torch.distributed.P2POp instances.",
                hints=[
                    "Construct a new torch.distributed.P2POp instead of mutating an existing one inside torch.compile.",
                ],
            )

        if name_str == "__class__":
            unimplemented(
                gb_type="__class__ assignment on user-defined object",
                context=f"object={self}, value={value}",
                explanation="Dynamo does not support reassigning __class__ on user-defined objects.",
                hints=[
                    "Move the __class__ assignment outside of the torch.compile region.",
                ],
            )

        if directly_update_dict:
            self.get_dict_vt(tx).setitem(name_str, value)
        else:
            tmp = self.try_get_descritor_and_setter_py_func(name_str)
            if tmp:
                descriptor, setter = tmp
                # Emulate
                # https://github.com/python/cpython/blob/3.11/Objects/object.c#L1371-L1452
                desc_source = None
                func_source = None
                if self.cls_source:
                    desc_source = self.get_source_by_walking_mro(tx, name_str)
                    # use `type(...)` to ignore instance attrs.
                    func_source = AttrSource(TypeSource(desc_source), "__set__")
                desc_var = VariableTracker.build(tx, descriptor, desc_source)
                func_var = VariableTracker.build(tx, setter, func_source, realize=True)
                if isinstance(descriptor, property):
                    args = [self, value]  # property.fset(self, value)
                else:
                    args = [desc_var, self, value]  # __set__(desc, self, value)
                return func_var.call_function(tx, args, {})

            # Handle Python property descriptors whose __set__ is a C slot
            # wrapper (not a Python function), which the above check misses.
            # Mirrors the property getter handling in var_getattr.
            descriptor = inspect.getattr_static(type(self.value), name_str, None)
            if isinstance(descriptor, property) and descriptor.fset is not None:
                fset_source = None
                if self.cls_source:
                    fset_source = AttrSource(
                        self.get_source_by_walking_mro(tx, name_str), "fset"
                    )
                fset_var = VariableTracker.build(
                    tx, descriptor.fset, source=fset_source
                )
                return fset_var.call_function(tx, [self, value], {})

            # NOTE: else we assume the descriptor (if any) has a
            # side-effect-free `__set__` as far as Dynamo tracing is concerned.

        # If the code reaches here, the attribute is either:
        #  1) a slot descriptor
        #  2) a plain attribute with no descriptor
        # If the object has no __dict__, only slot descriptors (member_descriptor)
        # allow mutation. Any other attribute assignment raises AttributeError.
        if not hasattr(self.value, "__dict__"):
            descriptor = self.lookup_class_mro_attr(name_str)
            if not inspect.ismemberdescriptor(descriptor):
                error_msg = VariableTracker.build(
                    tx,
                    f"'{type(self.value).__name__}' object has no attribute '{name_str}'",
                )
                raise_observed_exception(AttributeError, tx, args=[error_msg])

        tx.output.side_effects.store_attr(self, name_str, value)
        return variables.ConstantVariable.create(None)

    def needs_slow_setattr(self) -> bool:
        return not is_standard_setattr(
            inspect.getattr_static(self.value, "__setattr__", None)
        ) and not isinstance(self.value, threading.local)

    def unpack_var_sequence(self, tx: "InstructionTranslator") -> list[VariableTracker]:
        if self._base_vt is not None and self._base_methods is not None:
            iter_method = self._maybe_get_baseclass_method("__iter__")
            if iter_method is not None and iter_method in self._base_methods:
                return self._base_vt.unpack_var_sequence(tx)
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
        from .builder import SourcelessBuilder

        try:
            SourcelessBuilder.create(tx, iter).call_function(tx, [self], {})
            return True
        except ObservedTypeError:
            handle_observed_exception(tx)
            return False

    def force_unpack_var_sequence(
        self, tx: "InstructionTranslator"
    ) -> list[VariableTracker]:
        from .builder import SourcelessBuilder

        result = []
        iter_ = SourcelessBuilder.create(tx, iter).call_function(tx, [self], {})

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
                var = VariableTracker.build(tx, obj.mode)  # type: ignore[attr-defined]
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
            func_var = VariableTracker.build(tx, func, func_src, realize=True)
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
        #
        # C-level descriptors (getset_descriptor for __dict__, member_descriptor
        # for __slots__) are always safe to resolve — their __get__ is
        # implemented in C and doesn't run user code, so __getattribute__
        # overrides are irrelevant.  The NO_SUCH_SUBOBJ and
        # _is_c_defined_property cases DO require the absence of a custom
        # __getattribute__ because they fall back to
        # type(self.value).__getattribute__ which could be user-overridden.
        if inspect.ismemberdescriptor(subobj) or inspect.isgetsetdescriptor(subobj):
            subobj = type(self.value).__getattribute__(self.value, name)
        elif not self._object_has_getattribute and (
            subobj is NO_SUCH_SUBOBJ  # e.g., threading.local
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

    def lookup_class_mro_attr(self, name: str) -> object:
        """Walk type(obj).__mro__ to find *name* in the class hierarchy.

        This only searches the class chain (type(obj).__mro__), NOT the
        metaclass chain (type(type(obj)).__mro__).  The distinction matters
        because inspect.getattr_static conflates both chains — it can return
        metaclass descriptors (e.g. type.__dict__['__annotations__'], a
        getset_descriptor) when the attribute doesn't exist on the class MRO.
        Walking cls.__mro__ directly avoids that leak.
        """
        if name in self._subobj_from_class:
            return self._subobj_from_class[name]
        result = NO_SUCH_SUBOBJ
        for base in self.value.__class__.__mro__:
            if name in base.__dict__:
                result = base.__dict__[name]
                break
        self._subobj_from_class[name] = result
        return result

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
            elif self.dict_vt and self.dict_vt.contains(attr_name):
                return True
        return False

    def try_get_descritor_and_setter_py_func(
        self, attr_name: str
    ) -> tuple[object, object] | None:
        descriptor = inspect.getattr_static(type(self.value), attr_name, None)
        # Handle property descriptors with setters - call fset directly
        if isinstance(descriptor, property) and descriptor.fset is not None:
            return (descriptor, descriptor.fset)
        setter = inspect.getattr_static(type(descriptor), "__set__", None)
        if inspect.isfunction(setter):
            return (descriptor, setter)
        return None

    def has_key_in_generic_dict(self, tx: "InstructionTranslator", key: str) -> bool:
        if tx.output.side_effects.has_pending_mutation_of_attr(self, key):
            mutated_attr = tx.output.side_effects.load_attr(self, key, deleted_ok=True)
            return not isinstance(mutated_attr, variables.DeletedVariable)

        # TODO(guilhermeleobas): This can trigger a side effect
        return key in self.value.__dict__

    def get_source_by_walking_mro(
        self, tx: "InstructionTranslator", name: str
    ) -> DictGetItemSource:
        assert self.cls_source is not None

        for idx, klass in enumerate(type(self.value).__mro__):
            if name in klass.__dict__:
                descriptor = klass.__dict__[name]

                # Guard that intermediate MRO classes don't shadow this
                # attribute, deduplicating by (id(klass), name) across
                # subclasses that share the same intermediate MRO class.
                # Safe because TYPE_MATCH guards fix the MRO, so the same
                # id(klass) always refers to the same class object.
                for absent_idx in range(1, idx):
                    absent_klass = type(self.value).__mro__[absent_idx]
                    cache_key = (id(absent_klass), name)
                    if cache_key in tx.output.guarded_mro_absent_keys:
                        continue
                    tx.output.guarded_mro_absent_keys.add(cache_key)
                    mro_source = TypeMROSource(self.cls_source)
                    klass_source: Source = GetItemSource(mro_source, absent_idx)
                    dict_source = TypeDictSource(klass_source)
                    install_guard(
                        dict_source.make_guard(
                            functools.partial(GuardBuilder.DICT_NOT_CONTAINS, key=name)
                        )
                    )

                # Guard that the instance __dict__ does not shadow the
                # class attribute.  Skipped for data descriptors (those
                # with __set__, e.g. property) because Python gives data
                # descriptors priority over instance __dict__ in attribute
                # lookup — the instance dict can only be populated by
                # directly writing to obj.__dict__, not via setattr.
                if (
                    self.source
                    and hasattr(self.value, "__dict__")
                    and name not in self.value.__dict__
                    and not hasattr(descriptor, "__set__")
                ):
                    install_guard(
                        self.source.make_guard(
                            functools.partial(
                                GuardBuilder.NOT_PRESENT_IN_GENERIC_DICT, attr=name
                            )
                        )
                    )

                # Reuse the source if we've already resolved the same
                # descriptor object for the same attribute name (e.g. same
                # property reached via different subclasses) to avoid
                # redundant ID_MATCH guards.  We include name in the key
                # because distinct attributes can point to the same object
                # (e.g. a = b = some_obj, or interned small integers).
                cache_key = (id(descriptor), name)
                cache = tx.output.mro_source_cache
                if cache_key in cache:
                    return cache[cache_key]

                if idx != 0:
                    mro_source = TypeMROSource(self.cls_source)
                    klass_source = GetItemSource(mro_source, idx)
                else:
                    klass_source = self.cls_source
                dict_source = TypeDictSource(klass_source)
                out_source = DictGetItemSource(dict_source, name)
                cache[cache_key] = out_source
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

    def generic_getattr(
        self, tx: "InstructionTranslator", name: str
    ) -> VariableTracker:
        """Dynamo implementation of CPython's PyObject_GenericGetAttr.

        This mirrors object.__getattribute__ and is called from:
        - var_getattr (for objects without a custom __getattribute__)
        - SuperVariable.call_method (when super().__getattribute__() resolves
          to object.__getattribute__)

        The algorithm: MRO walk → data descriptor → instance __dict__ →
        non-data descriptor / plain class attr → dynamic fallback →
        __getattr__ → AttributeError.
        """
        source: Source | None = AttrSource(self.source, name) if self.source else None

        if tx.output.side_effects.has_pending_mutation_of_attr(self, name):
            result = tx.output.side_effects.load_attr(self, name, deleted_ok=True)
            if isinstance(result, variables.DeletedVariable):
                raise_observed_exception(
                    AttributeError,
                    tx,
                    args=[
                        f"'{type(self.value).__name__}' object has no attribute '{name}'",
                    ],
                )
            return result

        if name == "__dict__":
            if not hasattr(self.value, "__dict__"):
                raise_observed_exception(AttributeError, tx)
            return self.get_dict_vt(tx)

        # TODO(anijain2305) - Investigate if we need specialization for more
        # dunder attrs. inspect.getattr_static does not return correct value for
        # them.
        if name == "__class__":
            cls_source: Source | None = source
            if source is None:
                cls_source = self.cls_source
            else:
                cls_source = source
            return VariableTracker.build(tx, type(self.value), cls_source)

        from ..mutation_guard import unpatched_nn_module_init

        # ---- CPython attribute lookup algorithm ----
        # Mirror object.__getattribute__ (PyObject_GenericGetAttr):
        #   1. type_attr = lookup name in type(obj).__mro__
        #   2. if type_attr is a DATA descriptor → invoke it
        #   3. if name in obj.__dict__ → return as-is (no descriptor invocation)
        #   4. if type_attr is a non-data descriptor → invoke it
        #   5. if type_attr is a plain class variable → return it
        #   6. __getattr__ fallback
        #   7. raise AttributeError
        #
        # Between steps 5 and 6, we also handle objects with custom storage
        # that aren't visible via the MRO walk or instance __dict__ (step 5b).
        #
        # Step 1: Single MRO walk on the type (cached).
        type_attr = self.lookup_class_mro_attr(name)

        # Dynamo patches nn.Module.__init__ at import time to inject tracing
        # hooks.  Undo that here so the unpatched original is traced instead.
        if type_attr is torch.nn.Module.__init__:
            type_attr = unpatched_nn_module_init

        # Step 2: Data descriptors on the type take priority over instance dict.
        if type_attr is not NO_SUCH_SUBOBJ and is_data_descriptor(type_attr):
            return self.resolve_data_descriptor(tx, name, type_attr, source)

        # Step 3: Instance __dict__ — return as-is, no descriptor invocation.
        # TODO(guilhermeleobas): step 3 should look into dict_vt and not self.value.__dict__
        # as the object could have mutated an attribute via setattr
        if hasattr(self.value, "__dict__") and name in self.value.__dict__:
            subobj = self.value.__dict__[name]
            source = self.maybe_wrap_nn_module_source_for_instance(tx, name, source)
            return VariableTracker.build(tx, subobj, source)

        # Step 4-5: Non-data descriptor or plain class attribute.
        if type_attr is not NO_SUCH_SUBOBJ:
            return self.resolve_type_attr(tx, name, type_attr, source)

        # Step 5b: Dynamic fallback for attributes that exist on the live
        # object but aren't visible to the static MRO walk or instance
        # __dict__ check above.  This covers objects with custom storage
        # backends (e.g. threading.local uses a per-thread dict not
        # accessible via obj.__dict__) and C extensions that store data
        # outside the normal Python object layout.
        #
        # This is NOT the same as the C-level data descriptor fallback in
        # resolve_data_descriptor (step 2): that handles descriptors found
        # on the type MRO (like member_descriptor for __slots__), while this
        # handles attributes that aren't on the type MRO at all.
        #
        # Only safe when the class doesn't override __getattribute__,
        # otherwise we'd run arbitrary user code.
        if not self._object_has_getattribute:
            try:
                resolved = type(self.value).__getattribute__(self.value, name)
                source = self.maybe_wrap_nn_module_source_for_instance(tx, name, source)
                return VariableTracker.build(tx, resolved, source)
            except AttributeError:
                pass

        # Step 6: __getattr__ fallback.
        getattr_fn = self._check_for_getattr()
        if isinstance(getattr_fn, types.FunctionType):
            if (
                getattr_fn is unpatched_nn_module_getattr
                and isinstance(self, variables.UnspecializedNNModuleVariable)
                and istype(self.value._parameters, dict)  # type: ignore[attr-defined]
                and istype(self.value._buffers, dict)  # type: ignore[attr-defined]
                and istype(self.value._modules, dict)  # type: ignore[attr-defined]
            ):
                out = self.manually_trace_nn_module_getattr(tx, name)
            else:
                new_source = None
                if self.source:
                    new_source = AttrSource(self.source, "__getattr__")
                out = variables.UserMethodVariable(
                    getattr_fn, self, source=new_source
                ).call_function(tx, [variables.ConstantVariable.create(name)], {})

            if self.source and getattr_fn is torch.nn.Module.__getattr__:
                if isinstance(
                    out,
                    (
                        variables.UnspecializedNNModuleVariable,
                        variables.NNModuleVariable,
                    ),
                ):
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

        # Step 7: AttributeError.
        raise_observed_exception(
            AttributeError,
            tx,
            args=[f"'{type(self.value).__name__}' object has no attribute '{name}'"],
        )

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
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
                    source=new_source,
                ).call_function(tx, [VariableTracker.build(tx, name)], {})
            except ObservedAttributeError:
                # Pass through to __getattr__ if __getattribute__ fails
                handle_observed_exception(tx)

        return self.generic_getattr(tx, name)

    def resolve_data_descriptor(
        self,
        tx: "InstructionTranslator",
        name: str,
        type_attr: object,
        source: Source | None,
    ) -> VariableTracker:
        """Handle data descriptors found on the type MRO (property, _tuplegetter, etc.)."""
        if isinstance(type_attr, property) and not self._is_c_defined_property(
            type_attr
        ):
            # Python property — trace fget directly.
            if self.source:
                source = AttrSource(self.get_source_by_walking_mro(tx, name), "fget")
            fget_vt = VariableTracker.build(
                tx, type_attr.fget, source=source, realize=True
            )
            return fget_vt.call_function(tx, [self], {})

        get_fn = inspect.getattr_static(type(type_attr), "__get__", None)
        if isinstance(get_fn, types.FunctionType):
            # User-defined data descriptor with a Python __get__.
            return self.invoke_descriptor_get(tx, name, type_attr, source)

        # C-level data descriptor (property with C fget, member/getset
        # descriptors, Cython attrs, etc.) — resolve via
        # object.__getattribute__ which is side-effect free.
        # Uninitialized slots raise AttributeError which must be surfaced
        # as ObservedAttributeError so dynamo's try/except tracing works.
        try:
            resolved = type(self.value).__getattribute__(self.value, name)
        except AttributeError:
            raise_observed_exception(
                AttributeError,
                tx,
                args=[
                    f"'{type(self.value).__name__}' object has no attribute '{name}'"
                ],
            )
        return VariableTracker.build(tx, resolved, source)

    def resolve_type_attr(
        self,
        tx: "InstructionTranslator",
        name: str,
        type_attr: object,
        source: Source | None,
    ) -> VariableTracker:
        """Handle non-data descriptors and plain class attributes from the type MRO."""
        from ..mutation_guard import unpatched_nn_module_init

        if (
            type_attr is unpatched_nn_module_init
            or type_attr is torch.nn.Module.__init__
        ):
            type_attr = unpatched_nn_module_init

        can_use_mro_source = self.cls_source is not None and self.source is not None

        if isinstance(type_attr, staticmethod):
            # type_attr is the raw staticmethod wrapper from cls.__dict__
            # (not the unwrapped function).  We call __get__ to unwrap it,
            # but the *source* must go through __func__ on the descriptor
            # (not the resolved function) because the guard needs to watch
            # the descriptor object in the class dict, not the result.
            if can_use_mro_source:
                source = AttrSource(
                    self.get_source_by_walking_mro(tx, name), "__func__"
                )
            func = type_attr.__get__(self.value)
            return VariableTracker.build(tx, func, source)
        elif isinstance(type_attr, classmethod):
            source_fn = None
            if can_use_mro_source:
                source_fn = AttrSource(
                    self.get_source_by_walking_mro(tx, name), "__func__"
                )  # type: ignore[assignment]
            return variables.UserMethodVariable(
                type_attr.__func__,
                self.var_getattr(tx, "__class__"),
                source_fn=source_fn,
                source=source,
            )
        elif isinstance(type_attr, types.ClassMethodDescriptorType):
            func = type_attr.__get__(self.value, None)
            return VariableTracker.build(tx, func, source)
        elif is_lru_cache_wrapped_function(type_attr):
            return variables.WrapperUserMethodVariable(
                type_attr, "__wrapped__", self, source=source
            )
        elif isinstance(type_attr, types.FunctionType):
            while hasattr(type_attr, "_torchdynamo_inline"):
                type_attr = type_attr._torchdynamo_inline  # type: ignore[union-attr]
                source = AttrSource(source, "_torchdynamo_inline") if source else None
            # Function on the type MRO + not in instance dict → bound method.
            var_source = None
            if can_use_mro_source:
                var_source = self.get_source_by_walking_mro(tx, name)
            return variables.UserMethodVariable(
                type_attr, self, source_fn=var_source, source=source
            )
        # Check for a Python-level __get__ (non-data descriptor with traceable __get__).
        get_fn = inspect.getattr_static(type(type_attr), "__get__", None)
        if isinstance(get_fn, types.FunctionType):
            return self.invoke_descriptor_get(tx, name, type_attr, source)

        # C-level non-data descriptors / opaque callables — defer to runtime.
        # MethodDescriptorType: e.g. list.append (PyMethodDef)
        # WrapperDescriptorType: e.g. list.__add__ (slot wrappers)
        # MethodWrapperType: e.g. [].__add__ (bound slot wrappers)
        #
        # Exception: if the descriptor has a registered polyfill, return the
        # polyfill as a bound method so Dynamo can trace through it.
        if (
            isinstance(
                type_attr,
                (
                    types.MethodDescriptorType,
                    types.WrapperDescriptorType,
                    types.MethodWrapperType,
                ),
            )
            or torch._C._dynamo.utils.is_instancemethod(type_attr)  # type: ignore[attr-defined]
            or is_cython_function(type_attr)
        ):
            from .. import trace_rules

            if trace_rules.is_polyfilled_callable(type_attr):  # type: ignore[arg-type]
                from .functions import PolyfilledFunctionVariable

                polyfill_handlers = PolyfilledFunctionVariable._get_polyfill_handlers()
                wrapped: Any = polyfill_handlers.get(type_attr)  # type: ignore[arg-type]
                if wrapped is not None:
                    traceable_fn = wrapped.__torch_dynamo_polyfill__
                    return variables.UserMethodVariable(traceable_fn, self)
            return variables.GetAttrVariable(self, name, type(type_attr), source=source)

        # Plain class variable (or MethodType, C-level non-data descriptor
        # without __get__, etc.).
        if can_use_mro_source:
            source = self.get_source_by_walking_mro(tx, name)
        elif not source and self.cls_source is not None:
            source = AttrSource(self.cls_source, name)
        return VariableTracker.build(tx, type_attr, source)

    def invoke_descriptor_get(
        self,
        tx: "InstructionTranslator",
        name: str,
        descriptor: object,
        source: Source | None,
    ) -> VariableTracker:
        """Trace a descriptor's __get__(instance, owner) call."""
        descriptor_source = None
        descriptor_get_source = None
        if self.cls_source:
            descriptor_source = self.get_source_by_walking_mro(tx, name)
            descriptor_get_source = AttrSource(TypeSource(descriptor_source), "__get__")
            descriptor_var = VariableTracker.build(tx, descriptor, descriptor_source)
        else:
            descriptor_var = UserDefinedObjectVariable(descriptor)

        owner_var = UserDefinedClassVariable(type(self.value))
        return variables.UserMethodVariable(
            descriptor.__get__.__func__,  # type: ignore[union-attr]
            descriptor_var,
            source=descriptor_get_source,
        ).call_function(tx, [self, owner_var], {})

    def maybe_wrap_nn_module_source_for_instance(
        self,
        tx: "InstructionTranslator",
        name: str,
        source: Source | None,
    ) -> Source | None:
        """Wrap source for nn.Module instance dict attribute access if needed."""
        if (
            source
            and isinstance(self, variables.UnspecializedNNModuleVariable)
            and (not tx.output.export or torch._dynamo.config.install_free_tensors)
        ):
            if name in ("_buffers", "_parameters"):
                assert self.source is not None
                source = UnspecializedParamBufferSource(self.source, name)
            source = self._wrap_source(source)
        return source

    def call_obj_hasattr(
        self, tx: "InstructionTranslator", name: str
    ) -> "ConstantVariable":
        if self.source:
            install_guard(
                self.source.make_guard(
                    functools.partial(GuardBuilder.HASATTR, attr=name)
                )
            )

        try:
            var_vt = self.var_getattr(tx, name)
            return VariableTracker.build(
                tx, not isinstance(var_vt, variables.DeletedVariable)
            )
        except ObservedAttributeError:
            handle_observed_exception(tx)
            return variables.ConstantVariable.create(False)

    def is_python_hashable(self) -> bool:
        raise_on_overridden_hash(self.value, self)
        if self._base_vt is not None:
            return self._base_vt.is_python_hashable()
        return True

    def get_python_hash(self) -> int:
        if self._base_vt is not None:
            return self._base_vt.get_python_hash()
        return hash(self.value)

    def is_python_equal(self, other: object) -> bool:
        if (
            isinstance(other, VariableTracker)
            and self.is_python_constant()
            and other.is_python_constant()
        ):
            return self.as_python_constant() == other.as_python_constant()
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

    def call_tree_map_with_path_branch(
        self,
        tx: "InstructionTranslator",
        tree_map_fn: "variables.functions.UserFunctionVariable",
        map_fn: "VariableTracker",
        rest: "collections.abc.Sequence[VariableTracker]",
        tree_map_kwargs: "dict[str, VariableTracker]",
        keypath: "tuple[Any, ...]",
    ) -> "VariableTracker":
        """Emulate tree_map_with_path behavior for user-defined objects.

        Same logic as call_tree_map_branch but passes keypath to the map function.
        """
        tree_map_module = tree_map_fn.get_module()
        is_optree = tree_map_module.startswith("optree")

        if is_optree:
            try:
                import optree
                from optree.registry import _NODETYPE_REGISTRY

                is_registered = (
                    self.value_type in _NODETYPE_REGISTRY
                    or optree.is_namedtuple_class(self.value_type)
                    or optree.is_structseq_class(self.value_type)
                )

                if not is_registered:
                    namespace_var = tree_map_kwargs.get("namespace")
                    if namespace_var is not None:
                        try:
                            namespace = namespace_var.as_python_constant()
                            is_registered = (
                                namespace,
                                self.value_type,
                            ) in _NODETYPE_REGISTRY
                        except NotImplementedError:
                            return self._tree_map_with_path_fallback(
                                tx,
                                tree_map_fn,
                                map_fn,
                                rest,
                                tree_map_kwargs,
                                keypath,
                            )
            except ImportError:
                return self._tree_map_with_path_fallback(
                    tx,
                    tree_map_fn,
                    map_fn,
                    rest,
                    tree_map_kwargs,
                    keypath,
                )
        else:
            import torch.utils._pytree as pytree

            is_registered = (
                self.value_type in pytree.SUPPORTED_NODES
                or pytree.is_namedtuple_class(self.value_type)
                or pytree.is_structseq_class(self.value_type)
            )

        if not is_registered:
            keypath_var = variables.TupleVariable(
                [VariableTracker.build(tx, k) for k in keypath]
            )
            return map_fn.call_function(tx, [keypath_var, self, *rest], {})

        return self._tree_map_with_path_fallback(
            tx,
            tree_map_fn,
            map_fn,
            rest,
            tree_map_kwargs,
            keypath,
        )


class FrozenDataClassVariable(UserDefinedObjectVariable):
    """Frozen dataclass variable for as_proxy/as_python_constant/hashability.

    Construction is handled by the generic polyfill path (tracing through
    the auto-generated __init__). Field values are retrieved dynamically
    via var_getattr using InstructionTranslator.current_tx().
    """

    def _get_field_vt(self, field_name: str) -> VariableTracker:
        from torch._dynamo.symbolic_convert import InstructionTranslator

        tx = InstructionTranslator.current_tx()
        return self.var_getattr(tx, field_name)

    def as_python_constant(self) -> object:
        from dataclasses import fields

        import torch.utils._pytree as pytree

        if not istype(
            self.value, (pytree.TreeSpec, pytree.LeafSpec, pytree.ConstantNode)
        ):
            raise NotImplementedError(
                "currently can't reconstruct arbitrary frozen dataclass instances"
            )

        if istype(self.value, pytree.LeafSpec):
            return pytree.treespec_leaf()

        args: list[object] = []
        kwargs: dict[str, object] = {}
        for field in fields(self.value):  # type: ignore[arg-type]
            if field.init:
                data = self._get_field_vt(field.name).as_python_constant()
                if getattr(field, "kw_only", False):
                    kwargs[field.name] = data
                else:
                    args.append(data)

        return self.python_type()(*args, **kwargs)

    def as_proxy(self) -> object:
        from dataclasses import fields

        args: list[object] = []
        kwargs: dict[str, object] = {}
        for field in fields(self.value):  # type: ignore[arg-type]
            proxy = self._get_field_vt(field.name).as_proxy()
            if hasattr(field, "kw_only") and field.kw_only:
                kwargs[field.name] = proxy
            else:
                args.append(proxy)

        return self.python_type()(*args, **kwargs)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        if self.source is not None:
            codegen(self.source)
            return
        codegen.append_output(
            codegen.create_load_const_unchecked(self.as_python_constant())
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value_type.__name__})"

    def is_python_hashable(self) -> Literal[True]:
        return True

    def get_python_hash(self) -> int:
        from dataclasses import fields as dc_fields

        return hash(
            tuple(
                self._get_field_vt(f.name).get_python_hash()
                for f in dc_fields(self.value)  # type: ignore[arg-type]
            )
        )

    def is_python_equal(self, other: object) -> bool:
        if not isinstance(other, FrozenDataClassVariable):
            return False
        if self.python_type() is not other.python_type():
            return False
        from dataclasses import fields as dc_fields

        return all(
            self._get_field_vt(f.name).is_python_equal(other._get_field_vt(f.name))
            for f in dc_fields(self.value)  # type: ignore[arg-type]
        )


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
        init_args = kwargs.get("init_args", [])
        self.exc_vt = variables.ExceptionVariable(self.value_type, init_args)

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
            return variables.ConstantVariable.create(None)
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

    def var_getattr(self, tx: "InstructionTranslator", name: str):
        if name in (
            "args",
            "__cause__",
            "__context__",
            "__suppress_context__",
            "__traceback__",
        ):
            return self.exc_vt.var_getattr(tx, name)
        return super().var_getattr(tx, name)

    @property
    def __context__(self) -> "ConstantVariable":
        # type: ignore[return-value]
        return self.exc_vt.__context__

    @property
    def args(self) -> list[VariableTracker]:
        return self.exc_vt.args

    def set_context(self, context: "variables.ExceptionVariable") -> None:
        return self.exc_vt.set_context(context)

    @property
    def exc_type(self) -> type[BaseException]:
        return self.exc_vt.exc_type

    @property
    def python_stack(self) -> traceback.StackSummary | None:
        return self.exc_vt.python_stack

    def debug_repr(self) -> str:
        return self.exc_vt.debug_repr()

    @python_stack.setter
    def python_stack(self, value: traceback.StackSummary) -> None:
        self.exc_vt.python_stack = value


class InspectVariable(UserDefinedObjectVariable):
    """Handles inspect.Signature and inspect.Parameter objects.

    Short-circuits property accesses to avoid tracing property getters,
    redirecting them to the underlying private attributes directly.
    """

    _PROPERTY_REDIRECTS: dict[type, dict[str, str]] = {
        inspect.Signature: {"parameters": "_parameters"},
        inspect.Parameter: {"kind": "_kind", "name": "_name"},
    }

    @staticmethod
    def is_matching_object(obj: object) -> bool:
        return type(obj) in InspectVariable._PROPERTY_REDIRECTS

    @staticmethod
    def is_matching_class(obj: object) -> bool:
        return obj in InspectVariable._PROPERTY_REDIRECTS

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        redirects = self._PROPERTY_REDIRECTS.get(type(self.value), {})
        if name in redirects:
            return super().var_getattr(tx, redirects[name])
        return super().var_getattr(tx, name)


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


_CONSTANT_BASE_TYPES = (int, float, str)

_constant_base_methods: dict[type, set[Any]] = {
    t: {m for m in t.__dict__.values() if callable(m)} for t in _CONSTANT_BASE_TYPES
}


class UserDefinedConstantVariable(UserDefinedObjectVariable):
    """
    Represents user-defined objects that subclass immutable constant types
    (int, float, str).

    Uses a ConstantVariable as _base_vt for the underlying constant value.
    """

    def __init__(self, value: Any, **kwargs: Any) -> None:
        from .constant import ConstantVariable

        super().__init__(value, **kwargs)
        for base in type(value).__mro__:
            if base in _CONSTANT_BASE_TYPES:
                self._base_vt = ConstantVariable.create(base(value))
                self._base_methods = _constant_base_methods[base]
                break
        assert self._base_vt is not None

    def as_python_constant(self) -> Any:
        return self.value

    def as_proxy(self) -> object:
        assert self._base_vt is not None
        return self._base_vt.as_proxy()


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
            self._base_vt = ConstDictVariable(
                {},
                mutation_type=ValueMutationNew(),
            )
        else:
            self._base_vt = dict_vt
        self._base_methods = dict_methods
        assert self._base_vt is not None

    def len(self) -> int:
        # Used by nn_module.py to short-circuit the nn.Module forward method
        # when no hooks are registered.  Calling .len() directly avoids the
        # overhead of full call_method("__len__") dispatch during tracing.
        assert self._base_vt is not None
        return self._base_vt.len()  # type: ignore[union-attr]

    def sq_length(self, tx: "InstructionTranslator") -> VariableTracker:
        # Dict implements __len__ via mp_length (mapping protocol), not
        # sq_length (sequence protocol). Redirect so generic_len works.
        return self.mp_length(tx)

    def mp_subscript_impl(
        self,
        tx: "InstructionTranslator",
        key: VariableTracker,
    ) -> VariableTracker:
        # dict_subscript: https://github.com/python/cpython/blob/62a6e898e01/Objects/dictobject.c#L3673-L3706
        # TODO(follow-up): add test for unhashable/invalid key type, Counter missing key
        method = self._maybe_get_baseclass_method("__getitem__")
        if method in self._base_methods:
            assert self._base_vt is not None
            try:
                return self._base_vt.mp_subscript_impl(tx, key)
            except ObservedKeyError:
                if issubclass(
                    self.python_type(), dict
                ) and self._maybe_get_baseclass_method("__missing__"):
                    return self.call_method(tx, "__missing__", [key], {})
                else:
                    raise
        return super().mp_subscript_impl(tx, key)

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # Dict subclasses can override __missing__ to provide fallback
        # behavior instead of raising a KeyError. This is used, for example,
        # by collections.Counter.
        if (
            name == "__getitem__"
            and self._maybe_get_baseclass_method("__getitem__") in self._base_methods
            and self._maybe_get_baseclass_method("__missing__")
        ):
            assert self._base_vt is not None
            try:
                return self._base_vt.call_method(tx, name, args, kwargs)
            except ObservedKeyError:
                handle_observed_exception(tx)
                return self.call_method(tx, "__missing__", args, kwargs)
        return super().call_method(tx, name, args, kwargs)


# TODO: move to dicts.py alongside ConstDictVariable and DefaultDictVariable.
# Currently blocked by circular imports (dicts.py ↔ user_defined.py).
class OrderedDictVariable(UserDefinedDictVariable):
    """
    Represents collections.OrderedDict instances.

    CPython has both a pure-Python implementation:
    https://github.com/python/cpython/blob/v3.13.0/Lib/collections/__init__.py#L86-L339
    and a C accelerator that replaces it at runtime:
    https://github.com/python/cpython/blob/v3.13.0/Objects/odictobject.c

    The C accelerator is always active, so methods like move_to_end and
    popitem are C-level method_descriptors, not Python functions.

    Dict storage is delegated to _base_vt (a ConstDictVariable) via
    UserDefinedDictVariable.
    """

    def __init__(
        self,
        value: object,
        dict_vt: "ConstDictVariable | None" = None,
        **kwargs: Any,
    ) -> None:
        if dict_vt is None:
            from .dicts import ConstDictVariable

            dict_vt = ConstDictVariable(
                {},
                user_cls=collections.OrderedDict,
                mutation_type=ValueMutationNew(),
            )
        super().__init__(value, dict_vt=dict_vt, **kwargs)

    def is_python_constant(self) -> bool:
        assert self._base_vt is not None
        return self._base_vt.is_python_constant()

    def as_python_constant(self) -> Any:
        assert self._base_vt is not None
        return collections.OrderedDict(self._base_vt.as_python_constant())

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from .constant import ConstantVariable
        from .dicts import HashableTracker

        # OrderedDict-exclusive C methods that ConstDictVariable doesn't handle.
        # https://github.com/python/cpython/blob/v3.13.0/Objects/odictobject.c
        if name == "move_to_end":
            assert self._base_vt is not None
            self._base_vt.install_dict_keys_match_guard()  # type: ignore[union-attr]
            tx.output.side_effects.mutation(self._base_vt)
            if args[0] not in self._base_vt:  # type: ignore[operator]
                raise_observed_exception(KeyError, tx)

            last = True
            if len(args) == 2 and args[0].is_python_constant():
                last = args[1].as_python_constant()
            if kwargs and "last" in kwargs and kwargs["last"].is_python_constant():
                last = kwargs["last"].as_python_constant()

            key = HashableTracker(args[0])
            self._base_vt.items.move_to_end(key, last=last)  # type: ignore[union-attr]
            return ConstantVariable.create(None)
        elif name == "popitem":
            assert self._base_vt is not None
            if not self._base_vt.items:  # type: ignore[union-attr]
                raise_observed_exception(
                    KeyError, tx, args=["popitem(): dictionary is empty"]
                )

            last = True
            if len(args) == 1 and args[0].is_python_constant():
                last = args[0].as_python_constant()
            if kwargs and "last" in kwargs and kwargs["last"].is_python_constant():
                last = kwargs["last"].as_python_constant()

            k, v = self._base_vt.items.popitem(last=last)  # type: ignore[union-attr]
            self._base_vt.should_reconstruct_all = True  # type: ignore[union-attr]
            tx.output.side_effects.mutation(self._base_vt)
            return variables.TupleVariable([k.vt, v])
        return super().call_method(tx, name, args, kwargs)


# TODO: move to dicts.py alongside ConstDictVariable.
# Currently blocked by circular imports (dicts.py ↔ user_defined.py).
class DefaultDictVariable(UserDefinedDictVariable):
    """
    Represents collections.defaultdict instances.

    CPython's defaultdict is implemented in C:
    https://github.com/python/cpython/blob/v3.13.3/Modules/_collectionsmodule.c#L2177-L2180

    default_factory is a field on the C struct (defdictobject.default_factory),
    not a Python instance attribute, so we model it as a field on the VT.

    Dict storage is delegated to _base_vt (a ConstDictVariable) via
    UserDefinedDictVariable.
    """

    _cpython_type = collections.defaultdict

    def __init__(
        self,
        value: object,
        default_factory: VariableTracker | None = None,
        dict_vt: ConstDictVariable | None = None,
        **kwargs: Any,
    ) -> None:
        if dict_vt is None:
            from .dicts import ConstDictVariable

            dict_vt = ConstDictVariable(
                {},
                mutation_type=ValueMutationNew(),
            )
        super().__init__(value, dict_vt=dict_vt, **kwargs)
        if default_factory is None:
            from .constant import ConstantVariable

            default_factory = ConstantVariable.create(None)
        self.default_factory = default_factory

    @staticmethod
    def is_supported_factory(arg: VariableTracker) -> bool:
        """Check if arg is a valid default_factory (callable or None).

        CPython's defaultdict.__init__ checks ``callable(factory)`` and
        raises TypeError if not.  We mirror this by checking the
        underlying Python value when possible.
        """
        if isinstance(arg, variables.ConstantVariable):
            return arg.value is None
        # Check the real Python value for callable()
        try:
            val = arg.as_python_constant()
            return val is None or callable(val)
        except Exception:
            pass
        # Callables (functions, builtins, classes) are supported
        return isinstance(
            arg,
            (
                variables.BaseBuiltinVariable,
                variables.functions.BaseUserFunctionVariable,
                variables.functions.PolyfilledFunctionVariable,
                variables.UserDefinedClassVariable,
            ),
        )

    def is_python_constant(self) -> bool:
        assert self._base_vt is not None
        # An empty defaultdict with a non-constant factory can't be
        # constant-folded (we can't serialize the factory).
        if not self.default_factory.is_python_constant() and not self._base_vt.items:  # type: ignore[union-attr]
            return False
        return self._base_vt.is_python_constant()

    def as_python_constant(self) -> Any:
        assert self._base_vt is not None
        factory = None
        if self.default_factory.is_python_constant():
            factory = self.default_factory.as_python_constant()
        return collections.defaultdict(factory, self._base_vt.as_python_constant())

    def debug_repr(self) -> str:
        assert self.default_factory is not None
        assert self._base_vt is not None
        return (
            f"defaultdict({self.default_factory.debug_repr()}, "
            f"{self._base_vt.debug_repr()})"
        )

    def var_getattr(
        self,
        tx: "InstructionTranslator",
        name: str,
    ) -> VariableTracker:
        if name == "default_factory":
            return self.default_factory
        return super().var_getattr(tx, name)

    def _missing_impl(
        self,
        tx: "InstructionTranslator",
        key: "VariableTracker",
    ) -> "VariableTracker":
        """defaultdict.__missing__: auto-vivification via default_factory.

        https://github.com/python/cpython/blob/v3.13.3/Modules/_collectionsmodule.c#L2233-L2254
        """
        from .constant import ConstantVariable

        if (
            istype(self.default_factory, ConstantVariable)
            and self.default_factory.value is None
        ):
            raise_observed_exception(KeyError, tx, args=[key])
        default_var = self.default_factory.call_function(tx, [], {})
        assert self._base_vt is not None
        self._base_vt.call_method(tx, "__setitem__", [key, default_var], {})
        return default_var

    def mp_subscript_impl(
        self,
        tx: "InstructionTranslator",
        key: "VariableTracker",
    ) -> "VariableTracker":
        """defaultdict.__getitem__: dict lookup with __missing__ fallback."""
        assert self._base_vt is not None
        if key in self._base_vt:  # type: ignore[operator]
            return self._base_vt.getitem_const(tx, key)  # type: ignore[union-attr]
        return self._missing_impl(tx, key)

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from .constant import ConstantVariable

        if name == "__init__":
            # defaultdict.__init__(self, default_factory=None, *args, **kwargs)
            # https://github.com/python/cpython/blob/v3.13.3/Modules/_collectionsmodule.c#L2072
            # Extract default_factory, delegate rest to dict.__init__
            if len(args) >= 1:
                if self.is_supported_factory(args[0]):
                    self.default_factory = args[0]
                    tx.output.side_effects.store_attr(
                        self,
                        "default_factory",
                        self.default_factory,
                    )
                    args = list(args[1:])
                else:
                    # CPython raises TypeError for non-callable first arg
                    raise_observed_exception(
                        TypeError,
                        tx,
                        args=["first argument must be callable or None"],
                    )
            assert self._base_vt is not None
            return self._base_vt.call_method(tx, "__init__", args, kwargs)
        elif name == "__getitem__":
            if len(args) != 1:
                raise_args_mismatch(tx, name, "1 args", f"{len(args)} args")
            return self.mp_subscript_impl(tx, args[0])
        elif name == "__missing__":
            if len(args) != 1:
                raise_args_mismatch(tx, name, "1 args", f"{len(args)} args")
            return self._missing_impl(tx, args[0])
        elif name == "copy":
            # defaultdict.copy() creates a new defaultdict with same factory
            # https://github.com/python/cpython/blob/v3.13.3/Modules/_collectionsmodule.c#L2282
            from .builder import SourcelessBuilder

            assert self._base_vt is not None
            new_dd = tx.output.side_effects.track_new_user_defined_object(
                SourcelessBuilder.create(tx, dict),
                SourcelessBuilder.create(tx, collections.defaultdict),
                [],
            )
            assert isinstance(new_dd, DefaultDictVariable)
            new_dd.default_factory = self.default_factory
            new_dd._base_vt = self._base_vt.clone(
                mutation_type=ValueMutationNew(),
                source=None,
            )
            tx.output.side_effects.store_attr(
                new_dd, "default_factory", new_dd.default_factory
            )
            return new_dd
        elif name == "__setattr__":
            if len(args) != 2:
                raise_args_mismatch(tx, name, "2 args", f"{len(args)} args")
            if (
                istype(args[0], ConstantVariable) and args[0].value == "default_factory"
            ) and self.is_supported_factory(args[1]):
                self.default_factory = args[1]
                tx.output.side_effects.store_attr(
                    self, "default_factory", self.default_factory
                )
                return ConstantVariable.create(None)
            return super().call_method(tx, name, args, kwargs)
        elif name == "__eq__":
            if len(args) != 1:
                raise_args_mismatch(tx, name, "1 args", f"{len(args)} args")
            return VariableTracker.build(tx, polyfills.dict___eq__).call_function(
                tx, [self, args[0]], {}
            )
        return super().call_method(tx, name, args, kwargs)


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
        from .builder import SourcelessBuilder

        tx = kwargs.pop("tx", None)
        super().__init__(value, **kwargs)

        python_type = set if isinstance(value, set) else frozenset
        self._base_methods = set_methods if python_type is set else frozenset_methods

        if set_vt is None:
            assert self.source is None, (
                "set_vt must be constructed by builder.py when source is present"
            )
            if python_type is set:
                # set is initialized later
                self._base_vt = variables.SetVariable(
                    set(),
                    mutation_type=ValueMutationNew(),
                )
            else:
                init_args = kwargs.get("init_args", {})
                if tx is None:
                    tx = torch._dynamo.symbolic_convert.InstructionTranslator.current_tx()
                self._base_vt = SourcelessBuilder.create(tx, python_type).call_function(  # type: ignore[assignment]
                    tx, init_args, {}
                )
        else:
            self._base_vt = set_vt
        assert self._base_vt is not None

    def as_python_constant(self) -> object:
        assert self._base_vt is not None
        return self._base_vt.as_python_constant()

    @property
    def set_items(self) -> set[Any]:
        assert self._base_vt is not None
        return self._base_vt.set_items  # pyrefly: ignore[missing-attribute]

    @property
    def items(self) -> dict[HashableTracker, VariableTracker]:
        assert self._base_vt is not None
        return self._base_vt.items  # pyrefly: ignore[missing-attribute]

    def is_python_equal(self, other: object) -> bool:
        assert self._base_vt is not None
        return isinstance(
            other, UserDefinedSetVariable
        ) and self._base_vt.is_python_equal(other._base_vt)


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
            self._base_vt = ListVariable([], mutation_type=ValueMutationNew())
        else:
            self._base_vt = list_vt
        self._base_methods = list_methods
        assert self._base_vt is not None


class UserDefinedTupleVariable(UserDefinedObjectVariable):
    """
    Represents user defined objects that are subclasses of tuple.

    Internally, it uses a TupleVariable to represent the tuple part of the
    variable tracker. For everything else, it falls back to
    UserDefinedObjectVariable.

    NamedTupleVariable and StructSequenceVariable are subclasses that handle
    namedtuples and structseqs (torch.return_types.*) respectively.
    """

    _nonvar_fields = {
        "tuple_cls",
        *UserDefinedObjectVariable._nonvar_fields,
    }

    @staticmethod
    def get_vt_cls(cls: type) -> type["UserDefinedTupleVariable"]:
        if is_structseq_class(cls):
            return StructSequenceVariable
        return NamedTupleVariable

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
            self._base_vt = TupleVariable(elems, mutation_type=ValueMutationNew())
        else:
            self._base_vt = tuple_vt
        self.tuple_cls = type(value)
        self._base_methods = tuple_methods
        assert self._base_vt is not None

    @property
    def items(self) -> list[VariableTracker]:
        assert self._base_vt is not None
        return self._base_vt.items  # type: ignore[return-value]

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "__eq__":
            if len(args) != 1 or kwargs:
                raise ValueError("Improper arguments for method.")
            return VariableTracker.build(tx, self.is_python_equal(args[0]))
        elif name == "__ne__":
            if len(args) != 1 or kwargs:
                raise ValueError("Improper arguments for method.")
            return VariableTracker.build(tx, not self.is_python_equal(args[0]))
        return super().call_method(tx, name, args, kwargs)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # Sourceless namedtuples/structseqs (e.g. tensor subclass metadata from
        # SourcelessBuilder) aren't in id_to_variable so codegen_save_tempvars
        # never processes them. When they appear in return values, codegen falls
        # through to call_reconstruct. This is the same pattern as other
        # sourceless containers (ConstDictVariable, TupleVariable, etc.).
        # UserDefinedDictVariable doesn't need this because it's never created
        # sourceless — it only comes from VariableBuilder which always has a
        # source.
        assert self.source is None
        create_fn = self.get_construct_fn()
        codegen.add_push_null(
            lambda: codegen.append_output(
                codegen.create_load_const_unchecked(create_fn)
            )
        )
        codegen(self._base_vt)
        codegen.extend_output(create_call_function(1, False))

    def get_construct_fn(self) -> Callable[..., Any]:
        raise NotImplementedError

    def _validate_rest_for_tree_map(
        self, rest: "collections.abc.Sequence[VariableTracker]"
    ) -> list["UserDefinedTupleVariable"] | None:
        """Validate that rest args are compatible for tree_map fast-path."""
        others: list[UserDefinedTupleVariable] = []
        n = len(self.items)
        for candidate in rest:
            if (
                not isinstance(candidate, UserDefinedTupleVariable)
                or len(candidate.items) != n
                or candidate.tuple_cls is not self.tuple_cls
            ):
                return None
            others.append(candidate)
        return others

    def _make_tree_map_result(
        self, new_items: list[VariableTracker]
    ) -> "UserDefinedTupleVariable":
        from .lists import TupleVariable

        tuple_vt = TupleVariable(new_items, mutation_type=ValueMutationNew())
        return type(self)(
            self.value,
            tuple_vt=tuple_vt,
            mutation_type=ValueMutationNew(),
        )

    def _is_pytree_node(self) -> bool:
        from torch.utils._pytree import is_namedtuple_class

        return is_namedtuple_class(self.tuple_cls) or is_structseq_class(self.tuple_cls)

    def call_tree_map_branch(
        self,
        tx: "InstructionTranslator",
        tree_map_fn: "variables.functions.UserFunctionVariable",
        map_fn: "VariableTracker",
        rest: "collections.abc.Sequence[VariableTracker]",
        tree_map_kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if not self._is_pytree_node():
            return super().call_tree_map_branch(
                tx, tree_map_fn, map_fn, rest, tree_map_kwargs
            )
        others = self._validate_rest_for_tree_map(rest)
        if others is None:
            return self._tree_map_fallback(
                tx, tree_map_fn, map_fn, rest, tree_map_kwargs
            )

        new_items: list[VariableTracker] = []
        for idx, item in enumerate(self.items):
            sibling_leaves = [o.items[idx] for o in others]
            new_items.append(
                item.call_tree_map(
                    tx, tree_map_fn, map_fn, sibling_leaves, tree_map_kwargs
                )
            )

        return self._make_tree_map_result(new_items)

    def call_tree_map_with_path_branch(
        self,
        tx: "InstructionTranslator",
        tree_map_fn: "variables.functions.UserFunctionVariable",
        map_fn: "VariableTracker",
        rest: "collections.abc.Sequence[VariableTracker]",
        tree_map_kwargs: "dict[str, VariableTracker]",
        keypath: "tuple[Any, ...]",
    ) -> "VariableTracker":
        if not self._is_pytree_node():
            return super().call_tree_map_with_path_branch(
                tx, tree_map_fn, map_fn, rest, tree_map_kwargs, keypath
            )
        others = self._validate_rest_for_tree_map(rest)
        if others is None:
            return self._tree_map_with_path_fallback(
                tx, tree_map_fn, map_fn, rest, tree_map_kwargs, keypath
            )

        fields = namedtuple_fields(self.tuple_cls)
        new_items: list[VariableTracker] = []
        for idx, item in enumerate(self.items):
            sibling_leaves = [o.items[idx] for o in others]
            child_keypath = keypath + (GetAttrKey(fields[idx]),)
            new_items.append(
                item.call_tree_map_with_path(
                    tx,
                    tree_map_fn,
                    map_fn,
                    sibling_leaves,
                    tree_map_kwargs,
                    child_keypath,
                )
            )

        return self._make_tree_map_result(new_items)

    def is_python_equal(self, other: object) -> bool:
        assert self._base_vt is not None
        other = other._base_vt if isinstance(other, UserDefinedTupleVariable) else other
        return self._base_vt.is_python_equal(other)


class NamedTupleVariable(UserDefinedTupleVariable):
    """Represents Python namedtuples (created via collections.namedtuple).

    Namedtuples use _tuplegetter descriptors for field access and
    Type(*args) / Type._make(iterable) for construction.
    """

    def resolve_data_descriptor(
        self,
        tx: "InstructionTranslator",
        name: str,
        type_attr: object,
        source: Source | None,
    ) -> VariableTracker:
        if isinstance(type_attr, _collections._tuplegetter):
            # namedtuple fields are _tuplegetter descriptors implemented in C.
            # We emulate _tuplegetter.__get__ by indexing into the tracked
            # tuple items, because self.value may not hold actual runtime values.
            _, (idx, _) = type_attr.__reduce__()
            return self.items[idx]  # type: ignore[union-attr]
        return super().resolve_data_descriptor(tx, name, type_attr, source)

    def get_construct_fn(self) -> Callable[..., Any]:
        return self.tuple_cls._make  # type: ignore[attr-defined]

    def as_python_constant(self) -> Any:
        items = [x.as_python_constant() for x in self.items]
        return self.tuple_cls(*items)  # type: ignore[arg-type]

    def as_proxy(self) -> Any:
        items = [x.as_proxy() for x in self.items]
        return self.tuple_cls(*items)  # type: ignore[arg-type]


class StructSequenceVariable(UserDefinedTupleVariable):
    """Represents C-implemented PyStructSequence types (torch.return_types.*).

    Structseqs use Type(iterable) calling convention and reject tuple.__new__.
    """

    def resolve_data_descriptor(
        self,
        tx: "InstructionTranslator",
        name: str,
        type_attr: object,
        source: Source | None,
    ) -> VariableTracker:
        if isinstance(type_attr, types.MemberDescriptorType):
            # Structseq fields are member_descriptor objects. We emulate
            # field access by looking up the field name in _fields and
            # indexing into the tracked tuple items.
            fields = namedtuple_fields(self.tuple_cls)
            if name in fields:
                return self.items[fields.index(name)]
        return super().resolve_data_descriptor(tx, name, type_attr, source)

    def get_construct_fn(self) -> Callable[..., Any]:
        return self.tuple_cls

    def as_python_constant(self) -> Any:
        items = [x.as_python_constant() for x in self.items]
        return self.tuple_cls(items)

    def as_proxy(self) -> Any:
        items = [x.as_proxy() for x in self.items]
        return self.tuple_cls(items)


class MutableMappingVariable(UserDefinedObjectVariable):
    def __init__(self, value: object, **kwargs: Any) -> None:
        super().__init__(value, **kwargs)

    def method_setattr_standard(
        self,
        tx: "InstructionTranslator",
        name: VariableTracker,
        value: VariableTracker,
        directly_update_dict: bool = False,
    ) -> VariableTracker:
        """Override to handle property setters on MutableMapping subclasses.

        This is needed because property.__set__ is a slot wrapper (C function),
        not a Python function, so the base class's try_get_descritor_and_setter_py_func
        returns None for properties. But property.fset IS a Python function we can trace.

        Without this, property setters on newly created MutableMapping objects fail
        when accessing nested objects (which haven't been initialized yet on the
        example value). By tracing the fset, we capture the setter logic in the graph
        instead of running it on uninitialized example objects.

        TODO(compiler): This fix is scoped to MutableMapping only because tracing
        property setters on ALL UserDefinedObjectVariable can cause failures when
        the fset calls untraceable C++ functions (e.g., pybind functions). Ideally,
        this should be extended to all user-defined classes with a graceful fallback
        when tracing the fset hits an untraceable function.
        See: https://github.com/pytorch/pytorch/issues/172000
        """
        if isinstance(name, variables.ConstantVariable) and isinstance(name.value, str):
            name_str = name.value
            descriptor = inspect.getattr_static(type(self.value), name_str, None)
            if isinstance(descriptor, property) and descriptor.fset is not None:
                fset_source = None
                if self.cls_source:
                    desc_source = self.get_source_by_walking_mro(tx, name_str)
                    fset_source = AttrSource(desc_source, "fset")
                fset_vt = VariableTracker.build(tx, descriptor.fset, fset_source)
                return fset_vt.call_function(tx, [self, value], {})

        return super().method_setattr_standard(tx, name, value, directly_update_dict)

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        # A common pattern in the init code of MutableMapping objects is to
        # update the __dict__ attribute. The parent class
        # (UserDefinedObjectVariable) implements __dict__ lookups using a VT
        # (self.dict_vt) that uses the side effects table as source of truth.
        if name == "get" and type(self.value).get in (  # type: ignore[attr-defined]
            collections.abc.Mapping.get,
            dict.get,
        ):
            return variables.UserMethodVariable(polyfills.mapping_get, self)
        else:
            return super().var_getattr(tx, name)

    def mp_length(self, tx: "InstructionTranslator") -> VariableTracker:
        if self._maybe_get_baseclass_method("__len__") in dict_methods:
            return VariableTracker.build(tx, len(self.value))  # type: ignore[bad-argument-type]
        return super().mp_length(tx)


class RandomVariable(UserDefinedObjectVariable):
    pass
