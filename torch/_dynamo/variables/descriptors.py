"""
Variable trackers for CPython descriptor objects.

This module contains classes that model descriptor types during graph
compilation, including:
- C-level descriptors (wrapper, method, classmethod, member and getset descriptors)
- Bound builtin methods and method-wrappers
- staticmethod, classmethod and property
- collections._tuplegetter
"""

import _collections  # type: ignore[import-not-found]
import collections
import inspect
import sys
import types
from typing import Any, cast, TYPE_CHECKING, TypeAlias

from .. import graph_break_hints, variables
from ..bytecode_transformation import create_call_function
from ..exc import (
    raise_attribute_error,
    raise_observed_exception,
    raise_type_error,
    unimplemented,
)
from ..source import AttrSource, TypeMROSource, TypeSource
from ..utils import is_tensor_base_attr_getter
from .base import (
    AsPythonConstantNotImplementedError,
    GetSet,
    getset_build,
    getset_load_or_build,
    getset_set,
    Member,
    NO_SUCH_SUBOBJ,
    readonly_setter,
    store_attr_mutation,
    type_qualified_name,
    unmodeled_setter,
    VariableTracker,
)
from .constant import ConstantVariable
from .functions import (
    BaseUserFunctionVariable,
    UserFunctionVariable,
    UserMethodVariable,
)


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase
    from torch._guards import Source


# ---------------------------------------------------------------------------
# CPython descriptor VTs
#
# Each class mirrors a CPython descriptor type (PyWrapperDescr_Type,
# PyMethodDescr_Type, etc.) and implements tp_descr_get_impl to model
# the descriptor binding step faithfully.
# ---------------------------------------------------------------------------

DescriptorTypes: TypeAlias = (
    types.MethodDescriptorType
    | types.WrapperDescriptorType
    | types.MemberDescriptorType
    | types.GetSetDescriptorType
)


def _check_descriptor_obj_type(
    tx: "InstructionTranslatorBase",
    descriptor: DescriptorTypes,
    obj: "VariableTracker",
) -> None:
    """Check that obj's type is compatible with descriptor.__objclass__.

    Mirrors CPython's descr_check which raises TypeError when a C descriptor
    is bound to an object whose type is not a subtype of the descriptor's
    __objclass__.

    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L79-L96
    """
    obj_type = obj.python_type()
    if not issubclass(obj_type, descriptor.__objclass__):
        raise_type_error(
            tx,
            f"descriptor '{descriptor.__name__}' for "
            f"'{descriptor.__objclass__.__name__}' objects "
            f"doesn't apply to a '{obj_type.__name__}' object",
        )


class DescriptorVariable(VariableTracker):
    """Base class for CPython descriptor VTs.

    Mirrors the behavior of PyDescr_Type, which is the base type for all
    CPython descriptors.  The tp_descr_get slot on this type implements
    the descriptor binding step, which is mirrored by tp_descr_get_impl
    on this class.
    """

    tp_members = {
        "__objclass__": Member(
            getset_build(lambda s: s.descriptor.__objclass__), readonly_setter
        ),
        "__name__": Member(
            getset_build(lambda s: s.descriptor.__name__), readonly_setter
        ),
    }


# descr_members: __objclass__ and __name__ are PyMemberDef on all descriptor
# types. https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L641-L645
class WrapperDescriptorVariable(DescriptorVariable):
    """Unbound C slot wrapper (wrapper_descriptor on a type).

    CPython types define behavior through C-level slots on PyTypeObject
    (tp_richcompare, sq_length, nb_add, etc.).  When these slots are
    accessed from Python (e.g. list.__add__), CPython exposes them as
    wrapper_descriptor objects (PyWrapperDescr_Type).  A wrapper_descriptor
    is an unbound descriptor living on the type -- it is not tied to any
    instance.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L867

    When a wrapper_descriptor is accessed on an instance (e.g. [1,2].__add__),
    its tp_descr_get slot (wrapperdescr_get) is invoked, which calls
    PyWrapper_New to produce a bound method-wrapper (_PyMethodWrapper_Type).
    The tp_descr_get_impl method on this class mirrors that binding step.
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: types.WrapperDescriptorType,
        owner: VariableTracker,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.descriptor = descriptor
        self.owner = owner

    def __repr__(self) -> str:
        cls_name = self.descriptor.__objclass__.__name__
        return f"WrapperDescriptorVariable({cls_name}.{self.descriptor.__name__})"

    def python_type(self) -> type:
        return types.WrapperDescriptorType

    def as_python_constant(self) -> types.WrapperDescriptorType:
        return self.descriptor

    def get_real_python_backed_value(self) -> types.WrapperDescriptorType:
        return self.descriptor

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # Unbound call: list.__add__([1,2], [3,4]) -- first arg is self.
        # Mirrors wrapperdescr_call which invokes the C slot directly.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L535
        if not args:
            raise_type_error(
                tx,
                f"descriptor '{self.descriptor.__name__}' of "
                f"'{self.descriptor.__objclass__.__name__}' object needs an argument",
            )
        obj, *rest = args
        _check_descriptor_obj_type(tx, self.descriptor, obj)
        # Dispatch through the owner (UDCV for the defining class) rather
        # than obj.call_method, which would do MRO resolution from type(obj)
        # and find Python overrides on subclasses. Routing through the class
        # mirrors CPython's wrapperdescr_call which invokes the C slot directly.
        return self.owner.call_method(
            tx, self.descriptor.__name__, [obj, *rest], kwargs
        )

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        owner: VariableTracker,
    ) -> "MethodWrapperVariable":
        # Mirrors wrapperdescr_get which calls PyWrapper_New to produce
        # a bound method-wrapper.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L203-L213
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L1489-L1505
        _check_descriptor_obj_type(tx, self.descriptor, obj)
        return MethodWrapperVariable(self.descriptor, obj, source=self.source)


class MethodWrapperVariable(VariableTracker):
    """Bound method-wrapper (wrapper_descriptor bound to an instance).

    Produced by WrapperDescriptorVariable.tp_descr_get_impl, mirroring
    PyWrapper_New which stores a reference to the descriptor and the instance.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L1450
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: types.WrapperDescriptorType,
        obj: VariableTracker,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.descriptor = descriptor
        self.obj = obj

    def __repr__(self) -> str:
        cls_name = self.descriptor.__objclass__.__name__
        return (
            f"MethodWrapperVariable({cls_name}.{self.descriptor.__name__}, {self.obj})"
        )

    def python_type(self) -> type:
        return types.MethodWrapperType

    # A method-wrapper's own dunders come from the descriptor / bound obj and do
    # not depend on obj's contents. Resolving them via these tables (consulted
    # before the generic getset-descriptor path) avoids forcing self.obj to a
    # python constant, which would break e.g. a list holding non-constant items.
    # Every entry in CPython's wrapper_getsets has a NULL setter.
    tp_getset = {
        "__name__": GetSet(
            lambda s, tx: ConstantVariable.create(s.descriptor.__name__),
            readonly_setter,
        ),
        "__qualname__": GetSet(
            lambda s, tx: ConstantVariable.create(s.descriptor.__qualname__),
            readonly_setter,
        ),
    }
    tp_members = {
        "__self__": Member(lambda s, tx: s.obj, readonly_setter),
    }

    def get_real_python_backed_value(self) -> types.MethodWrapperType:
        return self.as_python_constant()

    def is_python_constant(self) -> bool:
        return self.obj.is_python_constant()

    def as_python_constant(self) -> types.MethodWrapperType:
        obj_value = self.obj.as_python_constant()
        try:
            return self.descriptor.__get__(obj_value, type(obj_value))
        except TypeError:
            raise AsPythonConstantNotImplementedError(self) from None

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # Only materialize the wrapper (which forces self.obj's python constant)
        # for the tensor-getter special case; the generic path below dispatches
        # via call_method and must not require a constant self.obj (e.g. a set /
        # frozenset whose as_python_constant() re-hashes its keys).
        if args and isinstance(args[0], variables.TensorVariable):
            try:
                method_wrapper = self.as_python_constant()
            except NotImplementedError:
                method_wrapper = None
            if method_wrapper is not None and is_tensor_base_attr_getter(
                method_wrapper
            ):
                if not (len(args) == 1 and len(kwargs) == 0):
                    raise_type_error(
                        tx, "tensor attribute getter takes exactly one argument"
                    )
                # Avoid the generic descriptor path's implicit owner lookup, which
                # would read __class__ on tensor subclasses during __torch_function__.
                descriptor = cast(Any, method_wrapper.__self__)
                return args[0].tp_getattro_impl(tx, descriptor.__name__)

        sd = self.obj.lookup_slotdefs(self.descriptor.__name__)
        if sd is None:
            return self.obj.call_method(tx, self.descriptor.__name__, args, kwargs)
        return sd(self.obj, tx, args, kwargs)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen(self.obj)
        codegen.extend_output(codegen.create_load_attrs(self.descriptor.__name__))

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        try:
            # CPython wrapper_hash:
            # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L1347
            return hash(self.as_python_constant()), False
        except NotImplementedError:
            return super().hash_impl(tx)

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: "VariableTracker", op: str
    ) -> "VariableTracker":
        from .object_protocol import python_constant_richcompare_impl

        return python_constant_richcompare_impl(self, tx, other, op)


class MethodDescriptorVariable(DescriptorVariable):
    """Unbound C method descriptor (method_descriptor on a type).

    CPython types expose their PyMethodDef-based C methods as
    method_descriptor objects (PyMethodDescr_Type) in the type's tp_dict.
    For example, list.append and dict.get are method_descriptors.  Like
    wrapper_descriptors, these are unbound descriptors living on the type.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L716

    When a method_descriptor is accessed on an instance (e.g. [].append),
    its tp_descr_get slot (method_get) is invoked, which calls
    PyCFunction_NewEx to produce a bound builtin_function_or_method
    (PyCFunction_Type).  The tp_descr_get_impl method mirrors that step.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L137-L159
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: types.MethodDescriptorType,
        owner: VariableTracker,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.descriptor = descriptor
        self.owner = owner

    def __repr__(self) -> str:
        cls_name = self.descriptor.__objclass__.__name__
        return f"MethodDescriptorVariable({cls_name}.{self.descriptor.__name__})"

    def python_type(self) -> type:
        return types.MethodDescriptorType

    def as_python_constant(self) -> types.MethodDescriptorType:
        return self.descriptor

    def get_real_python_backed_value(self) -> types.MethodDescriptorType:
        return self.descriptor

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # Mirrors methoddescr_call which invokes the C method directly.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L427
        if not args:
            raise_type_error(
                tx,
                f"descriptor '{self.descriptor.__name__}' of "
                f"'{self.descriptor.__objclass__.__name__}' object needs an argument",
            )
        obj, *rest = args
        name = self.descriptor.__name__
        _check_descriptor_obj_type(tx, self.descriptor, obj)
        # Dispatch through the owner (UDCV for the defining class) rather
        # than obj.call_method, which would do MRO resolution from type(obj)
        # and find Python overrides on subclasses.
        return self.owner.call_method(tx, name, [obj, *rest], kwargs)

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        owner: VariableTracker,
    ) -> "BoundBuiltinMethodVariable":
        # Mirrors method_get which calls PyCFunction_NewEx to produce a
        # bound builtin_function_or_method.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L137-L159
        # https://github.com/python/cpython/blob/3.13/Objects/methodobject.c#L40
        _check_descriptor_obj_type(tx, self.descriptor, obj)
        return BoundBuiltinMethodVariable(self.descriptor, obj, source=self.source)


class BoundBuiltinMethodVariable(VariableTracker):
    """Bound builtin_function_or_method (PyCFunction_Type).

    Produced by MethodDescriptorVariable.tp_descr_get_impl (binding a
    method_descriptor to an instance, e.g. [].append) or created by the
    builder for bound C methods (e.g. frozenset().__contains__,
    tuple.__new__).  The backing descriptor can be a MethodDescriptorType
    or a BuiltinFunctionType (for methods stored directly in type dicts).
    https://github.com/python/cpython/blob/3.13/Objects/methodobject.c#L331
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: types.MethodDescriptorType
        | types.BuiltinFunctionType
        | types.ClassMethodDescriptorType,
        obj: VariableTracker,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.descriptor = descriptor
        self.obj = obj

    def __repr__(self) -> str:
        cls_name = getattr(
            getattr(self.descriptor, "__objclass__", None), "__name__", "?"
        )
        return f"BoundBuiltinMethodVariable({cls_name}.{self.descriptor.__name__}, {self.obj})"

    def python_type(self) -> type:
        return types.BuiltinMethodType

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        # meth_hash: https://github.com/python/cpython/blob/e76aa128fe/Objects/methodobject.c#L319
        try:
            return hash(self.as_python_constant()), False
        except AsPythonConstantNotImplementedError:
            return id(self), True

    def tp_richcompare_impl(self, tx, other, op):
        from .object_protocol import object_richcompare

        return object_richcompare(self, tx, other, op)

    def as_python_constant(self) -> Any:
        obj = self.obj.as_python_constant()
        if isinstance(self.descriptor, types.ClassMethodDescriptorType):
            return self.descriptor.__get__(None, obj)
        if hasattr(self.descriptor, "__get__"):
            return self.descriptor.__get__(obj)  # type: ignore[union-attr]
        return getattr(obj, self.descriptor.__name__)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        return self.obj.call_method(tx, self.descriptor.__name__, list(args), kwargs)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen(self.obj)
        codegen.extend_output(codegen.create_load_attrs(self.descriptor.__name__))

    def _get_qualname(self, tx: "InstructionTranslatorBase") -> VariableTracker | None:
        # meth_get__qualname__: the receiver if it is a type, else its type.
        try:
            owner = self.obj.python_type()
            if issubclass(owner, type):
                owner = self.obj.as_python_constant()
        except NotImplementedError:
            return None
        name = self.descriptor.__name__
        return ConstantVariable.create(f"{owner.__qualname__}.{name}")

    # meth_getsets: https://github.com/python/cpython/blob/v3.13.0/Objects/methodobject.c#L249-L256
    tp_getset = {
        "__name__": GetSet(
            lambda s, tx: ConstantVariable.create(s.descriptor.__name__),
            readonly_setter,
        ),
        "__qualname__": GetSet(_get_qualname, readonly_setter),
        "__doc__": GetSet(
            lambda s, tx: ConstantVariable.create(s.descriptor.__doc__),
            readonly_setter,
        ),
    }


class ClassMethodDescriptorVariable(DescriptorVariable):
    """C-level classmethod descriptor (classmethod_descriptor on a type).

    CPython exposes C classmethods defined via PyMethodDef with METH_CLASS
    as classmethod_descriptor objects (PyClassMethodDescr_Type).  For
    example, dict.fromkeys is a classmethod_descriptor.  Like
    method_descriptor, these live on the type and are unbound.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L756

    classmethod_get binds the C method to the class (ignoring obj) via
    PyCMethod_New, producing a bound builtin_function_or_method.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L94-L134
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: types.ClassMethodDescriptorType,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.descriptor = descriptor

    def __repr__(self) -> str:
        cls_name = self.descriptor.__objclass__.__name__
        return f"ClassMethodDescriptorVariable({cls_name}.{self.descriptor.__name__})"

    def python_type(self) -> type:
        return types.ClassMethodDescriptorType

    def as_python_constant(self) -> types.ClassMethodDescriptorType:
        return self.descriptor

    def get_real_python_backed_value(self) -> types.ClassMethodDescriptorType:
        return self.descriptor

    tp_members = {
        "__objclass__": Member(
            getset_build(lambda s: s.descriptor.__objclass__), readonly_setter
        ),
        "__name__": Member(
            getset_build(lambda s: s.descriptor.__name__), readonly_setter
        ),
    }

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        owner: VariableTracker,
    ) -> BoundBuiltinMethodVariable:
        # classmethod_get binds the C method to the class (ignoring obj),
        # producing a builtin_function_or_method via PyCMethod_New.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L94-L134
        return BoundBuiltinMethodVariable(self.descriptor, owner, source=self.source)


# sm_init/cm_init run functools_wraps(), which copies these off the wrapped
# callable into the descriptor's instance dict, skipping the ones it does not
# have. The instance dict is read before the type's own attribute of the same
# name, so e.g. staticmethod(f).__doc__ is f.__doc__, not staticmethod.__doc__.
# Dynamo resolves them through the traced callable: materializing the descriptor
# to read that dict would rebuild the callable, which is inexact for synthesized
# functions (NestedUserFunctionVariable) and unavailable for wrapper VTs.
# https://github.com/python/cpython/blob/3.13/Objects/funcobject.c#L1090-L1117
_WRAPS_COPIED_ATTRS = frozenset(
    ("__module__", "__name__", "__qualname__", "__doc__", "__annotations__")
)


def _lookup_wraps_copied_attr(
    tx: "InstructionTranslatorBase",
    descriptor: VariableTracker,
    name: str,
) -> VariableTracker | None:
    if name not in _WRAPS_COPIED_ATTRS:
        return None
    from .object_protocol import generic_getattr

    return generic_getattr(tx, descriptor, name)


class StaticMethodVariable(VariableTracker):
    """staticmethod descriptor wrapping a callable.

    CPython's staticmethod (PyStaticMethod_Type) is a non-data descriptor
    whose tp_descr_get (sm_descr_get) simply returns the wrapped callable,
    ignoring both obj and type.
    https://github.com/python/cpython/blob/3.13/Objects/funcobject.c#L1520
    https://github.com/python/cpython/blob/3.13/Objects/funcobject.c#L1418-L1428
    """

    # sm_memberlist: both members are readonly aliases of the wrapped callable.
    # https://github.com/python/cpython/blob/3.13/Objects/funcobject.c#L1469-L1473
    tp_members = {
        "__func__": Member(lambda s, _: s.descriptor, readonly_setter),
        "__wrapped__": Member(lambda s, _: s.descriptor, readonly_setter),
    }

    def __init__(self, descriptor: VariableTracker, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.descriptor = descriptor

    @classmethod
    def from_descriptor(
        cls,
        tx: "InstructionTranslatorBase",
        descriptor: staticmethod,  # type: ignore[type-arg]
        source: "Source | None" = None,
    ) -> "StaticMethodVariable":
        func_source = AttrSource(source, "__func__") if source else None
        return cls(
            VariableTracker.build(tx, descriptor.__func__, func_source),
            source=source,
        )

    def __repr__(self) -> str:
        # A staticmethod built by user code can wrap any callable, so only a
        # traced function has a name to report.
        if isinstance(self.descriptor, BaseUserFunctionVariable):
            return f"StaticMethodVariable({self.descriptor.get_name()})"
        return f"StaticMethodVariable({self.descriptor})"

    def python_type(self) -> type:
        return staticmethod

    def as_python_constant(self) -> staticmethod:  # type: ignore[type-arg]
        return staticmethod(self.descriptor.as_python_constant())

    def lookup_instance_dict(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker | None:
        return _lookup_wraps_copied_attr(tx, self.descriptor, name)

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker | None,
        owner: VariableTracker,
    ) -> VariableTracker:
        # sm_descr_get returns sm->sm_callable unconditionally.
        # https://github.com/python/cpython/blob/3.13/Objects/funcobject.c#L1418-L1428
        return self.descriptor

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # Rebuild the descriptor around the traced callable rather than
        # restoring an original object, which may wrap an unrealized closure.
        # Descriptor-path instances are consumed by tp_descr_get_impl and do not
        # escape, so this effectively only serves values built by user code.
        codegen.add_push_null(
            lambda: codegen.load_import_from("builtins", "staticmethod")
        )
        codegen(self.descriptor)
        codegen.extend_output(create_call_function(1, False))


class ClassMethodVariable(VariableTracker):
    """classmethod descriptor wrapping a callable.

    CPython's classmethod (PyClassMethod_Type) is a non-data descriptor
    whose tp_descr_get (cm_descr_get) creates a bound method of the
    wrapped callable bound to the class (via PyMethod_New).
    https://github.com/python/cpython/blob/3.13/Objects/funcobject.c#L1314
    https://github.com/python/cpython/blob/3.13/Objects/funcobject.c#L1215-L1227
    """

    # cm_memberlist: both members are readonly aliases of the wrapped callable.
    # https://github.com/python/cpython/blob/3.13/Objects/funcobject.c#L1261-L1265
    tp_members = {
        "__func__": Member(lambda s, _: s.descriptor, readonly_setter),
        "__wrapped__": Member(lambda s, _: s.descriptor, readonly_setter),
    }

    def __init__(self, descriptor: VariableTracker, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.descriptor = descriptor

    @classmethod
    def from_descriptor(
        cls,
        tx: "InstructionTranslatorBase",
        descriptor: classmethod,  # type: ignore[type-arg]
        name: str,
        source: "Source | None" = None,
    ) -> "ClassMethodVariable":
        func_source = AttrSource(source, "__func__") if source else None
        # Being in a class __dict__ is what makes the bound method addressable
        # as `owner.<name>`, so build the VT that can source it.
        # A classmethod may wrap any callable, so let the builder pick the VT;
        # tp_descr_get_impl rejects the ones it cannot bind.
        return ClassAttrClassMethodVariable(
            name,
            VariableTracker.build(tx, descriptor.__func__, func_source, realize=True),
            source=source,
        )

    def __repr__(self) -> str:
        # A classmethod built by user code can wrap any callable, so only a
        # traced function has a name to report.
        if isinstance(self.descriptor, BaseUserFunctionVariable):
            return f"ClassMethodVariable({self.descriptor.get_name()})"
        return f"ClassMethodVariable({self.descriptor})"

    def python_type(self) -> type:
        return classmethod

    def as_python_constant(self) -> classmethod:  # type: ignore[type-arg]
        return classmethod(self.descriptor.as_python_constant())

    def lookup_instance_dict(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker | None:
        return _lookup_wraps_copied_attr(tx, self.descriptor, name)

    def _bound_source(
        self, owner: VariableTracker, func: BaseUserFunctionVariable
    ) -> "Source | None":
        """Source naming the bound method, or None when it has none.

        A `classmethod(...)` built by user code is not an entry in a class
        __dict__, so `owner.<func name>` does not name its bound method.
        """
        return None

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        owner: VariableTracker,
    ) -> VariableTracker:
        # cm_descr_get binds the wrapped function to the class.
        # https://github.com/python/cpython/blob/3.13/Objects/funcobject.c#L1215-L1227
        func = self.descriptor
        # NestedUserFunctionVariable is not a UserFunctionVariable, but it is a
        # Python function all the same and binds like one.
        if not isinstance(func, BaseUserFunctionVariable):
            unimplemented(
                gb_type="classmethod of non-Python function",
                context=f"tp_descr_get {self} on {owner}",
                explanation="Dynamo can only bind a classmethod wrapping a "
                "Python function to its class.",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        bound_source = self._bound_source(owner, func)
        if func.source is None and bound_source is not None:
            # The descriptor had no source of its own (C().f, where the class
            # came from obj.__class__), so take the function source from the
            # bound attribute. Without this the function would go unguarded.
            # Swap the source on the existing VT; get_function() would realize a
            # synthesized function and can graph break.
            # cast() is only for pyrefly.
            func = cast(
                BaseUserFunctionVariable,
                func.clone(source=AttrSource(bound_source, "__func__")),
            )
        return UserMethodVariable(func, owner, source=bound_source)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # See StaticMethodVariable.reconstruct.
        codegen.add_push_null(
            lambda: codegen.load_import_from("builtins", "classmethod")
        )
        codegen(self.descriptor)
        codegen.extend_output(create_call_function(1, False))


class ClassAttrClassMethodVariable(ClassMethodVariable):
    """classmethod found in a class __dict__, built by from_descriptor.

    Unlike `classmethod(f)` built by user code, the class attribute names its
    bound method, so `owner.<name>` is a valid source for guards and
    reconstruction. `name` is the attribute the descriptor is stored under,
    which is not the wrapped callable's name: a functools.wraps wrapper keeps
    its own code name, so `D.create` can wrap a function named `wrapper`.
    """

    def __init__(self, name: str, descriptor: VariableTracker, **kwargs: Any) -> None:
        super().__init__(descriptor, **kwargs)
        self.name = name

    def _bound_source(
        self, owner: VariableTracker, func: BaseUserFunctionVariable
    ) -> "Source | None":
        return AttrSource(owner.source, self.name) if owner.source else None


class MemberDescriptorVariable(DescriptorVariable):
    """C struct field descriptor (member_descriptor on a type).

    CPython exposes C struct fields defined via PyMemberDef as
    member_descriptor objects (PyMemberDescr_Type).  These are data
    descriptors used by __slots__ and C extension types to provide
    direct access to struct members.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L793

    member_get reads the field via PyMember_GetOne.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L162-L180
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: types.MemberDescriptorType,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.descriptor = descriptor

    def __repr__(self) -> str:
        cls_name = self.descriptor.__objclass__.__name__
        return f"MemberDescriptorVariable({cls_name}.{self.descriptor.__name__})"

    def python_type(self) -> type:
        return types.MemberDescriptorType

    def as_python_constant(self) -> types.MemberDescriptorType:
        return self.descriptor

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        owner: VariableTracker,
    ) -> VariableTracker:
        # Mirrors member_get which calls PyMember_GetOne to read the
        # C struct field.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L162-L180
        _check_descriptor_obj_type(tx, self.descriptor, obj)
        from .object_protocol import _UnhandledDescriptorError

        attr_name = self.descriptor.__name__
        # Prefer the VT's declarative table when it models this attribute, so a
        # VT with no concrete Python object behind it still resolves.
        entry = obj.lookup_tp_getset_member(attr_name)
        if entry is not None and entry.getter is not None:
            result = entry.getter(obj, tx)
            if result is not None:
                return result
        obj_value = obj.get_real_python_backed_value()
        if obj_value is NO_SUCH_SUBOBJ:
            raise _UnhandledDescriptorError(
                f"Cannot resolve member_descriptor '{attr_name}' "
                f"on {type(obj).__name__}"
            )
        try:
            resolved = self.descriptor.__get__(obj_value, type(obj_value))
        except (AttributeError, TypeError):
            raise_observed_exception(
                AttributeError,
                tx,
                args=[
                    f"'{type(obj_value).__name__}' object has no attribute '{attr_name}'"
                ],
            )
        result_source = obj.source and AttrSource(obj.source, attr_name)
        return VariableTracker.build(tx, resolved, result_source)

    def tp_descr_set_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        value: VariableTracker | None,
    ) -> VariableTracker:
        # Mirrors member_set: descr_setcheck, then PyMember_SetOne writes the
        # C struct field. value is None for __delete__.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L180-L196
        _check_descriptor_obj_type(tx, self.descriptor, obj)
        name = self.descriptor.__name__
        entry = obj.lookup_tp_getset_member(name)
        if entry is not None:
            # A READONLY PyMemberDef is modeled by readonly_setter, which raises
            # what PyMember_SetOne raises.
            entry.setter(obj, tx, value)
            return ConstantVariable.create(None)
        # No model for this member. STORE_ATTR itself applies the descriptor, so
        # replay via store_attr on the target (mirrors the __slots__ path in
        # UserDefinedObjectVariable).
        stored = variables.DeletedVariable() if value is None else value
        store_attr_mutation(tx, obj, name, stored)
        return ConstantVariable.create(None)


class GetSetDescriptorVariable(DescriptorVariable):
    """C getter/setter descriptor (getset_descriptor on a type).

    CPython exposes C getter/setter pairs defined via PyGetSetDef as
    getset_descriptor objects (PyGetSetDescr_Type).  These are data
    descriptors used for computed attributes backed by C functions
    (e.g. object.__class__, type.__dict__).
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L830

    getset_get calls the C getter function.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L183-L197
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(self, descriptor: types.GetSetDescriptorType, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.descriptor = descriptor

    def __repr__(self) -> str:
        cls_name = self.descriptor.__objclass__.__name__
        return f"GetSetDescriptorVariable({cls_name}.{self.descriptor.__name__})"

    def get_real_python_backed_value(self) -> types.GetSetDescriptorType:
        return self.descriptor

    def _getset_descriptor_get(
        self, tx: "InstructionTranslatorBase"
    ) -> "VariableTracker | None":
        if self.source is None:
            return None
        source = AttrSource(self.source, "__get__")
        return VariableTracker.build(tx, self.descriptor.__get__, source)

    tp_getset = {
        "__get__": GetSet(_getset_descriptor_get, readonly_setter),
    }

    def is_python_constant(self) -> bool:
        return True

    def as_python_constant(self) -> types.GetSetDescriptorType:
        return self.descriptor

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: "VariableTracker", op: str
    ) -> "VariableTracker":
        from .object_protocol import python_constant_richcompare_impl

        return python_constant_richcompare_impl(self, tx, other, op)

    def tp_descr_set_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        value: VariableTracker | None,
    ) -> VariableTracker:
        _check_descriptor_obj_type(tx, self.descriptor, obj)
        name = self.descriptor.__name__
        entry = obj.lookup_tp_getset_member(name)
        if entry is None:
            # No model for this getset. Dynamo cannot see whether the C setter
            # accepts the write, rejects it as read-only, or type-checks the
            # value, so it cannot decide between raising AttributeError and
            # applying an ordinary attribute write. Graph break rather than
            # guessing "writable" -- forwarding to the setattr/delattr
            # builtins here would run type(obj).__setattr__, which getset_set
            # bypasses, and would silently swallow a read-only rejection.
            unmodeled_setter(obj, tx, value)
        if entry.setter is readonly_setter:
            # getset_set's message is more specific than readonly_setter's, and
            # only this site knows the attribute name and its defining class.
            raise_attribute_error(
                tx,
                f"attribute '{name}' of "
                f"'{type_qualified_name(self.descriptor.__objclass__)}' objects "
                "is not writable",
            )
        entry.setter(obj, tx, value)
        return ConstantVariable.create(None)

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        owner: VariableTracker,
    ) -> VariableTracker:
        # Mirrors getset_get which calls the C getter function.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L183-L197
        attr_name = self.descriptor.__name__
        # Try to eagerly call the C getter when we can obtain the
        # concrete Python object (UDOV.value, or as_python_constant
        # for classes/constants). Fall back to tp_getattro_impl for
        # proxy-based VTs like TensorVariable.
        _check_descriptor_obj_type(tx, self.descriptor, obj)
        # Prefer the VT's declarative table when it models this attribute
        entry = obj.lookup_tp_getset_member(attr_name)
        if entry is not None and entry.getter is not None:
            result = entry.getter(obj, tx)
            if result is not None:
                return result
        # Otherwise call the C getter on the concrete Python object
        obj_value = obj.get_real_python_backed_value()
        if obj_value is NO_SUCH_SUBOBJ:
            from .object_protocol import _UnhandledDescriptorError

            raise _UnhandledDescriptorError(
                f"Cannot resolve getset_descriptor '{attr_name}' "
                f"on {type(obj).__name__}"
            )
        try:
            resolved = self.descriptor.__get__(obj_value, type(obj_value))
        except (AttributeError, TypeError):
            raise_observed_exception(
                AttributeError,
                tx,
                args=[
                    f"'{type(obj_value).__name__}' object has no attribute '{attr_name}'"
                ],
            )
        result_source = obj.source and AttrSource(obj.source, attr_name)
        if (
            obj.source
            and self.descriptor.__objclass__ is type
            and attr_name in ("__annotations__", "__dict__", "__mro__")
        ):
            # Direct descriptor calls still resolve the standard type slot even
            # when a metaclass shadows the same attribute. Only attach the
            # normal attribute source when runtime attribute lookup agrees.
            static_desc = inspect.getattr_static(type(obj_value), attr_name, None)
            if static_desc is not self.descriptor:
                result_source = None
            elif attr_name == "__mro__":
                result_source = TypeMROSource(obj.source)
        return VariableTracker.build(tx, resolved, result_source)


class PropertyVariable(VariableTracker):
    """Python property descriptor.

    The property type is a data descriptor with tp_descr_get =
    property_descr_get which calls fget(obj) to compute the value.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L2073
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L1660-L1693
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    tp_members = {
        "fget": Member(
            getset_load_or_build(lambda s: s.descriptor.fget, "fget"),
            readonly_setter,
        ),
        "fset": Member(
            getset_load_or_build(lambda s: s.descriptor.fset, "fset"),
            readonly_setter,
        ),
        "fdel": Member(
            getset_load_or_build(lambda s: s.descriptor.fdel, "fdel"),
            readonly_setter,
        ),
        "__doc__": Member(
            getset_load_or_build(lambda s: s.descriptor.__doc__, "__doc__"),
            getset_set("__doc__"),
        ),
    }

    def _name_getter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # property.__name__ only exists from 3.13
        if sys.version_info >= (3, 13):
            name = getattr(self.descriptor, "__name__", None)
            if name is not None:
                source = self.source and AttrSource(self.source, "__name__")
                return VariableTracker.build(tx, name, source)
        raise_attribute_error(
            tx, f"'{self.python_type_name()}' object has no attribute '__name__'"
        )

    def _isabstractmethod_getter(
        self, tx: "InstructionTranslatorBase"
    ) -> VariableTracker:
        from ..polyfills import property_isabstractmethod

        # Trace truth testing so user-defined __bool__ methods and their
        # exceptions are handled within the traced program.
        return UserFunctionVariable(property_isabstractmethod).call_function(
            tx, [self], {}
        )

    tp_getset = {
        "__name__": GetSet(_name_getter, getset_set("__name__")),
        "__isabstractmethod__": GetSet(
            _isabstractmethod_getter,
            readonly_setter,
        ),
    }

    def __init__(
        self,
        descriptor: property,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.descriptor = descriptor

    def __repr__(self) -> str:
        fget_name = getattr(self.descriptor.fget, "__name__", "?")
        return f"PropertyVariable({fget_name})"

    def python_type(self) -> type:
        return property

    def as_python_constant(self) -> property:
        return self.descriptor

    def tp_descr_set_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        value: VariableTracker | None,
    ) -> VariableTracker:
        attr = "fset" if value is not None else "fdel"
        fn = getattr(self.descriptor, attr)

        if fn is None:
            if sys.version_info >= (3, 13):
                display_name = getattr(self.descriptor, "__name__", "?")
            else:
                display_name = getattr(self.descriptor.fget, "__name__", "?")
            kind = "setter" if value is not None else "deleter"
            if sys.version_info >= (3, 11):
                # property_descr_set formats %R of the owner's *type*, whose
                # repr is its bare __qualname__ (no module prefix) -- unlike
                # python_qualified_name(), which mirrors the module-qualified
                # _PyType_GetFullyQualifiedName used elsewhere (e.g. __repr__).
                try:
                    cls_name = obj.python_type().__qualname__
                except NotImplementedError:
                    cls_name = obj.python_type_name()
                if display_name is not None:
                    msg = f"property '{display_name}' of '{cls_name}' object has no {kind}"
                else:
                    msg = f"property of '{cls_name}' object has no {kind}"
            else:
                # < 3.11: no owner/property-name in the message at all.
                verb = "set" if value is not None else "delete"
                if display_name is not None:
                    msg = f"can't {verb} attribute '{display_name}'"
                else:
                    msg = f"can't {verb} attribute"
            raise_attribute_error(tx, msg)

        args = [obj] if value is None else [obj, value]
        VariableTracker.build(
            tx, fn, source=self.source and AttrSource(self.source, attr)
        ).call_function(tx, args, {})
        return ConstantVariable.create(None)

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker | None,
        owner: VariableTracker,
    ) -> VariableTracker:
        # Mirrors property_descr_get: if obj is NULL or None, return self.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L1660-L1693
        if obj is None:
            return self
        fget_source = AttrSource(self.source, "fget") if self.source else None
        fget_vt = VariableTracker.build(
            tx, self.descriptor.fget, source=fget_source, realize=True
        )
        return fget_vt.call_function(tx, [obj], {})


class TupleGetterVariable(VariableTracker):
    """_tuplegetter descriptor used by namedtuple for field access.

    _tuplegetter is a C data descriptor that stores an index and returns
    self[index] on instance access. When accessed on the class (obj=None),
    it returns the descriptor itself.
    https://github.com/python/cpython/blob/3.13/Modules/_collectionsmodule.c#L2735
    https://github.com/python/cpython/blob/3.13/Modules/_collectionsmodule.c#L2636-L2663
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: "_collections._tuplegetter",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.descriptor = descriptor

    def __repr__(self) -> str:
        _, (idx, doc) = self.descriptor.__reduce__()
        return f"TupleGetterVariable(index={idx}, doc={doc!r})"

    def python_type(self) -> type:
        return _collections._tuplegetter

    def as_python_constant(self) -> "_collections._tuplegetter":
        return self.descriptor

    # _tuplegetter exposes __doc__ as a writable T_OBJECT member.
    # https://github.com/python/cpython/blob/v3.13.0/Modules/_collectionsmodule.c#L2717-L2721
    tp_members = {
        "__doc__": Member(
            getset_load_or_build(lambda s: s.descriptor.__doc__, "__doc__"),
            getset_set("__doc__"),
        )
    }

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker | None,
        owner: VariableTracker,
    ) -> VariableTracker:
        # https://github.com/python/cpython/blob/3.13/Modules/_collectionsmodule.c#L2636-L2663
        if obj is None:
            return self
        _, (idx, _) = self.descriptor.__reduce__()
        return obj.call_method(
            tx, "__getitem__", [variables.ConstantVariable.create(idx)], {}
        )

    def tp_descr_set_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        value: VariableTracker | None,
    ) -> VariableTracker:
        # _tuplegetter fields are read-only; tuplegetter_descr_set always
        # raises AttributeError for both set and delete.
        # https://github.com/python/cpython/blob/3.13/Modules/_collectionsmodule.c#L2665-L2673
        msg = "can't delete attribute" if value is None else "can't set attribute"
        raise_observed_exception(AttributeError, tx, args=[msg])


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


def _resolve_descriptor_get(
    tx: "InstructionTranslatorBase",
    type_attr: object,
    obj: VariableTracker,
    class_vt: VariableTracker,
    source: "Source | None",
    name: str,
) -> "VariableTracker | None":
    """Invoke tp_descr_get on a type attribute if it's a descriptor.

    Handles all descriptor types that have dedicated VTs with
    tp_descr_get_impl.  Returns None if type_attr is not a recognized
    descriptor type (plain class variable).
    """
    import types as _types

    from .object_protocol import mro_attr_source

    if isinstance(type_attr, property):
        # The property object lives on the type, not the instance: anchoring it at
        # obj.source would make PropertyVariable's fget source read the *result* of
        # the getter and then ask an int for .fget.
        prop_source = (
            mro_attr_source(tx, obj.python_type(), TypeSource(obj.source), name)
            if obj.source
            else None
        )
        prop_vt = variables.PropertyVariable(type_attr, source=prop_source)
        return prop_vt.tp_descr_get_impl(tx, obj, class_vt)
    if isinstance(type_attr, _types.MemberDescriptorType):
        md_vt = variables.MemberDescriptorVariable(type_attr, source=source)
        return md_vt.tp_descr_get_impl(tx, obj, class_vt)
    if isinstance(type_attr, _types.GetSetDescriptorType):
        gs_vt = variables.GetSetDescriptorVariable(type_attr, source=source)
        return gs_vt.tp_descr_get_impl(tx, obj, class_vt)
    _tuplegetter = collections._tuplegetter  # pyrefly: ignore[missing-attribute]
    if isinstance(type_attr, _tuplegetter):
        tg_vt = variables.TupleGetterVariable(type_attr, source=source)
        return tg_vt.tp_descr_get_impl(tx, obj, class_vt)
    if isinstance(type_attr, staticmethod):
        sm_vt = variables.StaticMethodVariable.from_descriptor(
            tx, type_attr, source=source
        )
        return sm_vt.tp_descr_get_impl(tx, obj, class_vt)
    if isinstance(type_attr, classmethod):
        cm_vt = variables.ClassMethodVariable.from_descriptor(
            tx, type_attr, name, source=source
        )
        return cm_vt.tp_descr_get_impl(tx, obj, class_vt)
    if isinstance(type_attr, _types.ClassMethodDescriptorType):
        cmd_vt = variables.ClassMethodDescriptorVariable(type_attr, source=source)
        return cmd_vt.tp_descr_get_impl(tx, obj, class_vt)
    if isinstance(type_attr, _types.WrapperDescriptorType):
        wd_vt = variables.WrapperDescriptorVariable(
            type_attr, owner=class_vt, source=source
        )
        return wd_vt.tp_descr_get_impl(tx, obj, class_vt)
    if isinstance(type_attr, _types.MethodDescriptorType):
        md_vt = variables.MethodDescriptorVariable(
            type_attr, owner=class_vt, source=source
        )
        return md_vt.tp_descr_get_impl(tx, obj, class_vt)
    if isinstance(type_attr, _types.FunctionType):
        return variables.UserMethodVariable(
            variables.UserFunctionVariable(
                type_attr, source=source and AttrSource(source, "__func__")
            ),
            obj,
            source=source,
        )

    return None
