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
import functools
import inspect
import sys
import types
from typing import Any, cast, NoReturn, TYPE_CHECKING, TypeAlias

from torch._C._dynamo import get_type_tp_name
from torch._guards import Source

from .. import graph_break_hints, variables
from ..bytecode_transformation import create_call_function
from ..exc import (
    handle_observed_exception,
    ObservedAttributeError,
    raise_attribute_error,
    raise_observed_exception,
    raise_type_error,
    unimplemented,
)
from ..guards import GuardBuilder, install_guard
from ..source import (
    AttrSource,
    DictGetItemSource,
    GetItemSource,
    GlobalSource,
    PropertyAccessorSource,
    PropertyNameSource,
    TypeDictSource,
    TypeMROSource,
    TypeSource,
)
from ..utils import (
    get_property_accessor,
    get_property_name,
    get_type_mro_no_user_code,
    is_tensor_base_attr_getter,
    is_torch_class,
    iter_mro_static_dicts,
    property_methods,
)
from .base import (
    _compute_slotdefs,
    AsPythonConstantNotImplementedError,
    AttributeMutationNew,
    GetSet,
    getset_build,
    getset_load_or_build,
    getset_read,
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
from .functions import BaseUserFunctionVariable, UserMethodVariable
from .object_protocol import mro_lookup, python_constant_repr_impl
from .user_defined import setattr_hackcheck, UserDefinedObjectVariable


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase


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


def _is_c_struct_member(descriptor: types.MemberDescriptorType) -> bool:
    """A member of a C type: may be T_OBJECT (del, then reads give None) or a
    numeric field (del raises). __slots__ members live on heap types and are
    always T_OBJECT_EX."""
    Py_TPFLAGS_HEAPTYPE = 1 << 9
    return not descriptor.__objclass__.__flags__ & Py_TPFLAGS_HEAPTYPE


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


def descr_repr(
    tx: "InstructionTranslatorBase", descr: Any, kind: str
) -> VariableTracker:
    # https://github.com/python/cpython/blob/v3.13.3/Objects/descrobject.c#L40-L48
    name, objclass = descr.__name__, descr.__objclass__
    if sys.version_info >= (3, 15) and objclass is object:
        return VariableTracker.build(tx, f"<{kind} '{name}'>")
    return VariableTracker.build(
        tx, f"<{kind} '{name}' of '{get_type_tp_name(objclass)}' objects>"
    )


def _vt_cls_modeling(vt: VariableTracker, py_type: type) -> type | None:
    if py_type is object:
        return variables.VariableTracker
    if py_type is type:
        return variables.TypeVariable

    from .base import VariableTrackerMeta

    for subclass in VariableTrackerMeta.all_subclasses:
        cpython_type = getattr(subclass, "_cpython_type", None)
        if cpython_type is None:
            continue
        types = cpython_type if isinstance(cpython_type, tuple) else (cpython_type,)
        if py_type not in types:
            continue
        # A SymNodeVariable also models int/float/bool, but other classes'
        # impls fold constants and cannot run on a symbolic value.
        if isinstance(vt, variables.SymNodeVariable) and not isinstance(vt, subclass):
            continue
        return subclass
    # An exact py_type instance has no Python overrides, so its own VT's slots
    # are py_type's C slots, e.g. SymNodeVariable for int.__neg__ on a SymInt.
    if vt.python_type() is py_type:
        return type(vt)
    return None


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

    def _get_descriptor_entry(
        self, obj: VariableTracker, name: str
    ) -> GetSet | Member | None:
        if self.as_python_constant().__objclass__ is property and isinstance(
            obj, variables.PropertyVariable
        ):
            return obj.raw_descriptor_entry(name)
        return obj.lookup_tp_getset_member(name)

    def call_getattr_fallback(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> "VariableTracker | None":
        if self.source is not None:
            install_guard(self.source.make_guard(GuardBuilder.TYPE_MATCH))
        msg = f"'{self.python_type_name()}' object has no attribute '{name}'"
        raise_observed_exception(
            AttributeError,
            tx,
            args=[msg],
            kwargs={"name": ConstantVariable.create(name), "obj": self},
        )


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

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # wrapperdescr_repr: https://github.com/python/cpython/blob/v3.13.3/Objects/descrobject.c#L71-L76
        return descr_repr(tx, self.descriptor, "slot wrapper")

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
        name, objclass = self.descriptor.__name__, self.descriptor.__objclass__
        if not issubclass(obj.python_type(), objclass):
            raise_type_error(
                tx,
                f"descriptor '{name}' requires a '{objclass.__name__}' object "
                f"but received a '{obj.python_type_name()}'",
            )
        # Dispatch through the owner (UDCV for the defining class) rather
        # than obj.call_method, which would do MRO resolution from type(obj)
        # and find Python overrides on subclasses. Routing through the class
        # mirrors CPython's wrapperdescr_call which invokes the C slot directly.
        obj = obj.realize()
        if name in ("__setattr__", "__delattr__"):
            setattr_hackcheck(tx, obj, self.descriptor)
        # The VT class modeling objclass, e.g. ConstDictVariable for
        # dict.__eq__ on an OrderedDict: its impl, not the subclass VT's.
        vt_cls = _vt_cls_modeling(obj, objclass)
        if (
            objclass is property
            and name in ("__get__", "__set__", "__delete__", "__init__")
            and isinstance(obj, variables.UserDefinedPropertyVariable)
        ):
            vt_cls = (
                variables.UserDefinedPropertyVariable
                if name == "__init__"
                else variables.PropertyVariable
            )
        sd = _compute_slotdefs(objclass).get(name)
        if vt_cls is not None and sd is not None:
            impl = getattr(vt_cls, sd.impl)
            return sd.wrapper(obj, tx, impl, rest, kwargs)
        return self.owner.call_method(
            tx, self.descriptor.__name__, [obj, *rest], kwargs
        )

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker | None,
        owner: VariableTracker,
    ) -> "WrapperDescriptorVariable | MethodWrapperVariable":
        # Mirrors wrapperdescr_get which calls PyWrapper_New to produce
        # a bound method-wrapper.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L203-L213
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L1489-L1505
        if obj is None:
            return self
        _check_descriptor_obj_type(tx, self.descriptor, obj)
        return MethodWrapperVariable(self.descriptor, obj, source=self.source)


class MethodWrapperVariable(VariableTracker):  # noqa: RVT001
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

    def get_real_python_backed_value(self) -> object:
        # The owner may be backed by a real object without being a constant
        # (e.g. an lru_cache wrapper).
        obj_value = self.obj.get_real_python_backed_value()
        if obj_value is NO_SUCH_SUBOBJ:
            return NO_SUCH_SUBOBJ
        return self.descriptor.__get__(obj_value, type(obj_value))

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

    def tp_hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        try:
            # CPython wrapper_hash:
            # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L1347
            return hash(self.as_python_constant()), False
        except NotImplementedError:
            return super().tp_hash_impl(tx)

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

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # method_repr: https://github.com/python/cpython/blob/v3.13.3/Objects/descrobject.c#L50-L55
        return descr_repr(tx, self.descriptor, "method")

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
                tx, f"unbound method {self.descriptor.__qualname__}() needs an argument"
            )
        obj, *rest = args
        name, objclass = self.descriptor.__name__, self.descriptor.__objclass__
        _check_descriptor_obj_type(tx, self.descriptor, obj)
        obj = obj.realize()

        if vt_cls := _vt_cls_modeling(obj, objclass):
            if meth := vt_cls._lookup_tp_table(name, "tp_methods"):
                result = meth(obj, tx, name, rest, kwargs)
                if result is not None:
                    return result
        if mro_lookup(obj.python_type(), name) is self.descriptor:
            # Nothing overrides it, so name lookup reaches the same C method.
            return obj.call_method(tx, name, rest, kwargs)
        unimplemented(
            gb_type="Unbound method descriptor call with unmodeled owner",
            context=f"{self.descriptor.__qualname__}({obj}, ...)",
            explanation=f"Dynamo has no model of {objclass.__qualname__}.{name} for a "
            f"{obj.python_type().__qualname__} receiver.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker | None,
        owner: VariableTracker,
    ) -> VariableTracker:
        # Mirrors method_get which calls PyCFunction_NewEx to produce a
        # bound builtin_function_or_method.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L137-L159
        # https://github.com/python/cpython/blob/3.13/Objects/methodobject.c#L40
        if obj is None:
            return self
        _check_descriptor_obj_type(tx, self.descriptor, obj)
        # PyCFunction_NewEx creates a fresh object; see ClassMethodDescriptorVariable.
        return BoundBuiltinMethodVariable(self.descriptor, obj)


@functools.cache
def _same_c_function(left: Any, right: Any) -> bool | None:
    """meth_richcompare's ml_meth check for two descriptors already known to
    share a receiver. ml_meth is only observable through that comparison, so
    bind both to one receiver and compare. Distinct descriptors can alias one C
    function, e.g. int.conjugate and int.__trunc__. None means undecidable.
    """
    if left is right:
        return True
    if type(left) is not type(right):
        return None
    if isinstance(left, types.BuiltinFunctionType):
        # Stored unbound in a type dict; m_self is the type itself.
        return left == right
    owner = left.__objclass__
    if issubclass(right.__objclass__, owner):
        owner = right.__objclass__
    if isinstance(left, types.ClassMethodDescriptorType):
        return left.__get__(None, owner) == right.__get__(None, owner)
    try:
        receiver = owner.__new__(owner)
    except TypeError:
        return None
    return left.__get__(receiver, owner) == right.__get__(receiver, owner)


def _bind_builtin_method(descriptor: Any, obj: Any) -> Any:
    if isinstance(descriptor, types.ClassMethodDescriptorType):
        return descriptor.__get__(None, obj)
    # METH_METHOD descriptors need an explicit owner on Python 3.12.
    return descriptor.__get__(obj, type(obj))


class BoundBuiltinMethodVariable(VariableTracker):  # noqa: RVT001
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
        "value",
        *VariableTracker._nonvar_fields,
    }

    def _qualname(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # https://github.com/python/cpython/blob/3.13/Objects/methodobject.c#L239
        # If __self__ is a module or NULL, return m.__name__
        #    (e.g. len.__qualname__ == 'len')
        #
        #    If __self__ is a type, return m.__self__.__qualname__ + '.' + m.__name__
        #    (e.g. dict.fromkeys.__qualname__ == 'dict.fromkeys')
        #
        #    Otherwise return type(m.__self__).__qualname__ + '.' + m.__name__
        #    (e.g. [].append.__qualname__ == 'list.append')
        from .object_protocol import generic_getattr

        name = self.descriptor.__name__
        obj = self.obj
        obj_type = obj.python_type()
        if issubclass(obj_type, types.ModuleType):
            return ConstantVariable.create(name)
        if issubclass(obj_type, type):
            owner = obj
        else:
            # Read type(obj).__qualname__ through a source so the result follows
            # a later reassignment of the class's __qualname__.
            if isinstance(obj, UserDefinedObjectVariable):
                owner_source = obj.cls_source
            else:
                owner_source = obj.source and TypeSource(obj.source)
            owner = VariableTracker.build(tx, obj_type, owner_source)
        # PyObject_GetAttr, so a __qualname__ assigned earlier in the frame is seen.
        owner_qualname = generic_getattr(tx, owner.realize(), "__qualname__")
        return ConstantVariable.create(f"{owner_qualname.as_python_constant()}.{name}")

    tp_getset = {
        "__doc__": GetSet(
            getset_build(lambda vt: vt.descriptor.__doc__),
            readonly_setter,
        ),
        "__name__": GetSet(
            getset_build(lambda vt: vt.descriptor.__name__),
            readonly_setter,
        ),
        "__qualname__": GetSet(_qualname, readonly_setter),
        "__self__": GetSet(
            getset_read(lambda vt: vt.obj),
            readonly_setter,
        ),
        "__text_signature__": GetSet(
            getset_build(lambda vt: vt.descriptor.__text_signature__),
            readonly_setter,
        ),
    }

    def __init__(
        self,
        descriptor: types.MethodDescriptorType
        | types.BuiltinFunctionType
        | types.ClassMethodDescriptorType,
        obj: VariableTracker,
        value: types.BuiltinMethodType | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.descriptor = descriptor
        self.obj = obj
        # The ID_MATCH-guarded runtime method, when built from one.
        self.value = value

    def __repr__(self) -> str:
        cls_name = getattr(
            getattr(self.descriptor, "__objclass__", None), "__name__", "?"
        )
        return f"BoundBuiltinMethodVariable({cls_name}.{self.descriptor.__name__}, {self.obj})"

    def python_type(self) -> type:
        return types.BuiltinMethodType

    def call_getattr_fallback(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> "VariableTracker | None":
        # builtin_function_or_method has no __getattr__ and no instance __dict__.
        msg = f"'builtin_function_or_method' object has no attribute '{name}'"
        raise_observed_exception(
            AttributeError,
            tx,
            args=[msg],
            kwargs={"name": ConstantVariable.create(name), "obj": self},
        )

    def tp_hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        # meth_hash: https://github.com/python/cpython/blob/e76aa128fe/Objects/methodobject.c#L319
        # It hashes m_self by identity, so only a guarded runtime method has a
        # real hash; a method bound during tracing gets a compile-time-only one.
        if self.value is not None:
            return hash(self.value), False
        return id(self), True

    def tp_richcompare_impl(self, tx, other, op):
        # meth_richcompare: equal iff both are builtin methods with the same
        # m_self and the same C function (ml_meth).
        # https://github.com/python/cpython/blob/3.13/Objects/methodobject.c#L289-L317
        if op not in ("__eq__", "__ne__") or not isinstance(
            other, BoundBuiltinMethodVariable
        ):
            return ConstantVariable.create(NotImplemented)

        if self.value is not None and other.value is not None:
            eq = self.value == other.value
            return ConstantVariable.create(eq if op == "__eq__" else not eq)

        from .object_protocol import vt_identity_compare

        same_self = vt_identity_compare(self.obj.realize(), other.obj.realize())
        eq = None
        if same_self is not None:
            eq = same_self.as_python_constant() and _same_c_function(
                self.descriptor, other.descriptor
            )
        if eq is None:
            unimplemented(
                gb_type="builtin method comparison with undecidable identity",
                context=f"{self} {op} {other}",
                explanation="Dynamo cannot determine the C-function and receiver identities of these builtin methods.",
                hints=[*graph_break_hints.DIFFICULT],
            )
        return ConstantVariable.create(eq if op == "__eq__" else not eq)

    def as_python_constant(self) -> Any:
        obj = self.obj.as_python_constant()
        if hasattr(self.descriptor, "__get__"):
            return _bind_builtin_method(self.descriptor, obj)
        return getattr(obj, self.descriptor.__name__)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from .object_protocol import mro_lookup

        # Handles calls like `list.append(L_subclass, item)`
        # https://github.com/python/cpython/blob/3.13/Objects/methodobject.c#L60-L90
        # CPython calls the C function the method was created with. Dynamo has
        # no function pointer, so it picks the implementation to trace from the
        # descriptor and the receiver.
        name = self.descriptor.__name__
        obj = self.obj.realize()
        descriptor = self.descriptor
        args = list(args)
        if isinstance(descriptor, types.ClassMethodDescriptorType):
            # PyCMethod_New bound the C function to the class in obj; the class
            # VT models the classmethod with itself as receiver.
            return obj.call_method(tx, name, args, kwargs)
        if not isinstance(descriptor, types.MethodDescriptorType):
            # METH_STATIC C functions stored directly in a type dict, e.g.
            # tuple.__new__; the receiver is the class itself.
            return obj.call_method(tx, name, args, kwargs)
        if is_torch_class(descriptor.__objclass__):
            # TorchInGraphFunctionVariable records the descriptor itself in the
            # graph, so a tensor subclass override never runs.
            fn = VariableTracker.build(tx, descriptor)
            return fn.call_function(tx, [obj, *args], kwargs)
        if mro_lookup(obj.python_type(), name) is descriptor:
            # Nothing overrides it, so name lookup reaches the same C method.
            return obj.call_method(tx, name, args, kwargs)
        if isinstance(obj, UserDefinedObjectVariable):
            base_methods = obj._base_methods
            if base_methods is not None and descriptor in base_methods:
                return obj.call_base_method(tx, name, args, kwargs)
        unimplemented(
            gb_type="Bound builtin method shadowed by an override",
            context=f"{self}",
            explanation=f"type({obj}) overrides {name}, and Dynamo has no base implementation of the bound C method to call instead.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # Rebuild through the descriptor: loading obj.<name> would resolve the
        # name on type(obj) and find a subclass override.
        descriptor = self.descriptor
        load_descriptor = codegen.create_load_const_unchecked(descriptor)
        if not hasattr(descriptor, "__get__"):
            # METH_STATIC functions are already bound, e.g. tuple.__new__.
            codegen.append_output(load_descriptor)
            return
        codegen.add_push_null(
            lambda: codegen.load_import_from(__name__, "_bind_builtin_method")
        )
        codegen.append_output(load_descriptor)
        codegen(self.obj)
        codegen.extend_output(create_call_function(2, False))


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

    @staticmethod
    def create(
        descriptor: types.ClassMethodDescriptorType,
        source: Source | None = None,
    ) -> "ClassMethodDescriptorVariable":
        return ClassMethodDescriptorVariable(descriptor, source=source)

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # method_repr: https://github.com/python/cpython/blob/v3.13.3/Objects/descrobject.c#L50-L55
        return descr_repr(tx, self.descriptor, "method")

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
        obj: VariableTracker | None,
        owner: VariableTracker,
    ) -> BoundBuiltinMethodVariable:
        # classmethod_get binds the C method to the class (ignoring obj),
        # producing a builtin_function_or_method via PyCMethod_New.
        # It first requires owner to be a type and a subtype of __objclass__.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L94-L134
        if obj is not None and owner.is_constant_none():
            owner = obj.tp_getattro_impl(tx, "__class__")
        owner_type = owner.python_type()
        if not issubclass(owner_type, type):
            raise_type_error(
                tx,
                f"descriptor '{self.descriptor.__name__}' for type "
                f"'{self.descriptor.__objclass__.__name__}' needs a type, not a "
                f"'{owner_type.__name__}' as arg 2",
            )
        owner_value = owner.get_real_python_backed_value()
        if not isinstance(owner_value, type):
            unimplemented(
                gb_type="Unresolved classmethod descriptor owner",
                context=f"{self} owner={owner}",
                explanation="Dynamo cannot resolve the class this classmethod descriptor is bound to.",
                hints=[*graph_break_hints.DYNAMO_BUG],
            )
        if not issubclass(owner_value, self.descriptor.__objclass__):
            raise_type_error(
                tx,
                f"descriptor '{self.descriptor.__name__}' requires a subtype of "
                f"'{self.descriptor.__objclass__.__name__}' but received "
                f"'{owner_value.__name__}'",
            )
        return BoundBuiltinMethodVariable(self.descriptor, owner)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # classmethoddescr_call: bind args[0] through classmethod_get, then call
        # the bound builtin_function_or_method with the remaining arguments.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L302-L326
        if not args:
            raise_type_error(
                tx,
                f"descriptor '{self.descriptor.__name__}' of "
                f"'{self.descriptor.__objclass__.__name__}' object needs an argument",
            )
        bound = self.tp_descr_get_impl(tx, None, args[0])
        return bound.call_function(tx, args[1:], kwargs)


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

    @staticmethod
    def create(fn: VariableTracker) -> "StaticMethodVariable":
        return StaticMethodVariable(fn)

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

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        return python_constant_repr_impl(self, tx)


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

    @staticmethod
    def create(fn: VariableTracker) -> "ClassMethodVariable":
        return ClassMethodVariable(fn)

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
        return ClassAttrClassMethodVariable(  # noqa: RVT001
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
        obj: VariableTracker | None,
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

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        return python_constant_repr_impl(self, tx)


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

    @staticmethod
    def create(
        descriptor: types.MemberDescriptorType,
        source: Source | None = None,
    ) -> "MemberDescriptorVariable":
        return MemberDescriptorVariable(descriptor, source=source)

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # member_repr: https://github.com/python/cpython/blob/v3.13.3/Objects/descrobject.c#L57-L62
        return descr_repr(tx, self.descriptor, "member")

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
        obj: VariableTracker | None,
        owner: VariableTracker,
    ) -> VariableTracker:
        # Mirrors member_get which calls PyMember_GetOne to read the
        # C struct field.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L162-L180
        if obj is None:
            return self
        _check_descriptor_obj_type(tx, self.descriptor, obj)
        from .object_protocol import _UnhandledDescriptorError

        attr_name = self.descriptor.__name__
        # Prefer the VT's declarative table when it models this attribute, so a
        # VT with no concrete Python object behind it still resolves.
        entry = self._get_descriptor_entry(obj, attr_name)
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
        entry = self._get_descriptor_entry(obj, name)
        if entry is not None:
            # A READONLY PyMemberDef is modeled by readonly_setter, which raises
            # what PyMember_SetOne raises.
            entry.setter(obj, tx, value)
            return ConstantVariable.create(None)
        if value is None and _is_c_struct_member(self.descriptor):
            unimplemented(
                gb_type="Delete of an unmodeled C member",
                context=f"del {obj}.{name}",
                explanation=f"Dynamo does not model deleting the C struct member '{name}'.",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
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

    @staticmethod
    def create(
        descriptor: types.GetSetDescriptorType,
        source: Source | None = None,
    ) -> "GetSetDescriptorVariable":
        return GetSetDescriptorVariable(descriptor, source=source)

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # getset_repr: https://github.com/python/cpython/blob/v3.13.3/Objects/descrobject.c#L64-L69
        if self.source:
            install_guard(self.source.make_guard(GuardBuilder.ID_MATCH))
            # descr_repr reads __objclass__.__name__, which a heap type can rename.
            objclass_source = AttrSource(self.source, "__objclass__")
            name_source = AttrSource(objclass_source, "__name__")
            install_guard(name_source.make_guard(GuardBuilder.EQUALS_MATCH))
        return descr_repr(tx, self.descriptor, "attribute")

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
        entry = self._get_descriptor_entry(obj, name)
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
        obj: VariableTracker | None,
        owner: VariableTracker,
    ) -> VariableTracker:
        # Mirrors getset_get which calls the C getter function.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L183-L197
        attr_name = self.descriptor.__name__
        # Try to eagerly call the C getter when we can obtain the
        # concrete Python object (UDOV.value, or as_python_constant
        # for classes/constants). Fall back to tp_getattro_impl for
        # proxy-based VTs like TensorVariable.
        if obj is None:
            return self
        _check_descriptor_obj_type(tx, self.descriptor, obj)
        # Prefer the VT's declarative table when it models this attribute
        entry = self._get_descriptor_entry(obj, attr_name)
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
        "fget": Member(lambda s, tx: s._get_accessor(tx, "fget"), readonly_setter),
        "fset": Member(lambda s, tx: s._get_accessor(tx, "fset"), readonly_setter),
        "fdel": Member(lambda s, tx: s._get_accessor(tx, "fdel"), readonly_setter),
        "__doc__": Member(
            getset_load_or_build(
                lambda s: s.descriptor.__doc__,
                "__doc__",
                lambda s: s.source and AttrSource(s.source, "__doc__"),
            ),
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

    def _getter_name(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        from .object_protocol import generic_getattr

        try:
            return generic_getattr(tx, self._get_accessor(tx, "fget"), "__name__")
        except ObservedAttributeError:
            handle_observed_exception(tx)
            raise_attribute_error(tx, "'property' object has no attribute '__name__'")

    def _isabstractmethod_getter(
        self, tx: "InstructionTranslatorBase"
    ) -> VariableTracker:
        from ..polyfills import property_isabstractmethod
        from .builder import SourcelessBuilder
        from .lists import TupleVariable

        # Trace truth testing so user-defined __bool__ methods and their
        # exceptions are handled within the traced program.
        fn = SourcelessBuilder.create_internal_user_function(property_isabstractmethod)
        return fn.call_function(
            tx,
            [
                TupleVariable(
                    [self._get_accessor(tx, name) for name in ("fget", "fset", "fdel")]
                )
            ],
            {},
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

    @staticmethod
    def create(
        descriptor: property, source: "Source | None" = None
    ) -> "PropertyVariable":
        return PropertyVariable(descriptor, source=source)

    def __repr__(self) -> str:
        getter = get_property_accessor(self.descriptor, "fget")
        if type(getter) is types.MethodType:
            getter = getter.__func__
        name = (
            getter.__name__
            if type(getter)
            in (
                types.FunctionType,
                types.BuiltinFunctionType,
                types.MethodDescriptorType,
            )
            else "?"
        )
        return f"PropertyVariable({name})"

    def python_type(self) -> type:
        return property

    def as_python_constant(self) -> property:
        return self.descriptor

    def get_real_python_backed_value(self) -> property:
        return self.descriptor

    def _get_accessor(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        source_cls = (
            AttrSource if type(self.descriptor) is property else PropertyAccessorSource
        )
        source = self.source and source_cls(self.source, name)
        return VariableTracker.build(
            tx, get_property_accessor(self.descriptor, name), source, realize=True
        )

    @staticmethod
    def from_descriptor(
        tx: "InstructionTranslatorBase",
        descriptor: property,
        source: "Source | None" = None,
    ) -> VariableTracker:
        if source is not None:
            return VariableTracker.build(tx, descriptor, source, realize=True)
        if type(descriptor) is property:
            return PropertyVariable.create(descriptor)
        if inspect.getattr_static(type(descriptor), "__get__") is not property.__get__:
            unimplemented(
                gb_type="Property subclass getter without a source",
                context="property subclass __get__ without a source",
                explanation="A source is required to trace a property descriptor and guard its state.",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        if descriptor in tx.output.side_effects:
            return tx.output.side_effects[descriptor]
        name = tx.output.install_global_by_id("___property", descriptor)
        return VariableTracker.build(tx, descriptor, GlobalSource(name), realize=True)

    def raw_descriptor_entry(self, name: str) -> GetSet | Member | None:
        # Explicit C descriptor calls bypass Python attribute lookup entirely.
        return VariableTracker.lookup_tp_getset_member(self, name)

    @staticmethod
    def _validate_init_args(
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> None:
        # Validate the C constructor's argument binding without invoking user code.
        try:
            property(*([None] * len(args)), **dict.fromkeys(kwargs))
        except TypeError as exc:
            raise_type_error(tx, str(exc))

    def _raise_missing_accessor(
        self, tx: "InstructionTranslatorBase", obj: VariableTracker, kind: str
    ) -> NoReturn:
        from .object_protocol import generic_getattr, generic_repr

        if sys.version_info < (3, 13):
            unimplemented(
                gb_type="Unsupported legacy property error message",
                context=f"property missing {kind}",
                explanation="Before Python 3.13 the stored property name is not exposed, so Dynamo cannot safely format its accessor error.",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        name = None
        try:
            name_vt = self._name_getter(tx)
        except ObservedAttributeError:
            handle_observed_exception(tx)
        else:
            try:
                name = str.__str__(generic_repr(tx, name_vt).as_python_constant())
            except AsPythonConstantNotImplementedError:
                unimplemented(
                    gb_type="Unsupported property name formatting",
                    context="non-constant property name representation",
                    explanation="Dynamo cannot format a property error with a non-constant name representation.",
                    hints=[*graph_break_hints.SUPPORTABLE],
                )
        cls = obj.python_type()
        if (
            inspect.getattr_static(type(cls), "__getattribute__")
            is not type.__getattribute__
        ):
            unimplemented(
                gb_type="Unsupported property owner name lookup",
                context="custom metaclass __getattribute__",
                explanation="Dynamo cannot format a property error through custom metaclass attribute access.",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        cls_source = getattr(obj, "cls_source", None) or (
            obj.source and TypeSource(obj.source)
        )
        cls_vt = VariableTracker.build(tx, cls, cls_source, realize=True)
        if tx.output.side_effects.has_pending_mutation_of_attr(cls_vt, "__qualname__"):
            owner_name = tx.output.side_effects.load_attr(cls_vt, "__qualname__")
        else:
            owner_name = generic_getattr(tx, cls_vt, "__qualname__")
        owner_repr = str.__str__(generic_repr(tx, owner_name).as_python_constant())
        prefix = "property" if name is None else f"property {name}"
        msg = f"{prefix} of {owner_repr} object has no {kind}"
        raise_attribute_error(tx, msg)

    def tp_descr_set_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        value: VariableTracker | None,
    ) -> VariableTracker:
        attr = "fset" if value is not None else "fdel"
        fn = self._get_accessor(tx, attr)

        if fn.is_constant_none():
            kind = "setter" if value is not None else "deleter"
            if type(self.descriptor) is not property:
                self._raise_missing_accessor(tx, obj, kind)
            if sys.version_info >= (3, 13):
                display_name = getattr(self.descriptor, "__name__", None)
            else:
                display_name = getattr(self.descriptor.fget, "__name__", None)
            try:
                cls_name = obj.python_type().__qualname__
            except NotImplementedError:
                cls_name = obj.python_type_name()
            prefix = (
                "property" if display_name is None else f"property '{display_name}'"
            )
            raise_attribute_error(tx, f"{prefix} of '{cls_name}' object has no {kind}")

        args = [obj] if value is None else [obj, value]
        fn.call_function(tx, args, {})
        return ConstantVariable.create(None)

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker | None,
        owner: VariableTracker,
    ) -> VariableTracker:
        # Mirrors property_descr_get: if obj is NULL or None, return self.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L1660-L1693
        if obj is None or obj.is_constant_none():
            return self
        fget_vt = self._get_accessor(tx, "fget")
        if fget_vt.is_constant_none() and type(self.descriptor) is not property:
            self._raise_missing_accessor(tx, obj, "getter")
        return fget_vt.call_function(tx, [obj], {})


class UserDefinedPropertyVariable(UserDefinedObjectVariable, PropertyVariable):
    _nonvar_fields = {
        *UserDefinedObjectVariable._nonvar_fields,
        *PropertyVariable._nonvar_fields,
    }

    def __init__(self, value: property, **kwargs: Any) -> None:
        super().__init__(value, descriptor=value, **kwargs)
        self._base_methods = property_methods
        self.property_args: list[VariableTracker] | None = None

    def _class_attribute_source(
        self, tx: "InstructionTranslatorBase", name: str, source: Source | None
    ) -> Source | None:
        if self.cls_source is not None:
            return self.get_source_by_walking_mro(tx, name)
        return source

    def raw_descriptor_entry(self, name: str) -> GetSet | Member | None:
        if name == "__doc__":
            return Member(type(self)._raw_doc_getter, unmodeled_setter)
        return super().raw_descriptor_entry(name)

    def _raw_doc_getter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        if self.property_args is not None:
            return ConstantVariable.create(None)
        source = self.source and PropertyAccessorSource(self.source, "__doc__")
        return VariableTracker.build(
            tx, get_property_accessor(self.descriptor, "__doc__"), source
        )

    def tp_init_impl(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from .object_protocol import (
            _mro_entry_source,
            generic_getattr,
            guard_mro_absent,
        )

        self._validate_init_args(tx, args, kwargs)
        cls = self.python_type()
        side_effects = tx.output.side_effects
        mro = get_type_mro_no_user_code(cls)
        if (
            sys.version_info < (3, 12)
            or not isinstance(self.mutation_type, AttributeMutationNew)
            or mro_lookup(cls, "__setattr__") is not object.__setattr__
            or any(
                base in side_effects
                and any(
                    side_effects.has_pending_mutation_of_attr(side_effects[base], name)
                    for name in ("__setattr__", "__doc__", "__bases__")
                )
                for base in mro
            )
        ):
            unimplemented(
                gb_type="Unsupported property subclass initialization",
                context="property subclass initialization",
                explanation="Dynamo cannot trace property subclass initialization on Python < 3.12, for existing objects, with custom __setattr__, or after a pending class metadata change.",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        doc_attr = mro_lookup(cls, "__doc__")
        has_doc_slot = type(doc_attr) is types.MemberDescriptorType and any(
            base is doc_attr.__objclass__ for base in mro
        )
        if not has_doc_slot and type(doc_attr) not in (str, type(None)):
            unimplemented(
                gb_type="Property subclass initialization with a custom doc setter",
                context="property subclass initialization with a custom __doc__ descriptor",
                explanation="Dynamo cannot reconstruct a property subclass with a custom __doc__ setter without repeating user code.",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        if self.cls_source is not None:
            # Rebuilding uses the default setter and the same kind of doc storage.
            for name in ("__setattr__", "__doc__"):
                for index, (_, namespace) in enumerate(iter_mro_static_dicts(cls)):
                    if name in namespace:
                        guard_mro_absent(tx, cls, self.cls_source, name, index)
                        owner = _mro_entry_source(cls, self.cls_source, index)
                        source = DictGetItemSource(TypeDictSource(owner), name)
                        guard = (
                            GuardBuilder.BUILTIN_MATCH
                            if name == "__setattr__" or has_doc_slot
                            else GuardBuilder.TYPE_MATCH
                        )
                        install_guard(source.make_guard(guard))
                        break
        names = ("fget", "fset", "fdel", "doc")
        bound: dict[str, VariableTracker] = dict.fromkeys(
            names, ConstantVariable.create(None)
        )
        bound.update(zip(names, args))
        bound.update(kwargs)
        bound = {name: value.realize() for name, value in bound.items()}
        side_effects.mutation(self)
        self.property_args = [bound[name] for name in names[:3]]
        doc = bound["doc"]
        getter_doc = False
        if doc.is_constant_none() and not bound["fget"].is_constant_none():
            try:
                fget = bound["fget"]
                if isinstance(
                    fget, variables.UserFunctionVariable
                ) and not tx.output.side_effects.has_pending_mutation_of_attr(
                    fget, "__doc__"
                ):
                    source = fget.source and AttrSource(fget.source, "__doc__")
                    doc = VariableTracker.build(tx, fget.get_function().__doc__, source)
                else:
                    doc = generic_getattr(tx, fget, "__doc__")
                doc = doc.realize()
                getter_doc = not doc.is_constant_none()
            except ObservedAttributeError:
                handle_observed_exception(tx)
                # CPython 3.12.5 stopped returning early for a missing getter doc.
                # https://github.com/python/cpython/commit/7545b7cb63a7b27de2a829b626f546088b24de47
                if sys.version_info < (3, 12, 5):
                    return ConstantVariable.create(None)
        if not type.__dict__["__dictoffset__"].__get__(cls) and not has_doc_slot:
            if getter_doc:
                raise_attribute_error(
                    tx,
                    f"'{self.python_type_name()}' object attribute '__doc__' is read-only",
                )
            return ConstantVariable.create(None)
        try:
            VariableTracker.build(tx, setattr).call_function(
                tx, [self, ConstantVariable.create("__doc__"), doc], {}
            )
        except ObservedAttributeError:
            if getter_doc:
                raise
            handle_observed_exception(tx)
        return ConstantVariable.create(None)

    def call_base_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "__init__":
            return self.tp_init_impl(tx, args, kwargs)
        return super().call_base_method(tx, name, args, kwargs)

    def lookup_tp_getset_member(self, name: str) -> GetSet | Member | None:
        if self._object_has_getattribute:
            return None
        entry = super().lookup_tp_getset_member(name)
        if name in PropertyVariable.tp_getset or name in PropertyVariable.tp_members:
            descriptor = inspect.getattr_static(property, name, NO_SUCH_SUBOBJ)
            if (
                descriptor is NO_SUCH_SUBOBJ
                or self.lookup_class_mro_attr(name) is not descriptor
            ):
                return None
            if (
                entry is not None
                and entry.getter is not None
                and self.cls_source is not None
            ):
                raw_getter = entry.getter

                def getter(obj, tx):
                    obj.get_source_by_walking_mro(tx, name)
                    return raw_getter(obj, tx)

                return type(entry)(getter, entry.setter)
        return entry

    def get_source_by_walking_mro(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> DictGetItemSource:
        source = super().get_source_by_walking_mro(tx, name)
        descriptor = self.lookup_class_mro_attr(name)
        if (
            self.cls_source is not None
            and isinstance(
                descriptor, (types.MemberDescriptorType, types.GetSetDescriptorType)
            )
            and descriptor.__objclass__ is property
        ):
            # The owner can move when __bases__ changes without gaining a shadow.
            owner_index = next(
                i
                for i, cls in enumerate(get_type_mro_no_user_code(self.python_type()))
                if cls is property
            )
            owner = GetItemSource(
                TypeMROSource(self.cls_source),
                owner_index,
            )
            install_guard(owner.make_guard(GuardBuilder.BUILTIN_MATCH))
        return source

    def _get_accessor(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        if self.property_args is not None:
            return self.property_args[("fget", "fset", "fdel").index(name)]
        return PropertyVariable._get_accessor(self, tx, name)

    def _name_getter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        if sys.version_info < (3, 13):
            return PropertyVariable._name_getter(self, tx)
        if self.property_args is not None:
            return self._getter_name(tx)
        fget = self._get_accessor(tx, "fget")
        raw_getter = get_property_accessor(self.descriptor, "fget")
        if (
            raw_getter is not None
            and type(raw_getter) not in (types.FunctionType, types.BuiltinFunctionType)
        ) or tx.output.side_effects.has_pending_mutation_of_attr(fget, "__name__"):
            unimplemented(
                gb_type="Unsupported existing property name lookup",
                context="property name may fall back to the getter's __name__",
                explanation="Dynamo cannot read an existing property's name when its getter has custom attribute access or a pending name mutation.",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        source = self.source and PropertyNameSource(self.source)
        names = VariableTracker.build(
            tx, get_property_name(self.descriptor), source, realize=True
        ).unpack_var_sequence(tx)
        if names:
            return names[0]
        raise_attribute_error(tx, "'property' object has no attribute '__name__'")

    tp_getset = {
        "__name__": GetSet(_name_getter, unmodeled_setter),
    }

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker | None,
        owner: VariableTracker,
    ) -> VariableTracker:
        if self.inherits_base_slot("__get__"):
            return PropertyVariable.tp_descr_get_impl(self, tx, obj, owner)
        return self._vectorcall_method(
            tx,
            "__get__",
            [ConstantVariable.create(None) if obj is None else obj, owner],
            {},
        )

    def tp_descr_set_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        value: VariableTracker | None,
    ) -> VariableTracker:
        if self.inherits_base_slot("__delete__" if value is None else "__set__"):
            return PropertyVariable.tp_descr_set_impl(self, tx, obj, value)
        return super().tp_descr_set_impl(tx, obj, value)

    def codegen_init(self, cg: "PyCodegen") -> None:
        if self.property_args is None:
            return

        cg.add_push_null(
            lambda: cg.load_import_from("torch._dynamo.utils", "initialize_property")
        )
        cg(self)
        for arg in self.property_args:
            cg(arg)
        cg.extend_output(create_call_function(4, False))
        cg.pop_top()


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

    @staticmethod
    def create(
        descriptor: "_collections._tuplegetter",
        source: Source | None = None,
    ) -> "TupleGetterVariable":
        return TupleGetterVariable(descriptor, source=source)

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

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        return python_constant_repr_impl(self, tx)
