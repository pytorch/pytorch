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
from typing import Any, cast, TYPE_CHECKING, TypeAlias

from torch._C._dynamo import get_type_tp_name

from .. import graph_break_hints, variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import (
    get_dynamo_observed_exception,
    raise_attribute_error,
    raise_observed_exception,
    raise_type_error,
    unimplemented,
)
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, TypeMROSource, TypeSource
from ..utils import is_tensor_base_attr_getter, is_torch_class, istype
from .base import (
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
    type_qualified_name,
    unmodeled_setter,
    VariableTracker,
)
from .constant import ConstantVariable
from .functions import (
    BaseUserFunctionVariable,
    LocalGeneratorFunctionVariable,
    NestedUserFunctionVariable,
    UserFunctionVariable,
    UserMethodVariable,
)
from .object_protocol import python_constant_repr_impl
from .user_defined import UserDefinedObjectVariable


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
        name = self.descriptor.__name__
        _check_descriptor_obj_type(tx, self.descriptor, obj)
        obj = obj.realize()
        if isinstance(obj, UserDefinedObjectVariable):
            base_methods = obj._base_methods
            if base_methods is not None and self.descriptor in base_methods:
                return obj.call_base_method(tx, name, rest, kwargs)
        method = obj.lookup_tp_method(name)
        if method is not None:
            result = method(obj, tx, name, rest, kwargs)
            if result is not None:
                return result
        return self.owner.call_method(tx, name, [obj, *rest], kwargs)

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        owner: VariableTracker,
    ) -> VariableTracker:
        # Mirrors method_get which calls PyCFunction_NewEx to produce a
        # bound builtin_function_or_method.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L137-L159
        # https://github.com/python/cpython/blob/3.13/Objects/methodobject.c#L40
        if obj.is_constant_none():
            return self
        _check_descriptor_obj_type(tx, self.descriptor, obj)
        # PyCFunction_NewEx creates a fresh object; see ClassMethodDescriptorVariable.
        return BoundBuiltinMethodVariable(self.descriptor, obj)

    def tp_richcompare_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        op: str,
    ) -> VariableTracker:
        # PyMethodDescr_Type leaves tp_richcompare NULL and inherits
        # object_richcompare from object.
        from .object_protocol import object_richcompare

        return object_richcompare(self, tx, other, op)


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

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
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
        obj: VariableTracker,
        owner: VariableTracker,
    ) -> BoundBuiltinMethodVariable:
        # classmethod_get binds the C method to the class (ignoring obj),
        # producing a builtin_function_or_method via PyCMethod_New.
        # It first requires owner to be a type and a subtype of __objclass__.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L94-L134
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
        bound = self.tp_descr_get_impl(tx, ConstantVariable.create(None), args[0])
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

    def __init__(
        self,
        descriptor: VariableTracker,
        value: staticmethod | None = None,  # type: ignore[type-arg]
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.value = value
        self.descriptor = descriptor
        self.initial_dict: dict[str, VariableTracker] = {}

    def _get_metadata_owner(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        owner = self.descriptor
        if istype(owner, UserMethodVariable):
            owner = owner.im_func
        if istype(owner, UserFunctionVariable):
            owner = owner._get_mutation_owner(tx)
        return owner

    @classmethod
    def create(
        cls, tx: "InstructionTranslatorBase", descriptor: VariableTracker
    ) -> "StaticMethodVariable":
        from .object_protocol import generic_getattr

        result = cls(descriptor, mutation_type=AttributeMutationNew())
        tx.output.side_effects.track_attribute_mutation_new(result)
        owner = result._get_metadata_owner(tx)
        wrapped = (
            NO_SUCH_SUBOBJ
            if isinstance(
                descriptor,
                (
                    NestedUserFunctionVariable,
                    LocalGeneratorFunctionVariable,
                    UserDefinedObjectVariable,
                    variables.UserDefinedClassVariable,
                ),
            )
            else descriptor.get_real_python_backed_value()
        )
        if wrapped is not NO_SUCH_SUBOBJ:
            # Read the callable's runtime metadata, including functools.wraps
            # updates, rather than the original name/doc in its code object.
            for name, value in staticmethod(cast(Any, wrapped)).__dict__.items():
                if (
                    name == "__annotations__"
                    and isinstance(owner, BaseUserFunctionVariable)
                ) or tx.output.side_effects.has_pending_mutation_of_attr(owner, name):
                    result.initial_dict[name] = generic_getattr(tx, owner, name)
                else:
                    result.initial_dict[name] = VariableTracker.build(
                        tx,
                        value,
                        owner.source and AttrSource(owner.source, name),
                    )
            return result
        attrs = ("__module__", "__name__", "__qualname__", "__doc__")
        if sys.version_info < (3, 14):
            attrs += ("__annotations__",)
        for name in attrs:
            try:
                value = generic_getattr(tx, owner, name)
            except get_dynamo_observed_exception(AttributeError):
                tx.exn_vt_stack.clear_current_exception()
                continue
            result.initial_dict[name] = value
        return result

    def get_real_python_backed_value(self) -> object:
        return self.value if self.value is not None else NO_SUCH_SUBOBJ

    def get_initial_instance_dict(self) -> dict[str, VariableTracker] | None:
        return self.initial_dict if self.value is None else None

    def reconstruct_for_side_effects(self, codegen: "PyCodegen") -> bool:
        self.reconstruct(codegen)
        return True

    def _is_abstract(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        from .object_protocol import generic_getattr, generic_is_true

        value = generic_getattr(
            tx, self.descriptor, "__isabstractmethod__", ConstantVariable.create(False)
        )
        return generic_is_true(tx, value)

    def _get_wrapped_attr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        from .object_protocol import generic_getattr

        return self.get_dict_vt(tx).getitem_or_default(
            name, lambda: generic_getattr(tx, self._get_metadata_owner(tx), name)
        )

    def _set_wrapped_attr(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        value: VariableTracker | None,
    ) -> None:
        attrs = self.get_dict_vt(tx)
        if value is None:
            if not attrs.contains(name):
                msg = f"'staticmethod' object has no attribute '{name}'"
                raise_observed_exception(AttributeError, tx, args=[msg])
            attrs.delitem(name)
        else:
            attrs.setitem(name, value)

    tp_getset = {
        "__dict__": GetSet(lambda s, tx: s.get_dict_vt(tx), unmodeled_setter),
        "__isabstractmethod__": GetSet(_is_abstract, readonly_setter),
    }
    if sys.version_info >= (3, 14):
        for attr in ("__annotations__", "__annotate__"):
            tp_getset[attr] = GetSet(
                lambda s, tx, name=attr: s._get_wrapped_attr(tx, name),
                lambda s, tx, value, name=attr: s._set_wrapped_attr(tx, name, value),
            )

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
            value=descriptor,
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
        if self.value is not None and self.dict_vt is None:
            return self.value
        # New descriptors have runtime identity; the real object also cannot
        # represent pending mutations once its instance dict is tracked.
        return super().as_python_constant()

    def lookup_instance_dict(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker | None:
        attrs = self.get_dict_vt(tx)
        return attrs.getitem(name) if attrs.contains(name) else None

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        try:
            return super().tp_getattro_impl(tx, name)
        except NotImplementedError:
            raise_observed_exception(
                AttributeError,
                tx,
                args=[f"'staticmethod' object has no attribute '{name}'"],
            )

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name in ("__setattr__", "__delattr__"):
            attr = args[0].as_python_constant()
            value = args[1] if name == "__setattr__" else None
            getset = self.lookup_tp_getset_member(attr)
            if getset is not None:
                getset.setter(self, tx, value)
            else:
                self._set_wrapped_attr(tx, attr, value)
            return ConstantVariable.create(None)
        return super().call_method(tx, name, args, kwargs)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        return self.descriptor.call_function(tx, args, kwargs)

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
        # Reconstruction may see the function after its metadata mutations.
        # Restore copies whose wrapped slot changed, or whose custom getter
        # could depend on another pending attribute mutation.
        side_effects = codegen.tx.output.side_effects
        owner = self._get_metadata_owner(codegen.tx)
        custom_getters = isinstance(
            self.descriptor,
            (UserDefinedObjectVariable, variables.UserDefinedClassVariable),
        ) and bool(side_effects.store_attr_mutations.get(self.descriptor))
        if custom_getters:
            # The constructor can run before the callable's attributes are
            # replayed. Restore the complete snapshot to preserve its key order.
            codegen.dup_top()
            codegen.append_output(codegen.create_load_attr("__dict__"))
            codegen.load_method("clear")
            codegen.call_method(0)
            codegen.append_output(create_instruction("POP_TOP"))
            codegen.clear_tos()
        for name, value in self.initial_dict.items():
            annotations_changed = (
                name == "__annotations__"
                and isinstance(owner, BaseUserFunctionVariable)
                and (
                    owner.annotations is not value
                    or (
                        isinstance(owner, UserFunctionVariable)
                        and value.source
                        != (owner.source and AttrSource(owner.source, name))
                    )
                )
            )
            if (
                custom_getters
                or annotations_changed
                or side_effects.has_pending_mutation_of_attr(owner, name)
            ):
                codegen.dup_top()
                codegen(value)
                codegen.extend_output(codegen.rot_n(2))
                codegen.store_attr(name)

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
        tx.output.side_effects.store_attr(obj, name, stored)
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
            getset_load_or_build(
                lambda s: s.descriptor.fget,
                "fget",
                lambda s: s.source and AttrSource(s.source, "fget"),
            ),
            readonly_setter,
        ),
        "fset": Member(
            getset_load_or_build(
                lambda s: s.descriptor.fset,
                "fset",
                lambda s: s.source and AttrSource(s.source, "fset"),
            ),
            readonly_setter,
        ),
        "fdel": Member(
            getset_load_or_build(
                lambda s: s.descriptor.fdel,
                "fdel",
                lambda s: s.source and AttrSource(s.source, "fdel"),
            ),
            readonly_setter,
        ),
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
            display_name = getattr(self.descriptor, "__name__", None)
            kind = "setter" if value is not None else "deleter"
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

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        return python_constant_repr_impl(self, tx)
