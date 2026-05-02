"""
Variable trackers for CPython descriptor types.

Each class mirrors a CPython descriptor type (PyWrapperDescr_Type,
PyMethodDescr_Type, etc.) and implements tp_descr_get_impl to model
the descriptor binding step faithfully.
"""

import types
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING

from ..exc import raise_observed_exception, raise_type_error
from .base import VariableTracker


if TYPE_CHECKING:
    from torch._guards import Source

    from ..codegen import PyCodegen
    from ..symbolic_convert import InstructionTranslator


# Avoid circular import; use late binding via the `variables` module.
from .. import variables


class WrapperDescriptorVariable(VariableTracker):
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
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        assert isinstance(descriptor, types.WrapperDescriptorType)
        self.descriptor = descriptor

    def __repr__(self) -> str:
        cls_name = self.descriptor.__objclass__.__name__
        return f"WrapperDescriptorVariable({cls_name}.{self.descriptor.__name__})"

    def python_type(self) -> type:
        return types.WrapperDescriptorType

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        # descr_members: __objclass__ and __name__ are PyMemberDef on all
        # descriptor types.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L641-L645
        if name == "__objclass__":
            return VariableTracker.build(tx, self.descriptor.__objclass__)
        if name == "__name__":
            return variables.ConstantVariable.create(self.descriptor.__name__)
        return super().var_getattr(tx, name)

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # Unbound call: list.__add__([1,2], [3,4]) -- first arg is self.
        # Mirrors wrapperdescr_call.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L535
        if not args:
            raise_type_error(
                tx,
                f"descriptor '{self.descriptor.__name__}' of "
                f"'{self.descriptor.__objclass__.__name__}' object needs an argument",
            )
        obj, *rest = args
        return obj.call_method(tx, self.descriptor.__name__, list(rest), kwargs)

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslator",
        obj: VariableTracker,
        name: str,
        **kwargs: Any,
    ) -> "MethodWrapperVariable":
        # Mirrors wrapperdescr_get which calls PyWrapper_New to produce
        # a bound method-wrapper.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L203-L213
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L1489-L1505
        return MethodWrapperVariable(self.descriptor, obj, name, **kwargs)


class MethodWrapperVariable(VariableTracker):
    """Bound method-wrapper (wrapper_descriptor bound to an instance).

    Produced by WrapperDescriptorVariable.tp_descr_get_impl, mirroring
    PyWrapper_New which stores a reference to the descriptor and the instance.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L1450
    """

    _nonvar_fields = {
        "descriptor",
        "name",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: types.WrapperDescriptorType,
        obj: VariableTracker,
        name: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        assert isinstance(descriptor, types.WrapperDescriptorType)
        assert isinstance(obj, VariableTracker)
        self.descriptor = descriptor
        self.obj = obj
        self.name = name

    def __repr__(self) -> str:
        return f"MethodWrapperVariable({self.descriptor}, {self.obj}, {self.name})"

    def python_type(self) -> type:
        return types.MethodWrapperType

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        return self.obj.call_method(tx, self.name, list(args), kwargs)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen(self.obj)
        codegen.extend_output(codegen.create_load_attrs(self.name))


class MethodDescriptorVariable(VariableTracker):
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
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        assert isinstance(descriptor, types.MethodDescriptorType)
        self.descriptor = descriptor

    def __repr__(self) -> str:
        cls_name = self.descriptor.__objclass__.__name__
        return f"MethodDescriptorVariable({cls_name}.{self.descriptor.__name__})"

    def python_type(self) -> type:
        return types.MethodDescriptorType

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        # descr_members: __objclass__ and __name__ are PyMemberDef on all
        # descriptor types.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L641-L645
        if name == "__objclass__":
            return VariableTracker.build(tx, self.descriptor.__objclass__)
        if name == "__name__":
            return variables.ConstantVariable.create(self.descriptor.__name__)
        return super().var_getattr(tx, name)

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # Unbound call: list.append([1,2,3], 4) -- first arg is self.
        if not args:
            raise_type_error(
                tx,
                f"descriptor '{self.descriptor.__name__}' of "
                f"'{self.descriptor.__objclass__.__name__}' object needs an argument",
            )
        obj, *rest = args
        return obj.call_method(tx, self.descriptor.__name__, list(rest), kwargs)

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslator",
        obj: VariableTracker,
        name: str,
        **kwargs: Any,
    ) -> "BuiltinMethodVariable":
        # Mirrors method_get which calls PyCFunction_NewEx to produce a
        # bound builtin_function_or_method.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L137-L159
        # https://github.com/python/cpython/blob/3.13/Objects/methodobject.c#L40
        return BuiltinMethodVariable(self.descriptor, obj, name, **kwargs)


class BuiltinMethodVariable(VariableTracker):
    """Bound builtin method (method_descriptor bound to an instance).

    Produced by MethodDescriptorVariable.tp_descr_get_impl, mirroring
    PyCFunction_NewEx which creates a PyCFunctionObject storing the
    PyMethodDef and the bound instance.
    https://github.com/python/cpython/blob/3.13/Objects/methodobject.c#L331
    """

    _nonvar_fields = {
        "descriptor",
        "name",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: types.MethodDescriptorType,
        obj: VariableTracker,
        name: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        assert isinstance(descriptor, types.MethodDescriptorType)
        assert isinstance(obj, VariableTracker)
        self.descriptor = descriptor
        self.obj = obj
        self.name = name

    def __repr__(self) -> str:
        cls_name = self.descriptor.__objclass__.__name__
        return (
            f"BuiltinMethodVariable({cls_name}.{self.descriptor.__name__}, {self.obj})"
        )

    def python_type(self) -> type:
        return types.BuiltinMethodType

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        return self.obj.call_method(tx, self.name, list(args), kwargs)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen(self.obj)
        codegen.extend_output(codegen.create_load_attrs(self.name))


class ClassMethodDescriptorVariable(VariableTracker):
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
        assert isinstance(descriptor, types.ClassMethodDescriptorType)
        self.descriptor = descriptor

    def __repr__(self) -> str:
        cls_name = self.descriptor.__objclass__.__name__
        return f"ClassMethodDescriptorVariable({cls_name}.{self.descriptor.__name__})"

    def python_type(self) -> type:
        return types.ClassMethodDescriptorType

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        # descr_members: __objclass__ and __name__ are PyMemberDef on all
        # descriptor types.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L641-L645
        if name == "__objclass__":
            return VariableTracker.build(tx, self.descriptor.__objclass__)
        if name == "__name__":
            return variables.ConstantVariable.create(self.descriptor.__name__)
        return super().var_getattr(tx, name)

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslator",
        obj: VariableTracker,
        name: str,
        source: "Source | None" = None,
    ) -> VariableTracker:
        # classmethod_get binds to the class, producing a
        # builtin_function_or_method.  It ignores obj and uses type.
        # When accessed on an instance, type = Py_TYPE(obj).
        # When accessed on a class, type is the class itself.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L94-L134
        try:
            obj_value = obj.as_python_constant()
        except NotImplementedError:
            obj_value = obj.value  # type: ignore[attr-defined]
        func = self.descriptor.__get__(obj_value, None)
        return VariableTracker.build(tx, func, source)


class StaticMethodVariable(VariableTracker):
    """staticmethod descriptor wrapping a callable.

    CPython's staticmethod (PyStaticMethod_Type) is a non-data descriptor
    whose tp_descr_get (sm_descr_get) simply returns the wrapped callable,
    ignoring both obj and type.
    https://github.com/python/cpython/blob/3.13/Objects/funcobject.c#L1520
    https://github.com/python/cpython/blob/3.13/Objects/funcobject.c#L1418-L1428
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: staticmethod,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        assert isinstance(descriptor, staticmethod)
        self.descriptor = descriptor

    def __repr__(self) -> str:
        func_name = getattr(self.descriptor.__func__, "__name__", "?")
        return f"StaticMethodVariable({func_name})"

    def python_type(self) -> type:
        return staticmethod

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslator",
        obj: VariableTracker | None,
        name: str,
        source: "Source | None" = None,
    ) -> VariableTracker:
        # sm_descr_get returns sm->sm_callable unconditionally.
        # https://github.com/python/cpython/blob/3.13/Objects/funcobject.c#L1418-L1428
        return VariableTracker.build(tx, self.descriptor.__func__, source)


class ClassMethodVariable(VariableTracker):
    """classmethod descriptor wrapping a callable.

    CPython's classmethod (PyClassMethod_Type) is a non-data descriptor
    whose tp_descr_get (cm_descr_get) creates a bound method of the
    wrapped callable bound to the class (via PyMethod_New).
    https://github.com/python/cpython/blob/3.13/Objects/funcobject.c#L1314
    https://github.com/python/cpython/blob/3.13/Objects/funcobject.c#L1215-L1227
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: classmethod,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        assert isinstance(descriptor, classmethod)
        self.descriptor = descriptor

    def __repr__(self) -> str:
        func_name = getattr(self.descriptor.__func__, "__name__", "?")
        return f"ClassMethodVariable({func_name})"

    def python_type(self) -> type:
        return classmethod

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslator",
        obj: VariableTracker,
        name: str,
        source: "Source | None" = None,
        source_fn: "Source | None" = None,
    ) -> VariableTracker:
        # cm_descr_get calls PyMethod_New(cm->cm_callable, type) to bind
        # the wrapped function to the class.
        # https://github.com/python/cpython/blob/3.13/Objects/funcobject.c#L1215-L1227
        cls_vt = obj.var_getattr(tx, "__class__")
        return variables.UserMethodVariable(
            self.descriptor.__func__, cls_vt, source_fn=source_fn, source=source
        )


class MemberDescriptorVariable(VariableTracker):
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
        assert isinstance(descriptor, types.MemberDescriptorType)
        self.descriptor = descriptor

    def __repr__(self) -> str:
        cls_name = self.descriptor.__objclass__.__name__
        return f"MemberDescriptorVariable({cls_name}.{self.descriptor.__name__})"

    def python_type(self) -> type:
        return types.MemberDescriptorType

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        # descr_members: __objclass__ and __name__ are PyMemberDef on all
        # descriptor types.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L641-L645
        if name == "__objclass__":
            return VariableTracker.build(tx, self.descriptor.__objclass__)
        if name == "__name__":
            return variables.ConstantVariable.create(self.descriptor.__name__)
        return super().var_getattr(tx, name)

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslator",
        obj: VariableTracker,
        name: str,
        source: "Source | None" = None,
    ) -> VariableTracker:
        # Mirrors member_get which calls PyMember_GetOne to read the
        # C struct field value.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L162-L180
        try:
            resolved = self.descriptor.__get__(obj.value)
        except AttributeError:
            raise_observed_exception(
                AttributeError,
                tx,
                args=[
                    f"'{type(obj.value).__name__}' object has no attribute '{name}'"
                ],
            )
        return VariableTracker.build(tx, resolved, source)


class GetSetDescriptorVariable(VariableTracker):
    """C getter/setter descriptor (getset_descriptor on a type).

    CPython exposes C getter/setter pairs defined via PyGetSetDef as
    getset_descriptor objects (PyGetSetDescr_Type).  These are data
    descriptors used for computed attributes backed by C functions
    (e.g. object.__class__, type.__dict__).
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L830

    getset_get calls the C getter function.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L183-L197
    """

    def __init__(self, desc: types.GetSetDescriptorType, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.desc = desc

    def get_real_python_backed_value(self) -> types.GetSetDescriptorType:
        return self.desc

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        from ..source import AttrSource

        if name == "__get__" and self.source:
            source = AttrSource(self.source, "__get__")
            return VariableTracker.build(tx, self.desc.__get__, source)
        elif name in ("__objclass__", "__name__"):
            source = self.source and AttrSource(self.source, name)
            return VariableTracker.build(tx, getattr(self.desc, name), source)
        else:
            return super().var_getattr(tx, name)

    def is_python_constant(self) -> bool:
        return True

    def as_python_constant(self) -> types.GetSetDescriptorType:
        return self.desc

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslator",
        obj: VariableTracker,
        name: str,
        source: "Source | None" = None,
    ) -> VariableTracker:
        # Mirrors getset_get which calls the C getter function.
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L183-L197
        try:
            resolved = self.desc.__get__(obj.value)
        except AttributeError:
            raise_observed_exception(
                AttributeError,
                tx,
                args=[
                    f"'{type(obj.value).__name__}' object has no attribute '{name}'"
                ],
            )
        return VariableTracker.build(tx, resolved, source)


class PropertyVariable(VariableTracker):
    """Python property descriptor.

    The property type is a data descriptor with tp_descr_get =
    property_descr_get which calls fget(obj) to compute the value.
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L2046
    https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L1660-L1693
    """

    _nonvar_fields = {
        "descriptor",
        *VariableTracker._nonvar_fields,
    }

    def __init__(
        self,
        descriptor: property,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        assert isinstance(descriptor, property)
        self.descriptor = descriptor

    def __repr__(self) -> str:
        fget_name = getattr(self.descriptor.fget, "__name__", "?")
        return f"PropertyVariable({fget_name})"

    def python_type(self) -> type:
        return property

    def tp_descr_get_impl(
        self,
        tx: "InstructionTranslator",
        obj: VariableTracker,
        name: str,
        source: "Source | None" = None,
    ) -> VariableTracker:
        # Mirrors property_descr_get which calls fget(obj).
        # https://github.com/python/cpython/blob/3.13/Objects/descrobject.c#L1660-L1693
        fget_vt = VariableTracker.build(
            tx, self.descriptor.fget, source=source, realize=True
        )
        return fget_vt.call_function(tx, [obj], {})
