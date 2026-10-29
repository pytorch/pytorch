import inspect
import sys
import types
from typing import Any, NoReturn, TYPE_CHECKING

from .. import graph_break_hints
from ..exc import (
    handle_observed_exception,
    ObservedAttributeError,
    raise_attribute_error,
    raise_type_error,
    unimplemented,
)
from ..source import (
    AttrSource,
    GlobalSource,
    PropertyAccessorSource,
    PropertyNameSource,
    TypeSource,
)
from ..utils import get_property_accessor, get_property_name
from .base import (
    AsPythonConstantNotImplementedError,
    AttrMutationKind,
    GetSet,
    getset_load_or_build,
    getset_set,
    Member,
    readonly_setter,
    VariableTracker,
)
from .constant import ConstantVariable


if TYPE_CHECKING:
    from torch._dynamo.source import Source
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase


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
        if sys.version_info >= (3, 13):
            if tx.output.side_effects.has_pending_mutation_of_attr(
                self, "__name__", AttrMutationKind.GENERIC_SETATTR
            ):
                from .misc import DeletedVariable

                name = tx.output.side_effects.load_attr(
                    self, "__name__", deleted_ok=True
                )
                if isinstance(name, DeletedVariable):
                    return self._getter_name(tx)
                return name
            fget = self._get_accessor(tx, "fget")
            raw_getter = get_property_accessor(self.descriptor, "fget")
            if (
                raw_getter is not None
                and type(raw_getter)
                not in (types.FunctionType, types.BuiltinFunctionType)
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
        from .functions import UserFunctionVariable
        from .lists import TupleVariable

        # Trace truth testing so user-defined __bool__ methods and their
        # exceptions are handled within the traced program.
        return UserFunctionVariable(property_isabstractmethod).call_function(
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

    def __repr__(self) -> str:
        return "PropertyVariable()"

    def python_type(self) -> type:
        return property

    def as_python_constant(self) -> property:
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
            return PropertyVariable(descriptor)
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
        from .base import unmodeled_setter
        from .user_defined import UserDefinedPropertyVariable

        entry = VariableTracker.lookup_tp_getset_member(self, name)
        if isinstance(self, UserDefinedPropertyVariable):
            if name == "__doc__":
                return Member(PropertyVariable._raw_doc_getter, unmodeled_setter)
            if name == "__name__" and (
                self.lookup_class_mro_attr(name) is not property.__dict__[name]
                or inspect.getattr_static(self.python_type(), "__setattr__")
                is not object.__setattr__
                or inspect.getattr_static(self.python_type(), "__delattr__")
                is not object.__delattr__
            ):
                return GetSet(type(self)._name_getter, unmodeled_setter)
        return entry

    def _raw_doc_getter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        from .user_defined import UserDefinedPropertyVariable

        if (
            isinstance(self, UserDefinedPropertyVariable)
            and self.property_args is not None
        ):
            return ConstantVariable.create(None)
        source = self.source and PropertyAccessorSource(self.source, "__doc__")
        return VariableTracker.build(
            tx, get_property_accessor(self.descriptor, "__doc__"), source
        )

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

    def tp_init_impl(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        self._validate_init_args(tx, args, kwargs)
        unimplemented(
            gb_type="Unsupported property initialization",
            context="property initialization",
            explanation="Dynamo cannot trace reinitialization of an existing property.",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    def tp_richcompare_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        op: str,
    ) -> VariableTracker:
        from .object_protocol import object_richcompare

        return object_richcompare(self, tx, other, op)

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
            self._raise_missing_accessor(
                tx, obj, "setter" if value is not None else "deleter"
            )

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
        if fget_vt.is_constant_none():
            self._raise_missing_accessor(tx, obj, "getter")
        return fget_vt.call_function(tx, [obj], {})
