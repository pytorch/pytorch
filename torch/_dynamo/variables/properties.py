import sys
from typing import Any, TYPE_CHECKING

from .. import graph_break_hints
from ..exc import (
    handle_observed_exception,
    ObservedAttributeError,
    raise_attribute_error,
    raise_type_error,
    unimplemented,
)
from ..source import AttrSource, PropertyNameSource
from ..utils import get_property_name
from .base import (
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
        if sys.version_info >= (3, 13):
            if tx.output.side_effects.has_pending_mutation_of_attr(
                self, "__name__", AttrMutationKind.GENERIC_SETATTR
            ):
                return tx.output.side_effects.load_attr(self, "__name__")
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

    def _isabstractmethod_getter(
        self, tx: "InstructionTranslatorBase"
    ) -> VariableTracker:
        from ..polyfills import property_isabstractmethod
        from .functions import UserFunctionVariable

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

    def _get_accessor(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        source = self.source and AttrSource(self.source, name)
        return VariableTracker.build(
            tx, getattr(self.descriptor, name), source, realize=True
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

    def tp_descr_set_impl(
        self,
        tx: "InstructionTranslatorBase",
        obj: VariableTracker,
        value: VariableTracker | None,
    ) -> VariableTracker:
        attr = "fset" if value is not None else "fdel"
        fn = self._get_accessor(tx, attr)

        if fn.is_constant_none():
            name = None
            if sys.version_info >= (3, 13):
                try:
                    name = repr(self._name_getter(tx).as_python_constant())
                except ObservedAttributeError:
                    handle_observed_exception(tx)
            kind = "setter" if value is not None else "deleter"
            if sys.version_info >= (3, 11):
                # property_descr_set formats %R of the owner's *type*, whose
                # repr is its bare __qualname__ (no module prefix) -- unlike
                # python_qualified_name(), which mirrors the module-qualified
                # _PyType_GetFullyQualifiedName used elsewhere (e.g. __repr__).
                try:
                    owner_name = obj.python_type().__qualname__
                except NotImplementedError:
                    owner_name = obj.python_type_name()
                if name is not None:
                    msg = f"property {name} of {owner_name!r} object has no {kind}"
                else:
                    msg = f"property of {owner_name!r} object has no {kind}"
            else:
                # < 3.11: no owner/property-name in the message at all.
                verb = "set" if value is not None else "delete"
                msg = f"can't {verb} attribute"
            raise_attribute_error(tx, msg)

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
        return fget_vt.call_function(tx, [obj], {})
