# mypy: ignore-errors

import collections
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .. import variables
from ..current_scope_id import current_scope_id
from ..exc import unimplemented
from ..source import AttrSource, Source
from ..utils import istype


if TYPE_CHECKING:
    from .symbolic_convert import InstructionTranslator, InstructionTranslatorBase


class SourceType(Enum):
    """
    This Enum divides VariableTracker into 2 cases, depending on the variable
    it represents:
    - already existed that Dynamo began tracking while introspection (Existing)
    - is a new variable that is created during Dynamo introspection (New)

    In general, we have these invariants:
    1. for `VariableTracker` associated with `Existing`, its `source` field must not be None.
    2. for `VariableTracker` associated with `New`, most of the time its
       `source` field is None, except for cases like side effect codegen for
       `AttributeMutationNew`, during which we generate a
       `LocalSource('tmp...')` for such variable, to facilitate codegen.
    """

    Existing = 0
    New = 1


class MutationType:
    """
    Base class for Variable.mutation_type. It encodes information about
    1. The type of mutation Dynamo allows on the variable.
    2. Whether the value represented by this variable already existed before
    Dynamo tracing.
    """

    def __init__(self, typ: SourceType) -> None:
        # In HigherOrderOperator tracing, we need to distinguish
        # between MutationTypes inside the HigherOrderOperator and
        # ones outside it. For example, it is not safe to mutate
        # `a` in the following example because it was constructed
        # in a different scope.
        #
        # def f(x):
        #     a = 1
        #     def g(x):
        #         nonlocal a
        #         a = 2
        #         return x
        #     return wrap(g, x) + a
        #
        # We use self.scope to distinguish this.
        # scope == 0: The object was an existing variable
        # scope == 1: The object was created while Dynamo
        #             was introspecting a function
        #             (and no HigherOrderOps were involved)
        # scope >= 2: The object was created through
        #             Dynamo introspection of a HigherOrderOp.
        #             The exact number corresponds to the level
        #             of nested HigherOrderOps.
        if typ is SourceType.Existing:
            self.scope = 0
        elif typ is SourceType.New:
            self.scope = current_scope_id()
        else:
            unimplemented(f"Unsupported SourceType: {typ}")


class ValueMutationNew(MutationType):
    """
    This case of VariableTracker.mutation_type marker indicates
    1. Dynamo allows mutation on the value itself (rather than its attributes).
    2. The value is created by the bytecode Dynamo is tracing through.

    For instance, Dynamo could model a newly created list with this marker,
    indicating that while we need to model mutations to this list, we don't have
    to emit bytecode for these mutations if the list doesn't escape into the
    Python world.
    """

    def __init__(self) -> None:
        super().__init__(SourceType.New)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


class ValueMutationExisting(MutationType):
    """
    This case of VariableTracker.mutation_type marker indicates
    1. Dynamo allows mutation on the value itself (rather than its attributes).
    2. The value exists before Dynamo tracing started.

    For instance, Dynamo could model a pre-existing list with this marker,
    indicating that if we encounter mutations to this list, we need to buffer
    and re-apply those mutations after the graph runs, since the list might be
    used afterwards in Python.
    """

    # A flag to indicate whether mutation happened on the associated
    # `VariableTracker`. This enables SideEffects to accurately and quickly
    # filter out which pre-existing values it needs to generate mutation for.
    is_modified: bool

    def __init__(self, is_modified: bool = False):
        super().__init__(SourceType.Existing)
        self.is_modified = is_modified


class AttributeMutation(MutationType):
    """
    This case of VariableTracker.mutation_type marker indicates that Dynamo
    allows mutation on the value's attributes.
    """

    def __init__(self, typ: SourceType):
        super().__init__(typ)


class AttributeMutationExisting(AttributeMutation):
    """
    This case of VariableTracker.mutation_type marker indicates
    1. Dynamo allows mutation on the value's attributes.
    2. The value exists before Dynamo tracing started.

    For instance, Dynamo could model a pre-existing object with this marker,
    indicating that if we encounter mutations to this object, we need to buffer
    then re-apply those mutations after the graph runs, since the object might
    be used afterwards in Python.
    """

    def __init__(self):
        super().__init__(SourceType.Existing)


class AttributeMutationNew(AttributeMutation):
    """
    This case of VariableTracker.mutation_type marker indicates
    1. Dynamo allows mutation on the value's attributes.
    2. The value is created by the bytecode Dynamo is tracing through.

    For instance, Dynamo could model a newly created object with this marker,
    indicating that while we need to model mutations to this object, we don't
    have to emit bytecode for these mutations if the object doesn't escape into
    the Python world.
    """

    def __init__(self, cls_source: Optional[Source] = None):
        super().__init__(SourceType.New)
        self.cls_source = cls_source


def _is_top_level_scope(scope_id):
    return scope_id == 1


def is_side_effect_safe(m: MutationType):
    scope_id = current_scope_id()

    # In the top-level scope (if no HigherOrderOperators are involved),
    # we are allowed to modify variables created in this scope as well
    # as existing variables.
    if _is_top_level_scope(scope_id):
        return True
    # Otherwise, only allow local mutation of variables created in the current scope
    return m.scope == scope_id


class VariableTrackerMeta(type):
    all_subclasses = []

    def __instancecheck__(cls, instance) -> bool:
        """Make isinstance work with LazyVariableTracker"""
        if type.__instancecheck__(
            variables.LazyVariableTracker, instance
        ) and cls not in (
            VariableTracker,
            variables.LazyVariableTracker,
        ):
            instance = instance.realize()
        return type.__instancecheck__(cls, instance)

    def __init__(cls, name, bases, attrs) -> None:
        super().__init__(name, bases, attrs)
        VariableTrackerMeta.all_subclasses.append(cls)


class VariableTracker(metaclass=VariableTrackerMeta):
    """
    Base class for tracked locals and stack values

    VariableTracker instances are immutable and should be copied in
    order to change them.

    Prefer the factory function VariableTracker.build() over VariableTracker.__init__().
    """

    # fields to leave unmodified in apply()
    _nonvar_fields = {
        "value",
        "guards",
        "source",
        "mutation_type",
        "parents_tracker",
        "user_code_variable_name",
    }

    def clone(self, **kwargs):
        """Shallow copy with some (optional) changes"""
        args = dict(self.__dict__)
        args.update(kwargs)
        return self.__class__(**args)

    @classmethod
    def visit(
        cls,
        fn: Callable[["VariableTracker"], None],
        value: Any,
        cache: Optional[Dict[int, Any]] = None,
    ) -> None:
        """
        Walk value and call fn on all the VariableTracker instances
        """
        if cache is None:
            cache = {}

        idx = id(value)
        if idx in cache:
            return
        # save `value` to keep it alive and ensure id() isn't reused
        cache[idx] = value

        if isinstance(value, VariableTracker):
            value = value.unwrap()
            fn(value)
            value = value.unwrap()  # calling fn() might have realized it
            nonvars = value._nonvar_fields
            for key, subvalue in value.__dict__.items():
                if key not in nonvars:
                    cls.visit(fn, subvalue, cache)
        elif istype(value, (list, tuple)):
            for subvalue in value:
                cls.visit(fn, subvalue, cache)
        elif istype(value, (dict, collections.OrderedDict)):
            for subvalue in value.values():
                cls.visit(fn, subvalue, cache)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def debug_repr(self):
        # Intended to be overridden to provide more info
        try:
            return repr(self.as_python_constant())
        except NotImplementedError:
            return repr(self)

    def python_type(self):
        """
        Abstract method to be implemented by subclasses of VariableTracker.

        This method should return the type represented by the instance of the subclass.
        The purpose is to provide a standardized way to retrieve the Python type information
        of the variable being tracked.

        Returns:
            type: The Python type (such as int, str, list, etc.) of the variable tracked by
                the subclass. If the type cannot be determined or is not relevant,
                leaving it undefined or invoking super() is always sound.

        Note:
            This is an abstract method and may be overridden in subclasses.

        Example:
            class SetVariable(VariableTracker):
                def python_type(self):
                    return set

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        try:
            return type(self.as_python_constant())
        except NotImplementedError:
            raise NotImplementedError(f"{self} has no type") from None

    def as_python_constant(self):
        """For constants"""
        raise NotImplementedError(f"{self} is not a constant")

    def guard_as_python_constant(self):
        """Similar to as_python_constant(), but add ID_MATCH guards to try to force things to become constants"""
        try:
            return self.as_python_constant()
        except NotImplementedError as e:
            unimplemented(str(e))

    def is_python_constant(self):
        try:
            self.as_python_constant()
            return True
        except NotImplementedError:
            return False

    def make_guard(self, fn):
        if self.source:
            return self.source.make_guard(fn)
        raise NotImplementedError

    def const_getattr(self, tx: "InstructionTranslator", name: str) -> Any:
        """getattr(self, name) returning a python constant"""
        raise NotImplementedError

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> "VariableTracker":
        """getattr(self, name) returning a new variable"""
        value = self.const_getattr(tx, name)
        if not variables.ConstantVariable.is_literal(value):
            raise NotImplementedError
        source = self.source and AttrSource(self.source, name)
        return variables.ConstantVariable.create(value, source=source)

    def is_proxy(self):
        try:
            self.as_proxy()
            return True
        except NotImplementedError:
            return False

    def as_proxy(self):
        raise NotImplementedError(str(self))

    def maybe_fx_node(self):
        try:
            proxy = self.as_proxy()
            import torch.fx

            if isinstance(proxy, torch.fx.Proxy):
                return proxy.node
            return None
        except NotImplementedError:
            return None

    def reconstruct(self, codegen):
        raise NotImplementedError

    def can_reconstruct(self, tx):
        """If it is possible to reconstruct the Python object this
        VariableTracker represents."""
        assert tx is tx.output.root_tx, "Only root tx can reconstruct"
        try:
            from ..codegen import PyCodegen

            cg = PyCodegen(tx)
            self.reconstruct(cg)
            return True
        except NotImplementedError:
            return False

    def unpack_var_sequence(self, tx) -> List["VariableTracker"]:
        raise NotImplementedError

    def force_unpack_var_sequence(self, tx) -> List["VariableTracker"]:
        # like unpack_var_sequence, but should only be used when it is
        # safe to eagerly (vs. lazily) unpack this variable.
        # e.g. map(f, x) is normally evaluated lazily but sometimes
        # we want to force eager unpacking, e.g. when converting to a list.
        # NOTE: this method is allowed to mutate the VariableTracker, so
        # it should only be called once.
        return self.unpack_var_sequence(tx)

    def has_unpack_var_sequence(self, tx) -> bool:
        try:
            self.unpack_var_sequence(tx)
            return True
        except NotImplementedError:
            return False

    # NB: don't call force_unpack_var_sequence, especially if it mutates!
    def has_force_unpack_var_sequence(self, tx) -> bool:
        return self.has_unpack_var_sequence(tx)

    def inspect_parameter_names(self) -> List[str]:
        unimplemented(f"inspect_parameter_names: {self}")

    def call_hasattr(self, tx: "InstructionTranslator", name: str) -> "VariableTracker":
        unimplemented(f"hasattr {self.__class__.__name__} {name}")

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        unimplemented(f"call_function {self} {args} {kwargs}")

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if name == "__len__" and self.has_unpack_var_sequence(tx):
            assert not (args or kwargs)
            return variables.ConstantVariable.create(len(self.unpack_var_sequence(tx)))
        elif (
            name == "__getattr__"
            and len(args) == 1
            and args[0].is_python_constant()
            and not kwargs
        ):
            return self.var_getattr(tx, args[0].as_python_constant())
        unimplemented(f"call_method {self} {name} {args} {kwargs}")

    def set_name_hint(self, name):
        pass

    def realize(self) -> "VariableTracker":
        """Used by LazyVariableTracker to build the real VariableTracker"""
        return self

    def unwrap(self) -> "VariableTracker":
        """Used by LazyVariableTracker to return the real VariableTracker if it already exists"""
        return self

    def is_realized(self):
        """Used by LazyVariableTracker to indicate an unrealized node"""
        return True

    def next_variable(self, tx):
        unimplemented(f"next({self})")

    def is_strict_mode(self, tx):
        return tx.strict_checks_fn and tx.strict_checks_fn(self)

    def is_mutable(self):
        """Whether Dynamo allows mutation on this variable."""
        return not self.is_immutable()

    def is_immutable(self):
        """Whether Dynamo bans mutation on this variable."""
        return self.mutation_type is None

    @staticmethod
    def build(
        tx: "InstructionTranslatorBase",
        value: Any,
        source: Optional[Source] = None,
    ) -> Any:
        """Create a new VariableTracker from a value and optional Source"""
        from . import builder

        if source is None:
            return builder.SourcelessBuilder.create(tx, value)
        else:
            return builder.VariableBuilder(tx, source)(value)

    def __init__(
        self,
        *,
        source: Source = None,
        mutation_type: MutationType = None,
    ) -> None:
        super().__init__()
        self.source = source
        self.mutation_type = mutation_type


def typestr(*objs):
    if len(objs) == 1:
        (obj,) = objs
        if isinstance(obj, VariableTracker):
            return str(obj)
        else:
            return type(obj).__name__
    else:
        return " ".join(map(typestr, objs))
