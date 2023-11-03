import collections
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from torch._guards import TracingContext

from .. import variables
from ..current_scope_id import current_scope_id
from ..exc import unimplemented
from ..source import AttrSource, Source
from ..utils import dict_values, identity, istype, odict_values


class MutableLocalSource(Enum):
    """
    If the VariableTracker.mutable_local represents a Variable that:
    - already existed that Dynamo began tracking while introspection (Existing)
    - is a new variable that is created during Dynamo introspection (Local)
    """

    Existing = 0
    Local = 1


class ParentsTracker:
    """
    This is a perf optimization to limit the number of objects we need to visit in tx.replace_all.
    This must be a seperate object so that it is not cloned in apply.
    """

    def __init__(self):
        # logically this is a set, but we use a dict to ensure deterministic ordering
        self.parents: Dict[ParentsTracker, bool] = dict()

    def add(self, parent):
        self.parents[parent] = True

    def recursive_parents(self):
        rv = dict(self.parents)
        worklist = list(self.parents)
        while worklist:
            for parent in worklist.pop().parents:
                if parent not in rv:
                    assert isinstance(parent, ParentsTracker)
                    rv[parent] = True
                    worklist.append(parent)
        return rv.keys()


class MutableLocalBase:
    """
    Base class for Variable.mutable_local
    """

    def __init__(self, typ: MutableLocalSource):
        # In HigherOrderOperator tracing, we need to distinguish
        # between MutableLocals inside the HigherOrderOperator and
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
        if typ is MutableLocalSource.Existing:
            self.scope = 0
        elif typ is MutableLocalSource.Local:
            self.scope = current_scope_id()
        else:
            unimplemented(f"Unsupported MutableLocalSource: {typ}")


class MutableLocal(MutableLocalBase):
    """
    Marker used to indicate this (list, iter, etc) was constructed in
    local scope and can be mutated safely in analysis without leaking
    state.
    """

    def __init__(self):
        super().__init__(MutableLocalSource.Local)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


def _is_top_level_scope(scope_id):
    return scope_id == 1


def is_side_effect_safe(m: MutableLocalBase):
    scope_id = current_scope_id()

    # In the top-level scope (if no HigherOrderOperators are involved),
    # we are allowed to modify variables created in this scope as well
    # as existing variables.
    if _is_top_level_scope(scope_id):
        return True
    # Otherwise, only allow local mutation of variables created in the current scope
    return m.scope == scope_id


# metaclass to call post_init
class HasPostInit(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__(*args, **kwargs)
        return obj


class VariableTracker(metaclass=HasPostInit):
    """
    Base class for tracked locals and stack values

    VariableTracker instances are immutable and should be copied in
    order to change them.
    """

    # fields to leave unmodified in apply()
    _nonvar_fields = {
        "value",
        "guards",
        "source",
        "mutable_local",
        "parents_tracker",
        "user_code_variable_name",
    }

    @staticmethod
    def propagate(*vars: List[List["VariableTracker"]]):
        """Combine the guards from many VariableTracker into **kwargs for a new instance"""
        guards = set()

        ret = {
            "guards": guards,
        }
        tcx = TracingContext.get()
        if tcx is not None and tcx.export:
            """
            Why we want to skip gurad propagation when exporting graph?
            For export path, we haven't found a way to meaningfully use generated guards from
            torch dynamo, since exported programs are also used in non Python environments.

            At the time of writing this today, we observed some slowness in dynamo that there's
            a substantial portion of time consumed in guard propagation logic, especially when
            we're dealing with immutable VTs. This doesn't improve the time complexity of VT
            propagation algorithm overall but greatly reduce the constant factor of it, thus still
            gives us great speedup for exporting time. We've discussed the resolution for the root
            cause but as a quick unblocker in the short term we might still want to skip guards to
            enable several use cases.
            """
            return ret

        def visit(var):
            if type(var) in (list, tuple, dict_values, odict_values):
                for i in var:
                    visit(i)
            else:
                assert isinstance(var, VariableTracker), typestr(var)
                guards.update(var.guards)

        visit(vars)
        return ret

    def clone(self, **kwargs):
        """Shallow copy with some (optional) changes"""
        args = dict(self.__dict__)
        args.update(kwargs)
        return self.__class__(**args)

    @classmethod
    def copy(cls, value):
        """Deeper (but not full) copy, leaving FX and user objects alone"""
        return cls.apply(identity, value)

    @classmethod
    def apply(
        cls,
        fn: Callable[["VariableTracker"], "VariableTracker"],
        value,
        cache=None,
        skip_fn=lambda _: False,  # Whether we should skip applying to this var
    ):
        """
        Walk this object and call fn on all the VariableTracker
        instances to produce a new VariableTracker with the results.
        """
        if cache is None:
            cache = dict()

        idx = id(value)
        if idx in cache:
            return cache[idx][0]

        if isinstance(value, VariableTracker):
            if not skip_fn(value):
                updated_dict = dict(value.__dict__)
                for key in updated_dict.keys():
                    if key not in value._nonvar_fields:
                        updated_dict[key] = cls.apply(
                            fn, updated_dict[key], cache, skip_fn
                        )
                result = fn(value.clone(**updated_dict))
            else:
                result = fn(value)

        elif istype(value, list):
            result = [cls.apply(fn, v, cache, skip_fn) for v in value]
        elif istype(value, tuple):
            result = tuple(cls.apply(fn, v, cache, skip_fn) for v in value)
        elif istype(value, collections.OrderedDict):
            result = collections.OrderedDict(
                cls.apply(fn, v, cache, skip_fn) for v in value.items()
            )
        elif istype(value, dict):
            assert "__name__" not in value, "_nonvar_fields should have excluded this"
            result = {
                k: cls.apply(fn, v, cache, skip_fn) for k, v in list(value.items())
            }
        else:
            result = value

        # save `value` to keep it alive and ensure id() isn't reused
        cache[idx] = (result, value)
        return result

    def add_guard(self, guard):
        return self.clone(guards=set.union(self.guards, {guard}))

    def add_guards(self, guards):
        if guards is None:
            return self
        assert isinstance(guards, set)
        return self.clone(guards=set.union(self.guards, guards))

    def add_options(self, options, *more):
        if more:
            return self.add_options(options).add_options(*more)
        if isinstance(options, VariableTracker):
            return self.add_guards(options.guards)
        assert isinstance(options, dict)
        return self.add_guards(options.get("guards", set()))

    def __str__(self):
        return f"{self.__class__.__name__}()"

    def __repr__(self):
        return str(self)

    def python_type(self):
        raise NotImplementedError(f"{self} has no type")

    def as_python_constant(self):
        """For constants"""
        raise NotImplementedError(f"{self} is not a constant")

    def is_python_constant(self):
        try:
            self.as_python_constant()
            return True
        except NotImplementedError:
            return False

    def can_make_guard(self):
        try:
            self.make_guard(None)
            return True
        except NotImplementedError:
            return False

    def make_guard(self, fn):
        if self.source:
            return self.source.make_guard(fn)
        raise NotImplementedError()

    def replace_guards(self, guards, *fns):
        name = self.source.name()
        new_guards = {g for g in (guards or []) if g.name != name}
        new_guards.update(self.source.make_guard(fn) for fn in fns)
        return new_guards

    def const_getattr(self, tx, name: str) -> Any:
        """getattr(self, name) returning a python constant"""
        raise NotImplementedError()

    def var_getattr(self, tx, name: str) -> "VariableTracker":
        """getattr(self, name) returning a new variable"""
        options = VariableTracker.propagate(self)

        value = self.const_getattr(tx, name)
        if not variables.ConstantVariable.is_literal(value):
            raise NotImplementedError()
        if self.source:
            options["source"] = AttrSource(self.source, name)
        return variables.ConstantVariable.create(value, **options)

    def is_proxy(self):
        try:
            self.as_proxy()
            return True
        except NotImplementedError:
            return False

    def as_proxy(self):
        raise NotImplementedError(str(self))

    def reconstruct(self, codegen):
        raise NotImplementedError()

    def unpack_var_sequence(self, tx):
        raise NotImplementedError()

    def has_unpack_var_sequence(self, tx):
        try:
            self.unpack_var_sequence(tx)
            return True
        except NotImplementedError:
            return False

    def num_parameters(self):
        unimplemented(f"num_parameters: {self}")

    def call_hasattr(self, tx, name: str) -> "VariableTracker":
        unimplemented(f"hasattr: {repr(self)}")

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
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
            return variables.ConstantVariable.create(
                len(self.unpack_var_sequence(tx)), **VariableTracker.propagate(self)
            )
        elif (
            name == "__getattr__"
            and len(args) == 1
            and args[0].is_python_constant()
            and not kwargs
        ):
            return self.var_getattr(tx, args[0].as_python_constant()).add_options(
                self, args[0]
            )
        raise unimplemented(f"call_method {self} {name} {args} {kwargs}")

    def rename(self, tx, name):
        new_name = tx.output.new_var(name)
        if not self.mutable_local or not isinstance(self.mutable_local, MutableLocal):
            # This is fine for objects that are not mutable locals
            self.user_code_variable_name = new_name
            return self
        new_vt = self.clone(user_code_variable_name=new_name)
        return tx.replace_all(self, new_vt)

    def __init__(
        self,
        *,
        guards: Optional[Set] = None,
        source: Source = None,
        mutable_local: MutableLocal = None,
        user_code_variable_name: str = None,
        parents_tracker: ParentsTracker = None,
    ):
        super().__init__()
        self.guards = guards or set()
        self.source = source
        self.mutable_local = mutable_local
        self.user_code_variable_name = user_code_variable_name
        self.parents_tracker = parents_tracker

    def __post_init__(self, *args, **kwargs):
        if self.parents_tracker is None:
            self.parents_tracker = ParentsTracker()
        # visit children 1 level deep and ensure parent is set properly
        VariableTracker.apply(
            lambda node: node.parents_tracker.add(self.parents_tracker),
            [v for k, v in self.__dict__.items() if k not in self._nonvar_fields],
            skip_fn=lambda _: True,
        )


def typestr(*objs):
    if len(objs) == 1:
        (obj,) = objs
        if isinstance(obj, VariableTracker):
            return str(obj)
        else:
            return type(obj).__name__
    else:
        return " ".join(map(typestr, objs))
