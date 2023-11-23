from __future__ import annotations

import dataclasses
import enum
import functools
import logging
import weakref
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
)

import torch
from torch.utils._traceback import CapturedTraceback

log = logging.getLogger(__name__)


if TYPE_CHECKING:
    # Import the following modules during type checking to enable code intelligence features,
    # such as auto-completion in tools like pylance, even when these modules are not explicitly
    # imported in user code.

    import sympy


"""
torch._guards is the definitional source of truth for general purpose guard structures.

An important thing to keep in mind here is the preservation of layering. There should be no dynamo notions,
and no guard installation notions here.
"""


class GuardSource(enum.Enum):
    LOCAL = 0
    GLOBAL = 1
    LOCAL_NN_MODULE = 2
    GLOBAL_NN_MODULE = 3
    CONSTANT = 4
    RANDOM_VALUE = 5
    SHAPE_ENV = 6
    LOCAL_FSDP_MODULE = 7
    GLOBAL_FSDP_MODULE = 8

    def is_fsdp_module(self) -> bool:
        return self in (GuardSource.GLOBAL_FSDP_MODULE, GuardSource.LOCAL_FSDP_MODULE)

    def is_nn_module(self) -> bool:
        return (
            self
            in (
                GuardSource.GLOBAL_NN_MODULE,
                GuardSource.LOCAL_NN_MODULE,
            )
            or self.is_fsdp_module()
        )

    def is_local(self):
        return self in (
            GuardSource.LOCAL,
            GuardSource.LOCAL_NN_MODULE,
            GuardSource.LOCAL_FSDP_MODULE,
        )


"""
Base class for a "GuardBuilder" role.

The GuardBuilderBase role is to represent a scope within which to build a guard. The name is a little
confusing, as its not a builder, but for the sake of avoiding a lot of renames and keeping the original reference
to torchdynamo's GuardBuilder.

Note: create_fn is invoked with a GuardBuilderBase and a Guard. A GuardBuilder is chosen based
on GuardSource's select function.

There is value in keeping this GuardBuilderBase empty to keep layering clean.
"""


class GuardBuilderBase:
    pass


class ShapeGuard(NamedTuple):
    expr: sympy.Expr
    stack: CapturedTraceback


@dataclasses.dataclass
class Guard:
    # originating_source is the source that called the make_guard method to
    # construct this guard object. The property name specifies what exactly it
    # is the guard is guarding on.  The meaning of the name is dependent on the
    # create_fn; you must look at the use-site inside create_fn to know what
    # name means.
    #
    # That being said, although you might think this is just a "name", name is
    # usually an arbitrary Python expression that will be evaluated with all
    # globals (and locals, if you create a LOCAL guard) to extract the Python
    # object that we want to perform guard tests on.  This evaluation
    # typically happens in GuardBuilder.eval.  In these cases, name is
    # typically produced by originating_source.name() (not to be confused with
    # GuardSource - the property source).
    #
    # Occasionally, name is not a valid Python expression; sometimes
    # it is meaningless.  Example create_fns that are like this include
    # GRAD_MODE and SHAPE_ENV.
    originating_source: Source
    create_fn: Callable[[GuardBuilderBase, Guard], None]

    # Export only. These values are written to at time of guard check_fn creation.
    guard_types: Optional[List[str]] = None
    code_list: Optional[List[str]] = None
    obj_weakref: Optional[object] = None
    guarded_class_weakref: Optional[type] = None

    stack = None
    user_stack = None
    _hash = None

    def __hash__(self):
        if self._hash is None:
            self._hash = hash((self.name, self.source, id(self.create_fn)))
        return self._hash

    def sort_key(self):
        return (
            self.source.value if self.source else -1,
            len(self.name),
            self.name,
            self.inner_create_fn().__code__.co_firstlineno,
        )

    def __lt__(self, other):
        return self.sort_key() < other.sort_key()

    def inner_create_fn(self):
        if isinstance(self.create_fn, functools.partial):
            return self.create_fn.func
        else:
            return self.create_fn

    @property
    def name(self) -> str:
        return self.originating_source.name()

    @property
    def source(self) -> GuardSource:
        return self.originating_source.guard_source()

    @staticmethod
    def weakref_to_str(obj_weakref):
        """
        This is a workaround of a Python weakref bug.

        `obj_weakref` is instance returned by `weakref.ref`,
        `str(obj_weakref)` is buggy if the original obj overrides __getattr__, e.g:

            class MyConfig(dict):
                def __getattr__(self, x):
                    return self[x]

            obj = MyConfig(offset=5)
            obj_weakref = weakref.ref(obj)
            str(obj_weakref)  # raise error: KeyError: '__name__'
        """
        if isinstance(obj_weakref, weakref.ReferenceType):
            obj = obj_weakref()
            if obj is not None:
                return f"<weakref at {hex(id(obj_weakref))}; to '{obj.__class__.__name__}' at {hex(id(obj))}>"
            else:
                return f"<weakref at {hex(id(obj_weakref))}; dead>"
        else:
            return str(obj_weakref)

    def __repr__(self):
        s = f"""
        {self.source.name.lower() if self.source else ""} {repr(self.name)} {self.inner_create_fn().__name__}
        {{
            'guard_types': {self.guard_types},
            'code': {self.code_list},
            'obj_weakref': {self.weakref_to_str(self.obj_weakref)}
            'guarded_class': {self.guarded_class_weakref}
        }}
        """
        return s

    def __str__(self):
        output = f"Name: {repr(self.name)}\n"
        source = self.source.name.lower() if self.source else ""
        output += f"    Source: {source}\n"
        output += f"    Create Function: {self.inner_create_fn().__name__}\n"
        output += f"    Guard Types: {self.guard_types}\n"
        output += f"    Code List: {self.code_list}\n"
        output += f"    Object Weakref: {self.weakref_to_str(self.obj_weakref)}\n"
        output += f"    Guarded Class Weakref: {self.guarded_class_weakref}\n"
        return output

    def create(self, builder: GuardBuilderBase):
        try:
            return self.create_fn(builder, self)
        except Exception:
            log.error("Error while creating guard:\n%s", str(self).rstrip())
            if self.stack:
                log.error("Created at:\n%s", "".join(self.stack.format()[-4:]).rstrip())
            raise

    def is_nn_module(self):
        return self.source.is_nn_module()

    def is_fsdp_module(self):
        return self.source.is_fsdp_module()

    def is_local(self):
        return self.source.is_local()

    def set_export_info(self, guard_type, guarded_class, code_list, obj_weakref):
        if not self.guard_types:
            self.guard_types = list()

        self.guard_types.append(guard_type)

        assert self.guarded_class_weakref in (
            guarded_class,
            None,
        ), "Guarded class id must be identical, or None"
        self.guarded_class_weakref = guarded_class

        if not self.code_list:
            self.code_list = code_list
        else:
            self.code_list.extend(code_list)

        assert self.obj_weakref in (
            obj_weakref,
            None,
        ), "Guarded object must be identical, or None"
        self.obj_weakref = obj_weakref


T = TypeVar("T")

"""
Parent structure for guard env expressions.
A GuardEnvExpr can have any subtype.
Note: All subtypes must be handled exhaustively in
torch._dynamo.guards._parse_guard_env_guards to avoid a RuntimeError.
"""


@dataclasses.dataclass
class GuardEnvExpr:
    pass


"""
A class representing a pair of duplicate inputs.
input_pos_a and input_pos_b are input positions we have deduped.
"""


@dataclasses.dataclass
class DuplicateInputs(GuardEnvExpr):
    input_source_a: Source
    input_source_b: Source

    def __post_init__(self):
        assert self.input_source_a != self.input_source_b


# TODO(voz): Move all the checkpointable, contexts, context state, etc
# into its own file
"""
Checkpointable is an interface for driving state snapshotting, left purposely vague for now.

copy_graphstate() -> T, a somewhat legacy name, is expected to emit a snapshot of any type that
can also be taken in at restore_graphstate(T) calls.

When to snapshot, is, at the moment, an implementation detail of upstream callers. Checkpointable
does not provide any garuantees around consistency, idempotency, or safety of calling its APIs, yet.

In the future, it will have a closer coupling to a generic Checkpoint management system.
"""


class Checkpointable(ABC, Generic[T]):
    @abstractmethod
    def copy_graphstate(self) -> T:
        ...

    @abstractmethod
    def restore_graphstate(self, state: T):
        ...


"""
The GuardCheckpointState - it is the T of Checkpointable[T] for GuardsContext
"""


class GuardsCheckpointState:
    dynamo_guards: Set[Guard] = set()

    def __init__(self, dynamo_guards):
        self.dynamo_guards = dynamo_guards

    """
    Produces a delta against another GuardsCheckpointState.

    Returns None if no delta is found, otherwise, return a set() of mismatched
    Guard type objects.
    """

    def diff(self, other):
        r = self.dynamo_guards.difference(other.dynamo_guards)
        if len(r) == 0:
            return None
        return r

    def __eq__(self, other):
        return self.diff(other) is None


class ModuleContextCheckpointState:
    nn_modules: Dict[str, torch.nn.Module] = {}

    def __init__(self, nn_modules):
        self.nn_modules = nn_modules

    """
    Produces a delta against another ModuleContextCheckpointState.

    Returns None if no delta is found, otherwise, return a set() of mismatched
    module key names.
    """

    def diff(self, other):
        r = set(self.nn_modules.keys()).difference(set(other.nn_modules.keys()))
        if len(r) == 0:
            return None
        return r

    def __eq__(self, other):
        return self.diff(other) is None


class ModuleContext(Checkpointable[ModuleContextCheckpointState]):
    def __init__(self):
        self.nn_modules: Dict[str, Any] = {}

    def copy_graphstate(self):
        return ModuleContextCheckpointState(dict(self.nn_modules))

    def restore_graphstate(self, state):
        assert isinstance(state, ModuleContextCheckpointState)
        self.nn_modules = state.nn_modules


class GlobalContextCheckpointState:
    global_state: Dict[str, Tuple[Callable, ...]] = {}

    def __init__(self, global_states):
        self.global_state = global_states

    """
    Produces a delta against another GlobalContextCheckpointState.

    Returns None if no delta is found, otherwise, return a set() of mismatched
    global key names.
    """

    def diff(self, other):
        r = set(self.global_state.keys()).difference(set(other.global_state.keys()))
        if len(r) == 0:
            return None
        return r

    def __eq__(self, other):
        return self.diff(other) is None


class GlobalContext(Checkpointable[GlobalContextCheckpointState]):
    """
    This keeps track of the global torch state during tracing of a function.
    For example, torch.is_grad_enabled.
    """

    _supported_global_states = {
        "grad_enabled",
        "torch_function_enabled",
        "autocast_enabled",
        "autocast_cpu_enabled",
        "autocast_gpu_dtype",
        "autocast_cpu_dtype",
        "autocast_cache_enabled",
    }

    def __init__(self):
        self.global_state: Dict[str, Tuple[Callable, ...]] = {}

    def copy_graphstate(self):
        return GlobalContextCheckpointState(dict(self.global_state))

    def restore_graphstate(self, state):
        assert isinstance(state, GlobalContextCheckpointState)
        self.global_state = state.global_state
        assert (
            len(self.global_state) == len(self._supported_global_states)
            and set(self.global_state.keys()) == self._supported_global_states
        ), "Global state mismatch"
        for func, args in self.global_state.values():
            func(args)


"""
A GuardsContext is a checkpointable representation of all the guards in the current tracing
context. It's lifecycle is bound 1:1 to the tracing context, and it should never be instantiated
directly outside of it. For passing around internal state representations of this object,
prefer to extract them with copy_graphstate to produce a GuardsCheckpointState.
"""


# Like a Set[Guard] but will record the user stack on all guards at the
# time they were installed at their destination
class GuardsSet:
    def __init__(self, inner=None):
        if inner is None:
            inner = set()
        self.inner = inner

    def __iter__(self):
        return iter(self.inner)

    def __len__(self):
        return len(self.inner)

    # Subtraction along with bool is typically used to determine the delta of
    # added guards between checkpoints for higher order ops
    def __sub__(self, other):
        return GuardsSet(self.inner - other.inner)

    def __bool__(self):
        return bool(self.inner)

    def add(self, guard: Guard, *, skip=0):
        if guard in self.inner:
            return
        if guard.stack is None:
            guard.stack = CapturedTraceback.extract(skip=1 + skip)
        if guard.user_stack is None:
            guard.user_stack = torch._tracing_context.TracingContext.extract_stack()
        self.inner.add(guard)

    def update(self, *others: Set[Guard]):
        for o in others:
            for g in o:
                self.add(g, skip=1)


class GuardsContext(Checkpointable[GuardsCheckpointState]):
    def __init__(self):
        self.dynamo_guards: GuardsSet = GuardsSet()
        self.aotautograd_guards: List[GuardEnvExpr] = []

    def copy_graphstate(self):
        return GuardsCheckpointState(set(self.dynamo_guards.inner))

    def restore_graphstate(self, state):
        # NB: "steals" the passed in state
        assert isinstance(state, GuardsCheckpointState)
        self.dynamo_guards = GuardsSet(state.dynamo_guards)


# Subclasses can be found in torch/_dynamo/source.py
# TODO(voz): Consider a toplevel torch/_source.py
@dataclasses.dataclass(frozen=True)
class Source:
    def reconstruct(self, codegen):
        raise NotImplementedError()

    def guard_source(self) -> GuardSource:
        raise NotImplementedError()

    def name(self) -> str:
        raise NotImplementedError()

    def make_guard(self, fn) -> Guard:
        if self.guard_source() is GuardSource.CONSTANT:
            raise NotImplementedError()
        return Guard(self, fn)

    def is_nn_module(self) -> bool:
        return self.guard_source().is_nn_module()


# Subclasses can be found in torch/_dynamo/source.py
@dataclasses.dataclass(frozen=True)
class ChainedSource(Source):
    base: Source
