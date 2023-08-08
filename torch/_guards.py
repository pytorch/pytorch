import contextlib

import dataclasses
import enum
import logging
import traceback
import unittest.mock
import weakref
from abc import ABC, abstractmethod
from contextlib import contextmanager
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
    TypeVar,
)

import torch

log = logging.getLogger(__name__)

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

    def select(self, locals_, globals_):
        # SHAPE_ENV counts as locals, because the guard expressions
        # created by shape env can reference f_locals
        #
        # RANDOM_VALUE counts as locals, because what we do is we run
        # Python RNG and assign it to a temporary, and then perform
        # guard tests on that temporary
        if self in (
            GuardSource.LOCAL,
            GuardSource.LOCAL_NN_MODULE,
            GuardSource.LOCAL_FSDP_MODULE,
            GuardSource.SHAPE_ENV,
            GuardSource.RANDOM_VALUE,
        ):
            return locals_
        if self in (
            GuardSource.GLOBAL,
            GuardSource.GLOBAL_NN_MODULE,
            GuardSource.GLOBAL_FSDP_MODULE,
        ):
            return globals_
        raise NotImplementedError(str(self))

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
    # TODO: store this in slightly less formatted form
    stack: str


@dataclasses.dataclass
class Guard:
    # The name of a Guard specifies what exactly it is the guard is guarding
    # on.  The meaning of the name is dependent on the create_fn; you must
    # look at the use-site inside create_fn to know what name means.
    #
    # That being said, although you might think this is just a "name", name is
    # usually an arbitrary Python expression that will be evaluated with all
    # globals (and locals, if you create a LOCAL guard) to extract the Python
    # object that we want to perform guard tests on.  This evaluation
    # typically happens in GuardBuilder.eval.  In these cases, name is
    # typically produced by Source.name() (not to be confused with
    # GuardSource)--morally, we could have stored a Source here.
    #
    # Occasionally, name is not a valid Python expression; sometimes
    # it is meaningless.  Example create_fns that are like this include
    # GRAD_MODE and SHAPE_ENV.
    name: str
    source: GuardSource
    create_fn: Callable[[GuardBuilderBase, "Guard"], None]
    is_volatile: bool = False

    # Export only. These values are written to at time of guard check_fn creation.
    guard_types: Optional[List[str]] = None
    code_list: Optional[List[str]] = None
    obj_weakref: Optional[object] = None
    guarded_class_weakref: Optional[type] = None

    def __hash__(self):
        return hash((self.name, self.source, id(self.create_fn)))

    def sort_key(self):
        return (
            self.source.value if self.source else -1,
            len(self.name),
            self.name,
            self.create_fn.__code__.co_firstlineno,
        )

    def __lt__(self, other):
        return self.sort_key() < other.sort_key()

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
        {self.source.name.lower() if self.source else ""} {repr(self.name)} {self.create_fn.__name__}
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
        output += f"    Create Function: {self.create_fn.__name__}\n"
        output += f"    Guard Types: {self.guard_types}\n"
        output += f"    Code List: {self.code_list}\n"
        output += f"    Object Weakref: {self.weakref_to_str(self.obj_weakref)}\n"
        output += f"    Guarded Class Weakref: {self.guarded_class_weakref}\n"
        return output

    def create(self, local_builder: GuardBuilderBase, global_builder: GuardBuilderBase):
        return self.create_fn(self.source.select(local_builder, global_builder), self)

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
    input_source_a: "Source"
    input_source_b: "Source"

    def __post_init__(self):
        assert self.input_source_a != self.input_source_b


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
        self.nn_modules: Dict[str, torch.nn.Module] = {}

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


class GuardsContext(Checkpointable[GuardsCheckpointState]):
    def __init__(self):
        self.dynamo_guards: Set[Guard] = set()
        self.aotautograd_guards: List[GuardEnvExpr] = []

    def copy_graphstate(self):
        return GuardsCheckpointState(set(self.dynamo_guards))

    def restore_graphstate(self, state):
        assert isinstance(state, GuardsCheckpointState)
        self.dynamo_guards = state.dynamo_guards


_CURRENT_TRACING_CONTEXT = None

"""
TracingContext is the source of truth for all currently accumulated information
needed to trace. Its lifecycle is kept 1:1 when using TorchDynamo, but other systems
are open to managing their own TracingContext with that in mind.

Currently, only guards live on the TracingContext, in the form of a GuardsContext.
However, future implementations will move FakeTensorMode (and its owned ShapeEnv), as well
as other structures into it.

The purpose of TracingContext is not to be a dumping ground, or god object, but rather to avoid
having to plumb complex subsystems across multiple verticals.

Ex: A common example is guard accumulation between dynamo, shape_env, aot_autograd, and inductor.
Accessing the current tracing context via
TracingContext.get() allows users to accumulate their own guards for processing, without needing to know how
to plumb objects back up to where frame interpretation happend.
"""


class TracingContext:
    """
    Provides the currently installed TracingContext, or None.

    Note that it is a staticmethod, and invocations outside of `with tracing()` (see below), are valid but
    will return None.
    """

    @staticmethod
    def get() -> Optional["TracingContext"]:
        return _CURRENT_TRACING_CONTEXT

    def __init__(self, fake_mode):
        self.guards_context = GuardsContext()
        self.module_context = ModuleContext()
        self.global_context = GlobalContext()
        self.fake_mode = fake_mode
        self.frame_summary_stack = []
        self.loc_in_frame = None
        # this is only set after aot_autograd
        self.fw_metadata = None
        self.params_flat = None

    @staticmethod
    def extract_stack():
        self = TracingContext.get()
        if self is None:
            return traceback.StackSummary()
        stack = list(self.frame_summary_stack)
        if self.loc_in_frame is not None:
            stack.append(self.loc_in_frame)
        return traceback.StackSummary.from_list(stack)

    # Call this when you want to call into some code that isn't necessarily
    # associated with the current frame state
    @staticmethod
    @contextlib.contextmanager
    def clear_frame():
        tc = TracingContext.get()
        assert (
            tc is not None
        ), "Frame context manager must be called within an ongoing trace."
        with unittest.mock.patch.object(
            tc, "frame_summary_stack", []
        ), unittest.mock.patch.object(tc, "loc_in_frame", None):
            yield

    @staticmethod
    @contextlib.contextmanager
    def current_frame(frame_summary):
        tc = TracingContext.get()
        assert (
            tc is not None
        ), "Frame context manager must be called within an ongoing trace."
        tc.frame_summary_stack.append(frame_summary)
        try:
            yield
        finally:
            tc.frame_summary_stack.pop()

    @staticmethod
    def set_current_loc(filename, lineno, frame_name):
        tc = TracingContext.get()
        assert (
            tc is not None
        ), "Loc context manager must be called within an ongoing trace."
        tc.loc_in_frame = traceback.FrameSummary(filename, lineno, frame_name)


"""
This function installs the passed in tracing context as a dynamic scoped global variable.

Calls to TracingContext.get() while not under a `with tracing()` context will return None.
"""


@contextmanager
def tracing(context: TracingContext):
    global _CURRENT_TRACING_CONTEXT
    old_context = _CURRENT_TRACING_CONTEXT
    _CURRENT_TRACING_CONTEXT = context
    try:
        yield _CURRENT_TRACING_CONTEXT
    finally:
        _CURRENT_TRACING_CONTEXT = old_context


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

    def make_guard(self, fn, is_volatile=False) -> Guard:
        if self.guard_source() is GuardSource.CONSTANT:
            raise NotImplementedError()
        return Guard(self.name(), self.guard_source(), fn, is_volatile)

    def is_nn_module(self) -> bool:
        return self.guard_source().is_nn_module()


# Subclasses can be found in torch/_dynamo/source.py
# Note - there is an odd exception to this invariant of a single base,
# see class SuperSource
@dataclasses.dataclass(frozen=True)
class ChainedSource(Source):
    base: Source


def detect_fake_mode(inputs: Any = None):
    """
    Attempts to "detect" what the current fake mode is.  If there is one ambiently
    available from TracingContext, we preferentially use that.  Otherwise, we
    heuristically detect the fake mode via the following sources, in order of
    priority:

        - Currently active fake mode on stack
        - Fake mode associated with passed in tensors (inputs does not
          have to be flattened)
    """
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    from torch.utils._pytree import tree_flatten

    fake_modes = []

    context = TracingContext.get()
    if context is not None:
        fake_mode = context.fake_mode
        if fake_mode is not None:
            fake_modes.append((fake_mode, "tracing context", 0))

    from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

    for i, m in enumerate(reversed(_get_current_dispatch_mode_stack())):
        if isinstance(m, FakeTensorMode):
            fake_modes.append((m, "active fake mode", i))

    flat_inputs, _ = tree_flatten(inputs)
    for i, flat_input in enumerate(flat_inputs):
        if isinstance(flat_input, FakeTensor):
            fake_modes.append((flat_input.fake_mode, "fake tensor input", i))

    if fake_modes:
        fake_mode, desc1, i1 = fake_modes[0]
        for m, desc2, i2 in fake_modes[1:]:
            assert (
                fake_mode is m
            ), f"fake mode ({fake_mode}) from {desc1} {i1} doesn't match mode ({m}) from {desc2} {i2}"
        return fake_mode
    else:
        return None
