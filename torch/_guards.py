from __future__ import annotations

import contextlib

import dataclasses
import enum
import functools
import logging
import threading
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
from torch.utils._traceback import CapturedTraceback

log = logging.getLogger(__name__)

import sympy

"""
torch._guards is the definitional source of truth for general purpose guard structures.

An important thing to keep in mind here is the preservation of layering. There should be no dynamo notions,
and no guard installation notions here.
"""


class CompileId(NamedTuple):
    frame_id: int
    # This id is per-frame, and counts how many times we've compiled this
    # frame.  This could have been a global id but having this be per-frame
    # gives you a better intuitive sense for how many recompiles have occurred
    # so far.
    frame_compile_id: int
    # TODO: consider also tracking the recompilation count

    def __str__(self):
        return f"{self.frame_id}/{self.frame_compile_id}"


class TraceId(NamedTuple):
    compile_id: CompileId
    # This starts off as 0, and every time we restart analysis it goes
    # up by one
    attempt: int

    def __str__(self):
        if self.attempt == 0:
            return str(self.compile_id)
        else:
            return f"{self.compile_id}_{self.attempt}"


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
    is_volatile: bool = False

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
    input_source_a: Source
    input_source_b: Source

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
            guard.user_stack = TracingContext.extract_stack()
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


_TLS = threading.local()

"""
TracingContext is the source of truth for all currently accumulated information
needed to trace. Its lifecycle is kept 1:1 when using TorchDynamo, but other systems
are open to managing their own TracingContext with that in mind.

The purpose of TracingContext is not to be a dumping ground, or god object, but rather to avoid
having to plumb complex subsystems across multiple verticals.

Ex: A common example is guard accumulation between dynamo, shape_env, aot_autograd, and inductor.
Accessing the current tracing context via
TracingContext.get() allows users to accumulate their own guards for processing, without needing to know how
to plumb objects back up to where frame interpretation happened.

Note that you can end up with multiple TracingContext for a single compilation
of a frame, as we reset the TracingContext whenever we restart analysis.
CompileContext is a more overarching context that encompasses multiple restarts.
"""


class CompileContext:
    @staticmethod
    def get() -> Optional[CompileContext]:
        return getattr(_TLS, "compile_context", None)

    def __init__(self, compile_id):
        assert compile_id is None or isinstance(compile_id, CompileId)
        self.compile_id: Optional[CompileId] = compile_id
        self.attempt = 0

    @staticmethod
    def current_compile_id():
        self = CompileContext.get()
        if self is None:
            return None
        return self.compile_id

    @staticmethod
    def current_trace_id():
        self = CompileContext.get()
        if self is None:
            return None
        if self.compile_id is None:
            return None
        return TraceId(self.compile_id, self.attempt)


class TracingContext:
    """
    Provides the currently installed TracingContext, or None.

    Note that it is a staticmethod, and invocations outside of `with tracing()` (see below), are valid but
    will return None.
    """

    @staticmethod
    def get() -> Optional[TracingContext]:
        return getattr(_TLS, "tracing_context", None)

    def __init__(self, fake_mode):
        self.guards_context = GuardsContext()
        self.module_context = ModuleContext()
        self.global_context = GlobalContext()
        self.fake_mode = fake_mode
        self.frame_summary_stack = []
        # This is morally part of frame_summary_stack, but it is kept separate
        # for clarity.  As we process a frame, this variable gets updated
        # to keep track of what line we are in the function.  We make a
        # function call, this gets cleared and the frame location is pushed
        # to frame_summary_stack (prepping this variable for the inner frame's
        # progress)
        self.loc_in_frame = None
        # this is only set after aot_autograd
        self.fw_metadata = None
        self.params_flat = None
        # this is for extended return calling convention from backend
        # compiler to aot_autograd
        # Per output, what the compiler specified stride of the output is,
        # or None if no stride is known.  This is always the HINT, it
        # is never a SymInt (it would be better if it was a SymInt, but
        # I can't conveniently get this from Inductor atm.  Also, be
        # careful not to accidentally induce guards on the SymInt if
        # you ever do change this in aot_autograd.py; you should check
        # on permutations preferentially.)
        self.output_strides: Optional[List[Optional[List[int]]]] = None
        # When this is True, whenever we encounter an int in Dynamo tracing,
        # we will (1) force unspec it and (2) force it as a size-like unbacked
        # integer.  This is currently used when processing certain lists of
        # ints that are known to be size-like and may have 0/1 entries that we
        # must not specialize on.
        self.force_unspec_int_unbacked_size_like = False

    @staticmethod
    @contextmanager
    def patch(**kwargs):
        prior = {}
        ctx = TracingContext.get()
        assert ctx is not None

        for key in kwargs.keys():
            # KeyError on invalid entry
            prior[key] = getattr(ctx, key)
        for key, val in kwargs.items():
            setattr(ctx, key, val)
        try:
            yield
        finally:
            for key, val in prior.items():
                setattr(ctx, key, val)

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
            try:
                yield
            except Exception as e:
                # Prevent real_stack from getting attached
                #
                # The invariant is that if an Exception as real_stack, we've
                # appropriately attached a user stack and we no longer need to
                # attach anything. Because we cannot conveniently interpose
                # when an exception is thrown, we instead interpose everywhere
                # we set what the user stack is set (using the context
                # manager). However, our compiler stack does "tail calls"
                # (when it calls into user compiler), at which point the
                # parent exception frames would incorrectly attach an
                # incorrect frame.
                #
                # However, if, somehow, someone raised an exception with this
                # scope that had a stack (for example, because they are
                # restoring the user stack state appropriately as they process
                # node by node), we should respect it. Thus, we cannot
                # unconditionally set None.
                if not hasattr(e, "real_stack"):
                    e.real_stack = None  # type: ignore[attr-defined]
                raise

    @staticmethod
    @contextlib.contextmanager
    def current_frame(frame_summary):
        # frame_summary can be None to solely take advantage of real_stack
        # attachment to thrown exceptions
        tc = TracingContext.get()
        assert (
            tc is not None
        ), "Frame context manager must be called within an ongoing trace."
        if frame_summary is not None:
            tc.frame_summary_stack.append(frame_summary)
        old = tc.loc_in_frame
        tc.loc_in_frame = None
        try:
            yield
        except Exception as e:
            if not hasattr(e, "real_stack"):
                e.real_stack = tc.extract_stack()  # type: ignore[attr-defined]
            raise
        finally:
            if frame_summary is not None:
                tc.frame_summary_stack.pop()
            tc.loc_in_frame = old

    @staticmethod
    @contextlib.contextmanager
    def report_output_strides():
        tc = TracingContext.get()
        if tc is None:
            yield None
            return
        old_output_strides = tc.output_strides
        tc.output_strides = []
        try:
            yield tc.output_strides
        finally:
            tc.output_strides = old_output_strides

    @staticmethod
    def set_current_loc(filename, lineno, frame_name):
        tc = TracingContext.get()
        assert (
            tc is not None
        ), "Loc context manager must be called within an ongoing trace."
        tc.loc_in_frame = traceback.FrameSummary(filename, lineno, frame_name)


@contextmanager
def compile_context(context: CompileContext):
    old_context = getattr(_TLS, "compile_context", None)
    _TLS.compile_context = context
    try:
        yield context
    finally:
        _TLS.compile_context = old_context


@contextmanager
def tracing(context: TracingContext):
    """
    This function installs the passed in tracing context as a dynamic scoped
    global variable.

    Calls to TracingContext.get() while not under a `with tracing()` context
    will return None.
    """
    old_context = getattr(_TLS, "tracing_context", None)
    _TLS.tracing_context = context
    try:
        yield context
    except Exception as e:
        if not hasattr(e, "real_stack") and context is not None:
            e.real_stack = context.extract_stack()  # type: ignore[attr-defined]
        raise
    finally:
        if (
            context is not None
            and context.fake_mode is not None
            and context.fake_mode.shape_env is not None
        ):
            context.fake_mode.shape_env.cleanup()
        _TLS.tracing_context = old_context


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
        return Guard(self, fn, is_volatile)

    def is_nn_module(self) -> bool:
        return self.guard_source().is_nn_module()


# Subclasses can be found in torch/_dynamo/source.py
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
            assert fake_mode is m, (
                f"fake mode ({fake_mode}) from {desc1} {i1} doesn't match mode ({m}) from {desc2} {i2}\n\n"
                f"fake mode from {desc1} {i1} allocated at:\n{fake_mode.stack}\n"
                f"fake mode from {desc2} {i2} allocated at:\n{m.stack}"
            )
        return fake_mode
    else:
        return None
