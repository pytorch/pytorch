from __future__ import annotations

import contextlib
import dataclasses
import enum
import functools
import logging
import re
import threading
import traceback
import unittest.mock
import weakref
from abc import abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generic, NamedTuple, Optional, TYPE_CHECKING, TypeVar, Union

import torch
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._traceback import CapturedTraceback, format_frame
from torch.utils.weak import WeakTensorKeyDictionary


log = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator
    from types import CodeType

    import sympy

    from torch._dynamo.backends.distributed import DDPOptimizerContext
    from torch._dynamo.codegen import PyCodegen
    from torch._functorch._aot_autograd.schemas import ViewAndMutationMeta
    from torch._subclasses.fake_tensor import FakeTensorMode


"""
torch._guards is the definitional source of truth for general purpose guard structures.

An important thing to keep in mind here is the preservation of layering. There should be no dynamo notions,
and no guard installation notions here.
"""

COMPILE_ID_PATTERN = re.compile(r"^(?P<frame_id>\d+)/(?P<frame_compile_id>\d+)$")
CA_COMPILE_ID_PATTERN = re.compile(
    r"^!(?P<compiled_autograd_id>\d+)(?:/(?P<frame_id>\d+)/(?P<frame_compile_id>\d+))?$"
)

# [Note: Updating CompiledId]
#
# CompiledId represents a unique program-level identifier, and we want to keep that
# property as the codebase evolves. This property is relied on even outside of the pytorch
# repo, e.g. tlparse or other internal tooling. The in-memory format can be freely changed,
# as those dependencies only consume the string serialization.
#
# The string form should be:
# 1. Program-level uid: CompileId can uniquely identify a compiled graph.
# 2. Storage efficient: This object is logged in nearly every entry. We should elide symbols when possible.
# 3. Compact: The string form is directly displayed by some tools. Special symbols are okay.


@dataclass(frozen=True, kw_only=True, slots=True)
class CompileId:
    frame_id: int | None
    # This id is per-frame, and counts how many times we've compiled this
    # frame.  This could have been a global id but having this be per-frame
    # gives you a better intuitive sense for how many recompiles have occurred
    # so far.
    frame_compile_id: int | None

    # torch.compiling a compiled autograd graph
    compiled_autograd_id: int | None = None

    # TODO: consider also tracking the recompilation count
    # See Note: Updating CompileId

    def __str__(self) -> str:
        # NOTE: Keep this in sync with both from_string and the tlparse repo
        if self.compiled_autograd_id is not None:
            assert (self.frame_id is None) == (self.frame_compile_id is None)
            frame_str = ""
            if self.frame_id is not None:
                frame_str = f"/{self.frame_id}/{self.frame_compile_id}"

            return f"!{self.compiled_autograd_id}{frame_str}"
        else:
            assert self.frame_id is not None and self.frame_compile_id is not None
            return f"{self.frame_id}/{self.frame_compile_id}"

    @classmethod
    def from_string(cls, compile_id: Optional[str]) -> Optional[CompileId]:
        """
        Factory method that creates a CompileId from its string representation.
        Keep this in sync with the __str__ method.
        """
        if compile_id is None:
            return None
        try:
            for pattern in (COMPILE_ID_PATTERN, CA_COMPILE_ID_PATTERN):
                if match := pattern.match(compile_id):
                    groups = match.groupdict()
                    for k, v in groups.items():
                        if v is not None:
                            groups[k] = int(v)
                    return cls(**groups)  # type: ignore[arg-type]
            else:
                raise ValueError

        except Exception as e:
            raise ValueError(f"Invalid compile_id '{compile_id}'") from e


class TraceId(NamedTuple):
    compile_id: CompileId
    # This starts off as 0, and every time we restart analysis it goes
    # up by one
    attempt: int

    def __str__(self) -> str:
        # Keep this in sync with tlparse repo
        if self.attempt == 0:
            return str(self.compile_id)
        else:
            return f"{self.compile_id}_{self.attempt}"


class GuardSource(enum.Enum):
    LOCAL = 0
    GLOBAL = 1
    LOCAL_SPECIALIZED_NN_MODULE = 2
    GLOBAL_SPECIALIZED_NN_MODULE = 3
    CONSTANT = 4
    RANDOM_VALUE = 5
    SHAPE_ENV = 6
    LOCAL_FSDP_MODULE = 7
    GLOBAL_FSDP_MODULE = 8
    BACKWARD_STATE = 9
    EPHEMERAL = 10
    SYNTHETIC_LOCAL = 11
    LOCAL_UNSPECIALIZED_NN_MODULE = 12
    GLOBAL_UNSPECIALIZED_NN_MODULE = 13
    LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE = 14
    GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE = 15
    TEMP_LOCAL = 16

    def is_fsdp_module(self) -> bool:
        return self in (GuardSource.GLOBAL_FSDP_MODULE, GuardSource.LOCAL_FSDP_MODULE)

    def is_specialized_nn_module(self) -> bool:
        import torch._dynamo.config as config

        if config._unsafe_skip_fsdp_module_guards:
            return (
                self
                in (
                    GuardSource.GLOBAL_SPECIALIZED_NN_MODULE,
                    GuardSource.LOCAL_SPECIALIZED_NN_MODULE,
                )
                or self.is_fsdp_module()
            )
        return self in (
            GuardSource.GLOBAL_SPECIALIZED_NN_MODULE,
            GuardSource.LOCAL_SPECIALIZED_NN_MODULE,
        )

    def is_unspecialized_nn_module(self) -> bool:
        return self in (
            GuardSource.GLOBAL_UNSPECIALIZED_NN_MODULE,
            GuardSource.LOCAL_UNSPECIALIZED_NN_MODULE,
            GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
            GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
        )

    def is_unspecialized_builtin_nn_module(self) -> bool:
        return self in (
            GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
            GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
        )

    def is_local(self) -> bool:
        return self in (
            GuardSource.LOCAL,
            GuardSource.LOCAL_SPECIALIZED_NN_MODULE,
            GuardSource.LOCAL_FSDP_MODULE,
            GuardSource.LOCAL_UNSPECIALIZED_NN_MODULE,
            GuardSource.LOCAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
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


@dataclasses.dataclass(frozen=True)
class SLoc:
    framework_loc: traceback.FrameSummary | str | None
    maybe_user_loc: str | None

    def __str__(self) -> str:
        floc = (
            self.framework_loc
            if isinstance(self.framework_loc, str)
            else format_frame(self.framework_loc)
        )
        if self.maybe_user_loc is not None:
            return f"{self.maybe_user_loc} ({floc})"
        else:
            return f"({floc})"


class ShapeGuard(NamedTuple):
    expr: sympy.logic.boolalg.Boolean
    sloc: SLoc
    size_oblivious: bool


@dataclasses.dataclass(slots=True)
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
    guard_types: Optional[list[str]] = None
    code_list: Optional[list[str]] = None
    obj_weakref: Optional[object] = None
    guarded_class_weakref: Optional[weakref.ReferenceType[Any]] = None

    stack: Optional[CapturedTraceback] = None
    user_stack: Optional[traceback.StackSummary] = None
    _hash: Optional[int] = None
    _unserializable: bool = False

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((self.name, self.source, id(self.create_fn)))
        return self._hash

    def sort_key(self) -> tuple[bool, int, int, str, int]:
        # Put the duplicate input guards at the end. The duplicate guards have
        # two sources while guard.name only considers one source.

        is_duplicate_input = (
            isinstance(self.create_fn, functools.partial)
            and self.create_fn.func is torch._dynamo.guards.GuardBuilder.DUPLICATE_INPUT
        )
        return (
            is_duplicate_input,
            self.source.value if self.source else -1,
            len(self.name),
            self.name,
            self.inner_create_fn().__code__.co_firstlineno,
        )

    def __lt__(self, other: Guard) -> bool:
        return self.sort_key() < other.sort_key()

    def inner_create_fn(self) -> Callable[[GuardBuilderBase, Guard], Any]:
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
    def weakref_to_str(obj_weakref: object) -> str:
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

    def __repr__(self) -> str:
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

    def __str__(self) -> str:
        output = f"Name: {repr(self.name)}\n"
        source = self.source.name.lower() if self.source else ""
        output += f"    Source: {source}\n"
        output += f"    Create Function: {self.inner_create_fn().__name__}\n"
        output += f"    Guard Types: {self.guard_types}\n"
        output += f"    Code List: {self.code_list}\n"
        output += f"    Object Weakref: {self.weakref_to_str(self.obj_weakref)}\n"
        output += f"    Guarded Class Weakref: {self.guarded_class_weakref}\n"
        return output

    def create(self, builder: GuardBuilderBase) -> Any:
        try:
            return self.create_fn(builder, self)
        except Exception:
            log.exception("Error while creating guard:\n%s", str(self).rstrip())
            if self.stack:
                log.error("Created at:\n%s", "".join(self.stack.format()[-4:]).rstrip())
            raise

    def is_specialized_nn_module(self) -> bool:
        return self.source.is_specialized_nn_module()

    def is_fsdp_module(self) -> bool:
        return self.source.is_fsdp_module()

    def is_local(self) -> bool:
        return self.source.is_local()

    def create_fn_name(self) -> str:
        if isinstance(self.create_fn, functools.partial):
            create_fn = self.create_fn.func  # type: ignore[attr-defined]
        else:
            create_fn = self.create_fn
        return create_fn.__name__

    def set_export_info(
        self,
        guard_type: str,
        guarded_class: Optional[weakref.ReferenceType[Any]],
        code_list: list[str],
        obj_weakref: object,
    ) -> None:
        if not self.guard_types:
            self.guard_types = []

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

        # Some objects are ephemeral, e.g., list[slice(1, 2)]. If we have
        # multiple guards on the same object, the weakref can die between the
        # invocation of set_export_info calls. So a dead weakref is also
        # acceptable.
        assert (
            self.obj_weakref in (obj_weakref, None)
            or callable(self.obj_weakref)
            and self.obj_weakref() is None
        ), "Guarded object must be identical, None or ephemeral (dead weakref)"
        self.obj_weakref = obj_weakref


T = TypeVar("T")

"""
Parent structure for guard env expressions.
A GuardEnvExpr can have any subtype.
Note: All subtypes must be handled exhaustively in
torch._dynamo.guards._parse_guard_env_guards to avoid a RuntimeError.
"""


@dataclasses.dataclass(frozen=True)
class GuardEnvExpr:
    pass


"""
A class representing a pair of duplicate inputs.
input_pos_a and input_pos_b are input positions we have deduped.
"""


@dataclasses.dataclass(frozen=True)
class DuplicateInputs(GuardEnvExpr):
    input_source_a: Source
    input_source_b: Source

    def __post_init__(self) -> None:
        assert self.input_source_a != self.input_source_b


"""
A class representing storage overlap relations among inputs that aliases the same storage.

Given that a set of tensors alias the same storage, this guard checks whether they actually
have overlapping storages.

While non_overlapping_sources represent input tensors that definitely don't have any storage
overlapping with any other input, overlapping_sources represent tensors that either:

1. Do overlap some other input tensor
2. Might not overlap some other input tensor, but we are not sure
"""


@dataclasses.dataclass(frozen=True)
class StorageOverlap(GuardEnvExpr):
    overlapping_sources: list[Source]
    non_overlapping_sources: list[Source]


"""
Checkpointable is an interface for driving state snapshotting, left purposely vague for now.

copy_graphstate() -> T, a somewhat legacy name, is expected to emit a snapshot of any type that
can also be taken in at restore_graphstate(T) calls.

When to snapshot, is, at the moment, an implementation detail of upstream callers. Checkpointable
does not provide any guarantees around consistency, idempotency, or safety of calling its APIs, yet.

In the future, it will have a closer coupling to a generic Checkpoint management system.
"""


class Checkpointable(Generic[T]):
    @abstractmethod
    def copy_graphstate(self) -> T: ...

    @abstractmethod
    def restore_graphstate(self, state: T) -> None: ...


class GuardsCheckpointState:
    """
    The GuardCheckpointState - it is the T of Checkpointable[T] for GuardsContext
    """

    dynamo_guards: set[Guard] = set()

    def __init__(self, dynamo_guards: set[Guard]) -> None:
        self.dynamo_guards = dynamo_guards

    def diff(self, other: GuardsCheckpointState) -> Optional[set[Guard]]:
        """
        Produces a delta against another GuardsCheckpointState.

        Returns None if no delta is found, otherwise, return a set() of mismatched
        Guard type objects.
        """
        r = self.dynamo_guards.difference(other.dynamo_guards)
        if len(r) == 0:
            return None
        return r

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GuardsCheckpointState):
            return False
        return self.diff(other) is None


class ModuleContextCheckpointState:
    nn_modules: dict[str, torch.nn.Module] = {}

    def __init__(self, nn_modules: dict[str, torch.nn.Module]) -> None:
        self.nn_modules = nn_modules

    def diff(self, other: ModuleContextCheckpointState) -> Optional[set[str]]:
        """
        Produces a delta against another ModuleContextCheckpointState.

        Returns None if no delta is found, otherwise, return a set() of mismatched
        module key names.
        """
        r = set(self.nn_modules.keys()).difference(set(other.nn_modules.keys()))
        if len(r) == 0:
            return None
        return r

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModuleContextCheckpointState):
            return False
        return self.diff(other) is None


class ModuleContext(Checkpointable[ModuleContextCheckpointState]):
    def __init__(self) -> None:
        self.nn_modules: dict[str, Any] = {}

    def copy_graphstate(self) -> ModuleContextCheckpointState:
        return ModuleContextCheckpointState(dict(self.nn_modules))

    def restore_graphstate(self, state: ModuleContextCheckpointState) -> None:
        assert isinstance(state, ModuleContextCheckpointState)
        self.nn_modules = state.nn_modules


class GlobalContextCheckpointState:
    global_state: dict[str, tuple[Callable, Any]] = {}

    def __init__(self, global_states: dict[str, tuple[Callable, Any]]) -> None:
        self.global_state = global_states

    def diff(self, other: GlobalContextCheckpointState) -> Optional[set[str]]:
        """
        Produces a delta against another GlobalContextCheckpointState.

        Returns None if no delta is found, otherwise, return a set() of mismatched
        global key names.
        """
        r = set(self.global_state.keys()).difference(set(other.global_state.keys()))
        if len(r) == 0:
            return None
        return r

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GlobalContextCheckpointState):
            return False
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

    def __init__(self) -> None:
        self.global_state: dict[str, tuple[Callable, Any]] = {}

    def copy_graphstate(self) -> GlobalContextCheckpointState:
        return GlobalContextCheckpointState(self.global_state)

    def restore_graphstate(self, state: GlobalContextCheckpointState) -> None:
        assert isinstance(state, GlobalContextCheckpointState)
        self.global_state = state.global_state
        assert (
            len(self.global_state) == len(self._supported_global_states)
            and set(self.global_state.keys()) == self._supported_global_states
        ), "Global state mismatch"
        for func, args in self.global_state.values():
            func(args)


# Like a Set[Guard] but will record the user stack on all guards at the
# time they were installed at their destination
class GuardsSet:
    def __init__(self, inner: Optional[set[Guard]] = None) -> None:
        if inner is None:
            inner = set()
        self.inner = inner

    def __iter__(self) -> Iterator[Guard]:
        return iter(self.inner)

    def __len__(self) -> int:
        return len(self.inner)

    # Subtraction along with bool is typically used to determine the delta of
    # added guards between checkpoints for higher order ops
    def __sub__(self, other: GuardsSet) -> GuardsSet:
        return GuardsSet(self.inner - other.inner)

    def __bool__(self) -> bool:
        return bool(self.inner)

    def add(
        self, guard: Guard, *, collect_debug_stack: bool = True, skip: int = 0
    ) -> None:
        if guard in self.inner:
            return
        if collect_debug_stack:
            if guard.stack is None:
                guard.stack = CapturedTraceback.extract(skip=1 + skip)
        if guard.user_stack is None:
            guard.user_stack = TracingContext.extract_stack()
        self.inner.add(guard)

    def update(self, *others: set[Guard]) -> None:
        for o in others:
            for g in o:
                self.add(g, skip=1)

    def remove_guards_with_source(self, source: Source) -> None:
        """Delete all guards that contains a given source"""
        from ._dynamo.source import is_from_source

        self.inner = {
            g for g in self.inner if not is_from_source(g.originating_source, source)
        }


"""
A GuardsContext is a checkpointable representation of all the guards in the current tracing
context. It's lifecycle is bound 1:1 to the tracing context, and it should never be instantiated
directly outside of it. For passing around internal state representations of this object,
prefer to extract them with copy_graphstate to produce a GuardsCheckpointState.
"""


class GuardsContext(Checkpointable[GuardsCheckpointState]):
    def __init__(self) -> None:
        self.dynamo_guards: GuardsSet = GuardsSet()
        self.aotautograd_guards: list[GuardEnvExpr] = []

    def copy_graphstate(self) -> GuardsCheckpointState:
        return GuardsCheckpointState(set(self.dynamo_guards.inner))

    def restore_graphstate(self, state: GuardsCheckpointState) -> None:
        # NB: "steals" the passed in state
        assert isinstance(state, GuardsCheckpointState)
        self.dynamo_guards = GuardsSet(state.dynamo_guards)


class HopSubgraphCache:
    @abstractmethod
    def add_dynamo_installed_submodule(self, fn_id: int, identifier: str) -> None: ...

    @abstractmethod
    def get_dynamo_installed_submodules(self, fn_id: int) -> list[str]: ...

    @abstractmethod
    def add_autograd_key_entry(self, identifier: str, key: Callable) -> None: ...

    @abstractmethod
    def get_autograd_key_entry(self, identifier: str) -> Optional[Callable]: ...

    @abstractmethod
    def add_proxy_dispatch_entry(self, identifier: str, key: Callable) -> None: ...

    @abstractmethod
    def get_proxy_dispatch_entry(self, identifier: str) -> Optional[Callable]: ...

    @abstractmethod
    def add_lazy_bwd_entry(
        self,
        identifier: str,
        tangent_metadata: tuple[object],
        gmod: torch.fx.GraphModule,
    ) -> int: ...

    @abstractmethod
    def get_lazy_bwd_entry(
        self, identifier: str, tangent_metadata: tuple[object]
    ) -> tuple[Optional[torch.fx.GraphModule], Optional[int]]: ...


class InvokeSubgraphCache(HopSubgraphCache):
    def __init__(self) -> None:
        self.autograd_cache: dict[str, Callable] = {}
        self.proxy_dispatch_cache: dict[str, Callable] = {}
        self.dynamo_installed_submodules: dict[int, list[str]] = defaultdict(list)
        self.lazy_bwd_cache: dict[
            str, dict[tuple[object], tuple[torch.fx.GraphModule, int]]
        ] = defaultdict(dict)
        self.effects_cache: dict[
            str, set
        ] = {}  # Maps identifier -> set of effect types

    def add_dynamo_installed_submodule(self, fn_id: int, identifier: str) -> None:
        self.dynamo_installed_submodules[fn_id].append(identifier)

    def get_dynamo_installed_submodules(self, fn_id: int) -> list[str]:
        return self.dynamo_installed_submodules.get(fn_id, [])

    def add_autograd_key_entry(self, identifier: str, key: Callable) -> None:
        self.autograd_cache[identifier] = key

    def get_autograd_key_entry(self, identifier: str) -> Optional[Callable]:
        return self.autograd_cache.get(identifier, None)

    def add_proxy_dispatch_entry(self, identifier: str, key: Callable) -> None:
        self.proxy_dispatch_cache[identifier] = key

    def get_proxy_dispatch_entry(self, identifier: str) -> Optional[Callable]:
        return self.proxy_dispatch_cache.get(identifier, None)

    def add_lazy_bwd_entry(
        self,
        identifier: str,
        tangent_metadata: tuple[object],
        gmod: torch.fx.GraphModule,
    ) -> int:
        # Save the number of existing graph modules in the dictionary to get the suffix
        num_gmods = len(self.lazy_bwd_cache[identifier])
        self.lazy_bwd_cache[identifier][tangent_metadata] = (gmod, num_gmods)
        return num_gmods

    def get_lazy_bwd_entry(
        self, identifier: str, tangent_metadata: tuple[object]
    ) -> tuple[Optional[torch.fx.GraphModule], Optional[int]]:
        if identifier not in self.lazy_bwd_cache:
            return (None, None)

        return self.lazy_bwd_cache[identifier].get(tangent_metadata, (None, None))

    def add_effects(self, identifier: str, effects: set) -> None:
        """Store the effect types for a given invoke_subgraph identifier."""
        if prev_effects := self.effects_cache.get(identifier, None):
            assert effects == prev_effects, (
                "Different number of effects were found for invoke_subgraph "
                f"call with identifier {identifier}. \n"
                f"Previously we had the following effects: {prev_effects}.\n"
                f"But now we have: {effects}."
            )
        self.effects_cache[identifier] = effects

    def get_effects(self, identifier: str) -> Optional[set]:
        """Retrieve the effect types for a given invoke_subgraph identifier."""
        return self.effects_cache.get(identifier, None)


class HopDispatchSetCache:
    def __init__(self) -> None:
        # Delayed import to avoid circular dependency
        from torch._higher_order_ops.invoke_subgraph import invoke_subgraph

        self.hop_cache_map = {invoke_subgraph: InvokeSubgraphCache()}

    def get_cache(self, op: torch._ops.HigherOrderOperator) -> HopSubgraphCache | None:
        if op not in self.hop_cache_map:
            return None
        return self.hop_cache_map[op]  # type: ignore[index]


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
    def get() -> CompileContext:
        assert _TLS.compile_context is not None
        return _TLS.compile_context

    @staticmethod
    def try_get() -> CompileContext | None:
        return getattr(_TLS, "compile_context", None)

    def __init__(self, compile_id: Optional[CompileId]) -> None:
        assert compile_id is None or isinstance(compile_id, CompileId)
        self.compile_id: CompileId | None = compile_id
        self.attempt = 0
        # Verbose ShapeEnv guards produced.
        self.shape_env_guards: list[str] = []

    @staticmethod
    def current_compile_id() -> Optional[CompileId]:
        self = CompileContext.try_get()
        if self is None:
            return None
        return self.compile_id

    @staticmethod
    def current_trace_id() -> Optional[TraceId]:
        self = CompileContext.try_get()
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
    def try_get() -> TracingContext | None:
        return getattr(_TLS, "tracing_context", None)

    @staticmethod
    def get() -> TracingContext:
        if ctx := TracingContext.try_get():
            return ctx
        raise RuntimeError(
            "TracingContext.get() must be called within an ongoing trace."
        )

    def __init__(self, fake_mode: Optional[FakeTensorMode]) -> None:
        self.guards_context = GuardsContext()
        self.module_context = ModuleContext()
        self.global_context = GlobalContext()
        self.previously_inlined_functions: dict[Any, Any] = dict()
        self.previously_cleaned_instructions: dict[Any, Any] = dict()
        self.fake_mode: Optional[FakeTensorMode] = fake_mode
        self.frame_summary_stack: list[traceback.FrameSummary] = []
        # This is morally part of frame_summary_stack, but it is kept separate
        # for clarity.  As we process a frame, this variable gets updated
        # to keep track of what line we are in the function.  We make a
        # function call, this gets cleared and the frame location is pushed
        # to frame_summary_stack (prepping this variable for the inner frame's
        # progress)
        self.loc_in_frame: Optional[tuple[str, int, str]] = None
        # this is only set after aot_autograd
        self.fw_metadata: Optional[ViewAndMutationMeta] = None
        # this is only set when the DDPOptimizer is used
        self.ddp_optimizer_ctx: Optional[DDPOptimizerContext] = None
        # this is only set after aot_autograd
        self.aot_graph_name: Optional[list[str]] = None
        self.params_flat: Optional[list[Any]] = None
        self.params_flat_unwrap_subclasses: Optional[list[Any]] = None
        self.params_unwrapped_to_flat_index: Optional[list[Any]] = None
        # this is for extended return calling convention from backend
        # compiler to aot_autograd
        # Per output, what the compiler specified stride of the output is,
        # or None if no stride is known.  This is always the HINT, it
        # is never a SymInt (it would be better if it was a SymInt, but
        # I can't conveniently get this from Inductor atm.  Also, be
        # careful not to accidentally induce guards on the SymInt if
        # you ever do change this in aot_autograd.py; you should check
        # on permutations preferentially.)
        self.output_strides: list[tuple[int, ...] | None] | None = None
        # When this is True, whenever we encounter an int in Dynamo tracing,
        # we will (1) force unspec it and (2) force it as a size-like unbacked
        # integer.  This is currently used when processing certain lists of
        # ints that are known to be size-like and may have 0/1 entries that we
        # must not specialize on.
        self.force_unspec_int_unbacked_size_like = False
        # See note [Tensor Fakification and Symbol Caching]
        self.tensor_to_context = WeakTensorKeyDictionary()

        # If this true, Aot Autograd will return output Fake Tensors with appropriate
        # meta on the first invocation
        # see note: [Returning Fake Tensors on First AOT Autograd Call]
        self.fakify_first_call = False
        self.hop_dispatch_set_cache = HopDispatchSetCache()
        # list of code objects for inlined functions
        self.traced_code: list[CodeType] = []

    def clear(self) -> None:
        # Look at the note in output_graph.py in function `save_global_state`
        # for the context on clearing global context.
        self.global_context.global_state = {}
        self.previously_inlined_functions.clear()
        self.previously_cleaned_instructions.clear()

    @staticmethod
    @contextmanager
    def patch(**kwargs: Any) -> Generator[None, None, None]:
        prior = {}
        ctx = TracingContext.get()

        for key in kwargs:
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
    def extract_stack() -> traceback.StackSummary:
        self = TracingContext.try_get()
        if self is None:
            return traceback.StackSummary()
        stack = self.frame_summary_stack
        if self.loc_in_frame is not None:
            stack = stack + [self._populate_loc_in_frame_summary()]
        return traceback.StackSummary.from_list(stack)

    def _populate_loc_in_frame_summary(self) -> traceback.FrameSummary:
        assert self.loc_in_frame is not None
        filename, lineno, frame_name = self.loc_in_frame
        return traceback.FrameSummary(filename, lineno, frame_name, lookup_line=False)

    # Call this when you want to call into some code that isn't necessarily
    # associated with the current frame state
    @staticmethod
    @contextlib.contextmanager
    def clear_frame() -> Generator[None, None, None]:
        tc = TracingContext.get()
        with (
            unittest.mock.patch.object(tc, "frame_summary_stack", []),
            unittest.mock.patch.object(tc, "loc_in_frame", None),
        ):
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
    def current_frame(
        frame_summary: Optional[traceback.FrameSummary],
    ) -> Generator[None, None, None]:
        # frame_summary can be None to solely take advantage of real_stack
        # attachment to thrown exceptions
        tc = TracingContext.get()
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
    def report_output_strides() -> Generator[
        Optional[list[Optional[tuple[int, ...]]]], None, None
    ]:
        tc = TracingContext.try_get()
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
    def set_current_loc(filename: str, lineno: int, frame_name: str) -> None:
        # Save the current location in the frame. Lazily generate the
        # framesummary.
        TracingContext.get().loc_in_frame = (filename, lineno, frame_name)

    @staticmethod
    def get_traced_code() -> Optional[list[CodeType]]:
        tc = TracingContext.try_get()
        if tc is None:
            return None
        return tc.traced_code


@contextmanager
def compile_context(
    context: Optional[CompileContext],
) -> Generator[Optional[CompileContext], None, None]:
    old_context = getattr(_TLS, "compile_context", None)
    _TLS.compile_context = context
    try:
        yield context
    finally:
        _TLS.compile_context = old_context


@contextmanager
def tracing(
    context: Optional[TracingContext],
) -> Generator[Optional[TracingContext], None, None]:
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
    def is_dict_key(self) -> bool:
        return False

    def is_ephemeral(self) -> bool:
        return False

    def reconstruct(self, codegen: PyCodegen) -> None:
        raise NotImplementedError

    def guard_source(self) -> GuardSource:
        raise NotImplementedError

    def name(self) -> str:
        raise NotImplementedError

    def make_guard(self, fn: Callable[..., Any]) -> Guard:
        if self.guard_source() is GuardSource.CONSTANT:
            raise NotImplementedError
        return Guard(self, fn)

    def is_specialized_nn_module(self) -> bool:
        return self.guard_source().is_specialized_nn_module()

    def subguards_allowed(self) -> bool:
        """True if you can guard on attributes of this"""
        return self.guard_source() != GuardSource.SYNTHETIC_LOCAL


# Subclasses can be found in torch/_dynamo/source.py
@dataclasses.dataclass(frozen=True)
class ChainedSource(Source):
    base: Source

    def is_dict_key(self) -> bool:
        # Recurse until you either hit a ConstDictKey or a Source
        return self.base.is_dict_key()

    def is_ephemeral(self) -> bool:
        return self.base.is_ephemeral()

    def get_base(self) -> Source:
        current: Source = self
        while isinstance(current, ChainedSource):
            current = current.base
        return current


def detect_fake_mode(inputs: Any = None) -> Optional[FakeTensorMode]:
    """
    Attempts to "detect" what the current fake mode is.  If there is one ambiently
    available from TracingContext, we preferentially use that.  Otherwise, we
    heuristically detect the fake mode via the following sources, in order of
    priority:

        - Currently active fake mode on stack
        - Fake mode associated with passed in tensors (inputs does not
          have to be flattened)
    """
    from torch._subclasses.fake_tensor import (
        FakeTensor,
        FakeTensorMode,
        get_plain_tensors,
    )

    fake_modes = []

    if context := TracingContext.try_get():
        fake_mode = context.fake_mode
        if fake_mode is not None:
            fake_modes.append((fake_mode, "tracing context", 0))

    from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

    for i, m in enumerate(reversed(_get_current_dispatch_mode_stack())):
        if isinstance(m, FakeTensorMode):
            # pyrefly: ignore [bad-argument-type]
            fake_modes.append((m, "active fake mode", i))

    flat_inputs = pytree.tree_leaves(inputs)
    for i, flat_input in enumerate(flat_inputs):
        if isinstance(flat_input, FakeTensor):
            # pyrefly: ignore [bad-argument-type]
            fake_modes.append((flat_input.fake_mode, "fake tensor input", i))
        if is_traceable_wrapper_subclass(flat_input):
            out: list[Union[torch.Tensor, int, torch.SymInt]] = []
            get_plain_tensors(flat_input, out=out)  # type: ignore[arg-type]
            fake_tensors: list[FakeTensor] = [
                x for x in out if isinstance(x, FakeTensor)
            ]
            fake_modes.extend(
                # pyrefly: ignore [bad-argument-type]
                [
                    (tensor.fake_mode, f"subclass input {i}", ix)
                    for ix, tensor in enumerate(fake_tensors)
                ]
            )

    if fake_modes:
        fake_mode, desc1, i1 = fake_modes[0]
        for m, desc2, i2 in fake_modes[1:]:
            assert fake_mode is m, (
                f"fake mode ({fake_mode}) from {desc1} {i1} doesn't match mode ({m}) from {desc2} {i2}\n\n"
                # pyrefly: ignore [missing-attribute]
                f"fake mode from {desc1} {i1} allocated at:\n{fake_mode.stack}\n"
                # pyrefly: ignore [missing-attribute]
                f"fake mode from {desc2} {i2} allocated at:\n{m.stack}"
            )
        # pyrefly: ignore [bad-return]
        return fake_mode
    else:
        return None


def active_fake_mode() -> Optional[FakeTensorMode]:
    """
    Inspects the dispatch mode stack for an active fake mode and returns it.
    Returns None if no fake mode is active.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

    for _, m in enumerate(reversed(_get_current_dispatch_mode_stack())):
        if isinstance(m, FakeTensorMode):
            return m

    return None
