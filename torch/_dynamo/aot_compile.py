import builtins
import dataclasses
import enum
import importlib
import inspect
import io
import logging
import os
import pickle
import re
import tempfile
import types
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, ExitStack, nullcontext
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

import torch
import torch.fx
from torch._dynamo.convert_frame import GraphRuntimeEnv
from torch._dynamo.graph_utils import _collapse_device_types, _graph_device_types
from torch._dynamo.package import FunctionPicklerBase, SerializedCode, SystemInfo

from . import convert_frame
from .aot_compile_types import (
    BundledAOTAutogradSerializableCallable,
    SerializableCallable,
)
from .hooks import Hooks


if TYPE_CHECKING:
    from .guards import GuardManagerWrapper
    from .package import SourceInfo


log = logging.getLogger(__name__)

_EXTERNAL_DATA_HINT = (
    "Mark the value(s) as external data by using `external_data={'key': ...}`."
)


# A guard failure that is exactly a missing top-level global: the verbose code
# part a guard tree reports for one ("KeyError on G['CONFIG']"). A trailing
# subscript ("KeyError on G['CONFIG']['scale']") means the global itself
# resolved and only a key inside it is absent, so the advice to define the
# global would be wrong.
_MISSING_GLOBAL_RE = re.compile(r"KeyError on G\[[^\[\]]*\]")


def _names_a_missing_global(text: str) -> bool:
    # Matched whole, against one verbose code part: matching a substring of the
    # GuardDebugInfo string would also fire for the nested-key failure above.
    return _MISSING_GLOBAL_RE.fullmatch(text) is not None


class _GuardScope(enum.Enum):
    """Which dict the artifact's global guards resolve names against."""

    # Never serialized: the guards still hold the tracing process's globals.
    CAPTURED = "captured"
    # A live scope the load path re-rooted the guards at, e.g. a function
    # load's f_globals.
    SUPPLIED = "supplied"
    # Rebuilt from the serialized bytecode, so it holds only the globals the
    # graph lifted -- no name this process defines can reach it.
    RECONSTRUCTED = "reconstructed"


def bind_locals(
    signature: inspect.Signature, *args: Any, **kwargs: Any
) -> dict[str, Any]:
    bound_arguments = signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()
    return bound_arguments.arguments


@dataclass
class CompileArtifacts:
    signature: inspect.Signature
    guard_manager: Optional["GuardManagerWrapper"]
    guards_state: bytes
    backend_id: str
    compiled_fn: SerializableCallable
    original_code: types.CodeType
    runtime_env: GraphRuntimeEnv
    source_info: "SourceInfo"
    device_type: str
    backend_name: str
    system_info: SystemInfo = dataclasses.field(default_factory=SystemInfo.current)

    def check_compatibility(self) -> None:
        current_system = SystemInfo.current()
        current_system.check_compatibility(self.system_info, self.device_type)


@dataclasses.dataclass
class _ProbeState:
    """Shared by an AOTCompilePickler and the throwaway probe picklers its
    _dumps_cleanly spawns, so the whole probe tree sees one memo."""

    # id(value) -> picklable; without the memo a probe tree is exponential.
    cache: dict[int, bool] = dataclasses.field(default_factory=dict)
    inflight: set[int] = dataclasses.field(default_factory=set)
    # id(value) -> the unmarked nn.Modules the probe reached inside it, so the
    # warning can name the actual reason and the offending modules.
    unmarked_modules: dict[int, list[Any]] = dataclasses.field(default_factory=dict)
    # id(value) -> the exception type that failed its probe, so the warning
    # tells a value that does not pickle from a reducer bug (an AttributeError
    # out of this file's own machinery, a RecursionError).
    failures: dict[int, str] = dataclasses.field(default_factory=dict)
    # id(function) -> its picklable __dict__ entries; a function that closes
    # over itself is reduced twice, and the second pass must not re-probe or
    # re-warn.
    attributes: dict[int, dict[str, Any]] = dataclasses.field(default_factory=dict)
    # id(function) -> its kept __doc__ (None when pruned), for the same warn-once
    # reason.
    docs: dict[int, Any] = dataclasses.field(default_factory=dict)
    # id(function) -> its kept annotations, for the same warn-once reason.
    annotations: dict[int, dict[str, Any]] = dataclasses.field(default_factory=dict)
    type_params: dict[int, tuple[Any, ...] | None] = dataclasses.field(
        default_factory=dict
    )
    # Whether a probe short-circuited on an in-flight id; such a verdict is
    # not cached as final but parked (as unpicklable) for the rest of the
    # probe tree.
    leaned: bool = False
    parked: set[int] = dataclasses.field(default_factory=set)


class AOTCompilePickler(FunctionPicklerBase):
    def __init__(
        self,
        external_data: dict[str, object],
        buf: io.BytesIO,
        *,
        probe_state: _ProbeState | None = None,
    ) -> None:
        super().__init__(buf)
        self.external_data = external_data
        self.id_map: dict[int, str] = {
            id(value): key for key, value in external_data.items()
        }
        self.errors = {}
        # A probe pickler shares its parent's state; only the real dump reports
        # what it drops, since a probe's verdict may not be final.
        self._probing = probe_state is not None
        self._probe_state = probe_state or _ProbeState()

    def persistent_id(self, obj: object) -> int | str | None:
        if id(obj) in self.id_map:
            return self.id_map[id(obj)]
        elif isinstance(obj, torch.nn.Module):
            self.errors[id(obj)] = obj
            return id(obj)
        else:
            return None

    # pyrefly: ignore [bad-override]
    def reducer_override(self, obj: Any) -> Any:
        if isinstance(obj, types.CellType):
            return self._reduce_cell(obj)
        elif inspect.iscode(obj):
            return type(self)._unpickle_code, (SerializedCode.from_code_object(obj),)
        elif inspect.ismodule(obj):
            return type(self)._unpickle_python_module, (obj.__name__,)
        elif inspect.ismethod(obj):
            receiver = obj.__self__
            # A receiver in external_data is served by persistent_id, so it is
            # the LIVE object at load and pickle's default getattr(receiver,
            # name) resolves on it; the shared reducer's __getattr__ gate would
            # instead pickle __func__ (every nn.Module defines __getattr__),
            # rebuilding a local subclass's method by value and failing on its
            # __class__ cell. An unmarked nn.Module is recorded in errors and
            # fails serialize() anyway, so for it this only keeps the dump going.
            live = id(receiver) in self.id_map or isinstance(receiver, torch.nn.Module)
            reduced = self._reduce_bound_method(obj, receiver_is_live=live)
            if reduced is not None:
                return reduced
        elif inspect.isfunction(obj) and not self._fqn_resolves(obj):
            # The runtime env has to RUN this function, so what a call needs
            # (defaults, keyword defaults, closure) is carried verbatim, while
            # annotations, type params, __dict__ entries and __doc__ are pruned
            # per value: the runtime assigns those back and never forces a pruned
            # one, so a value this pickler cannot serialize (a <locals>
            # annotation class, a PEP 695 function-scoped TypeVar, or a __dict__
            # entry like the __wrapped__ functools.wraps stashes, which can drag
            # an unrelated lock/Module in) is dropped rather than left to fail
            # the whole dump. Known limitation: the top-level function's own
            # annotations ride on CompileArtifacts.signature, which serialize()
            # dumps unpruned, so this only protects the nested functions reached
            # here.
            return self._reduce_function(
                obj,
                defaults=obj.__defaults__,
                kwdefaults=obj.__kwdefaults__,
                closure=obj.__closure__,
                attributes=self._pickleable_attributes(obj),
                annotations=self._pickleable_annotations(obj),
                doc=self._pickleable_doc(obj),
                type_params=self._pickleable_type_params(obj),
                globals_snapshot=None,
            )

        return NotImplemented

    def _warn_dropped(self, obj: Any, slot: str, value: Any) -> None:
        # The body may read a pruned attribute (`with helper.lock:`), so the
        # drop is a warning that names the fix, not a silent debug line; the
        # user can hand the object over as external data and it is kept. It
        # names the exception type too: a reducer bug then reads as one in a
        # bug report instead of as the user's value not pickling. The
        # function is named by its code object: functools.wraps overwrites
        # __qualname__ with the wrappee's, which would make the two drops of a
        # wrapper and its wrappee indistinguishable.
        if self._probing:
            return
        code = obj.__code__
        modules = self._probe_state.unmarked_modules.get(id(value))
        if modules is not None:
            names = ", ".join(type(m).__name__ for m in modules)
            reason = f"it holds nn.Module(s) not marked as external data ({names})"
        else:
            failure = self._probe_state.failures.get(id(value))
            reason = (
                f"it does not pickle ({failure})" if failure else "it does not pickle"
            )
        # co_qualname is 3.11+; the bare co_name on 3.10 cannot tell a wraps
        # wrapper from its wrappee, but __qualname__ could not either.
        log.warning(
            "dropping %s.%s (%s) from the artifact: %s; pass it in external_data to keep it (function defined at %s:%d)",
            getattr(code, "co_qualname", code.co_name),
            slot,
            type(value).__name__,
            reason,
            code.co_filename,
            code.co_firstlineno,
        )

    def _pickleable_attributes(self, obj: Any) -> dict[str, Any]:
        # Memoized for the REAL dump only, where every verdict consulted is
        # final, so a function reduced twice (it closes over itself) is neither
        # re-probed nor re-warned. A probe's answer may lean on an in-flight
        # value and must not be reused. A snapshot of the items: a probe runs
        # user __reduce__ code that may write back onto the function.
        state = self._probe_state
        if not self._probing and id(obj) in state.attributes:
            return state.attributes[id(obj)]
        attributes = {}
        for name, value in list(obj.__dict__.items()):
            if self._dumps_cleanly(value):
                attributes[name] = value
            else:
                self._warn_dropped(obj, name, value)
        if not self._probing:
            state.attributes[id(obj)] = attributes
        return attributes

    def _pickleable_doc(self, obj: Any) -> Any:
        # Nothing on the load path forces __doc__ (_apply_function_state
        # assigns it, that is all), so an unpicklable docstring is dropped like
        # a pruned attribute rather than failing the dump. A plain str is not
        # probed. Memoized for the real dump so a function reduced twice (it
        # closes over itself) warns once; unlike the attributes memo there is no
        # write-back to snapshot against, a single read cannot be mutated.
        state = self._probe_state
        if not self._probing and id(obj) in state.docs:
            return state.docs[id(obj)]
        doc = obj.__doc__
        if not self._dumps_cleanly(doc):
            self._warn_dropped(obj, "__doc__", doc)
            doc = None
        if not self._probing:
            state.docs[id(obj)] = doc
        return doc

    def _pickleable_annotations(self, obj: Any) -> dict[str, Any]:
        # The runtime must SERIALIZE these, so on 3.14 ask for evaluated VALUEs
        # rather than the FORWARDREF proxies the guard pickler reads: a proxy
        # must not be carried (it holds its owner and may drag the owner's
        # globals along). Evaluating runs the function's __annotate__ and
        # caches the result on it, the same thing inspect.signature does; when
        # it raises -- a TYPE_CHECKING-only name is the common case -- the whole
        # set is dropped, since __annotate__ is one function returning the whole
        # dict (a FORWARDREF retry that keeps the proxy-free values would
        # salvage the siblings; not done). That drop is a debug line where a
        # per-value drop below warns: a set that does not evaluate was never
        # readable at runtime in this process either (typing.get_type_hints
        # raises the same), so nothing the body does can depend on it, while a
        # value that resolved but does not pickle is a live object the body may
        # read. Below 3.14 the read hands back the live __annotations__ dict
        # (materializing an empty one on a function that has none); its items
        # are snapshotted, since a probe runs user __reduce__ code that may
        # write back onto the function, and the kept values go into a fresh
        # dict. A value can still be unpicklable -- a <locals> class resolves
        # fine yet pickle cannot reference it -- so probe each and keep only the
        # ones that dump, warning per drop like a __dict__ entry. The key is
        # dropped rather than kept with a sentinel as the guard pickler does:
        # this function is CALLED after load, and a sentinel where the body
        # expects a type is a wrong object, while a missing key is the KeyError
        # the warning predicts. Memoized for the real dump like the attributes.
        state = self._probe_state
        if not self._probing and id(obj) in state.annotations:
            return state.annotations[id(obj)]
        annotations: dict[str, Any] = {}
        try:
            annotations = self._read_raw_annotations(obj, evaluate=True)
        except Exception as e:
            code = obj.__code__
            log.debug(
                "dropping the annotations of %s (%s:%d): %s",
                getattr(code, "co_qualname", code.co_name),
                code.co_filename,
                code.co_firstlineno,
                e,
            )
        kept = {}
        for name, value in list(annotations.items()):
            if self._dumps_cleanly(value):
                kept[name] = value
            else:
                self._warn_dropped(obj, f"__annotations__[{name!r}]", value)
        if not self._probing:
            state.annotations[id(obj)] = kept
        return kept

    def _dumps_cleanly(self, value: Any) -> bool:
        # "does it pickle?" has no cheaper predicate than trying. A throwaway
        # pickler of this exact class keeps external_data/persistent_id behaviour
        # identical to the real dump. The cache stops a value from being probed
        # twice, not from being dumped again inside an ancestor's probe, so the
        # total work is the reachable bytes times the nesting depth, and user
        # __reduce__ code runs once per probe that reaches it; a tensor stashed
        # on a function is serialized into the throwaway buffer too (a transient
        # copy of its storage, and its hook warning fires once more). A
        # recursion overflow counts as unpicklable (the value is pruned) rather
        # than re-raising: a deep-but-finite value in an optional slot must not
        # fail a save that has nothing wrong with it; the guard pickler makes
        # the opposite call for the same condition, since it has a bypass to
        # fall back to and this pickler does not.
        if self._is_literal(value):
            return True
        state = self._probe_state
        vid = id(value)
        cached = state.cache.get(vid)
        if cached is not None:
            return cached
        if vid in state.parked:
            return False
        if vid in state.inflight:
            # Re-entered mid-probe (a value whose attributes reach back to
            # itself). Say picklable to break the cycle -- pickle's memo handles
            # the reference -- and record the lean so a verdict computed on top
            # of it is not cached as final.
            state.leaned = True
            return True
        # Every probed value is reachable from the object being dumped, which
        # the REAL pickler's memo keeps alive until dump() returns, so an id is
        # not reused within one dump; the cache lives as long as this pickler,
        # one per serialize(). (A function a user __reduce__ manufactures inside
        # a probe is not held that way; a later object at its address would
        # inherit its verdict.)
        probe = type(self)(self.external_data, io.BytesIO(), probe_state=state)
        state.inflight.add(vid)
        leaned_before = state.leaned
        state.leaned = False
        try:
            probe.dump(value)
        except Exception as exc:
            # No %r of the value: a repr can raise or be huge.
            log.debug(
                "pruning an unpicklable %s from a nested function: %s",
                type(value).__name__,
                exc,
            )
            state.failures[vid] = type(exc).__name__
            result = False
        else:
            # persistent_id records an unmarked nn.Module rather than raising, so
            # such a value dumps here but would fail the real serialize(); treat
            # it as unpicklable so it is pruned now instead of failing the whole
            # dump later.
            result = not probe.errors
            if not result:
                state.unmarked_modules[vid] = list(probe.errors.values())
                log.debug(
                    "pruning unmarked nn.Module(s) %s from a nested function",
                    list(probe.errors.values()),
                )
        finally:
            state.inflight.discard(vid)
            # The lean travels back through this shared flag because the nested
            # probe is reached through pickle's own dump stack (probe.dump ->
            # reducer_override -> _pickleable_attributes -> _dumps_cleanly), so
            # no return value of the child can reach this frame; restore it here
            # so an aborting dump cannot leave the child's value behind.
            leaned = state.leaned
            state.leaned = leaned_before or leaned
        # A False that leaned on an in-flight True may be a false negative, so
        # it is not cached as final. It is parked for the rest of this probe
        # tree -- re-deriving it is exponential on a cyclic cluster -- and
        # dropped when the tree finishes, so the real dump never consults it.
        # Consulting a park is not a lean: it can make a probe over-prune, and
        # over-pruning CAN flip a probe's verdict False -> True, and the real
        # dump may read that True straight from the cache. It cannot hurt: the
        # real dump never reuses a probe's ATTRIBUTE SET (the memo above is
        # gated on not _probing), so every prunable edge below such a value is
        # re-decided by an outermost probe of its own, and the non-prunable
        # slots (defaults, kwdefaults, closure) are traversed identically in
        # every probe, so a failure there would have failed the earlier probe
        # too. A True, or a False that leaned on nothing, is final. So is the
        # OUTERMOST probe's verdict, leaned or not: the only in-flight id it can
        # lean on is its own, and that lean is exact because pickle's memo
        # resolves the back-reference; the caller acts on it irrevocably.
        if result or not leaned or not state.inflight:
            state.cache[vid] = result
        else:
            state.parked.add(vid)
        if not state.inflight:
            state.parked.clear()
            state.leaned = False
        return result

    def _pickleable_type_params(self, obj: Any) -> tuple[Any, ...] | None:
        # A PEP 695 function-scoped TypeVar pickles by name as typing.<name> and
        # fails pickle's identity check against it (or the lookup, for a name
        # typing lacks), so drop the whole tuple when any element will not dump:
        # a generic cannot be rebuilt around a missing parameter. Ordinary
        # functions carry (), which dumps and is kept.
        # A TypeVar the body itself references sits in a closure cell, which is
        # never pruned (the body needs it), so that shape still fails the dump.
        # Memoized for the real dump like the other slots, so a function reduced
        # twice warns once; below 3.12 the tuple lives in __dict__ and the
        # attributes pass has already reported it, so this pass stays quiet.
        state = self._probe_state
        if not self._probing and id(obj) in state.type_params:
            return state.type_params[id(obj)]
        type_params = getattr(obj, "__type_params__", None)
        kept = type_params
        if type_params:
            # next() short-circuits: every probe serializes the reachable graph
            # into a throwaway buffer, and only the first failure is reported.
            bad = next((p for p in type_params if not self._dumps_cleanly(p)), None)
            if bad is not None:
                if "__type_params__" not in obj.__dict__:
                    self._warn_dropped(obj, "__type_params__", bad)
                kept = None
        if not self._probing:
            state.type_params[id(obj)] = kept
        return kept


class AOTCompileUnpickler(pickle.Unpickler):
    def __init__(self, external_data: dict[str, object], file: io.BytesIO) -> object:
        super().__init__(file)
        self.external_data = external_data

    def persistent_load(self, key: str) -> object:
        if key not in self.external_data:
            raise RuntimeError(
                f"Missing required external reference to data: {key}. "
                "Please load AOT compiled function with "
                "`external_data=<external data dictionary>`"
                f"{self.external_data}"
            )
        return self.external_data[key]


@dataclass
class AOTCompileSaveResult:
    serialized_data: bytes


def atomic_write_binary(file_path: str, data: bytes):
    dir_name = os.path.dirname(file_path) or "."

    with tempfile.NamedTemporaryFile(
        dir=dir_name, delete=False, mode="wb"
    ) as temp_file:
        temp_path = temp_file.name
        temp_file.write(data)
        temp_file.flush()
        os.fsync(temp_file.fileno())

    os.replace(temp_path, file_path)


@dataclass
class AOTCompiledFunction:
    _artifacts: CompileArtifacts
    _guard_check_enabled: bool = True
    _extra_globals: dict[str, object] | None = None
    # Guard-only scope, held by reference; kept apart from _extra_globals so
    # nothing in it reaches the compiled bytecode except the names __post_init__
    # picks out of it, and only where a kept guard certifies them.
    _guard_globals: dict[str, object] | None = None
    # Which of the three scopes the artifact's guards resolve against, so a
    # guard failure can say something actionable about the dict the name was
    # looked up in. Not init-settable: it stays CAPTURED unless a load path in
    # __post_init__ re-roots the guards, so it can never contradict
    # _guard_globals.
    _guard_scope: _GuardScope = dataclasses.field(
        init=False, default=_GuardScope.CAPTURED
    )
    # Reason describing why model.forward could not be resolved to a Python
    # function; read only by _missing_global_hint, to name that forward when a
    # guard fails on a global the rebuilt scope does not carry.
    _forward_not_resolved_reason: str | None = None
    # Whether any of this artifact's guards is rooted at a global other than the
    # recorded __import_* aliases, which a rebuilt scope carries freshly
    # imported, and the builtins-dict key, which the recorded scope carries
    # whether or not a guard reads it -- and which the load seeds into whichever
    # scope it resolved when one does; recorded on the load path, where it is
    # half the gate arming the live-value pick below, and read again by
    # AOTCompiledModel.deserialize, to warn once that a fallback scope makes
    # them check nothing useful. False on a freshly captured artifact.
    _has_global_guards: bool = dataclasses.field(init=False, default=False)
    # Whether the compiled bytecode reads the guarded names out of
    # _guard_globals too. Set by the module load path, which supplies one dict
    # for both roles and passes no _extra_globals of its own.
    _bytecode_reads_guard_scope: bool = False
    # The rebuilt callable, set by __post_init__ (never absent on a live
    # artifact); a declared field rather than an attribute setattr'd onto the
    # instance.
    fn: Callable[..., Any] = dataclasses.field(init=False)

    def prepare_f_locals(self, *args: object, **kwargs: object) -> dict[str, object]:
        f_locals: dict[str, object] = {}
        env = self._artifacts.runtime_env
        if env.closure:
            if not env.bytecode.co_freevars or len(env.closure) != len(
                env.bytecode.co_freevars
            ):
                raise AssertionError("closure length must match co_freevars length")
            f_locals = {
                name: cell.cell_contents
                for name, cell in zip(env.bytecode.co_freevars, env.closure)
            }
        f_locals.update(bind_locals(self._artifacts.signature, *args, **kwargs))
        return f_locals

    def guard_check(self, *args: Any, **kwargs: Any) -> bool:
        f_locals = self.prepare_f_locals(*args, **kwargs)
        if self._artifacts.guard_manager is None:
            raise AssertionError("guard_manager must not be None")
        return self._artifacts.guard_manager.check(f_locals)

    def __post_init__(self) -> None:
        from .package import load_guard_manager, load_guards_state

        self._artifacts.check_compatibility()

        extra_globals = self._extra_globals
        guards_state = None
        guard_scope = self._guard_globals
        if self._artifacts.guard_manager is None:
            guards_state = load_guards_state(self._artifacts.guards_state)
            output_graph = guards_state.output_graph
            # The serialized global_scope is the serializer's own record of the
            # names the kept guards read, so it also carries the ones reached
            # only through a shape or DUPLICATE_INPUT source, which no guard's
            # originating_source names. The builtins dict key rides along
            # whether or not a guard reads it, so it is not one of them.
            builtins_key = output_graph.name_of_builtins_dict_key_in_fglobals or ""
            guarded_globals = set(output_graph.global_scope) - {builtins_key}
            # Dynamo's own __import_* aliases are not user globals: a rebuilt
            # scope carries every recorded one freshly imported, so a guard
            # rooted at one resolves there and neither half of the warning the
            # fallback path logs applies to it. They stay in guarded_globals,
            # which a live scope still has to be read for.
            aliases = set(self._artifacts.runtime_env.import_sources)
            self._has_global_guards = bool(guarded_globals - aliases)
            if guard_scope is not None:
                # A live scope: a name it lacks must fail the guard rather than
                # fall back to the value serialized with the artifact.
                self._guard_scope = _GuardScope.SUPPLIED
                if self._bytecode_reads_guard_scope and self._has_global_guards:
                    # The guarded names only: a passing guard is what certifies
                    # that a live value is the one the graph was compiled for,
                    # and a filter may keep the guard on one global while
                    # dropping another's, so substituting the whole namespace
                    # would feed the graph values nothing checks.
                    extra_globals = {
                        **(extra_globals or {}),
                        **{
                            name: guard_scope[name]
                            for name in guarded_globals
                            if name in guard_scope
                        },
                    }

        self.fn = self._artifacts.runtime_env.forward_callable(
            self._artifacts.backend_id,
            self._artifacts.compiled_fn,
            extra_globals=extra_globals,
        )

        if guards_state is not None:
            if guard_scope is None:
                self._guard_scope = _GuardScope.RECONSTRUCTED
                guard_scope = self.fn.__globals__
            self._seed_guard_scope(guard_scope, guards_state)
            self._artifacts.guard_manager = load_guard_manager(
                guards_state,
                self._artifacts.original_code,
                guard_scope,
            )

    def _seed_guard_scope(self, guard_scope: dict[str, Any], guards_state: Any) -> None:
        # Dynamo mints __import_* aliases and a __builtins_dict___N key into the
        # tracing process's globals and roots guards at them; a process that only
        # loads never traced, so seed them here. Every guarded name is gated on a
        # kept guard being rooted at it: the seeding mutates a scope that may be a
        # user module's live namespace and installs no CleanupHook, so a name
        # nothing checks must not be written. __builtins__ is the one exception --
        # it is what the builtins dict is derived from rather than a name a guard
        # reads -- and it is written only when the gated builtins key itself is.
        # This diverges from the precompile load path in torch/_dynamo/package.py:
        # it leaves an already-bound name in place, whereas install()'s builtins
        # branch raises on a mismatched binding. A wrong binding fails the guard
        # rather than passing it, and a caller-supplied guard_scope may
        # legitimately already carry these -- so keep what is there rather than
        # fight over it.
        from .output_graph import get_builtins_dict
        from .source import get_global_source_name
        from .utils import CleanupHook

        # The serialized global_scope is pruned to the names the kept guards read,
        # so it gates the aliases. It cannot gate the builtins key: the serializer
        # writes that key into the pruned scope whether or not a guard reads it.
        # Only a caller-supplied guard_scope ever needs an alias -- forward_callable
        # builds fn.__globals__ with every recorded alias already imported, so on
        # the default path the loop below adds nothing; what can still be missing
        # there is the builtins key.
        output_graph = guards_state.output_graph
        guarded_globals = output_graph.global_scope
        for alias, module_name in self._artifacts.runtime_env.import_sources.items():
            if alias in guarded_globals and alias not in guard_scope:
                guard_scope[alias] = importlib.import_module(module_name)
        builtins_key = output_graph.name_of_builtins_dict_key_in_fglobals
        if not builtins_key:
            return
        # Every source that can root at the builtins key: guard_on_key_order roots
        # a dict-order check without appearing as any guard's originating_source.
        # The serializer's pruning scan reads two channels beyond this list --
        # the shape-env sources substituted for a ShapeEnvSource guard, and
        # DUPLICATE_INPUT's source_b, unioned in as additional_used_global_vars
        # -- but both root at a graph input, never at the builtins dict.
        sources = [guard.originating_source for guard in output_graph.guards]
        sources += output_graph.guard_on_key_order
        if builtins_key not in {get_global_source_name(source) for source in sources}:
            return
        # A pre-reset compile's CleanupHook may still own this name even when we
        # leave its value alone; drop it so it can't delete the binding once
        # collected.
        CleanupHook.disown(guard_scope, builtins_key)
        if builtins_key not in guard_scope:
            # Neither a caller-supplied f_globals nor the scope rebuilt from the
            # serialized bytecode need carry __builtins__; exec would seed it, so
            # fall back to the real builtins here.
            if "__builtins__" not in guard_scope:
                guard_scope["__builtins__"] = builtins.__dict__
            guard_scope[builtins_key] = get_builtins_dict(guard_scope)

    def _missing_global_hint(self, *, forward: str | None = None) -> str:
        """Advice for a guard that failed on a global its scope does not define,
        worded for the scope the guards were actually resolved against. A bare
        sentence: a caller that continues a line of its own adds the separator.
        ``forward`` names the model class's ``forward``, which only the module
        path knows; a module load resolves the live scope from the INSTANCE
        attribute, which is why the SUPPLIED wording below names the function a
        rebind put there too. Honoured only in the SUPPLIED branch -- the other
        two scopes are not resolved from a model attribute at all."""
        if self._guard_scope is _GuardScope.RECONSTRUCTED:
            rebuilt = (
                "a guarded global is missing from the scope rebuilt from the "
                "artifact, which holds only the globals the graph lifted"
            )
            if self._forward_not_resolved_reason is not None:
                # No module load path takes an f_globals (neither
                # OptimizedModule._load_aot_compiled_module nor
                # AOTCompiledModel.deserialize), so the only way to reach a live
                # scope here is to make forward resolvable again.
                return (
                    f"{rebuilt}. That scope was rebuilt because get_traced_fn "
                    f"cannot resolve {self._forward_not_resolved_reason} to a "
                    "Python function; make model.forward a plain function or "
                    "bound method so its own globals are used instead."
                )
            return (
                f"{rebuilt}; load with f_globals= set to a complete live scope "
                "that carries it -- normally vars() of the module that defined "
                "the function, which is usually not the module doing the loading "
                "-- so the guard can resolve it."
            )
        if self._guard_scope is _GuardScope.SUPPLIED:
            where = (
                # A module load resolves the scope from the INSTANCE attribute,
                # so the dict is the globals of the function that attribute
                # resolves to. Resolving forward on the class lands in that
                # same dict whenever it reaches that same function, inheritance
                # from another module included, and can land elsewhere once an
                # instance rebinds forward; naming the class's forward alone
                # would send that reader to a dict these guards never read.
                f"the globals of the function {forward} resolves to -- or, for "
                "an instance that rebound forward before the load, of the "
                "function it was rebound to, since that is the one the load "
                "resolved"
                if forward is not None
                else "the live scope this artifact was loaded against"
            )
            return (
                f"a guarded global is missing from {where}; define it there "
                "so the guard can resolve it."
            )
        # CAPTURED: the guards hold the globals they were traced against BY
        # REFERENCE, so a name deleted after capture can be defined there again
        # to make the guard resolve -- the same advice as SUPPLIED, worded for
        # the dict this path actually used.
        return (
            "a guarded global is missing from the globals of the module the "
            "compiled function was traced in, which its guards still resolve "
            "against; define it there so the guard can resolve it."
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._artifacts.guard_manager is None:
            raise AssertionError("guard_manager must not be None")
        if self._guard_check_enabled and not self.guard_check(*args, **kwargs):
            f_locals = self.prepare_f_locals(*args, **kwargs)
            debug_info = self._artifacts.guard_manager.check_verbose(f_locals)
            msg = f"GuardManager check failed, reason: {debug_info}"
            if any(
                _names_a_missing_global(part) for part in debug_info.verbose_code_parts
            ):
                # What the f-string interpolated is str(GuardDebugInfo), which
                # ends in a newline, so the hint has to be appended to the
                # stripped message: otherwise its inline continuation lands on a
                # line of its own, starting with a stray space.
                msg = msg.rstrip() + " -- " + self._missing_global_hint()
            raise RuntimeError(msg)
        return self.fn(*args, **kwargs)

    def source_info(self) -> "SourceInfo":
        return self._artifacts.source_info

    def save_compiled_function(
        self, path: str, external_data: dict[str, Any] | None = None
    ) -> AOTCompileSaveResult:
        result = type(self).serialize(self, external_data)
        atomic_write_binary(path, result.serialized_data)
        return result

    @classmethod
    def serialize(
        cls, fn: "AOTCompiledFunction", external_data: dict[str, Any] | None = None
    ) -> AOTCompileSaveResult:
        state = fn._artifacts.__dict__.copy()
        state["guard_manager"] = None
        state["runtime_env"] = dataclasses.replace(
            state["runtime_env"],
            bytecode=SerializedCode.from_code_object(state["runtime_env"].bytecode),
        )
        compiled_fn = state["compiled_fn"]
        # The backend pickles itself here, deliberately outside the handler
        # below: external_data cannot fix an unpicklable graph module, so that
        # failure must not be dressed up with guidance pointing at it.
        state["compiled_fn"] = (
            type(compiled_fn).deserialize_compile_artifacts,
            type(compiled_fn).serialize_compile_artifacts(compiled_fn),
        )
        state["original_code"] = SerializedCode.from_code_object(state["original_code"])
        buf = io.BytesIO()
        pickler = AOTCompilePickler(external_data or {}, buf)
        try:
            pickler.dump(state)
        except (pickle.PicklingError, TypeError, AttributeError, RecursionError) as e:
            # Preserve the original exception object -- callers and tests match
            # on it (e.g. "cannot pickle '_thread.lock' object") -- and append
            # guidance. Mutate args and re-raise rather than type(e)(msg): a
            # TypeError subclass from a user __reduce__ may take a non-message
            # constructor, so reconstructing would swap the real error for a
            # constructor failure. (A subclass whose __str__ ignores args still
            # renders without the guidance; add_note() would cover it but is
            # 3.11+.) The args tail is kept for a consumer that reads it.
            # AttributeError is caught too: the C _pickle accelerator raises a
            # bare AttributeError "Can't get local object" for a <locals> class
            # in a default/kwdefault (3.14+ raises PicklingError), so it needs
            # the same guidance. RecursionError as well: a deep-but-finite value
            # in an unpruned slot overflows the C pickler, and external_data is
            # its fix too. Unmarked modules recorded before the failure are
            # reported here rather than on the next attempt.
            # str(e) is repr(args) for a 2+-argument exception, which would nest
            # the tuple repr and escape the newline; the head argument is the
            # message. str(e) of the result is still a tuple repr in that case,
            # which is the price of keeping the tail.
            message = str(e.args[0]) if e.args else ""
            if _EXTERNAL_DATA_HINT in message:
                raise  # a re-raised singleton already carries the guidance
            prefix = f"{message}\n" if message else ""
            modules = ""
            if pickler.errors:
                # Class names, not reprs: nn.Module.__repr__ renders the whole
                # child tree and runs user code inside this handler.
                names = ", ".join(type(m).__name__ for m in pickler.errors.values())
                modules = f" It also reached unmarked nn.Modules ({names})."
            e.args = (
                prefix + "Some value reached by the artifact is not picklable (a "
                "closure cell, a default/kwdefault, or the top-level function's "
                "own signature annotations, which ride unpruned, are the common "
                f"sources).{modules} {_EXTERNAL_DATA_HINT}",
                *e.args[1:],
            )
            raise
        if pickler.errors:
            raise RuntimeError(
                f"Failed to serialize the following objects: {list(pickler.errors.values())}\n"
                f"{_EXTERNAL_DATA_HINT}"
            )
        return AOTCompileSaveResult(serialized_data=buf.getvalue())

    @classmethod
    def deserialize(
        cls,
        data: bytes,
        f_globals: dict[str, object] | None = None,
        external_closure_data: dict[str, Any] | None = None,
        *,
        guard_globals: dict[str, object] | None = None,
        bytecode_reads_guard_scope: bool = False,
        forward_not_resolved_reason: str | None = None,
    ) -> "AOTCompiledFunction":
        """Rebuild a compiled function from ``serialize()`` output.

        ``f_globals`` and ``guard_globals`` have distinct contracts and must not
        be conflated. ``f_globals`` is MERGED over the scope reconstructed from
        the serialized bytecode (extra names the compiled fn may reference), so a
        name it omits still resolves to the baked-in value. ``guard_globals``
        REPLACES the guard scope with no such fallback -- a name it lacks fails
        the guard rather than resolving to a serialized value, and an empty dict
        is an empty scope rather than "no scope" -- so it is the live namespace
        global guards are re-rooted at on load. It is WRITTEN into as well as
        read: the load seeds the recorded import aliases a kept guard is rooted
        at, the recorded builtins-dict key if a kept guard reads it, and
        ``__builtins__`` if that key has to be built -- never replacing a name it
        already binds -- so pass the dict those names should land in.
        Passing neither resolves global guards against the scope rebuilt from
        the artifact: a global the graph lifted is checked against the value
        serialized with it, one it did not lift is simply absent and fails the
        guard, and either way a rebinding in this process is invisible.

        ``bytecode_reads_guard_scope`` additionally serves the graph the live
        value of each global a kept guard reads -- apart from the recorded
        ``__builtins_dict___N`` key, which is excluded by name -- picked out of
        ``guard_globals`` one name at a time: a caller that cannot inspect the
        guards itself, i.e. the module load path, gets the substitution only
        where a guard certifies it, and never for a global whose guard a filter
        dropped.
        """
        f = io.BytesIO(data)
        f.seek(0)
        unpickler = AOTCompileUnpickler(external_closure_data or {}, f)
        state = unpickler.load()
        f.close()
        state["runtime_env"] = dataclasses.replace(
            state["runtime_env"],
            bytecode=SerializedCode.to_code_object(state["runtime_env"].bytecode),
        )
        deserializer, compiled_fn_state = state["compiled_fn"]
        with torch._inductor.config.patch(enable_autograd_for_aot=True):
            state["compiled_fn"] = deserializer(compiled_fn_state)
        state["original_code"] = SerializedCode.to_code_object(state["original_code"])

        artifacts = CompileArtifacts(**state)
        return cls(
            artifacts,
            _extra_globals=f_globals,
            _guard_globals=guard_globals,
            _bytecode_reads_guard_scope=bytecode_reads_guard_scope,
            _forward_not_resolved_reason=forward_not_resolved_reason,
        )

    def disable_guard_check(self) -> None:
        self._guard_check_enabled = False


def aot_compile_fullgraph(
    model: Any,
    example_inputs: tuple[tuple[Any, ...], dict[str, Any]],
    hooks: Hooks,
    backend: Callable[[torch.fx.GraphModule, list[torch.Tensor]], SerializableCallable],
    dynamic: bool | None = None,
) -> AOTCompiledFunction:
    from torch._dynamo.guards import CheckFunctionManager
    from torch._dynamo.package import SourceInfo
    from torch._dynamo.utils import dynamo_timed, get_metrics_context
    from torch._dynamo.variables.torch_function import (
        torch_function_mode_stack_state_mgr,
    )
    from torch._guards import TracingContext

    args, kwargs = example_inputs

    dynamic_ctx = nullcontext()
    if dynamic is not None:
        from torch._dynamo.eval_frame import set_enable_dynamic

        dynamic_ctx = set_enable_dynamic(dynamic)

    with (
        get_metrics_context(),
        dynamo_timed("fullgraph_capture"),
        torch._functorch.config.patch(strict_autograd_cache=True),
        dynamic_ctx,
        torch_function_mode_stack_state_mgr,
    ):
        capture_output = convert_frame.fullgraph_capture(model, args, kwargs)
        graph_capture_output = capture_output.graph_capture_output
        if graph_capture_output.output_graph is None:
            raise AssertionError("output_graph must not be None")

        if not hooks.guard_filter_fn:
            from torch._dynamo.types import GuardFilterEntry

            def new_guard_filter_fn(
                guard_entries: Sequence[GuardFilterEntry],
            ) -> Sequence[bool]:
                # NB: dropping every global guard is what
                # torch.compiler.skip_guard_on_globals_unsafe does explicitly,
                # and the "unsafe" in that name applies here too: a dropped
                # global guard does not fail, it silently reuses a graph traced
                # under a different global value. Narrowing this default needs
                # every load to supply a scope binding every global a kept
                # guard reads, because a load that supplies none rebuilds one
                # from the artifact: there a global the graph lifted resolves to
                # the value serialized with it, so the kept guard certifies that
                # rather than the loading process's, and one the graph did not
                # lift is simply absent, failing every call with KeyError on
                # G['...']. So callers who need a specific global guarded must
                # pass guard_filter_fn.
                return [
                    (
                        not (
                            g.is_global
                            or g.guard_type
                            in CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
                        )
                    )
                    for g in guard_entries
                ]

            hooks.guard_filter_fn = new_guard_filter_fn

        fn, _ = convert_frame.get_traced_fn(model)

        backend_input = capture_output.backend_input
        if backend_input is None:
            raise AssertionError("backend_input must not be None")
        backend_input.graph_module._backend_id = backend_input.backend_id  # type: ignore[assignment]
        graph_devices = _graph_device_types(backend_input.graph_module.graph)
        device_type = _collapse_device_types(graph_devices)
        if (
            backend_input.fake_mode.shape_env
            is not graph_capture_output.output_graph.shape_env
        ):
            raise AssertionError(
                "fake_mode.shape_env must be the same as output_graph.shape_env"
            )
        tracing_context = TracingContext(backend_input.fake_mode)
        tracing_context.tensor_to_context = backend_input.tensor_to_context
        with (
            torch._guards.tracing(tracing_context),
            torch._functorch.config.patch(
                {
                    "strict_autograd_cache": True,
                    "bypass_autograd_cache_key": True,
                    "bundled_autograd_cache": True,
                    "force_non_lazy_backward_lowering": True,
                    "force_autograd_cache": True,
                }
            ),
        ):
            compiled_fn = backend(
                backend_input.graph_module, backend_input.example_inputs
            )
            # If Inductor backend or AOTAutograd-based backend is used,
            # wrap the compiled_fn for serialization.
            # TODO: this should be replaced once we make the backend return the SerializableCallable directly.
            if (
                isinstance(backend, torch._TorchCompileInductorWrapper)
                or (
                    hasattr(backend, "compiler_fn")
                    and isinstance(
                        backend.compiler_fn, torch._dynamo.backends.common.AotAutograd
                    )
                )
                or (
                    hasattr(compiled_fn, "serialize")
                    and compiled_fn.serialize is not None
                )
            ):
                compiled_fn = BundledAOTAutogradSerializableCallable(compiled_fn)

        if not isinstance(compiled_fn, SerializableCallable):
            if hasattr(backend, "compiler_fn"):
                compiler_fn = backend.compiler_fn
            else:
                compiler_fn = backend
            raise RuntimeError(
                f"Compiled function type {type(compiled_fn)} (produced "
                + f"from backend {compiler_fn}) does not implement SerializableCallable."
            )

        # Temporarily restore the mode stack so guard expressions that
        # reference modes can evaluate, matching the compile_inner path.
        build_guards_ctx = ExitStack()
        if torch_function_mode_stack_state_mgr.stack:
            build_guards_ctx.enter_context(
                torch_function_mode_stack_state_mgr.temp_restore_stack()
            )
        with build_guards_ctx:
            check_fn = graph_capture_output.build_guards(
                fn.__code__, hooks=hooks, save=True, strict_error=True
            )

        if check_fn.guards_state is None:
            raise AssertionError("guards_state must not be None")

        source_info = SourceInfo(inlined_sources=set())
        for traced_code in graph_capture_output.traced_code:
            source_info.add_code(traced_code)

        artifacts = CompileArtifacts(
            signature=convert_frame._get_signature(fn),
            guard_manager=check_fn.guard_manager,
            guards_state=check_fn.guards_state,
            backend_id=backend_input.backend_id,
            compiled_fn=compiled_fn,
            original_code=fn.__code__,
            runtime_env=graph_capture_output.get_runtime_env(),
            source_info=source_info,
            device_type=device_type,
            backend_name=getattr(backend, "compiler_name", "unknown"),
        )
        aot_compiled_fn = AOTCompiledFunction(
            _artifacts=artifacts, _extra_globals=fn.__globals__
        )

    return aot_compiled_fn


def _unwrap_optimized_module(model: torch.nn.Module) -> torch.nn.Module:
    # isinstance, not getattr(model, "_orig_mod", model): _orig_mod is a
    # registrable submodule name, and unwrapping to a child would run the
    # parent's graph against the child's parameters.
    from torch._dynamo.eval_frame import OptimizedModule

    return model._orig_mod if isinstance(model, OptimizedModule) else model


@dataclass
class ModelInput:
    """
    WIP type: represents a single model input
    Which consists of a tuple of arguments and a set of contexts in which to run the model.

    For each ModelInput, we'll compile one full graph of the model, and then use the guards generated
    to dispatch between the compiled graphs.


    """

    args: tuple[object, ...]
    kwargs: dict[str, object]
    contexts: Sequence[AbstractContextManager[object]]


def _warn_dropped_module_dispatch(model: torch.nn.Module) -> None:
    # nn.Module's hook dispatch in _call_impl never runs -- the capture traces
    # model.forward and the load calls what it compiled to -- so every hook on
    # model itself is dropped, a silently different answer rather than an error.
    # These four dicts are the per-instance subset of _call_impl's eight-dict
    # fast-path test; the four _global_* ones still fire on the wrapper's own
    # _call_impl when the artifact is served through OptimizedModule, so only a
    # direct AOTCompiledModel drops them, which deserialize's docstring covers
    # instead.
    # The *_with_kwargs and *_always_called side tables are keyed by handles
    # already in these dicts, so they cannot be non-empty alone. Forward and
    # backward are worded separately because the redirect is unconditional only
    # for the forward dicts: tracing torch.compile(model).forward keeps those,
    # while a module-level backward hook needs compiled autograd enabled around
    # the same trace ("Module-level backwards hooks require compiled autograd"),
    # which only compiled_autograd._enable does -- aot_compile never enters the
    # dynamo context that reads the config flag the graph break's hint names --
    # and the artifact that produces cannot be reloaded. A dropped backward hook
    # also leaves the forward result alone and changes the gradients, so it needs
    # different wording than "the result may differ".
    forward_hooked = [
        label
        for label, attr in (
            ("forward pre-hooks", "_forward_pre_hooks"),
            ("forward hooks", "_forward_hooks"),
        )
        if getattr(model, attr, None)
    ]
    backward_hooked = [
        label
        for label, attr in (
            ("backward pre-hooks", "_backward_pre_hooks"),
            ("backward hooks", "_backward_hooks"),
        )
        if getattr(model, attr, None)
    ]
    if forward_hooked:
        log.warning(
            "%s has %s registered; the AOT compiled forward calls %s.forward "
            "directly, so those hooks do NOT run and its result may differ from "
            "eager -- to keep them, AOT compile torch.compile(model).forward "
            "instead, which traces __call__ and runs them; call the artifact it "
            "returns with the module as its first argument",
            type(model).__name__,
            ", ".join(forward_hooked),
            type(model).__name__,
        )
    if backward_hooked:
        log.warning(
            "%s has %s registered; the AOT compiled forward calls %s.forward "
            "directly, so those hooks do NOT run and the gradients it produces "
            "may differ from eager while the forward result does not -- to keep "
            "them, AOT compile torch.compile(model).forward with compiled "
            "autograd enabled around the capture "
            "(torch._dynamo.compiled_autograd._enable); call the artifact it "
            "returns with the module as its first argument, and note that it "
            "cannot be saved and reloaded",
            type(model).__name__,
            ", ".join(backward_hooked),
            type(model).__name__,
        )
    # Eager dispatches through type(model).__call__ while the artifact calls what
    # forward compiled to, so an overridden __call__ is dropped just like a hook
    # -- and no hook dict records it, so the lists above see nothing to report.
    # The probe is that same type lookup: an instance attribute named __call__ is
    # not an override, because CPython resolves a special method on the type, so
    # eager ignores it too and the artifact matches.
    if type(model).__call__ is not torch.nn.Module.__call__:
        log.warning(
            "%s overrides __call__; the AOT compiled forward runs what "
            "%s.forward compiled to, so that override does NOT run and its "
            "result may differ from eager -- to keep it, AOT compile "
            "torch.compile(model).forward instead, which traces __call__; call "
            "the artifact it returns with the module as its first argument",
            type(model).__name__,
            type(model).__name__,
        )


@dataclass
class AOTCompiledModel:
    # Represents a single forward function of a model along with dispatch
    # compiled_results is serializable. We require the model to deserialize again.
    model: torch.nn.Module
    compiled_results: list[AOTCompiledFunction]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # guard_check() evaluates guards regardless of _guard_check_enabled, so
        # scan EVERY result for a real match first -- skipping opted-out results
        # here would, when all of them opted out, fall through to the first
        # result below and silently serve the wrong graph.
        for result in self.compiled_results:
            # guard_check binds the call before evaluating anything, so a call
            # the signature cannot bind surfaces as bind_locals' TypeError, as
            # the plain module call would -- a caller error no ModelInput could
            # fix, rather than a no-match report.
            if result.guard_check(self.model, *args, **kwargs):
                # guard_check already passed; call fn directly so result()
                # does not re-run the guard eval on this hot dispatch path.
                return result.fn(self.model, *args, **kwargs)
        # check() can reject from the recursive dict-tag fast path without ever
        # running the tree, so a rejection above is not yet an answer about this
        # call -- but the node that rejected and its ancestors have their
        # _disable_dict_tag_matching set, which nothing resets, so a second
        # check() re-evaluates those in full (a sibling tag-safe root the first
        # pass never reached stays armed, and can still answer from its fast
        # path). That is the rescue the old fall-through to compiled_results[0]
        # got from re-entering AOTCompiledFunction.__call__, here extended to
        # every result rather than only the first, and the extra pass
        # costs a genuine mismatch nothing: such a tree still short-circuits at
        # its first failing guard. Opted-out results are skipped -- nobody asked
        # about their guards, and the last resort below serves them anyway.
        for result in self.compiled_results:
            if result._guard_check_enabled and result.guard_check(
                self.model, *args, **kwargs
            ):
                return result.fn(self.model, *args, **kwargs)
        # A result that opted out via disable_guard_check() accepts anything, but
        # only after both passes above have failed to find a real match.
        for result in self.compiled_results:
            if not result._guard_check_enabled:
                return result.fn(self.model, *args, **kwargs)
        raise RuntimeError(self._no_match_report(*args, **kwargs))

    def _no_match_report(self, *args: Any, **kwargs: Any) -> str:
        """A report naming every compiled input and what its guards said."""
        lines = [
            "No AOT compiled graph matched this call. Tried "
            f"{len(self.compiled_results)} compiled input(s):"
        ]
        missing_global_result: AOTCompiledFunction | None = None
        for i, result in enumerate(self.compiled_results):
            # Narrowing for pyrefly, not a live check: __post_init__ always
            # leaves a populated guard_manager (only serialize() nulls it, on a
            # copy), and the dispatch scan in __call__ already required one on
            # every result before we got here.
            guard_manager = result._artifacts.guard_manager
            if guard_manager is None:
                raise AssertionError("live artifact must have a guard_manager")
            f_locals = result.prepare_f_locals(self.model, *args, **kwargs)
            reason = guard_manager.check_verbose(f_locals)
            if reason.result:
                # Both dispatch passes rejected this call and check_verbose, which
                # never takes the dict-tag fast path, accepts it: either a guard
                # here does not answer consistently, or a still-armed tag-safe
                # root neither rejection reached refused from its fast path.
                # Quoting the guards it just passed as a reason the call failed
                # would name the wrong thing.
                lines.append(
                    f"  [{i}] <guards rejected this call twice and then accepted "
                    "it here: a guard that does not answer consistently, or a "
                    "tag-safe fast path that refused without running the tree>"
                )
                continue
            if not reason.verbose_code_parts:
                # A failing accessor can report no parts at all (a set index past
                # the end of a shorter set answers GuardDebugInfo(false, 0)), so
                # an empty list is not the passing signal reason.result is. The
                # call did not match: advise as for any other mismatch.
                lines.append(f"  [{i}] <guard check failed without naming a guard>")
                continue
            parts = [str(p) for p in reason.verbose_code_parts]
            if any(_names_a_missing_global(p) for p in parts):
                if missing_global_result is None:
                    missing_global_result = result
            joined = "; ".join(parts).replace("\n", " ")
            lines.append(f"  [{i}] {joined}")
        # Each advice line below stands on its own entries, and more than one can
        # apply: an entry that named a missing global says nothing about a call no
        # input covers, and an input captured for the branch this call takes need
        # not read that global at all, which the report cannot know either way.
        if missing_global_result is not None:
            # Read off the entry that failed on the missing global, which is
            # every entry's scope in practice -- deserialize hands them all one
            # guard_globals.
            forward = f"{type(self.model).__name__}.forward"
            lines.append(missing_global_result._missing_global_hint(forward=forward))
        # Every entry above answered in both dispatch passes, so an input
        # covering this call is on the table whatever its reason named.
        lines.append(
            "Add a ModelInput covering this call, or check whether "
            "guard_filter_fn kept a guard this call cannot satisfy."
        )
        return "\n".join(lines)

    def serialize(self) -> bytes:
        # Nothing threads external_data down this path (_save_aot_compiled_module
        # has no parameter for it either), so the guidance a failed save appends
        # is only actionable by saving the offending function on its own.
        data: list[bytes] = []
        for result in self.compiled_results:
            data.append(AOTCompiledFunction.serialize(result).serialized_data)
        return pickle.dumps(data)

    @classmethod
    def deserialize(cls, model: torch.nn.Module, data: bytes) -> "AOTCompiledModel":
        """Rebuild the compiled forward of ``model`` from ``serialize()`` output.

        ``model`` may be the module itself or the wrapper ``torch.compile``
        returned for it, which is unwrapped to the module that was traced.

        Guards on globals are evaluated, by reference, against the live
        ``__globals__`` of the function ``model.forward`` resolves to. That dict
        is mutated only for the names a kept guard is rooted at: a recorded
        ``__import_*`` alias the serialized scope still carries, and the
        ``__builtins_dict___N`` key when a guard source names it, are inserted
        (never overwriting an existing key) so guards rooted at them resolve in a
        process that never traced. A guarded global the dict lacks fails the
        guard; there is no fallback to the serialized scope. The compiled bytecode
        reads a snapshot, taken here, of the globals serialized with the artifact,
        with the names a kept guard reads taken from that live dict instead --
        apart from the recorded ``__builtins_dict___N`` key, which is excluded
        from that substitution by name whether or not a guard reads it -- so a
        guard on a substituted name certifies the value the graph will actually
        use and every other global is the one the graph was traced with. A
        global rebound after the load keeps being served from the snapshot
        unless a guard on its value refuses the call, and a kept
        ``TENSOR_MATCH`` checks metadata, not values.
        Only when ``model.forward`` cannot be resolved to a Python function by
        ``get_traced_fn`` is there no live scope to use; guards then resolve
        against the scope rebuilt from the artifact, with a warning when a guard
        is rooted at a global other than those aliases and that key, and there
        they are useless in both directions: a guard on a global the graph
        lifted is compared against the value serialized with it and cannot fail,
        and one on a global that scope does not carry can never be satisfied.

        Hooks registered on ``model`` do not run: the artifact calls ``forward``
        directly. The artifact records none, so what is warned about here is what
        ``model`` carries at this call; the ones it carried at capture were
        warned about there, and one registered after this call is dropped
        silently. An overridden ``__call__`` is dropped the same way, and warned
        about the same way. Only ``model`` itself is inspected: a hook or a
        ``__call__`` override a SUBMODULE carries only in the loading process is
        dropped silently too, because the graph baked in whatever the capture
        traced through that submodule's ``nn.Module.__call__``. Hooks registered
        globally (``register_module_forward_hook`` and friends) are dropped
        without a warning only when the returned ``AOTCompiledModel`` is called
        directly: they still run on the wrapper's own ``_call_impl`` when the
        artifact is served through ``OptimizedModule``, so only the per-instance
        dicts are worth warning about.
        """
        from torch._dynamo.utils import get_metrics_context
        from torch._guards import compile_context, CompileContext

        # An OptimizedModule's forward is a wrapper defined in eval_frame, so
        # resolving from it would root every global guard in that module's
        # namespace (and seed it); dispatch also has to pass the module that was
        # actually traced as self. eval_frame's own loader unwraps for us; a
        # caller of this classmethod may not have.
        model = _unwrap_optimized_module(model)

        _warn_dropped_module_dispatch(model)
        # Resolve from model.forward, not the model: for a hooked module
        # get_traced_fn would return Module._wrapped_call_impl and nn.Module's
        # namespace.
        forward = model.forward
        forward_not_resolved_reason = None
        try:
            traced_fn, _ = convert_frame.get_traced_fn(forward)
            scope = traced_fn.__globals__
        except (RuntimeError, AttributeError):
            # Format forward in a bounded way to avoid dumping the entire module
            # repr (functools.partial embeds the module's full repr).
            forward_type = type(forward).__name__
            forward_qualname = getattr(forward, "__qualname__", "")
            forward_not_resolved_reason = (
                f"{type(model).__name__}.forward ({forward_type}"
                f"{f' named {forward_qualname}' if forward_qualname else ''})"
            )
            scope = None

        results: list[bytes] = pickle.loads(data)
        compiled_results = []
        for result in results:
            with (
                compile_context(CompileContext(convert_frame.get_compile_id({}))),
                get_metrics_context(),
            ):
                compiled_results.append(
                    AOTCompiledFunction.deserialize(
                        result,
                        guard_globals=scope,
                        bytecode_reads_guard_scope=True,
                        forward_not_resolved_reason=forward_not_resolved_reason,
                    )
                )
        # Model-level, so warn once here rather than once per ModelInput.
        if forward_not_resolved_reason is not None and any(
            result._has_global_guards for result in compiled_results
        ):
            log.warning(
                "%s, from which no live guard scope could be resolved "
                "(get_traced_fn cannot resolve model.forward to a Python "
                "function); global guards on this artifact resolve against "
                "the scope rebuilt from the serialized bytecode instead, "
                "where they check nothing useful: one on a global the graph "
                "lifted is compared against the value serialized with it and "
                "cannot fail, and one on a global that scope does not carry "
                "cannot be satisfied, so the call will report no match",
                forward_not_resolved_reason,
            )
        return cls(model, compiled_results)


def aot_compile_module(
    model: torch.nn.Module,
    inputs: list[ModelInput],
    hooks: Hooks,
    backend: Callable[[torch.fx.GraphModule, list[torch.Tensor]], SerializableCallable],
) -> AOTCompiledModel:
    """
    Compiles a single nn.Module with any number of inputs, and returns a compiled forward function.

    ``model`` may be the module itself or the wrapper ``torch.compile`` returned
    for it, which is unwrapped to the module to trace. ``model.forward`` is what
    gets traced, so the per-instance hooks ``nn.Module.__call__`` would dispatch
    on, and an overridden ``__call__`` eager reaches instead of it, are dropped
    from the result; both are warned about, as they are on the load path.
    """
    # eval_frame's caller hands us _orig_mod; a caller of this function may not
    # have. Everything below needs the module that was traced: tracing an
    # OptimizedModule.forward reaches eval_frame's compile_wrapper and dies on
    # set_eval_frame, the wrapper would be stored as the self the recorded
    # type-id guard is checked against, and warning about it would report a
    # __call__ override nobody wrote while missing the hooks that are dropped.
    model = _unwrap_optimized_module(model)
    _warn_dropped_module_dispatch(model)

    def compile_single_graph(model_input: ModelInput) -> AOTCompiledFunction:
        example_inputs = (model_input.args, model_input.kwargs)
        orig_forward = model.forward
        with ExitStack() as stack:
            for ctx in model_input.contexts:
                stack.enter_context(ctx)
            return aot_compile_fullgraph(
                orig_forward,
                example_inputs,
                hooks=hooks,
                backend=backend,
            )

    # pyrefly: ignore [implicit-any]
    compiled_results = []
    for model_input in inputs:
        log.info("Compiling input %s..", model_input)
        compiled_results.append(compile_single_graph(model_input))

    if len(compiled_results) == 0:
        raise AssertionError("Expected at least one compiled result")

    return AOTCompiledModel(model, compiled_results)
