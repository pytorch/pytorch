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
import sys
import tempfile
import types
import weakref
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, ExitStack, nullcontext
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

import torch
import torch.fx
from torch._dynamo.convert_frame import GraphRuntimeEnv
from torch._dynamo.graph_utils import _graph_device_types
from torch._dynamo.package import (
    _collapse_device_types,
    FunctionPicklerBase,
    SerializedCode,
    SystemInfo,
)

from . import convert_frame, external_utils
from .aot_compile_types import (
    BundledAOTAutogradSerializableCallable,
    SerializableCallable,
)
from .hooks import Hooks


if TYPE_CHECKING:
    from .guards import GuardManagerWrapper, GuardsState
    from .output_graph import OutputGraphGuardsState
    from .package import SourceInfo


log = logging.getLogger(__name__)

_EXTERNAL_DATA_HINT = (
    "Mark the value(s) as external data by using `external_data={'key': ...}`."
)

# What a raise can cost the tree it came out of, said once: the no-match report
# and the warning the serving paths log both name it. Phrased for one tree or
# several, since the report's caveat names every tree it rests on. Hedged,
# because only a throw skips the reset on check_nopybind_template's exits: a
# tree that returns with an error set (the SystemError _unwrapped_raise reads
# through) reset on its way out, and neither the last-resort veto nor this
# clause tells the two apart.
_STALE_AFTER_THROW = (
    "a C++ throw out of a tree can leave that tree's relational guard state "
    "stale, so its next check can reject a call it fits or accept one it does not"
)


# A guard failure that is exactly a missing top-level global: the verbose code
# part a guard tree reports for one ("KeyError on G['CONFIG']"). A trailing
# subscript ("KeyError on G['CONFIG']['scale']") means the global itself
# resolved and only a key inside it is absent, so the advice to define the
# global would be wrong.
_MISSING_GLOBAL_RE = re.compile(r"KeyError on G\[(?P<name>[^\[\]]*)\]")
# The G['NAME'] operands of a symbolic-shape guard installed as a Python lambda.
# Anchored: shape exprs are source names, so L['self'].myG['k'] carries no global.
_SHAPE_GUARD_GLOBAL_RE = re.compile(r"\bG\['([^']*)'\]")
_UNBOUND = object()

# Names Dynamo mints into the scope the guards resolve against, rather than
# names the caller wrote: the __import_* module aliases, the __builtins_dict___N
# key, and the ___unnamed_scope_<id>_c<n> key an inlined frame whose globals
# belong to no module is guarded through. A load seeds each of the three a kept
# guard is rooted at, so a KeyError on one reports a gap in that seeding -- for
# the last, a namespace the graph only specialized on: the key embeds id() of a
# dict in the tracing process, so no module's vars() in a loading process holds
# it, and the artifact carries the dict only where the graph lifted a value read
# through it. None of the three is a name the advice below can send a caller to
# define. The list is complete because a report here needs a serializable guard
# that resolves through a GlobalSource on the name, as get_global_source_name
# decides it: that helper walks a chained source to its base, so a guard on
# G['__import_torch'].nn.modules.module._global_forward_hooks counts for
# __import_torch, and prune_variable records the name through it. Every other
# minted family builds no Source (the codegen-only installs) or a guard type in
# UNSUPPORTED_SERIALIZATION_GUARD_TYPES, which ___unnamed_scope's was not -- so
# moving a type off that list means re-checking this one.
_UNNAMED_SCOPE_PREFIX = "___unnamed_scope"
_MINTED_GLOBAL_PREFIXES = ("__import_", "__builtins_dict__", _UNNAMED_SCOPE_PREFIX)


def _picklable_unnamed_scope(scope: dict[str, Any]) -> dict[str, Any]:
    # exec inserts the LIVE builtins dict under __builtins__ into a namespace
    # that lacks one, and used_globals records an inlined frame's unnamed scope
    # whole. By value that is every builtin plus whatever extension modules
    # stash there (pybind11 < 2.13 on CPython < 3.12 keeps its internals in a
    # PyCapsule), while nothing at load reads it: the artifact only subscripts
    # the dict. Send the guards pickler's stand-in, which loads as the live one.
    from .guards import _live_builtins

    if scope.get("__builtins__") is not builtins.__dict__:
        return scope
    return {**scope, "__builtins__": _live_builtins}


def _names_a_missing_global(text: str) -> bool:
    # Matched whole, against one verbose code part: matching a substring of the
    # GuardDebugInfo string would also fire for the nested-key failure above.
    match = _MISSING_GLOBAL_RE.fullmatch(text)
    if match is None:
        return False
    return not match["name"].strip("\"'").startswith(_MINTED_GLOBAL_PREFIXES)


def _unwrapped_raise(e: Exception) -> tuple[str, BaseException]:
    """What a guard tree meant to raise, as ``(type name, exception)``; the one
    reading the report line and the warning share, so they cannot drift."""
    # A tree returning to pybind with an exception still set arrives as a
    # SystemError whose str() is the bound method's repr, so report what
    # _PyErr_FormatFromCause chained behind it. __cause__, not __context__: that
    # call sets both, but PEP 3134 sets __context__ for ANY exception raised while
    # another was handled, so it would quote what the CALLER was handling.
    cause = e.__cause__ if isinstance(e, SystemError) else None
    reason: BaseException = e if cause is None else cause
    return type(reason).__name__, reason


def _raise_text(e: Exception) -> str:
    kind, reason = _unwrapped_raise(e)
    # Keyed on that chain, not on where the raise came from: the clause explains
    # why the text quotes a chained exception rather than the tree's own, so a
    # raise with nothing chained (a TORCH_CHECK, which pybind translates at the
    # same boundary into a plain RuntimeError) gets the text without it.
    boundary = "" if reason is e else " (through the guard tree's pybind boundary)"
    # str(reason) is user text and the report is read back with splitlines().
    return " ".join(f"{kind}: {reason}{boundary}".splitlines())


def _raised_line(index: int, e: Exception) -> str:
    return f"  [{index}] <guard check raised {_raise_text(e)}>"


class _GuardScope(enum.Enum):
    """Which dict the artifact's global guards resolve names against."""

    # Never serialized: the guards still hold the tracing process's globals.
    CAPTURED = "captured"
    # A live scope the load path re-rooted the guards at, e.g. a function
    # load's f_globals.
    SUPPLIED = "supplied"
    # Rebuilt for this load: the globals the graph lifted, the graph's freshly
    # imported module aliases, the backend id, and whatever an f_globals= was
    # merged over them. That merge copies, so a name bound in the f_globals
    # after the load is invisible to these guards.
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


def _module_namespace_name(scope: dict[str, Any]) -> str | None:
    """The name of the module whose live namespace ``scope`` is, else None."""
    # Identity, not __name__ alone: a GraphModule's forward is exec'd into a
    # private copy of its codegen globals, and a caller can hand over any dict.
    name = scope.get("__name__")
    module = sys.modules.get(name) if isinstance(name, str) else None
    return name if module is not None and vars(module) is scope else None


def _guard_source_globals(output_graph: "OutputGraphGuardsState") -> set[str]:
    """The global names a kept guard's own originating_source IS."""
    # A guard certifies its own source, not the object that source is reached
    # through, so a CHAINED source does not count: a TENSOR_MATCH on
    # G['D']['a'] certifies that one item, while the name a load can substitute
    # is D, whose every other key the graph would then read live and unchecked.
    # get_global_source_name would walk such a source up to D.
    # guard_on_key_order is deliberately not unioned in, even though a
    # dict-order check roots a global: guard_filter_fn never prunes that set, so
    # a name only it contributes is precisely a name no surviving guard checks
    # the value of. On the default aot_compile filter, which drops every global
    # guard, an iterated global dict is exactly that shape.
    # Narrower than the serialized global_scope for the same reason: the
    # serializer also fills that from a ShapeEnvSource guard's shape_env_sources
    # and from DUPLICATE_INPUT's source_b -- names whose value no guard checks,
    # and which DUPLICATE_INPUT records before its optimizer-source early return,
    # so an optimizer-rooted pair records one with no guard installed at all.
    from .source import GlobalSource

    return {
        guard.originating_source.global_name
        for guard in output_graph.guards
        if isinstance(guard.originating_source, GlobalSource)
    }


def _recorded_guard_globals(guards_state: "GuardsState") -> set[str]:
    """Every global name the kept guards read at check time."""
    # Wider than their originating_sources: the serialized global_scope is the
    # serializer's own record of the names the kept guards resolve, so it also
    # carries a DUPLICATE_INPUT's source_b and a cpp-form SHAPE_ENV guard's
    # shape_env_sources. A SHAPE_ENV guard installed as a Python lambda reads
    # its G['NAME'] operands from the same scope, and none of them reaches
    # global_scope -- shape_env_sources is filled from the cpp code parts alone
    # -- so they are recovered from the lambda's own text.
    # A filter that drops SHAPE_ENV drops them too: the builder records
    # shape_code_parts on the save pass only, which runs over the kept guards.
    names = set(guards_state.output_graph.global_scope)
    shape_code_parts = guards_state.shape_code_parts
    if shape_code_parts is not None and shape_code_parts.python_fallback:
        for expr in shape_code_parts.python_code_parts.exprs:
            names.update(_SHAPE_GUARD_GLOBAL_RE.findall(expr))
    return names


@dataclass
class AOTCompiledFunction:
    _artifacts: CompileArtifacts
    _guard_check_enabled: bool = True
    _extra_globals: dict[str, object] | None = None
    # Guard-only scope, held by reference; kept apart from _extra_globals so
    # nothing in it reaches the compiled bytecode but the names _serve re-takes,
    # which are the ones __post_init__ certified.
    _guard_globals: dict[str, object] | None = None
    # Which of the three scopes the artifact's guards resolve against, so a
    # guard failure can say something actionable about the dict the name was
    # looked up in. Not init-settable: it stays CAPTURED unless __post_init__
    # itself resolves a scope, which it does only for a load that has guards
    # left to re-root.
    _guard_scope: _GuardScope = dataclasses.field(
        init=False, default=_GuardScope.CAPTURED
    )
    # Why no live guard scope could be resolved from model.forward, with the
    # advice for that shape; read only by _missing_global_hint, whose
    # RECONSTRUCTED template splices it after "rebuilt because" and appends
    # ", or pass ... a guard_globals= scope", so every reason must end in
    # "; <imperative advice>" for the two verb phrases to read as parallel.
    _forward_not_resolved_reason: str | None = None
    # Whether a kept guard is rooted at a user global; False until a load
    # decides it. Arms the live-value pick, and deserialize's fallback warning.
    _has_global_guards: bool = dataclasses.field(init=False, default=False)
    # The globals a kept guard's own source IS (not one reached only through a
    # sub-path of it), armed only for a supplied live scope. _serve re-takes
    # them out of _guard_globals before every call: the guards read that dict by
    # reference while the bytecode's globals are a dict of their own, so leaving
    # them at their load-time values would let a rebind the guards ACCEPT
    # compute with whatever the load happened to see.
    _live_global_names: tuple[str, ...] = dataclasses.field(init=False, default=())
    # The rebuilt callable, set by __post_init__ (never absent on a live
    # artifact); a declared field rather than an attribute setattr'd onto the
    # instance. Out of repr and eq because the field has no default: leaving it
    # in either makes both raise AttributeError on an instance __post_init__
    # abandoned -- check_compatibility and forward_callable both raise there --
    # which is what a traceback rendering frame locals would report instead of
    # the real failure. It is a per-instance FunctionType, so it is not
    # equality state either.
    fn: Callable[..., Any] = dataclasses.field(init=False, repr=False, compare=False)

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

    def _live_guard_manager(self) -> "GuardManagerWrapper":
        # Narrowing for pyrefly, not a live check: __post_init__ always leaves a
        # populated guard_manager (only serialize() nulls it, on a copy).
        if self._artifacts.guard_manager is None:
            raise AssertionError("live artifact must have a guard_manager")
        return self._artifacts.guard_manager

    def guard_check(self, *args: Any, **kwargs: Any) -> bool:
        f_locals = self.prepare_f_locals(*args, **kwargs)
        return self._live_guard_manager().check(f_locals)

    def __post_init__(self) -> None:
        from .package import load_guard_manager, load_guards_state

        self._artifacts.check_compatibility()

        extra_globals = self._extra_globals
        guards_state = None
        guard_scope = self._guard_globals
        if self._artifacts.guard_manager is None:
            guards_state = load_guards_state(self._artifacts.guards_state)
            output_graph = guards_state.output_graph
            # The wide set: every name the kept guards read, which also gates the
            # seeding below. Enough to arm the pick, but not to decide what it
            # takes -- see _guard_source_globals. The builtins dict key rides
            # along whether or not a guard reads it, so it is not evidence.
            builtins_key = output_graph.name_of_builtins_dict_key_in_fglobals or ""
            recorded_globals = _recorded_guard_globals(guards_state) - {builtins_key}
            # Dynamo's own __import_* aliases are not user globals: a rebuilt
            # scope carries every recorded one freshly imported, so a guard
            # rooted at one resolves there and neither half of the warning the
            # fallback path logs applies to it.
            aliases = set(self._artifacts.runtime_env.import_sources)
            self._has_global_guards = bool(recorded_globals - aliases)
            if guard_scope is not None:
                # A live scope: a name it lacks must fail the guard rather than
                # fall back to the value serialized with the artifact.
                self._guard_scope = _GuardScope.SUPPLIED
                if self._has_global_guards:
                    # The narrow set, because a passing guard is the only thing
                    # that certifies a live value is the one the graph was
                    # compiled for. The builtins dict key is left out as well: a
                    # guard rooted at it certifies the live dict itself, which
                    # the graph's globals never copy. Not intersected with the
                    # scope: a name it does not bind yet fails the guard rooted
                    # at it, so nothing is served on that name until a caller
                    # who populates the dict after the load binds it -- and then
                    # the re-read is what the graph gets, not the value the
                    # artifact was traced with.
                    certified = _guard_source_globals(output_graph) - {builtins_key}
                    self._live_global_names = tuple(sorted(certified))
                    # Bound at load as well as re-taken per call in _serve: a
                    # certified name the bytecode reads but the graph never
                    # lifted -- a global the forward mutates or returns -- is
                    # in external_refs and in no serialized scope, so
                    # forward_callable's check fails unless it is bound here or
                    # by f_globals. Taken with .get, as in _serve, so a del
                    # racing the load cannot raise KeyError out of it.
                    live = {
                        n: v
                        for n in certified
                        if (v := guard_scope.get(n, _UNBOUND)) is not _UNBOUND
                    }
                    extra_globals = {**(extra_globals or {}), **live}

        self.fn = self._artifacts.runtime_env.forward_callable(
            self._artifacts.backend_id,
            self._artifacts.compiled_fn,
            extra_globals=extra_globals,
        )

        if guards_state is not None:
            if guard_scope is None:
                self._guard_scope = _GuardScope.RECONSTRUCTED
                guard_scope = self.fn.__globals__
            # Seeded AFTER forward_callable, never before: on the default path this
            # IS fn.__globals__, and PyFunction_New caches __builtins__ at creation,
            # so the __builtins__ written below cannot rewire the bytecode's lookups.
            # The builtins-dict key below is an ordinary global and does; see there.
            self._seed_guard_scope(guard_scope, guards_state)
            self._artifacts.guard_manager = load_guard_manager(
                guards_state,
                self._artifacts.original_code,
                guard_scope,
            )

    def _seed_guard_scope(
        self, guard_scope: dict[str, Any], guards_state: "GuardsState"
    ) -> None:
        # Dynamo mints __import_* aliases, a __builtins_dict___N key and the
        # ___unnamed_scope_<id>_c<n> key of an inlined frame's globals into the
        # TRACING process's globals and roots guards at them; a process that only
        # loads never traced. Each name is gated on the artifact showing a kept
        # guard reads it -- the aliases and the unnamed-scope key on every name
        # the kept guards read, the builtins key on the deserialized guards' own
        # roots -- because this writes into a scope that may be a user module's live
        # namespace and installs no CleanupHook. A binding this process already
        # had is left alone: a wrong binding fails the guard rather than passing
        # it. The one value replaced is one this load itself put there, in the
        # builtins branch below.
        from .output_graph import get_builtins_dict
        from .source import get_global_source_name
        from .utils import CleanupHook

        output_graph = guards_state.output_graph
        # A scope lacking any name the kept guards read fails with a KeyError on
        # G[...], whichever channel reads it, so that whole set gates the aliases
        # and the unnamed-scope key: the same set the arming in __post_init__
        # starts from, before it subtracts the aliases, which a seeding must
        # not. Only a caller-supplied scope can be missing one --
        # forward_callable imports every recorded alias and spreads used_globals.
        guarded_globals = _recorded_guard_globals(guards_state)
        # That set cannot gate the builtins key -- the serializer writes it into
        # global_scope whether or not a guard reads it -- so match the
        # deserialized guards' own roots instead. The wider channels never root
        # at this key: load_builtin_from_argval is the only site that mints a
        # source under it, and only for a callable builtin.
        builtins_key = output_graph.name_of_builtins_dict_key_in_fglobals
        sources = [guard.originating_source for guard in output_graph.guards]
        roots = {get_global_source_name(source) for source in sources}
        seeds_builtins = bool(builtins_key) and builtins_key in roots
        # The scopes the key is seeded into. Two dicts can hold the recording it
        # replaces, and only on the default path are they one: the guard scope,
        # which the deserialized guard tree reads, and fn.__globals__, which the
        # generated bytecode subscripts -- visited only when it is a different
        # object with a recording to replace. Every scope's __builtins__ is
        # validated here, before anything below writes or disowns, so a refused
        # load leaves the caller's scopes exactly as it found them; the check
        # reads only __builtins__ and has to run whenever the key is seeded, not
        # only when it is derived, or a pre-bound key would let a bad binding load.
        scopes: list[tuple[dict[str, Any], str]] = []
        snapshot = None
        if seeds_builtins and builtins_key is not None:
            snapshot = self._artifacts.runtime_env.used_globals.get(builtins_key)
            # Name the dict: get_builtins_dict would otherwise raise a bare
            # AttributeError out of Dynamo internals. A module's namespace by
            # its module, since the module load path resolves one from
            # model.forward and no parameter names it; a hand-built dict by
            # the parameter it arrived by, where load_compiled_function
            # forwards one dict as both, so a dict that arrived by both routes
            # is named by the public one -- guard_globals is not in that
            # signature. The TYPE and not the value -- a repr on a load
            # failure path runs user code.
            arrived_as_f_globals = (
                self._guard_globals is None
                or self._guard_globals is self._extra_globals
            )
            param = "f_globals" if arrived_as_f_globals else "guard_globals"
            namespace = _module_namespace_name(guard_scope)
            where = param if namespace is None else f"vars({namespace})"
            scopes.append((guard_scope, where))
            if self.fn.__globals__ is not guard_scope and snapshot is not None:
                scopes.append((self.fn.__globals__, "f_globals"))
            for scope, where in scopes:
                bound = scope.get("__builtins__", builtins.__dict__)
                if not isinstance(bound, (dict, types.ModuleType)):
                    raise TypeError(
                        f"{where}['__builtins__'] must be a dict or a module, got "
                        f"{type(bound).__name__}"
                    )
        # No disown for an alias: import_source and CompilePackage._install_global
        # bind one by plain dict assignment, and only install_global_unsafe
        # creates a CleanupHook, never for an alias.
        for alias, module_name in self._artifacts.runtime_env.import_sources.items():
            if alias in guarded_globals and alias not in guard_scope:
                guard_scope[alias] = importlib.import_module(module_name)
        # The unnamed-scope key embeds id() of a dict in the tracing process, so
        # no live scope carries it. Where the graph lifted a value read through
        # that dict, used_globals recorded the dict under the key, and that
        # recording is what the rebuilt scope hands the guard, so a supplied
        # scope is handed the same object; a key used_globals lacks names a
        # namespace the graph only specialized on, and there is nothing to bind.
        # install_global_by_id binds through install_global_unsafe, so a compile
        # in this process may still own a leftover it left here; disowned as the
        # builtins key is below, whether or not this load binds it, so the hook
        # cannot delete a binding these guards read once its code is collected.
        used_globals = self._artifacts.runtime_env.used_globals
        for name in guarded_globals:
            if name.startswith(_UNNAMED_SCOPE_PREFIX) and name in used_globals:
                CleanupHook.disown(guard_scope, name)
                if name not in guard_scope:
                    guard_scope[name] = used_globals[name]
        if not seeds_builtins or builtins_key is None:
            return
        # A pre-reset compile's CleanupHook may still own this name even when we
        # leave its value alone; drop it so it can't delete the binding once
        # collected.
        CleanupHook.disown(guard_scope, builtins_key)
        # The dict this key resolves to has to be the LIVE builtins: the guard
        # rooted here is an ID_MATCH on a builtin, so a snapshot goes on passing
        # after that builtin is rebound. That is what this load binds when the
        # scope has no __builtins__; a caller who pre-binds one chooses the dict
        # the guard watches. Two bindings reach a snapshot, and the
        # second is this load's own -- when the generated bytecode reads this key,
        # get_runtime_env records a pickle-filtered COPY of the tracing builtins
        # under it and forward_callable spreads that copy into fn.__globals__,
        # which on the default path IS the guard scope. Re-derive over that
        # recording; a binding from anywhere else is a value this process chose and
        # stays.
        # Re-deriving it also decides what the bytecode subscripts, since that
        # recording exists only because the bytecode reads this key, and it is
        # filtered for picklability alone: a builtin the tracing process had and
        # this one lacks stops being readable -- a kept guard on that name reports
        # it, and without one the bytecode raises KeyError.
        # Each dict is derived from its OWN __builtins__, so a guard-only scope
        # cannot rewire what the bytecode resolves.
        for scope, _ in scopes:
            if builtins_key in scope and (
                snapshot is None or scope[builtins_key] is not snapshot
            ):
                continue
            # forward_callable builds fn.__globals__ as a plain dict, so unlike an
            # exec'd module namespace it carries no __builtins__ to derive from.
            if "__builtins__" not in scope:
                scope["__builtins__"] = builtins.__dict__
            scope[builtins_key] = get_builtins_dict(scope)

    def _missing_global_hint(self, *, forward: str | None = None) -> str:
        """Advice for a guard that failed on a global its scope does not define,
        worded for the scope the guards were actually resolved against. Returns a
        bare sentence; a caller that continues a line of its own adds the
        separator. ``forward`` names the model's instance attribute, passed only
        when the guards hold the very dict it resolves to -- resolved by the load
        or supplied by the caller -- and honoured only in the SUPPLIED branch."""
        if self._guard_scope is _GuardScope.RECONSTRUCTED:
            rebuilt = (
                "a guarded global is missing from the scope rebuilt from the artifact"
            )
            if self._forward_not_resolved_reason is not None:
                # A module load takes no f_globals=, which is the function load's
                # parameter; _load_aot_compiled_module takes only the bytes.
                return (
                    f"{rebuilt}. That scope was rebuilt because "
                    f"{self._forward_not_resolved_reason}, or pass "
                    "AOTCompiledModel.deserialize a guard_globals= scope that "
                    "carries the name."
                )
            return (
                f"{rebuilt}; load with an f_globals= that is a complete live "
                "scope carrying the name -- normally vars(mod) for the module "
                "mod that defined the function, which is usually not the module "
                "doing the loading -- so the guard can resolve it."
            )
        if self._guard_scope is _GuardScope.SUPPLIED:
            # SUPPLIED implies a scope; a module's namespace is named by its
            # module, since a module load that resolved it from model.forward had
            # no caller's dict to send the reader back to.
            namespace = _module_namespace_name(self._guard_globals or {})
            named = "" if namespace is None else f", here vars({namespace})"
            where = (
                # A rebind to a Dynamo wrapper is resolved THROUGH it
                # (_resolve_guard_scope), so the sentence sends the reader through too.
                f"the globals of the function {forward} resolves to, seen through "
                "the wrappers torch.compile, torch._dynamo.disable, run and "
                "optimize return and through any functools.wraps'd "
                "torch._dynamo.external_utils function to the function they wrap, "
                f"which is the dict the guards hold{named}"
                if forward is not None
                else f"the live scope this artifact was loaded against{named}"
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
        if self._guard_check_enabled and not self.guard_check(*args, **kwargs):
            f_locals = self.prepare_f_locals(*args, **kwargs)
            debug_info = self._live_guard_manager().check_verbose(f_locals)
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
        return self._serve(*args, **kwargs)

    def _serve(self, *args: Any, **kwargs: Any) -> Any:
        """Run the graph, re-reading the globals a kept guard certifies.

        A global a kept guard's own source IS -- not one reached only through a
        sub-path of it, which the guard does not certify -- is re-read from the
        guard scope here, so a rebind that scope took and the guards accepted (a
        same-metadata swap under a ``TENSOR_MATCH``, which checks metadata, not
        values) is what the graph computes with. Every other global keeps the
        value the bytecode's globals were built with at load, a container a
        guard reaches only through a sub-path such as ``G['D']['a']`` included:
        that guard certifies the one item, not the container's other members. A
        name the scope does not bind is skipped rather than deleted, so it keeps
        whatever it last held -- the serialized value if the scope never bound
        it -- and, unless the check is disabled, the guard rooted at it refuses
        the call. A global the graph itself rebinds is re-read from the scope on
        the next call too: the replayed ``STORE_GLOBAL`` lands in the bytecode's
        globals, not in the scope, so the stored value is one no guard certified
        and the scope's is what the check before the call just passed. That is a
        deliberate trade-off: a forward that accumulates into a guarded global
        (``global W; W = W * 2``) serves the scope's value on every call, where
        eager, whose store lands in the dict its guards read, counts up. Leaving
        a stored name out of the re-read instead would serve the stored value
        after a rebind of the scope the guards accepted -- the check certifying
        one value while the graph reads another, which is the stale read this
        re-read removes -- and writing the store back into the scope is a
        behaviour neither load path has. Under the default filter, which keeps
        no global guard, nothing is re-read and such a store accumulates in the
        bytecode's globals, unchecked.

        Only a load handed a live scope arms this. An artifact compiled in this
        process has none, so its globals stay at the values the capture copied
        while its guards read the module dict they were rooted in: the same two
        dicts, left to diverge as they already did rather than widened here.

        An artifact that opted out of the check re-reads the same names with
        nothing certifying them and serves whatever it finds, a value a kept
        guard would have rejected included, or, for a name the scope no longer
        binds, the last value read: the opt-out is unsafe by construction, and
        holding those names at their load-time values instead would be no more
        checked, only stale.
        The re-read is not atomic with the guard check before it, so a rebind
        landing between the two is served unchecked -- the same window an eager
        compiled frame has between guard evaluation and LOAD_GLOBAL. The write
        lands in this artifact's own ``fn.__globals__``, which every call of it
        shares, so two threads serving one loaded artifact against scopes that
        differ, or against a scope rebound concurrently, race on that dict and
        one can run the graph on the value the other just wrote; a caller who
        needs isolation loads the artifact once per thread, since each load
        builds its own dict."""
        if self._live_global_names and (scope := self._guard_globals) is not None:
            f_globals = self.fn.__globals__
            for name in self._live_global_names:
                value = scope.get(name, _UNBOUND)
                if value is not _UNBOUND:
                    f_globals[name] = value
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
        runtime_env = state["runtime_env"]
        state["runtime_env"] = dataclasses.replace(
            runtime_env,
            bytecode=SerializedCode.from_code_object(runtime_env.bytecode),
            used_globals={
                name: _picklable_unnamed_scope(value)
                if name.startswith(_UNNAMED_SCOPE_PREFIX) and isinstance(value, dict)
                else value
                for name, value in runtime_env.used_globals.items()
            },
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
        forward_not_resolved_reason: str | None = None,
    ) -> "AOTCompiledFunction":
        """Rebuild a compiled function from ``serialize()`` output.

        ``f_globals`` is MERGED over the scope reconstructed from the serialized
        bytecode, so a name it omits still resolves to the baked-in value.
        ``guard_globals`` REPLACES the guard scope with no such fallback -- a name
        it lacks fails the guard, and an EMPTY dict is an empty scope rather than
        "no scope" -- and the load WRITES into it, seeding the recorded aliases,
        builtins-dict key and unnamed-scope key a kept guard is rooted at without
        replacing a name it already binds, so pass the dict those should land in.
        Supplying it also binds the certified globals into the bytecode's globals
        at load and arms the per-call re-read ``_serve`` performs: a global a kept
        guard's own source IS -- not one reached only through a sub-path of it,
        which the guard does not certify, and apart from the recorded
        ``__builtins_dict___N`` key, excluded by name -- is taken from that dict at
        load and on every call, one name at a time, so the graph reads only what a
        guard certifies and never a global whose guard a filter dropped; the
        load-time bind is what lets a ``guard_globals``-only load succeed where the
        serialized scope lacks such a global, and it is written over ``f_globals``:
        a certified name both bind is taken from ``guard_globals`` from the load
        onwards, not only from the first call. The public loader passes one dict
        as both, so it is unaffected. Passing neither resolves global guards
        against the scope rebuilt from the artifact, where a rebinding in this
        process is invisible.
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
                # NB: the is_global clause dropping every global guard is
                # deliberate, not a gap: narrowing it would need every load to
                # supply a scope binding every global a kept guard reads.
                # Callers who need one guarded pass their own guard_filter_fn.
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
        graph = backend_input.graph_module.graph
        device_type = _collapse_device_types(_graph_device_types(graph))
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


def _resolve_guard_scope(
    model: torch.nn.Module,
) -> tuple[dict[str, Any] | None, str | None]:
    # From model.forward, not the model: for a hooked module get_traced_fn would
    # return Module._wrapped_call_impl and nn.Module's namespace. An nn.Module
    # forward is refused for the same reason -- get_traced_fn rewrites an
    # nn.Module argument to THAT module's forward, rooting the guards in its
    # namespace with nothing raising. Plain assignment cannot produce one
    # (nn.Module.__setattr__ files a Module under _modules, and the class
    # attribute keeps winning the lookup), but object.__setattr__ can; refusing
    # here also keeps get_traced_fn's Module branch, whose hook reads can raise
    # AttributeError on an uninitialized module, off this path entirely.
    from torch._dynamo.eval_frame import _static_getattr, innermost_fn

    forward = model.forward
    # Describe forward AS GIVEN, not what the unwrap below reached, in a bounded
    # way that avoids dumping the entire module repr (functools.partial embeds
    # the module's full repr).
    forward_type = type(forward).__name__
    forward_qualname = getattr(forward, "__qualname__", "")
    described = (
        f"{type(model).__name__}.forward ({forward_type}"
        f"{f' named {forward_qualname}' if forward_qualname else ''})"
    )
    if isinstance(forward, torch.nn.Module):
        return None, (
            f"{described} is an nn.Module, which get_traced_fn would rewrite to "
            "that module's forward, rooting the guards in its defining namespace; "
            "bind a plain function or bound method as model.forward instead"
        )
    # innermost_fn follows the _torchdynamo_orig_callable chain the wrappers
    # torch.compile, torch._dynamo.disable, run and optimize return carry. A
    # compile that wrapped its target in external_utils.wrap_inline
    # (config.wrap_top_frame, or a forward defined under torch/) ends that chain
    # on wrap_inline's inner, which only forwards to what it wraps, so a function
    # OWNING external_utils' dict is followed to its __wrapped__, one hop per
    # stacked compile. Keyed on the dict's identity, not __module__ (functools.wraps
    # copies it): torch._dynamo.decorators' wraps'd wrappers are the root frame a
    # capture traces, so they stay put. A bound method owns no __globals__ for
    # _static_getattr (object.__getattribute__) to read, so one is never hopped.
    resolved = innermost_fn(forward)
    while _static_getattr(resolved, "__globals__") is vars(external_utils):
        wrapped = _static_getattr(resolved, "__wrapped__")
        if wrapped is None:
            # wrap_dunder_call_ctx_manager's inner skips functools.wraps on purpose,
            # as does wrap_inline_with_error_on_graph_break's wrapper, which only
            # compile_wrapper._torchdynamo_inline holds: no public API binds it.
            hopped = resolved is not forward
            via = "resolves through a Dynamo wrapper to" if hopped else "is"
            return None, (
                f"{described} {via} a torch._dynamo.external_utils function with "
                "no __wrapped__ to see through to the forward it wraps -- the "
                "wrapper torch._dynamo.error_on_graph_break, patch_dynamo_config, "
                "disable_nested_graph_breaks and override_cudagraphs return skips "
                "functools.wraps; bind the forward that decorator wrapped as "
                "model.forward instead"
            )
        resolved = wrapped
    # torch.compile(mod).forward wraps the module's DISPATCH, not the forward
    # the capture traced: the module itself under config.wrap_top_frame or a
    # skip rule (OptimizedModule._initialize hands it to wrap_inline), which
    # only the hop can reach, or otherwise the bound nn.Module.__call__, whose
    # __func__ owns torch.nn.modules.module's namespace and fails the namespace
    # test below. So does mod.forward = other.__call__ with no wrapper in
    # front, a shape a capture did record that namespace for: refusing forgoes
    # that agreement rather than seed a process-wide torch namespace.
    if isinstance(resolved, torch.nn.Module):
        return None, (
            f"{described} resolves through a Dynamo wrapper to an nn.Module, "
            "the module's dispatch rather than its forward; bind that module's "
            "forward, or a wrapper over the forward rather than over the "
            "module, as model.forward instead"
        )
    # get_traced_fn raises RuntimeError on a callable that is neither a
    # function nor a method, and its __self__ branch returns __func__
    # unchecked, so a C-implemented bound method (a tensor's sum) raises
    # AttributeError there or on the __globals__ read.
    try:
        traced_fn = convert_frame.get_traced_fn(resolved)[0]
        scope = traced_fn.__globals__
    except (RuntimeError, AttributeError):
        if resolved is forward:
            return None, (
                f"get_traced_fn cannot resolve {described} to a Python function; "
                "make model.forward a plain function or bound method so its own "
                "globals are used instead"
            )
        # torch.compile over a functools.partial or a tensor method wraps it in
        # wrap_inline (no source file, not a function), so the unwrap lands on
        # it; the cannot-resolve advice above would describe the
        # compile_wrapper, a plain function that resolves fine.
        return None, (
            f"{described} resolves through a Dynamo wrapper to an instance of "
            f"{type(resolved).__name__}, which get_traced_fn cannot resolve to "
            "a Python function; bind a plain function or bound method as "
            "model.forward instead"
        )
    # A forward that resolves to a function torch itself defines -- the
    # nn.Module.forward a module never overrode, _LazyGraphModule._lazy_forward
    # before a real recompile, the bound Module._wrapped_call_impl a wrapper over
    # the module's dispatch hops to -- owns a torch module's namespace, which a
    # load must neither root guards in nor seed: the seeding is permanent and
    # installs no CleanupHook. The test is the namespace, not the function's
    # __module__, which functools.wraps copies; a GraphModule's forward is
    # exec'd into a private per-instance copy of its codegen globals, no
    # module's namespace, and resolves.
    namespace = _module_namespace_name(scope)
    if namespace is not None and namespace.partition(".")[0] == "torch":
        hopped = resolved is not forward
        via = "resolves through a Dynamo wrapper to" if hopped else "resolves to"
        what = traced_fn.__qualname__
        wrapped = _static_getattr(traced_fn, "__wrapped__")
        if wrapped is not None:
            # functools.wraps copied the wrappee's __qualname__ onto the wrapper
            # (a class-body @torch.compiler.wrap_numpy forward binds external_utils'
            # wrap), which would read "X resolves to X". co_qualname is 3.11+.
            code = traced_fn.__code__
            what = (
                f"{getattr(code, 'co_qualname', code.co_name)}, a functools.wraps'd "
                f"wrapper over {getattr(wrapped, '__qualname__', what)}"
            )
        return None, (
            f"{described} {via} {what}, whose globals are {namespace}'s namespace, "
            "a torch module a load neither roots guards in nor seeds; bind the "
            "module's own forward, defined outside torch, as model.forward instead"
        )
    return scope, None


def _unwrap_optimized_module(model: torch.nn.Module) -> torch.nn.Module:
    # isinstance, not getattr(model, "_orig_mod", model): _orig_mod is a
    # registrable submodule name, and unwrapping to a child would run the
    # parent's graph against the child's parameters. A loop, because wrappers
    # nest: OptimizedModule.__reduce__ rebuilds a deepcopied or unpickled
    # wrapper without the metadata innermost_fn follows, so torch.compile
    # wraps it again instead of collapsing onto the module.
    from torch._dynamo.eval_frame import OptimizedModule

    while isinstance(model, OptimizedModule):
        model = model._orig_mod
    return model


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


# The redirect's artifact takes the module as its first argument only on the
# _wrapped_call_impl branch of OptimizedModule._initialize; the wrap_inline branch
# it takes for config.wrap_top_frame or a skipped model.forward closes over the
# module instead, so measured, passing it there fails len(L['args']) == 1. What
# decides that skip is the FILE model.forward is DEFINED in -- _forward_has_skip_rule
# is trace_rules.check(mod.forward) -- and not what the class is: measured, a
# subclass of nn.Linear that does not override forward is skipped too. No list
# spells that rule: check_file consults LEGACY_MOD_INLINELIST before MOD_SKIPLIST,
# so a file inside a skipped directory can still be inlined (measured, QuantStub's
# forward under torch/ao/ is), which is why the clause names the predicate itself.
_REDIRECT_CALL = (
    "call the artifact it returns with the module as its first argument if you "
    "define forward yourself; if forward is instead defined in a file dynamo "
    "skips, as torch.nn's stock modules are -- inheriting it unoverridden counts, "
    "since dynamo decides on the file forward is defined in rather than on the "
    "class, and torch._dynamo.eval_frame.OptimizedModule._forward_has_skip_rule("
    "model) is the exact test -- or config.wrap_top_frame is set, that capture "
    "wrapped the module rather than its __call__ and the artifact takes only the "
    "forward arguments"
)


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
    # and the artifact that produces saves but cannot be reloaded. A dropped
    # backward hook also leaves the forward result alone and changes the
    # gradients, so it needs different wording than "the result may differ".
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
        # LazyModuleMixin registers its initializer as a forward pre-hook
        # (nn/modules/lazy.py:178), so every uninitialized lazy module lands here,
        # and for one OptimizedModule._initialize pins the wrapper's forward to
        # _call_lazy_check (eval_frame.py:547-549), which carries no aot_compile.
        # The pin outlives the initializer, so the redirect needs a module already
        # materialized when torch.compile saw it, not just a materialized one.
        lazy = ""
        if inspect.getattr_static(model, "_initialize_hook", None) is not None:
            lazy = (
                " -- but not for this module yet: it carries a lazy initializer, "
                "and while it does torch.compile pins the wrapper's forward to "
                "_call_lazy_check, which has no aot_compile, so call the module "
                "once to materialize it and then wrap it again"
            )
        log.warning(
            "%s has %s registered; the AOT compiled forward calls %s.forward "
            "directly, so those hooks do NOT run and its result may differ from "
            "eager -- to keep them, AOT compile torch.compile(model).forward "
            "instead, which traces __call__ and runs them%s; %s",
            type(model).__name__,
            ", ".join(forward_hooked),
            type(model).__name__,
            lazy,
            _REDIRECT_CALL,
        )
    if backward_hooked:
        log.warning(
            "%s has %s registered; the AOT compiled forward calls %s.forward "
            "directly, so those hooks do NOT run and the gradients it produces "
            "may differ from eager while the forward result does not -- to keep "
            "them, AOT compile torch.compile(model).forward with compiled "
            "autograd enabled around the capture -- only the private "
            "torch._dynamo.compiled_autograd._enable does that, since the "
            "config flag of the same name is not read on this path; %s. Note "
            "that the artifact saves but cannot be reloaded",
            type(model).__name__,
            ", ".join(backward_hooked),
            type(model).__name__,
            _REDIRECT_CALL,
        )
    # Eager dispatches through type(model).__call__ while the artifact calls what
    # forward compiled to, so an overridden __call__ is dropped just like a hook
    # -- and no hook dict records it, so the lists above see nothing to report.
    # The probe is that same type lookup, walked over the MRO: an instance
    # attribute named __call__ is not an override, because CPython resolves a
    # special method on the type, so eager ignores it too and the artifact
    # matches.
    # fx.GraphModule installs a wrapper as its per-instance class's __call__ on
    # every GraphModule.recompile, so a bare lookup reports an override for
    # every one of them, ExportedProgram.module() included. With no class
    # __call__ to wrap, the wrapper only prettifies tracebacks and delegates to
    # super(cls, obj), so skip every class carrying one and take the next
    # __call__ the MRO offers, which is the one that delegation reaches. The
    # wrapper is told apart by what it is rather than by the class it sits on:
    # recompile installs it on whatever type(self) is at the time, so a
    # __class__ swap (FSDP, replicate, parametrize all rebind it after the
    # trace) followed by another recompile leaves two wrapper classes on the
    # MRO, while a real __call__ assigned onto the per-instance class afterwards
    # sits on the very class FX wrapped and still has to count. Skipping those
    # classes rather than starting past them keeps an override on a base AHEAD
    # of them visible; a GraphModule subclass that defines __call__ carries it
    # on a base BEHIND them, where the delegation finds it. A wrapper whose
    # cls_call was set delegates there instead of to super -- functional_export
    # hooks a hooked root's wrapper that way -- so it is not skipped. The
    # wrapper is call_wrapped, a closure GraphModule.recompile mints anew on
    # every run, so no one function is there to compare by identity; its def
    # site (module and qualname) is, along with the _WrappedCall it delegates
    # to, which recompile installs on the same class. Moving either turns this
    # back into a warning on every GraphModule, which
    # test_aot_compile_module_fx_call_wrapper_is_not_warned_about catches.
    # _LazyGraphModule defers that recompile to the first call or code access,
    # so until then no class on its MRO owns a __call__ and the walk lands on
    # nn.Module's: silent for want of a wrapper rather than by skipping one.
    fx_module = torch.fx.graph_module
    fx_wrapper = (fx_module.__name__, "GraphModule.recompile.<locals>.call_wrapped")

    def is_fx_wrapper(c: type) -> bool:
        fn, wrapped = vars(c)["__call__"], vars(c).get("_wrapped_call")
        def_site = (getattr(fn, "__module__", None), getattr(fn, "__qualname__", None))
        return (
            def_site == fx_wrapper
            and isinstance(wrapped, fx_module._WrappedCall)
            and wrapped.cls_call is None
        )

    # nn.Module defines __call__ in its own vars, so the default is unreachable,
    # and only keeps a StopIteration out of a warning helper.
    call = next(
        (
            vars(c)["__call__"]
            for c in type(model).__mro__
            if "__call__" in vars(c) and not is_fx_wrapper(c)
        ),
        torch.nn.Module.__call__,
    )
    if call is not torch.nn.Module.__call__:
        log.warning(
            "%s overrides __call__; the AOT compiled forward runs what "
            "%s.forward compiled to, so that override does NOT run and its "
            "result may differ from eager -- to keep it, AOT compile "
            "torch.compile(model).forward instead, which traces __call__; %s",
            type(model).__name__,
            type(model).__name__,
            _REDIRECT_CALL,
        )


# Parameters as (name, kind, default id), co_freevars, closure cell ids.
_BindingKey = tuple[tuple[tuple[str, int, int], ...], tuple[str, ...], tuple[int, ...]]


def _binding_key(artifacts: CompileArtifacts) -> _BindingKey:
    # What prepare_f_locals reads, with defaults and cells by identity. Signature
    # equality is unusable here: Parameter.__eq__ takes bool() of
    # `default == default`, which raises for a tensor default.
    env, params = artifacts.runtime_env, artifacts.signature.parameters.values()
    return (
        tuple([(p.name, p.kind, id(p.default)) for p in params]),
        env.bytecode.co_freevars,
        tuple([id(cell) for cell in env.closure or ()]),
    )


def _same_results(
    prior: tuple[weakref.ref[AOTCompiledFunction], ...],
    results: tuple[AOTCompiledFunction, ...],
) -> bool:
    return len(prior) == len(results) and all(w() is r for w, r in zip(prior, results))


@dataclass
class AOTCompiledModel:
    """A module's forward compiled for several calls, with dispatch over them.

    Private and experimental, like ``_aot_compile`` which builds one. Only
    ``compiled_results`` serializes; ``deserialize`` needs the model again.
    ``compiled_results`` must hold at least one result: ``aot_compile_module``
    refuses an empty list, and neither the constructor nor ``deserialize``
    checks, so an empty one is the caller's error.

    ``compiled_results`` may be edited between calls. A call judges the list
    it began with, and results that bind alike -- equal parameter names, kinds
    and default objects, the same closure cells, as every result of one
    ``_aot_compile`` has -- share one binding of the call; whether they do is
    decided again whenever the list's contents change.

    Dispatch walks ``compiled_results`` in order and serves the first result
    whose guard check accepts the call. One exit of ``check()`` refuses without
    evaluating the tree -- the no-tensor-aliasing exit of the recursive
    dict-tag fast path in ``GuardManager::check_nopybind``, reached only with
    ``use_recursive_dict_tags_for_guards`` on -- so if no check accepted, every
    result is checked once more before dispatch gives up; a result whose guards
    would pass can therefore be outranked by a later result whose first check
    accepted. When neither pass accepts, the call is served by the first result
    that opted out through ``disable_guard_check()``, from any index, and only
    when none did does it raise the ``No AOT compiled graph matched this call``
    report below. That is all the flag does here: ``check()`` never reads it,
    so an opted-out result is scanned and re-checked like any other and is
    served in index order when its check accepts, and on the strength of its
    opt-out alone only after both the scan and the re-check found no match and
    no checked input's guard tree raised while being evaluated; one opt-out
    replaces the ``No AOT compiled graph matched this call`` error for the
    whole model. A tree that raises rejects nothing, so a raise from a checked
    input withholds the opt-out, and the call raises the no-match
    ``RuntimeError`` with the raise chained as its ``__cause__`` and the report
    naming the input that raised. A raise beside an input whose guards did
    match is served over: the matching graph runs, and the raise is logged once
    per ``(input index, exception type)`` per model on the
    ``torch._dynamo.aot_compile`` logger, starting over when ``compiled_results``
    changes. A ``KeyboardInterrupt`` or ``SystemExit`` out of a guard tree is
    never read as an answer and propagates.

    When no result matches and none opted out, the call raises ``RuntimeError``
    with a report headed ``No AOT compiled graph matched this call``: one line
    per compiled result quoting the verbose parts of the guard that refused it,
    or, for a result
    whose guards accept the call on the report's own evaluation after not
    accepting it in dispatch, a ``<guards did not accept this call in dispatch
    and accepted it here: ...>`` explanation in place of any guards, or, for a
    result whose refusal quotes nothing -- an accessor that answered false with
    no parts, or a guard that raised with a blank message -- ``<guard check
    failed without naming a guard>``; one
    ``For [i, j]:`` line per distinct missing-global hint naming the entries
    whose guards failed on a global the process does not define; and -- when
    some checked tree reached an answer, or the artifact holds no input at all
    -- the advice to add a ``ModelInput`` or check which guards
    ``guard_filter_fn`` kept. When every rejection that advice rests on followed
    a raise from its own tree, it names and quotes those raises and says to fix
    them first; when no checked tree ever answered and two or more trees raised,
    a line saying every guard tree raised replaces it, unless an opted-out
    result's line has already said the raise withheld it; a single raiser's own
    line already says as much. When some checked input's guard tree raised, that
    exception is the ``__cause__`` of the ``RuntimeError`` rather than the
    exception the caller sees, so a caller
    catching the tree's own type (``SystemError`` for a leaf that returned with
    an error set, ``RuntimeError`` for a ``TORCH_CHECK``) catches the report
    instead.
    """

    model: torch.nn.Module
    compiled_results: list[AOTCompiledFunction]
    # The results last judged, weakly so a dropped one is not kept alive, and
    # whether one bind of a call serves them all; the default is the verdict
    # over no results. One field so one store publishes both and a reader never
    # sees one list's contents beside another's verdict. A hint, not a lock:
    # the last writer wins, and a call that finds the contents changed decides
    # again.
    _binding_verdict: tuple[tuple[weakref.ref[AOTCompiledFunction], ...], bool] = (
        dataclasses.field(default=((), False), init=False, compare=False, repr=False)
    )
    # The results warn_swallowed last logged about and the (index, exception type)
    # pairs it logged, so a hot loop over a broken artifact logs once per defect.
    # Per model, not torch._logging.warning_once, whose cache is process-global.
    # Kept beside the results because a changed compiled_results can put another
    # artifact at a warned-about index; judged where the warning is logged, since
    # a one-result model never re-decides the binding verdict. One field, as above.
    _warned: tuple[
        tuple[weakref.ref[AOTCompiledFunction], ...], set[tuple[int, str]]
    ] = dataclasses.field(
        default_factory=lambda: ((), set()), init=False, compare=False, repr=False
    )

    def _binds_alike(self, results: tuple[AOTCompiledFunction, ...]) -> bool:
        prior, shared = self._binding_verdict
        if _same_results(prior, results):
            return shared
        key = _binding_key(results[0]._artifacts) if results else None
        shared = key is not None and all(
            _binding_key(result._artifacts) == key for result in results[1:]
        )
        self._binding_verdict = (tuple(weakref.ref(r) for r in results), shared)
        return shared

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # compiled_results is public, so read it once: every stage below judges
        # the results this call began with, on the binding decided over them.
        results = tuple(self.compiled_results)
        # check() ignores _guard_check_enabled, which only the last resort and
        # the report read, so scan every result.
        raised: dict[int, Exception] = {}
        # Indices whose LAST evaluation reached no answer: no guard to quote.
        unanswered: set[int] = set()
        # Indices that ever reached an answer, which a ModelInput could have covered.
        answered: set[int] = set()
        # Answered with no raise of their own on record (see warn_swallowed).
        trusted: set[int] = set()
        # Per-result bindings, filled on first use, kept for the re-check and report.
        bound: dict[int, dict[str, object]] = {}
        # Whether results that bind alike reuse the first one's binding (a bind
        # costs more than a check()), decided once a second result is reached so
        # a call the first result serves pays nothing for it; the reuse rests on
        # check() only reading the f_locals it is handed.
        shared: bool | None = None

        def accepts(i: int, result: AOTCompiledFunction) -> bool:
            nonlocal shared
            # prepare_f_locals stays outside the try, so a call the signature cannot
            # bind surfaces as bind_locals' TypeError, not as a tree that did not match.
            f_locals = bound.get(i)
            if f_locals is None:
                if bound and shared is None:
                    shared = self._binds_alike(results)
                if shared:
                    f_locals = bound[0]
                else:
                    f_locals = result.prepare_f_locals(self.model, *args, **kwargs)
                bound[i] = f_locals
            try:
                answer = result._live_guard_manager().check(f_locals)
            except Exception as e:
                # Keep going so another result can still match.
                raised[i] = e
                unanswered.add(i)
                return False
            if answer:
                return True
            # Recorded on a rejection only: an accept serves and builds no report.
            # Measured 0.06us for the four against a 1.3us check().
            unanswered.discard(i)
            answered.add(i)
            if i not in raised:
                trusted.add(i)
            return False

        def warn_swallowed(served: int) -> None:
            # A raise is not a rejection, so it says nothing about the result
            # that did answer -- but no report is built on this path, so nothing
            # else records it. Also where the result that answered IS the one that
            # raised: its accept is the only answer about this call, and vetoing
            # it would not contain the stale relational state below, which a scan
            # raise leaves for the NEXT call, with no raise on record at all.
            over, warned = self._warned
            if not _same_results(over, results):
                warned = set()
                self._warned = (tuple(weakref.ref(r) for r in results), warned)
            for i, e in raised.items():
                kind, reason = _unwrapped_raise(e)
                if (i, kind) in warned:
                    continue
                warned.add((i, kind))
                if results[i]._guard_check_enabled:
                    advice = (
                        f"Fix or drop input [{i}]: a tree that raises rejects "
                        f"nothing, and {_STALE_AFTER_THROW}."
                    )
                else:
                    # The last resort serves an opted-out result whatever its
                    # guards say, so a stale rejection costs it nothing; what the
                    # raise costs it is the match the scan can never find.
                    advice = (
                        f"Input [{i}] opted out of guard checks, but a tree that "
                        "raises never matches in the scan, so its graph is "
                        "reachable only through the last resort, which a raise "
                        "from any enabled tree withholds."
                    )
                log.warning(
                    "AOT compiled input [%d]'s guard check raised %s: %s; "
                    "dispatch served [%d] rather than propagating it. %s",
                    i,
                    kind,
                    reason,
                    served,
                    advice,
                )

        for i, result in enumerate(results):
            if accepts(i, result):
                if raised:
                    warn_swallowed(i)
                # The guard manager already passed; go through _serve rather
                # than result(), which would re-run the ~1us guard eval on this
                # hot dispatch path. _serve costs one Python frame plus a scope
                # probe and a dict write per certified global instead (about 0.4us
                # with one name, under the ~0.85us guard eval), and is what hands
                # the graph the globals that check just accepted. Gating that frame out per site is
                # not worth it: _serve is the only caller of the raw fn, and a
                # site that got the gate wrong would serve a stale global with
                # every guard passing.
                return result._serve(self.model, *args, **kwargs)
        # One exit of check() refuses without running the tree: a tag-safe root's
        # no-tensor-aliasing fast check (GuardManager::check_nopybind). It
        # disarms that root, so a second check() runs the tree it skipped. With
        # use_recursive_dict_tags_for_guards off (the default) no root is tag
        # safe and this pass re-runs trees that genuinely failed, lambda guards
        # included, bumping the failing node's _fail_count a second time; about
        # 1us per result, accepted.
        for i, result in enumerate(results):
            if accepts(i, result):
                if raised:
                    warn_swallowed(i)
                return result._serve(self.model, *args, **kwargs)
        # A result that opted out via disable_guard_check() accepts anything, but
        # only after both passes failed to find a real match and no tree whose
        # guards someone did ask about raised -- even if a later pass answered: a
        # rejection after a throw can be about the relational guard state the
        # throw left stale (see warn_swallowed), not about this call. A raise from
        # the opted-out result itself withholds nothing: nobody wanted its answer.
        if not any(results[i]._guard_check_enabled for i in raised):
            for i, result in enumerate(results):
                if not result._guard_check_enabled:
                    if raised:
                        warn_swallowed(i)
                    return result._serve(self.model, *args, **kwargs)
        report = self._no_match_report(
            results, raised, unanswered, answered, trusted, bound
        )
        if raised:
            # `raised` is in recording order, so this chains the first index that
            # raised, not always the raiser the advice names: they differ when an
            # opted-out result raised first, whose line quotes no exception text.
            raise RuntimeError(report) from next(iter(raised.values()))
        # Not `from None`: an ordinary no-match must not suppress an exception
        # this call was made while handling.
        raise RuntimeError(report)

    def _no_match_report(
        self,
        results: tuple[AOTCompiledFunction, ...],
        raised: dict[int, Exception],
        unanswered: set[int],
        answered: set[int],
        trusted: set[int],
        bound: dict[int, dict[str, object]],
    ) -> str:
        """A report naming every compiled input and what its guard check said or raised.

        ``results`` and ``bound`` are the results ``AOTCompiledModel.__call__``
        judged and the f_locals it judged them on, one per result, so the report
        explains the same call rather than a fresh one."""
        lines = [
            "No AOT compiled graph matched this call. Tried "
            f"{len(results)} compiled input(s):"
        ]
        # Hint text -> the entries it is for, in first-seen order: entries whose
        # advice reads alike share a line, whatever scope each resolves against,
        # and one whose advice differs keeps its own rather than being read
        # another entry's. Two unnamed supplied dicts word alike and so share a
        # line; the sentence names no dict either way.
        hinted: dict[str, list[int]] = {}
        resolved: dict[str, Any] | None = None
        tried_forward = False
        # An opted-out result is reported at all only because a raise vetoed the
        # last resort above; without one it is served and there is no report.
        # One read, before check_verbose runs user code that could opt a result
        # out under the loop: the entries and the raiser they name must agree.
        enabled = [result._guard_check_enabled for result in results]
        raiser = next((i for i in raised if enabled[i]), None)
        # An entry that answered in either dispatch pass rejected this call, so a
        # ModelInput could have covered it even where its line below is a raise.
        coverable = any(results[i]._guard_check_enabled for i in answered)
        trusted_rejection = any(results[i]._guard_check_enabled for i in trusted)
        withheld = False
        for i, result in enumerate(results):
            if not enabled[i]:
                # Nobody asked about this result's guards, so quoting them -- a
                # raise out of them included -- would name the wrong thing, and no
                # ModelInput covers what kept it from serving: the raise above.
                # Decided before the raise below, so an opted-out tree that raised
                # is reported once, as the opt-out it is.
                lines.append(
                    f"  [{i}] <opted out of guard checks; withheld because "
                    f"[{raiser}]'s guard check raised>"
                )
                withheld = True
                continue
            if i in unanswered:
                # No rejection to quote, so report the raise rather than evaluate
                # the tree a third time, whose answer would not be the one dispatch
                # acted on. An entry that raised and THEN answered is not here: its
                # rejection is quoted below, and its raise survives only where it
                # is the one the chain carries -- the FIRST index that raised.
                lines.append(_raised_line(i, raised[i]))
                continue
            try:
                reason = result._live_guard_manager().check_verbose(bound[i])
            except Exception as e:
                # check_verbose runs paths check() did not (a repr of a user
                # object, for one); one entry raising must not cost the others.
                lines.append(_raised_line(i, e))
                continue
            if reason.result:
                lines.append(
                    f"  [{i}] <guards did not accept this call in dispatch and "
                    "accepted it here: a guard that does not answer consistently, or "
                    "guarded state that changed between those evaluations>"
                )
                continue
            parts = reason.verbose_code_parts
            # Collapse every separator splitlines() reads the report back on.
            # Done here, not in get_verbose_code_part: the recompile logs consume
            # the same parts and are out of this report's scope.
            joined = " ".join("; ".join(parts).splitlines())
            if not joined.strip():
                # A failing accessor can answer false with no parts to quote, and
                # a guard that raised quotes str(exc), which can be blank.
                lines.append(f"  [{i}] <guard check failed without naming a guard>")
                continue
            if any(map(_names_a_missing_global, parts)):
                forward: str | None = None
                if result._guard_scope is _GuardScope.SUPPLIED and not tried_forward:
                    tried_forward = True
                    # Resolving forward runs user code: get_traced_fn formats a
                    # forward it refuses into its error, and that repr can raise past
                    # what _resolve_guard_scope catches. The report must still arrive.
                    try:
                        resolved, unresolved = _resolve_guard_scope(self.model)
                    except Exception as exc:
                        # The type only: str(exc) can run the same repr again.
                        unresolved = f"resolving it raised {type(exc).__name__}"
                    if resolved is None:
                        log.debug(
                            "the no-match report's hint names no %s.forward: %s",
                            type(self.model).__name__,
                            unresolved,
                        )
                if resolved is not None and resolved is result._guard_globals:
                    # Named as the instance attribute: the guards hold the dict it
                    # resolves to, whether the load resolved that dict from it or the
                    # caller passed the same one, and a rebound instance reads another
                    # function's dict.
                    forward = f"this {type(self.model).__name__} instance's forward"
                hint = result._missing_global_hint(forward=forward)
                hinted.setdefault(hint, []).append(i)
            lines.append(f"  [{i}] {joined}")
        for hint, at in hinted.items():
            lines.append(f"For [{', '.join(map(str, at))}]: {hint}")
        if withheld:
            lines.append(
                f"[{raiser}]'s raise, not a guard failure, is what withheld "
                "the opted-out input(s) above; fix or drop that artifact."
            )
        elif raiser is not None:
            # Same advice with no opt-out to withhold: a tree that raised has to
            # be fixed whether or not its raise also cost the caller a graph.
            lines.append(
                f"[{raiser}]'s guard check raised while checking this call; fix "
                "or drop that artifact."
            )
        # An artifact holding no inputs at all -- which deserialize() accepts --
        # has no entry to answer, and adding an input is exactly the advice for it.
        if coverable or not results:
            advice = (
                "Add a ModelInput covering this call, or check whether "
                "guard_filter_fn kept a guard this call cannot satisfy -- both "
                "belong to the process that compiles the artifacts, which need "
                "not be the one that loaded them."
            )
            if coverable and not trusted_rejection:
                # Keyed on what dispatch recorded, not on the entry lines: the
                # re-check may have printed a raise or an accept instead. Named
                # and quoted here because nothing else on the report carries
                # such a raise: the entry line quotes the rejection, the raiser
                # line names the FIRST enabled raiser, which need not be one of
                # these, and the chain carries the first raise of all. Bracketed
                # as on the entry line and joined with a semicolon: the quoted
                # text is arbitrary user text that may hold commas, and
                # _raise_text has a parenthetical of its own. Enabled only:
                # nobody asked about an opted-out tree's guards, and its withheld
                # line already blames the raiser. Under the gate above no trusted
                # entry is enabled, so raised[i] is populated for every entry
                # here, by the recording in accepts().
                untrusted = [
                    f"[{i}] <{_raise_text(raised[i])}>"
                    for i in sorted(answered)
                    if results[i]._guard_check_enabled
                ]
                plural = "s" if len(untrusted) > 1 else ""
                advice += (
                    f" Fix the raise{plural} out of {'; '.join(untrusted)} first: "
                    "every rejection this advice rests on followed a raise from "
                    f"its own tree, and {_STALE_AFTER_THROW}."
                )
            lines.append(advice)
        if len(raised) > 1 and not withheld and not coverable:
            # `not coverable`: no checked tree answered, so every entry line above
            # is a raise and the advice above is off. Not beside a withheld line,
            # which has already said what the raise cost, and only where the
            # raiser line above names one raiser of several: for a single entry
            # it already says all of this.
            lines.append(
                "Every guard tree raised while checking this call; the reasons "
                "above are those raises, not guards this call failed."
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
    def deserialize(
        cls,
        model: torch.nn.Module,
        data: bytes,
        *,
        guard_globals: dict[str, Any] | None = None,
    ) -> "AOTCompiledModel":
        """Rebuild the compiled forward of ``model`` from ``serialize()`` output.

        ``model`` may be the module itself or the wrapper ``torch.compile``
        returned for it, which is unwrapped to the module that was traced.

        Guards on globals are evaluated, by reference, against the live
        ``__globals__`` of the function ``model.forward`` resolves to, and the
        compiled bytecode reads the globals serialized with the artifact except
        for the names a kept guard's own source IS -- not a global reached only
        through a sub-path of it, and never the recorded ``__builtins_dict___N``
        key -- which are re-taken from that live dict on every call. So a value
        the graph reads live is one a passing guard certifies, every other global
        is the one it was traced with, and a guarded global the live dict lacks
        fails the guard rather than falling back to the serialized value. The
        re-take writes into the artifact's own ``fn.__globals__``, which every
        call of it shares, so two threads serving one loaded model while either
        rebinds a guarded global race on that dict, with or without the GIL; a
        caller who needs isolation loads once per thread.
        Rebinding a guarded global after the load is therefore what the graph
        computes with once the guards accept it, and the certification is only as
        strong as the guard's type: a kept ``TENSOR_MATCH`` accepts a same-metadata
        swap, checking metadata and not values, and a root ``TYPE_MATCH`` on a
        container checks its type, not the members the graph reads through it.
        Loading also MUTATES that dict: a recorded ``__import_*`` alias a kept guard
        still reads, that builtins key when a guard source names it, and the
        ``___unnamed_scope_*`` key of an inlined frame's globals when the graph
        lifted a value through it -- bound to the dict serialized with the artifact,
        since the key embeds an ``id()`` from the tracing process that no live
        namespace holds -- are inserted (never overwriting an existing key) so
        guards rooted at them resolve in a process that never traced.

        A symbolic-shape guard on a global with a dynamic dim resolves its
        operands in that live dict as well, whether it installs as a Python
        lambda (the default) or as a C++ guard under
        ``enable_cpp_symbolic_shape_guards``, and counts as a global guard for
        the warning below; the graph still keeps the tensor serialized with it,
        since a size check certifies no value.

        There is no live scope only when ``model.forward`` does not resolve to a
        Python function of its own: ``get_traced_fn`` cannot resolve it, or it is
        an ``nn.Module``, as given or reached through a Dynamo wrapper (the
        module's dispatch under ``config.wrap_top_frame``), or it is a
        ``torch._dynamo.external_utils`` function with no ``__wrapped__`` to see
        through (``torch._dynamo.error_on_graph_break``'s wrapper), or it resolves
        to a function torch itself defines, whose globals are a torch module's
        namespace (the module's dispatch otherwise, ``torch.compile(mod).forward``,
        included), or the target a Dynamo wrapper is seen through to does not
        resolve itself; guards then resolve against the scope rebuilt from the
        artifact, where they check nothing useful, and a guard rooted at any
        global but those aliases and that key warns to say so, naming the cause.

        The function ``model.forward`` resolves to is the one bound as ``forward``
        seen through Dynamo's own wrappers -- the ones ``torch.compile``,
        ``torch._dynamo.disable``, ``run`` and ``optimize`` return, and any
        ``functools.wraps``'d ``torch._dynamo.external_utils`` function, of which
        ``torch.compiler.wrap_numpy`` is the one a caller applies -- but not
        through any other wrapper the caller applied: a ``functools.wraps``'d
        decorator over it, in the class body or rebound on the instance, resolves
        to the decorator's own function, so the scope is the decorator's module.
        That is the scope a capture of the decorated forward records as well --
        Dynamo traces the decorator as the root frame -- so an artifact captured
        through the same decorator loads and reads that module's guarded globals
        live, and one captured from the undecorated forward fails its global
        guards there, with ``KeyError on G['NAME']`` and a hint naming that
        module; load an artifact onto the forward it was captured from.
        ``wrap_numpy`` rebound on the instance is seen through, so the scope is
        the forward's own module -- what an artifact captured from the
        undecorated forward, the only artifact that shape can load, recorded;
        applied in the class body it binds a method, which is not seen through
        and resolves to ``wrap`` in ``external_utils``' own namespace, refused as
        any torch namespace is.

        ``guard_globals``, when supplied, is that scope instead of anything
        resolved from ``model.forward``, so a caller who wants neither the live
        read nor the write passes its own dict; it is seeded, bound into the
        bytecode's globals at load, and re-read from on the same terms.

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
        forward_not_resolved_reason = None
        scope = guard_globals
        if scope is None:
            scope, forward_not_resolved_reason = _resolve_guard_scope(model)

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
                        forward_not_resolved_reason=forward_not_resolved_reason,
                    )
                )
        # Model-level, so warn once here rather than once per ModelInput.
        if forward_not_resolved_reason is not None and any(
            result._has_global_guards for result in compiled_results
        ):
            log.warning(
                "no live guard scope could be resolved from model.forward, so "
                "global guards on this artifact resolve against the scope "
                "rebuilt from the serialized bytecode instead, where they check "
                "nothing useful: one on a global the graph lifted is compared "
                "against the value serialized with it and cannot fail, and one "
                "on a global that scope does not carry cannot be satisfied, so "
                "the call will report no match. %s.",
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
