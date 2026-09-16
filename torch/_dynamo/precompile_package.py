"""
Ahead-of-time precompilation of a callable into MANY graphs: the multi-graph
counterpart of ``torch.compile(fn, fullgraph=True).aot_compile(...)``. Every
frame Dynamo produces while the caller's calls run -- the entry
frame, each ``torch_dynamo_resume_in_*`` continuation created by a graph break,
and every recompiled variant of each -- is captured into one serializable
artifact on top of CompilePackage.

Everything here is internal; the session that drives it and the
``torch.compiler.precompile.capture(..., tracer=DynamoTracer())`` entry point build on
these in later commits. This is distinct from
``torch._dynamo.config.caching_precompile``, which caches ``torch.compile``
artifacts transparently without an explicit capture.

Capture is by execution, and the caller drives it: the session hands back a
callable, the caller invokes it with real inputs inside their own loop, and
every frame Dynamo produces is recorded. Runtime guards stay intact during
capture; ``guard_filter_fn`` applies only to the serialized copy, and every
dropped guard is reported in ``PrecompileSummary.dropped_guards``.

    with torch.compiler.precompile.capture(
        step, artifact_path="m.py", cache_path="m.cache", backend="inductor"
    ) as cap:
        y1 = cap(model, x1)  # runs step(model, x1), returns its result
        y2 = cap(model, x2)  # exercises another variant

    # later, in a fresh process
    compiled = torch.compiler.precompile.load("m.py", "m.cache")
    with compiled, torch.no_grad():
        compiled(model, x1)

The caller's calls ARE the capture: each ``cap(...)`` runs the callable for
real, returns its result, and records every frame, break continuation and
guarded variant it exercises. ``precompile.accumulate`` is the same model, rewriting the artifact on every
call instead of once at block exit.

Calls run with the grad mode the caller sets -- capture does not force
``no_grad()`` or ``enable_grad()``. ``training=True`` lowers the backward
eagerly so the artifact carries one and a served output can be backpropagated.
No loss is needed for that: the joint trace synthesizes tangents from the
forward outputs' own metadata.

Live capture retains every runtime guard, so later examples trigger the same
recompilations as ordinary ``torch.compile``. ``guard_filter_fn`` applies only
to the serialized copy. If serialization drops a configuration-dependent guard,
the artifact is refused by default rather than written with variants whose
dispatch would be ambiguous after load. ``invariants`` writes a readable report
that separates, per frame, the guards holding in EVERY variant from the ones
that differed: the first are preconditions the artifact is only valid under,
the second are what tell its graphs apart. Guards from different frames are not
comparable -- an entry frame guards its arguments, a resume frame guards
whatever crossed the break -- so the intersection is per frame.

Capture is by execution: a resume function only exists once the frame ahead of
it has actually run, so every variant must be exercised. Whatever you do not
run is not in the artifact, and ``summary().complete`` means complete only for
the observed capture, not for every possible input to the callable. A captured
call that raises marks the session incomplete even if caller code catches it.

Know these before relying on an artifact in production:

* An inference artifact is the default: the caller runs the calls under
  ``torch.no_grad()``. For a training artifact pass ``training=True``, which
  traces with grad on and lowers the backward eagerly -- without it, AOTAutograd
  defers the backward to the first ``.backward()`` call, so a grad-enabled
  capture that never makes one records no backends and cannot be written.
* A non-tensor argument, and any value that crosses a graph break, is guarded
  by equality, so an int/bool/str argument or a break coming from ``.item()``
  yields an artifact that only serves calls reproducing those exact values.
  ``summary().wont_generalize`` lists them; exercise every value you need to
  serve with a ``cap(...)`` call, or expect poor coverage on new data.
  ``dynamic=True`` helps with shapes but not with pinned values.
* Identity guards cannot be serialized, so precompiling gives up on noticing
  that a guarded object was rebound. ``summary().dropped_guards`` is the
  authoritative list. ``risky_dropped_guards`` includes every drop observed to
  distinguish captured variants plus a lint for configuration-like sources; it
  is still not a proof for unobserved deployments. See ``_is_risky_drop``. The
  public ``torch.compiler.precompile`` facade rejects the RISKY subset by
  default. Refusing every drop is opt-in: every model drops the identity guards
  precompile cannot serialize, so ``require_no_dropped_guards=True`` refuses
  essentially every real artifact. Some models trip the lint on
  library internals: measured on stock models, torchvision resnet18 and
  mobilenet_v3 report none, timm's ViT reports one (a re-exported
  ``torch._assert``) and transformers' Qwen2 reports 33 built from a two-layer
  config, 55 for the pretrained 24-layer, of which only the
  attention-implementation registry looks genuinely config-selected. Report
  counts are per model, not per library: torchvision's efficientnet_b0 reports
  2 and timm's swin reports 5, one of which is a real config slot. Audit the
  list before relying on the relaxed dropped-guard default, and before
  relaxing the risky-drop rail on top of it.
* Some models do not capture yet. For example, T5 raises ``PackageError: Cannot
  find module for code <code object __init__`` from ``_get_code_source``, which
  is byte-identical to base and which plain ``caching_precompile`` also raises.
* The model must live in an importable module. Source is checksummed, so a
  class defined in ``__main__`` or a REPL cannot be loaded elsewhere.
* ``install()`` writes compiled and resume functions into module globals, but
  guarded dispatch is scoped to the isolated compile region owned by the
  returned callable. Call the returned object rather than another instance of
  the same class. Multiple loaded artifacts can share entry, inner, and resume
  code objects without taking each other's entries; ``unload()`` removes only
  its own region and the globals it still owns.

This wraps CompilePackage, which is the low-level component and is not meant to
be used directly.

The public surface is ``torch.compiler.precompile.capture(...)``, a caller-driven
capture used as a context manager: the caller's own calls inside the block drive
the capture, and the ``(python_code, cache)`` artifact is written to the given files
when the block exits (its default ``tracer=DynamoTracer()`` records many calls;
``tracer=MakeFxTracer()`` produces a self-contained Python source artifact from one
call); ``torch.compiler.precompile.accumulate(...)``, the counterpart that rewrites
the files after every call; and ``torch.compiler.precompile.load``.
The helpers in this module, including the capture session, implement that surface
and remain internal. All of it is distinct from ``torch._dynamo.config.caching_precompile``,
which caches ``torch.compile`` artifacts transparently without an explicit
capture block.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import threading
from typing import Any, TYPE_CHECKING

import torch
import torch._functorch.config as functorch_config
from torch.compiler._precompile_types import (
    FrameInvariants,
    GuardFact as _GuardFact,
    PrecompileSummary,
)

from .convert_frame import CatchErrorsWrapper
from .exc import PackageError
from .package import CompilePackage


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from .convert_frame import ConvertFrameReturn

    from .eval_frame import OptimizeContext
    from .package import _BackendId
    from .types import CacheEntry, DynamoFrameType, GuardFilterEntry
    from .variables.builder import FrameStateSizeEntry

import contextvars
import logging

from .guards import CheckFunctionManager


log = logging.getLogger(__name__)


# Built once: config.patch() allocates a class and a ContextVar each time it is
# called, and this runs on every frame Dynamo compiles for a package.
_ALLOW_EMPTY_GRAPHS = torch._dynamo.config._make_closure_patcher(
    allow_empty_graphs=True
)


# Not a public surface -- see the module docstring. This exists so `from ...
# import *` in a debugging session pulls the entry points rather than every
# private helper, and so linters do not flag them as unused.
__all__ = [
    "FrameInvariants",
    "PrecompileSession",
    "PrecompileSummary",
    "precompile_capture",
]


# Depth per context so overlapping sessions on one thread patch once and
# restore once; a worker thread starts at zero and patches for itself.
_CAPTURE_CONFIG_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "_CAPTURE_CONFIG_DEPTH", default=0
)


_CAPTURE_CONFIG_STACK: contextvars.ContextVar[contextlib.ExitStack | None] = (
    contextvars.ContextVar("_CAPTURE_CONFIG_STACK", default=None)
)


@contextlib.contextmanager
def _capture_config(training: bool) -> Iterator[None]:
    # Backends serialize into the artifact rather than the process-local
    # inductor cache. AOTAutograd lowers the backward lazily on the first
    # .backward(), so a training capture that never makes one forces it eager.
    depth = _CAPTURE_CONFIG_DEPTH.get()
    if depth == 0:
        functorch_patch: dict[str, Any] = {
            "bundled_autograd_cache": True,
            # AOTAutogradCache refuses to KEY a graph it cannot address soundly
            # -- a graph calling anything outside its allowlist -- and a refusal
            # means it never saves, so the bundled artifact precompile needs is
            # never recorded and the capture ends with nothing to serialize.
            # That gate asks whether the key tells this graph's behaviour apart
            # from another's, which a precompile artifact does not depend on: it
            # is addressed by backend id and pinned to one torch build, so fall
            # back to a nonce key rather than declining, as
            # torch._dynamo.aot_compile and aot_compile_joint_with_descriptors
            # already do.
            "bypass_autograd_cache_key": True,
        }
        if training:
            functorch_patch["force_non_lazy_backward_lowering"] = True
        stack = contextlib.ExitStack()
        stack.enter_context(functorch_config.patch(functorch_patch))
        # allow_empty_graphs keeps an empty graph as a compiled frame so its
        # guards reach the artifact. It also extends the lifetime of objects the
        # frame holds: with it on, a weakref callback on a value the frame
        # captured does not fire when the caller drops its reference
        # (test/dynamo/test_repros.py ReproTests.test_weakref_callback).
        try:
            stack.enter_context(torch._dynamo.config.patch(allow_empty_graphs=True))
        except BaseException:
            stack.close()
            raise
        _CAPTURE_CONFIG_STACK.set(stack)
    _CAPTURE_CONFIG_DEPTH.set(depth + 1)
    try:
        yield
    finally:
        remaining = _CAPTURE_CONFIG_DEPTH.get() - 1
        _CAPTURE_CONFIG_DEPTH.set(remaining)
        if remaining == 0:
            stack = _CAPTURE_CONFIG_STACK.get()
            _CAPTURE_CONFIG_STACK.set(None)
            if stack is not None:
                stack.close()


class _AllowEmptyGraphsCallback(CatchErrorsWrapper):
    """The package's Dynamo callback, compiling its frames with allow_empty_graphs.

    An uncovered no-op branch must become a guarded variant rather than Dynamo's
    ordinary eager-only SkipFrame, or one fallback call permanently skips that
    frame and serving() can no longer detect it. Patched here as well as in
    _capture_config so the package's own frames get it even when the callback
    runs outside a capture-config scope.
    """

    def __call__(
        self,
        frame: DynamoFrameType,
        cache_entry: CacheEntry | None,
        frame_state: dict[str, int | FrameStateSizeEntry],
    ) -> ConvertFrameReturn:
        revert = _ALLOW_EMPTY_GRAPHS()
        try:
            return super().__call__(frame, cache_entry, frame_state)
        finally:
            revert()


def _compose_with_default(
    user: Callable[[Sequence[GuardFilterEntry]], Sequence[bool]],
) -> Callable[[Sequence[GuardFilterEntry]], Sequence[bool]]:
    """AND a caller's filter with the default rather than replacing it.

    ``default_guard_filter_fn`` is not a default in the "sensible starting point"
    sense -- it is what drops the identity guards that CANNOT be serialized at
    all. Replacing it means a caller who wanted to drop three of their own guards
    silently re-admits every unserializable one, and the failure surfaces as
    "ID_MATCH guard cannot be serialized" in frames that have nothing to do with
    their filter. A custom filter can only ever want to drop MORE, so composing
    is the only reading that makes sense.
    """

    def composed(entries: Sequence[GuardFilterEntry]) -> Sequence[bool]:
        base = default_guard_filter_fn(entries)
        chosen = user(entries)
        if len(chosen) != len(entries):
            raise ValueError(
                f"guard_filter_fn returned {len(chosen)} decisions for "
                f"{len(entries)} guards; it must return one per entry."
            )
        return [bool(a) and bool(b) for a, b in zip(base, chosen)]

    return composed


def default_guard_filter_fn(
    guard_entries: Sequence[GuardFilterEntry],
) -> Sequence[bool]:
    """
    Drop the guard types that cannot be serialized, and keep everything else.

    Read this before trusting an artifact. The unserializable set is exactly the
    IDENTITY guards -- ID_MATCH, FUNCTION_MATCH, CLOSURE_MATCH, MODULE_MATCH,
    NN_MODULE, CLASS_MATCH, DICT_VERSION, WEAKREF_ALIVE -- so precompiling
    inherently gives up on noticing that a guarded object was REBOUND to a
    different object of the same shape. Most such guards are on modules and
    builtins and are stable in practice, but one on a global holding a function
    is not: rebind it between capture and load and the artifact serves the graph
    traced against the old one, with no error.

    Keeping these makes serialization raise for essentially every function, so
    every drop is recorded with its source name in
    ``PrecompileSummary.dropped_guards``. A caller is not refused for having
    them, because requiring none would refuse essentially every model. The rail
    that is on is the risky-drop
    lint, and a lint is not a proof. See ``risky_dropped_guards``.
    """
    unsupported = CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
    return [
        g.guard_type not in unsupported
        and not any(d in unsupported for d in g.derived_guard_types)
        for g in guard_entries
    ]


@functools.singledispatch
def _entry_fn_of(fn: object) -> Callable[..., object]:
    if not callable(fn):
        raise TypeError(f"expected a callable or nn.Module, got {type(fn).__name__}")
    if not hasattr(fn, "__code__"):
        raise TypeError(
            f"expected a function or nn.Module, got {type(fn).__name__}, which "
            f"has no __code__ for Dynamo to capture or to load an artifact "
            f"onto. Pass partial.func for a functools.partial, or obj.__call__ "
            f"for an object that only defines __call__."
        )
    return fn  # type: ignore[return-value]


@_entry_fn_of.register(torch.nn.Module)
def _(fn: torch.nn.Module) -> Callable[..., object]:
    forward = fn.forward
    if not hasattr(forward, "__code__"):
        raise TypeError(
            f"{type(fn).__name__}.forward is a {type(forward).__name__}, which "
            f"has no __code__ for Dynamo to capture or to load an artifact onto. "
            f"Binding it in __init__ -- self.forward = functools.partial(...) -- "
            f"shadows the class method and lands here; keep forward a method."
        )
    return forward


def _identify_graph(gm: torch.fx.GraphModule) -> str:
    """Name a graph well enough to find it, from inside a backend.

    The module class alone does not: every graph a model produces reports the
    same one, so a capture that recompiled nine graphs at serve time said
    "GraphModule" nine times. The compile id keys tlparse, the backend id keys
    the artifact, and the first node carrying a stack trace names the user line
    -- including, for a continuation, the resume frame Dynamo minted for it,
    which is the only thing that tells one break in a chain from another.
    """
    parts: list[str] = []
    compile_id = torch._guards.CompileContext.current_compile_id()
    if compile_id is not None:
        parts.append(f"compile id {compile_id}")
    backend_id = gm.meta.get("backend_id") or getattr(gm, "_backend_id", None)
    if backend_id is not None:
        parts.append(f"backend id {backend_id}")
    for node in gm.graph.nodes:
        if node.op not in ("placeholder", "output") and node.stack_trace:
            first = node.stack_trace.strip().splitlines()[0].strip()
            parts.append(f"first traced at {first}")
            break
    return f" Graph: {'; '.join(parts)}." if parts else ""


class _PrecompileBackend:
    """Give one explicit session its own Dynamo cache identity."""

    def __init__(
        self, backend: str, keep_graphs: bool = False, serving: bool = False
    ) -> None:
        inner = torch._dynamo.lookup_backend(backend)
        self._torchdynamo_orig_backend = inner
        self.backend_ctx_ctor = getattr(
            inner, "backend_ctx_ctor", contextlib.nullcontext
        )
        # Rendering a subgraph as source needs the graph, which only exists
        # here. Kept only where something will render it (see the caller):
        # retaining deepcopies every compiled graph for the session.
        self._keep_graphs = keep_graphs
        self.graphs: dict[str, tuple[torch.fx.GraphModule, list[Any]]] = {}
        # Serving an INSTALLED artifact answers a guard miss by compiling,
        # because a frame reachable only through the frame evaluator has no
        # other way to run. Counted, and said out loud once per graph: an
        # artifact that quietly compiles more of itself on every batch looks
        # exactly like one that is serving.
        self.serving = serving
        self.serve_time_compiles = 0

    def __call__(self, gm: torch.fx.GraphModule, inputs: list[torch.Tensor]) -> Any:
        if self.serving:
            self.serve_time_compiles += 1
            log.warning(
                "precompile: serving compiled a NEW graph -- no captured variant "
                "matched this call, so the artifact is serving less than it was "
                "measured to. Recapture with an example that covers it.%s",
                _identify_graph(gm),
            )
        if self._keep_graphs:
            backend_id = gm.meta.get("backend_id")
            if backend_id is not None and str(backend_id) not in self.graphs:
                # Deep-copy before the inner backend runs: inductor lowering
                # mutates the graph it is handed, and a rendered copy has to be
                # the graph Dynamo produced, not the leftovers.
                placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
                # Render against the placeholders' FAKES, never the real inputs
                # below: compile_fx re-fakifies real tensors into a fresh symbol
                # set and dedups by value, which silently unifies a batch dim
                # with any other dim of the same size.
                fakes = [
                    n.meta.get("example_value", n.meta.get("val")) for n in placeholders
                ]
                if all(f is not None for f in fakes):
                    self.graphs[str(backend_id)] = (copy.deepcopy(gm), fakes)
        return self._torchdynamo_orig_backend(gm, inputs)

    def get_compiler_config(self) -> Any:
        getter = getattr(self._torchdynamo_orig_backend, "get_compiler_config", None)
        return None if getter is None else getter()


# Guards whose check IS object identity, directly or through a derived guard,
# which is the same test default_guard_filter_fn drops on.
def _optimize_isolated(
    backend: _PrecompileBackend,
    package: CompilePackage,
    *,
    recompile_limit: int,
    dynamic: bool | None,
    guard_filter_fn: Callable[[Sequence[GuardFilterEntry]], Sequence[bool]] | None,
) -> OptimizeContext:
    from .eval_frame import OptimizeContext

    optimize_ctx = torch._dynamo.optimize(
        backend,
        package=package,
        recompile_limit=recompile_limit,
        dynamic=dynamic,
        guard_filter_fn=guard_filter_fn,
        isolate_recompiles=True,
    )
    if not isinstance(optimize_ctx, OptimizeContext):
        raise PackageError("torch.compiler.precompile requires Dynamo to be enabled")
    callback = optimize_ctx.callback
    if not isinstance(callback, CatchErrorsWrapper):
        raise AssertionError(f"expected a CatchErrorsWrapper, got {type(callback)}")
    optimize_ctx.callback = _AllowEmptyGraphsCallback(
        callback._torchdynamo_orig_backend, callback.hooks
    )
    return optimize_ctx


class PrecompileSession:
    """
    A caller-driven capture in progress. Enter as a context manager to get the
    callable to exercise, invoke it with real inputs inside the block, and
    ``save()`` to write the artifact -- repeatedly to checkpoint mid-block, and
    once more on exit. The compiled region stays alive for the whole block, so
    every call reuses the variants the earlier ones produced.
    """

    def __init__(
        self,
        fn: Callable[..., object],
        *,
        backend: str = "inductor",
        guard_filter_fn: Callable[[Sequence[GuardFilterEntry]], Sequence[bool]]
        | None = None,
        recompile_limit: int = 256,
        dynamic: bool | None = None,
        training: bool = False,
        keep_graphs: bool = False,
        invariants: str | None = None,
    ) -> None:
        self._fn = fn
        self._backend = backend
        self._custom_guard_filter = guard_filter_fn is not None
        # A training capture traces with grad on and lowers the backward
        # eagerly, so the artifact carries AOTAutograd's CompiledFunction and
        # calling .backward() on a served output runs precompiled code.
        self._training = training
        # Retaining a graph deepcopies it for the session; the guard probe and
        # captures whose graphs are never rendered leave it off.
        self._keep_graphs = keep_graphs
        self._invariants_path = invariants
        self._backend_obj: _PrecompileBackend | None = None
        # Slots dropped by the invariance policy. The policy itself is not part
        # of this build, so the set stays empty; summary() subtracts it anyway.
        self._policy_dropped_guards: set[tuple[str, str]] = set()
        # slot -> the check it rendered as, for every slot dropped by any
        # route. See PrecompileSummary.dropped_guard_code for why the slot
        # tuple alone cannot be audited.
        self._dropped_guard_code: dict[tuple[str, str], str] = {}
        self._dropped_guards: set[tuple[str, str]] = set()
        self._kept_guards: set[tuple[str, str]] = set()
        self._risky_dropped_guards: set[tuple[str, str]] = set()
        self._capture_errors: list[str] = []
        # How many capture errors predate the capture block. Set at __enter__;
        # a render counts only the errors raised since.
        self._gate_error_mark = 0
        self._recorded_exception_keys: set[tuple[type[BaseException], str]] = set()
        # (co_name, co_filename, co_firstlineno) -> one fact set per compilation
        self._guard_sets: dict[tuple[str, str, int], list[frozenset[_GuardFact]]] = {}
        self._undetermined: dict[tuple[str, str, int], set[_GuardFact]] = {}
        self._guard_filter_fn = (
            default_guard_filter_fn
            if guard_filter_fn is None
            else _compose_with_default(guard_filter_fn)
        )
        self._recompile_limit = recompile_limit
        self._dynamic = dynamic
        self._entry_fn = _entry_fn_of(fn)
        # The guard filter rides on the optimize context rather than the
        # package, so it applies to the live guards as well as the serialized
        # ones, exactly as caching_precompile does today.
        self._package = CompilePackage(self._entry_fn)
        self._backend_artifacts: dict[_BackendId, Any] = {}
        self._stack: contextlib.ExitStack | None = None
        self._optimized: Callable[..., object] | None = None
        self._compiled: Callable[..., object] | None = None
        self._state = threading.Condition()
        self._active_calls = 0
        self._closing = False
        self._finished = False

    def _take_backend_artifacts(self) -> None:
        from torch._dynamo.output_graph import noop_graph_call
        from torch._dynamo.precompile_context import (
            EagerCacheArtifact,
            PrecompileContext,
        )

        for backend_id in self._package.cache_entry().backend_ids:
            artifact = PrecompileContext.take_artifact(backend_id)
            if artifact is not None:
                self._backend_artifacts[backend_id] = artifact
            elif self._package.cached_backends.get(backend_id) is noop_graph_call:
                # output_graph short-circuits an empty graph to noop_graph_call
                # without filing anything under its id, which the bytecode still
                # names. Record the no-op so the served frame dispatches to it
                # rather than running eager. Here rather than in
                # _collect_backends: teardown clears cached_backends first.
                self._backend_artifacts[backend_id] = EagerCacheArtifact(
                    key=backend_id, content=noop_graph_call
                )

    def _record_capture_error(self, error: BaseException) -> None:
        message = str(error)
        key = (type(error), message)
        if key in self._recorded_exception_keys:
            return
        self._recorded_exception_keys.add(key)
        self._capture_errors.append(f"{type(error).__name__}: {message}")

    def _release(self) -> None:
        # The compiled variants stay in the entry's ordinary Dynamo cache, as
        # they would after torch.compile; clearing them per capture needs the
        # region-scoped cache entries that are not part of this build. The
        # eager backends stay until the render collects them.
        self._optimized = None
        if self._backend != "eager":
            self._package.cached_backends.clear()

    def _call(self, *args: object, **kwargs: object) -> object:
        with self._state:
            if self._compiled is None or self._closing:
                raise RuntimeError("PrecompileSession is not active")
            compiled = self._compiled
            self._active_calls += 1
        try:
            with _capture_config(self._training):
                result = compiled(*args, **kwargs)
        except BaseException as e:
            self._record_capture_error(e)
            raise
        finally:
            with self._state:
                self._active_calls -= 1
                if self._active_calls == 0:
                    self._state.notify_all()
        return result

    def __enter__(self) -> Callable[..., object]:
        if self._finished:
            raise RuntimeError("PrecompileSession cannot be re-entered")
        if self._stack is not None:
            raise PackageError(
                "PrecompileSession is already active: a session runs one capture "
                "block at a time, so serialize concurrent entries."
            )
        self._gate_error_mark = len(self._capture_errors)
        stack = contextlib.ExitStack()
        # The grad-mode/config patch is per call, in _call, not block-level:
        # user code between calls (optimizer.step, data loading, save()) must
        # run in the ambient mode, not the capture's.
        self._stack = stack
        try:
            if self._optimized is None:
                self._backend_obj = _PrecompileBackend(self._backend, self._keep_graphs)
                optimize_ctx = _optimize_isolated(
                    self._backend_obj,
                    self._package,
                    recompile_limit=self._recompile_limit,
                    dynamic=self._dynamic,
                    guard_filter_fn=self._guard_filter_fn,
                )
                self._optimized = optimize_ctx(self._fn)
            self._compiled = self._optimized
        except BaseException as e:
            self._record_capture_error(e)
            # A __enter__ that raises never gets its __exit__, so without this
            # the session is wedged: save() reports the block as still open.
            self._stack = None
            self._compiled = None
            # Drain in-flight calls FIRST, before any teardown, exactly as
            # __exit__ does: a concurrent call can still be compiling against a
            # borrowed cache entry, and stack.close() mutates state it reads.
            # The cleanup chain sits in the drain's finally so an interrupt
            # raised out of wait() (e.g. KeyboardInterrupt) still releases the
            # session rather than leaking it until process exit.
            try:
                with self._state:
                    self._closing = True
                    while self._active_calls:
                        self._state.wait()
            finally:
                try:
                    stack.close()
                finally:
                    try:
                        self._take_backend_artifacts()
                    finally:
                        try:
                            self._release()
                            self._finished = True
                        finally:
                            with self._state:
                                self._state.notify_all()
            raise
        return self._call

    def __exit__(self, *exc: object) -> None:
        if isinstance(exc[1], BaseException):
            self._record_capture_error(exc[1])
        with self._state:
            self._closing = True
            try:
                while self._active_calls:
                    self._state.wait()
            except BaseException:
                self._closing = False
                self._state.notify_all()
                raise
            stack = self._stack
            self._stack = None
            self._compiled = None
        if stack is not None:
            try:
                stack.close()
            except BaseException as e:
                self._record_capture_error(e)
                raise
            finally:
                try:
                    self._take_backend_artifacts()
                finally:
                    self._release()
                    self._finished = True
                    with self._state:
                        self._state.notify_all()
        self._recorded_exception_keys.clear()


def precompile_capture(
    fn: Callable[..., object],
    *,
    backend: str = "inductor",
    guard_filter_fn: Callable[[Sequence[GuardFilterEntry]], Sequence[bool]]
    | None = None,
    recompile_limit: int = 256,
    dynamic: bool | None = None,
    training: bool = False,
    keep_graphs: bool = False,
    invariants: str | None = None,
) -> PrecompileSession:
    r"""Begin capturing ``fn`` into a multi-graph artifact.

    ``recompile_limit`` defaults well above Dynamo's usual 8 because a
    precompile deliberately wants one compiled variant per condition, whereas
    the normal limit exists to catch runaway recompilation. It also raises a
    lower ambient ``accumulated_recompile_limit`` for this capture so that the
    explicit API limit is the effective one.

    The capture is caller-driven: enter the session to get a callable, invoke it
    exactly as you would ``fn`` inside the ``with`` body, and the calls fold into
    the artifact in the ambient grad mode. The compiled region stays alive for
    the whole block, so every call reuses the variants the earlier ones
    produced. ``invariants`` names a file written when the block exits without
    an exception.

    Runtime guards remain intact during capture. ``guard_filter_fn`` applies
    only to the serialized guard state, so every call observes the same
    recompilation behavior as ordinary ``torch.compile``. ``save()`` refuses
    the risky subset by default rather than every drop, and a drop a custom
    filter adds beyond the default's counts as risky.
    """
    return PrecompileSession(
        fn,
        backend=backend,
        guard_filter_fn=guard_filter_fn,
        recompile_limit=recompile_limit,
        dynamic=dynamic,
        training=training,
        keep_graphs=keep_graphs,
        invariants=invariants,
    )
