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
import os
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
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from .convert_frame import ConvertFrameReturn

    from .eval_frame import OptimizeContext
    from .package import _BackendId, _DynamoCacheEntry
    from .types import CacheEntry, DynamoFrameType, GuardFilterEntry
    from .variables.builder import FrameStateSizeEntry

import contextvars
import logging
import re

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


_OBJ_ID = re.compile(r"(?<=, )\d+(?=\), type=)")


_SAVED_HOOK_IDS = re.compile(r"(?<=top_saved_tensors_hooks ids == )\(\d+(?:, \d+)*\)")


_DYNAMO_COUNTER = re.compile(
    r"(__builtins_dict__|__compiled_fn|__resume_at)_*\d+(_\d+)?"
)


# OutputGraph.install_global_by_id names a global "<prefix>_<id(value)>_c<n>",
# so a guard reading one carries BOTH an address and a compile counter inside
# an identifier, where neither pattern above can see it. Real models reach this
# -- transformers' Qwen2 installs three -- and the report then differs run to
# run, which is exactly what the "commit and diff" contract rules out.
_DYNAMO_GLOBAL_BY_ID = re.compile(r"_\d{9,}_c\d+\b")


def _normalize(text: str) -> str:
    text = _SAVED_HOOK_IDS.sub("(<ids>)", text)  # see _saved_hooks_fingerprint
    text = _DYNAMO_GLOBAL_BY_ID.sub("_<id>_c<n>", _OBJ_ID.sub("<id>", text))
    return _DYNAMO_COUNTER.sub(r"\1_<n>", text)


def _render_code(code_list: Sequence[str] | None) -> tuple[str, ...]:
    # Keep the _dynamo_*_indices parts: they carry TENSOR_MATCH's dimension
    # marking, so mark_static on one variant and not the next shows up only here.
    return tuple(_normalize(part) for part in (code_list or ()))


_SHAPE_BEARING_GUARD_TYPES = frozenset(
    {
        "TENSOR_MATCH",
        "SEQUENCE_LENGTH",
        # Value-equality guards belong here for the same reason and are the
        # half that bites hardest: they pin a Python value the graph
        # specialized on -- an int or bool argument, `module.training`, an
        # `.item()` result, `mask=None`. Dropped, the artifact serves the
        # captured branch for every other value, with correct-looking numerics
        # and nothing in the header to say so. Shapes at least crash inside a
        # kernel; these do not.
        "CONSTANT_MATCH",
        "EQUALS_MATCH",
        "DUPLICATE_INPUT",
        # And the one that pins whether an attribute is THERE. hasattr is a
        # branch like any other, so dropping it serves the captured side to a
        # caller on the other one -- the same silent wrong answer as a dropped
        # CONSTANT_MATCH. Reachable on the DEFAULT gates, because a
        # single-variant capture makes every slot look invariant and the drop
        # is not classed risky.
        "HASATTR",
        # And the guard that pins an input's KIND. Dropped, a graph traced for
        # one class is served to another and returns the first one's answer,
        # silently -- there is no shape to crash on. Upstream depends on this
        # specifically: an AsyncCollectiveTensor's tensor-class guards are
        # deliberately removed so an ACT-traced graph can be reused for the
        # resolved tensor, and the observation sites reinstall exactly this
        # guard to keep that sound. FAKE_SCRIPT_TYPE_MATCH is the same pin for
        # a reference-type opaque object (type(unwrapped) is T).
        "TYPE_MATCH",
        "FAKE_SCRIPT_TYPE_MATCH",
        # And the default-device pin: the graph specialized on
        # utils_device.CURRENT_DEVICE, so a capture under the default None
        # served under torch.set_default_device("cuda") returns CPU tensors
        # with no refusal.
        "DEFAULT_DEVICE",
        # And every guard that pins a Python fact about a value or a container's
        # contents: whether a key is in a dict or a set, which keys a dict has,
        # whether an attribute is absent from an instance __dict__, how long a
        # tuple iterator is, where a range/count iterator stands, whether a
        # value is None or a given bool. Each one is a branch the graph
        # specialized on, so a drop serves the captured branch to the other
        # side, silently -- and a module-owned dict (self.opts = {}) is
        # environment-rooted, which is exactly where the policy used to drop it.
        "BOOL_MATCH",
        "CONSTANT_SUBCLASS_MATCH",
        "COUNT_ITERATOR_MATCH",
        "DICT_CONTAINS",
        "DICT_KEYS_MATCH",
        "DICT_NOT_CONTAINS",
        "MAPPING_KEYS_CHECK",
        "NONE_MATCH",
        "NOT_NONE_MATCH",
        "NOT_PRESENT_IN_GENERIC_DICT",
        "RANGE_ITERATOR_MATCH",
        "SET_CONTAINS",
        "SET_NOT_CONTAINS",
        "TUPLE_ITERATOR_LEN",
    }
)


# Guards whose check IS object identity, directly or through a derived guard,
# which is the same test default_guard_filter_fn drops on.
_UNMODELLED_GUARD_TYPES = frozenset(
    {
        "DISPATCH_KEY_SET_MATCH",
        "DTENSOR_SPEC_MATCH",
        # Its builder is a no-op like GRAD_MODE's, but GlobalStateGuard does not
        # snapshot FSDP training state and the state is per param group, so
        # nothing here can model or vouch for it.
        "FSDP_TRAINING_STATE",
        "GLOBAL_STATE",
        "OPAQUE_OBJ_GUARD_FN_MATCH",
        "SHAPE_ENV",
        "TENSOR_SUBCLASS_METADATA_MATCH",
        "TORCH_FUNCTION_STATE",
    }
)


# Guard types whose GuardBuilder method is `pass`: the guard is a marker, and
# the check it names is made by GLOBAL_STATE's leaf. Nothing about them is
# serialized or dropped, so they never appear in a dropped-guard report --
# listing GRAD_MODE as "a precondition nothing checks" would be false, since
# GlobalStateGuard checks it on every call. Their facts ARE compared, from the
# same process state GlobalStateGuard snapshots (see _value_fingerprint).
_NOOP_GUARD_TYPES = frozenset({"DETERMINISTIC_ALGORITHMS", "GRAD_MODE"})


def _is_noop_guard_type(guard_type: str) -> bool:
    # EMPTY_NN_MODULE_HOOKS_DICT is a no-op by config: under
    # skip_nnmodule_hook_guards, the default, GuardBuilder emits nothing for it.
    return guard_type in _NOOP_GUARD_TYPES or (
        guard_type == "EMPTY_NN_MODULE_HOOKS_DICT"
        and torch._dynamo.config.skip_nnmodule_hook_guards
    )


# The ONLY guard types the invariance policy may drop, and only when proven
# invariant across every captured variant: identity guards (which the default
# filter drops anyway, as unserializable) and process-wide compiler state. The
# four sets form a total, disjoint classification of GuardBuilder's
# guard-producing methods, pinned by
# test_precompile_package.test_guard_policy_classification_is_total: a guard type in
# none of them -- i.e. any type added to GuardBuilder after this list -- is
# KEPT unconditionally until someone classifies it here, so a new value-pinning
# guard can never become silently droppable by default.
def _fact_order(fact: _GuardFact) -> tuple[str, str, str, str]:
    # value is part of the key: once the boilerplate code parts are filtered a
    # TENSOR_MATCH renders no code, so two shape specializations would otherwise
    # tie and sort unstably, making the file differ run to run.
    return (fact.source, fact.guard_type, " ".join(fact.code), fact.value)


def _summarize(
    entry: _DynamoCacheEntry,
    dropped: set[tuple[str, str]],
    kept: set[tuple[str, str]],
    policy_dropped: set[tuple[str, str]],
    risky: set[tuple[str, str]],
    truncated: frozenset[str],
    uncovered: frozenset[str],
    capture_errors: Sequence[str],
    guard_sets: Mapping[tuple[str, str, int], Sequence[frozenset[_GuardFact]]],
    dropped_code: Mapping[tuple[str, str], str],
) -> PrecompileSummary:
    # The value-pinning analysis behind wont_generalize is not part of this build.
    wont_generalize: tuple[str, ...] = ()
    return PrecompileSummary(
        frames=len(entry.codes),
        resume_functions=sum(1 for c in entry.codes if c.install_to_global),
        guarded_codes=sum(len(c.guarded_codes) for c in entry.codes),
        backend_graphs=len(entry.backend_ids),
        bypassed=tuple(c.python_code.co_name for c in entry.codes if c.bypassed),
        truncated=tuple(sorted(truncated)),
        uncovered_frames=tuple(sorted(uncovered)),
        wont_generalize=wont_generalize,
        dropped_guards=tuple(sorted(dropped)),
        dropped_guard_code=tuple(
            (gtype, name, dropped_code[(gtype, name)])
            for gtype, name in sorted(dropped | policy_dropped | risky)
            if (gtype, name) in dropped_code
        ),
        kept_guards=tuple(sorted(kept)),
        risky_dropped_guards=tuple(sorted(risky)),
        policy_dropped_guards=tuple(sorted(policy_dropped)),
        capture_errors=tuple(capture_errors),
    )


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


def _warn_risky_drops(risky: Sequence[tuple[str, str]]) -> None:
    """Report accepted risky drops, shape-bearing ones first.

    Ordering by type only puts the candidates where they can be seen; whether a
    guarded VALUE can differ at serve time is not something the type answers.
    """
    by_type: dict[str, list[str]] = {}
    for guard_type, name in sorted(risky):
        by_type.setdefault(guard_type, []).append(name)

    # Grouped rather than a flat cut, and capped PER TYPE: a flat list is
    # dominated by whichever type happens to be most numerous, which can bury a
    # lone SEQUENCE_LENGTH behind a crowd of CONSTANT_MATCH and CLOSURE_MATCH.
    def render(types: list[str], per_type: int) -> str:
        parts = []
        for t in types:
            names = by_type[t]
            shown = ", ".join(names[:per_type])
            more = f", +{len(names) - per_type} more" if len(names) > per_type else ""
            parts.append(f"{t} x{len(names)}: {shown}{more}")
        return "; ".join(parts)

    shape_types = [t for t in by_type if t in _SHAPE_BEARING_GUARD_TYPES]
    other_types = [t for t in by_type if t not in _SHAPE_BEARING_GUARD_TYPES]
    # Says "could" rather than "can", and points at the distinction that
    # actually decides it. This is a classification by guard TYPE, and the
    # question a reader has is whether the guarded VALUE can differ at serve
    # time -- which the type does not answer. The first four this ordering
    # surfaced on a real model were all reached through a class or function
    # definition (__mro__ walks to __defaults__, __code__) and were therefore
    # compile-time constants that no batch could change. Distinguishing those
    # properly needs the structured source, not the name.
    shape_report = (
        f" COULD BEAR ON SHAPE ({sum(len(by_type[t]) for t in shape_types)}), "
        f"unlike the rest, so check these first -- but check whether each one "
        f"can actually differ at serve time: a guard reached through a class or "
        f"function definition (an __mro__ walk, __defaults__, __code__) is a "
        f"compile-time constant and cannot: {render(shape_types, 3)}."
        if shape_types
        else ""
    )
    log.warning(
        "precompile: %d dropped guard(s) can affect dispatch, so nothing checks "
        "them at load.%s The rest are identity slots to audit: %s. "
        "summary().risky_dropped_guards has all of them; this warning appears "
        "only because require_no_risky_drops=False explicitly accepted them.",
        len(risky),
        shape_report,
        render(other_types, 2) or "none",
    )


def _missing_backends_message(
    total: int, missing: Sequence[object], backend: str = "inductor"
) -> str:
    """Why some compiled subgraphs never reached the artifact.

    Reports the recorded/total split rather than asserting nothing was
    recorded: a single missing id is fatal here, and saying so as "never
    recorded" reads as total failure when most of the capture succeeded.
    """
    shown = ", ".join(str(b) for b in missing[:8])
    if len(missing) > 8:
        shown += f", ... ({len(missing) - 8} more)"
    if backend not in ("inductor", "eager"):
        # A session takes any backend Dynamo can resolve, but only these two
        # leave something a served artifact can run: "eager" keeps the fx
        # graphs and "inductor" bundles compiled code. Anything else captures
        # cleanly and records nothing, so say so here rather than let it read
        # as a defect in the model. aot_eager is the one people reach for,
        # since it is how you isolate AOTAutograd.
        return (
            f"Precompilation recorded {total - len(missing)} of {total} "
            f"compiled backend(s) because backend={backend!r} does not produce "
            f"anything serializable; precompile can record only 'inductor' or "
            f"'eager'. To isolate AOTAutograd without inductor, use plain "
            f"torch.compile(backend='aot_eager') -- that needs no precompile."
        )
    from torch._dynamo.utils import counters

    # Rendering re-enters AOTAutograd outside the pinned bypass_autograd_cache_key
    # config and bypasses there routinely, so this is a diagnostic, not a diagnosis.
    bypasses = counters["aot_autograd"].get("autograd_cache_bypass", 0)
    bypass_note = (
        f" It bypassed {bypasses} time(s) here, which rendering does for any "
        "graph the cache cannot key, and which does not by itself explain a gap."
        if bypasses
        else ""
    )
    return (
        f"Precompilation recorded {total - len(missing)} of {total} compiled "
        f"backend(s), so {len(missing)} graph(s) would reach the artifact with "
        f"no code behind them: {shown}. Capture pins functorch's "
        "bypass_autograd_cache_key, so AOTAutograd keys every graph it lowers "
        "and no longer declines to record one it cannot address."
        + bypass_note
        + " A gap therefore means a graph whose backward never compiled, which "
        "is a forward-only capture with grad enabled. Pass training=True to "
        "lower the backward eagerly (the joint trace synthesizes tangents, so "
        "no loss is needed), capture under torch.no_grad() / "
        "torch.inference_mode() for an inference artifact, or run .backward() "
        "inside the capture block. Re-run with "
        "TORCH_LOGS=+torch._functorch._aot_autograd to see each graph as it "
        "lowers."
    )


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
        self._guard_filter_fn = self._recording_filter(
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
        if self._invariants_path is None:
            return
        if exc[0] is None:
            self.write_invariants(self._invariants_path)
        else:
            # A partial capture's report reads exactly like a complete one, so
            # it is not written; say why rather than leaving the user looking
            # for a file that never appeared.
            log.warning(
                "precompile: the capture block raised %s, so no invariants "
                "report was written to %s. Call write_invariants() for the "
                "partial one.",
                getattr(exc[0], "__name__", exc[0]),
                self._invariants_path,
            )

    def _record_dropped_code(
        self, slot: tuple[str, str], code: Sequence[str] | None
    ) -> None:
        """Remember what a dropped slot actually checked.

        Accumulate distinct renderings rather than keeping the first: sibling
        guards can share a (guard_type, name) slot yet check different things --
        HASATTR for two attributes of one base -- so first-writer-wins would
        hide the second. Repeats within a slot (once per variant and per
        re-serialization) collapse because the renderings agree up to the ids
        _normalize already masks.
        """
        rendered = " ; ".join(_render_code(code))
        if not rendered:
            return
        existing = self._dropped_guard_code.get(slot)
        if existing is None:
            self._dropped_guard_code[slot] = rendered
        elif rendered not in existing.split(" | "):
            self._dropped_guard_code[slot] = f"{existing} | {rendered}"

    def _recording_filter(
        self,
        inner: Callable[[Sequence[GuardFilterEntry]], Sequence[bool]],
    ) -> Callable[[Sequence[GuardFilterEntry]], Sequence[bool]]:
        """
        Remember which guard types were discarded. A dropped guard does not
        fail at serving time, it silently widens what a graph is reused for, so
        the set has to be inspectable rather than invisible.
        """

        # One object per distinct fact, shared by every compilation that
        # produced it. A recompiled frame repeats nearly all of its guards, so
        # storing a copy per compilation would make the session grow with
        # variants rather than with facts.
        pool: dict[_GuardFact, _GuardFact] = {}

        def filter_fn(entries: Sequence[GuardFilterEntry]) -> Sequence[bool]:
            decisions = inner(entries)
            # A custom filter composes with the default, so the identity drops
            # the default makes anyway are judged as they always are; only a
            # drop the custom filter ADDED is risky by construction, because
            # nothing here can say what the caller gave up.
            default_kept = (
                default_guard_filter_fn(entries)
                if self._custom_guard_filter
                else decisions
            )
            # A no-op type's check, where it has one, is GLOBAL_STATE's leaf,
            # made whatever the filter said about the marker: it is dropped
            # only with GLOBAL_STATE.
            global_state_kept = any(
                keep and e.guard_type == "GLOBAL_STATE"
                for keep, e in zip(decisions, entries)
            )
            facts: set[_GuardFact] = set()
            undetermined: set[_GuardFact] = set()
            for keep, by_default, entry in zip(decisions, default_kept, entries):
                slot = (entry.guard_type, entry.name)
                enforced = keep or (
                    _is_noop_guard_type(entry.guard_type) and global_state_kept
                )
                if not enforced:
                    self._record_dropped_code(slot, entry.orig_guard.code_list)
                target = self._kept_guards if enforced else self._dropped_guards
                target.add(slot)
                # Risky here means a drop the default filter would not have made;
                # the finer lint over identity guards is not part of this build.
                if not enforced and by_default:
                    self._risky_dropped_guards.add(slot)
                unmodelled = entry.guard_type in _UNMODELLED_GUARD_TYPES
                fact = _GuardFact(
                    guard_type=entry.guard_type,
                    source=_normalize(entry.name),
                    code=_render_code(entry.orig_guard.code_list),
                    value="" if unmodelled else _plain_value(entry),
                    enforced=enforced,
                )
                fact = pool.setdefault(fact, fact)
                # Never compared, so never claimed to hold: see
                # _UNMODELLED_GUARD_TYPES.
                (undetermined if unmodelled else facts).add(fact)
            # One filter call is one compilation, and only the package knows
            # which frame is being compiled.
            entry = self._package._current_entry
            if entry is not None:
                code = entry.python_code
                key = (code.co_name, code.co_filename, code.co_firstlineno)
            else:
                key = ("<unknown>", "<unknown>", 0)
            self._guard_sets.setdefault(key, []).append(frozenset(facts))
            self._undetermined.setdefault(key, set()).update(undetermined)
            return decisions

        return filter_fn

    def invariants(self) -> tuple[FrameInvariants, ...]:
        """
        Per frame, the guards that held in EVERY compiled variant of it.

        Intersection is per frame rather than global because guards from
        different frames are not comparable: the entry frame guards its
        arguments, a resume frame guards whatever crossed the graph break, so a
        global intersection would be empty for any model that breaks.

        A frame compiled once reports everything as invariant, which is true but
        uninformative -- exercise more than one variant for the diff to mean
        anything.

        GLOBAL_STATE, TORCH_FUNCTION_STATE and FSDP_TRAINING_STATE carry no
        value of their own, so nothing here can say whether two variants agreed
        on, say, autocast. They are listed as UNDETERMINED rather than compared
        -- see _UNMODELLED_GUARD_TYPES, and note that calling them equal is
        precisely how the report would assert a precondition that does not
        hold. GRAD_MODE and DETERMINISTIC_ALGORITHMS are fingerprinted instead
        from the process state GlobalStateGuard snapshots, because the capture
        path itself produces the grad-mode split and the fingerprint can model
        it.
        """
        out = []
        for (name, filename, lineno), sets in sorted(self._guard_sets.items()):
            shared = frozenset.intersection(*sets) if sets else frozenset()
            everything: set[_GuardFact] = set()
            for one in sets:
                everything |= one
            out.append(
                FrameInvariants(
                    frame=name,
                    filename=filename,
                    lineno=lineno,
                    variants=len(sets),
                    invariant=tuple(sorted(shared, key=_fact_order)),
                    varying=tuple(sorted(everything - shared, key=_fact_order)),
                    undetermined=tuple(
                        sorted(
                            self._undetermined.get((name, filename, lineno), set()),
                            key=_fact_order,
                        )
                    ),
                )
            )
        return tuple(out)

    def write_invariants(self, path: str) -> None:
        """
        Write :meth:`invariants` to ``path`` in human-readable form.

        ``path`` is a FILE, written exactly as given, with parent directories
        created -- ``snapshots/invariants.txt`` is a text file. Same contract as
        :meth:`save`.

        Output is stable across runs of the same capture: object ids and
        Dynamo's per-process counters are normalized away, so the file can be
        committed and diffed to see what a model change did to its guards.
        """
        frames = self.invariants()
        target = getattr(self._fn, "__qualname__", None) or type(self._fn).__qualname__
        lines = [
            f"# precompile invariants for {target}",
            "#",
            "# Conditions that held in EVERY compiled variant of a frame. A call",
            "# violating one cannot be served by any graph in this artifact, so",
            "# these are the preconditions the artifact is only valid under.",
            "# 'varies' lists what differed between variants -- those are what",
            "# distinguish one compiled graph from another, not preconditions.",
            "# 'unknown' lists guards whose check this report cannot model, so it",
            "# cannot say whether they held across variants. Treat them as",
            "# neither: they may or may not be preconditions.",
            "#",
            "# enforced = the guard is serialized and rechecked when the artifact",
            "#            is loaded.",
            "# dropped  = it was not serialized, so it is a precondition",
            "#            NOTHING checks at serving time. See",
            "#            PrecompileSummary.dropped_guards.",
            "#",
            f"# {len(frames)} frame(s), "
            f"{sum(f.variants for f in frames)} compilation(s)",
        ]
        if any(f.variants < 2 for f in frames):
            lines.append(
                "# NOTE: some frames were compiled once, so their invariants are"
                " just every guard. Exercise more variants for a real diff."
            )
        for f in frames:
            where = f"{os.path.basename(f.filename)}:{f.lineno}"
            lines.append("")
            lines.append(
                f"frame {f.frame} ({where})  {f.variants} variant(s), "
                f"{len(f.invariant)} invariant, {len(f.varying)} varying, "
                f"{len(f.undetermined)} undetermined"
            )
            if not f.invariant:
                lines.append("  invariant: (none)")
            for fact in f.invariant:
                lines.append(f"  invariant {fact.render()}")
            for fact in f.varying:
                lines.append(f"  varies    {fact.render()}")
            for fact in f.undetermined:
                lines.append(f"  unknown   {fact.render()}")
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        # UTF-8 explicitly: the report renders user identifiers (module, class and
        # parameter names) and the ambient locale can be ASCII in a container, where
        # a non-ASCII name would raise UnicodeEncodeError and leave a truncated file
        # behind -- from the one call that exists to explain the artifact.
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        log.info(
            "precompile: wrote invariants for %d frame(s) to %s", len(frames), path
        )

    def summary(self) -> PrecompileSummary:
        # Risky only when the SAME source held DIFFERENT values across variants;
        # merely being absent from one variant (a MODULE_MATCH a branch does not
        # touch) flags every ordinary multi-branch capture. Grouped by source,
        # not (guard_type, source): a rebind can change the guard type too.
        values_by_source: dict[str, set[str]] = {}
        for frame in self.invariants():
            for fact in frame.varying:
                if not fact.enforced:
                    values_by_source.setdefault(fact.source, set()).add(fact.value)
        varying_dropped = {
            (fact.guard_type, fact.source)
            for frame in self.invariants()
            for fact in frame.varying
            if not fact.enforced and len(values_by_source.get(fact.source, ())) > 1
        }
        # Normalized: a Dynamo per-process counter (__builtins_dict___14) makes
        # one logical drop appear once per compilation, under names that change
        # every run, which inflates the count save() truncates at 8 and can push
        # the genuinely config-selected drop out of the operator's view.
        risky = {
            (guard_type, _normalize(name))
            for guard_type, name in self._risky_dropped_guards
        } | {
            (guard_type, _normalize(name))
            for guard_type, name in self._dropped_guards
            if (guard_type, _normalize(name)) in varying_dropped
        }
        from .package import SerializedCode

        entry = self._package.cache_entry()
        return _summarize(
            entry,
            self._dropped_guards,
            self._kept_guards - self._policy_dropped_guards,
            self._policy_dropped_guards,
            risky,
            # Truncated frames need the package's capture-mode bookkeeping,
            # which is not part of this build; an uncovered frame is one the
            # package recorded without a single guarded variant.
            frozenset(),
            frozenset(
                SerializedCode.to_code_object(code.python_code).co_name
                for code in entry.codes
                if not code.bypassed and not code.guarded_codes
            ),
            self._capture_errors,
            self._guard_sets,
            self._dropped_guard_code,
        )

    def _gated_summary(
        self,
        *,
        require_complete: bool,
        require_no_risky_drops: bool,
        require_no_dropped_guards: bool,
    ) -> PrecompileSummary:
        """Run the coverage and guard gates, or raise saying which one failed.

        Callable mid-block: ``save()`` renders while the region is still live.
        """
        summary = self.summary()
        # Only the errors raised since the block was entered. A call that failed
        # already raised to the caller, who saw it and carried on; counting a
        # pre-block error here would refuse a render over something the caller
        # already handled.
        fresh_errors = list(summary.capture_errors)[self._gate_error_mark :]
        if require_complete and fresh_errors:
            raise PackageError(
                "Precompilation is incomplete because capture raised: "
                f"{fresh_errors}. Re-run every example successfully, "
                "or pass require_complete=False to save the partial artifact."
            )
        if require_no_dropped_guards and summary.dropped_guards:
            raise PackageError(
                f"Precompilation dropped {len(summary.dropped_guards)} guard(s) that "
                f"were not serialized: {list(summary.dropped_guards)}. Rebinding any "
                f"of those sources between capture and load can silently serve a graph "
                f"traced against the old value. Pass require_no_dropped_guards=False "
                f"only to select the relaxed risky-drop policy."
            )
        if summary.risky_dropped_guards and require_no_risky_drops:
            raise PackageError(
                f"Precompilation dropped guard(s) that can affect dispatch on "
                f"{[n for _, n in summary.risky_dropped_guards]}. Each of those names "
                f"either a configuration-dependent identity slot or a guard discarded "
                f"by a custom filter. Nothing checks it at load time, so a different "
                f"value can silently select the wrong graph instead of recompiling. "
                f"Make the value reachable through a serializable guard, pin both "
                f"machines to the same value, or pass "
                f"require_no_risky_drops=False to accept the risk explicitly."
            )
        elif summary.risky_dropped_guards:
            # The caller explicitly accepted the risk.
            _warn_risky_drops(summary.risky_dropped_guards)
        if require_complete:
            if summary.guarded_codes == 0:
                raise PackageError(
                    "Precompilation captured no compiled code. Capture happens by "
                    "execution, so the callable must actually be run inside the "
                    "capture block. A call Dynamo could not turn into guarded code "
                    "is reported separately as an uncovered frame."
                )
            if summary.backend_graphs == 0:
                raise PackageError(
                    "Precompilation compiled no graph: every captured frame was "
                    "empty, so the artifact carries no compiled compute. This is "
                    "what a callable whose whole body sits behind "
                    "torch._dynamo.disable looks like. Pass require_complete=False "
                    "to write the guards-only artifact anyway."
                )
            if summary.truncated:
                raise PackageError(
                    f"Precompilation is incomplete: at least "
                    f"{len(summary.truncated)} frame(s) exceeded recompile_limit "
                    f"(currently {self._recompile_limit}) and are missing variants: "
                    f"{list(summary.truncated)}. That list is a lower bound -- hitting "
                    f"the limit also puts every frame called beneath the named one "
                    f"into run-only mode, so those stop capturing too and never "
                    f"re-enter Dynamo to report it. A frame needs one slot per "
                    f"variant, and frames shared across module instances accumulate "
                    f"them. Raise recompile_limit, or pass require_complete=False to "
                    f"accept an artifact that is more incomplete than this list shows."
                )
            if summary.uncovered_frames:
                raise PackageError(
                    f"Precompilation exercised frame(s) that produced NO guarded code "
                    f"at all: {list(summary.uncovered_frames)}. Those paths are absent "
                    f"from the artifact, and such a frame is skipped at install and runs "
                    f"eager, so serving() cannot report that gap. This is expected for a "
                    f"frame that only dispatches to covered submodules; it also looks "
                    f"exactly like a frame Dynamo gave up on (check "
                    f"TORCH_LOGS=graph_breaks for gb0124). A frame that hit the recompile "
                    f"limit has working variants and is reported as truncated instead. "
                    f"Pass require_complete=False once you have confirmed which."
                )
            if summary.bypassed:
                raise PackageError(
                    f"Precompilation is incomplete: {len(summary.bypassed)} frame(s) "
                    f"were bypassed and will serve nothing: {list(summary.bypassed)}. "
                    f"This usually means their guards could not be serialized. Pass "
                    f"require_complete=False to accept a partial artifact."
                )
        if summary.wont_generalize:
            log.warning(
                "precompile: %d value(s) are pinned to what capture saw (%s). A call "
                "supplying anything else misses every graph, so exercise each value "
                "you need to serve inside the capture block.",
                len(summary.wont_generalize),
                list(summary.wont_generalize),
            )
        return summary

    def rendered_backends(
        self, backend_ids: Sequence[str]
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Compiled subgraphs as READABLE source, and the reason each of the rest
        stayed pickled, both keyed by backend id.

        The pickled bundle is the fallback, not the goal: a subgraph is Inductor
        output, which has a source form (the make_fx tracer emits exactly this),
        unlike the guard trees and transformed bytecode beside it. Anything that
        fails to render -- an effectful op, a graph with no compute, a training
        shape the composer refuses -- stays pickled, and its reason is warned
        here and written into the artifact header so the fallback is visible.

        Rendering re-runs AOTAutograd + Inductor on the retained graph, so it is
        a second lowering, paid once per subgraph that reaches the artifact.
        """
        from torch._functorch import aot_autograd

        if self._backend_obj is None or self._backend == "eager":
            return {}, {}
        rendered: dict[str, str] = {}
        refused: dict[str, str] = {}
        for backend_id in backend_ids:
            held = self._backend_obj.graphs.get(str(backend_id))
            if held is None:
                continue
            gm, fakes = held
            try:
                # grad_enabled is what makes AOTAutograd emit the joint
                # forward+backward for a training capture; without it the
                # backward is silently absent and the served output loses its
                # grad_fn.
                source, _ = aot_autograd.compile_to_python(gm, fakes)
            except Exception as e:
                reason = " ".join(f"{type(e).__name__}: {e}".split())
                log.warning(
                    "precompile: subgraph %s stays pickled, not rendered as source: %s",
                    backend_id,
                    reason,
                )
                refused[str(backend_id)] = reason
                continue
            rendered[str(backend_id)] = source
        from torch._functorch._aot_autograd.to_standalone_python import (
            namespace_module_names,
        )

        keys = list(rendered)
        namespaced = namespace_module_names([rendered[k] for k in keys])
        return dict(zip(keys, namespaced)), refused

    def snapshot_artifact(
        self,
        *,
        require_complete: bool = True,
        require_no_risky_drops: bool = True,
        require_no_dropped_guards: bool = False,
    ) -> tuple[str, bytes]:
        """Render everything captured SO FAR, leaving this session able to capture more.

        The render reads the package's records without consuming them, which is
        what lets ``save()`` be called repeatedly within one capture block.
        """
        summary = self._gated_summary(
            require_complete=require_complete,
            require_no_risky_drops=require_no_risky_drops,
            require_no_dropped_guards=require_no_dropped_guards,
        )
        from torch._precompile import _build_multigraph_artifact

        backends = self._collect_backends()
        entry = self._package.cache_entry()
        rendered, refused = self.rendered_backends(list(backends))
        return _build_multigraph_artifact(
            entry,
            backends,
            summary,
            self._backend,
            _entry_fn_of(self._fn),
            rendered,
            refused,
        )

    def _collect_backends(self) -> dict[str, Any]:
        """The compiled subgraphs this capture produced, keyed by backend id."""
        from torch._dynamo.precompile_context import (
            EagerCacheArtifact,
            PrecompileContext,
        )

        self._take_backend_artifacts()
        collected = dict(self._backend_artifacts)
        if self._backend == "eager":
            # Eager "backends" are fx graphs with no compiled artifact of their
            # own, so they have to be gathered explicitly.
            for backend_id, backend in self._package.cached_backends.items():
                collected[backend_id] = EagerCacheArtifact(
                    key=backend_id, content=backend
                )
        entry = self._package.cache_entry()
        for backend_id in entry.backend_ids:
            if backend_id not in collected:
                artifact = PrecompileContext.take_artifact(backend_id)
                if artifact is not None:
                    collected[backend_id] = artifact
        missing = [b for b in entry.backend_ids if b not in collected]
        if missing:
            raise PackageError(
                _missing_backends_message(
                    len(entry.backend_ids), missing, self._backend
                )
            )
        return {str(b): collected[b] for b in entry.backend_ids}


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


def _plain_value(entry: GuardFilterEntry) -> str:
    """A stable rendering of a guard's value, for the invariants report."""
    if not entry.has_value:
        return ""
    text = _normalize(repr(entry.value))
    return text if len(text) <= 200 else text[:197] + "..."
