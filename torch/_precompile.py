"""Ahead-of-time precompilation (``make_fx`` tracer by default; Dynamo available).

    python_code, cache = torch.compiler.precompile(
        fn, example_inputs=[(model, *example_inputs)]
    )
    f_c = torch.compiler.precompile.load(python_code, cache)
    out = f_c(model, *example_inputs)   # pass the model again at runtime

precompile captures your computation with ``make_fx`` -- a NON-STRICT trace of the ATen
ops that run when ``fn`` executes once on the example inputs. It does not analyze your
Python, so it comes with an explicit contract (the programming model): stay inside it
and the artifact faithfully reproduces ``fn``; step outside it and you get an artifact
that computes the wrong thing.

With ``tracer="dynamo"``, precompile executes every tuple in ``example_inputs`` and
captures the guarded specializations and recompilations Dynamo produces. Graph breaks
are not supported yet. Guards derived from explicit inputs are retained for dispatch;
named guards covering the Python environment (module globals, imports) are dropped
because that environment is a caller-provided invariant. This invariant is unchecked,
so changing the environment can silently run code specialized for its capture-time
state (the dropped guards and their capture-time values are listed in the artifact's
``_DROPPED_GUARDS``). Ambient torch state is checked per call: autocast, default
dtype, deterministic-algorithms and torch-function state must match the capture; grad
mode is pinned to the capture-time mode by the artifact and the thread count is not
checked. The artifact never compiles after loading; a call that fails every retained
guard set raises, naming the failing guard. Compiled graphs and kernels remain Python
source, while guard trees and transformed bytecode are stored as opaque inline data.

With ``tracer="dynamo"``, differentiability is inferred per captured graph exactly
as ``torch.compile`` infers it: capture runs under the caller's ambient grad mode,
and under grad mode a graph whose inputs require grad is differentiable. A capture
under ``torch.no_grad()`` yields inference graphs only. On the
inductor backend each differentiable graph contains AOTAutograd's forward and
backward as readable Inductor source; the served output retains its ``grad_fn`` and
a later ``backward()`` executes those captured backward kernels across captured
recompilations, with backward variants specialized to output-tangent patterns
observed while running the examples (an unseen pattern falls back to the always
covered all-tangents-defined backward, which materializes the missing tangents,
instead of compiling at runtime). On the eager backend the backward is live eager
autograd through the emitted forward ops -- not captured, like the eager forward's
kernels.

``precompile`` returns a self-contained, executable ``python_code`` string plus a
companion integrity-tagged ``cache``. With ``backend="inductor"`` (the default) the
captured graph is lowered through the AOT backend contract
(``torch._functorch.aot_autograd.compile_to_python``, AOTAutograd + Inductor);
``python_code`` JIT-compiles kernels on first call and the cache primes them so a warm
reload skips JIT. With ``backend="eager"`` ``python_code`` inlines the captured graph and
runs on its own. Reload with ``torch.compiler.precompile.load(python_code, cache)``.

The full contract, the calling convention, and the cache / code_hash design all live in
Note [precompile programming model] below; every public entry point and guard references
it.
"""

# Note [precompile programming model]
#
# ``fn`` is the WHOLE computation, e.g. ``lambda model, x: model(x)`` for inference
# or ``lambda model, x, t: loss_fn(model(x), t).backward()`` for a training step.
# Within each tuple in example_inputs, the nn.Module arguments have their parameters
# and buffers lifted to explicit graph inputs (via functional reparametrization), so
# nothing live is baked in; the remaining args are the runtime inputs. The artifact
# embeds NO weights -- you pass the model again at runtime.
#
# Because make_fx is a non-strict trace, precompile offers a contract, not a
# guarantee against misuse. The caller MUST uphold the invariants below. The ones
# that are cheaply knowable from the captured graph are ENFORCED (a violation
# raises PrecompileError); the rest are the caller's responsibility and, if broken,
# produce a SILENTLY INCORRECT artifact -- the ordinary consequence of tracing.
#
# 1. Everything live is an input. Every tensor the computation reads must be passed to
#    fn as an explicit tensor argument -- EXCEPT tensors held inside an nn.Module
#    argument, which precompile handles for you. For an nn.Module argument you do NOT
#    enumerate its tensors yourself: precompile lifts every registered parameter and
#    buffer (recursively, including submodules, tied weights collapsed by identity) to
#    explicit graph inputs for you via functional reparametrization, and re-derives the
#    same list from the runtime model you pass to load(). Passing the module is enough --
#    that is the whole point of accepting modules as arguments. What is NOT lifted is
#    anything not reachable
#    through that protocol: tensors closed over by ``fn`` (globals, captured locals)
#    and plain (non-registered) module attributes -- a bare ``self.weight = t`` rather
#    than a registered parameter/buffer. Those are not inputs; a vanilla make_fx trace
#    would bake them in as get_attr constants. Fix by registering them on the module
#    (register_parameter / register_buffer) or passing them as explicit tensor args.
#    ENFORCED: _check_no_constant_tensors rejects any baked tensor constant.
#
# 2. The runtime model must match the traced model structurally. At load time you
#    pass the model again; precompile re-derives the parameter/buffer list from the
#    runtime model in the SAME order (parameters then buffers, interned by tensor
#    identity so tied weights collapse to a single input). The runtime model must
#    have the same named_parameters()/named_buffers() ordering and count and the
#    same weight tying as the example model. Same architecture with different
#    weights is the intended use (swap in a checkpoint); a structurally different
#    model is undefined. requires_grad is ALSO part of the structural contract: which
#    params get a scattered grad is fixed at capture time from the example model's
#    requires_grad (invariant 5), so flipping a param's requires_grad at runtime does
#    not change what the artifact computes. ENFORCED: the driver compares the runtime
#    model's full param/buffer NAME list (order and identity, tied weights collapsed)
#    against the traced list, AND each runtime param/buffer's SHAPE, DTYPE, AND DEVICE
#    against the baked example values, so a reordered or otherwise structurally-different
#    model -- even one with the same count and names but a differently shaped, typed, or
#    placed weight (e.g. a Linear(4,4) swapped for a Linear(4,8), or a CPU weight where a
#    CUDA one was traced) -- is rejected (it cannot silently scatter grads onto the wrong
#    slot, fail deep in a kernel, or compute the wrong thing). Different WEIGHT VALUES with
#    the same shapes/dtypes/devices are the intended use -- WITH ONE INDUCTOR-BACKEND
#    CAVEAT: the inductor backend ALSO specializes each param/buffer's LAYOUT (memory
#    format), since it bakes assert_size_stride on every weight the graph reads. So a
#    same-shape/same-dtype checkpoint whose weight has a DIFFERENT layout (e.g. a
#    non-contiguous view, or a channels_last weight where the example was contiguous) is
#    REJECTED at runtime by the inductor backend (invariant 6). Match the example weight's
#    layout (.contiguous() to match a contiguous example), or use backend='eager' for
#    layout-flexible weights.
#
# 3. Control flow (and, by default, shapes) is specialized to the example. A non-strict
#    trace follows the single path taken for the example inputs: Python ``if``/``for``
#    over tensor values, ``.item()``, and shape-dependent branching are resolved at
#    trace time and baked. Shapes are static BY DEFAULT (capture uses make_fx in its
#    "real" mode, so each size is baked as a constant). You can opt specific user-input
#    dims into being dynamic by marking them with
#    ``torch._dynamo.decorators.mark_unbacked`` before calling: those dims are
#    captured as UNBACKED symints (symbolic capture), which CANNOT be guarded on -- so
#    the artifact is valid for any runtime size of those dims, and a graph that needs to
#    guard on / specialize a marked dim fails LOUDLY at capture (PrecompileError) instead
#    of baking a silently-wrong result.
#
# 4. Boundary effects. Input mutation (including module buffers -- e.g. BatchNorm
#    running stats in training mode), tensor-subclass wrap/unwrap (e.g. DTensor),
#    outputs that alias inputs, and functionalized RNG are SUPPORTED: the inductor
#    backend lowers through torch._functorch.aot_autograd.compile_to_python, which
#    composes AOTAutograd's own codegen'd prelude/epilogue into the artifact (the
#    effect is reflected onto the runtime model / inputs). Effectful ops are not
#    supported yet and raise at capture time (_assert_supported) with a concrete
#    reason; this is an implementation gap, not a fundamental limit. Every other
#    runtime wrapper that can appear in a composable (cacheable) forward graph is
#    codegen'd as source and composed in; the one non-codegen'd wrapper
#    (FakifiedOutWrapper) only activates under fakify_first_call, which makes the graph
#    non-cacheable, so such a graph is rejected before composition ever runs.
#
# 5. Backward is part of the computation. Yes: if you trace ``forward -> loss ->
#    backward``, running the artifact re-runs that whole computation and puts the
#    resulting parameter gradients onto the runtime model. Concretely: the parameter
#    gradients are harvested inside the (functional) graph as extra outputs, and the
#    driver scatters them back onto the runtime model's ``parameters()`` ``.grad``
#    fields -- ACCUMULATING (``p.grad += g``), not overwriting, exactly like eager
#    ``.backward()``, so a ``zero_grad()`` / ``optimizer.step()`` loop works unchanged
#    (skip the zero and grads pile up, by design). WHICH params get a grad is fixed at
#    TRACE time, not runtime: only params that actually received a gradient during the
#    traced backward are harvested (recorded by index in GRAD_PARAM_INDICES); a frozen
#    (``requires_grad=False``) or non-contributing param keeps ``.grad = None``, exactly
#    as eager leaves it -- precompile does NOT zero-fill such params, and flipping a
#    param's requires_grad at runtime does not change what gets scattered (invariant 2).
#    Buffers are never harvested (a requires_grad buffer that got a grad is rejected at
#    capture). The artifact therefore returns ``fn``'s own result (``None`` for a bare
#    ``.backward()`` step), not the grads. The grad scatter is the ONLY mutation
#    precompile performs, and it happens in Python outside the graph, so the graph stays
#    functional. precompile does not own optimizer state; bring your own optimizer and
#    zero grads as usual.
#
# 6. Shapes are static by default (dynamic dims are opt-in via mark_unbacked, invariant
#    3), each input's dtype/device is baked, and the inductor backend also specializes
#    on input layout. Each dense user-input leaf's dtype and device are recorded at
#    capture and checked at runtime (both backends): a dtype- or device-mismatched input
#    is rejected with a PrecompileError rather than crashing deep in a kernel or reading
#    a wrong value. The graph is specialized to the example input shapes (invariant 3);
#    tensor-subclass outputs in particular are rebuilt with constant outer sizes/strides,
#    so a different runtime shape is undefined. The inductor backend ADDITIONALLY bakes
#    each read input's stride / memory format (it emits assert_size_stride) -- and this
#    applies to model PARAMETERS/BUFFERS too, not only user inputs, since they are graph
#    inputs the kernels read. So a same-shape runtime input OR a same-shape/same-dtype
#    checkpoint WEIGHT with a DIFFERENT layout (e.g. a contiguous tensor when the example
#    was transposed or channels_last, or a non-contiguous view of a weight) is rejected
#    with a clear PrecompileError; match the example layout or use backend='eager'.
#    This guard is deliberately CONSERVATIVE: a layout-agnostic kernel (e.g. matmul) may
#    well have computed the right answer on the new layout, but precompile cannot
#    recompile to specialize it the way torch.compile does, so it rejects to stay safe
#    rather than risk a silently-wrong result from a layout-sensitive kernel. Pass inputs
#    in the example's layout (``.contiguous()`` to match a contiguous example), or use the
#    layout-flexible eager backend. ENFORCED for read inputs (a layout mismatch raises
#    rather than crashing in assert_size_stride or reading wrong strides).
#
# 7. Both python_code and the cache are trusted, EXECUTABLE input to load(). The cache
#    outer envelope is a plain {"artifact": bytes, ...} dict (read with
#    weights_only=True) carrying a format/version + backend tag AND a code_hash
#    (sha256 of the python_code it accelerates) that load() verifies (raising
#    PrecompileError on mismatch). load() feeds those bytes to
#    torch.compiler.load_cache_artifacts to PRIME the inductor kernel caches, then always
#    EXECs python_code -- with the caches primed the kernels load from the precompiled
#    binaries instead of JIT-compiling. Both the cache priming (it unpickles) and the exec run
#    code you supplied; treat both python_code and the cache like code you are about to
#    run. The code_hash binds the cache to its python_code:
#    load() rejects a make_fx (code, cache) pair from different precompile() calls
#    (same backend) rather than silently running the cache's graph under foreign
#    metadata. Only load_files (the on-disk pair a stateful capture rewrites)
#    degrades such a mismatch to a cold cache with a warning: python_code is fully
#    self-contained, and a rewrite interrupted between its two renames leaves
#    exactly such a pair.
#
# self-contained: ``python_code`` runs on its own -- it inlines the composed graph
# module (inductor: kernels JIT-compiled on first call, plus AOTAutograd's codegen'd
# prelude/epilogue) or the captured graph (eager), plus all calling-convention
# metadata. It NEVER reads the cache, and it is the SINGLE SOURCE OF TRUTH for the
# calling convention. The ``cache`` holds ONLY the compiled INDUCTOR artifact and is
# purely an ACCELERATION consumed only by load(): load AST-scrapes the module-level
# calling convention out of python_code, primes the inductor kernel caches from the bundle
# (torch.compiler.load_cache_artifacts), then execs python_code -- so its kernels load
# from the precompiled binaries instead of JIT. With the cache you skip JIT; with only
# python_code you JIT -- same results either way. The
# eager backend has no kernels to accelerate, so the eager cache carries no compiled
# artifact (artifact=None) but is still a full integrity-tagged envelope, and load()
# always runs the graph inlined in python_code. The metadata
# lives in one place (python_code); the envelope carries a code_hash (sha256 of
# python_code) alongside the format/version + backend tag, so load() rejects a
# (python_code, cache) pair that did not come from the same precompile() call.
#
# backend: "inductor" (default) lowers the captured graph through
# torch._functorch.aot_autograd.compile_to_python (AOTAutograd + Inductor, emitting a
# self-contained module). "eager" skips lowering and runs the captured
# ATen graph as-is (analogous to torch.compile(backend="eager")), for inspecting or
# debugging exactly what was traced. The contract above is identical for both
# backends with ONE exception (invariant 6): the inductor backend additionally
# specializes on each input's stride / memory format, while the eager backend is
# layout-flexible. Otherwise the same graph is captured; only its realization differs.
# Two mechanical consequences: the eager backend runs the graph directly on the
# (subclass-level) inputs, so it does not exercise the dense subclass
# flatten/unflatten path that the inductor backend's calling convention requires;
# and because there are no kernels, the eager cache carries no compiled artifact
# (artifact=None) but is still a full integrity-tagged envelope (python_code is the
# whole runnable artifact).
#
# tracer: the capture front-end, orthogonal to backend. "make_fx" (default) is a
# non-strict trace. "dynamo" analyzes Python bytecode, captures one variant per
# specialization/recompilation exercised by example_inputs, filters each variant's
# guards by provenance as Dynamo creates them (input-derived guards kept, named
# environment guards dropped), and dispatches among the variants at runtime,
# newest first. It currently requires one full graph (no graph breaks).
#
# The dynamo tracer's artifact format depends on these torch._dynamo internals
# (changes to them are format/behavior changes here): package.CompilePackage /
# load_guards_state / load_guard_manager / SerializedCode and the guards_state
# bytes CompilePackage records, the guard_filter_fn hook of torch._dynamo.optimize
# with types.GuardFilterEntry / Guard.provenance (Note [Guard provenance]), the
# _backend_id Dynamo stamps on each graph it hands a backend and the
# _dynamo_source it stamps on placeholders, output_graph.global_state_guard
# (GlobalStateGuard.__getstate__/__setstate__ JSON, rewritten at load),
# GuardManagerWrapper.check_verbose, exc.GuardSerializationError under
# config.strict_precompile, pgo._use_code_state, and the teardown entry points
# eval_frame.remove_cached_backend / utils.clear_guard_failures_for_code.

from __future__ import annotations

import contextlib
import hashlib
import io
import logging
import threading
import weakref
from types import MappingProxyType
from typing import Any, cast, NamedTuple, NewType, TYPE_CHECKING
from typing_extensions import Self

import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


log = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    from torch._dynamo.types import GuardFilterEntry
    from torch._subclasses.fake_tensor import FakeTensorMode


# ``precompile`` and ``PrecompileError`` are exposed under the compiler namespace as
# ``torch.compiler.precompile`` / ``torch.compiler.precompile.PrecompileError``
# (re-exported from torch/compiler/__init__.py and registered in
# ``torch.compiler.__all__``); they are deliberately kept out of this private module's
# ``__all__`` so test_public_bindings sees a consistent single public location.
__all__: list[str] = []


# Integrity tag baked into the cache envelope and verified by load() (with the
# code_hash) to reject a foreign / mismatched cache; see Note [precompile programming
# model], invariant 7.
_CACHE_FORMAT = "torch.compiler.precompile"
_CACHE_VERSION = 1
_DYNAMO_COMPILE_LOCK = threading.RLock()


# Index into the caller's positional nn.Module arguments (0-based over the modules,
# not over all args), used to qualify tied-across-modules param/buffer names as m<i>.<n>.
_ModuleIndex = NewType("_ModuleIndex", int)


# Decoded mark_unbacked spec for one dim: (shape_id, min, max, hint_override). shape_id is
# an opaque hashable grouping label (dims sharing it collapse to one unbacked symbol); the
# other three are optional integer sizes and are None wherever the decorator left them unset.
_MarkSpec = tuple[object, int | None, int | None, int | None]
# Per-user-input-leaf runtime bounds harvested from the marks: {dim: (min, max)} (either
# may be None), or None when the leaf has no bounded marked dim.
_LeafBounds = dict[int, tuple[int | None, int | None]] | None


# Reused read-only empty mapping for the mark_unbacked getattr fallbacks (Note [precompile
# reads private dynamo mark attributes]), so the common unmarked leaf reads its (absent)
# _dynamo_* dicts without allocating a throwaway {} per call.
# The value type is Any because these back three DISTINCT private dynamo dicts (dim ->
# shape_id label / (min, max) tuple / hint int); one shared empty default cannot name all
# three, and the private _dynamo_* attrs are untyped (Note above), so Any is the isolated
# boundary here.
_NO_MARKS: Mapping[int, Any] = MappingProxyType({})


class PrecompileError(RuntimeError):
    """The error type raised by ``torch.compiler.precompile`` and its artifacts.

    Raised when capture, lowering, ``load``, or a runtime call violates the precompile
    contract -- e.g. a tensor baked as a constant (invariant 1), an unsupported /
    effectful op, a non-tensor output the inductor backend cannot lower, or a runtime
    input whose shape or memory format differs from the example (invariants 3 and 6).
    See Note [precompile programming model] in this module for the full contract.
    """


def _dense_shape(t: object) -> tuple[int, ...] | None:
    """Return the shape of a plain dense tensor, else ``None`` (non-tensor / subclass).

    Tensor subclasses (e.g. DTensor) go through AOTAutograd's flatten path, so their
    outer shape is not the dense shape the inductor artifact bakes; record ``None`` and
    skip them in the shape check.
    """
    if isinstance(t, torch.Tensor) and not is_traceable_wrapper_subclass(t):
        return tuple(t.shape)
    return None


def _dense_dtype(t: object) -> str | None:
    """Return the dtype of a plain dense tensor as a string, else ``None``.

    Recorded as a string (e.g. ``"torch.float32"``) so it serializes into the artifact
    metadata as a literal and compares cleanly against ``str(t.dtype)`` at runtime;
    mirrors the _dense_shape convention (None for non-tensor / subclass leaves). The
    graph is specialized to the example dtype (invariant 6).
    """
    if isinstance(t, torch.Tensor) and not is_traceable_wrapper_subclass(t):
        return str(t.dtype)
    return None


def _dense_device(t: object) -> str | None:
    """Return the device (as a string) of a plain dense tensor, else ``None``.

    Recorded as a string so it serializes into the artifact metadata as a literal and
    compares cleanly at runtime; mirrors _dense_shape (None for non-tensor / subclass
    leaves). The graph is specialized to the example device (invariant 6).
    """
    if isinstance(t, torch.Tensor) and not is_traceable_wrapper_subclass(t):
        return str(t.device)
    return None


def _resolved_get_attrs(
    gm: torch.fx.GraphModule,
) -> list[tuple[str, object]]:
    """Return ``(target, attr)`` for every ``get_attr`` node, resolving dotted
    qualnames the same way for both capture guards below (missing attr -> None)."""
    resolved = []
    for node in gm.graph.find_nodes(op="get_attr"):
        attr: object = gm
        for part in node.target.split("."):
            attr = getattr(attr, part, None)
        resolved.append((node.target, attr))
    return resolved


# Note [precompile reads private dynamo mark attributes]
#
# The functions below read PRIVATE per-tensor attributes that
# torch._dynamo.decorators.mark_unbacked stamps onto a tensor: it consumes
# _dynamo_unbacked_indices / _dynamo_strict_unbacked_indices / _dynamo_shape_ids /
# _dynamo_unbacked_bounds / _dynamo_hint_overrides, and rejects _dynamo_dynamic_indices
# / _specialize_on (marks it cannot honor). This is a deliberate coupling to a private
# dynamo contract -- mark_unbacked is the documented entry point, and precompile reads
# what it leaves behind rather than exposing its own dynamic-shape kwarg. A stable
# dynamo-owned accessor is the eventual home; until then these names are load-bearing.
def _has_unbacked_marks(args: tuple[object, ...]) -> bool:
    """True if any tensor reachable in ``args`` carries a mark_unbacked dim (backed or
    strict)."""
    return any(
        isinstance(t, torch.Tensor)
        and (
            getattr(t, "_dynamo_unbacked_indices", None)
            or getattr(t, "_dynamo_strict_unbacked_indices", None)
        )
        for t in pytree.tree_leaves(args)
    )


def _reject_unsupported_marks(user_flat: list[object]) -> None:
    """Reject mark options precompile cannot honor, loudly (invariant 3).

    precompile only honors mark_unbacked (backed unbacked dims) and mark_unbacked's
    strict variant. Backed dynamic marks (mark_dynamic -> _dynamo_dynamic_indices) and
    per-dim specialization (_specialize_on) have no analogue in the static/unbacked
    capture path -- silently dropping them would bake a wrong artifact, so reject rather
    than ignore. (mark_unbacked's hint_override is NOT rejected: it is a perf-only
    autotuning size hint, never a guard, so the single artifact is valid regardless; it
    is threaded into the capture ShapeEnv in _fakeify_with_unbacked.) A mark_unbacked dim
    on a tensor SUBCLASS (e.g. DTensor) is rejected: the dynamic capture cannot preserve
    the subclass through the refake, so it too would bake a wrong artifact.
    """
    for t in user_flat:
        if not isinstance(t, torch.Tensor):
            continue
        # mark_unbacked on a tensor subclass (e.g. DTensor) stamps its marks on the OUTER
        # subclass as well as the inner tensor, so precompile's dynamic path picks it up --
        # but _fakeify_with_unbacked refakes a marked leaf via torch.empty, which yields a
        # plain dense tensor and DROPS the subclass, so the trace would run on the wrong
        # type. Reject loudly here rather than silently capturing a subclass-stripped tensor
        # (mirrors the decorator itself, which raises for every non-DTensor subclass).
        if is_traceable_wrapper_subclass(t) and (
            getattr(t, "_dynamo_unbacked_indices", None)
            or getattr(t, "_dynamo_strict_unbacked_indices", None)
        ):
            raise PrecompileError(
                "precompile: an input is a tensor subclass (e.g. DTensor) with a "
                "mark_unbacked dynamic dim, which precompile cannot honor: the dynamic "
                "capture cannot preserve the subclass. Mark a dense input instead, or "
                "capture that dim static (do not mark_unbacked it)."
            )
        if getattr(t, "_dynamo_dynamic_indices", None):
            raise PrecompileError(
                "precompile: an input has a mark_dynamic (backed dynamic) dim, which "
                "precompile cannot honor; it supports only mark_unbacked dynamic dims. "
                "Use torch._dynamo.decorators.mark_unbacked, or leave the dim static."
            )
        specialize_on = getattr(t, "_specialize_on", None)
        if specialize_on and any(v for v in specialize_on.values()):
            raise PrecompileError(
                "precompile: an input has a mark_unbacked specialize_on list, which "
                "precompile cannot honor (it produces a single artifact, not per-value "
                "specializations). Remove specialize_on."
            )


def _read_unbacked_marks(user_flat: list[object]) -> list[dict[int, _MarkSpec]]:
    """Read ``torch._dynamo.decorators.mark_unbacked`` marks off the user-input tensors.

    Dynamic shapes are opt-in via that decorator (the caller marks dims before calling
    precompile), NOT via a precompile kwarg -- so the precompile signature stays simple.
    Returns a per-leaf list aligned to ``user_flat``; each entry maps a marked dim to
    ``(shape_id, min, max, hint_override)`` (None when unset), empty when the leaf has no
    marks. Dims sharing a ``shape_id`` get the SAME unbacked symbol (so they are equal by
    construction); ``min``/``max`` become runtime range asserts; ``hint_override`` is a
    perf-only autotuning size hint applied to the symbol in _fakeify_with_unbacked.
    """
    marks: list[dict[int, _MarkSpec]] = []
    for t in user_flat:
        if not isinstance(t, torch.Tensor):
            marks.append({})
            continue
        # Union the non-strict and strict unbacked index sets. mark_unbacked(strict=True)
        # records ONLY _dynamo_strict_unbacked_indices; precompile already enforces
        # strict's error-on-specialize semantics via the GuardOnDataDependentSymNode ->
        # PrecompileError path, so both are honored identically here. NOTE: the decorator's
        # strict branch returns early, so a strict dim carries no shape_id/min/max/
        # hint_override (those are dropped at mark time) -- combine strict with shape_id/
        # min/max only if that limitation is acceptable; use non-strict to get them.
        idx = set(getattr(t, "_dynamo_unbacked_indices", None) or ())
        idx |= set(getattr(t, "_dynamo_strict_unbacked_indices", None) or ())
        if not idx:
            marks.append({})
            continue
        shape_ids = getattr(t, "_dynamo_shape_ids", _NO_MARKS) or _NO_MARKS
        bounds = getattr(t, "_dynamo_unbacked_bounds", _NO_MARKS) or _NO_MARKS
        hints = getattr(t, "_dynamo_hint_overrides", _NO_MARKS) or _NO_MARKS
        marks.append(
            {
                d: (shape_ids.get(d), *bounds.get(d, (None, None)), hints.get(d))
                for d in idx
            }
        )
    return marks


def _read_input_bounds(marks: list[dict[int, _MarkSpec]]) -> list[_LeafBounds]:
    """Build the per-leaf runtime min/max bounds from the already-read mark_unbacked
    marks, aligned to ``user_flat`` (so ``marks`` is the output of _read_unbacked_marks).

    mark_unbacked promises (in its own docstring) a runtime check that the dim is >= min
    and <= max; those bounds are applied as capture-time torch._check constraints in
    _fakeify_with_unbacked, but unbacked symints cannot be guarded on, so they never
    become a runtime guard on their own. We record them here so the driver enforces them.
    Each entry is None when the leaf has no bounded marked dim, else a dict mapping a
    marked dim index to ``(lo, hi)`` (either may be None); mirrors USER_INPUT_DTYPES.
    """
    bounds: list[_LeafBounds] = []
    for per in marks:
        per_leaf: dict[int, tuple[int | None, int | None]] = {}
        for d, (_shape_id, lo, hi, _hint) in per.items():
            if lo is not None or hi is not None:
                per_leaf[d] = (lo, hi)
        bounds.append(per_leaf or None)
    return bounds


def _detect_memory_format(t: torch.Tensor) -> torch.memory_format:
    """Return the example leaf's memory format so a refaked marked input preserves it.

    A mark_unbacked dim refakes the leaf via torch.empty; defaulting to contiguous would
    bake a contiguous assert_size_stride and reject a channels_last / transposed input
    even at its own layout. Probe the recognized formats and raise on an exotic /
    ambiguous layout we cannot capture rather than silently forcing contiguous.
    """
    if t.is_contiguous(memory_format=torch.contiguous_format):
        return torch.contiguous_format
    if t.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    if t.is_contiguous(memory_format=torch.channels_last_3d):
        return torch.channels_last_3d
    raise PrecompileError(
        "precompile: a mark_unbacked input has a memory format that is neither "
        "contiguous, channels_last, nor channels_last_3d (e.g. a transposed or "
        "otherwise non-standard layout); the dynamic-shape capture cannot preserve it. "
        "Pass the input in one of those layouts (.contiguous() to make it contiguous), "
        "or capture the dim static (do not mark_unbacked it)."
    )


def _fakeify_with_unbacked(
    pb_flat: list[Tensor], user_flat: list[object], marks: list[dict[int, _MarkSpec]]
) -> tuple[list[object], FakeTensorMode]:
    """Fakeify the flat capture inputs for an unbacked dynamic-shape capture.

    Params/buffers and unmarked dims become static fakes; each mark_unbacked dim becomes
    an UNBACKED SymInt (unguardable, so the artifact is valid for any runtime size and a
    graph that needs to guard on it fails at capture). Dims sharing a ``shape_id`` reuse
    one symbol; ``min``/``max`` add runtime asserts. Returns ``(flat_fake, fake_mode)``;
    the fake_mode (ShapeEnv) is threaded to the lowering via from_tracing_context.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    fake_mode = FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True)
    # shape_id -> unbacked symint (a dynamic SymInt); untyped so grouped dims share one symbol.
    shared: dict[object, Any] = {}
    with fake_mode:
        fake_pb = [fake_mode.from_tensor(t, static_shapes=True) for t in pb_flat]
        fake_user: list[object] = []
        for leaf, per in zip(user_flat, marks):
            if not isinstance(leaf, torch.Tensor):
                fake_user.append(leaf)
            elif not per:
                fake_user.append(fake_mode.from_tensor(leaf, static_shapes=True))
            else:
                sizes: list[Any] = []  # mix of static ints and unbacked SymInts
                for i, s in enumerate(leaf.shape):
                    if i not in per:
                        sizes.append(int(s))
                        continue
                    shape_id, lo, hi, hint = per[i]
                    if shape_id is not None and shape_id in shared:
                        u = shared[shape_id]
                        # Reusing the shared symbol still applies THIS occurrence's
                        # bounds: distinct dims grouped by shape_id may each carry their
                        # own (min, max), and dropping them would lose a runtime assert.
                        if lo is not None:
                            torch._check(u >= lo)
                        if hi is not None:
                            torch._check(u <= hi)
                        sizes.append(u)
                        continue
                    u = shape_env.create_unbacked_symint()
                    torch._check(u >= 0)
                    if lo is not None:
                        torch._check(u >= lo)
                    if hi is not None:
                        torch._check(u <= hi)
                    # hint_override is a perf-only autotuning size hint (not a guard):
                    # thread it onto the fresh symbol so inductor autotuning sees it. For a
                    # shared shape_id the group's one symbol keeps the first hint set here.
                    if hint is not None:
                        shape_env._set_unbacked_var_to_hint_override(u, hint)
                    if shape_id is not None:
                        shared[shape_id] = u
                    sizes.append(u)
                memory_format = _detect_memory_format(leaf)
                f = torch.empty(
                    sizes,
                    dtype=leaf.dtype,
                    device=leaf.device,
                    memory_format=memory_format,
                )
                f.requires_grad_(leaf.requires_grad)
                fake_user.append(f)
    return [*fake_pb, *fake_user], fake_mode


def _check_no_constant_tensors(gm: torch.fx.GraphModule) -> None:
    """Enforce invariant 1 of Note [precompile programming model]: everything live
    is an input.

    Every legitimate tensor in a non-strict capture is a placeholder (a lifted
    parameter/buffer or user input) or the result of a ``call_function`` node.
    A ``get_attr`` pointing at a tensor therefore means some tensor was closed
    over (a global, captured local, or non-registered module attribute) and would
    be baked into the graph as a constant, which we forbid.
    """
    offending = [
        (target, tuple(attr.shape), str(attr.dtype))
        for target, attr in _resolved_get_attrs(gm)
        if isinstance(attr, torch.Tensor)
    ]
    if offending:
        raise PrecompileError(
            "precompile traced a tensor that is neither a graph input "
            "(module parameter/buffer or user input) nor an intermediate. Such "
            "tensors would be hard-coded into the graph. This fires for a tensor "
            "closed over by fn (a global or captured local) or a plain "
            "(non-registered) module attribute, and also for a tensor literal "
            "constructed inside fn (e.g. torch.tensor([...])). Offending constants "
            f"(target, shape, dtype): {offending}. Fix by passing the tensor as an "
            "explicit argument; for module state register it as a parameter/buffer, "
            "and for a literal hoist it out of fn and pass it as an argument."
        )


def _assert_no_control_flow_subgraphs(gm: torch.fx.GraphModule) -> None:
    """Reject captured control-flow HOP subgraphs (e.g. from ``torch.cond``).

    They appear as ``get_attr`` nodes pointing at nested ``GraphModule`` submodules.
    The eager backend inlines ``gm.code`` and cannot reach such submodules (they are
    not on the standalone ``_GraphSelf`` holder), and the standalone composition does
    not inline them either, so the artifact would crash at runtime. Fail at capture
    with a concrete reason instead, like ``_assert_supported``.
    """
    offending = [
        target
        for target, attr in _resolved_get_attrs(gm)
        if isinstance(attr, torch.fx.GraphModule)
    ]
    if offending:
        raise PrecompileError(
            "precompile cannot lower a captured control-flow subgraph (e.g. from "
            f"torch.cond / torch.while_loop); not supported yet. Offending get_attr "
            f"targets: {offending}."
        )


def _intern_param_buffers(
    mods: list[torch.nn.Module],
) -> tuple[
    list[Tensor], list[str], list[str], list[tuple[_ModuleIndex, str, int]], int
]:
    """Lift each module's parameters then buffers to a flat list, interning by
    tensor identity so a tied weight becomes a single entry (one optimizer step,
    accumulated gradient -- not one per name).

    Returns ``(pb_flat, param_names, buffer_names, alias_entries, num_params)``,
    where ``alias_entries`` maps each ``(module_index, name)`` to its index in
    ``pb_flat`` (used to reparametrize during capture). This same params-then-
    buffers, intern-by-identity order is reproduced at runtime against the
    user-supplied modules, so the dense list lines up with the compiled graph.

    INVARIANT: the all-modules' params then all-modules' buffers, dedup-by-id ordering
    here is load-bearing and is reproduced VERBATIM by
    ``torch._precompile_driver._extract_param_buffers`` (emitted into the inlined/eager
    load paths). The cached load path uses this function directly, so both must stay
    in sync; ``test_cached_and_inlined_paths_agree`` cross-checks them.
    """
    if len(mods) > 1:

        def _name(mi: _ModuleIndex, n: str) -> str:
            return f"m{mi}.{n}"
    else:

        def _name(mi: _ModuleIndex, n: str) -> str:
            return n

    unique: list[Tensor] = []
    id_to_uidx: dict[int, int] = {}
    alias_entries: list[tuple[_ModuleIndex, str, int]] = []

    def _intern(mi: _ModuleIndex, n: str, t: Tensor, names_out: list[str]) -> None:
        uidx = id_to_uidx.get(id(t))
        if uidx is None:
            uidx = len(unique)
            id_to_uidx[id(t)] = uidx
            unique.append(t)
            names_out.append(_name(mi, n))
        alias_entries.append((mi, n, uidx))

    param_names: list[str] = []
    for mi, m in enumerate(mods):
        for n, p in m.named_parameters(remove_duplicate=False):
            _intern(_ModuleIndex(mi), n, p, param_names)
    num_params = len(unique)
    buffer_names: list[str] = []
    for mi, m in enumerate(mods):
        for n, b in m.named_buffers(remove_duplicate=False):
            _intern(_ModuleIndex(mi), n, b, buffer_names)
    return unique, param_names, buffer_names, alias_entries, num_params


def _capture(
    fn: Callable[..., object],
    args: tuple[object, ...],
    decompositions: dict | None = None,
) -> _Capture:
    """Trace the computation ``fn(*args)`` to an ATen graph.

    See Note [precompile programming model] for the contract. ``fn`` is the whole
    computation, e.g. ``lambda model, x: model(x)`` or a training step
    ``lambda model, x, t: loss_fn(model(x), t).backward()``. Among ``args``, the
    ``nn.Module`` arguments have their parameters/buffers lifted to explicit graph
    inputs (via reparametrization, so nothing is baked -- invariant 1); the
    remaining arguments are the runtime inputs. Whatever ``fn`` returns becomes the
    graph's result outputs, and if ``fn`` ran a backward, the resulting parameter
    gradients (read off ``param.grad``) are harvested as additional, trailing graph
    outputs. They are kept separate from the result so the driver can scatter them
    onto the runtime model's ``.grad`` fields rather than return them (invariant 5).

    This is a NON-STRICT trace (invariant 3): make_fx records only the ATen ops
    that run for THIS example. Python-level control flow over tensor values, data-
    dependent branches, and shapes are specialized to ``args`` and baked. The
    interning/order established here for params then buffers is the calling
    convention the runtime model must reproduce (invariant 2).
    """
    import contextlib

    args = tuple(args)
    module_positions = [i for i, a in enumerate(args) if isinstance(a, torch.nn.Module)]
    module_pos_set = set(module_positions)
    mods = [a for a in args if isinstance(a, torch.nn.Module)]
    user_inputs = tuple(a for i, a in enumerate(args) if i not in module_pos_set)

    # Lift the example modules' params/buffers for tracing only. Their VALUES are
    # never stored in the cache -- the user passes the model(s) again at runtime
    # (mirroring fn's signature), and the same interning is reproduced there.
    pb_flat, param_names, buffer_names, alias_entries, num_params = (
        _intern_param_buffers(mods)
    )
    num_pb = len(pb_flat)
    # Record each interned param's / buffer's example SHAPE, DTYPE, and DEVICE (aligned to
    # param_names / buffer_names) so the structural check (invariant 2) compares not just
    # names but also each runtime tensor's shape, dtype, and device. The graph is specialized
    # to the example param/buffer shapes (and can bake a device literal via a factory op), so
    # a same-named runtime tensor with a different shape / dtype / device would otherwise
    # silently compute the wrong thing (eager has no assert_size_stride backstop).
    param_shapes = [tuple(t.shape) for t in pb_flat[:num_params]]
    buffer_shapes = [tuple(t.shape) for t in pb_flat[num_params:]]
    param_dtypes = [str(t.dtype) for t in pb_flat[:num_params]]
    buffer_dtypes = [str(t.dtype) for t in pb_flat[num_params:]]
    param_devices = [str(t.device) for t in pb_flat[:num_params]]
    buffer_devices = [str(t.device) for t in pb_flat[num_params:]]

    user_flat, in_spec = pytree.tree_flatten(user_inputs)
    # Reject mark options precompile cannot honor (mark_dynamic, specialize_on) loudly
    # here, before tracing, rather than silently dropping them. (hint_override is honored,
    # not rejected -- it is a perf-only autotuning hint threaded onto the capture symbol.)
    _reject_unsupported_marks(user_flat)
    flat_args = [*pb_flat, *user_flat]
    # The REAL example tensors (params/buffers and user inputs). flat_args is reassigned
    # to FAKE tensors in the unbacked path below, but the saved-grad snapshot/clear/restore
    # block must protect the real example model's .grad fields (those are what the user
    # owns and what a backward in fn populates), not the throwaway fakes. list() snapshots
    # the real tensors here, so the later flat_args rebind does not affect real_flat.
    real_flat = list(flat_args)
    # Record the example user inputs' dense shapes/dtypes/devices so the drivers can
    # reject a shape (invariant 3) or dtype/device (invariant 6) mismatch up front; see
    # the inlined driver checks (torch._precompile_driver). Stride is NOT recorded --
    # memory-format mismatches are enforced by inductor's own (pinned-on)
    # assert_size_stride. Subclasses -> None.
    # Widened element type (a marked-dynamic dim becomes None within the tuple in the
    # unbacked path below); _dense_shape's static tuples conform to it.
    user_input_shapes: list[tuple[int | None, ...] | None] = [
        _dense_shape(t) for t in user_flat
    ]
    user_input_dtypes = [_dense_dtype(t) for t in user_flat]
    user_input_devices = [_dense_device(t) for t in user_flat]

    # Dynamic shapes (opt-in, UNBACKED only): dims the caller tagged with
    # torch._dynamo.decorators.mark_unbacked are refakeified as unbacked symints, then
    # traced symbolically with the fake_mode's ShapeEnv threaded to the lowering. Unbacked
    # dims cannot be guarded on, so the artifact is valid across runtime sizes; a graph
    # that would need to guard on a marked dim fails loudly at capture
    # (GuardOnDataDependentSymNode) rather than baking it. Reading the marks here (instead
    # of a precompile kwarg) keeps the precompile signature simple.
    marks = _read_unbacked_marks(user_flat)
    # Record each marked dim's declared min/max so the driver enforces them at runtime;
    # the capture-time torch._check on an unbacked symint never becomes a runtime guard,
    # so without this the documented mark_unbacked min/max check would be a silent no-op.
    user_input_bounds = _read_input_bounds(marks)
    # Snapshot and clear the REAL example tensors' .grad BEFORE fakeifying and tracing.
    # A backward in fn accumulates (``p.grad = p.grad + new``), so a live pre-existing
    # grad would be read into the graph and baked by make_fx as a get_attr constant --
    # tripping the invariant-1 guard with a misleading "tensor closed over by fn" error on
    # the common warmup-step-then-precompile flow. The clear MUST precede
    # _fakeify_with_unbacked: fake_mode.from_tensor copies .grad onto the fakes we trace
    # on, so clearing the reals first keeps the fakes grad-free too. Restored in finally;
    # precompile does not mutate the user's example .grad (params/buffers AND user inputs).
    # Snapshot the ORIGINAL .grad object (no clone) and restore that SAME object below, so
    # grad IDENTITY is preserved -- a caller holding a prior p.grad reference, or optimizer
    # state keyed on grad identity, is not invalidated. The unbacked path traces on fakes,
    # so the reals' .grad is untouched there; the STATIC path (fake_mode is None) traces on
    # the real interned params, so a backward in fn DOES write .grad in place -- but onto a
    # fresh grad object, since .grad was snapshotted and cleared to None just above. The
    # finally-restore below puts the snapshotted object back, so both grad identity and
    # value are preserved regardless of which path ran.
    saved_grads = [a.grad if isinstance(a, torch.Tensor) else None for a in real_flat]
    for a in real_flat:
        if isinstance(a, torch.Tensor):
            a.grad = None
    fake_mode = None
    if any(marks):
        flat_args, fake_mode = _fakeify_with_unbacked(pb_flat, user_flat, marks)
        user_input_shapes = [
            None
            if base is None
            else tuple(None if i in per else s for i, s in enumerate(base))
            for base, per in zip(user_input_shapes, marks)
        ]

    # flat_fn (traced by make_fx) writes these back so _capture can thread the output
    # structure and the harvested-grad param indices into the _Capture result.
    captured_out_spec: pytree.TreeSpec | None = None
    captured_grad_param_indices: list[int] = []

    def flat_fn(flat: list[object]) -> list[object]:
        nonlocal captured_out_spec, captured_grad_param_indices
        # The pb region is entirely interned params/buffers (Tensors); the user region
        # (flat[num_pb:]) is arbitrary pytree leaves.
        pb = cast("list[Tensor]", flat[:num_pb])
        runtime_inputs = pytree.tree_unflatten(flat[num_pb:], in_spec)
        with contextlib.ExitStack() as stack:
            for mi, m in enumerate(mods):
                reparam = {n: pb[uidx] for emi, n, uidx in alias_entries if emi == mi}
                stack.enter_context(
                    stateless._reparametrize_module(m, reparam, tie_weights=True)
                )
            # Reconstruct fn's full positional args: reparametrized modules at
            # their original positions, runtime inputs at theirs.
            full: list[object] = []
            ui = 0
            for i in range(len(args)):
                if i in module_pos_set:
                    full.append(args[i])
                else:
                    full.append(runtime_inputs[ui])
                    ui += 1
            result = fn(*full)
            # Harvest parameter gradients produced by any backward in fn.
            param_proxies = pb[:num_params]
            harvested = [p.grad for p in param_proxies]
            # Buffers are not harvested (only params get scattered grads). A registered
            # buffer with requires_grad=True that received a gradient would be silently
            # dropped, so reject it -- a cheaply-knowable invariant-5 violation.
            if any(getattr(b, "grad", None) is not None for b in pb[num_params:]):
                raise PrecompileError(
                    "precompile: a registered buffer received a gradient (it has "
                    "requires_grad=True), but precompile only harvests gradients for "
                    "parameters. Register it as an nn.Parameter instead."
                )
            # User-input leaves are not harvested either (only params get scattered
            # grads), so a requires_grad user input that received a gradient during the
            # traced backward would be silently dropped. Reject it, mirroring the buffer
            # case -- another cheaply-knowable invariant-5 violation.
            if any(getattr(t, "grad", None) is not None for t in flat[num_pb:]):
                raise PrecompileError(
                    "precompile: a user input received a gradient; precompile only "
                    "harvests gradients for parameters, so an input gradient would be "
                    "silently dropped. Pass the tensor as a module parameter if its "
                    "gradient is needed."
                )

        # The result (fn's own return) and the harvested grads are kept as separate
        # output regions: the driver returns the result and scatters the grads onto
        # the runtime model's .grad fields. We emit a grad output ONLY for params that
        # actually received a gradient -- mirroring eager .backward(), which leaves
        # .grad = None for frozen / non-contributing params -- and record which unique
        # param index each emitted grad belongs to, so the driver scatters onto exactly
        # those params. grad_flat is empty when fn ran no backward.
        result_flat, result_spec = pytree.tree_flatten(result)
        grad_flat = []
        grad_param_indices = []
        for i, g in enumerate(harvested):
            if g is not None:
                grad_flat.append(g)
                grad_param_indices.append(i)
        captured_out_spec = result_spec
        captured_grad_param_indices = grad_param_indices
        return [*result_flat, *grad_flat]

    # Trace with grad enabled so any backward in ``fn`` is built as graph ops; the
    # forward graph is the same as under no_grad. Restore in finally so a make_fx
    # failure (e.g. fn raising after running a backward) does not leave the user's
    # example model with clobbered .grad fields.
    from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode

    tracing_mode = "symbolic" if fake_mode is not None else "real"
    capture_cm = fake_mode if fake_mode is not None else contextlib.nullcontext()
    try:
        with torch.enable_grad(), capture_cm:
            try:
                gm = make_fx(
                    flat_fn,
                    decomposition_table=decompositions,
                    tracing_mode=tracing_mode,
                )(flat_args)
            except GuardOnDataDependentSymNode as e:
                # A mark_unbacked dim was captured as an unbacked symint (no hint), but
                # the computation needs to guard on / specialize its size (e.g. a
                # shape-dependent branch or a reshape that pins it). Unbacked dims cannot
                # be guarded, so rather than bake a silently-wrong artifact, fail here.
                raise PrecompileError(
                    "precompile: fn needs to guard on a dim marked with mark_unbacked "
                    "(it branches on or specializes that size), which is not allowed for "
                    "an unbacked dynamic dim. Do not mark that dim (capture it static), "
                    "or restructure fn to avoid the size-dependent operation. Underlying: "
                    f"{str(e).splitlines()[0]}"
                ) from e
    finally:
        for a, g in zip(real_flat, saved_grads):
            if isinstance(a, torch.Tensor):
                a.grad = g
    _check_no_constant_tensors(gm)
    _assert_no_control_flow_subgraphs(gm)
    _assert_supported(gm)

    # flat_fn always runs during the make_fx trace above, so captured_out_spec is set.
    return _Capture(
        gm=gm,
        flat_args=flat_args,
        module_positions=module_positions,
        num_positional_args=len(args),
        param_names=param_names,
        buffer_names=buffer_names,
        param_shapes=param_shapes,
        buffer_shapes=buffer_shapes,
        param_dtypes=param_dtypes,
        buffer_dtypes=buffer_dtypes,
        param_devices=param_devices,
        buffer_devices=buffer_devices,
        in_spec=in_spec,
        out_spec=cast("pytree.TreeSpec", captured_out_spec),
        grad_param_indices=captured_grad_param_indices,
        user_input_shapes=user_input_shapes,
        user_input_dtypes=user_input_dtypes,
        user_input_devices=user_input_devices,
        user_input_bounds=user_input_bounds,
        fake_mode=fake_mode,
    )


class _Capture:
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        flat_args: list[object],
        module_positions: list[int],
        num_positional_args: int,
        param_names: list[str],
        buffer_names: list[str],
        param_shapes: list[tuple[int, ...]],
        buffer_shapes: list[tuple[int, ...]],
        param_dtypes: list[str],
        buffer_dtypes: list[str],
        param_devices: list[str],
        buffer_devices: list[str],
        in_spec: pytree.TreeSpec,
        out_spec: pytree.TreeSpec,
        grad_param_indices: list[int],
        user_input_shapes: list[tuple[int | None, ...] | None],
        user_input_dtypes: list[str | None],
        user_input_devices: list[str | None],
        user_input_bounds: list[_LeafBounds],
        fake_mode: FakeTensorMode | None = None,
    ) -> None:
        self.gm = gm
        self.flat_args = flat_args
        self.module_positions = module_positions
        self.num_positional_args = num_positional_args
        self.param_names = param_names
        self.buffer_names = buffer_names
        self.param_shapes = param_shapes
        self.buffer_shapes = buffer_shapes
        self.param_dtypes = param_dtypes
        self.buffer_dtypes = buffer_dtypes
        self.param_devices = param_devices
        self.buffer_devices = buffer_devices
        self.in_spec = in_spec
        self.out_spec = out_spec
        self.grad_param_indices = grad_param_indices
        self.user_input_shapes = user_input_shapes
        self.user_input_dtypes = user_input_dtypes
        self.user_input_devices = user_input_devices
        self.user_input_bounds = user_input_bounds
        # The fake_mode (with ShapeEnv) used for a dynamic-shape capture, threaded to the
        # lowering (dynamic_shapes="from_tracing_context"); None for a static capture.
        self.fake_mode = fake_mode


_GENERATED_HEADER = """\
# Generated by torch.compiler.precompile -- do not edit.
#
# This is a SELF-CONTAINED, EXECUTABLE artifact: it runs on its own, needing no
# companion cache. You provide the model(s) at runtime, exactly as the original fn
# took them, e.g.:
#
#     ns = {}
#     exec(open("this_file.py").read(), ns)
#     out = ns["forward"](model, my_input)      # same args as the traced fn
#
# The runtime model must be STRUCTURALLY IDENTICAL to the one precompile traced
# (same parameter/buffer names, order, and weight tying); only the weight VALUES
# may differ (swap in a checkpoint). This artifact was produced by a non-strict
# make_fx trace, so control flow and shapes are specialized to the example inputs,
# and (inductor backend) each input's stride / memory format is baked too: pass
# runtime inputs in the example's layout (.contiguous() to match a contiguous
# example). See Note [precompile programming model] in torch/_precompile.py.
#
# It contains, in order:
#   1. The composed graph module from aot_autograd.compile_to_python: the inlined
#      Inductor kernels (JIT-compiled from the embedded source on first use -- no
#      external cache required) plus AOTAutograd's own codegen'd prelude/epilogue
#      (tensor-subclass wrap/unwrap, input-mutation reflection, output aliasing),
#      exposing ``call(flat_inputs) -> outputs``.
#   2. Calling-convention metadata.
#   3. A small driver that extracts each runtime module's params/buffers (in the
#      same order as capture), passes them with the runtime inputs to ``call``, and
#      scatters any harvested gradients onto the model's .grad fields. No model
#      weights are embedded (you bring the model).
#
# The companion ``cache`` returned by precompile is purely an ACCELERATION used by
# torch.compiler.precompile.load: it primes the inductor kernel caches so exec'ing this
# file loads its kernels from the precompiled binaries (no JIT). This file does not read
# it; running this file alone just JITs.
"""


def _build_metadata_section(compiled: PrecompiledModule) -> list[str]:
    if compiled._out_spec is None or compiled._in_spec is None:
        raise PrecompileError("internal: cannot build metadata before _compile()")
    # OUT_SPEC is load-bearing: the driver rebuilds fn's output via tree_unflatten, so
    # unlike IN_SPEC it cannot degrade to None. If fn's output structure is not
    # JSON-serializable (an unregistered namedtuple, or a registered pytree node with a
    # non-JSON-dumpable context), fail with a clear PrecompileError rather than leaking
    # a raw pytree NotImplementedError/TypeError.
    try:
        out_spec_str = pytree.treespec_dumps(compiled._out_spec)
    except (NotImplementedError, TypeError) as e:
        raise PrecompileError(
            "precompile cannot serialize the output structure of fn (its pytree "
            "TreeSpec is not JSON-serializable). This fires when fn returns an "
            "unregistered collections.namedtuple, or a registered pytree node with a "
            "non-JSON-dumpable context. Register the namedtuple via "
            "torch.utils._pytree._register_namedtuple(...) (or supply a JSON-dumpable "
            "to_dumpable_context), or return a plain tuple/list/dict of tensors."
        ) from e
    # IN_SPEC drives the runtime input-structure check, but is best-effort: some specs
    # are not JSON-serializable -- an unregistered namedtuple raises NotImplementedError,
    # and a registered pytree node whose context is not JSON-dumpable (no
    # to_dumpable_context serializer, or one yielding non-JSON output) raises TypeError.
    # Such inputs still compile -- emit IN_SPEC = None and the driver skips the
    # structure check rather than regressing.
    try:
        in_spec_str: str | None = pytree.treespec_dumps(compiled._in_spec)
    except (NotImplementedError, TypeError):
        in_spec_str = None
    parts = [
        "# " + "=" * 70,
        "# 2. Calling-convention metadata",
        "# " + "=" * 70,
        "import torch as _torch",
        "import torch.utils._pytree as _pytree",
        "",
        # python_code is the single source of truth for the calling convention; the
        # cache holds ONLY the compiled/captured artifact. load() reads these
        # constants back out of python_code (see _parse_artifact_metadata).
        f"BACKEND = {compiled._backend!r}",
        f"MODULE_POSITIONS = {compiled._module_positions!r}",
        # Number of positional args the traced fn took (modules + runtime inputs); the
        # driver checks the runtime call passes the same count up front, so a wrong
        # arity raises a clear PrecompileError instead of a raw IndexError.
        f"NUM_POSITIONAL_ARGS = {compiled._num_positional_args}",
        f"PARAM_NAMES = {compiled._param_names!r}",
        f"BUFFER_NAMES = {compiled._buffer_names!r}",
        # Per interned param / buffer example shape / dtype / device (aligned to
        # PARAM_NAMES / BUFFER_NAMES); the driver checks each runtime param/buffer against
        # these for the structural contract (invariant 2).
        f"PARAM_SHAPES = {compiled._param_shapes!r}",
        f"BUFFER_SHAPES = {compiled._buffer_shapes!r}",
        f"PARAM_DTYPES = {compiled._param_dtypes!r}",
        f"BUFFER_DTYPES = {compiled._buffer_dtypes!r}",
        f"PARAM_DEVICES = {compiled._param_devices!r}",
        f"BUFFER_DEVICES = {compiled._buffer_devices!r}",
        # Which unique-param index each trailing grad output belongs to (see invariant 5);
        # the driver scatters grad k onto params[GRAD_PARAM_INDICES[k]].
        f"GRAD_PARAM_INDICES = {compiled._grad_param_indices!r}",
        # The pytree structure of the runtime inputs, or None if not serializable (the
        # driver validates against it when present, else skips the structure check).
        f"IN_SPEC = {in_spec_str!r}",
        f"OUT_SPEC = {out_spec_str!r}",
        # Per user-input-leaf example shape / dtype / device (None for a non-tensor /
        # subclass leaf); the drivers reject a runtime mismatch (invariants 3 and 6).
        # Memory-format mismatches are caught by the inductor artifact's own
        # assert_size_stride (pinned on at capture).
        f"USER_INPUT_SHAPES = {compiled._user_input_shapes!r}",
        f"USER_INPUT_DTYPES = {compiled._user_input_dtypes!r}",
        f"USER_INPUT_DEVICES = {compiled._user_input_devices!r}",
        # Per user-input-leaf mark_unbacked min/max bounds: None for a leaf with no bounded
        # marked dim, else {dim: (lo, hi)} (either may be None). The drivers reject a
        # runtime size outside the declared range (invariant 3); see the inlined drivers.
        f"USER_INPUT_BOUNDS = {compiled._user_input_bounds!r}",
        "",
    ]
    return parts


def _parse_artifact_metadata(python_code: str) -> dict[str, object]:
    """Read the calling-convention constants back out of ``python_code`` WITHOUT
    executing it (exec'ing the inlined Inductor output would JIT the kernels, the
    very work the cache exists to skip).

    python_code is the single source of truth: ``_build_metadata_section`` emits the
    constants below as top-level literal assignments, so an AST walk + literal_eval
    recovers them safely. The cache then only needs to carry the compiled artifact.
    """
    import ast

    make_fx_metadata = {
        "MODULE_POSITIONS",
        "NUM_POSITIONAL_ARGS",
        "PARAM_NAMES",
        "BUFFER_NAMES",
        "PARAM_SHAPES",
        "BUFFER_SHAPES",
        "PARAM_DTYPES",
        "BUFFER_DTYPES",
        "PARAM_DEVICES",
        "BUFFER_DEVICES",
        "GRAD_PARAM_INDICES",
        "IN_SPEC",
        "OUT_SPEC",
        "USER_INPUT_SHAPES",
        "USER_INPUT_DTYPES",
        "USER_INPUT_DEVICES",
        "USER_INPUT_BOUNDS",
    }
    # _DYNAMO_BACKEND_SOURCES is intentionally absent: its value is not a plain
    # literal (each entry is a subscripted string), so ast.literal_eval rejects it.
    dynamo_metadata = {
        "_DYNAMO_GRAD_ENABLED",
        "_DYNAMO_MUTATES_INPUTS",
        "_DYNAMO_PYTHON_VERSION",
        "_DYNAMO_TORCH_VERSION",
        "_DYNAMO_BACKEND_IDS",
        "_DROPPED_GUARDS",
        "_DYNAMO_STATE",
    }
    wanted = {"BACKEND", "TRACER", *make_fx_metadata, *dynamo_metadata}
    found: dict[str, object] = {}
    try:
        tree = ast.parse(python_code)
    except SyntaxError as e:
        raise PrecompileError(
            "python_code is not valid Python; it does not look like a "
            "torch.compiler.precompile artifact."
        ) from e
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id == "_DYNAMO_STATE":
            # Presence only: the opaque pickle literal is megabytes and consumed
            # by the exec'd driver, not here.
            found[target.id] = None
        elif target.id in wanted:
            try:
                found[target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError, RecursionError) as e:
                # A truncated or hand-edited artifact can leave a metadata name
                # assigned a non-literal; surface the documented error type
                # instead of ast's raw "malformed node" ValueError.
                raise PrecompileError(
                    f"python_code assigns non-literal metadata to {target.id!r}; "
                    "it does not look like a torch.compiler.precompile artifact."
                ) from e
        else:
            # Not a metadata name we consume (the driver section emits only
            # function defs today, but a future artifact revision could add a
            # driver-internal top-level assignment). Skipped by design, but log
            # it at debug so a malformed / renamed artifact is diagnosable
            # rather than silently dropped.
            log.debug(
                "precompile: ignoring unrecognized top-level assignment %r while "
                "parsing artifact calling-convention metadata",
                target.id,
            )
    tracer = found.get("TRACER", "make_fx")
    if tracer not in ("make_fx", "dynamo"):
        raise PrecompileError(
            f"python_code has an unsupported TRACER value {tracer!r}; it does not "
            "look like a compatible torch.compiler.precompile artifact."
        )
    if tracer == "make_fx":
        required = {"BACKEND", *make_fx_metadata}
    else:
        required = {"BACKEND", "TRACER", *dynamo_metadata}
    missing = required - found.keys()
    if missing:
        raise PrecompileError(
            f"python_code is missing calling-convention metadata {sorted(missing)}; "
            "it does not look like a torch.compiler.precompile artifact."
        )
    return found


def _build_python_source(
    compiled: PrecompiledModule,
    graph_python: str,
) -> str:
    parts = [_GENERATED_HEADER, ""]
    parts.append("# " + "=" * 70)
    parts.append("# 1. Compiled graph (AOTAutograd + Inductor): exposes ``call``")
    parts.append("# " + "=" * 70)
    # The composed graph module from aot_autograd.compile_to_python: the inlined
    # Inductor kernels plus AOTAutograd's codegen'd prelude/epilogue, exposing
    # ``call(flat_inputs) -> outputs`` (subclass + mutation handled inside).
    parts.append(graph_python)
    parts.append("")
    parts.extend(_build_metadata_section(compiled))
    parts.append("# " + "=" * 70)
    parts.append(
        "# 3. Driver: module params/buffers + grad scatter + calling convention"
    )
    parts.append("# " + "=" * 70)
    parts.append(_emit_driver_source("_inductor_forward"))
    return "\n".join(parts)


_EAGER_GENERATED_HEADER = """\
# Generated by torch.compiler.precompile (backend="eager") -- do not edit.
#
# Self-contained, executable artifact: the captured ATen graph is inlined below (both
# the human-readable rendering and the executable code) and runs on its own. Provide
# the model(s) at runtime, exactly as the original fn took them:
#
#     ns = {}
#     exec(open("this_file.py").read(), ns)
#     out = ns["forward"](model, my_input)      # same args as the traced fn
#
# The runtime model must be structurally identical to the traced one (only weight
# VALUES may differ), and control flow / shapes are specialized to the example inputs.
# See Note [precompile programming model] in torch/_precompile.py for the full contract.
"""


def _build_eager_python_source(compiled: PrecompiledModule) -> str:
    gm = compiled._gm
    # gm.code defines ``def forward(self, flat)`` that references fx_pytree / pytree
    # and self._in_spec / self._out_spec. Rename it so it does not collide with the
    # driver's public ``forward``, and supply the specs via a tiny holder object so
    # the inlined graph runs standalone.
    in_spec = gm._in_spec if gm is not None else None
    out_spec = gm._out_spec if gm is not None else None
    if gm is None or in_spec is None or out_spec is None:
        raise PrecompileError("internal: eager graph missing before _compile()")
    graph_src = gm.code.replace("def forward(", "def _graph_forward(", 1)
    in_spec_str = pytree.treespec_dumps(in_spec)
    out_spec_str = pytree.treespec_dumps(out_spec)
    parts = [_EAGER_GENERATED_HEADER, ""]
    parts.append("# " + "=" * 70)
    parts.append("# 1. Captured ATen graph (eager backend) -- executable and readable")
    parts.append("# " + "=" * 70)
    # gm.code relies on fx's custom builtins (torch, device, inf, nan, NoneType,
    # fx_pytree, pytree) being in scope -- fx injects them when a real GraphModule
    # runs. Reproduce the FULL set (not just torch/pytree) so a graph that bakes a
    # device / inf / nan constant (e.g. BatchNorm, masked_fill to -inf) runs
    # standalone instead of raising NameError. Sourced from fx so it stays correct.
    from torch.fx.graph import _custom_builtins

    for _cb in _custom_builtins.values():
        parts.append(_cb.import_str)
    parts.append(graph_src)
    parts.append("")
    parts.append("class _GraphSelf:")
    parts.append(f"    _in_spec = pytree.treespec_loads({in_spec_str!r})")
    parts.append(f"    _out_spec = pytree.treespec_loads({out_spec_str!r})")
    parts.append("")
    parts.append("")
    parts.append("def call(args):")
    parts.append("    out = _graph_forward(_GraphSelf(), list(args))")
    parts.append("    return list(out) if isinstance(out, (list, tuple)) else [out]")
    parts.append("")
    parts.extend(_build_metadata_section(compiled))
    parts.append("# " + "=" * 70)
    parts.append("# 3. Driver: run the inlined captured graph eagerly")
    parts.append("# " + "=" * 70)
    parts.append(_emit_driver_source("_eager_forward"))
    return "\n".join(parts)


_DRIVER_MAIN = """\
if __name__ == "__main__":
    print("forward() is ready; call it with the model(s) and inputs the traced")
    print("fn took, e.g. forward(model, x).")
"""


def _emit_driver_source(forward_fn_name: str) -> str:
    """Emit the runtime driver as text for inlining into python_code.

    The driver lives as real, type-checked code in torch._precompile_driver; here we read
    it back with inspect.getsource (LAZILY -- only on this emit path; load() never runs
    it, so a stripped-source environment only affects capture, not reload) and rename the
    selected forward variant to the public ``forward``. Emitting the TEXT (rather than
    importing the module from the artifact) keeps python_code self-contained and
    version-frozen (Note [precompile programming model], invariant 7)."""
    import inspect

    from torch import _precompile_driver as driver

    forward_fn = getattr(driver, forward_fn_name)
    blocks = [
        inspect.getsource(driver._extract_param_buffers),
        inspect.getsource(driver._fail),
        inspect.getsource(driver._check_structure),
        inspect.getsource(forward_fn).replace(
            f"def {forward_fn_name}(", "def forward(", 1
        ),
    ]
    body = "\n\n".join(block.rstrip() for block in blocks)
    return "\n" + body + "\n\n\n" + _DRIVER_MAIN


def _assert_supported(gm: torch.fx.GraphModule) -> None:
    """Enforce invariant 4 of Note [precompile programming model]: reject boundary
    effects the AOT backend's standalone composition does not handle. Detected
    directly from the captured graph -- no AOTAutograd coupling.

    Input mutation (incl. module buffers, e.g. BatchNorm running stats), tensor-
    subclass wrap/unwrap, output aliasing, and functionalized RNG are SUPPORTED:
    AOTAutograd's codegen'd prelude/epilogue is composed into the artifact (see
    torch._functorch.aot_autograd.compile_to_python), so they are not rejected here.

    Effectful ops are not supported yet (an implementation gap, not a fundamental
    limit), so raise here with a concrete reason rather than let the failure surface
    deep in the cache layer. See _unsupported for the mechanical cause.
    """
    from torch._higher_order_ops.effects import _get_effect

    for node in gm.graph.nodes:
        # Only ATen ops can be in the effect registry; skip plain call_functions
        # like operator.getitem (which _get_effect rejects).
        if node.op == "call_function" and isinstance(
            node.target, torch._ops.OpOverload
        ):
            if _get_effect(node.target) is not None:
                raise _unsupported(f"effectful op {node.target}")


def _unsupported(reason: str) -> PrecompileError:
    return PrecompileError(
        f"precompile cannot compile this computation: {reason}. The graph contains an "
        "effectful op, which is not supported yet: its with_effects HOP is "
        "non-cacheable, so the compiled artifact cannot be saved and lowered to "
        "standalone source."
    )


class _DynamoPythonBackend:
    """A capture-time callable backed by standalone Python graph source."""

    def __init__(
        self,
        python_code: str,
        cache: bytes | None,
        is_dynamic: bool,
        mutates_inputs: bool,
        call: Callable[[list[object]], object],
        compile_state: Any | None = None,
    ) -> None:
        self.python_code = python_code
        self.cache = cache
        self.is_dynamic = is_dynamic
        self.mutates_inputs = mutates_inputs
        self._call = call
        self._compile_state = compile_state
        if compile_state is not None:
            compile_state.install_capture(call.__globals__)

    def __call__(self, *args: object) -> object:
        return self._call(list(args))

    def observed_masks(self) -> set[int]:
        """Output-tangent patterns the emitted backward recorded while examples ran."""
        if self._compile_state is None:
            return set()
        return self._compile_state.observed_masks()

    def finalize_training(
        self, keep_capture: bool = False, extra_masks: Iterable[int] = ()
    ) -> None:
        """Compose the training module over the backward variants observed so far.

        ``extra_masks`` carries patterns observed on SIBLING backends of the same
        code (an automatic-dynamic recompile supersedes earlier backends in
        newest-first dispatch, so this backend must cover their masks too); they are
        re-canonicalized against this backend's metadata. Mask 0 (every tangent
        defined) is always covered. ``keep_capture`` leaves the live variant-compiler
        hook installed so a stateful capture keeps accumulating.
        """
        if self._compile_state is None:
            return
        observed = {
            self._compile_state.canonical_mask(mask)
            for mask in self.observed_masks() | set(extra_masks)
        }
        masks = (0, *sorted(observed - {0}))
        try:
            self.python_code, self.cache = self._compile_state.finalize(masks)
        except PrecompileError:
            raise
        except Exception as e:
            failed = sorted(set(masks) - self._compile_state.compiled_masks())
            raise PrecompileError(
                "precompile tracer='dynamo' could not compile the backward for the "
                f"observed output-tangent pattern(s) {[bin(mask) for mask in failed]} "
                f"({type(e).__name__}: {e}). Capture again without the example whose "
                "backward produced them."
            ) from e
        finally:
            if not keep_capture:
                self._compile_state.uninstall_capture()
                self._compile_state = None


def _build_dynamo_eager_graph_source(gm: torch.fx.GraphModule) -> str:
    """Render one Dynamo FX graph as standalone eager Python source."""
    get_attrs = list(gm.graph.find_nodes(op="get_attr"))
    if get_attrs:
        raise PrecompileError(
            "precompile tracer='dynamo' with backend='eager' does not yet support "
            "FX get_attr nodes; use backend='inductor'."
        )

    from torch.fx.graph import _custom_builtins

    parts = [
        "# Dynamo captured graph (eager backend).",
        *(_cb.import_str for _cb in _custom_builtins.values()),
        gm.code.replace("def forward(", "def _graph_forward(", 1),
        "",
        "class _GraphSelf:",
        "    pass",
        "",
        "",
        "def call(args):",
        "    return _graph_forward(_GraphSelf(), *args)",
        "",
    ]
    return "\n".join(parts)


def _graph_mutates_inputs(gm: torch.fx.GraphModule) -> bool:
    """Conservatively detect in-place ops in a Dynamo FX graph.

    In-place tensor methods (``add_``), in-place operators (``setitem``,
    ``iadd``), ``out=`` / ``inplace=True`` calls and mutable aten schemas all count,
    whether or not they hit a graph input: a false positive only keeps the runtime
    storage-overlap check on, a miss would skip it for a mutating graph.
    """
    import operator

    inplace_targets = {
        operator.setitem,
        operator.delitem,
        operator.iadd,
        operator.iand,
        operator.iconcat,
        operator.ifloordiv,
        operator.ilshift,
        operator.imatmul,
        operator.imod,
        operator.imul,
        operator.ior,
        operator.ipow,
        operator.irshift,
        operator.isub,
        operator.itruediv,
        operator.ixor,
    }
    for node in gm.graph.nodes:
        if node.op not in ("call_function", "call_method"):
            continue
        if node.kwargs.get("inplace") is True or "out" in node.kwargs:
            return True
        target = node.target
        if node.op == "call_method":
            name = target
        elif target in inplace_targets:
            return True
        elif isinstance(target, torch._ops.OpOverload):
            if target._schema.is_mutable:
                return True
            continue
        else:
            name = getattr(target, "__name__", "")
        if name.endswith("_") and not name.endswith("__"):
            return True
    return False


def _reject_environment_placeholders(gm: torch.fx.GraphModule, fn_name: str) -> None:
    # Dynamo tags each placeholder with the Source it was read from; anything not
    # rooted at the frame's arguments is a tensor the artifact's namespace cannot
    # reproduce (it would resolve to a NameError at serve time).
    from torch._guards import GuardProvenance

    for node in gm.graph.find_nodes(op="placeholder"):
        source = getattr(node, "_dynamo_source", None)
        if source is not None and source.provenance is not GuardProvenance.INPUT:
            raise PrecompileError(
                f"precompile tracer='dynamo' cannot capture {fn_name!r}: it reads "
                f"the tensor {source.name} from the Python environment (a global, "
                "class, or module attribute) rather than from its arguments. Every "
                "tensor must be an explicit input; pass it as an argument."
            )


class _DynamoCaptureSession:
    """What one capture's Dynamo hooks (backend compiler, guard filter) record.

    Those hooks are held by Dynamo's process-global registries until teardown, so
    this record must not reference the PrecompileState that owns it: otherwise an
    unclosed state could never be garbage collected (and never warn).
    """

    def __init__(self, fn_name: str, backend: str, grad_enabled: bool) -> None:
        self.fn_name = fn_name
        self.backend = backend
        self.grad_enabled = grad_enabled
        # Compiled graphs keyed by Dynamo's backend id (the __compiled_fn_* global
        # the transformed bytecode calls).
        self.backends: dict[str, _DynamoPythonBackend] = {}
        # Named environment guards dropped from dispatch (see _filter_dynamo_guards).
        self.dropped_guards: set[tuple[str, str, str]] = set()
        # Sources of environment tensors Dynamo guarded on (rejected after compile).
        self.environment_tensors: set[str] = set()


def _make_dynamo_backend_compiler(
    session: _DynamoCaptureSession,
) -> Callable[..., _DynamoPythonBackend]:
    def compile_graph(
        gm: torch.fx.GraphModule, example_inputs: list[object]
    ) -> _DynamoPythonBackend:
        from torch._functorch import aot_autograd, config as functorch_config
        from torch._functorch._aot_autograd.to_standalone_python import (
            _compile_to_python_with_state,
            _graph_has_dynamic_shapes,
        )

        _reject_environment_placeholders(gm, session.fn_name)
        is_dynamic = _graph_has_dynamic_shapes(gm)
        mutates_inputs = _graph_mutates_inputs(gm)
        compile_state = None
        if session.backend == "eager":
            python_code = _build_dynamo_eager_graph_source(gm)
            cache = None
            namespace: dict[str, object] = {"__name__": "_dynamo_eager_graph"}
            exec(compile(python_code, "<dynamo-eager-graph>", "exec"), namespace)
            call = cast("Callable[[list[object]], object]", namespace["call"])
        else:
            # Dynamo's runtime examples may have concrete tensor sizes while the graph
            # metadata carries the symbolic sizes and sources selected for this variant.
            graph_inputs = [
                node.meta["example_value"]
                for node in gm.graph.nodes
                if node.op == "placeholder"
            ]
            # Compile-time functorch knobs, patched only around this lowering (which
            # runs under Dynamo's compile lock, so no other thread's compile observes
            # them): eager backward lowering so the joint graph's backward source is
            # in hand at capture, and the undefined-tangent specialization machinery
            # that precompile's training artifacts are built on. The emitted training
            # module bakes the latter at emission time, so nothing needs the patch at
            # run time.
            with functorch_config.patch(
                force_non_lazy_backward_lowering=session.grad_enabled,
                aot_autograd_prune_unused_outputs=session.grad_enabled,
            ):
                python_code, cache, compile_state = _compile_to_python_with_state(
                    gm,
                    graph_inputs,
                    options={"size_asserts": True},
                    grad_enabled=session.grad_enabled,
                )
            call = aot_autograd.load_from_python(python_code, cache)
            if compile_state is not None:
                mutates_inputs = mutates_inputs or any(
                    info.mutates_data or info.mutates_metadata
                    for info in compile_state.spec.fw_metadata.input_info
                )
        compiled = _DynamoPythonBackend(
            python_code, cache, is_dynamic, mutates_inputs, call, compile_state
        )
        # Dynamo names the __compiled_fn_* global its transformed bytecode calls
        # before handing the graph over; the same name lands in code.backend_ids.
        session.backends[str(gm._backend_id)] = compiled  # type: ignore[attr-defined]
        return compiled

    return compile_graph


def _guard_value_repr(entry: GuardFilterEntry) -> str:
    if not entry.has_value:
        return ""
    try:
        return repr(entry.value)[:200]
    except Exception:
        return f"<{type(entry.value).__name__}>"


def _filter_dynamo_guards(
    session: _DynamoCaptureSession, entries: Sequence[GuardFilterEntry]
) -> list[bool]:
    """Dynamo guard filter: one pass at variant creation (Note [Guard provenance]).

    Keeps input-rooted guards (they dispatch), nameless guards (GLOBAL_STATE,
    SHAPE_ENV, TORCH_FUNCTION_STATE, DEFAULT_DEVICE, BACKWARD_STATE: checked per
    call) and tracing-internal SYNTHETIC guards. Named GLOBAL/AMBIENT guards cover
    the Python environment, a caller-provided invariant, and are dropped and
    recorded for ``_DROPPED_GUARDS``; this also removes the identity guards on
    globals that Dynamo cannot serialize. A dropped TENSOR_MATCH means fn touched a
    tensor that lives in the environment, which the artifact cannot reproduce; it is
    recorded and rejected right after the compile (_run_dynamo_examples).
    """
    from torch._guards import GuardProvenance

    keep = []
    for entry in entries:
        if (
            entry.provenance in (GuardProvenance.INPUT, GuardProvenance.SYNTHETIC)
            or entry.orig_guard.name == ""
        ):
            keep.append(True)
            continue
        if entry.guard_type == "TENSOR_MATCH":
            session.environment_tensors.add(entry.name)
        session.dropped_guards.add(
            (entry.guard_type, entry.name, _guard_value_repr(entry))
        )
        keep.append(False)
    return keep


def _reject_environment_globals(code: Any, fn_name: str) -> None:
    """Reject transformed bytecode that touches globals the artifact does not carry.

    The driver runs Dynamo's bytecode in a namespace holding only the import
    sources and backend ids Dynamo installed (plus builtins); a user global it
    loads (side-effect replay, output reconstruction) or stores would be a NameError
    or a silently dropped mutation at serve time.
    """
    import builtins
    import dis
    import types

    from torch._dynamo.package import SerializedCode

    allowed = set(code.import_sources) | set(code.backend_ids) | set(dir(builtins))

    def global_accesses(co: types.CodeType) -> Iterator[tuple[str, str]]:
        for instruction in dis.get_instructions(co):
            if instruction.opname in (
                "LOAD_GLOBAL",
                "STORE_GLOBAL",
                "DELETE_GLOBAL",
            ) and isinstance(instruction.argval, str):
                yield instruction.opname, instruction.argval
        for constant in co.co_consts:
            if isinstance(constant, types.CodeType):
                yield from global_accesses(constant)

    for guarded in code.guarded_codes:
        accesses = set(
            global_accesses(SerializedCode.to_code_object(guarded.dynamo_code))
        )
        stored = sorted({name for opname, name in accesses if opname != "LOAD_GLOBAL"})
        if stored:
            raise PrecompileError(
                f"precompile tracer='dynamo' cannot capture {fn_name!r}: it mutates "
                f"the global(s) {stored}, which the artifact cannot replay (it runs "
                "in its own namespace). Return the value instead."
            )
        loaded = sorted({name for opname, name in accesses if name not in allowed})
        if loaded:
            raise PrecompileError(
                f"precompile tracer='dynamo' cannot capture {fn_name!r}: its "
                f"transformed bytecode reads the global(s) {loaded} (to reconstruct "
                "or mutate them), which the artifact's namespace does not carry. "
                "Pass such values as arguments and return computed values only."
            )


def _dynamo_backend_source_literal(source: str) -> str:
    escaped = source.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    return f'    """\n{escaped}\n"""[1:-1],'


def _build_dynamo_python_source(
    *,
    backend: str,
    grad_enabled: bool,
    mutates_inputs: bool,
    state: dict[str, Any],
    backend_ids: list[str],
    compiled_backends: list[_DynamoPythonBackend],
    dropped_guards: tuple[tuple[str, str, str], ...],
) -> str:
    import base64
    import inspect
    import pickle
    import sys

    from torch import _precompile_driver as driver

    try:
        encoded_state = base64.b64encode(pickle.dumps(state)).decode("ascii")
    except Exception as e:
        raise PrecompileError(
            "precompile tracer='dynamo' could not serialize its guards and transformed "
            f"bytecode ({type(e).__name__}: {e})."
        ) from e

    dynamic_count = sum(compiled.is_dynamic for compiled in compiled_backends)
    parts = [
        '# Generated by torch.compiler.precompile (tracer="dynamo") -- do not edit.',
        "#",
        "# The compiled graphs and kernels below remain Python source. Dynamo's guard",
        "# trees and transformed code objects have no source form, so section 2 stores",
        "# the retained dispatch guard state plus bytecode as base64-encoded pickle data.",
        "",
        "import enum",
        "import types",
        "",
        "import torch as _torch",
        "import torch.utils._pytree as _pytree",
        "from torch._precompile import PrecompileError",
        "",
        "# " + "=" * 70,
        "# 1. Capture metadata and compiled Python graph sources",
        "# " + "=" * 70,
        f"BACKEND = {backend!r}",
        'TRACER = "dynamo"',
        "# The ambient grad mode at capture; dispatch is pinned to it (see forward).",
        f"_DYNAMO_GRAD_ENABLED = {grad_enabled!r}",
        "# Whether any captured graph mutates an input; only then must the driver check",
        "# runtime inputs for storage overlap (invariant 2).",
        f"_DYNAMO_MUTATES_INPUTS = {mutates_inputs!r}",
        f"VARIANT_COUNT = {len(state['variants'])}",
        f"GRAPH_COUNT = {len(compiled_backends)}",
        f"DYNAMIC_GRAPH_COUNT = {dynamic_count}",
        f"_DYNAMO_PYTHON_VERSION = {tuple(sys.version_info[:2])!r}",
        f"_DYNAMO_TORCH_VERSION = {torch.__version__!r}",
        f"_DYNAMO_BACKEND_IDS = {tuple(backend_ids)!r}",
        "# (guard type, source, capture-time value) triples dropped from at least one",
        "# variant's dispatch: they only cover the Python environment, which is a",
        "# caller-provided invariant. Every retained input-derived guard still gates",
        "# dispatch.",
        "_DROPPED_GUARDS = (",
        *(f"    {entry!r}," for entry in dropped_guards),
        ")",
        "# Each block is a standalone backend module. Keep them in separate strings so",
        "# load can execute each in an isolated namespace without graph-global collisions.",
        "_DYNAMO_BACKEND_SOURCES = (",
    ]
    for index, compiled in enumerate(compiled_backends):
        parts.append(f"    # Backend graph {index}")
        parts.append(_dynamo_backend_source_literal(compiled.python_code))
    parts.extend(
        [
            ")",
            "",
            "# " + "=" * 70,
            "# 2. Guard trees and transformed Dynamo bytecode (opaque)",
            "# " + "=" * 70,
            f"_DYNAMO_STATE = {encoded_state!r}",
            "",
            "# " + "=" * 70,
            "# 3. Python runtime glue: rebuild guards and dispatch variants",
            "# " + "=" * 70,
            inspect.getsource(driver._instance_values),
            "",
            inspect.getsource(driver._has_storage_overlap),
            "",
            inspect.getsource(driver._detach_fresh_outputs),
            "",
            inspect.getsource(driver._build_dynamo_forward),
            "",
            "forward = _build_dynamo_forward()",
            "",
            _DRIVER_MAIN,
        ]
    )
    return "\n".join(parts)


def _dynamo_cache_bytes(
    python_code: str, backend: str, artifacts: list[bytes | None]
) -> bytes:
    buf = io.BytesIO()
    torch.save(
        {
            "format": _CACHE_FORMAT,
            "version": _CACHE_VERSION,
            "backend": backend,
            "code_hash": hashlib.sha256(python_code.encode()).hexdigest(),
            "artifact": artifacts if backend == "inductor" else None,
        },
        buf,
    )
    return buf.getvalue()


def _validate_dynamo_capture(
    fn: Callable[..., object],
    example_inputs: Sequence[tuple[object, ...]],
    decompositions: dict | None,
) -> Callable[..., object]:
    """Reject unsupported dynamo-capture inputs up front; return the target function."""
    import inspect
    import pickle

    from torch._dynamo.eval_frame import innermost_fn
    from torch._precompile_driver import _has_storage_overlap, _instance_values

    if decompositions is not None:
        raise PrecompileError(
            "precompile decompositions are not yet supported with tracer='dynamo'."
        )

    try:
        import numpy
    except ImportError:
        numpy = None  # type: ignore[assignment]

    for example in example_inputs:
        values = list(_instance_values(example))
        if any(isinstance(value, torch.nn.Module) for value in values):
            raise PrecompileError(
                "precompile tracer='dynamo' does not yet support nn.Module arguments "
                "(including inside containers) because Dynamo's module identity "
                "guards are not serializable."
            )
        if numpy is not None and any(
            isinstance(value, (numpy.ndarray, numpy.generic)) for value in values
        ):
            # Dynamo traces ndarrays AND numpy scalars (numpy.generic) via
            # ___from_numpy sources whose TENSOR_MATCH guard construction fails
            # under the package/save-guards path, so capture would die with an
            # internal error; reject up front.
            raise PrecompileError(
                "precompile tracer='dynamo' does not yet support numpy array or "
                "numpy scalar arguments (including inside containers); convert "
                "them with torch.from_numpy / float(...) and pass those instead."
            )
        if _has_storage_overlap(example):
            raise PrecompileError(
                "precompile tracer='dynamo' does not support distinct tensor "
                "inputs that share or overlap storage; pass the same tensor "
                "object, or clone the views into separate tensors."
            )

    target = innermost_fn(fn)
    if not inspect.isfunction(target):
        raise PrecompileError(
            "precompile tracer='dynamo' currently requires a Python function and does "
            "not accept an nn.Module or bound method directly as fn."
        )
    if target.__closure__ is not None:
        raise PrecompileError(
            "precompile tracer='dynamo' does not yet support functions with closure "
            "cells; pass captured values as explicit arguments."
        )

    defaults = (target.__defaults__, target.__kwdefaults__)
    if any(isinstance(value, torch.Tensor) for value in _instance_values(defaults)):
        raise PrecompileError(
            "precompile tracer='dynamo' cannot serialize tensor-valued function "
            "defaults; pass every tensor as an explicit example input."
        )

    # Defaults travel inside the pickled artifact state; probe them up front so an
    # unpicklable default is reported as such rather than as a guard/bytecode failure.
    try:
        pickle.dumps(defaults)
    except Exception as e:
        raise PrecompileError(
            "precompile tracer='dynamo' cannot serialize the function's default "
            f"values ({type(e).__name__}: {e}); pass them as explicit example inputs."
        ) from e
    return target


def _make_dynamo_capture_target(
    target: Callable[..., object],
) -> Callable[..., object]:
    import types

    capture_globals = dict(target.__globals__)
    capture_target = types.FunctionType(
        target.__code__.replace(),
        capture_globals,
        target.__name__,
        target.__defaults__,
        target.__closure__,
    )
    capture_target.__kwdefaults__ = target.__kwdefaults__
    capture_target.__module__ = target.__module__
    capture_target.__qualname__ = target.__qualname__
    return capture_target


@contextlib.contextmanager
def _dynamo_capture_context(state: PrecompileState) -> Iterator[None]:
    from torch._dynamo.pgo import _use_code_state

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            torch._dynamo.config.patch(
                accumulated_recompile_limit=max(
                    torch._dynamo.config.accumulated_recompile_limit,
                    state.capture_limit,
                ),
                fail_on_recompile_limit_hit=True,
                # suppress_errors would silently fall back to eager (an
                # artifact must never be built from an uncaptured run), and
                # eval_frame asserts it is off whenever
                # fail_on_recompile_limit_hit is on -- a process-level
                # TORCHDYNAMO_SUPPRESS_ERRORS=1 would otherwise fail every
                # capture with that raw assertion.
                suppress_errors=False,
                # Otherwise every capture compile records the private package
                # into the process-global PrecompileContext (DynamoCache).
                caching_precompile=False,
                # A guard Dynamo cannot serialize must abort the capture with its
                # typed GuardSerializationError; the non-strict path would bypass
                # the package and silently leave the variant out of the artifact.
                strict_precompile=True,
            )
        )
        stack.enter_context(_use_code_state(state.pgo_state))
        stack.enter_context(torch.inference_mode(False))
        stack.enter_context(torch.set_grad_enabled(state.grad_enabled))
        yield


@contextlib.contextmanager
def _translate_dynamo_capture_errors(
    capture_limit: int, *, stateful: bool = False
) -> Iterator[None]:
    from torch._dynamo.exc import (
        BackendCompilerFailed,
        FailOnRecompileLimitHit,
        GuardSerializationError,
        PackageError,
        RecompileError,
        Unsupported,
    )

    try:
        yield
    except Unsupported as e:
        if e.gb_type == "Call to `torch._dynamo.graph_break()`":
            raise PrecompileError(
                "precompile tracer='dynamo' does not support graph breaks yet; "
                f"capture must produce one full graph. Dynamo reported: {e}"
            ) from e
        raise PrecompileError(
            "precompile tracer='dynamo' could not capture fn as one full graph. "
            f"Dynamo reported: {e}"
        ) from e
    except BackendCompilerFailed as e:
        if isinstance(e.inner_exception, PrecompileError):
            raise e.inner_exception from e
        raise
    except GuardSerializationError as e:
        # Input-derived identity guards (ID_MATCH and friends) survive the
        # guard filter, so they are the common cause.
        detail = f" (on {e.guard_name})" if e.guard_name else ""
        raise PrecompileError(
            f"precompile tracer='dynamo' encountered a {e.guard_type} guard{detail} "
            "that Dynamo cannot serialize yet (for example a module or callable "
            "identity guard)."
        ) from e
    except PackageError as e:
        raise PrecompileError(
            f"precompile tracer='dynamo' could not serialize the capture: {e}"
        ) from e
    except (FailOnRecompileLimitHit, RecompileError) as e:
        # A state's limit is fixed at creation (the optimize wrapper bakes it
        # in), so "pass a larger recompile_limit" alone would send a stateful
        # caller into the resume-mismatch error.
        advice = (
            "a state's recompile_limit is fixed when it is created, so close() "
            "this state and precompile again from scratch with a larger "
            "recompile_limit"
            if stateful
            else "pass a larger recompile_limit"
        )
        raise PrecompileError(
            "precompile tracer='dynamo' could not capture every example before "
            f"recompile_limit={capture_limit}; {advice}. Dynamo reported: {e}"
        ) from e


def _make_dynamo_capture_optimizer(
    state: PrecompileState,
) -> tuple[Callable[..., object], Callable[..., object]]:
    import functools

    compile_graph = _make_dynamo_backend_compiler(state.session)
    compiled = torch._dynamo.optimize(
        backend=compile_graph,
        nopython=True,
        guard_filter_fn=functools.partial(_filter_dynamo_guards, state.session),
        package=state.package,
        dynamic=state.dynamic,
        recompile_limit=state.capture_limit,
        isolate_recompiles=True,
    )(state.capture_target)
    if compiled is state.capture_target:
        # torch._dynamo.optimize returned its null decorator: capture would
        # silently run eager and fail later with a misleading error.
        raise PrecompileError(
            "precompile tracer='dynamo' cannot capture because Dynamo is "
            "disabled in this process (TORCHDYNAMO_DISABLE or a compiler "
            "kill switch)."
        )
    return compiled, compile_graph


class PrecompileStateSummary(NamedTuple):
    """What the most recently rendered artifact carries.

    ``dropped_guards`` lists the (guard type, source, capture-time value) triples
    the guard filter removed from at least one variant's dispatch (deduplicated
    across variants). They only covered the Python environment, which the
    programming model declares invariant between capture and serving; the recorded
    value is the environment state the artifact is specialized to.
    """

    calls: int
    examples: int
    variants: int
    graphs: int
    dynamic_graphs: int
    dropped_guards: tuple[tuple[str, str, str], ...]


def _warn_unclosed_dynamo_state(fn_name: str) -> None:
    log.warning(
        "A torch.compiler.precompile.PrecompileState for %r was garbage collected "
        "without close(); its capture session stays pinned by Dynamo's "
        "process-global registries until torch._dynamo.reset().",
        fn_name,
    )


class PrecompileState:
    """Accumulated capture state of ``torch.compiler.precompile.stateful``.

    Returned by (and passed back to) ``stateful`` calls. It owns the live Dynamo
    capture session: the cloned capture function with its installed code caches,
    the CompilePackage new variants accumulate into, the optimize wrapper (whose
    isolate-recompiles bucket keeps earlier variants visible to later calls), the
    isolated PGO state that drives automatic dynamic shapes across calls, the
    compiled backends, and the environment guards dropped so far. The ambient grad
    mode at creation is fixed for the state's lifetime. It is process-local and not
    serializable.

    Release the session with :meth:`close`, or use the state as a context manager
    (``with state:``): Dynamo's recompile-logging registry otherwise pins it until
    ``torch._dynamo.reset()``. A closed state cannot be resumed; artifact files
    written by earlier calls remain valid. Do not call ``torch._dynamo.reset()``
    between calls that share a state: later calls may raise, or silently duplicate
    variants in the rewritten artifact.
    """

    def __init__(
        self,
        *,
        target: Callable[..., object],
        capture_target: Callable[..., object],
        package: Any,
        pgo_state: Any,
        backend: str,
        capture_limit: int,
        dynamic: bool | None,
        grad_enabled: bool,
    ) -> None:
        self.target = target
        self.capture_target = capture_target
        self.package = package
        self.pgo_state = pgo_state
        self.backend = backend
        self.capture_limit = capture_limit
        self.dynamic = dynamic
        self.grad_enabled = grad_enabled
        self.session = _DynamoCaptureSession(target.__name__, backend, grad_enabled)
        self.compiled: Callable[..., object] | None = None
        self.backend_fn: Callable[..., object] | None = None
        self.examples = 0
        self.calls = 0
        self.last_summary: PrecompileStateSummary | None = None
        self.closed = False
        # Warn-only: the session outlives the state object (Dynamo's registries
        # pin it by code object, not through this instance), and tearing down
        # global compiler state from a GC callback is not safe. atexit=False
        # keeps a state that is simply alive at interpreter exit quiet.
        self._finalizer = weakref.finalize(
            self, _warn_unclosed_dynamo_state, target.__name__
        )
        self._finalizer.atexit = False

    def summary(self) -> PrecompileStateSummary | None:
        """Coverage of the most recently written artifact; None before one exists."""
        return self.last_summary

    def close(self) -> None:
        """Release the capture session (installed code caches and registries).

        Idempotent. A closed state cannot be resumed; artifact files written by
        earlier calls remain valid.
        """
        with _DYNAMO_COMPILE_LOCK:
            if self.closed:
                return
            self.closed = True
            self._finalizer.detach()
            self.compiled = None
            _teardown_dynamo_capture(
                self.package, self.capture_target, self.pgo_state, self.backend_fn
            )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        status = ", closed" if self.closed else ""
        return (
            f"<torch.compiler.precompile.PrecompileState: {self.examples} "
            f"example(s), backend={self.backend!r}{status}>"
        )


def _new_dynamo_state(
    target: Callable[..., object],
    *,
    backend: str,
    capture_limit: int,
    dynamic: bool | None,
) -> PrecompileState:
    from torch._dynamo.package import CompilePackage
    from torch._dynamo.pgo import _new_code_state

    capture_target = _make_dynamo_capture_target(target)
    return PrecompileState(
        target=target,
        capture_target=capture_target,
        package=CompilePackage(capture_target),
        pgo_state=_new_code_state(),
        backend=backend,
        capture_limit=capture_limit,
        dynamic=dynamic,
        # Differentiability mirrors torch.compile: capture runs under the caller's
        # ambient grad mode and each graph infers it from requires_grad inputs.
        grad_enabled=torch.is_grad_enabled(),
    )


def _run_dynamo_examples(
    state: PrecompileState, example_inputs: Sequence[tuple[object, ...]]
) -> list[object]:
    import inspect

    if state.compiled is None:
        state.compiled, state.backend_fn = _make_dynamo_capture_optimizer(state)
    # Probe every example's arity up front, so a caller mistake raises before
    # any example of the batch is run or counted.
    signature = inspect.signature(state.capture_target)
    for example in example_inputs:
        try:
            signature.bind(*example)
        except TypeError as e:
            raise TypeError(
                f"precompile example does not match the positional signature of "
                f"{state.target.__name__!r}: {e}. No example from this call was "
                "recorded."
            ) from e
    results = []
    for example in example_inputs:
        state.examples += 1
        results.append(state.compiled(*example))
        if state.session.environment_tensors:
            raise PrecompileError(
                f"precompile tracer='dynamo' cannot capture {state.target.__name__!r}: "
                f"it uses the tensor(s) {sorted(state.session.environment_tensors)} from the "
                "Python environment (a global, class, or module attribute) rather "
                "than from its arguments. Every tensor must be an explicit input; "
                "pass them as arguments."
            )
    return results


def _build_dynamo_artifact(
    state: PrecompileState, *, keep_capture: bool = False
) -> tuple[str, bytes, PrecompileStateSummary]:
    """Render the state's accumulated capture as (python_code, cache) bytes.

    A pure read of the live package and compiled backends (plus, for training, a
    snapshot compose of the backward variants recorded so far when keep_capture
    is set), so a stateful capture can rebuild after every example call.
    """
    # An automatic-dynamic recompile supersedes earlier backends of the same
    # code in the artifact's newest-first dispatch, so a tangent mask observed
    # only on a superseded backend (e.g. a partial backward run before the
    # recompile) would be served by a newer backend that never saw it. Union
    # the observed masks across backends and cover the union everywhere; each
    # backend's finalize re-canonicalizes them against its own metadata.
    backends = state.session.backends
    observed_masks: set[int] = set()
    for compiled_backend in backends.values():
        observed_masks |= compiled_backend.observed_masks()
    for compiled_backend in backends.values():
        compiled_backend.finalize_training(
            keep_capture=keep_capture, extra_masks=observed_masks
        )

    cache_entry = state.package.cache_entry()
    active_codes = [code for code in cache_entry.codes if not code.bypassed]
    if len(active_codes) != 1:
        raise PrecompileError(
            "precompile tracer='dynamo' does not support graph breaks or separately "
            "compiled nested frames yet."
        )
    code = active_codes[0]
    if code.install_to_global or not code.guarded_codes:
        raise PrecompileError(
            "precompile tracer='dynamo' did not capture a runnable entry frame."
        )
    _reject_environment_globals(code, state.target.__name__)

    compiled_backends = []
    for backend_id in code.backend_ids:
        compiled_backend = backends.get(backend_id)
        if compiled_backend is None:
            raise PrecompileError(
                "precompile tracer='dynamo' encountered a graph that could not be "
                "represented as standalone Python source."
            )
        compiled_backends.append(compiled_backend)

    dynamo_state: dict[str, Any] = {
        "code": code.python_code,
        "import_sources": dict(code.import_sources),
        "defaults": state.capture_target.__defaults__,
        "kwdefaults": state.capture_target.__kwdefaults__,
        # Newest-first: the driver serves the first variant whose guards pass,
        # and live Dynamo checks recompilations LRU-front-first -- an input
        # matching both an early static variant and a later dynamic one (the
        # automatic-dynamic revisit pattern) must serve the later one, whose
        # backend is the one that observed any training tangent masks.
        # guarded_codes is chronological (CompilePackage appends); note that
        # CompilePackage.install serves the same list oldest-first.
        "variants": [
            {
                "guards_state": guarded.guards_state,
                "dynamo_code": guarded.dynamo_code,
            }
            for guarded in code.guarded_codes
        ][::-1],
    }
    dropped_guards = tuple(sorted(state.session.dropped_guards))
    python_code = _build_dynamo_python_source(
        backend=state.backend,
        grad_enabled=state.grad_enabled,
        mutates_inputs=any(compiled.mutates_inputs for compiled in compiled_backends),
        state=dynamo_state,
        backend_ids=[str(backend_id) for backend_id in code.backend_ids],
        compiled_backends=compiled_backends,
        dropped_guards=dropped_guards,
    )
    cache = _dynamo_cache_bytes(
        python_code,
        state.backend,
        [compiled.cache for compiled in compiled_backends],
    )
    summary = PrecompileStateSummary(
        calls=state.calls,
        examples=state.examples,
        variants=len(code.guarded_codes),
        graphs=len(compiled_backends),
        dynamic_graphs=sum(c.is_dynamic for c in compiled_backends),
        dropped_guards=dropped_guards,
    )
    return python_code, cache, summary


def _teardown_dynamo_capture(
    package: Any,
    capture_target: Callable[..., object],
    pgo_state: Any,
    backend_fn: Callable[..., object] | None = None,
) -> None:
    from torch._dynamo.eval_frame import remove_cached_backend
    from torch._dynamo.utils import clear_guard_failures_for_code

    try:
        if package is not None:
            package.uninstall()
    finally:
        try:
            torch._dynamo.reset_code(capture_target.__code__)
        finally:
            pgo_state.clear()
            # Recompile logging strong-keys the capture code object, transitively
            # pinning the whole session (including the copied fn globals).
            clear_guard_failures_for_code(capture_target.__code__)
            if backend_fn is not None:
                remove_cached_backend(backend_fn)


def _precompile_dynamo(
    fn: Callable[..., object],
    example_inputs: Sequence[tuple[object, ...]],
    *,
    backend: str,
    decompositions: dict | None,
    recompile_limit: int | None = None,
    dynamic: bool | None = None,
) -> tuple[str, bytes]:
    target = _validate_dynamo_capture(fn, example_inputs, decompositions)
    capture_limit = (
        recompile_limit
        if recompile_limit is not None
        else max(torch._dynamo.config.recompile_limit, len(example_inputs) + 1)
    )
    with _DYNAMO_COMPILE_LOCK:
        state = None
        try:
            with _translate_dynamo_capture_errors(capture_limit):
                state = _new_dynamo_state(
                    target,
                    backend=backend,
                    capture_limit=capture_limit,
                    dynamic=dynamic,
                )
                with _dynamo_capture_context(state):
                    _run_dynamo_examples(state, example_inputs)
                    python_code, cache, _summary = _build_dynamo_artifact(state)
                    return python_code, cache
        finally:
            if state is not None:
                state.close()


def _write_dynamo_artifact_files(
    python_code: str, cache: bytes, artifact_path: str, cache_path: str
) -> None:
    from torch._inductor.codecache import write_atomic

    # Artifact first, then cache, each an fsync'd write plus rename. A crash between
    # the two leaves the NEW artifact with the OLD cache, which load_files degrades
    # on (code_hash mismatch -> cold cache with a warning) and the next successful
    # rewrite repairs; a crash before the first rename leaves the previous pair.
    write_atomic(
        artifact_path, python_code, make_dirs=True, encode_utf_8=True, fsync=True
    )
    write_atomic(cache_path, cache, make_dirs=True, fsync=True)


def _precompile_dynamo_stateful(
    fn: Callable[..., object],
    example_inputs: Sequence[tuple[object, ...]],
    *,
    backend: str,
    recompile_limit: int | None,
    dynamic: bool | None,
    state: PrecompileState | None,
    artifact_path: str,
    cache_path: str,
) -> tuple[list[object], PrecompileState]:
    if state is not None and not isinstance(state, PrecompileState):
        raise TypeError(
            "precompile state must be the state returned by a previous "
            f"stateful precompile call, got {type(state).__name__}."
        )
    target = _validate_dynamo_capture(fn, example_inputs, None)
    with _DYNAMO_COMPILE_LOCK:
        fresh = state is None
        if state is None:
            # Accumulating captures outgrow the config default quickly, so the
            # stateful default is a real budget rather than the example count.
            capture_limit = (
                recompile_limit
                if recompile_limit is not None
                else max(torch._dynamo.config.recompile_limit, 256)
            )
        else:
            if state.closed:
                raise ValueError(
                    "precompile cannot resume a closed state; start fresh with "
                    "state=None."
                )
            capture_limit = state.capture_limit
            mismatches = []
            if backend != state.backend:
                mismatches.append(
                    f"backend={backend!r} (the state was created with "
                    f"{state.backend!r})"
                )
            if recompile_limit is not None and recompile_limit != state.capture_limit:
                mismatches.append(
                    f"recompile_limit={recompile_limit!r} (the state was created "
                    f"with {state.capture_limit!r})"
                )
            if dynamic is not None and dynamic != state.dynamic:
                mismatches.append(
                    f"dynamic={dynamic!r} (the state was created with "
                    f"{state.dynamic!r})"
                )
            if torch.is_grad_enabled() != state.grad_enabled:
                mismatches.append(
                    f"grad mode enabled={torch.is_grad_enabled()!r} (the state was "
                    f"created with {state.grad_enabled!r})"
                )
            if target is not state.target:
                mismatches.append(
                    "fn (a state resumes only the function that created it)"
                )
            if mismatches:
                raise ValueError(
                    "precompile cannot resume a state under different settings; "
                    "that would produce a mixed artifact: "
                    + "; ".join(mismatches)
                    + "."
                )
        try:
            # A FRESH call that hits the limit self-closes and never returns its
            # state, so the close()-and-recapture advice only fits resumed calls.
            with _translate_dynamo_capture_errors(capture_limit, stateful=not fresh):
                if state is None:
                    state = _new_dynamo_state(
                        target,
                        backend=backend,
                        capture_limit=capture_limit,
                        dynamic=dynamic,
                    )
                with _dynamo_capture_context(state):
                    results = _run_dynamo_examples(state, example_inputs)
                    try:
                        python_code, cache, summary = _build_dynamo_artifact(
                            state, keep_capture=True
                        )
                    except PrecompileError as e:
                        if fresh:
                            raise
                        # The failing variant stays in the accumulated capture, so
                        # rebuilding will keep failing; say so instead of implying
                        # the state can make further progress.
                        raise PrecompileError(
                            f"{e} The accumulated capture can no longer be rendered, "
                            "so further calls on this state will fail the same way; "
                            "the last successfully written artifact files remain "
                            "valid. close() the state and precompile again without "
                            "the offending example."
                        ) from e
            _write_dynamo_artifact_files(python_code, cache, artifact_path, cache_path)
            state.calls += 1
            state.last_summary = summary._replace(calls=state.calls)
        except BaseException:
            # A fresh call that failed returns no state, so tear its session
            # down; a resumed call leaves the state (and the last successfully
            # written artifact) intact.
            if fresh and state is not None:
                state.close()
            raise
        # Always a list, one entry per example of THIS call: with a single
        # conditional shape, a fn that itself returns a list would be
        # indistinguishable from a multi-example call.
        return results, state


class PrecompiledModule:
    """Internal holder for a precompiled computation / a loaded runnable."""

    def __init__(
        self,
        fn: Callable[..., object],
        *,
        backend: str = "inductor",
        decompositions: dict | None = None,
    ) -> None:
        # ``fn`` is the whole computation: an nn.Module, or a callable that closes
        # over the module(s) it uses (e.g. ``lambda x: model(x)``, or a training
        # step that computes a loss and torch.autograd.grad). This holder implements
        # the make_fx calling convention; load() reuses it for dynamo artifacts too,
        # recording which tracer produced the loaded forward.
        self._fn = fn
        self._backend = backend
        self._tracer = "make_fx"
        self._decompositions = decompositions
        self._artifact: object = None
        self._module_positions: list[int] = []
        self._num_positional_args: int = 0
        # Interned param / buffer names and their example shape, dtype, and device
        # (aligned lists); the driver checks each runtime param/buffer against these for
        # the structural contract (invariant 2). Populated by _compile().
        self._param_names: list[str] = []
        self._buffer_names: list[str] = []
        self._param_shapes: list[tuple[int, ...]] = []
        self._buffer_shapes: list[tuple[int, ...]] = []
        self._param_dtypes: list[str] = []
        self._buffer_dtypes: list[str] = []
        self._param_devices: list[str] = []
        self._buffer_devices: list[str] = []
        self._in_spec: pytree.TreeSpec | None = None
        self._out_spec: pytree.TreeSpec | None = None
        self._gm: torch.fx.GraphModule | None = None
        # Inductor backend: the composed self-contained graph module (from
        # aot_autograd.compile_to_python, exposing ``call(flat_inputs)``) and the
        # opaque artifact-cache bytes (None if uncacheable), populated by _compile().
        self._graph_python: str = ""
        self._artifact_bytes: bytes | None = None
        # Which unique-param index each emitted (trailing) grad output belongs to; its
        # length is the number of grad outputs. Lets the driver scatter grads onto
        # exactly the params that received one, leaving frozen / non-contributing
        # params' .grad as None.
        self._grad_param_indices: list[int] = []
        # Per user-input-leaf example shape, dtype, and device (None for a subclass /
        # non-tensor leaf; a marked-dynamic dim is None within the shape tuple); the drivers
        # reject a runtime mismatch (invariants 3 and 6). Stride / memory format is enforced
        # by the inductor artifact's own assert_size_stride, not recorded here. Populated by
        # _compile().
        self._user_input_shapes: list[tuple[int | None, ...] | None] = []
        self._user_input_dtypes: list[str | None] = []
        self._user_input_devices: list[str | None] = []
        # Per user-input-leaf mark_unbacked min/max bounds (None for a leaf with no
        # bounded marked dim, else {dim: (lo, hi)}). The drivers reject a runtime size
        # outside the declared range (invariant 3). Populated by _compile().
        self._user_input_bounds: list[Any] = []
        # Set only on the load() path, where we wrap a reconstructed callable.
        self._loaded_forward: Callable[..., object] | None = None

    @classmethod
    def _from_loaded(
        cls,
        forward: Callable[..., object],
        *,
        backend: str,
        tracer: str,
    ) -> PrecompiledModule:
        """Build a runnable from load()'s reconstructed forward.

        load() does not re-run capture/_compile, so reuse ``__init__`` for all the
        defaults (the single definition of this object's state) and override only the
        reconstructed forward. All the calling-convention metadata lives in the inlined
        driver (``forward``) itself, so the __init__ fields (``_fn``, ``_gm``,
        ``_module_positions``, ``_out_spec``, ...) stay at their defaults; inspect the
        artifact via python_code.
        """
        obj = cls(None, backend=backend)  # type: ignore[arg-type]
        obj._tracer = tracer
        obj._loaded_forward = forward
        return obj

    def _compile(self, example_inputs: Sequence[tuple[object, ...]]) -> None:
        if len(example_inputs) != 1:
            raise ValueError(
                "precompile with tracer='make_fx' requires exactly one example "
                "input tuple."
            )
        args = example_inputs[0]
        if self._backend == "eager" and _has_unbacked_marks(args):
            raise PrecompileError(
                "precompile: mark_unbacked (dynamic shapes) is only supported with "
                "backend='inductor'; eager + unbacked is not supported."
            )
        capture = _capture(self._fn, args, self._decompositions)
        self._module_positions = capture.module_positions
        self._num_positional_args = capture.num_positional_args
        self._param_names = capture.param_names
        self._buffer_names = capture.buffer_names
        self._param_shapes = capture.param_shapes
        self._buffer_shapes = capture.buffer_shapes
        self._param_dtypes = capture.param_dtypes
        self._buffer_dtypes = capture.buffer_dtypes
        self._param_devices = capture.param_devices
        self._buffer_devices = capture.buffer_devices
        self._user_input_shapes = capture.user_input_shapes
        self._user_input_dtypes = capture.user_input_dtypes
        self._user_input_devices = capture.user_input_devices
        self._user_input_bounds = capture.user_input_bounds
        self._in_spec = capture.in_spec
        self._out_spec = capture.out_spec
        self._grad_param_indices = capture.grad_param_indices
        self._gm = capture.gm

        if self._backend == "eager":
            # No Inductor lowering: the captured ATen graph IS the artifact. It is
            # run directly on the (subclass-level) inputs, so there is no inductor
            # ``call`` to inline and no dense flatten/unflatten -- the graph runs
            # exactly as captured (see Note [precompile programming model]).
            return

        # Lower through the AOT backend contract: it returns a self-contained module
        # exposing ``call(flat_inputs) -> outputs`` (with AOTAutograd's own codegen'd
        # prelude/epilogue -- subclass wrap/unwrap, input-mutation reflection, output
        # aliasing -- composed in, not reimplemented here) plus an opaque cache (the
        # save_cache_artifacts bundle that primes the inductor cache on load, or None
        # for uncacheable graphs).
        import torch._inductor.config as _ind_config
        from torch._functorch import aot_autograd
        from torch._inductor.exc import InductorError
        from torch._inductor.standalone_compile import NoRunnableInductorModuleError

        # Pin size_asserts ON so the artifact ALWAYS bakes assert_size_stride for the
        # inputs the graph reads -- this enforces the input memory-format contract
        # (invariant 6) at runtime regardless of the user's ambient size_asserts config
        # (off would otherwise elide the asserts and silently read wrong strides). The
        # guard is conservative (see the inlined driver checks): an input the graph never
        # reads gets no assert and stays layout-flexible, but a read input is asserted on
        # the example layout even for layout-agnostic ops (matmul/addmm), since precompile
        # cannot recompile to specialize a new layout the way torch.compile would. A
        # dynamic (unbacked) capture additionally pins scalar_asserts so the make_fx
        # ShapeEnv's runtime range asserts survive into the artifact.
        #
        # These are inductor config keys, so they ride in as ``options`` (aot_autograd.
        # compile_to_python merges them into the inductor config.patch it wraps the
        # compile in) rather than being patched around the call. The AOT layer detects
        # dynamic (symbolic) shapes off the captured graph and threads the make_fx
        # ShapeEnv through automatically, so there is no dynamic_shapes knob to pass and
        # no manual TracingContext to install: a static capture specializes to the
        # example shapes, an unbacked capture keeps the symbols.
        options: dict[str, Any] = {"size_asserts": True}
        if capture.fake_mode is not None and hasattr(_ind_config, "scalar_asserts"):
            options["scalar_asserts"] = True
        try:
            self._graph_python, self._artifact_bytes = aot_autograd.compile_to_python(
                capture.gm, capture.flat_args, options=options
            )
        except NoRunnableInductorModuleError as e:
            # Inductor emits no runnable module for a graph with no compute to lower --
            # one that returns inputs or Python constants unchanged (e.g. ``lambda x: x``,
            # ``x.detach()``, ``return 7``, or a bare ``return None``). The eager backend
            # (above) handles these; surface a clear PrecompileError instead of the raw
            # lowering error.
            raise PrecompileError(
                "the inductor backend cannot lower a graph with no compute -- the traced "
                "fn returns its inputs or Python constants unchanged, producing no "
                "Inductor kernel. Return a computed tensor, or use backend='eager'."
            ) from e
        except InductorError as e:
            # Inductor codegen asserts on certain non-tensor Python values in the output
            # structure ("Unexpected output types: [<class 'float'>]" -- also complex,
            # str, ...); int/bool/None outputs lower fine, and the eager backend handles
            # them too. Surface a clear PrecompileError instead of the raw assertion.
            if "Unexpected output types" in str(e):
                raise PrecompileError(
                    "the inductor backend cannot lower a graph whose output mixes a "
                    "non-tensor Python value (e.g. float / complex / str) with computed "
                    "tensors (int / bool / None outputs are fine). Return only tensors, "
                    "or use backend='eager'."
                ) from e
            raise

    def __call__(self, *args: object, **kwargs: object) -> object:
        # A PrecompiledModule is runnable only after load(); precompile() itself
        # returns (python_code, cache) rather than a runnable.
        if self._loaded_forward is None:
            raise PrecompileError(
                "this object is not runnable; build one with "
                "torch.compiler.precompile.load(python_code, cache)."
            )
        if kwargs and self._tracer != "dynamo":
            raise PrecompileError(
                "precompile: make_fx artifacts take positional arguments only (the "
                "model(s) and inputs in the traced fn's positions); keyword arguments "
                "are supported by tracer='dynamo' artifacts."
            )
        return self._loaded_forward(*args, **kwargs)

    def to_python_code(self) -> str:
        """Return the self-contained, executable Python artifact as a string.

        It runs on its own, needing no cache (Note [precompile programming model],
        "self-contained"). For the inductor backend it embeds the composed graph
        module from aot_autograd.compile_to_python (kernels JIT-compile on first
        call; AOTAutograd's prelude/epilogue inlined), the calling-convention
        metadata, and a ``forward()`` that takes the same args the traced fn took
        (the model(s) plus runtime inputs). For the eager backend it embeds the
        captured ATen graph (both readable and executable) plus a driver that runs it
        eagerly. No weights are embedded.
        """
        if self._loaded_forward is not None:
            raise PrecompileError(
                "this object was produced by torch.compiler.precompile.load(); the "
                "python_code you passed in is the source artifact (load() does not "
                "re-capture, so there is no python_code to re-emit from this object)."
            )
        if self._backend == "eager":
            if self._gm is None:
                raise PrecompileError("internal: not compiled; call _compile() first")
            return _build_eager_python_source(self)
        if not self._graph_python:
            raise PrecompileError("internal: not compiled; call _compile() first")
        return _build_python_source(self, self._graph_python)

    def to_cache_bytes(self, python_code: str | None = None) -> bytes:
        """Return the binary cache as bytes -- an ACCELERATION, not required to run.

        ``python_code`` is the single source of truth for the calling convention, so the
        cache holds only the compiled artifact plus the integrity tag and code_hash. For
        the inductor backend that artifact is the ``save_cache_artifacts`` bundle (load
        primes the kernel caches with it, so a warm reload skips JIT); for the eager
        backend it is None. See Note [precompile programming model], invariant 7.

        ``python_code`` defaults to what ``to_python_code()`` would emit; ``__call__``
        threads in the exact string it already built so code_hash matches the bytes
        returned to the user and the metadata is not rebuilt.
        """
        # _artifact_bytes is the inductor cache bundle (None if uncacheable, and always
        # None for eager); the envelope is a plain str/int/bytes dict (weights_only-safe)
        # carrying the tag + code_hash that binds it to python_code (invariant 7).
        if self._loaded_forward is not None:
            raise PrecompileError(
                "this object was produced by torch.compiler.precompile.load(); the cache "
                "you passed in is the source artifact (load() does not re-capture, so "
                "there is no cache to re-emit from this object)."
            )
        if python_code is None:
            python_code = self.to_python_code()
        code_hash = hashlib.sha256(python_code.encode()).hexdigest()
        buf = io.BytesIO()
        torch.save(
            {
                "format": _CACHE_FORMAT,
                "version": _CACHE_VERSION,
                "backend": self._backend,
                "code_hash": code_hash,
                "artifact": self._artifact_bytes,
            },
            buf,
        )
        return buf.getvalue()


def _make_inlined_forward(python_code: str) -> Callable[..., object]:
    """Fallback: execute the self-contained python string (JITs kernels).

    ``python_code`` needs no cache -- the kernels (inductor) or graph (eager) are
    inlined, so we just exec it and hand back its ``forward``. The returned
    ``forward`` takes the same args the traced fn took (model(s) plus runtime
    inputs).

    The untrusted-input warning is emitted by ``load`` BEFORE any cache
    processing, not here: the cache is unpickled (``load_cache_artifacts``)
    before this exec runs, and a warning after that unpickle would fire only
    once the risk had already been taken."""
    module_ns: dict[str, object] = {"__name__": "_precompiled_artifact"}
    exec(compile(python_code, "<precompile>", "exec"), module_ns)
    return cast("Callable[..., object]", module_ns["forward"])


def _normalize_example_inputs(
    example_args: tuple[object, ...],
    example_inputs: Sequence[tuple[object, ...]] | None,
) -> list[tuple[object, ...]]:
    from collections.abc import Sequence as SequenceABC

    if example_inputs is None:
        return [tuple(example_args)]
    if example_args:
        raise TypeError(
            "precompile cannot take both positional examples and example_inputs."
        )
    if isinstance(example_inputs, (str, bytes)) or not isinstance(
        example_inputs, SequenceABC
    ):
        raise TypeError(
            "precompile example_inputs must be a sequence (list or tuple) of "
            f"positional-argument tuples, got {type(example_inputs).__name__}."
        )
    if len(example_inputs) == 0:
        raise ValueError("precompile requires at least one example input tuple.")
    for example in example_inputs:
        if not isinstance(example, tuple):
            raise TypeError(
                "precompile example_inputs must be a sequence of positional-"
                f"argument tuples, got {type(example).__name__}."
            )
    return list(example_inputs)


def _load_artifact(
    python_code: str, cache: bytes | None, *, degrade_on_mismatch: bool
) -> Callable[..., object]:
    """Shared body of load / load_files; see their docstrings for the contract.

    ``degrade_on_mismatch`` selects the on-disk pairing rule: a cache whose
    backend/code_hash does not match python_code is a wrong pairing for an
    in-memory pair (hard failure), but for the file pair a stateful capture
    rewrites it is exactly what a rewrite interrupted between its two renames
    leaves behind, so it degrades to a cold cache with a warning.
    """
    # Unpickling the cache references classes in AOTAutograd's runtime; import
    # dynamo first so that import completes in a non-circular order (otherwise
    # a cold load can hit a runtime_wrappers <-> _dynamo circular import).
    import torch._dynamo

    # The whole calling convention (MODULE_POSITIONS, OUT_SPEC, USER_INPUT_*, PARAM_*,
    # BUFFER_*, IN_SPEC, ...) is consumed by the driver INLINED in python_code
    # (emitted from torch._precompile_driver), so the loaded object needs none of it.
    # _parse_artifact_metadata still runs to validate python_code is a precompile
    # artifact and to read BACKEND for the cache-pairing check below.
    meta = _parse_artifact_metadata(python_code)
    backend = cast(str, meta["BACKEND"])
    tracer = cast(str, meta.get("TRACER", "make_fx"))

    # Both halves are untrusted EXECUTABLE input: priming below unpickles the
    # cache's inductor bundle, and the exec of python_code runs whatever it
    # contains (JIT-compiling inlined kernels or running the inlined graph).
    # Warn per load (not warning_once), BEFORE either risk is taken.
    log.warning(
        "torch.compiler.precompile.load is about to unpickle the cache and EXEC "
        "python_code, which are untrusted executable inputs (they run inlined "
        "kernels / graph code). Only load a (python_code, cache) pair you "
        "produced or otherwise trust (Note [precompile programming model], "
        "invariant 7)."
    )

    # weights_only=True is safe (plain str/int/bytes dict). The inner artifact bytes
    # are the inductor save_cache_artifacts bundle, used below to prime the kernel
    # caches. The cache is acceleration only, so an unreadable envelope or a FORMAT /
    # VERSION mismatch degrades to JIT'ing from python_code rather than crashing. A
    # BACKEND or CODE_HASH mismatch is different -- it signals a wrong (python_code,
    # cache) pairing -- so it hard-fails rather than running under foreign metadata,
    # except on the on-disk path form (degrade_on_mismatch).
    artifact = None
    try:
        blob = torch.load(io.BytesIO(cache), weights_only=True) if cache else None
        if blob is not None and (
            blob.get("format") != _CACHE_FORMAT or blob.get("version") != _CACHE_VERSION
        ):
            log.warning(
                "torch.compiler.precompile.load got a cache with format=%r "
                "version=%r, expected %r / %r; it is likely from a different torch "
                "build. Falling back to JIT from python_code.",
                blob.get("format"),
                blob.get("version"),
                _CACHE_FORMAT,
                _CACHE_VERSION,
            )
            blob = None
        if blob is not None:
            expected_code_hash = hashlib.sha256(python_code.encode()).hexdigest()
            mismatched = (
                blob.get("backend") != backend
                or blob.get("code_hash") != expected_code_hash
            )
            if mismatched and degrade_on_mismatch:
                log.warning(
                    "torch.compiler.precompile.load_files got a cache whose "
                    "backend/code_hash does not match the artifact (likely a "
                    "rewrite interrupted between the artifact and cache renames). "
                    "Falling back to JIT from python_code."
                )
                blob = None
            elif blob.get("backend") != backend:
                raise PrecompileError(
                    f"cache backend {blob.get('backend')!r} does not match the "
                    f"python_code backend {backend!r}; the cache and python_code "
                    "came from different precompile() calls."
                )
            elif blob.get("code_hash") != expected_code_hash:
                raise PrecompileError(
                    "cache does not match python_code (its code_hash "
                    f"{blob.get('code_hash')!r} != sha256(python_code) "
                    f"{expected_code_hash!r}); the cache and python_code came "
                    "from different precompile() calls. Pair each cache with "
                    "the python_code from the same precompile() call."
                )
        if blob is not None:
            artifact = blob.get("artifact")
    except PrecompileError:
        raise
    except Exception as e:
        log.warning(
            "torch.compiler.precompile.load could not read the cache envelope (%s: %s); the "
            "cache is likely corrupt or from a different torch build. Falling back "
            "to JIT from python_code.",
            type(e).__name__,
            e,
        )
    if artifact is not None:
        # Prime the inductor kernel caches from the bundle so the exec of python_code
        # below loads the precompiled kernels (Triton binaries / autotune results)
        # instead of recompiling them. The composed python_code runs its inlined
        # kernels directly (no compile_fx re-entry, so no FxGraphCache lookup); the
        # acceleration is the warm kernel cache. This is a pure acceleration: a stale /
        # cross-torch-version / corrupt bundle that fails to load just leaves the caches
        # cold, and python_code JITs -- same result, no crash.
        artifacts = (
            artifact
            if tracer == "dynamo" and isinstance(artifact, (list, tuple))
            else [artifact]
        )
        for artifact_bundle in artifacts:
            if artifact_bundle is None:
                continue
            try:
                torch.compiler.load_cache_artifacts(artifact_bundle)
            except Exception as e:
                log.warning(
                    "torch.compiler.precompile.load could not prime the cache from "
                    "an artifact bundle (%s: %s); it is likely stale or from a "
                    "different torch build. Falling back to JIT from python_code.",
                    type(e).__name__,
                    e,
                )
    # Run the driver inlined in python_code. It carries the full calling convention and
    # runtime safety checks (subclass wrap/unwrap, param/buffer lifting, grad harvest,
    # input/model validation) and JITs the kernels -- which hit the primed cache when
    # the bundle above loaded, so the "cache" path is exec-with-warm-kernels rather than
    # a separate runtime.
    forward = _make_inlined_forward(python_code)

    return PrecompiledModule._from_loaded(forward, backend=backend, tracer=tracer)


class _PrecompileApi:
    """Callable namespace implementing ``torch.compiler.precompile`` and its members.

    A single instance is exposed as ``torch.compiler.precompile``; calling it precompiles a
    computation, ``torch.compiler.precompile.load`` / ``load_files`` reload the resulting
    artifacts, and ``stateful`` captures incrementally. It is a class (rather than a
    function with attached attributes) so the call, the loaders, the state types, and
    the error type are explicit members.

    The contract for every entry point is Note [precompile programming model] in this
    module.
    """

    # Reported so test_public_bindings / introspection see this as ``torch.compiler``.
    __module__ = "torch.compiler"

    # The error type raised by precompile, reachable as
    # ``torch.compiler.precompile.PrecompileError``.
    PrecompileError = PrecompileError
    # The stateful capture's state and summary types, reachable as
    # ``torch.compiler.precompile.PrecompileState`` / ``.PrecompileStateSummary``.
    PrecompileState = PrecompileState
    PrecompileStateSummary = PrecompileStateSummary

    def __reduce__(self) -> str:
        # torch.compiler.precompile is a process-wide singleton; pickle/deepcopy must
        # round-trip to the SAME object (the instance carries no per-call state) rather
        # than fail to pickle a bound-method-bearing instance. Returning the qualified
        # name resolves back to this singleton on unpickle.
        return "precompile"

    def __repr__(self) -> str:
        return "torch.compiler.precompile"

    def __call__(
        self,
        fn: Callable[..., object],
        *example_args: object,
        example_inputs: Sequence[tuple[object, ...]] | None = None,
        backend: str = "inductor",
        tracer: str = "make_fx",
        decompositions: dict | None = None,
        recompile_limit: int | None = None,
        dynamic: bool | None = None,
    ) -> tuple[str, bytes]:
        """Ahead-of-time precompile ``fn`` against ``example_inputs``.

        .. note::

            ``torch.compiler.precompile`` is NOT
            ``torch._dynamo.config.caching_precompile`` (a ``torch.compile``
            guard-serialization caching mode); it captures ``fn`` ahead of time and
            lowers it to a self-contained Python source artifact.

        With the default ``make_fx`` tracer this is a non-strict trace with an explicit
        contract; read Note [precompile programming model] before using it. The artifact
        faithfully reproduces ``fn`` only for callers that uphold that contract.

        ``example_inputs`` is a sequence of positional-argument tuples for ``fn``.
        For compatibility, positional arguments after ``fn`` describe one example call;
        they cannot be combined with ``example_inputs``.
        The outer sequence supports capture front-ends that can specialize one artifact
        from multiple calls. The ``make_fx`` tracer accepts exactly one tuple because it
        records only one execution; ``dynamo`` executes every tuple and records the
        guarded recompilations they trigger. The Dynamo artifact retains serialized
        input-derived guards and treats the Python environment as invariant.

        THREADING: the inductor lowering step drives process-global compiler state
        and is serialized by an internal lock, so concurrent ``backend="inductor"``
        calls lower one at a time. Dynamo capture is also serialized because it uses
        process-global frame-evaluation and compilation state; the lock only covers
        precompile itself. For the duration of a Dynamo capture (including the
        caller's real steps in ``stateful``) these ``torch._dynamo.config`` values are
        patched process-wide, because they must be in effect while Dynamo converts
        the frame: ``fail_on_recompile_limit_hit=True`` and ``suppress_errors=False``
        (an artifact must never be built from a silently uncaptured eager fallback),
        ``caching_precompile=False`` (the private package must not be recorded into
        the global DynamoCache), ``strict_precompile=True`` (an unserializable guard
        aborts the capture instead of bypassing the package), and a raised
        ``accumulated_recompile_limit``. An unrelated ``torch.compile`` on another
        thread during a capture can observe those values. The functorch knobs the
        inductor lowering needs (``force_non_lazy_backward_lowering``,
        ``aot_autograd_prune_unused_outputs``) are patched only around each graph's
        lowering, under Dynamo's compile lock, so no other compile observes them.
        The make_fx capture phase and its ``backend="eager"`` path are NOT serialized.

        ``backend`` selects how the captured graph is realized:

        - ``"inductor"`` (default): lower the graph through
          ``torch._functorch.aot_autograd.compile_to_python`` (the full AOTAutograd +
          Inductor pipeline, composed into one self-contained module). ``python_code``
          is the inlined Inductor output with AOTAutograd's prelude/epilogue; the cache
          holds the save_cache_artifacts bundle that primes the inductor cache on load.
        - ``"eager"``: do NOT lower -- keep the captured ATen graph and run it as-is
          (analogous to ``torch.compile(backend="eager")``). ``python_code`` inlines
          the readable captured graph (both the inspectable rendering and the
          executable artifact); the eager cache carries no compiled artifact
          (artifact=None) but is still a full integrity-tagged envelope -- with no
          kernels there is nothing to accelerate, so ``load`` runs the inlined graph.
          Useful for
          inspecting/debugging exactly what was traced without an Inductor dependency.

        ``tracer`` selects the capture front-end:

        - ``"make_fx"`` (default): a NON-STRICT make_fx trace -- it records the ATen ops
          that actually run when ``fn`` executes once on the sole example-input tuple
          and does not analyze your Python, so control flow and shapes are specialized
          to the example (the source of the programming-model contract).
        - ``"dynamo"``: analyze a Python function's bytecode and capture every guarded
          specialization/recompilation exercised by ``example_inputs``. The emitted
          artifact retains guards derived from explicit inputs. Named guards covering
          the Python environment (module globals, imports) are dropped because that
          environment is required to be unchanged between capture and runtime. This
          assumption is unchecked, so changing the environment can silently run code
          specialized for its capture-time state; the dropped guards and their
          capture-time values are recorded in the artifact as ``_DROPPED_GUARDS``.
          Ambient torch state is checked per call: autocast, default dtype,
          deterministic-algorithms and torch-function state must match the capture
          (a mismatch raises, naming what differs); grad mode is pinned to the
          capture-time mode by the artifact and the thread count is not checked.
          A call that fails every retained input guard set raises, naming the failing
          guard, instead of compiling at runtime. Variants are dispatched newest
          first (the automatic-dynamic recompile of a shape supersedes the static
          variant it grew out of), which differs from ``CompilePackage.install``'s
          oldest-first order for the same serialized guards.
          This initial path requires one full graph (graph breaks
          are rejected), a function without closure cells, and positional tensor/scalar
          arguments or containers of those values (``nn.Module`` arguments are not
          supported yet). Every tensor the function uses must arrive through its
          arguments: a tensor read from a global, class, or module attribute is
          rejected, naming it, as are functions whose transformed bytecode reads or
          mutates a module global (the artifact runs in its own namespace, so the
          global would be missing or the mutation lost), distinct tensor inputs
          that share or overlap
          storage (the same tensor object may repeat; the loaded artifact also
          raises on overlapping runtime inputs when a captured graph mutates an
          input), and non-strided input layouts other than sparse (which surfaces
          Dynamo's own rejection).

        ``decompositions`` is an optional decomposition table (a dict mapping each
        ``OpOverload`` to a decomposition function) forwarded to ``make_fx`` as its
        ``decomposition_table`` during capture, so you can control how ATen ops are
        broken down in the captured graph. Defaults to ``None`` (make_fx's default) and
        is not yet supported with ``tracer="dynamo"``.

        With ``tracer="dynamo"``, differentiability is inferred per captured graph
        exactly as ``torch.compile`` infers it: capture runs under the caller's
        ambient grad mode, and under grad mode a graph whose inputs require grad is
        differentiable while no-grad inputs stay inference graphs. Examples may mix
        requires_grad states for the same input: requires_grad is guarded per input
        per variant, so each runtime call dispatches to a variant captured for its
        inputs' requires_grad states, and raises if none was. A capture under
        ``torch.no_grad()`` produces an inference artifact: it serves under
        ``torch.no_grad()`` (or with grad enabled when no input requires grad) and
        raises if called with grad enabled on a requires_grad input, since eager
        would build autograd history the artifact cannot. A grad-mode artifact
        called under ``torch.no_grad()`` returns eager no_grad results: fresh
        outputs carry no history and a view of an input is a no_grad view of it.
        How the backward is realized depends on the backend,
        mirroring ``torch.compile``: with ``backend="inductor"`` a differentiable
        graph is compiled as a joint forward+backward -- readable AOTAutograd source
        bridged by an emitted ``torch.autograd.Function`` -- so served outputs retain
        ``grad_fn`` and ``backward()`` executes the precompiled backward kernels;
        each example's actual backward records its output-tangent presence pattern
        and gets a backward specialized to it; a pattern not observed during
        capture falls back to the always-covered all-tangents-present backward
        (materializing the missing tangents, as torch.compile does) instead of
        compiling during serving. With ``backend="eager"`` the artifact's forward runs under live
        autograd, so ``backward()`` is ordinary eager autograd through the emitted
        ops -- neither captured nor specialized (any tangent pattern and higher-order
        grad work), and, like the eager forward itself, resolved against the loaded
        torch rather than frozen in the artifact.

        With ``tracer="dynamo"``, shape variation across ``example_inputs`` uses
        Dynamo's ordinary automatic dynamic-shape policy: for example, a static first
        graph can recompile into a symbolic graph when a later tuple changes a dimension.
        The symbolic graphs and their input-derived dispatch guards are retained in the
        artifact.

        To capture incrementally from a loop the caller owns (one example per
        call, an always-loadable artifact rewritten on disk), use the sibling
        entry point :meth:`precompile.stateful`; ``recompile_limit`` there
        defaults to ``max(torch._dynamo.config.recompile_limit, 256)``, while
        the one-shot default here is
        ``max(torch._dynamo.config.recompile_limit, len(example_inputs) + 1)``.

        With ``tracer="make_fx"``, dynamic shapes are opt-in via
        ``torch._dynamo.decorators.mark_unbacked``
        (inductor backend only), NOT a precompile kwarg: mark dims on the inputs before
        calling, e.g. ``mark_unbacked(x, 0); precompile(fn,
        example_inputs=[(model, x)])`` frees ``x``'s batch dim. Marked dims are captured
        as UNBACKED symints, which cannot be guarded on, so one artifact serves any
        runtime size of them (invariant 3); a graph that needs to guard on / specialize
        a marked dim fails at capture with a ``PrecompileError``. Dims sharing a
        ``shape_id`` reuse one symbol (equal by construction); ``min``/``max`` become
        runtime asserts. Other dims stay static.
        Dims that MUST be equal at runtime (e.g. two inputs combined by a broadcast that
        requires equal sizes, ``model(a) + model(b)``) MUST be given a SHARED ``shape_id``
        so a mismatch is rejected; marking two such dims INDEPENDENTLY currently bakes a
        SILENT equal-size assumption and a runtime mismatch does NOT raise the loud failure
        eager gives (invariant 3). This is a harvesting gap, not an inherent limit of the
        standalone artifact: the capture ShapeEnv DOES record the equality (as a deferred
        runtime assert, e.g. ``Eq(u0, u1)``), but precompile does not yet harvest/enforce
        those relational asserts in the driver -- only the decorator's declared min/max feed
        the runtime bound checks. A shared ``shape_id`` is the way to get the check today.

        Returns ``(python_code, cache)`` -- a self-contained, executable Python
        source string (the single source of truth for the calling convention) and a
        binary cache holding ONLY the backend artifact (NO metadata, NO weights).
        Reload a runnable with ``torch.compiler.precompile.load(python_code, cache)``.

        ``fn`` is the whole computation, e.g.::

            python_code, cache = torch.compiler.precompile(
                lambda model, x: model(x), example_inputs=[(model, x)]
            )


            def train_step(model, x, t):
                loss_fn(model(x), t).backward()  # or return autograd.grad(...)


            python_code, cache = torch.compiler.precompile(
                train_step, example_inputs=[(model, x, t)]
            )

        Within each tuple in ``example_inputs``, the ``nn.Module`` arguments have their
        params/buffers lifted to graph inputs (no weights are baked into the artifact --
        invariant 1); the rest are the runtime inputs. The reloaded callable is invoked
        with the SAME argument structure -- pass the model(s) again at runtime, e.g.
        ``f_c(model, x)``, and that runtime model must match the example model's
        parameter/buffer structure (invariant 2). Example inputs are positional
        tuples on both tracers. A ``make_fx`` artifact is called positionally too (a
        keyword call raises ``PrecompileError``); a ``dynamo`` artifact binds its
        arguments like the traced fn, so keyword arguments and defaults work. A wrong
        arity raises ``PrecompileError`` on both tracers. With ``tracer="make_fx"``,
        if ``fn`` ran a
        backward, the resulting parameter gradients are scattered (accumulated) onto
        that runtime model's ``parameters()`` ``.grad`` fields, exactly like eager ``.backward()``,
        so a ``zero_grad()`` / ``optimizer.step()`` loop works unchanged; the artifact
        returns ``fn``'s own result (``None`` for a bare ``.backward()`` step), not the
        grads (invariant 5).

        Input mutation (incl. module buffers, e.g. BatchNorm running stats in
        training mode), tensor subclasses (e.g. DTensor), and outputs aliasing inputs
        are supported -- AOTAutograd's prelude/epilogue is composed into the artifact
        (invariant 4), as is functionalized RNG. Caller responsibilities NOT checked
        here (see the Note): the runtime model must be structurally identical to the
        example, and control flow / shapes are specialized to the sole ``make_fx``
        example tuple (invariants 2 and 3). Violations that ARE checked raise
        ``PrecompileError``: a tensor baked as a constant (invariant 1), effectful ops
        (invariant 4), and -- for the
        inductor backend -- a runtime input whose stride / memory format differs from
        the example's (invariant 6). Error types: anything about the captured
        computation or its inputs (an unsupported construct, an environment tensor, a
        closure, an ``nn.Module`` or numpy argument on the Dynamo tracer, a rejected
        runtime input) raises ``PrecompileError``; ``TypeError`` / ``ValueError`` are
        reserved for Python-level misuse of this API itself (the types or values of
        ``example_inputs``, ``backend``, ``tracer``, and the Dynamo-only kwargs, or
        an example tuple that does not fit ``fn``'s signature).
        """
        torch._C._log_api_usage_once("torch.compiler.precompile")
        example_inputs = _normalize_example_inputs(example_args, example_inputs)
        if backend not in ("inductor", "eager"):
            raise ValueError(
                f"precompile backend must be 'inductor' or 'eager', got {backend!r}."
            )
        if tracer not in ("make_fx", "dynamo"):
            raise ValueError(
                f"precompile tracer must be 'make_fx' or 'dynamo', got {tracer!r}."
            )
        if (recompile_limit is not None or dynamic is not None) and tracer != "dynamo":
            raise ValueError(
                "precompile recompile_limit and dynamic require tracer='dynamo'."
            )
        if tracer == "dynamo":
            return _precompile_dynamo(
                fn,
                example_inputs,
                backend=backend,
                decompositions=decompositions,
                recompile_limit=recompile_limit,
                dynamic=dynamic,
            )
        compiled = PrecompiledModule(fn, backend=backend, decompositions=decompositions)
        compiled._compile(example_inputs)
        # Build the (expensive) python_code ONCE and thread it into to_cache_bytes so
        # the full metadata + embedded kernel source is not rebuilt, and so code_hash is
        # sha256 over exactly the bytes returned to the caller (a matched pair loads).
        python_code = compiled.to_python_code()
        return python_code, compiled.to_cache_bytes(python_code)

    def stateful(
        self,
        fn: Callable[..., object],
        *,
        example_inputs: Sequence[tuple[object, ...]],
        artifact_path: str,
        cache_path: str,
        state: PrecompileState | None = None,
        backend: str = "inductor",
        recompile_limit: int | None = None,
        dynamic: bool | None = None,
    ) -> tuple[list[object], PrecompileState]:
        """Capture ``fn`` incrementally from a loop the caller owns (Dynamo tracer).

        Some capture regions can only be exercised from the caller's own loop
        (e.g. a pipelined train step whose batch deque only advances between
        iterations), so ``precompile`` cannot drive the examples itself. Each
        ``stateful`` call runs its example tuples for real, records whatever
        guarded variants they newly exercise into the returned
        :class:`PrecompileState`, REWRITES the artifact and cache files at
        ``artifact_path``/``cache_path`` atomically, and returns
        ``(results, state)``: ``results`` is always a list with one entry per
        example tuple of THIS call (never unwrapped, so a fn that itself
        returns a list is unambiguous). The capture semantics -- tracer,
        rejections, guard filtering, grad mode, the programming-model contract --
        are exactly ``precompile(..., tracer="dynamo")``'s; only the delivery
        differs. Feed it the calls that add variants (new shapes, new
        branches), not every batch of a training loop -- rewriting is
        proportional to everything captured so far::

            state = None
            try:
                for batch in representative_batches:
                    [result], state = torch.compiler.precompile.stateful(
                        step,
                        example_inputs=[(batch,)],
                        state=state,
                        artifact_path="step.py",
                        cache_path="step.cache",
                    )
            finally:
                if state is not None:
                    state.close()

        A state you already hold is also a context manager that closes on exit::

            with state:
                _, state = torch.compiler.precompile.stateful(
                    step,
                    example_inputs=[(extra_batch,)],
                    state=state,
                    artifact_path="step.py",
                    cache_path="step.cache",
                )

        ``state=None`` starts fresh; passing a returned state resumes -- with
        the same ``fn``, ``backend``, ``recompile_limit``, ``dynamic``, and
        ambient grad mode, else the call raises ``ValueError`` rather than
        produce a mixed artifact. The files on disk are always a loadable
        artifact for everything captured so far. A call whose guards all hit
        adds nothing. Later calls see earlier variants because the state carries
        one isolate-recompiles bucket and one PGO record across calls (a
        dimension that varies between calls recompiles into a symbolic graph
        exactly as it would within one call). Each example runs on the caller's
        live objects and nothing is retained afterwards: a step may freely
        mutate its inputs.
        ``recompile_limit`` caps the variants per capture and defaults to
        ``max(torch._dynamo.config.recompile_limit, 256)`` because
        accumulating captures outgrow the config default. ``dynamic`` is
        forwarded to Dynamo (``None`` keeps the automatic policy). After each
        rewrite, ``state.summary()`` reports what the artifact carries -- calls,
        examples, variants, graphs, dynamic graphs, and the environment guards
        dropped from dispatch with their capture-time values (also embedded in
        the artifact as ``_DROPPED_GUARDS``). Load the on-disk pair directly
        with ``precompile.load_files(artifact_path, cache_path)``. The state is
        process-local and not serializable; call ``state.close()`` (or use
        ``with state:``) when done capturing to release the session, and do not
        call ``torch._dynamo.reset()`` between calls that share a state (later
        calls may raise or duplicate variants).
        """
        torch._C._log_api_usage_once("torch.compiler.precompile.stateful")
        example_inputs = _normalize_example_inputs((), example_inputs)
        if backend not in ("inductor", "eager"):
            raise ValueError(
                f"precompile backend must be 'inductor' or 'eager', got {backend!r}."
            )
        # Equal paths would make the cache write clobber the artifact, destroying
        # the previously loadable artifact.
        import os

        if os.path.realpath(artifact_path) == os.path.realpath(cache_path):
            raise ValueError(
                "precompile.stateful requires distinct artifact_path and "
                f"cache_path, got {artifact_path!r} for both."
            )
        return _precompile_dynamo_stateful(
            fn,
            example_inputs,
            backend=backend,
            recompile_limit=recompile_limit,
            dynamic=dynamic,
            state=state,
            artifact_path=artifact_path,
            cache_path=cache_path,
        )

    def load(self, python_code: str, cache: bytes | None) -> Callable[..., object]:
        """Reconstruct a runnable from the ``(python_code, cache)`` pair ``precompile`` returned.

        The driver runs from ``python_code`` -- the single source of truth for the whole
        calling convention. ``load`` reads the cache's ``BACKEND`` (to check the pairing)
        and, for the inductor backend, primes the inductor kernel caches from its
        ``save_cache_artifacts`` bundle (via ``torch.compiler.load_cache_artifacts``) so a
        warm reload loads precompiled kernels instead of JIT-compiling; then it exec's
        ``python_code``. ``cache=None`` (or an empty cache) means no cache: it degrades
        to JIT'ing from ``python_code`` with a warning. For the file pair a stateful
        capture writes, use :meth:`load_files`.

        Call the result with the SAME argument structure ``fn`` took -- the
        model(s) in their original positions plus the runtime inputs. Per invariant
        2 of Note [precompile programming model], the runtime model must match the
        example model's parameter/buffer structure; precompile re-derives the
        param/buffer list from it (same interning/order as capture).

        Raises ``TypeError`` if ``python_code`` is not a ``str`` or ``cache`` is not
        ``bytes`` / ``None``. Raises ``PrecompileError`` if ``python_code`` is
        malformed or is not a ``torch.compiler.precompile`` artifact (it fails to
        parse, or is missing the calling-convention metadata), if the cache's
        ``backend`` tag or ``code_hash`` does not match ``python_code`` (the cache and
        python_code came from different ``precompile()`` calls -- on both tracers),
        or if a Dynamo artifact is loaded under a different Python minor version or
        torch version than produced it. A cache whose ``format``/``version`` does not
        match (a foreign or different-build envelope) is NOT fatal: the cache is
        acceleration only, so ``load`` degrades to JIT'ing from ``python_code`` rather
        than crashing.
        """
        if not isinstance(python_code, str):
            raise TypeError(
                f"precompile.load python_code must be a str, got "
                f"{type(python_code).__name__}."
            )
        if cache is not None and not isinstance(cache, bytes):
            raise TypeError(
                f"precompile.load cache must be bytes or None, got {type(cache).__name__}."
            )
        if not cache:
            log.warning(
                "torch.compiler.precompile.load got an empty cache (None or b''); "
                "falling back to JIT from python_code."
            )
        return _load_artifact(python_code, cache or None, degrade_on_mismatch=False)

    def load_files(self, artifact_path: str, cache_path: str) -> Callable[..., object]:
        """Reconstruct a runnable from the file pair a ``stateful`` capture wrote.

        Reads ``artifact_path`` (the self-contained Python artifact) and
        ``cache_path`` (the binary acceleration cache) and loads them like
        :meth:`load`, with the pairing rules a rewritten-on-disk pair needs: a
        stateful rewrite renames the artifact and then the cache, so a crash
        between the two leaves a NEW artifact with the OLD cache, and the first
        rewrite's crash window leaves an artifact with no cache file at all. Both
        degrade to a cold cache with a warning (the artifact is fully
        self-contained), and the next successful rewrite repairs the pair.

        Raises ``FileNotFoundError`` if ``artifact_path`` does not exist, and
        ``PrecompileError`` as :meth:`load` does for a malformed artifact or a
        version mismatch. A cache whose ``backend``/``code_hash`` does not match the
        artifact degrades to a cold cache with a warning instead of raising.
        """
        with open(artifact_path, encoding="utf-8") as f:
            python_code = f.read()
        try:
            with open(cache_path, "rb") as f:
                cache: bytes | None = f.read()
        except FileNotFoundError:
            log.warning(
                "torch.compiler.precompile.load_files found no cache file at %r "
                "(likely a first rewrite interrupted between the artifact and "
                "cache renames). Falling back to JIT from python_code.",
                cache_path,
            )
            cache = None
        return _load_artifact(python_code, cache or None, degrade_on_mismatch=True)


precompile = _PrecompileApi()
# ``torch.compiler.precompile`` is a callable instance, not a function, so give it the
# name/doc introspection (Sphinx autosummary, help(), IDEs) expects to find on a
# public callable; the rich usage docs live on ``__call__``.
precompile.__name__ = "precompile"  # type: ignore[attr-defined]
precompile.__qualname__ = "precompile"  # type: ignore[attr-defined]
precompile.__doc__ = _PrecompileApi.__call__.__doc__

# These are public under torch.compiler.precompile, so report their module/qualname there
# (mirroring the singleton fixup above) -- otherwise Sphinx autoexception/autofunction
# would anchor them under this private module. The loaders are bound methods; patch the
# underlying functions so introspection on precompile.load reports torch.compiler too.
PrecompileError.__module__ = "torch.compiler"
PrecompileError.__qualname__ = "precompile.PrecompileError"
PrecompileState.__module__ = "torch.compiler"
PrecompileState.__qualname__ = "precompile.PrecompileState"
PrecompileStateSummary.__module__ = "torch.compiler"
PrecompileStateSummary.__qualname__ = "precompile.PrecompileStateSummary"
_PrecompileApi.load.__module__ = "torch.compiler"
_PrecompileApi.load.__qualname__ = "precompile.load"
_PrecompileApi.load_files.__module__ = "torch.compiler"
_PrecompileApi.load_files.__qualname__ = "precompile.load_files"
_PrecompileApi.stateful.__module__ = "torch.compiler"
_PrecompileApi.stateful.__qualname__ = "precompile.stateful"
