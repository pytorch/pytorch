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
guards covering the Python environment may be dropped because that environment is a
caller-provided invariant. This invariant is unchecked, so changing the environment can
silently run code specialized for its capture-time state. The artifact never compiles
after loading; a call that fails every retained guard set raises. Compiled graphs and kernels remain Python source, while
guard trees and transformed bytecode are stored as opaque inline data.

With ``tracer="dynamo", training=True`` (inductor backend only), every compiled graph
contains AOTAutograd's forward and backward as readable Inductor source. The served
output retains its ``grad_fn`` and a later ``backward()`` executes those captured
backward kernels across captured recompilations. Backward variants are specialized to
output-tangent patterns observed while running the examples, and an unseen pattern fails
instead of compiling at runtime.

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
#    Distributed capture: a ``compile_on_one_rank`` flag (trace on a single rank and
#    broadcast the artifact to the rest, so every rank need not re-capture) is
#    anticipated and scheduled for a follow-up later in this stack.
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
#    metadata; for dynamo artifacts the mismatch degrades to a cold cache with a
#    warning (python_code is fully self-contained, and a stateful rewrite
#    interrupted between its two renames leaves exactly such a pair).
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
# specialization/recompilation exercised by example_inputs, minimizes guard records
# while preserving dispatch for those calls, and dispatches among the variants at
# runtime. It currently requires one full graph (no graph breaks).
#
# The dynamo tracer's artifact format depends on these torch._dynamo internals
# (changes to them are format/behavior changes here): package.CompilePackage /
# load_guards_state / load_guard_manager / SerializedCode, guards.
# CheckFunctionManager rebuilt over a dataclasses.replace'd OutputGraphCommon
# (guard minimization), CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES,
# convert_frame.GUARDS_STATE_NONE_MESSAGE, pgo._use_code_state, and the
# eval_frame.cached_backends / utils.guard_failures registries (teardown).

from __future__ import annotations

import contextlib
import hashlib
import io
import logging
import threading
import weakref
from types import MappingProxyType
from typing import Any, cast, NamedTuple, NewType, TYPE_CHECKING

import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


log = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

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
        "TRAINING",
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
        if target.id in wanted:
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


def _load_dynamo_state(python_code: str) -> dict[str, Any]:
    """Decode the opaque Dynamo state pickled into a tracer='dynamo' artifact.

    The AST walk (rather than line matching) keeps callers robust to formatting
    changes in the emitted assignment; it is the read-side counterpart of
    _build_dynamo_python_source's ``_DYNAMO_STATE = ...`` line.
    """
    import ast
    import base64
    import pickle

    try:
        tree = ast.parse(python_code)
    except SyntaxError as e:
        raise PrecompileError(
            f"python_code is not parseable Python ({e}); it does not look like a "
            "torch.compiler.precompile artifact."
        ) from e
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_DYNAMO_STATE"
        ):
            encoded = ast.literal_eval(node.value)
            return pickle.loads(base64.b64decode(encoded))
    raise PrecompileError(
        "python_code has no _DYNAMO_STATE assignment; it is not a tracer='dynamo' "
        "precompile artifact."
    )


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
        call: Callable[[list[object]], object],
        compile_state: Any | None = None,
    ) -> None:
        self.python_code = python_code
        self.cache = cache
        self.is_dynamic = is_dynamic
        self._call = call
        self._compile_state = compile_state
        if compile_state is not None:
            globals_dict = getattr(call, "__globals__", None)
            if not isinstance(globals_dict, dict):
                raise AssertionError("training backend call must have Python globals")
            compile_state.install_capture(globals_dict)

    def __call__(self, *args: object) -> object:
        return self._call(list(args))

    def _finalize_masks(self) -> tuple[int, ...]:
        globals_dict = getattr(self._call, "__globals__", {})
        observed = globals_dict.get("_AOT_OBSERVED_UNDEFINED_TANGENT_MASKS", ())
        # Always cover the ordinary all-tangents-defined backward (mask 0): a
        # caller who only ran partial backwards between stateful calls must not
        # regress the rewritten artifact's default path.
        return (0, *sorted(set(observed) - {0}))

    def finalize_training(self, keep_capture: bool = False) -> None:
        # keep_capture snapshots the training module (finalize is a pure compose
        # over the variants recorded so far) while leaving the live variant
        # compiler hook installed, so a stateful capture can keep accumulating.
        if self._compile_state is None:
            return
        globals_dict = getattr(self._call, "__globals__", {})
        masks = self._finalize_masks()
        if keep_capture:
            try:
                self.python_code, self.cache = self._compile_state.finalize(masks)
            except Exception:
                # A mask whose deferred compile fails must not poison every later
                # rebuild (it stays in the observed set forever): drop it from
                # this snapshot with a warning. Serving that pattern then fails
                # like any unobserved pattern.
                compiled = set(self._compile_state._observed_variants)
                usable = tuple(mask for mask in masks if mask in compiled)
                if set(usable) == set(masks):
                    raise
                log.warning(
                    "precompile stateful snapshot could not compile tangent "
                    "mask(s) %s; the rewritten artifact covers only %s.",
                    sorted(set(masks) - set(usable)),
                    list(usable) or [0],
                )
                self.python_code, self.cache = self._compile_state.finalize(
                    usable or (0,)
                )
            return
        try:
            self.python_code, self.cache = self._compile_state.finalize(masks)
        finally:
            globals_dict["_AOT_BACKWARD_VARIANT_COMPILER"] = None
            variants = globals_dict.get("_AOT_BACKWARD_VARIANTS")
            if isinstance(variants, dict):
                variants.clear()
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


def _dynamo_backend_compiler(
    backend: str, training: bool
) -> Callable[..., _DynamoPythonBackend]:
    def compile_graph(
        gm: torch.fx.GraphModule, example_inputs: list[object]
    ) -> _DynamoPythonBackend:
        from torch._functorch import aot_autograd
        from torch._functorch._aot_autograd.to_standalone_python import (
            _compile_to_python_with_state,
            _graph_has_dynamic_shapes,
        )

        is_dynamic = _graph_has_dynamic_shapes(gm)
        if backend == "eager":
            python_code = _build_dynamo_eager_graph_source(gm)
            cache = None
            namespace: dict[str, object] = {"__name__": "_dynamo_eager_graph"}
            exec(compile(python_code, "<dynamo-eager-graph>", "exec"), namespace)
            call = cast("Callable[[list[object]], object]", namespace["call"])
            compile_state = None
        else:
            # Dynamo's runtime examples may have concrete tensor sizes while the graph
            # metadata carries the symbolic sizes and sources selected for this variant.
            graph_inputs = [
                node.meta["example_value"]
                for node in gm.graph.nodes
                if node.op == "placeholder"
            ]
            python_code, cache, compile_state = _compile_to_python_with_state(
                gm,
                graph_inputs,
                options={"size_asserts": True},
                grad_enabled=training,
            )
            call = aot_autograd.load_from_python(python_code, cache)
        return _DynamoPythonBackend(python_code, cache, is_dynamic, call, compile_state)

    return compile_graph


def _filter_dynamo_guards(
    target: Callable[..., object],
    guarded_codes: Sequence[Any],
    example_inputs: Sequence[tuple[object, ...]],
) -> tuple[list[bytes], tuple[tuple[str, str], ...]]:
    """Drop environment guards while preserving every input-derived guard.

    Returns the re-serialized guard states plus the dropped guards as sorted,
    deduplicated (guard type, source) pairs aggregated across variants: a
    listed pair was dropped from at least one variant's dispatch, but may
    remain checked in a variant where dropping it would have changed how the
    capture examples dispatch.
    """
    import dataclasses
    import functools
    import inspect

    from torch._dynamo.guards import CheckFunctionManager, GuardBuilder
    from torch._dynamo.output_graph import OutputGraphCommon
    from torch._dynamo.package import load_guard_manager, load_guards_state
    from torch._dynamo.source import LocalSource
    from torch._guards import ChainedSource, GuardsSet
    from torch.utils._ordered_set import OrderedSet

    signature = inspect.signature(target)
    example_scopes: list[dict[str, object]] = []
    for example in example_inputs:
        bound = signature.bind(*example)
        bound.apply_defaults()
        example_scopes.append(dict(bound.arguments))
    input_names = {name for scope in example_scopes for name in scope}

    def is_input_guard(guard: Any) -> bool:
        # Classify by the originating source's root, not by a string prefix on
        # the rendered name: many input-rooted sources render as call
        # expressions (___tuple_iterator_getitem(L['it'], 0),
        # ___from_numpy(L['x']), type(L['x']), list(dict.keys(L['d']))[0], ...)
        # that an "L['name']"-prefix test misclassifies as environment guards;
        # dropping one always succeeds (a variant's guards pass on the examples
        # that produced it) and the artifact then silently serves the
        # capture-time specialization for calls whose values differ.
        source = guard.originating_source
        while isinstance(source, ChainedSource):
            source = source.base
        return isinstance(source, LocalSource) and source.local_name in input_names

    def fresh_guard(guard: Any, *, final: bool = False) -> Any:
        create_fn = guard.create_fn
        if (
            final
            and isinstance(create_fn, functools.partial)
            and create_fn.func is GuardBuilder.TENSOR_MATCH
        ):
            create_fn = GuardBuilder.TENSOR_MATCH
        return dataclasses.replace(
            guard,
            create_fn=create_fn,
            guard_types=None,
            code_list=None,
            obj_weakref=None,
            guarded_class_weakref=None,
            _hash=None,
        )

    def manager_for(state: Any, output_graph: Any | None = None) -> Any:
        if output_graph is not None:
            state = dataclasses.replace(state, output_graph=output_graph)
        return load_guard_manager(state, target.__code__, target.__globals__)

    def outcomes(
        state: Any,
        *,
        guards: Sequence[Any],
        aot_guards: Sequence[Any],
        key_order: Sequence[Any],
    ) -> list[bool]:
        output_graph = dataclasses.replace(
            state.output_graph,
            _guards=GuardsSet(OrderedSet(fresh_guard(guard) for guard in guards)),
            _aotautograd_guards=list(aot_guards),
            guard_on_key_order=set(key_order),
        )
        manager = manager_for(state, output_graph)
        return [manager.check(scope) for scope in example_scopes]

    filtered_states: list[bytes] = []
    dropped_guards: set[tuple[str, str]] = set()
    for guarded in guarded_codes:
        state = load_guards_state(guarded.guards_state)
        kept_guards = list(state.output_graph.guards)
        kept_aot_guards = list(state.output_graph.aotautograd_guards)
        kept_key_order = sorted(
            state.output_graph.guard_on_key_order, key=lambda source: source.name
        )
        baseline = outcomes(
            state,
            guards=kept_guards,
            aot_guards=kept_aot_guards,
            key_order=kept_key_order,
        )
        matching_scopes = [
            scope
            for scope, matches in zip(example_scopes, baseline, strict=True)
            if matches
        ]
        if not matching_scopes:
            raise PrecompileError(
                "precompile tracer='dynamo' captured a variant that does not match "
                "any example input."
            )

        # AOT and key-order guard records are input-derived or structurally
        # required, so only kept_guards is a minimization candidate.
        def try_drop(index: int) -> bool:
            guard = kept_guards[index]
            if not guard.name or is_input_guard(guard):
                return False
            candidate = kept_guards[:index] + kept_guards[index + 1 :]
            try:
                return (
                    outcomes(
                        state,
                        guards=candidate,
                        aot_guards=kept_aot_guards,
                        key_order=kept_key_order,
                    )
                    == baseline
                )
            except Exception:
                # Some records construct relational checks jointly. If removing one
                # leaves an invalid manager, it is a required dependency.
                return False

        changed = True
        while changed:
            changed = False
            index = 0
            while index < len(kept_guards):
                if try_drop(index):
                    guard = kept_guards[index]
                    dropped_guards.add((guard.create_fn_name(), guard.name or ""))
                    del kept_guards[index]
                    changed = True
                else:
                    index += 1

        output_graph = dataclasses.replace(
            state.output_graph,
            local_scope={**state.output_graph.local_scope, **matching_scopes[0]},
            _guards=GuardsSet(
                OrderedSet(fresh_guard(guard, final=True) for guard in kept_guards)
            ),
            _aotautograd_guards=kept_aot_guards,
            guard_on_key_order=set(kept_key_order),
        )
        shape_code_parts = (
            state.shape_code_parts
            if any(guard.create_fn_name() == "SHAPE_ENV" for guard in kept_guards)
            else None
        )
        check_fn = CheckFunctionManager(
            target.__code__,
            OutputGraphCommon(output_graph),
            shape_code_parts=shape_code_parts,
            runtime_global_scope=target.__globals__,
            save_guards=True,
            strict_error=True,
            guard_build_local_state=state.local_state,
        )
        if check_fn.guards_state is None:
            raise PrecompileError(
                "precompile tracer='dynamo' could not re-serialize its minimized "
                "guards."
            )
        filtered_state = load_guards_state(check_fn.guards_state)
        filtered_manager = manager_for(filtered_state)
        filtered_outcomes = [filtered_manager.check(scope) for scope in example_scopes]
        if filtered_outcomes != baseline:
            raise PrecompileError(
                "precompile tracer='dynamo' guard filtering changed captured "
                "example dispatch."
            )
        filtered_states.append(check_fn.guards_state)

    return filtered_states, tuple(sorted(dropped_guards))


def _dynamo_backend_source_literal(source: str) -> str:
    escaped = source.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    return f'    """\n{escaped}\n"""[1:-1],'


def _build_dynamo_python_source(
    *,
    backend: str,
    training: bool,
    state: dict[str, Any],
    backend_ids: list[str],
    compiled_backends: list[_DynamoPythonBackend],
    dropped_guards: tuple[tuple[str, str], ...],
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
        "# only minimized dispatch guard state plus bytecode as base64-encoded pickle data.",
        "",
        "# " + "=" * 70,
        "# 1. Capture metadata and compiled Python graph sources",
        "# " + "=" * 70,
        f"BACKEND = {backend!r}",
        'TRACER = "dynamo"',
        f"TRAINING = {training!r}",
        f"VARIANT_COUNT = {len(state['variants'])}",
        f"GRAPH_COUNT = {len(compiled_backends)}",
        f"DYNAMIC_GRAPH_COUNT = {dynamic_count}",
        f"_DYNAMO_PYTHON_VERSION = {tuple(sys.version_info[:2])!r}",
        f"_DYNAMO_TORCH_VERSION = {torch.__version__!r}",
        f"_DYNAMO_BACKEND_IDS = {tuple(backend_ids)!r}",
        "# (guard type, source) pairs dropped from at least one variant's dispatch:",
        "# they only cover the Python environment, which is a caller-provided",
        "# invariant. A listed pair may still be checked by other variants, and every",
        "# retained input-derived guard still gates dispatch.",
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
    *,
    environment_scan: dict[str, tuple[frozenset[int], bool]] | None = None,
) -> tuple[Callable[..., object], dict[str, tuple[frozenset[int], bool]]]:
    """Reject unsupported dynamo-capture inputs; return (target, environment_scan).

    ``environment_scan`` maps each referenced global to (reachable object ids,
    tensor reached). Walking those object graphs is the expensive part of
    validation and is input-independent, so stateful capture computes it once
    when the state is created and passes it back on every resume (the
    programming model declares the environment invariant during capture, so a
    caller who mutates it after creation is outside the contract and a
    post-creation alias will NOT be re-detected -- it can silently serve
    capture-time results); only
    the cheap per-call id intersection against the current example inputs runs
    every time. Passing a scan also skips the tensor-global rejection, which
    was decided at scan time.
    """
    import dis
    import enum
    import functools
    import inspect
    import pickle
    import sys
    import types

    if not example_inputs:
        raise ValueError(
            "precompile with tracer='dynamo' requires at least one example input tuple."
        )
    if decompositions is not None:
        raise NotImplementedError(
            "precompile decompositions are not yet supported with tracer='dynamo'."
        )

    def is_library_module(module_name: str | None) -> bool:
        root = (module_name or "").partition(".")[0]
        return root == "torch" or root in sys.stdlib_module_names

    def instance_values(root: object) -> Iterator[object]:
        # The narrow, per-argument walker: an argument is judged only by what
        # it carries -- containers, __dict__ contents, and slot values -- not
        # by attributes of its type or by modules. Functions, modules, types,
        # and tensors are yielded but not descended. The wide environment walk
        # (classes through the MRO, user modules, instance types) lives in
        # environment_reachable below.
        seen: set[int] = set()
        stack: list[object] = [root]
        while stack:
            value = stack.pop()
            if value is None or type(value) in (bool, int, float, complex, str, bytes):
                continue
            if id(value) in seen:
                continue
            seen.add(id(value))
            yield value
            if isinstance(
                value, (enum.Enum, torch.dtype, torch.layout, torch.memory_format)
            ):
                # Value-guarded singleton leaves: an enum member's __dict__
                # carries __objclass__ (its class), which would otherwise read
                # as an environment alias of the enum's global.
                continue
            if isinstance(value, dict):
                stack.extend(value.keys())
                stack.extend(value.values())
                continue
            if isinstance(value, (tuple, list, set, frozenset)):
                stack.extend(value)
                continue
            if isinstance(
                value, (types.FunctionType, types.ModuleType, type, torch.Tensor)
            ):
                continue
            if hasattr(value, "__dict__"):
                stack.extend(vars(value).values())
            for cls in type(value).__mro__:
                for descriptor in vars(cls).values():
                    if isinstance(descriptor, types.MemberDescriptorType):
                        try:
                            stack.append(descriptor.__get__(value, type(value)))
                        except AttributeError:
                            pass

    def reaches(value: object, predicate: Callable[[object], bool]) -> bool:
        return any(predicate(item) for item in instance_values(value))

    def environment_reachable(root: object) -> tuple[frozenset[int], bool]:
        # The wide environment walk: enumerate every object id reachable from a
        # referenced global -- through containers, instance state, slot values,
        # class attributes (via the MRO), each instance's type, and user
        # (non-library) modules -- plus whether a tensor is reachable. A tensor
        # or an aliased input on a class or module attribute is invisible to
        # the narrow walk, would have its guard classified as environment, and
        # would serve a raw NameError or the capture-time branch.
        ids: set[int] = set()
        found_tensor = False
        stack: list[object] = [root]
        while stack:
            value = stack.pop()
            if value is None or type(value) in (
                bool,
                int,
                float,
                complex,
                str,
                bytes,
            ):
                continue
            value_id = id(value)
            if value_id in ids:
                continue
            ids.add(value_id)
            if isinstance(value, torch.Tensor):
                found_tensor = True
                continue
            if isinstance(value, dict):
                stack.extend(value.keys())
                stack.extend(value.values())
                continue
            if isinstance(value, (tuple, list, set, frozenset)):
                stack.extend(value)
                continue
            if isinstance(value, types.ModuleType):
                # Library modules are part of the environment; user modules can
                # carry tensors or aliased inputs as attributes.
                if not is_library_module(value.__name__):
                    stack.extend(vars(value).values())
                continue
            if isinstance(value, type):
                if not is_library_module(value.__module__):
                    stack.extend(
                        item
                        for cls in value.__mro__
                        if not is_library_module(cls.__module__)
                        for item in vars(cls).values()
                    )
                continue
            if isinstance(value, types.FunctionType):
                # A function carries state too: tensors or input aliases can
                # hide in its defaults, closure cells, attributes, or -- for a
                # user function Dynamo may inline -- in globals its own code
                # loads that the root fn's bytecode never names. Library
                # functions are opaque leaves, like library modules.
                if is_library_module(getattr(value, "__module__", None)):
                    continue
                stack.extend(value.__defaults__ or ())
                stack.extend((value.__kwdefaults__ or {}).values())
                for cell in value.__closure__ or ():
                    try:
                        stack.append(cell.cell_contents)
                    except ValueError:
                        pass
                stack.extend(vars(value).values())
                stack.extend(
                    value.__globals__[name]
                    for name in loaded_global_names(value.__code__)
                    if name in value.__globals__
                )
                continue
            if isinstance(value, types.MethodType):
                stack.extend((value.__func__, value.__self__))
                continue
            if isinstance(value, functools.partial):
                stack.append(value.func)
                stack.extend(value.args)
                stack.extend(value.keywords.values())
                continue
            if hasattr(value, "__dict__"):
                stack.extend(vars(value).values())
            for cls in type(value).__mro__:
                for descriptor in vars(cls).values():
                    if isinstance(descriptor, types.MemberDescriptorType):
                        try:
                            stack.append(descriptor.__get__(value, type(value)))
                        except AttributeError:
                            pass
            stack.append(type(value))
        return frozenset(ids), found_tensor

    def has_storage_overlap(values: object) -> bool:
        # The same tensor object passed twice is a supported identity relation
        # (Dynamo's serialized aliasing guards cover it); DISTINCT tensors that
        # share or overlap storage are not: their AOT StorageOverlap relation
        # does not survive serialization, so a mutating variant could silently
        # serve wrong results. The check is deliberately STORAGE-granular
        # (non-overlapping views of one buffer are also rejected) because
        # AOTAutograd's synthetic-base mutation handling keys on shared
        # storage, not on element overlap, so storage identity is the
        # boundary the missing guards would have enforced. Only SPARSE layouts
        # are skipped (Dynamo rejects those with its own clearer diagnostics);
        # any other non-strided layout (e.g. jagged) is rejected outright,
        # regardless of tensor count, because its aliasing cannot be verified.
        sparse_layouts = (
            torch.sparse_coo,
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        )
        # Tensor enumeration must be DEEP (containers, __dict__, slot values):
        # a tensor inside a custom, non-pytree argument is invisible to
        # tree_leaves and an aliased pair would bypass the check. Keep this
        # enumeration in sync with the driver copy in
        # torch/_precompile_driver.py (test_precompile pins the parity).
        tensors = []
        for value in instance_values(values):
            if not isinstance(value, torch.Tensor) or value.layout in sparse_layouts:
                continue
            if value.layout is not torch.strided:
                raise PrecompileError(
                    "precompile tracer='dynamo' cannot verify storage overlap "
                    f"for a {value.layout} layout tensor input."
                )
            tensors.append(value)
        if len(tensors) < 2:
            return False
        storage_ranges: dict[tuple[str, int | None], list[tuple[int, int]]] = {}
        storage_ids: set[tuple[str, int | None, int]] = set()
        seen_objects: set[int] = set()
        for tensor in tensors:
            if id(tensor) in seen_objects:
                continue
            seen_objects.add(id(tensor))
            try:
                storage = tensor.untyped_storage()
                start = storage.data_ptr()
                size = storage.nbytes()
            except RuntimeError as e:
                raise PrecompileError(
                    "precompile tracer='dynamo' cannot verify storage overlap "
                    "for this tensor input."
                ) from e
            storage_key = (tensor.device.type, tensor.device.index, storage._cdata)
            if storage_key in storage_ids:
                return True
            storage_ids.add(storage_key)
            # data_ptr() == 0 (meta/fake or unallocated storages) is excluded
            # from the range check; identity via _cdata above still applies.
            if start != 0 and size > 0:
                storage_ranges.setdefault(
                    (tensor.device.type, tensor.device.index), []
                ).append((start, start + size))
        for ranges in storage_ranges.values():
            furthest_end = 0
            for start, end in sorted(ranges):
                if start < furthest_end:
                    return True
                furthest_end = max(furthest_end, end)
        return False

    try:
        import numpy
    except ImportError:
        numpy = None  # type: ignore[assignment]

    for example in example_inputs:
        if not isinstance(example, tuple):
            raise TypeError(
                "precompile example_inputs must be a sequence of positional-argument "
                f"tuples, got {type(example).__name__}."
            )
        if reaches(example, lambda v: isinstance(v, torch.nn.Module)):
            raise NotImplementedError(
                "precompile tracer='dynamo' does not yet support nn.Module arguments "
                "(including inside containers) because Dynamo's module identity "
                "guards are not serializable."
            )
        if numpy is not None and reaches(
            example, lambda v: isinstance(v, numpy.ndarray)
        ):
            # Dynamo traces ndarrays via ___from_numpy sources whose TENSOR_MATCH
            # guard construction fails under the package/save-guards path, so
            # capture would die with an internal error; reject up front.
            raise NotImplementedError(
                "precompile tracer='dynamo' does not yet support numpy.ndarray "
                "arguments (including inside containers); convert them to tensors "
                "with torch.from_numpy and pass those instead."
            )
        if has_storage_overlap(example):
            raise PrecompileError(
                "precompile tracer='dynamo' does not support distinct tensor "
                "inputs that share or overlap storage; pass the same tensor "
                "object, or clone the views into separate tensors."
            )

    from torch._dynamo.eval_frame import innermost_fn

    target = innermost_fn(fn)
    if not inspect.isfunction(target):
        raise NotImplementedError(
            "precompile tracer='dynamo' currently requires a Python function and does "
            "not accept an nn.Module or bound method directly as fn."
        )
    if target.__closure__ is not None:
        raise NotImplementedError(
            "precompile tracer='dynamo' does not yet support functions with closure "
            "cells; pass captured values as explicit arguments."
        )

    defaults = (target.__defaults__, target.__kwdefaults__)
    if reaches(defaults, lambda v: isinstance(v, torch.Tensor)):
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

    def loaded_global_names(code: types.CodeType) -> set[str]:
        names = {
            instruction.argval
            for instruction in dis.get_instructions(code)
            if instruction.opname
            in ("LOAD_GLOBAL", "LOAD_NAME", "LOAD_FROM_DICT_OR_GLOBALS")
            and isinstance(instruction.argval, str)
        }
        for constant in code.co_consts:
            if isinstance(constant, types.CodeType):
                names.update(loaded_global_names(constant))
        return names

    def mutates_globals(code: types.CodeType) -> bool:
        return any(
            instruction.opname in ("STORE_GLOBAL", "DELETE_GLOBAL")
            for instruction in dis.get_instructions(code)
        ) or any(
            mutates_globals(constant)
            for constant in code.co_consts
            if isinstance(constant, types.CodeType)
        )

    # A capture runs against a copy of fn.__globals__ and the artifact runs
    # against its own namespace, so a global mutation would never reach the
    # caller's module -- at capture or at serve. Reject rather than silently
    # drop the side effect.
    if mutates_globals(target.__code__):
        raise PrecompileError(
            "precompile tracer='dynamo' cannot capture a Python function that "
            "mutates globals; return the value instead."
        )

    fresh_scan = environment_scan is None
    if fresh_scan:
        environment_scan = {
            name: environment_reachable(target.__globals__[name])
            for name in sorted(loaded_global_names(target.__code__))
            if name in target.__globals__
        }

    # torch.dtype/layout/memory_format are process-wide singletons whose
    # guards are value-based and whose pickles return the same object, so
    # sharing one with the environment carries no identity hazard
    # (torch.device is NOT exempt: equal devices need not be identical). Enum
    # members are exempt for a different reason: Dynamo puts an ID_MATCH on
    # every realized enum argument and ID_MATCH is unserializable, so a USED
    # enum argument always fails capture loudly with the accurate
    # identity-guard error, while an unused pass-through enum (which this
    # exemption newly enables) never influences dispatch.
    # The id set must be as deep as what an argument carries (instance_values:
    # containers, __dict__, slot values): a tensor inside a custom object that
    # aliases the environment has the same dropped-guard hazard as a bare
    # aliased leaf. instance_values already skips primitive leaves.
    # Interpreter-wide singletons ((), Ellipsis, NotImplemented) are exempt
    # like the torch singletons above: Dynamo value-guards them, and their
    # process-wide identity would otherwise read any helper default or class
    # attribute holding one as an alias of the caller's input. Functions,
    # classes, and modules an argument merely carries are exempt for the enum
    # rationale: a USED one gets an unserializable identity guard and fails
    # capture loudly; an unused reference never influences dispatch.
    singleton_ids = {id(()), id(Ellipsis), id(NotImplemented)}
    input_ids = {
        id(value)
        for example in example_inputs
        for value in instance_values(example)
        if not isinstance(
            value,
            (
                enum.Enum,
                torch.dtype,
                torch.layout,
                torch.memory_format,
                types.FunctionType,
                types.MethodType,
                types.ModuleType,
                type,
            ),
        )
        and id(value) not in singleton_ids
    }
    # An input aliased through the environment (including a global's class or
    # module attributes) would have its identity guard classified as
    # environment and dropped, silently serving the capture-time branch for
    # inputs that no longer alias.
    aliased_globals = [
        name
        for name, (reachable_ids, _) in environment_scan.items()
        if not reachable_ids.isdisjoint(input_ids)
    ]
    if aliased_globals:
        raise PrecompileError(
            "precompile tracer='dynamo' cannot capture an explicit input that aliases "
            "the Python environment; pass the value only as an input. Aliased global: "
            f"{aliased_globals[0]!r}."
        )
    if fresh_scan:
        # A referenced tensor-valued global would be guarded by identity and
        # read through the fn's globals at serve time, where the artifact
        # namespace has no such name: capture would succeed and serving would
        # fail with a raw NameError. Deliberately conservative: a global is
        # rejected when ANY tensor is reachable from it, even if the fn never
        # reads the tensor itself.
        tensor_globals = [
            name for name, (_, found_tensor) in environment_scan.items() if found_tensor
        ]
        if tensor_globals:
            raise PrecompileError(
                "precompile tracer='dynamo' cannot capture a Python global whose "
                f"object graph contains a tensor (global {tensor_globals[0]!r}); "
                "every tensor must be an explicit input, so pass the values the "
                "function needs as arguments instead."
            )
    return target, environment_scan


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


def _keep_serializable_capture_guards(guards: Sequence[Any]) -> list[bool]:
    from torch._dynamo.guards import CheckFunctionManager

    unsupported = CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
    return [
        not (
            guard.is_global
            and (
                guard.guard_type in unsupported
                or any(kind in unsupported for kind in guard.derived_guard_types)
            )
        )
        for guard in guards
    ]


@contextlib.contextmanager
def _dynamo_capture_context(
    pgo_state: Any, training: bool, capture_limit: int
) -> Iterator[None]:
    import torch._functorch.config as functorch_config
    from torch._dynamo.pgo import _use_code_state

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            torch._dynamo.config.patch(
                accumulated_recompile_limit=max(
                    torch._dynamo.config.accumulated_recompile_limit, capture_limit
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
            )
        )
        stack.enter_context(
            functorch_config.patch(
                bundled_autograd_cache=True,
                bypass_autograd_cache_key=True,
                force_non_lazy_backward_lowering=training,
            )
        )
        stack.enter_context(_use_code_state(pgo_state))
        stack.enter_context(torch.inference_mode(False))
        stack.enter_context(torch.set_grad_enabled(training))
        yield


@contextlib.contextmanager
def _translate_dynamo_capture_errors(
    capture_limit: int, *, stateful: bool = False
) -> Iterator[None]:
    from torch._dynamo.exc import (
        BackendCompilerFailed,
        FailOnRecompileLimitHit,
        PackageError,
        RecompileError,
        Unsupported,
    )

    try:
        yield
    except Unsupported as e:
        raise PrecompileError(
            "precompile tracer='dynamo' does not support graph breaks yet; capture must "
            f"produce one full graph. Dynamo reported: {e}"
        ) from e
    except BackendCompilerFailed as e:
        if isinstance(e.inner_exception, PrecompileError):
            raise e.inner_exception from e
        raise
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
    except AssertionError as e:
        # torch/_dynamo/convert_frame.py raises this assertion when a kept guard
        # cannot be serialized into the CompilePackage (non-strict
        # CheckFunctionManager swallows the PackageError and leaves guards_state
        # None). Input-derived identity guards (ID_MATCH and friends) survive
        # _keep_serializable_capture_guards, so this is their failure surface.
        from torch._dynamo.convert_frame import GUARDS_STATE_NONE_MESSAGE

        if GUARDS_STATE_NONE_MESSAGE not in str(e):
            raise
        raise PrecompileError(
            "precompile tracer='dynamo' encountered an identity guard that Dynamo "
            "cannot serialize yet (for example a module or callable guard)."
        ) from e


def _make_dynamo_capture_optimizer(
    capture_target: Callable[..., object],
    package: Any,
    backend: str,
    training: bool,
    capture_limit: int,
    dynamic: bool | None,
) -> tuple[Callable[..., object], Callable[..., object]]:
    compile_graph = _dynamo_backend_compiler(backend, training)
    compiled = torch._dynamo.optimize(
        backend=compile_graph,
        nopython=True,
        guard_filter_fn=_keep_serializable_capture_guards,
        package=package,
        dynamic=dynamic,
        recompile_limit=capture_limit,
        isolate_recompiles=True,
    )(capture_target)
    if compiled is capture_target:
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

    ``dropped_guards`` lists the (guard type, source) pairs guard minimization
    removed from at least one variant's dispatch (deduplicated across
    variants; a pair may remain checked elsewhere). They only covered the
    Python environment, which the programming model declares invariant
    between capture and serving.
    """

    calls: int
    examples: int
    variants: int
    graphs: int
    dynamic_graphs: int
    dropped_guards: tuple[tuple[str, str], ...]


def _build_dynamo_artifact(
    package: Any,
    capture_target: Callable[..., object],
    example_inputs: Sequence[tuple[object, ...]],
    *,
    backend: str,
    training: bool,
    keep_capture: bool = False,
) -> tuple[str, bytes, PrecompileStateSummary]:
    """Render the package's accumulated capture as (python_code, cache) bytes.

    A pure read of the live package (plus, for training, a snapshot compose of
    the backward variants recorded so far when keep_capture is set), so a
    stateful capture can rebuild after every example call.
    """
    for compiled_backend in package.cached_backends.values():
        if isinstance(compiled_backend, _DynamoPythonBackend):
            compiled_backend.finalize_training(keep_capture=keep_capture)

    cache_entry = package.cache_entry()
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
    filtered_guard_states, dropped_guards = _filter_dynamo_guards(
        capture_target, code.guarded_codes, example_inputs
    )

    compiled_backends = []
    for backend_id in code.backend_ids:
        compiled_backend = package.cached_backends.get(backend_id)
        if not isinstance(compiled_backend, _DynamoPythonBackend):
            raise PrecompileError(
                "precompile tracer='dynamo' encountered a graph that could not be "
                "represented as standalone Python source."
            )
        compiled_backends.append(compiled_backend)

    dynamo_state: dict[str, Any] = {
        "code": code.python_code,
        "import_sources": dict(code.import_sources),
        "defaults": capture_target.__defaults__,
        "kwdefaults": capture_target.__kwdefaults__,
        # Newest-first: the driver serves the first variant whose guards pass,
        # and live Dynamo checks recompilations LRU-front-first -- an input
        # matching both an early static variant and a later dynamic one (the
        # automatic-dynamic revisit pattern) must serve the later one, whose
        # backend is the one that observed any training tangent masks.
        # guarded_codes is chronological (CompilePackage appends).
        "variants": [
            {
                "guards_state": guards_state,
                "dynamo_code": guarded.dynamo_code,
            }
            for guarded, guards_state in zip(code.guarded_codes, filtered_guard_states)
        ][::-1],
    }
    backend_ids = [str(backend_id) for backend_id in code.backend_ids]
    python_code = _build_dynamo_python_source(
        backend=backend,
        training=training,
        state=dynamo_state,
        backend_ids=backend_ids,
        compiled_backends=compiled_backends,
        dropped_guards=dropped_guards,
    )
    cache = _dynamo_cache_bytes(
        python_code,
        backend,
        [compiled.cache for compiled in compiled_backends],
    )
    summary = PrecompileStateSummary(
        calls=0,
        examples=len(example_inputs),
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
    from torch._dynamo.eval_frame import cached_backends
    from torch._dynamo.utils import guard_failures

    try:
        if package is not None:
            package.uninstall()
    finally:
        try:
            torch._dynamo.reset_code(capture_target.__code__)
        finally:
            pgo_state.clear()
            # Recompile logging strong-keys the capture code object in this
            # module-level registry, transitively pinning the whole session
            # (including the copied fn globals); pop it so the session can be
            # garbage collected without a torch._dynamo.reset().
            guard_failures.pop(capture_target.__code__, None)
            if backend_fn is not None:
                cached_backends.pop(id(backend_fn), None)


def _freeze_tensor_metadata(tensor: torch.Tensor) -> torch.Tensor:
    # A metadata-frozen alias: shares storage (no data copy; guards check
    # metadata, not data) but owns its sizes/strides/requires_grad, so a later
    # in-place METADATA mutation of the input (resize_, transpose_,
    # requires_grad_) cannot invalidate the recorded example. Built under
    # no_grad so the alias is a leaf that keeps no autograd graph alive.
    with torch.no_grad():
        frozen = tensor.as_strided(
            tensor.size(), tensor.stride(), tensor.storage_offset()
        )
    frozen.requires_grad = tensor.requires_grad
    return frozen


def _snapshot_example(example: tuple[object, ...]) -> tuple[object, ...]:
    # Guard minimization re-checks every recorded example AFTER it has run
    # (and stateful rebuilds re-check them again on every later call), so a fn
    # or caller that mutates an input would otherwise fail the re-check -- or
    # permanently poison a stateful state -- for a computation plain
    # torch.compile supports. Record a pre-execution copy instead. Tensor
    # leaves become metadata-frozen storage aliases (copying data would double
    # example memory; the shared memo preserves identity and aliasing
    # relations); a tensor that cannot be re-aliased, or an example deepcopy
    # cannot handle, is recorded live, which only matters if it is then
    # mutated.
    import copy

    memo: dict[int, object] = {}
    for leaf in pytree.tree_leaves(example):
        if isinstance(leaf, torch.Tensor) and id(leaf) not in memo:
            try:
                memo[id(leaf)] = _freeze_tensor_metadata(leaf)
            except Exception:
                memo[id(leaf)] = leaf
    try:
        return copy.deepcopy(example, memo)
    except Exception:
        return example


def _precompile_dynamo(
    fn: Callable[..., object],
    example_inputs: Sequence[tuple[object, ...]],
    *,
    backend: str,
    decompositions: dict | None,
    training: bool,
    recompile_limit: int | None = None,
    dynamic: bool | None = None,
) -> tuple[str, bytes]:
    from torch._dynamo.package import CompilePackage
    from torch._dynamo.pgo import _new_code_state

    target, _environment_scan = _validate_dynamo_capture(
        fn, example_inputs, decompositions
    )
    capture_target = _make_dynamo_capture_target(target)
    capture_limit = (
        recompile_limit
        if recompile_limit is not None
        else max(torch._dynamo.config.recompile_limit, len(example_inputs) + 1)
    )
    with _DYNAMO_COMPILE_LOCK:
        package = None
        backend_fn = None
        pgo_state = _new_code_state()
        try:
            with (
                _dynamo_capture_context(pgo_state, training, capture_limit),
                _translate_dynamo_capture_errors(capture_limit),
            ):
                package = CompilePackage(capture_target)
                compiled, backend_fn = _make_dynamo_capture_optimizer(
                    capture_target, package, backend, training, capture_limit, dynamic
                )
                recorded = []
                for example in example_inputs:
                    recorded.append(_snapshot_example(example))
                    compiled(*example)
                python_code, cache, _summary = _build_dynamo_artifact(
                    package,
                    capture_target,
                    recorded,
                    backend=backend,
                    training=training,
                )
                return python_code, cache
        finally:
            _teardown_dynamo_capture(package, capture_target, pgo_state, backend_fn)


def _warn_unclosed_dynamo_state(fn_name: str) -> None:
    log.warning(
        "A stateful torch.compiler.precompile state for %r was garbage collected "
        "without close(); its capture session stays pinned by Dynamo's "
        "process-global registries until torch._dynamo.reset().",
        fn_name,
    )


class _PrecompileDynamoState:
    """Opaque accumulated capture state for ``torch.compiler.precompile.stateful``.

    Returned by (and passed back to) ``precompile.stateful`` calls. It owns
    the live Dynamo capture session:
    the cloned capture function with its installed code caches, the
    CompilePackage new variants accumulate into, the optimize wrapper (whose
    isolate-recompiles bucket keeps earlier variants visible to later calls),
    the isolated PGO state that drives automatic dynamic shapes across calls,
    and a pre-execution snapshot of every example tuple seen so far (guard
    minimization re-checks all of them on every rebuild, so their tensors stay
    alive for the state's lifetime; see _snapshot_example). Treat it as
    opaque: it is process-local and not serializable.
    Call :meth:`close` when done capturing -- Dynamo's recompile-logging
    registry otherwise pins the session until ``torch._dynamo.reset()``. Do
    not call ``torch._dynamo.reset()`` between calls that share a state: later
    calls may raise, or silently duplicate variants in the rewritten artifact.
    """

    def __init__(
        self,
        *,
        target: Callable[..., object],
        capture_target: Callable[..., object],
        package: Any,
        pgo_state: Any,
        backend: str,
        training: bool,
        capture_limit: int,
        dynamic: bool | None,
        environment_scan: dict[str, tuple[frozenset[int], bool]],
    ) -> None:
        self.target = target
        self.capture_target = capture_target
        self.package = package
        self.pgo_state = pgo_state
        self.backend = backend
        self.training = training
        self.capture_limit = capture_limit
        self.dynamic = dynamic
        # Cached wide walk of the fn's referenced globals (see
        # _validate_dynamo_capture): resumed calls reuse it instead of
        # re-walking whole object graphs per call.
        self.environment_scan = environment_scan
        self.compiled: Callable[..., object] | None = None
        self.backend_fn: Callable[..., object] | None = None
        self.examples: list[tuple[object, ...]] = []
        self.calls = 0
        self.last_summary: PrecompileStateSummary | None = None
        self.closed = False
        # Warn-only: the session outlives the state object (Dynamo's registries
        # pin it by code object, not through this instance), and tearing down
        # global compiler state from a GC callback is not safe. atexit=False
        # keeps a state that is simply alive at interpreter exit quiet.
        self._finalizer = weakref.finalize(
            self, _warn_unclosed_dynamo_state, getattr(target, "__name__", "<fn>")
        )
        self._finalizer.atexit = False

    def summary(self) -> PrecompileStateSummary | None:
        """Coverage of the most recently written artifact; None before one exists."""
        return self.last_summary

    def close(self) -> None:
        """Release the capture session (installed code caches and registries).

        Idempotent. A closed state cannot be resumed; artifact files written by
        earlier calls remain valid. Without close(), the session (including its
        copy of the fn's globals) stays pinned by Dynamo's process-global
        recompile-logging registry until ``torch._dynamo.reset()``.
        """
        with _DYNAMO_COMPILE_LOCK:
            if self.closed:
                return
            self.closed = True
            self._finalizer.detach()
            self.compiled = None
            self.examples.clear()
            _teardown_dynamo_capture(
                self.package, self.capture_target, self.pgo_state, self.backend_fn
            )

    def __repr__(self) -> str:
        status = ", closed" if self.closed else ""
        return (
            f"<torch.compiler.precompile dynamo state: {len(self.examples)} "
            f"example call(s), backend={self.backend!r}, "
            f"training={self.training}{status}>"
        )


def _write_dynamo_artifact_files(
    python_code: str, cache: bytes, artifact_path: str, cache_path: str
) -> None:
    # Two-phase: write and fsync both temp files first, then rename back to
    # back, so the window in which a crash leaves a mismatched
    # (code_hash-rejected) pair on disk is two renames, not a full cache write.
    # A failure before the first rename cleans its temp files up and leaves the
    # previous pair; a failure between the renames leaves the NEW artifact with
    # the OLD cache, which load() degrades on (code_hash mismatch -> cold cache
    # with a warning) and the next successful rewrite repairs.
    import os

    renames = []
    try:
        for path, data, mode, encoding in (
            (artifact_path, python_code, "w", "utf-8"),
            (cache_path, cache, "wb", None),
        ):
            parent = os.path.dirname(os.fspath(path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp"
            renames.append((tmp, path))
            with open(tmp, mode, encoding=encoding) as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        for tmp, path in renames:
            os.replace(tmp, path)
    except BaseException:
        # A tmp already renamed away just fails its unlink with ENOENT.
        for tmp, _ in renames:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise


def _precompile_dynamo_stateful(
    fn: Callable[..., object],
    example_inputs: Sequence[tuple[object, ...]],
    *,
    backend: str,
    decompositions: dict | None,
    training: bool,
    recompile_limit: int | None,
    dynamic: bool | None,
    state: _PrecompileDynamoState | None,
    artifact_path: str,
    cache_path: str,
) -> tuple[list[object], _PrecompileDynamoState]:
    from torch._dynamo.package import CompilePackage
    from torch._dynamo.pgo import _new_code_state

    if state is not None and not isinstance(state, _PrecompileDynamoState):
        raise TypeError(
            "precompile state must be the state returned by a previous "
            f"stateful precompile call, got {type(state).__name__}."
        )
    target, environment_scan = _validate_dynamo_capture(
        fn,
        example_inputs,
        decompositions,
        environment_scan=None if state is None else state.environment_scan,
    )
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
            capture_target = _make_dynamo_capture_target(target)
            state = _PrecompileDynamoState(
                target=target,
                capture_target=capture_target,
                package=CompilePackage(capture_target),
                pgo_state=_new_code_state(),
                backend=backend,
                training=training,
                capture_limit=capture_limit,
                dynamic=dynamic,
                environment_scan=environment_scan,
            )
        else:
            if state.closed:
                raise ValueError(
                    "precompile cannot resume a closed state; start fresh with "
                    "state=None."
                )
            mismatches = [
                f"{name}={got!r} (the state was created with {want!r})"
                for name, got, want in (
                    ("backend", backend, state.backend),
                    ("training", training, state.training),
                )
                if got != want
            ]
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
            with (
                _dynamo_capture_context(state.pgo_state, training, state.capture_limit),
                # A FRESH call that hits the limit self-closes and never
                # returns its state, so the close()-and-recapture advice only
                # fits resumed calls.
                _translate_dynamo_capture_errors(
                    state.capture_limit, stateful=not fresh
                ),
            ):
                if state.compiled is None:
                    state.compiled, state.backend_fn = _make_dynamo_capture_optimizer(
                        state.capture_target,
                        state.package,
                        backend,
                        training,
                        state.capture_limit,
                        state.dynamic,
                    )
                import inspect

                # An unbindable example (a caller arity mistake) must never be
                # recorded: guard minimization signature.bind()s every recorded
                # example on every rebuild, so it would poison the state. Probe
                # every example of the call up front, so a bad one raises
                # before any example of the batch is recorded or run.
                signature = inspect.signature(state.capture_target)
                for example in example_inputs:
                    try:
                        signature.bind(*example)
                    except TypeError as e:
                        raise TypeError(
                            f"precompile example does not match the positional "
                            f"signature of {state.target.__name__!r}: {e}. No "
                            "example from this call was recorded."
                        ) from e
                results = []
                for example in example_inputs:
                    # Record BEFORE running: Dynamo installs a new guarded
                    # variant at frame entry, so a step that raises after
                    # compiling would otherwise leave a variant matching no
                    # recorded example and every later rebuild would fail.
                    # Record a pre-execution snapshot (see _snapshot_example);
                    # the step itself runs on the caller's live objects.
                    state.examples.append(_snapshot_example(example))
                    results.append(state.compiled(*example))
                try:
                    python_code, cache, summary = _build_dynamo_artifact(
                        state.package,
                        state.capture_target,
                        state.examples,
                        backend=backend,
                        training=training,
                        keep_capture=True,
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
            if fresh:
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
        tracer: str = "make_fx",
        decompositions: dict | None = None,
    ) -> None:
        # ``fn`` is the whole computation: an nn.Module, or a callable that closes
        # over the module(s) it uses (e.g. ``lambda x: model(x)``, or a training
        # step that computes a loss and torch.autograd.grad).
        self._fn = fn
        self._backend = backend
        self._tracer = tracer
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
        obj._loaded_forward = forward
        return obj

    def _compile(self, example_inputs: Sequence[tuple[object, ...]]) -> None:
        # The Dynamo path is handled directly by _PrecompileApi; this holder implements
        # only the make_fx calling convention.
        if self._tracer != "make_fx":
            raise NotImplementedError(
                f"precompile tracer={self._tracer!r} is not implemented yet; use "
                "tracer='make_fx' (the default)."
            )
        if len(example_inputs) != 1:
            raise ValueError(
                "precompile with tracer='make_fx' requires exactly one example "
                "input tuple."
            )
        args = example_inputs[0]
        if self._backend == "eager" and _has_unbacked_marks(args):
            raise NotImplementedError(
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

    def __call__(self, *args: object) -> object:
        # A PrecompiledModule is runnable only after load(); precompile() itself
        # returns (python_code, cache) rather than a runnable.
        if self._loaded_forward is None:
            raise PrecompileError(
                "this object is not runnable; build one with "
                "torch.compiler.precompile.load(python_code, cache)."
            )
        return self._loaded_forward(*args)

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


class _PrecompileApi:
    """Callable namespace implementing ``torch.compiler.precompile`` and ``.load``.

    A single instance is exposed as ``torch.compiler.precompile``; calling it precompiles a
    computation and ``torch.compiler.precompile.load`` reloads the resulting artifacts. It
    is a class (rather than a function with attached attributes) so the call, the
    loader, and the error type are explicit members.

    The contract for both ``__call__`` and ``load`` is Note [precompile programming
    model] in this module.
    """

    # Reported so test_public_bindings / introspection see this as ``torch.compiler``.
    __module__ = "torch.compiler"

    # The error type raised by precompile, reachable as
    # ``torch.compiler.precompile.PrecompileError``.
    PrecompileError = PrecompileError

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
        training: bool = False,
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
        precompile itself, and the capture temporarily patches process-global Dynamo
        and functorch config (e.g. ``fail_on_recompile_limit_hit``), so an unrelated
        ``torch.compile`` on another thread during a capture can observe those patched
        values. The make_fx capture phase and its ``backend="eager"`` path are NOT
        serialized.

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
          artifact retains guards derived from explicit inputs. Guards covering the
          Python environment may be removed because that environment is required to be
          unchanged between capture and runtime. This assumption is unchecked, so
          changing the environment can silently run code specialized for its capture-time
          state. A call that fails every retained input guard set raises instead of
          compiling at runtime. This initial path requires one full graph (graph breaks
          are rejected), a function without closure cells, and positional tensor/scalar
          arguments or containers of those values (``nn.Module`` arguments are not
          supported yet). A global whose object graph contains a tensor is rejected
          conservatively (every tensor must be an explicit input), as are functions
          that mutate globals, pytree-leaf inputs also reachable through the Python
          environment (checked once at state creation for stateful capture;
          dtypes, layouts, and memory formats are exempt as value-guarded
          singletons, and enum members because a used enum argument fails
          loudly on its unserializable identity guard), distinct tensor inputs
          that share or overlap
          storage (the same tensor object may repeat; the loaded artifact also
          raises on overlapping runtime inputs), and non-strided input layouts
          other than sparse (which surfaces Dynamo's own rejection).

        ``decompositions`` is an optional decomposition table (a dict mapping each
        ``OpOverload`` to a decomposition function) forwarded to ``make_fx`` as its
        ``decomposition_table`` during capture, so you can control how ATen ops are
        broken down in the captured graph. Defaults to ``None`` (make_fx's default) and
        is not yet supported with ``tracer="dynamo"``.

        ``training=True`` is supported with ``tracer="dynamo"`` and
        ``backend="inductor"``. Capture runs with grad enabled, and each compiled graph
        carries a readable AOTAutograd forward and backward bridged by an emitted
        ``torch.autograd.Function``. A served output therefore retains its ``grad_fn``;
        calling ``backward()`` executes the precompiled backward kernels. The input
        tensors that require gradients must do so in every example and at runtime.
        Each example's actual backward records its output-tangent presence pattern; an
        unseen pattern raises instead of compiling during serving. If the examples only
        run forwards, only the all-tangents-present backward is covered.

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
        parameter/buffer structure (invariant 2). Arguments are matched POSITIONALLY:
        pass the model(s) and inputs positionally both here and at load time; keyword-
        argument calling conventions are not supported (a keyword call raises
        ``TypeError``; a wrong positional arity raises ``PrecompileError``, on both
        tracers). With ``tracer="make_fx"``, if ``fn`` ran a
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
        the example's (invariant 6).
        """
        torch._C._log_api_usage_once("torch.compiler.precompile")
        if example_inputs is None:
            example_inputs = [tuple(example_args)]
        elif example_args:
            raise TypeError(
                "precompile cannot take both positional examples and example_inputs."
            )
        if len(example_inputs) == 0:
            raise ValueError("precompile requires at least one example input tuple.")
        for example in example_inputs:
            if not isinstance(example, tuple):
                raise TypeError(
                    "precompile example_inputs must be a sequence of positional-"
                    f"argument tuples, got {type(example).__name__}."
                )
        if backend not in ("inductor", "eager"):
            raise ValueError(
                f"precompile backend must be 'inductor' or 'eager', got {backend!r}."
            )
        if tracer not in ("make_fx", "dynamo"):
            raise ValueError(
                f"precompile tracer must be 'make_fx' or 'dynamo', got {tracer!r}."
            )
        if training and (tracer != "dynamo" or backend != "inductor"):
            raise NotImplementedError(
                "precompile training=True currently requires tracer='dynamo' and "
                "backend='inductor'."
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
                training=training,
                recompile_limit=recompile_limit,
                dynamic=dynamic,
            )
        compiled = PrecompiledModule(
            fn, backend=backend, tracer=tracer, decompositions=decompositions
        )
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
        state: _PrecompileDynamoState | None = None,
        backend: str = "inductor",
        training: bool = False,
        recompile_limit: int | None = None,
        dynamic: bool | None = None,
    ) -> tuple[list[object], _PrecompileDynamoState]:
        """Capture ``fn`` incrementally from a loop the caller owns (Dynamo tracer).

        Some capture regions can only be exercised from the caller's own loop
        (e.g. a pipelined train step whose batch deque only advances between
        iterations), so ``precompile`` cannot drive the examples itself. Each
        ``stateful`` call runs its example tuples for real, records whatever
        guarded variants they newly exercise into the returned opaque
        ``state``, REWRITES the artifact and cache files at
        ``artifact_path``/``cache_path`` atomically, and returns
        ``(results, state)``: ``results`` is always a list with one entry per
        example tuple of THIS call (never unwrapped, so a fn that itself
        returns a list is unambiguous). The capture semantics -- tracer,
        rejections, guard minimization, the programming-model contract -- are
        exactly ``precompile(..., tracer="dynamo")``'s; only the delivery
        differs. API note: this began as a mode of ``precompile`` itself
        (keyed on passing the paths), and was split into this sibling entry
        point after review -- both modes returned 2-tuples, so calling the
        wrong mode failed silently at unpack time and the return annotation
        was an unhelpful union::

            state = None
            try:
                for batch in batches:
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

        ``state=None`` starts fresh; passing a returned state resumes -- with
        the same ``fn``, ``backend``, ``training``, ``recompile_limit``, and
        ``dynamic``, else the call raises rather than produce a mixed
        artifact. The files on disk are always a loadable artifact for
        everything captured so far. A call whose guards all hit adds nothing;
        guard minimization is re-run over every example seen so far on each
        rebuild, so the state keeps a pre-execution snapshot of every example
        tuple alive (tensor data by reference, tensor metadata frozen; a step
        may freely mutate its container inputs and its tensor inputs in place,
        except an exotic input ``deepcopy`` cannot copy, which is recorded
        live), and later calls see earlier variants because the
        state carries one isolate-recompiles bucket and one PGO record across
        calls (a dimension that varies between calls recompiles into a
        symbolic graph exactly as it would within one call).
        ``recompile_limit`` caps the variants per capture and defaults to
        ``max(torch._dynamo.config.recompile_limit, 256)`` because
        accumulating captures outgrow the config default. ``dynamic`` is
        forwarded to Dynamo (``None`` keeps the automatic policy). After each
        rewrite, ``state.summary()`` reports what the artifact carries --
        calls, examples, variants, graphs, dynamic graphs, and the environment
        guards minimization dropped from at least one variant (also embedded
        in the artifact as ``_DROPPED_GUARDS``). Load the on-disk pair
        directly with ``precompile.load(artifact_path=..., cache_path=...)``.
        Rewriting is proportional to everything captured so far, not to the
        call, so a long loop over a large capture pays it every time; feed
        stateful the calls that add variants rather than all of them. The
        state is process-local and not serializable; call ``state.close()``
        when done capturing to release the session, and do not call
        ``torch._dynamo.reset()`` between calls that share a state (later
        calls may raise or duplicate variants).
        """
        torch._C._log_api_usage_once("torch.compiler.precompile.stateful")
        if len(example_inputs) == 0:
            raise ValueError(
                "precompile.stateful requires at least one example input tuple."
            )
        for example in example_inputs:
            if not isinstance(example, tuple):
                raise TypeError(
                    "precompile.stateful example_inputs must be a sequence of "
                    f"positional-argument tuples, got {type(example).__name__}."
                )
        if backend not in ("inductor", "eager"):
            raise ValueError(
                f"precompile backend must be 'inductor' or 'eager', got {backend!r}."
            )
        if training and backend != "inductor":
            raise NotImplementedError(
                "precompile training=True currently requires backend='inductor'."
            )
        return _precompile_dynamo_stateful(
            fn,
            example_inputs,
            backend=backend,
            decompositions=None,
            training=training,
            recompile_limit=recompile_limit,
            dynamic=dynamic,
            state=state,
            artifact_path=artifact_path,
            cache_path=cache_path,
        )

    def load(
        self,
        python_code: str | None = None,
        cache: bytes | None = None,
        *,
        artifact_path: str | None = None,
        cache_path: str | None = None,
    ) -> Callable[..., object]:
        """Reconstruct a runnable from ``(python_code, cache)`` from precompile.

        Pass either the in-memory pair or -- the natural companion of stateful
        capture's on-disk rewrites -- ``artifact_path``/``cache_path`` to read
        the pair from the files a capture wrote.

        The driver runs from ``python_code`` -- the single source of truth for the whole
        calling convention. ``load`` reads the cache's ``BACKEND`` (to check the pairing)
        and, for the inductor backend, primes the inductor kernel caches from its
        ``save_cache_artifacts`` bundle (via ``torch.compiler.load_cache_artifacts``) so a
        warm reload loads precompiled kernels instead of JIT-compiling; then it exec's
        ``python_code``. With no usable cache it degrades to JIT'ing from ``python_code``.

        Call the result with the SAME argument structure ``fn`` took -- the
        model(s) in their original positions plus the runtime inputs. Per invariant
        2 of Note [precompile programming model], the runtime model must match the
        example model's parameter/buffer structure; precompile re-derives the
        param/buffer list from it (same interning/order as capture).

        Raises ``PrecompileError`` if ``python_code`` is malformed or is not a
        ``torch.compiler.precompile`` artifact (it fails to parse, or is missing the
        calling-convention metadata), or -- for ``make_fx`` artifacts -- if the
        cache's ``backend`` tag or ``code_hash`` does not match ``python_code``,
        i.e. the cache and python_code came from different ``precompile()`` calls.
        For ``dynamo`` artifacts a ``backend``/``code_hash`` mismatch instead
        degrades to a cold cache with a warning: the python_code is fully
        self-contained, and stateful capture's two-rename rewrite can
        legitimately leave a mismatched pair after a crash. A MISSING cache
        file on the path form (the first-rewrite crash window) degrades with a
        warning for BOTH tracers -- it is checked before the artifact's tracer
        is known, and an absent cache is no cache, not a wrong pairing.
        A cache whose ``format``/``version`` does not match (a
        foreign or different-build envelope) is NOT fatal: the cache is acceleration
        only, so ``load`` degrades to JIT'ing from ``python_code`` rather than crashing.
        """
        if (artifact_path is None) != (cache_path is None):
            raise ValueError(
                "precompile.load requires both artifact_path and cache_path."
            )
        if artifact_path is not None and cache_path is not None:
            if python_code is not None or cache is not None:
                raise TypeError(
                    "precompile.load takes either (python_code, cache) or "
                    "(artifact_path, cache_path), not both."
                )
            with open(artifact_path, encoding="utf-8") as f:
                python_code = f.read()
            try:
                with open(cache_path, "rb") as f:
                    cache = f.read()
            except FileNotFoundError:
                # A crash between the two renames of a stateful capture's
                # FIRST rewrite leaves an artifact with no cache file; the
                # cache is acceleration only, so degrade instead of raising.
                log.warning(
                    "torch.compiler.precompile.load found no cache file at %r "
                    "(likely a first rewrite interrupted between the artifact "
                    "and cache renames). Falling back to JIT from python_code.",
                    cache_path,
                )
                cache = b""
        elif not cache and cache is not None:
            # The in-memory form must not degrade silently: an empty cache is
            # a truncated or misread file, and the pre-envelope-check code
            # warned here too (torch.load raised into the fallback warning).
            log.warning(
                "torch.compiler.precompile.load got an empty cache; falling "
                "back to JIT from python_code."
            )
        if python_code is None or cache is None:
            raise TypeError(
                "precompile.load requires python_code and cache (or artifact_path "
                "and cache_path)."
            )
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
        # cache) pairing -- so it hard-fails rather than running under foreign metadata.
        artifact = None
        try:
            blob = torch.load(io.BytesIO(cache), weights_only=True) if cache else None
            if blob is not None and (
                blob.get("format") != _CACHE_FORMAT
                or blob.get("version") != _CACHE_VERSION
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
                # Reject a cache whose backend or code_hash does not match this
                # python_code (a mismatched pairing); see Note [precompile
                # programming model], invariant 7.
                expected_code_hash = hashlib.sha256(python_code.encode()).hexdigest()
                mismatched = (
                    blob.get("backend") != backend
                    or blob.get("code_hash") != expected_code_hash
                )
                if mismatched and tracer == "dynamo":
                    # A dynamo python_code is fully self-contained (every graph
                    # source is inlined), so the cache is pure acceleration.
                    # Stateful capture rewrites the pair as two back-to-back
                    # renames; a crash between them leaves exactly this mismatch
                    # (a backend mismatch is the same window, over an older
                    # artifact at the same paths), and degrading keeps the
                    # on-disk artifact always loadable.
                    log.warning(
                        "torch.compiler.precompile.load got a cache whose "
                        "backend/code_hash does not match python_code (likely a "
                        "rewrite interrupted between the artifact and cache "
                        "renames, or a mismatched pairing). Falling back to JIT "
                        "from python_code."
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

        return PrecompiledModule._from_loaded(forward, backend=backend)


precompile = _PrecompileApi()
# ``torch.compiler.precompile`` is a callable instance, not a function, so give it the
# name/doc introspection (Sphinx autosummary, help(), IDEs) expects to find on a
# public callable; the rich usage docs live on ``__call__``.
precompile.__name__ = "precompile"  # type: ignore[attr-defined]
precompile.__qualname__ = "precompile"  # type: ignore[attr-defined]
precompile.__doc__ = _PrecompileApi.__call__.__doc__

# These are public under torch.compiler.precompile, so report their module/qualname there
# (mirroring the singleton fixup above) -- otherwise Sphinx autoexception/autofunction
# would anchor them under this private module. load is a bound method; patch the
# underlying function so introspection on precompile.load reports torch.compiler too.
PrecompileError.__module__ = "torch.compiler"
PrecompileError.__qualname__ = "precompile.PrecompileError"
_PrecompileApi.load.__module__ = "torch.compiler"
_PrecompileApi.load.__qualname__ = "precompile.load"
_PrecompileApi.stateful.__module__ = "torch.compiler"
_PrecompileApi.stateful.__qualname__ = "precompile.stateful"
