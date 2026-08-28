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
captures the guarded specializations, recompilations, and graph-break resume frames
Dynamo produces. Closure-free functions wrapped with ``torch._dynamo.disable`` are
embedded for eager execution between compiled segments. The serialized guard records
are filtered under the input-only recompilation contract while preserving how every
example dispatches among the captured variants. The Python environment, including globals
and context-manager state, must remain semantically unchanged at runtime; guards that only
enforce that promise may be omitted, including guards through process-local values that
cannot be reconstructed. By default, every portable input-derived guard is retained,
rebuilt from frozen capture state, and checked for predicate drift. Distinct tensor inputs
must not share or overlap storage, and an explicit input must not also be reachable through
the Python environment. Statically visible identity relations are rejected; dynamic native
indirection that hides one is unsupported. Python functions that mutate globals are
rejected, as is unverified behavior on mutable environment objects. Every artifact raises
when no captured variant matches. Captured nested frames that are reachable only by an
ordinary Python call use an isolated installed mode; loading prepares it, the first call
installs it, and ``unload()`` removes the installation. Compiled graph bodies and kernels
remain Python source. Eager higher-order ops retain opaque FX structure where their runtime
interpreters require a real ``Graph``; guard trees and transformed/disabled bytecode are
also stored as opaque inline data.

With ``tracer="dynamo", training=True``, captured graphs remain differentiable on both
backends. Inductor artifacts contain AOTAutograd's forward and backward as readable
source. The served output retains its ``grad_fn`` and a later ``backward()`` executes
the captured backward, including across captured recompilations and graph breaks. Serving
pins grad mode to ``training`` rather than inheriting the caller's current grad mode.

``precompile`` returns an executable ``python_code`` string plus a companion
integrity-tagged ``cache``. Make-fx artifacts are self-contained. Dynamo artifacts may
import modules referenced by transformed globals, and installed artifacts additionally
import the defining Python modules whose nested code objects they serve. With
``backend="inductor"`` (the default) the captured graph is lowered through the AOT
backend contract (``torch._functorch.aot_autograd.compile_to_python``, AOTAutograd +
Inductor);
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
#    load() rejects a (code, cache) pair from different precompile() calls (same
#    backend) rather than silently running the cache's graph under foreign metadata.
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
# specialization/recompilation exercised by example_inputs, omits guards for the
# caller-promised invariant environment, and retains every portable input-derived
# guard. Graph breaks are reconstructed from their captured resume frames. The eager
# backend preserves higher-order graph bodies as Python plus the opaque FX structure
# their runtime interpreters require; it does not symbolically retrace them at load.

from __future__ import annotations

import contextvars
import dataclasses
import dis
import functools
import gc
import hashlib
import io
import itertools
import logging
import os
import re
import sys
import threading
import types
import weakref
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import CodeType, MappingProxyType
from typing import Any, cast, NewType, TYPE_CHECKING
from typing_extensions import Self

import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch.compiler._precompile_types import (
    _DynamoArtifactState,
    _DynamoCodeState,
    _DynamoDisabledFunction,
    _DynamoGuardedVariant,
    _DynamoInputContract,
    _DynamoInputContractVariant,
    ExampleInput,
    FrameInvariants,
    GuardFact,
    PrecompiledCallable,
    PrecompileSummary,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


log = logging.getLogger(__name__)


if TYPE_CHECKING:
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
    shape_env = None
    initial_runtime_asserts: set[Any] = set()
    if any(marks):
        flat_args, fake_mode = _fakeify_with_unbacked(pb_flat, user_flat, marks)
        shape_env = fake_mode.shape_env
        if shape_env is None:
            raise AssertionError("mark_unbacked capture requires a ShapeEnv")
        initial_runtime_asserts = {
            runtime_assert.expr
            for assertions in shape_env.deferred_runtime_asserts.values()
            for runtime_assert in assertions
        }
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
    if shape_env is not None:
        added_runtime_asserts = sorted(
            {
                str(runtime_assert.expr)
                for assertions in shape_env.deferred_runtime_asserts.values()
                for runtime_assert in assertions
                if runtime_assert.expr not in initial_runtime_asserts
            }
        )
        if added_runtime_asserts:
            raise PrecompileError(
                "precompile: fn introduced deferred runtime shape constraints that the "
                "standalone driver cannot enforce yet: "
                f"{added_runtime_asserts}. Rewrite fn to avoid the deferred constraint. "
                "For equality constraints between input dimensions, give those "
                "dimensions a shared shape_id or capture them static."
            )
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
    from torch._dynamo.graph_utils import _graph_device_types

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
    graph_devices = (
        ()
        if compiled._gm is None
        else tuple(sorted(_graph_device_types(compiled._gm.graph)))
    )
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
        f"GRAPH_DEVICES = {graph_devices!r}",
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
    wanted = {
        "BACKEND",
        "TRACER",
        "TRAINING",
        "_DYNAMO_BACKENDS",
        "_DYNAMO_BACKEND_IDS",
        "_DYNAMO_PYTHON_VERSION",
        "_DYNAMO_STATE",
        "_DYNAMO_TORCH_VERSION",
        "SERVING_MODE",
        "UNREACHABLE_WITHOUT_INSTALL",
        "CAPTURE_COMPLETE",
        "DROPPED_GUARDS",
        "RISKY_DROPPED_GUARDS",
        "POLICY_DROPPED_GUARDS",
        "WONT_GENERALIZE",
        *make_fx_metadata,
    }
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
            except (SyntaxError, ValueError) as e:
                raise PrecompileError(
                    f"python_code has invalid calling-convention metadata "
                    f"{target.id!r}."
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
    required = {"BACKEND", "TRACER"}
    if tracer == "make_fx":
        required = {"BACKEND", *make_fx_metadata}
    else:
        required.update(
            {
                "TRAINING",
                "_DYNAMO_BACKENDS",
                "_DYNAMO_BACKEND_IDS",
                "_DYNAMO_PYTHON_VERSION",
                "_DYNAMO_STATE",
                "_DYNAMO_TORCH_VERSION",
            }
        )
    missing = required - found.keys()
    if missing:
        raise PrecompileError(
            f"python_code is missing calling-convention metadata {sorted(missing)}; "
            "it does not look like a torch.compiler.precompile artifact."
        )
    if tracer == "dynamo":
        if type(found["TRAINING"]) is not bool:
            raise PrecompileError(
                "python_code has invalid calling-convention metadata 'TRAINING'."
            )
        backend_ids = found["_DYNAMO_BACKEND_IDS"]
        if not (
            isinstance(backend_ids, tuple)
            and all(isinstance(backend_id, str) for backend_id in backend_ids)
        ):
            raise PrecompileError(
                "python_code has invalid calling-convention metadata "
                "'_DYNAMO_BACKEND_IDS'."
            )
        if found["_DYNAMO_BACKENDS"] != {}:
            raise PrecompileError(
                "python_code has invalid calling-convention metadata "
                "'_DYNAMO_BACKENDS'."
            )
        python_version = found["_DYNAMO_PYTHON_VERSION"]
        if not (
            isinstance(python_version, tuple)
            and len(python_version) == 2
            and all(type(part) is int for part in python_version)
        ):
            raise PrecompileError(
                "python_code has invalid calling-convention metadata "
                "'_DYNAMO_PYTHON_VERSION'."
            )
        if not isinstance(found["_DYNAMO_TORCH_VERSION"], str):
            raise PrecompileError(
                "python_code has invalid calling-convention metadata "
                "'_DYNAMO_TORCH_VERSION'."
            )
        if not isinstance(found["_DYNAMO_STATE"], str):
            raise PrecompileError(
                "python_code has invalid calling-convention metadata '_DYNAMO_STATE'."
            )
    return found


def _parse_dynamo_state(python_code: str) -> _DynamoArtifactState:
    """Read the serialized Dynamo state from an artifact without executing it."""
    import ast
    import base64
    import pickle

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
        if isinstance(target, ast.Name) and target.id == "_DYNAMO_STATE":
            try:
                encoded_state = ast.literal_eval(node.value)
                if not isinstance(encoded_state, str):
                    raise TypeError("_DYNAMO_STATE must be a string")
                state = pickle.loads(base64.b64decode(encoded_state, validate=True))
            except Exception as e:
                raise PrecompileError(
                    "python_code contains invalid serialized Dynamo state."
                ) from e
            if not isinstance(state, _DynamoArtifactState):
                raise PrecompileError(
                    "python_code contains invalid serialized Dynamo state."
                )
            from torch import _precompile_driver as driver

            return driver._validate_dynamo_artifact_state(state)
    raise PrecompileError("python_code is missing serialized Dynamo state.")


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
    importing the module from the artifact) keeps the driver source version-frozen
    (Note [precompile programming model], invariant 7)."""
    import inspect

    from torch import _precompile_driver as driver

    forward_fn = getattr(driver, forward_fn_name)
    blocks = [
        inspect.getsource(driver._extract_param_buffers),
        inspect.getsource(driver._fail),
        inspect.getsource(driver._check_structure),
        inspect.getsource(driver._autocast_off),
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

    def finalize_training(self) -> None:
        if self._compile_state is None:
            return
        globals_dict = getattr(self._call, "__globals__", {})
        masks = globals_dict.get("_AOT_OBSERVED_UNDEFINED_TANGENT_MASKS", ())
        try:
            self.python_code, self.cache = self._compile_state.finalize(tuple(masks))
        finally:
            globals_dict["_AOT_BACKWARD_VARIANT_COMPILER"] = None
            variants = globals_dict.get("_AOT_BACKWARD_VARIANTS")
            if isinstance(variants, dict):
                variants.clear()
            self._compile_state = None


def _build_dynamo_eager_graph_source(gm: torch.fx.GraphModule) -> str:
    """Render one Dynamo FX graph as standalone eager Python source."""
    import base64
    import pickle

    from torch.fx._graph_pickler import GraphPickler, Options
    from torch.fx._lazy_graph_module import _LazyGraphModule
    from torch.fx.graph_module import _format_import_block
    from torch.package import sys_importer

    def recompile(module: torch.fx.GraphModule) -> Any:
        return (
            module._real_recompile()
            if isinstance(module, _LazyGraphModule)
            else module.recompile()
        )

    options = Options(
        ops_filter=None,
        node_metadata_key_filter=lambda key: key
        not in (
            "example_value",
            "tensor_dict",
            "source_fn_stack",
            "nn_module_stack",
            "fwd_source_fn_stack",
        ),
    )
    python_code = recompile(gm)
    readable_subgraphs: list[tuple[str, str, str]] = []

    def collect_subgraphs(module: torch.fx.GraphModule, prefix: str = "") -> None:
        for name, child in module._modules.items():
            if not isinstance(child, torch.fx.GraphModule):
                continue
            path = f"{prefix}.{name}" if prefix else name
            child_code = recompile(child)
            readable_subgraphs.append(
                (
                    path,
                    _format_import_block(child_code.globals, sys_importer),
                    child.code,
                )
            )
            collect_subgraphs(child, path)

    collect_subgraphs(gm)
    body = gm.__dict__.copy()
    body.pop("_graph", None)
    marker = "torch.compiler.precompile.eager_subgraph"
    body["_modules"] = {
        name: (marker, GraphPickler.dumps(module, options))
        if isinstance(module, torch.fx.GraphModule)
        else module
        for name, module in body.get("_modules", {}).items()
    }
    encoded_body = base64.b64encode(pickle.dumps(body)).decode("ascii")
    import_block = _format_import_block(python_code.globals, sys_importer)

    from torch.fx.graph import _custom_builtins

    parts = [
        "# Dynamo captured graph (eager backend).",
        "import base64",
        "import pickle",
        "import torch",
        "from torch._subclasses import FakeTensorMode",
        "from torch.fx._graph_pickler import GraphPickler",
        "from torch.fx.experimental.symbolic_shapes import ShapeEnv",
        *(_cb.import_str for _cb in _custom_builtins.values()),
        import_block,
    ]
    for index, (path, imports, source) in enumerate(readable_subgraphs):
        parts.extend(
            [
                f"# Nested FX graph {path!r}; executable structure is restored below.",
                imports,
                source.replace("def forward(", f"def _eager_subgraph_{index}(", 1),
                "",
            ]
        )
    parts.extend(
        [
            gm.code.replace("def forward(", "def _graph_forward(", 1),
            "",
            "# Opaque FX state: eager HOPs require real Graph objects at runtime.",
            f"_EAGER_GRAPH_BODY = {encoded_body!r}",
            f"_EAGER_SUBGRAPH_MARKER = {marker!r}",
            "",
            "",
            "def _load_eager_subgraph(value):",
            "    if not (isinstance(value, tuple) and len(value) == 2 and",
            "            value[0] == _EAGER_SUBGRAPH_MARKER):",
            "        return value",
            "    graph = GraphPickler.loads(",
            "        value[1], FakeTensorMode(shape_env=ShapeEnv())",
            "    )",
            "    graph.recompile()",
            "    return graph",
            "",
            "",
            "class _GraphSelf(torch.nn.Module):",
            "    def __init__(self):",
            "        super().__init__()",
            "        body = pickle.loads(base64.b64decode(_EAGER_GRAPH_BODY))",
            "        body['_modules'] = {",
            "            name: _load_eager_subgraph(value)",
            "            for name, value in body.get('_modules', {}).items()",
            "        }",
            "        self.__dict__.update(body)",
            "",
            "",
            "_GRAPH_SELF = _GraphSelf()",
            "",
            "",
            "def call(args):",
            "    return _graph_forward(_GRAPH_SELF, *args)",
            "",
        ]
    )
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


_DYNAMO_UNMODELLED_GUARD_TYPES = frozenset(
    {
        "DETERMINISTIC_ALGORITHMS",
        "DISPATCH_KEY_SET_MATCH",
        "DTENSOR_SPEC_MATCH",
        "FSDP_TRAINING_STATE",
        "GLOBAL_STATE",
        "OPAQUE_OBJ_GUARD_FN_MATCH",
        "SHAPE_ENV",
        "TENSOR_SUBCLASS_METADATA_MATCH",
        "TORCH_FUNCTION_STATE",
    }
)
_DYNAMO_ENVIRONMENT_GUARD_TYPES = frozenset(
    {
        "AUTOGRAD_SAVED_TENSORS_HOOKS",
        "DEFAULT_DEVICE",
        "DETERMINISTIC_ALGORITHMS",
        "GRAD_MODE",
        "GLOBAL_STATE",
        "TORCH_FUNCTION_STATE",
    }
)
_DYNAMO_VALUE_GUARD_TYPES = frozenset({"CONSTANT_MATCH", "EQUALS_MATCH"})
_DYNAMO_OBJ_ID = re.compile(r"(?<=, )\d+(?=\), type=)")
_DYNAMO_SAVED_HOOK_IDS = re.compile(
    r"(?<=top_saved_tensors_hooks ids == )\(\d+(?:, \d+)*\)"
)
_DYNAMO_COUNTER = re.compile(
    r"(__builtins_dict__|__compiled_fn|__resume_at)_*\d+(_\d+)?"
)
_DYNAMO_GLOBAL_BY_ID = re.compile(r"_\d{9,}_c\d+\b")


def _normalize_dynamo_guard_text(text: str) -> str:
    text = _DYNAMO_SAVED_HOOK_IDS.sub("(<ids>)", text)
    text = _DYNAMO_GLOBAL_BY_ID.sub("_<id>_c<n>", _DYNAMO_OBJ_ID.sub("<id>", text))
    return _DYNAMO_COUNTER.sub(r"\1_<n>", text)


def _stable_code_constants(constants: tuple[object, ...]) -> tuple[object, ...]:
    stable = (str, int, float, complex, bytes, bool, type(None))
    result: list[object] = []
    for value in constants:
        if isinstance(value, stable):
            result.append(value)
        elif isinstance(value, CodeType):
            result.append(_dynamo_code_fingerprint(value))
        elif isinstance(value, tuple):
            result.append(_stable_code_constants(value))
        elif isinstance(value, frozenset):
            result.append(tuple(sorted(_stable_code_constants(tuple(value)), key=repr)))
    return tuple(result)


def _dynamo_code_fingerprint(code: CodeType) -> str:
    data = (
        code.co_code,
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        _stable_code_constants(code.co_consts),
    )
    return hashlib.sha256(repr(data).encode()).hexdigest()[:12]


def _dynamo_object_identity(value: object) -> str:
    if isinstance(value, types.ModuleType):
        return f"is module {value.__name__}"
    name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if isinstance(name, str):
        code = getattr(value, "__code__", None)
        where = ""
        if isinstance(code, CodeType):
            site = os.path.basename(code.co_filename or "?")
            where = f"@{site}:{code.co_firstlineno}#{_dynamo_code_fingerprint(code)} "
        owner = getattr(value, "__module__", None) or "?"
        return _normalize_dynamo_guard_text(f"is {where}{owner}.{name}")[:160]
    return f"is a {type(value).__module__}.{type(value).__qualname__}"[:160]


def _dynamo_guard_value(entry: Any) -> str:
    if entry.guard_type == "AUTOGRAD_SAVED_TENSORS_HOOKS":
        try:
            from torch._functorch._aot_autograd.utils import top_saved_tensors_hooks

            hooks = top_saved_tensors_hooks()
        except Exception:
            return ""
        return "hooks=None" if not hooks else f"hooks={len(hooks)}"
    if entry.guard_type == "GRAD_MODE":
        return f"grad_enabled={torch.is_grad_enabled()}"
    if not entry.has_value:
        return ""
    value = entry.value
    if isinstance(value, torch.Tensor):
        try:
            from torch._dynamo.guards import (
                convert_to_concrete_values,
                get_tensor_guard_code_part,
            )

            pytype = getattr(value, "pytype", type(value))
            is_python_fake = isinstance(  # noqa: ISINSTANCE_FAKE_TENSOR
                value, torch._subclasses.FakeTensor
            )
            if is_python_fake and pytype is type(value):
                pytype = torch.Tensor
            dispatch_keys = getattr(value, "dispatch_keys", None)
            if dispatch_keys is None:
                dispatch_keys = torch._C._dispatch_keys(value)

            return get_tensor_guard_code_part(
                value,
                "",
                convert_to_concrete_values(value.size()),
                convert_to_concrete_values(value.stride()),
                pytype,
                dispatch_keys,
            )
        except Exception:
            return f"type={type(value).__name__}, dtype={value.dtype}, <unrenderable>"
    unsupported = set(
        torch._dynamo.guards.CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
    )
    if entry.guard_type in unsupported or unsupported.intersection(
        entry.derived_guard_types
    ):
        return _dynamo_object_identity(value)
    return ""


def _dynamo_guard_fact(entry: Any, *, enforced: bool) -> GuardFact:
    code = getattr(entry, "code", ()) or tuple(entry.orig_guard.code_list or ())
    return GuardFact(
        guard_type=entry.guard_type,
        source=_normalize_dynamo_guard_text(entry.name),
        code=tuple(_normalize_dynamo_guard_text(part) for part in code),
        value=(
            ""
            if entry.guard_type in _DYNAMO_UNMODELLED_GUARD_TYPES
            else _dynamo_guard_value(entry)
        ),
        enforced=enforced,
    )


def _dynamo_guard_slot(guard: Any) -> tuple[str, str]:
    from torch._dynamo.guards import strip_local_scope

    return (
        guard.create_fn_name(),
        _normalize_dynamo_guard_text(strip_local_scope(guard.name)),
    )


@dataclasses.dataclass(frozen=True)
class _DynamoGuardFinalization:
    states: tuple[bytes, ...]
    kept_slots: tuple[frozenset[tuple[str, str]], ...]
    policy_dropped: frozenset[tuple[str, str]]


@dataclasses.dataclass(frozen=True)
class _DynamoCapturedGuardSet:
    example_index: int
    facts: tuple[GuardFact, ...]
    dropped: frozenset[tuple[str, str]]
    risky_dropped: frozenset[tuple[str, str]]
    environment: frozenset[tuple[str, str]]


def _dynamo_frame_invariants(
    code_entries: Sequence[Any],
    captured: dict[int, list[_DynamoCapturedGuardSet]],
    kept_by_entry: dict[int, list[frozenset[tuple[str, str]]]],
) -> tuple[FrameInvariants, ...]:
    from torch._dynamo.package import SerializedCode

    result = []
    for entry in code_entries:
        code = SerializedCode.to_code_object(entry.python_code)
        records = captured.get(id(entry), [])
        final_slots = kept_by_entry.get(id(entry), [])
        variants: list[frozenset[GuardFact]] = []
        undetermined: set[GuardFact] = set()
        for index, record in enumerate(records):
            kept = final_slots[index] if index < len(final_slots) else frozenset()
            facts = set()
            for fact in record.facts:
                enforced = fact.enforced and (fact.guard_type, fact.source) in kept
                updated = dataclasses.replace(fact, enforced=enforced)
                if fact.guard_type in _DYNAMO_UNMODELLED_GUARD_TYPES:
                    undetermined.add(updated)
                else:
                    facts.add(updated)
            variants.append(frozenset(facts))
        shared = frozenset.intersection(*variants) if variants else frozenset()
        every: set[GuardFact] = set()
        for facts in variants:
            every.update(facts)

        def order(fact: GuardFact) -> tuple[str, str, str, str]:
            return (
                fact.source,
                fact.guard_type,
                " ".join(fact.code),
                fact.value,
            )

        result.append(
            FrameInvariants(
                frame=code.co_name,
                filename=code.co_filename,
                lineno=code.co_firstlineno,
                variants=len(entry.guarded_codes),
                variant_examples=tuple(record.example_index for record in records),
                invariant=tuple(sorted(shared, key=order)),
                varying=tuple(sorted(every - shared, key=order)),
                undetermined=tuple(sorted(undetermined, key=order)),
            )
        )
    return tuple(result)


def _dynamo_wont_generalize(
    kept: set[tuple[str, str]],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                source
                for guard_type, source in kept
                if guard_type in _DYNAMO_VALUE_GUARD_TYPES
                and "." not in source
                and "[" not in source
            }
        )
    )


def _write_dynamo_invariants(
    path: str, target: Callable[..., object], frames: Sequence[FrameInvariants]
) -> None:
    from pathlib import Path

    lines = [
        f"# precompile invariants for {target.__module__}.{target.__qualname__}",
        "# Generated from one execution of each supplied example.",
        "",
    ]
    for frame in frames:
        lines.extend(
            [
                f"[{frame.frame} at {frame.filename}:{frame.lineno}]",
                f"variants = {frame.variants}",
                f"variant_examples = {frame.variant_examples!r}",
                "invariant:",
                *([f"  {fact.render()}" for fact in frame.invariant] or ["  <none>"]),
                "varying:",
                *([f"  {fact.render()}" for fact in frame.varying] or ["  <none>"]),
                "undetermined:",
                *(
                    [f"  {fact.render()}" for fact in frame.undetermined]
                    or ["  <none>"]
                ),
                "",
            ]
        )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def _filter_dynamo_guards(
    target_code: CodeType,
    runtime_global_scope: dict[str, object],
    guarded_codes: Sequence[Any],
    captured: Sequence[_DynamoCapturedGuardSet],
    live_leaf_sets: Sequence[frozenset[tuple[str, str, str]]],
) -> _DynamoGuardFinalization:
    """Rebuild and validate guards from frozen capture state."""
    import dataclasses
    import functools

    from torch._dynamo.guards import (
        _companion_attribute_guards,
        CheckFunctionManager,
        GuardBuilder,
        strip_local_scope,
    )
    from torch._dynamo.output_graph import OutputGraphCommon
    from torch._dynamo.package import load_guard_manager, load_guards_state
    from torch._guards import GuardsSet
    from torch.utils._ordered_set import OrderedSet

    def fresh_guard(guard: Any, *, final: bool = False) -> Any:
        create_fn = guard.create_fn
        if (
            final
            and isinstance(create_fn, functools.partial)
            and create_fn.func is GuardBuilder.TENSOR_MATCH
        ):
            # Drop only the capture-time TensorWeakRef; any other bound options remain
            # part of the guard's semantics.
            keywords = dict(create_fn.keywords)
            keywords.pop("value", None)
            create_fn = functools.partial(
                GuardBuilder.TENSOR_MATCH, *create_fn.args, **keywords
            )
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
        return load_guard_manager(state, target_code, runtime_global_scope)

    states = [load_guards_state(guarded.guards_state) for guarded in guarded_codes]
    if len(states) != len(captured):
        raise PrecompileError(
            "precompile tracer='dynamo' did not record one guard set for every "
            "captured variant."
        )
    if len(states) != len(live_leaf_sets):
        raise PrecompileError(
            "precompile tracer='dynamo' did not record one live guard tree for "
            "every captured variant."
        )
    filtered_states: list[bytes] = []
    kept_slots: list[frozenset[tuple[str, str]]] = []
    policy_dropped: set[tuple[str, str]] = set()
    for state, record, live_leaves in zip(
        states, captured, live_leaf_sets, strict=True
    ):
        live_facts: dict[tuple[str, str], set[GuardFact]] = {}
        for fact in record.facts:
            if fact.enforced:
                live_facts.setdefault((fact.guard_type, fact.source), set()).add(fact)
        kept_guards = [
            guard
            for guard in state.output_graph.guards
            if _dynamo_guard_slot(guard) not in record.environment
        ]
        kept_aot_guards = list(state.output_graph.aotautograd_guards)
        environment_sources = {source for _, source in record.environment}
        kept_key_order = sorted(
            (
                source
                for source in state.output_graph.guard_on_key_order
                if _normalize_dynamo_guard_text(strip_local_scope(source.name))
                not in environment_sources
            ),
            key=lambda source: source.name,
        )

        output_graph = dataclasses.replace(
            state.output_graph,
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
        failures: list[tuple[Any, Exception]] = []
        drifted: list[tuple[Any, GuardFact]] = []

        def keep_unchanged_guards(entries: Sequence[Any]) -> list[bool]:
            for entry in entries:
                fact = _dynamo_guard_fact(entry, enforced=True)
                slot = (fact.guard_type, fact.source)
                if fact not in live_facts.get(slot, set()):
                    drifted.append((entry.orig_guard, fact))
            companions = _companion_attribute_guards(
                [entry.orig_guard for entry in entries],
                [
                    (guard, RuntimeError("the rebuilt guard changed"))
                    for guard, _ in drifted
                ],
            )
            dropped = {id(guard) for guard, _ in drifted}
            dropped.update(id(guard) for guard, _ in companions)
            return [id(entry.orig_guard) not in dropped for entry in entries]

        check_fn = CheckFunctionManager(
            target_code,
            OutputGraphCommon(output_graph),
            shape_code_parts=shape_code_parts,
            runtime_global_scope=runtime_global_scope,
            save_guards=True,
            strict_error=True,
            guard_build_local_state=state.local_state,
            serialization_guard_filter_fn=keep_unchanged_guards,
            collect_guard_failures=failures,
        )
        failed_slots = {_dynamo_guard_slot(guard) for guard, _ in failures}
        input_failures = failed_slots - record.environment
        if input_failures:
            guard, error = next(
                (guard, error)
                for guard, error in failures
                if _dynamo_guard_slot(guard) in input_failures
            )
            raise PrecompileError(
                "precompile tracer='dynamo' cannot rebuild input-derived guard "
                f"{_dynamo_guard_slot(guard)!r}: {type(error).__name__}: {error}"
            ) from error
        input_drift = [
            (guard, fact)
            for guard, fact in drifted
            if _dynamo_guard_slot(guard) not in record.environment
        ]
        if input_drift:
            guard, fact = input_drift[0]
            raise PrecompileError(
                "precompile tracer='dynamo' rebuilt input-derived guard "
                f"{_dynamo_guard_slot(guard)!r} with a changed predicate: "
                f"{fact.render()}"
            )
        if check_fn.guards_state is None:
            raise PrecompileError(
                "precompile tracer='dynamo' did not serialize its rebuilt guard state."
            )
        filtered_state = load_guards_state(check_fn.guards_state)
        rebuilt = manager_for(filtered_state)
        dropped_sources = {
            source for _, source in record.environment | record.dropped if source
        }
        dropped_root_types = {
            guard_type
            for guard_type, source in record.environment | record.dropped
            if not source
        }
        root_environment_types = {
            "DEFAULT_DEVICE",
            "GLOBAL_STATE",
            "TORCH_FUNCTION_MODE_STACK",
        }

        def normalize_leaf(
            leaf: tuple[str, str, str],
        ) -> tuple[str, str, str]:
            source, guard_type, payload = leaf
            return (
                _normalize_dynamo_guard_text(source),
                guard_type,
                _normalize_dynamo_guard_text(payload),
            )

        def is_dropped_leaf(leaf: tuple[str, str, str]) -> bool:
            source, guard_type, payload = leaf
            if not source:
                return guard_type in root_environment_types or (
                    guard_type == "LAMBDA_GUARD"
                    and (
                        "top_saved_tensors_hooks" in payload
                        or "SHAPE_ENV" in dropped_root_types
                    )
                )
            return any(
                source == root or source.startswith((f"{root}.", f"{root}["))
                for root in dropped_sources
            )

        normalized_live_leaves = map(normalize_leaf, live_leaves)
        expected_leaves = frozenset(
            leaf for leaf in normalized_live_leaves if not is_dropped_leaf(leaf)
        )
        rebuilt_leaves = frozenset(
            normalize_leaf(leaf) for leaf in rebuilt.leaf_fingerprint()
        )
        changed_leaves = expected_leaves ^ rebuilt_leaves
        if changed_leaves:
            examples = ", ".join(
                f"{source or '<root>'}: {guard_type}: {payload}"
                for source, guard_type, payload in sorted(changed_leaves)[:3]
            )
            raise PrecompileError(
                "precompile tracer='dynamo' rebuilt a guard tree with changed "
                f"input-derived checks: {examples}"
            )
        filtered_states.append(check_fn.guards_state)
        final_slots = frozenset(
            _dynamo_guard_slot(guard) for guard in filtered_state.output_graph.guards
        )
        kept_slots.append(final_slots)
        policy_dropped.update(
            {_dynamo_guard_slot(guard) for guard in state.output_graph.guards}
            - final_slots
        )

    facts_by_variant = [
        {
            slot: frozenset(
                dataclasses.replace(fact, enforced=True)
                for fact in record.facts
                if (fact.guard_type, fact.source) == slot
            )
            for slot in {(fact.guard_type, fact.source) for fact in record.facts}
        }
        for record in captured
    ]
    for index, facts in enumerate(facts_by_variant):
        for earlier in range(index):
            if not any(
                facts_by_variant[earlier].get(slot) != facts.get(slot)
                for slot in kept_slots[earlier]
            ):
                differing = sorted(
                    slot
                    for slot in facts_by_variant[earlier].keys() | facts.keys()
                    if facts_by_variant[earlier].get(slot) != facts.get(slot)
                )
                raise PrecompileError(
                    "precompile tracer='dynamo' dropped guards that can affect "
                    f"dispatch between captured variants {earlier} and {index}: "
                    f"{differing}"
                )

    return _DynamoGuardFinalization(
        states=tuple(filtered_states),
        kept_slots=tuple(kept_slots),
        policy_dropped=frozenset(policy_dropped),
    )


def _dynamo_code_names(code: CodeType) -> set[str]:
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, CodeType):
            names.update(_dynamo_code_names(const))
    return names


def _dynamo_code_writes_grad(code: CodeType) -> bool:
    if any(
        instruction.opname in ("STORE_ATTR", "DELETE_ATTR")
        and instruction.argval == "grad"
        for instruction in dis.get_instructions(code)
    ):
        return True
    return any(
        isinstance(constant, CodeType) and _dynamo_code_writes_grad(constant)
        for constant in code.co_consts
    )


def _reachable_dynamo_frames(codes: Sequence[_DynamoCodeState]) -> set[int]:
    from torch._dynamo.package import SerializedCode

    reachable = {0}
    while True:
        names: set[str] = set()
        for index in reachable:
            names.update(
                *(
                    _dynamo_code_names(
                        SerializedCode.to_code_object(variant.dynamo_code)
                    )
                    for variant in codes[index].variants
                )
            )
        added = {
            index
            for index, code in enumerate(codes)
            if index not in reachable
            and code.install_to_global
            and any(name in names for name in code.function_names)
        }
        if not added:
            return reachable
        reachable.update(added)


def _dynamo_serving_mode(codes: Sequence[_DynamoCodeState]) -> str:
    reachable = _reachable_dynamo_frames(codes)
    return (
        "installed"
        if any(
            code.variants and index not in reachable for index, code in enumerate(codes)
        )
        else "standalone"
    )


def _build_dynamo_python_source(
    *,
    backend: str,
    training: bool,
    state: _DynamoArtifactState,
    backend_ids: list[str],
    compiled_backends: list[_DynamoPythonBackend],
) -> str:
    import base64
    import inspect
    import pickle

    from torch import _precompile_driver as driver
    from torch._functorch._aot_autograd.to_standalone_python import (
        namespace_module_names,
    )

    try:
        encoded_state = base64.b64encode(pickle.dumps(state)).decode("ascii")
    except Exception as e:
        raise PrecompileError(
            "precompile tracer='dynamo' could not serialize its guards and transformed "
            f"bytecode ({type(e).__name__}: {e})."
        ) from e

    dynamic_count = sum(compiled.is_dynamic for compiled in compiled_backends)
    variant_count = sum(len(code.variants) for code in state.codes)
    summary = state.summary
    reachable = _reachable_dynamo_frames(state.codes)
    unreachable = tuple(
        code.code.co_name
        for index, code in enumerate(state.codes)
        if code.variants and index not in reachable
    )
    serving_description = (
        [
            "# This artifact installs captured entries into an isolated Dynamo cache",
            "# region on first call because ordinary Python calls make some captured",
            "# frames unreachable from a standalone bytecode dispatcher. Loading",
            "# prepares its backends and guards without installing them. An uncovered",
            "# call raises instead of compiling; unload()",
            "# removes only this artifact's entries and installed globals.",
        ]
        if state.serving_mode == "installed"
        else [
            "# This artifact dispatches directly among captured entry/resume variants.",
            "# It installs nothing and never compiles after loading.",
        ]
    )
    parts = [
        '# Generated by torch.compiler.precompile (tracer="dynamo") -- do not edit.',
        "#",
        "# Graph entry points and kernels below are ordinary Python source. Nested FX",
        "# graph structure used by eager higher-order ops, Dynamo guard trees, and",
        "# required code objects are stored as labelled base64-encoded pickle data.",
        *serving_description,
        "",
        "# " + "=" * 70,
        "# 1. Capture metadata",
        "# " + "=" * 70,
        f"BACKEND = {backend!r}",
        'TRACER = "dynamo"',
        f"TRAINING = {training!r}",
        f"FRAME_COUNT = {len(state.codes)}",
        f"VARIANT_COUNT = {variant_count}",
        f"GRAPH_COUNT = {len(compiled_backends)}",
        f"DYNAMIC_GRAPH_COUNT = {dynamic_count}",
        f"SERVING_MODE = {state.serving_mode!r}",
        f"UNREACHABLE_WITHOUT_INSTALL = {unreachable!r}",
        f"CAPTURE_COMPLETE = {summary.complete if summary is not None else True!r}",
        f"DROPPED_GUARDS = {list(summary.dropped_guards) if summary else []!r}",
        f"RISKY_DROPPED_GUARDS = "
        f"{list(summary.risky_dropped_guards) if summary else []!r}",
        f"POLICY_DROPPED_GUARDS = "
        f"{list(summary.policy_dropped_guards) if summary else []!r}",
        f"WONT_GENERALIZE = {summary.wont_generalize if summary else ()!r}",
        f"_DYNAMO_PYTHON_VERSION = {tuple(sys.version_info[:2])!r}",
        f"_DYNAMO_TORCH_VERSION = {torch.__version__!r}",
        f"_DYNAMO_BACKEND_IDS = {tuple(backend_ids)!r}",
    ]
    namespaced_sources = namespace_module_names(
        [compiled.python_code for compiled in compiled_backends]
    )
    parts.extend(
        [
            "",
            "# " + "=" * 70,
            "# 2. Compiled graph and kernel source",
            "# " + "=" * 70,
            "_DYNAMO_BACKENDS = {}",
        ]
    )
    for index, (backend_id, source) in enumerate(
        zip(backend_ids, namespaced_sources, strict=True)
    ):
        parts.extend(
            [
                "",
                "# " + "-" * 70,
                f"# Backend graph {index}: {backend_id}",
                "# " + "-" * 70,
                source,
                f"_DYNAMO_BACKENDS[{backend_id!r}] = call_s{index}",
            ]
        )
    parts.extend(
        [
            "",
            "# " + "=" * 70,
            "# 3. Guard trees and Dynamo/disabled-function bytecode (opaque)",
            "# " + "=" * 70,
            "# This pickle has the same trust boundary as the executable source itself.",
            f"_DYNAMO_STATE = {encoded_state!r}",
            "",
            "# " + "=" * 70,
            "# 4. Python runtime glue: rebuild guards/resume frames and dispatch",
            "# " + "=" * 70,
            inspect.getsource(driver._validate_dynamo_artifact_state),
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


def _dynamo_input_contract(
    example_inputs: Sequence[ExampleInput],
) -> _DynamoInputContract:
    def tensor_inputs(
        leaves: Sequence[object],
    ) -> tuple[list[torch.Tensor], set[int]]:
        tensors: list[torch.Tensor] = []
        decomposed = set()
        pending = list(leaves)
        seen = set()
        while pending:
            value = pending.pop()
            value_id = id(value)
            if value_id in seen:
                continue
            seen.add(value_id)
            if isinstance(value, torch.Tensor):
                tensors.append(value)
                tensor_flatten = getattr(value, "__tensor_flatten__", None)
                if type(value) is not torch.Tensor and callable(tensor_flatten):
                    try:
                        flattened = tensor_flatten()
                        if not isinstance(flattened, tuple) or len(flattened) != 2:
                            raise TypeError("invalid __tensor_flatten__ result")
                        names = flattened[0]
                        if not isinstance(names, (tuple, list)):
                            raise TypeError("invalid __tensor_flatten__ names")
                        children = [getattr(value, name) for name in names]
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        children = []
                    if any(isinstance(child, torch.Tensor) for child in children):
                        decomposed.add(value_id)
                        pending.extend(children)
                continue
            if isinstance(value, (dict, MappingProxyType)):
                pending.extend(value.keys())
                pending.extend(value.values())
                if type(value) in (dict, MappingProxyType):
                    continue
            elif isinstance(value, (tuple, list, set, frozenset)):
                pending.extend(value)
                if type(value) in (tuple, list, set, frozenset):
                    continue
            if isinstance(value, torch.nn.Module):
                pending.extend(value.modules())
                pending.extend(
                    tensor
                    for _, tensor in value.named_parameters(remove_duplicate=False)
                )
                pending.extend(
                    tensor for _, tensor in value.named_buffers(remove_duplicate=False)
                )
            if isinstance(
                value, (CodeType, type, types.FunctionType, types.ModuleType)
            ):
                continue
            pending.extend(_dynamo_object_state_values(value))
        return tensors, decomposed

    def has_storage_alias(leaves: Sequence[object]) -> bool:
        tensors, decomposed = tensor_inputs(leaves)
        storage_ranges: dict[tuple[str, int | None], list[tuple[int, int]]] = {}
        storage_ids: set[tuple[str, int | None, int]] = set()
        seen_objects: set[int] = set()
        for tensor in tensors:
            object_id = id(tensor)
            if object_id in seen_objects:
                continue
            seen_objects.add(object_id)
            try:
                storage = tensor.untyped_storage()
            except RuntimeError as e:
                if object_id in decomposed:
                    continue
                raise PrecompileError(
                    "precompile tracer='dynamo' cannot verify storage aliasing for "
                    "this tensor input."
                ) from e
            storage_key = (tensor.device.type, tensor.device.index, storage._cdata)
            if storage_key in storage_ids:
                return True
            storage_ids.add(storage_key)
            try:
                start = storage.data_ptr()
                size = storage.nbytes()
            except RuntimeError as e:
                if object_id in decomposed:
                    continue
                raise PrecompileError(
                    "precompile tracer='dynamo' cannot verify storage aliasing for "
                    "this tensor input."
                ) from e
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

    def module_signature(module: torch.nn.Module) -> tuple[object, ...]:
        tensors: list[tuple[str, str, torch.Tensor]] = [
            ("parameter", name, tensor)
            for name, tensor in module.named_parameters(remove_duplicate=False)
        ]
        tensors.extend(
            ("buffer", name, tensor)
            for name, tensor in module.named_buffers(remove_duplicate=False)
        )
        aliases: dict[int, int] = {}
        metadata = []
        for kind, name, tensor in tensors:
            alias = aliases.setdefault(id(tensor), len(aliases))
            metadata.append(
                (
                    kind,
                    name,
                    alias,
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    str(tensor.dtype),
                    str(tensor.device),
                    tensor.requires_grad,
                )
            )
        return (
            type(module).__module__,
            type(module).__qualname__,
            module.training,
            tuple(metadata),
        )

    groups: dict[str, list[list[object]]] = {}
    for example in example_inputs:
        if has_storage_alias((example.args, example.kwargs)):
            raise PrecompileError(
                "precompile tracer='dynamo' does not support storage aliasing between "
                "distinct tensor inputs."
            )
        leaves, spec = pytree.tree_flatten((example.args, example.kwargs))
        try:
            serialized_spec = pytree.treespec_dumps(spec)
        except Exception as e:
            raise PrecompileError(
                "precompile tracer='dynamo' cannot serialize the input structure "
                "needed for runtime safety checks. Register a serializable pytree "
                "node or use only built-in pytree containers."
            ) from e
        groups.setdefault(serialized_spec, []).append(leaves)

    variants = []
    for serialized_spec, leaves_by_example in groups.items():
        leaf_contracts: list[dict[str, object] | None] = []
        for leaves in zip(*leaves_by_example):
            if all(isinstance(leaf, torch.nn.Module) for leaf in leaves):
                signatures = tuple(
                    dict.fromkeys(
                        module_signature(cast("torch.nn.Module", leaf))
                        for leaf in leaves
                    )
                )
                leaf_contracts.append({"kind": "module", "variants": signatures})
                continue
            if not all(isinstance(leaf, torch.Tensor) for leaf in leaves):
                leaf_contracts.append(None)
                continue
            tensors = cast("tuple[torch.Tensor, ...]", leaves)

            def common(values: Sequence[object]) -> object | None:
                first = values[0]
                return first if all(value == first for value in values[1:]) else None

            ranks = [tensor.dim() for tensor in tensors]
            rank = cast("int | None", common(ranks))
            shapes = [tuple(tensor.shape) for tensor in tensors]
            strides = [tuple(tensor.stride()) for tensor in tensors]
            marked_dims = set().union(
                *(
                    set(getattr(tensor, "_dynamo_unbacked_indices", None) or ())
                    | set(
                        getattr(tensor, "_dynamo_strict_unbacked_indices", None) or ()
                    )
                    for tensor in tensors
                )
            )
            shape = (
                tuple(
                    None
                    if dim in marked_dims
                    else common([dims[dim] for dims in shapes])
                    for dim in range(rank)
                )
                if rank is not None
                else None
            )
            stride = (
                None
                if marked_dims
                else tuple(
                    common([dims[dim] for dims in strides]) for dim in range(rank)
                )
                if rank is not None
                else None
            )
            leaf_contracts.append(
                {
                    "kind": "tensor",
                    "type": common(
                        [(type(t).__module__, type(t).__qualname__) for t in tensors]
                    ),
                    "dtype": common([str(t.dtype) for t in tensors]),
                    "device": common([str(t.device) for t in tensors]),
                    "requires_grad": common([t.requires_grad for t in tensors]),
                    "shape": shape,
                    "stride": stride,
                }
            )
        variants.append(
            _DynamoInputContractVariant(serialized_spec, tuple(leaf_contracts))
        )
    return _DynamoInputContract(tuple(variants))


def _dynamo_object_state_values(value: object) -> list[object]:
    try:
        values = list(vars(value).values())
    except TypeError:
        values = []
    if not isinstance(value, type):
        value_type = type(value)
        for cls in value_type.__mro__:
            for descriptor in vars(cls).values():
                if not isinstance(descriptor, types.MemberDescriptorType):
                    continue
                try:
                    values.append(descriptor.__get__(value, value_type))
                except AttributeError:
                    pass
    values.extend(
        item
        for item in gc.get_referents(value)
        if not isinstance(item, (CodeType, types.ModuleType, type))
    )
    if isinstance(value, contextvars.ContextVar):
        try:
            values.append(value.get())
        except LookupError:
            pass
    elif isinstance(value, contextvars.Context):
        values.extend(value.values())
    return values


def _dynamo_reachable_object_ids(
    values: Sequence[object], *, skip_literals: bool = False
) -> set[int]:
    stack = list(values)
    seen: set[int] = set()
    literal_types = (bool, int, float, complex, str, bytes)
    while stack:
        value = stack.pop()
        literal = value is None or type(value) in literal_types
        if skip_literals and literal and value == value:
            continue
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        if isinstance(value, (dict, MappingProxyType)):
            stack.extend(value.keys())
            stack.extend(value.values())
            if type(value) not in (dict, MappingProxyType):
                stack.extend(_dynamo_object_state_values(value))
        elif isinstance(value, (list, tuple, set, frozenset)):
            stack.extend(value)
            if type(value) not in (list, tuple, set, frozenset):
                stack.extend(_dynamo_object_state_values(value))
        elif isinstance(value, torch.nn.Module):
            stack.extend(value.modules())
            stack.extend(value.parameters())
            stack.extend(value.buffers())
            stack.extend(_dynamo_object_state_values(value))
        elif isinstance(value, weakref.ReferenceType):
            referent = value()
            if referent is not None:
                stack.append(referent)
            callback = value.__callback__
            if callback is not None:
                stack.append(callback)
        elif isinstance(value, types.MethodType):
            stack.extend((value.__func__, value.__self__))
        elif isinstance(value, types.FunctionType):
            stack.extend(value.__defaults__ or ())
            stack.extend((value.__kwdefaults__ or {}).values())
            stack.extend(value.__dict__.values())
            for cell in value.__closure__ or ():
                try:
                    stack.append(cell.cell_contents)
                except ValueError:
                    pass
        elif isinstance(value, types.GeneratorType):
            continue
        elif not isinstance(value, (CodeType, type, types.ModuleType)):
            stack.extend(_dynamo_object_state_values(value))
    return seen


def _dynamo_input_object_ids(
    fn: Callable[..., object], example_inputs: Sequence[ExampleInput]
) -> set[int]:
    values = [
        value
        for example in example_inputs
        for value in (*example.args, *example.kwargs.values())
    ]
    if isinstance(fn, types.FunctionType):
        values.extend(fn.__defaults__ or ())
        values.extend((fn.__kwdefaults__ or {}).values())
    elif isinstance(fn, types.MethodType):
        values.extend(fn.__func__.__defaults__ or ())
        values.extend((fn.__func__.__kwdefaults__ or {}).values())
    if isinstance(fn, torch.nn.Module):
        values.append(fn)
    return _dynamo_reachable_object_ids(values)


def _dynamo_example_grads(
    fn: Callable[..., object], example_inputs: Sequence[ExampleInput]
) -> dict[int, tuple[torch.Tensor, torch.Tensor | None]]:
    tensors: dict[int, torch.Tensor] = {}

    def visit(value: object) -> None:
        if isinstance(value, torch.nn.Module):
            for tensor in itertools.chain(value.parameters(), value.buffers()):
                tensors[id(tensor)] = tensor
        elif isinstance(value, torch.Tensor):
            tensors[id(value)] = value

    receiver = fn if isinstance(fn, torch.nn.Module) else getattr(fn, "__self__", None)
    if receiver is not None:
        visit(receiver)
    for example in example_inputs:
        for value in (*example.args, *example.kwargs.values()):
            visit(value)
        for value in pytree.tree_leaves((example.args, example.kwargs)):
            visit(value)
    return {key: (tensor, tensor.grad) for key, tensor in tensors.items()}


def _precompile_dynamo(
    fn: Callable[..., object],
    example_inputs: Sequence[tuple[object, ...] | ExampleInput],
    *,
    backend: str,
    decompositions: dict | None,
    training: bool,
    recompile_limit: int,
    dynamic: bool | None,
    guard_filter_fn: Callable[[Sequence[Any]], Sequence[bool]] | None,
    invariants: str | None,
    require_complete: bool,
    require_no_risky_drops: bool,
    require_no_dropped_guards: bool,
) -> tuple[str, bytes]:
    import contextlib
    import dis
    import importlib
    import inspect
    import operator
    import types

    import torch._functorch.config as functorch_config

    if not example_inputs:
        raise AssertionError(
            "precompile with tracer='dynamo' requires at least one example input tuple."
        )
    if decompositions is not None:
        raise NotImplementedError(
            "precompile decompositions are not yet supported with tracer='dynamo'."
        )
    if training and torch.is_inference_mode_enabled():
        raise PrecompileError(
            "precompile tracer='dynamo' cannot capture training=True inside "
            "torch.inference_mode()."
        )
    examples: list[ExampleInput] = []
    for example in example_inputs:
        if isinstance(example, ExampleInput):
            examples.append(example)
        elif isinstance(example, tuple):
            examples.append(ExampleInput(example))
        else:
            raise TypeError(
                "precompile example_inputs must contain positional-argument tuples "
                f"or torch.compiler.ExampleInput values, got {type(example).__name__}."
            )

    def check_module_state(module: torch.nn.Module) -> None:
        state = (
            ("parameter", module.named_parameters()),
            ("buffer", module.named_buffers()),
        )
        for kind, tensors in state:
            for name, tensor in tensors:
                if torch.is_inference(tensor):
                    raise PrecompileError(
                        "precompile tracer='dynamo' found inference tensor "
                        f"{kind} {name!r} on {type(module).__name__}; create the "
                        "module outside torch.inference_mode()."
                    )

    if isinstance(fn, torch.nn.Module):
        check_module_state(fn)
    for example in examples:
        for value in pytree.tree_leaves((example.args, example.kwargs)):
            if isinstance(value, torch.Tensor) and torch.is_inference(value):
                raise PrecompileError(
                    "precompile tracer='dynamo' example_inputs cannot contain "
                    "inference tensors; create them outside torch.inference_mode()."
                )
            if isinstance(value, torch.nn.Module):
                check_module_state(value)

    from torch._dynamo.eval_frame import innermost_fn
    from torch._dynamo.exc import (
        BackendCompilerFailed,
        FailOnRecompileLimitHit,
        InternalTorchDynamoError,
        PackageError,
        RecompileError,
        Unsupported,
    )
    from torch._dynamo.guards import CheckFunctionManager
    from torch._dynamo.package import CompilePackage, SerializedCode
    from torch._dynamo.pgo import _new_code_state, _use_code_state

    entry = fn.forward if isinstance(fn, torch.nn.Module) else fn
    if inspect.ismethod(fn):
        raise NotImplementedError(
            "precompile tracer='dynamo' does not support bound methods; pass the "
            "unbound function and its receiver as an explicit input."
        )
    target_callable = innermost_fn(entry)
    target = (
        target_callable.__func__
        if inspect.ismethod(target_callable)
        else target_callable
    )
    input_object_ids = _dynamo_input_object_ids(fn, examples)
    if not inspect.isfunction(target):
        raise NotImplementedError(
            "precompile tracer='dynamo' currently requires a Python function and does "
            f"not support {type(target).__name__}."
        )
    if target.__closure__ is not None:
        raise NotImplementedError(
            "precompile tracer='dynamo' does not yet support functions with closure "
            "cells; pass captured values as explicit arguments."
        )
    if target.__code__.co_cellvars:
        raise NotImplementedError(
            "precompile tracer='dynamo' does not yet support nested functions that "
            "capture local variables."
        )

    def is_literal(value: object) -> bool:
        if value is None or value is Ellipsis or value is NotImplemented:
            return True
        if type(value) in (
            bool,
            int,
            float,
            complex,
            str,
            bytes,
        ):
            return True
        if type(value) is tuple:
            return all(is_literal(item) for item in value)
        if type(value) is frozenset:
            return all(is_literal(item) for item in value)
        return False

    def importable_global(value: object) -> tuple[str, tuple[str, ...]] | None:
        if isinstance(value, types.ModuleType):
            if value.__name__ == "__main__" or value.__spec__ is None:
                return None
            try:
                imported = importlib.import_module(value.__name__)
            except ImportError:
                return None
            return (value.__name__, ()) if imported is value else None
        module_name = getattr(value, "__module__", None)
        if not isinstance(module_name, str):
            return None
        names = [getattr(value, "__qualname__", None), getattr(value, "__name__", None)]
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            return None
        for name in names:
            if not isinstance(name, str) or "<locals>" in name:
                continue
            path = tuple(name.split("."))
            resolved: object = module
            try:
                for attr in path:
                    resolved = getattr(resolved, attr)
            except AttributeError:
                continue
            if resolved is value:
                return module_name, path
        return None

    def loaded_global_paths(code: CodeType) -> list[tuple[str, ...]]:
        paths = []
        instructions = list(dis.get_instructions(code))
        for index, instruction in enumerate(instructions):
            if instruction.opname not in (
                "LOAD_GLOBAL",
                "LOAD_NAME",
                "LOAD_FROM_DICT_OR_GLOBALS",
            ) or not isinstance(instruction.argval, str):
                continue
            path = [instruction.argval]
            for following in instructions[index + 1 :]:
                if following.opname not in ("LOAD_ATTR", "LOAD_METHOD"):
                    break
                if not isinstance(following.argval, str):
                    break
                path.append(following.argval)
            paths.append(tuple(path))
        for const in code.co_consts:
            if isinstance(const, CodeType):
                paths.extend(loaded_global_paths(const))
        return paths

    def is_stable_singleton(value: object) -> bool:
        return (
            value is None
            or type(value) is bool
            or value is Ellipsis
            or value is NotImplemented
        )

    def is_safely_reflexive(value: object, seen: set[int] | None = None) -> bool:
        if value is None or type(value) in (bool, int, str, bytes):
            return True
        if type(value) in (float, complex):
            return value == value
        if type(value) in (dict, deque, list, tuple, set, frozenset):
            seen = set() if seen is None else seen
            if id(value) in seen:
                return True
            seen.add(id(value))
            if type(value) is dict:
                items = (
                    item
                    for pair in cast("dict[object, object]", value).items()
                    for item in pair
                )
            else:
                items = iter(cast("Iterable[object]", value))
            return all(is_safely_reflexive(item, seen) for item in items)
        if isinstance(value, (functools.partial, functools.partialmethod)):
            return all(
                is_safely_reflexive(item, seen)
                for item in (*value.args, *(value.keywords or {}).values())
            )
        return False

    def is_identity_container(value: object) -> bool:
        return isinstance(
            value,
            (dict, MappingProxyType, deque, list, tuple, set, frozenset),
        )

    def is_identity_mapping(value: object) -> bool:
        return isinstance(value, (dict, MappingProxyType))

    def is_safely_reflexive_container_lookup(value: object) -> bool:
        if isinstance(value, (dict, MappingProxyType)):
            return all(is_safely_reflexive(item) for item in value)
        if isinstance(value, (deque, list, tuple, set, frozenset)):
            return all(is_safely_reflexive(item) for item in value)
        return is_safely_reflexive(value)

    def is_potentially_mutable(value: object) -> bool:
        if is_stable_singleton(value) or type(value) in (
            int,
            float,
            complex,
            str,
            bytes,
        ):
            return False
        if type(value) in (tuple, frozenset):
            items = cast("Iterable[object]", value)
            return any(is_potentially_mutable(item) for item in items)
        return True

    environment_objects: dict[int, object] = {}
    local_function_codes: dict[int, CodeType] = {}
    local_function_payloads: dict[
        int, tuple[bool, bool, frozenset[tuple[str, ...]]]
    ] = {}

    def environment_value_dependency(
        value: object,
    ) -> tuple[bool, bool, frozenset[tuple[str, ...]]]:
        environment_objects[id(value)] = value
        markers = {
            ("<environment-object>", str(id(value))),
            ("<singleton>",)
            if is_stable_singleton(value)
            else ("<identity-unstable>",),
        }
        if not is_safely_reflexive(value):
            markers.add(("<identity-sensitive>",))
        if is_identity_container(value) and not is_safely_reflexive_container_lookup(
            value
        ):
            markers.add(("<lookup-identity-sensitive>",))
        if is_identity_mapping(value) and not all(
            is_safely_reflexive(item) for item in cast("Mapping[object, object]", value)
        ):
            markers.add(("<mapping-key-identity-sensitive>",))
        if is_identity_mapping(value):
            markers.add(("<mapping>",))
        if is_potentially_mutable(value):
            markers.add(("<mutable-environment-object>",))
        return False, True, frozenset(markers)

    def bytecode_dependencies(
        code: CodeType,
        globals_scope: dict[str, object],
        parameter_dependencies: Mapping[
            str, tuple[bool, bool, frozenset[tuple[str, ...]]]
        ]
        | Sequence[tuple[bool, bool, frozenset[tuple[str, ...]]]]
        | None = None,
        active: set[int] | None = None,
        dependency_scopes: dict[str, dict[str, object]] | None = None,
    ) -> tuple[
        bool,
        set[tuple[str, ...]],
        tuple[bool, bool, frozenset[tuple[str, ...]]],
        bool,
        bool,
    ]:
        dependency = tuple[bool, bool, frozenset[tuple[str, ...]]]
        stack_slot = dependency | None
        analysis_state = tuple[tuple[stack_slot, ...], dict[str, dependency]]
        called_globals: set[tuple[str, ...]] = set()
        return_dependencies: list[dependency] = []
        found_identity = False
        call_graph_incomplete = False
        mutates_environment = False
        active = set() if active is None else active
        dependency_scopes = {} if dependency_scopes is None else dependency_scopes
        scope_key = str(id(globals_scope))
        dependency_scopes[scope_key] = globals_scope
        if id(code) in active:
            return False, set(), (False, True, frozenset()), True, False
        active.add(id(code))

        def combine(
            values: Sequence[stack_slot],
        ) -> tuple[bool, bool, frozenset[tuple[str, ...]]]:
            present = [value for value in values if value is not None]
            return (
                any(value[0] for value in present),
                any(value[1] for value in present),
                frozenset(path for value in present for path in value[2]),
            )

        def merge_dependency(left: stack_slot, right: stack_slot) -> stack_slot:
            if left is None:
                return right
            if right is None:
                return left
            return combine((left, right))

        def merge_state(
            current: analysis_state | None, incoming: analysis_state
        ) -> tuple[analysis_state, bool]:
            if current is None:
                return incoming, True
            current_stack, current_locals = current
            incoming_stack, incoming_locals = incoming
            if len(current_stack) == len(incoming_stack):
                merged_stack = tuple(
                    merge_dependency(left, right)
                    for left, right in zip(current_stack, incoming_stack, strict=True)
                )
            else:
                merged = combine((*current_stack, *incoming_stack))
                merged_stack = (merged,) * max(len(current_stack), len(incoming_stack))
            merged_locals = {
                name: combine(
                    tuple(
                        value
                        for scope in (current_locals, incoming_locals)
                        if (value := scope.get(name)) is not None
                    )
                )
                for name in current_locals.keys() | incoming_locals.keys()
            }
            merged_state = merged_stack, merged_locals
            return merged_state, merged_state != current

        def environment_dependency(
            path: tuple[str, ...],
        ) -> tuple[bool, bool, frozenset[tuple[str, ...]]]:
            return False, True, frozenset((path,))

        def is_synthetic_path(path: tuple[str, ...]) -> bool:
            return path[0].startswith("<") and path[0] != "<scope>"

        def scoped_path(path: tuple[str, ...]) -> tuple[str, ...]:
            return ("<scope>", scope_key, *path)

        def unscoped_path(path: tuple[str, ...]) -> tuple[str, ...]:
            return path[2:] if path[0] == "<scope>" else path

        def resolve_environment_path(path: tuple[str, ...]) -> tuple[object, bool]:
            if path[0] == "<environment-object>" and len(path) >= 2:
                try:
                    value = environment_objects[int(path[1])]
                    for attribute in path[2:]:
                        value = inspect.getattr_static(value, attribute)
                except (AttributeError, KeyError, ValueError):
                    return None, False
                return value, True
            if is_synthetic_path(path):
                return None, False
            resolved_value, _, resolved = resolve_called(path)
            return resolved_value, resolved

        def has_identity_unstable_environment(value: dependency) -> bool:
            if not value[1]:
                return False
            saw_source = False
            for path in value[2]:
                if (
                    path[0] == "<identity-unstable>"
                    or path[0] == "<identity-sensitive>"
                ):
                    return True
                if path[0] == "<singleton>":
                    saw_source = True
                    continue
                resolved_value, resolved = resolve_environment_path(path)
                if not resolved:
                    continue
                saw_source = True
                if not is_stable_singleton(resolved_value):
                    return True
            return not saw_source

        def has_identity_sensitive_environment(value: dependency) -> bool:
            if not value[1]:
                return False
            saw_source = False
            for path in value[2]:
                if path[0] == "<identity-sensitive>":
                    return True
                if path[0] in ("<identity-unstable>", "<singleton>"):
                    saw_source = True
                    continue
                resolved_value, resolved = resolve_environment_path(path)
                if not resolved:
                    continue
                saw_source = True
                if not is_safely_reflexive(resolved_value):
                    return True
            return not saw_source

        def has_lookup_identity_sensitive_environment(value: dependency) -> bool:
            if not value[1]:
                return False
            saw_source = False
            for path in value[2]:
                if path[0] == "<lookup-identity-sensitive>":
                    return True
                if path[0] in (
                    "<identity-sensitive>",
                    "<identity-unstable>",
                    "<singleton>",
                ):
                    continue
                resolved_value, resolved = resolve_environment_path(path)
                if not resolved:
                    continue
                saw_source = True
                if not is_safely_reflexive_container_lookup(resolved_value):
                    return True
            return not saw_source

        def has_mapping_key_identity_sensitive_environment(
            value: dependency,
        ) -> bool:
            if not value[1]:
                return False
            saw_source = False
            for path in value[2]:
                if path[0] == "<mapping-key-identity-sensitive>":
                    return True
                if is_synthetic_path(path):
                    continue
                resolved_value, resolved = resolve_environment_path(path)
                if not resolved:
                    continue
                saw_source = True
                if is_identity_mapping(resolved_value) and not all(
                    is_safely_reflexive(item)
                    for item in cast("Mapping[object, object]", resolved_value)
                ):
                    return True
            return not saw_source

        def has_input_container(value: dependency) -> bool:
            return value[0] and any(
                path[0] in ("<input-container>", "<input-container-content>")
                for path in value[2]
            )

        def has_input_mapping(value: dependency) -> bool:
            return value[0] and any(path[0] == "<input-mapping>" for path in value[2])

        def has_identity_sensitive_input_container(value: dependency) -> bool:
            return value[0] and any(
                path[0] == "<input-identity-sensitive-container>" for path in value[2]
            )

        def has_lookup_identity_sensitive_input_container(
            value: dependency,
        ) -> bool:
            return value[0] and any(
                path[0] == "<input-lookup-identity-sensitive-container>"
                for path in value[2]
            )

        def has_identity_sensitive_input_mapping_keys(
            value: dependency,
        ) -> bool:
            return value[0] and any(
                path[0] == "<input-mapping-key-identity-sensitive>" for path in value[2]
            )

        def has_environment_container(value: dependency) -> bool:
            for path in value[2]:
                if path[0] == "<container>":
                    return True
                resolved_value, resolved = resolve_environment_path(path)
                if resolved and isinstance(
                    resolved_value,
                    (dict, MappingProxyType, deque, list, tuple, set, frozenset),
                ):
                    return True
            return False

        def has_environment_mapping(value: dependency) -> bool:
            for path in value[2]:
                if path[0] == "<mapping>":
                    return True
                resolved_value, resolved = resolve_environment_path(path)
                if resolved and is_identity_mapping(resolved_value):
                    return True
            return False

        def has_mutable_environment_reference(value: dependency) -> bool:
            if not value[1]:
                return False
            for path in value[2]:
                if path[0] == "<mutable-environment-object>":
                    return True
                resolved_value, resolved = resolve_environment_path(path)
                if resolved and is_potentially_mutable(resolved_value):
                    return True
            return False

        def environment_values_satisfy(
            value: dependency, predicate: Callable[[object], bool]
        ) -> bool:
            if not value[1]:
                return False
            saw_value = False
            for path in value[2]:
                if path[0] == "<environment>":
                    return False
                resolved_value, resolved = resolve_environment_path(path)
                if not resolved:
                    if not is_synthetic_path(path):
                        return False
                    continue
                saw_value = True
                if not predicate(resolved_value):
                    return False
            return saw_value

        def has_unsafe_environment_behavior(value: dependency) -> bool:
            return has_mutable_environment_reference(
                value
            ) and not environment_values_satisfy(
                value,
                lambda item: isinstance(item, torch.Tensor)
                or type(item)
                in (dict, MappingProxyType, deque, list, set, frozenset, tuple),
            )

        def may_have_unguarded_input_alias(left: dependency, right: dependency) -> bool:
            if not left[0] or not right[0]:
                return False
            left_types = {path[1:] for path in left[2] if path[0] == "<input-type>"}
            right_types = {path[1:] for path in right[2] if path[0] == "<input-type>"}
            if not left_types or not right_types:
                return True
            return any(
                left_type == right_type and left_type[0] != "tensor"
                for left_type in left_types
                for right_type in right_types
            )

        def has_environment_reference(value: dependency) -> bool:
            return value[1] and any(
                path[0]
                in (
                    "<environment>",
                    "<environment-object>",
                    "<mutable-environment-object>",
                )
                or not is_synthetic_path(path)
                for path in value[2]
            )

        def extend_path(path: tuple[str, ...], attribute: str) -> tuple[str, ...]:
            if path[0] in (
                "<environment-object>",
                "<mutable-environment-object>",
            ):
                return (*path, attribute)
            if is_synthetic_path(path):
                return path
            if len(path) >= 8:
                return ("<environment>",)
            return (*path, attribute)

        def input_method_dependency(value: dependency, name: str) -> dependency:
            paths: set[tuple[str, ...]] = {
                ("<input-method>", path[1], name)
                for path in value[2]
                if path[0] == "<input-object>"
            }
            if value[0]:
                paths.add(("<input>", name))
            return value[0], False, frozenset(paths)

        def environment_method_dependency(value: dependency, name: str) -> dependency:
            return (
                False,
                value[1],
                frozenset(
                    extend_path(path, name)
                    for path in value[2]
                    if path[0] == "<environment-object>" or not is_synthetic_path(path)
                ),
            )

        def resolve_called(
            path: tuple[str, ...],
        ) -> tuple[object | None, dependency | None, bool]:
            if path[0] == "<input-method>":
                receiver_value = input_objects.get(int(path[1]))
                if receiver_value is None:
                    return None, None, False
                try:
                    value = inspect.getattr_static(type(receiver_value), path[2])
                except AttributeError:
                    return None, None, False
                if isinstance(value, (classmethod, staticmethod)):
                    value = value.__func__
                receiver_markers: set[tuple[str, ...]] = {
                    ("<input-object>", path[1]),
                    (
                        "<input-type>",
                        "tensor"
                        if isinstance(receiver_value, torch.Tensor)
                        else "value",
                        type(receiver_value).__module__,
                        type(receiver_value).__qualname__,
                    ),
                }
                if isinstance(
                    receiver_value,
                    (dict, MappingProxyType, deque, list, tuple, set, frozenset),
                ):
                    receiver_markers.add(("<input-container>",))
                    if not is_safely_reflexive(receiver_value):
                        receiver_markers.add(("<input-identity-sensitive-container>",))
                    if not is_safely_reflexive_container_lookup(receiver_value):
                        receiver_markers.add(
                            ("<input-lookup-identity-sensitive-container>",)
                        )
                if is_identity_mapping(receiver_value):
                    receiver_markers.add(("<input-mapping>",))
                    if not all(
                        is_safely_reflexive(item)
                        for item in cast("Mapping[object, object]", receiver_value)
                    ):
                        receiver_markers.add(
                            ("<input-mapping-key-identity-sensitive>",)
                        )
                receiver_dependency = (
                    True,
                    False,
                    frozenset(receiver_markers),
                )
                return value, receiver_dependency, True
            if path[0] == "<environment-object>" and len(path) >= 2:
                try:
                    value = environment_objects[int(path[1])]
                except (KeyError, ValueError):
                    return None, None, False
                owner = None
                try:
                    for index, attribute in enumerate(path[2:], 2):
                        owner = value
                        value = inspect.getattr_static(owner, attribute)
                        if isinstance(value, staticmethod):
                            value = value.__func__
                        elif isinstance(value, classmethod):
                            value = value.__func__
                            if index == len(path) - 1:
                                return (
                                    value,
                                    environment_dependency(path[:-1]),
                                    True,
                                )
                        elif (
                            index == len(path) - 1
                            and isinstance(value, types.FunctionType)
                            and not isinstance(owner, (type, types.ModuleType))
                        ):
                            return (
                                value,
                                environment_dependency(path[:-1]),
                                True,
                            )
                except AttributeError:
                    return None, None, False
                if isinstance(value, types.MethodType):
                    return value.__func__, environment_dependency(path), True
                if (
                    owner is not None
                    and callable(value)
                    and not isinstance(owner, (type, types.ModuleType))
                ):
                    return value, environment_dependency(path[:-1]), True
                return value, None, True
            path_globals = globals_scope
            if path[0] == "<scope>":
                path_globals = dependency_scopes.get(path[1], {})
                path = path[2:]
                if not path:
                    return None, None, False
            if len(path) == 2 and path[0] == "super":
                owner: object | None = None
                qualname = getattr(code, "co_qualname", None)
                if isinstance(qualname, str):
                    owner = path_globals
                    try:
                        for name in qualname.split(".")[:-1]:
                            if name == "<locals>":
                                return None, None, False
                            owner = (
                                owner[name]
                                if isinstance(owner, dict)
                                else inspect.getattr_static(owner, name)
                            )
                    except (AttributeError, KeyError):
                        return None, None, False
                else:
                    for candidate in path_globals.values():
                        if not isinstance(candidate, type):
                            continue
                        method = inspect.getattr_static(candidate, code.co_name, None)
                        method = getattr(method, "__func__", method)
                        if isinstance(method, types.FunctionType) and (
                            method.__code__ is code
                        ):
                            owner = candidate
                            break
                if isinstance(owner, type):
                    for base in owner.__mro__[1:]:
                        called = inspect.getattr_static(base, path[1], None)
                        if isinstance(called, (classmethod, staticmethod)):
                            called = called.__func__
                        if called is not None:
                            return called, environment_dependency(path), True
                return None, None, False
            if path[0] in path_globals:
                value = path_globals[path[0]]
            else:
                builtins_scope = path_globals.get("__builtins__", {})
                if isinstance(builtins_scope, types.ModuleType):
                    builtins_scope = vars(builtins_scope)
                elif not isinstance(builtins_scope, Mapping):
                    builtins_scope = {}
                if path[0] in builtins_scope:
                    value = builtins_scope[path[0]]
                else:
                    try:
                        value = importlib.import_module(path[0])
                    except ImportError:
                        return None, None, False
            owner = None
            try:
                for index, attribute in enumerate(path[1:], 1):
                    owner = value
                    value = inspect.getattr_static(owner, attribute)
                    if isinstance(value, staticmethod):
                        value = value.__func__
                    elif isinstance(value, classmethod):
                        value = value.__func__
                        if index == len(path) - 1:
                            return value, environment_dependency(path[:-1]), True
                    elif (
                        index == len(path) - 1
                        and isinstance(value, types.FunctionType)
                        and not isinstance(owner, (type, types.ModuleType))
                    ):
                        return value, environment_dependency(path[:-1]), True
            except AttributeError:
                return None, None, False
            if isinstance(value, types.MethodType):
                return value.__func__, environment_dependency(path), True
            if (
                owner is not None
                and callable(value)
                and not isinstance(owner, (type, types.ModuleType))
            ):
                return value, environment_dependency(path[:-1]), True
            return value, None, True

        def python_callables(
            called: object | None,
            path: tuple[str, ...],
            receiver: dependency | None,
            seen: set[int] | None = None,
        ) -> list[
            tuple[
                types.FunctionType,
                dependency | None,
                tuple[object, ...],
                dict[str, object],
            ]
        ]:
            seen = set() if seen is None else seen
            if id(called) in seen:
                return []
            seen.add(id(called))
            if isinstance(called, (functools.partial, functools.partialmethod)):
                function = called.func
                if isinstance(function, types.MethodType):
                    function = function.__func__
                    receiver = environment_dependency(path)
                if isinstance(function, types.FunctionType):
                    return [
                        (
                            function,
                            receiver,
                            called.args,
                            called.keywords or {},
                        )
                    ]
                called = function
            if isinstance(called, types.MethodType):
                return [(called.__func__, environment_dependency(path), (), {})]
            if isinstance(called, types.FunctionType):
                return [(called, receiver, (), {})]
            if isinstance(called, type):
                constructors: list[
                    tuple[
                        types.FunctionType,
                        dependency | None,
                        tuple[object, ...],
                        dict[str, object],
                    ]
                ] = []
                metaclass_call = inspect.getattr_static(type(called), "__call__", None)
                if isinstance(metaclass_call, types.FunctionType):
                    constructors.append(
                        (metaclass_call, environment_dependency(path), (), {})
                    )
                for name in ("__new__", "__init__"):
                    constructor = inspect.getattr_static(called, name, None)
                    if isinstance(constructor, (classmethod, staticmethod)):
                        constructor = constructor.__func__
                    if isinstance(constructor, types.FunctionType):
                        constructors.append(
                            (
                                constructor,
                                environment_dependency(path)
                                if name == "__new__"
                                else (False, False, frozenset()),
                                (),
                                {},
                            )
                        )
                return constructors
            if callable(called) and not isinstance(called, types.ModuleType):
                call = inspect.getattr_static(type(called), "__call__", None)
                if isinstance(call, types.FunctionType):
                    return [(call, environment_dependency(path), (), {})]
            if isinstance(called, (dict, MappingProxyType)):
                values = list(called.values())
            elif isinstance(called, (tuple, list, set, frozenset)):
                values = list(called)
            else:
                values = []
            if values:
                candidates = []
                for value in values:
                    candidates.extend(python_callables(value, path, None, seen))
                return candidates
            return []

        def bind_call_dependencies(
            function: types.FunctionType,
            args: Sequence[stack_slot],
            keyword_names: tuple[str, ...],
            receiver: dependency | None,
            bound_positional: tuple[object, ...] = (),
            bound_keywords: Mapping[str, object] | None = None,
        ) -> dict[str, dependency]:
            function_code = function.__code__
            positional_names = list(
                function_code.co_varnames[: function_code.co_argcount]
            )
            kwonly_start = function_code.co_argcount
            kwonly_end = kwonly_start + function_code.co_kwonlyargcount
            kwonly_names = list(function_code.co_varnames[kwonly_start:kwonly_end])
            bound: dict[str, dependency] = {}
            positional_index = 0
            if receiver is not None and positional_names:
                bound[positional_names[0]] = receiver
                positional_index = 1
            for name, value in zip(
                positional_names[positional_index:], bound_positional
            ):
                bound[name] = environment_value_dependency(value)
            positional_index += len(bound_positional)
            positional_count = len(args) - len(keyword_names)
            positional_values = args[:positional_count]
            bound.update(
                {
                    name: value
                    for name, value in zip(
                        positional_names[positional_index:], positional_values
                    )
                    if value is not None
                }
            )
            remaining = positional_values[
                max(len(positional_names) - positional_index, 0) :
            ]
            next_name = kwonly_end
            if function_code.co_flags & inspect.CO_VARARGS:
                bound[function_code.co_varnames[next_name]] = combine(remaining)
                next_name += 1
            extra_keywords = []
            for name, value in zip(keyword_names, args[positional_count:]):
                if value is None:
                    continue
                if name in positional_names or name in kwonly_names:
                    bound[name] = value
                else:
                    extra_keywords.append(value)
            if function_code.co_flags & inspect.CO_VARKEYWORDS:
                bound[function_code.co_varnames[next_name]] = combine(extra_keywords)
            for name, value in (bound_keywords or {}).items():
                if name in positional_names or name in kwonly_names:
                    bound.setdefault(name, environment_value_dependency(value))
            defaults = function.__defaults__ or ()
            for name, value in zip(
                positional_names[len(positional_names) - len(defaults) :], defaults
            ):
                bound.setdefault(name, environment_value_dependency(value))
            for name, value in (function.__kwdefaults__ or {}).items():
                bound.setdefault(name, environment_value_dependency(value))
            return bound

        def bind_local_function_dependencies(
            function_code: CodeType,
            args: Sequence[stack_slot],
            keyword_names: tuple[str, ...],
            captured: dependency,
        ) -> dict[str, dependency]:
            positional_names = list(
                function_code.co_varnames[: function_code.co_argcount]
            )
            kwonly_start = function_code.co_argcount
            kwonly_end = kwonly_start + function_code.co_kwonlyargcount
            kwonly_names = list(function_code.co_varnames[kwonly_start:kwonly_end])
            positional_count = len(args) - len(keyword_names)
            bound = {
                name: value
                for name, value in zip(positional_names, args[:positional_count])
                if value is not None
            }
            remaining = args[positional_count:]
            parameter_names = positional_names + kwonly_names
            bound.update(
                {
                    name: value
                    for name, value in zip(keyword_names, remaining)
                    if value is not None and name in parameter_names
                }
            )
            next_name = kwonly_end
            if function_code.co_flags & inspect.CO_VARARGS:
                bound[function_code.co_varnames[next_name]] = combine(
                    args[len(positional_names) : positional_count]
                )
                next_name += 1
            if function_code.co_flags & inspect.CO_VARKEYWORDS:
                bound[function_code.co_varnames[next_name]] = combine(
                    tuple(
                        value
                        for name, value in zip(keyword_names, remaining)
                        if value is not None and name not in parameter_names
                    )
                )
            for name in (*positional_names, *kwonly_names, *function_code.co_freevars):
                bound.setdefault(name, captured)
            return bound

        def keyword_names_for_call(index: int) -> tuple[str, ...]:
            for previous in reversed(instructions[max(0, index - 4) : index]):
                if previous.opname == "KW_NAMES" and previous.arg is not None:
                    names = code.co_consts[previous.arg]
                elif previous.opname == "LOAD_CONST":
                    names = previous.argval
                elif previous.opname not in ("CACHE", "EXTENDED_ARG", "PRECALL"):
                    break
                else:
                    continue
                if isinstance(names, tuple) and all(
                    isinstance(name, str) for name in names
                ):
                    return names
                break
            return ()

        inplace_operator_names = (
            "iadd",
            "iand",
            "iconcat",
            "ifloordiv",
            "ilshift",
            "imatmul",
            "imod",
            "imul",
            "ior",
            "ipow",
            "irshift",
            "isub",
            "itruediv",
            "ixor",
        )
        mutating_method_names = {
            "__delattr__",
            "__delitem__",
            "__setattr__",
            "__setitem__",
            "add",
            "append",
            "appendleft",
            "clear",
            "difference_update",
            "discard",
            "extend",
            "extendleft",
            "insert",
            "intersection_update",
            "pop",
            "popitem",
            "remove",
            "reverse",
            "rotate",
            "setdefault",
            "sort",
            "symmetric_difference_update",
            "update",
            *(f"__{name}__" for name in inplace_operator_names),
        }
        identity_method_names = {
            "__contains__",
            "__eq__",
            "__getitem__",
            "__ne__",
            "count",
            "get",
            "index",
        }
        inplace_operators = tuple(
            candidate
            for name in inplace_operator_names
            if (candidate := getattr(operator, name, None)) is not None
        )
        binary_operator_methods = {
            "+": ("__add__", "__radd__"),
            "-": ("__sub__", "__rsub__"),
            "*": ("__mul__", "__rmul__"),
            "@": ("__matmul__", "__rmatmul__"),
            "/": ("__truediv__", "__rtruediv__"),
            "//": ("__floordiv__", "__rfloordiv__"),
            "%": ("__mod__", "__rmod__"),
            "**": ("__pow__", "__rpow__"),
            "<<": ("__lshift__", "__rlshift__"),
            ">>": ("__rshift__", "__rrshift__"),
            "&": ("__and__", "__rand__"),
            "^": ("__xor__", "__rxor__"),
            "|": ("__or__", "__ror__"),
            "BINARY_ADD": ("__add__", "__radd__"),
            "BINARY_SUBTRACT": ("__sub__", "__rsub__"),
            "BINARY_MULTIPLY": ("__mul__", "__rmul__"),
            "BINARY_MATRIX_MULTIPLY": ("__matmul__", "__rmatmul__"),
            "BINARY_TRUE_DIVIDE": ("__truediv__", "__rtruediv__"),
            "BINARY_FLOOR_DIVIDE": ("__floordiv__", "__rfloordiv__"),
            "BINARY_MODULO": ("__mod__", "__rmod__"),
            "BINARY_POWER": ("__pow__", "__rpow__"),
            "BINARY_LSHIFT": ("__lshift__", "__rlshift__"),
            "BINARY_RSHIFT": ("__rshift__", "__rrshift__"),
            "BINARY_AND": ("__and__", "__rand__"),
            "BINARY_XOR": ("__xor__", "__rxor__"),
            "BINARY_OR": ("__or__", "__ror__"),
        }
        readonly_native_methods: Mapping[type[object], frozenset[str]] = {
            contextvars.ContextVar: frozenset({"get"}),
            object: frozenset({"__getattribute__"}),
            dict: frozenset(
                {
                    "__contains__",
                    "__eq__",
                    "__getitem__",
                    "__iter__",
                    "__len__",
                    "__ne__",
                    "__or__",
                    "__reversed__",
                    "copy",
                    "get",
                    "items",
                    "keys",
                    "values",
                }
            ),
            list: frozenset(
                {
                    "__add__",
                    "__contains__",
                    "__eq__",
                    "__getitem__",
                    "__iter__",
                    "__len__",
                    "__mul__",
                    "__ne__",
                    "__reversed__",
                    "__rmul__",
                    "copy",
                    "count",
                    "index",
                }
            ),
            deque: frozenset(
                {
                    "__add__",
                    "__contains__",
                    "__eq__",
                    "__getitem__",
                    "__iter__",
                    "__len__",
                    "__mul__",
                    "__ne__",
                    "__reversed__",
                    "__rmul__",
                    "copy",
                    "count",
                    "index",
                }
            ),
            tuple: frozenset(
                {
                    "__add__",
                    "__contains__",
                    "__eq__",
                    "__getitem__",
                    "__iter__",
                    "__len__",
                    "__mul__",
                    "__ne__",
                    "__rmul__",
                    "count",
                    "index",
                }
            ),
            MappingProxyType: frozenset(
                {
                    "__contains__",
                    "__eq__",
                    "__getitem__",
                    "__iter__",
                    "__len__",
                    "__or__",
                    "__reversed__",
                    "copy",
                    "get",
                    "items",
                    "keys",
                    "values",
                }
            ),
            set: frozenset(
                {
                    "__and__",
                    "__contains__",
                    "__eq__",
                    "__iter__",
                    "__len__",
                    "__ne__",
                    "__or__",
                    "__sub__",
                    "__xor__",
                    "copy",
                    "difference",
                    "intersection",
                    "isdisjoint",
                    "issubset",
                    "issuperset",
                    "symmetric_difference",
                    "union",
                }
            ),
        }
        protocol_consuming_builtins = {
            abs,
            all,
            any,
            ascii,
            bin,
            bool,
            bytes,
            complex,
            dict,
            dir,
            float,
            format,
            frozenset,
            hash,
            hex,
            int,
            iter,
            len,
            list,
            max,
            min,
            next,
            oct,
            print,
            repr,
            reversed,
            round,
            set,
            sorted,
            str,
            sum,
            tuple,
        }
        native_input_only_callables = {
            all,
            any,
            filter,
            functools.reduce,
            map,
        }
        environment_retaining_native_callables = (
            classmethod,
            functools.partial,
            functools.partialmethod,
            memoryview,
            property,
            slice,
            staticmethod,
            MappingProxyType,
        )
        safe_native_free_callables = {
            callable,
            getattr,
            hasattr,
            id,
            isinstance,
            issubclass,
            operator.contains,
            operator.countOf,
            operator.eq,
            operator.getitem,
            operator.indexOf,
            operator.is_,
            operator.is_not,
            operator.ne,
        }
        native_argument_protocol_methods: Mapping[type[object], frozenset[str]] = {
            dict: frozenset({"__contains__", "__eq__", "__getitem__", "__ne__", "get"}),
            MappingProxyType: frozenset(
                {"__contains__", "__eq__", "__getitem__", "get"}
            ),
            list: frozenset(
                {
                    "__contains__",
                    "__eq__",
                    "__getitem__",
                    "__mul__",
                    "__ne__",
                    "__rmul__",
                    "count",
                    "index",
                }
            ),
            deque: frozenset(
                {
                    "__contains__",
                    "__eq__",
                    "__getitem__",
                    "__mul__",
                    "__ne__",
                    "__rmul__",
                    "count",
                    "index",
                }
            ),
            tuple: frozenset(
                {
                    "__contains__",
                    "__eq__",
                    "__getitem__",
                    "__mul__",
                    "__ne__",
                    "__rmul__",
                    "count",
                    "index",
                }
            ),
            set: readonly_native_methods[set] - {"__iter__", "__len__", "copy"},
        }

        def has_unverified_protocol_value(
            value: object, seen: set[int] | None = None
        ) -> bool:
            if isinstance(value, torch.Tensor) or not is_potentially_mutable(value):
                return False
            if isinstance(
                value,
                (
                    functools.partial,
                    functools.partialmethod,
                    types.FunctionType,
                    types.MethodType,
                ),
            ):
                return False
            seen = set() if seen is None else seen
            if id(value) in seen:
                return False
            seen.add(id(value))
            if type(value) in (dict, MappingProxyType):
                return any(
                    has_unverified_protocol_value(item, seen)
                    for pair in cast("Mapping[object, object]", value).items()
                    for item in pair
                )
            if type(value) in (deque, list, tuple, set, frozenset):
                return any(
                    has_unverified_protocol_value(item, seen)
                    for item in cast("Iterable[object]", value)
                )
            return True

        def has_unsafe_native_arguments(value: dependency) -> bool:
            return has_mutable_environment_reference(
                value
            ) and not environment_values_satisfy(
                value, lambda item: not has_unverified_protocol_value(item)
            )

        def environment_container_contents_dependency(
            value: dependency, kind: str
        ) -> dependency | None:
            contents = []
            seen = set()
            for path in value[2]:
                container, resolved = resolve_environment_path(path)
                if not resolved or id(container) in seen:
                    continue
                seen.add(id(container))
                if type(container) in (dict, MappingProxyType):
                    mapping = cast("Mapping[object, object]", container)
                    if kind in ("iteration", "keys"):
                        selected = mapping.keys()
                    elif kind == "items":
                        selected = (item for pair in mapping.items() for item in pair)
                    else:
                        selected = mapping.values()
                elif type(container) in (deque, list, tuple, set, frozenset):
                    selected = cast("Iterable[object]", container)
                else:
                    continue
                contents.extend(environment_value_dependency(item) for item in selected)
            if not contents:
                return None
            selected = combine(contents)
            input_paths = frozenset(
                path for path in value[2] if path[0].startswith("<input")
            )
            return (
                value[0],
                selected[1],
                selected[2]
                | input_paths
                | (frozenset((("<input-value>",),)) if value[0] else frozenset()),
            )

        def native_method_result_dependency(
            called: object, receiver: dependency
        ) -> dependency | None:
            name = getattr(called, "__name__", None)
            if name in ("__getitem__", "get", "values"):
                result = environment_container_contents_dependency(receiver, "values")
                if result is not None or name != "get":
                    return result
                context_values = []
                for path in receiver[2]:
                    context_var, resolved = resolve_environment_path(path)
                    if not resolved or not isinstance(
                        context_var, contextvars.ContextVar
                    ):
                        continue
                    try:
                        context_values.append(
                            environment_value_dependency(context_var.get())
                        )
                    except LookupError:
                        pass
                return combine(context_values) if context_values else None
            if name == "__iter__":
                return environment_container_contents_dependency(receiver, "iteration")
            if name == "keys":
                return environment_container_contents_dependency(receiver, "keys")
            if name == "items":
                return environment_container_contents_dependency(receiver, "items")
            if name == "copy":
                return receiver
            return None

        def native_method_uses_argument_protocol(
            called: object, receiver: dependency
        ) -> bool:
            name = getattr(called, "__name__", None)
            if not isinstance(name, str):
                return False
            owner = getattr(called, "__objclass__", None)

            def uses_protocol(value: object) -> bool:
                methods = native_argument_protocol_methods.get(type(value))
                if (
                    methods is None
                    and isinstance(owner, type)
                    and isinstance(value, owner)
                ):
                    methods = native_argument_protocol_methods.get(owner)
                return methods is not None and name in methods

            return environment_values_satisfy(receiver, uses_protocol)

        def native_method_protocol_values(value: object, name: str) -> Iterable[object]:
            if type(value) in (dict, MappingProxyType):
                mapping = cast("Mapping[object, object]", value)
                if name in ("__contains__", "__getitem__", "get"):
                    return mapping.keys()
                if name in ("__eq__", "__ne__"):
                    return (item for pair in mapping.items() for item in pair)
            elif type(value) in (deque, list, tuple) and name in (
                "__contains__",
                "__eq__",
                "__ne__",
                "count",
                "index",
            ):
                return cast("Iterable[object]", value)
            elif type(value) is set and name not in ("__iter__", "__len__", "copy"):
                return cast("Iterable[object]", value)
            return ()

        def is_readonly_native_method(called: object, receiver: dependency) -> bool:
            name = getattr(called, "__name__", None)
            if not isinstance(name, str):
                return False

            def is_readonly(value: object) -> bool:
                if isinstance(value, torch.Tensor):
                    return True
                allowed = readonly_native_methods.get(type(value), frozenset())
                owner = getattr(called, "__objclass__", None)
                if (
                    name not in allowed
                    and isinstance(owner, type)
                    and isinstance(value, owner)
                ):
                    allowed = readonly_native_methods.get(owner, frozenset())
                if name not in allowed:
                    return False
                return not any(
                    has_unverified_protocol_value(item)
                    for item in native_method_protocol_values(value, name)
                )

            return environment_values_satisfy(
                receiver,
                is_readonly,
            )

        def record_mutation() -> None:
            nonlocal mutates_environment
            mutates_environment = True

        def analyze_binary_behavior(
            opname: str,
            argrepr: str,
            values: Sequence[stack_slot],
        ) -> bool:
            pair = dependency_pair(values)
            if pair is None:
                return False
            operator_name = (
                argrepr.removesuffix("=") if opname == "BINARY_OP" else opname
            )
            methods = binary_operator_methods.get(operator_name)
            if methods is None:
                return False
            left, right = pair
            for receiver, argument, method in (
                (left, right, methods[0]),
                (right, left, methods[1]),
            ):
                analyze_call(
                    (
                        combine(
                            (
                                input_method_dependency(receiver, method),
                                environment_method_dependency(receiver, method),
                            )
                        ),
                    ),
                    (argument,),
                )
            return True

        def analyze_call(
            callable_values: Sequence[stack_slot],
            args: Sequence[stack_slot],
            keyword_names: tuple[str, ...] = (),
            *,
            packed: bool = False,
        ) -> dependency:
            nonlocal call_graph_incomplete, found_identity, mutates_environment
            callable_dependency = combine(callable_values)
            called_globals.update(
                unscoped_path(path)
                for path in callable_dependency[2]
                if not is_synthetic_path(path)
            )
            args_dependency = combine(args)
            call_inputs = args_dependency[0] or callable_dependency[0]
            result = (
                call_inputs,
                args_dependency[1],
                frozenset(
                    path
                    for path in args_dependency[2]
                    if path[0]
                    not in (
                        "<container>",
                        "<input-container>",
                        "<input-container-content>",
                    )
                )
                | (frozenset((("<input-value>",),)) if call_inputs else frozenset()),
            )
            nested_results = []
            native_results = []
            callback_results = []
            retains_unverified_environment = False
            for path in callable_dependency[2]:
                if path[0] == "<local-function>":
                    try:
                        function_id = int(path[1])
                        local_code = local_function_codes[function_id]
                    except (IndexError, KeyError, ValueError):
                        call_graph_incomplete = True
                        continue
                    captured = local_function_payloads.get(
                        function_id, (False, False, frozenset())
                    )
                    (
                        nested_identity,
                        nested_calls,
                        nested_result,
                        nested_incomplete,
                        nested_mutates_environment,
                    ) = bytecode_dependencies(
                        local_code,
                        globals_scope,
                        bind_local_function_dependencies(
                            local_code, args, keyword_names, captured
                        ),
                        active,
                        dependency_scopes,
                    )
                    found_identity |= nested_identity
                    call_graph_incomplete |= nested_incomplete
                    mutates_environment |= nested_mutates_environment
                    called_globals.update(nested_calls)
                    nested_results.append(nested_result)
                    continue
                if path[0] == "<input>":
                    method_name = path[-1]
                    input_container = (
                        has_input_mapping(callable_dependency)
                        if method_name in ("__getitem__", "get")
                        else has_input_container(callable_dependency)
                    )
                    input_container_sensitive = (
                        has_identity_sensitive_input_mapping_keys(callable_dependency)
                        if method_name in ("__getitem__", "get")
                        else (
                            has_identity_sensitive_input_container(callable_dependency)
                            if method_name in ("__eq__", "__ne__")
                            else has_lookup_identity_sensitive_input_container(
                                callable_dependency
                            )
                        )
                    )
                    if (
                        method_name in identity_method_names
                        and input_container
                        and (
                            has_identity_sensitive_environment(args_dependency)
                            or (args_dependency[0] and input_container_sensitive)
                        )
                    ):
                        found_identity = True
                    continue
                if path[0] == "<mutable-environment-object>":
                    if (
                        len(path) > 1
                        and path[-1] in mutating_method_names
                        and has_mutable_environment_reference(callable_dependency)
                    ):
                        record_mutation()
                    continue
                if is_synthetic_path(path) and path[0] not in (
                    "<environment-object>",
                    "<input-method>",
                ):
                    continue
                called, receiver, resolved = resolve_called(path)
                if called is super:
                    result = environment_dependency(("super",))
                    continue
                if path[-1] in identity_method_names and receiver is not None:
                    method_name = path[-1]
                    input_receiver = (
                        has_input_mapping(receiver)
                        if method_name in ("__getitem__", "get")
                        else has_input_container(receiver)
                    )
                    input_receiver_sensitive = (
                        has_identity_sensitive_input_mapping_keys(receiver)
                        if method_name in ("__getitem__", "get")
                        else (
                            has_identity_sensitive_input_container(receiver)
                            if method_name in ("__eq__", "__ne__")
                            else has_lookup_identity_sensitive_input_container(receiver)
                        )
                    )
                    environment_receiver = (
                        has_environment_mapping(receiver)
                        if method_name in ("__getitem__", "get")
                        else has_environment_container(receiver)
                    )
                    environment_receiver_sensitive = (
                        has_mapping_key_identity_sensitive_environment(receiver)
                        if method_name in ("__getitem__", "get")
                        else (
                            has_identity_sensitive_environment(receiver)
                            if method_name in ("__eq__", "__ne__")
                            else has_lookup_identity_sensitive_environment(receiver)
                        )
                    )
                    found_identity |= (
                        input_receiver
                        and (
                            has_identity_sensitive_environment(args_dependency)
                            or (args_dependency[0] and input_receiver_sensitive)
                        )
                    ) or (
                        args_dependency[0]
                        and environment_receiver
                        and environment_receiver_sensitive
                    )
                if (
                    receiver is not None
                    and not isinstance(called, types.FunctionType)
                    and has_mutable_environment_reference(receiver)
                    and path[-1] in mutating_method_names
                ):
                    record_mutation()
                identity_callable = called
                if isinstance(
                    identity_callable, (functools.partial, functools.partialmethod)
                ):
                    identity_callable = identity_callable.func
                if any(
                    called is candidate
                    for candidate in environment_retaining_native_callables
                ) and has_mutable_environment_reference(args_dependency):
                    retains_unverified_environment = True
                native_receiver = receiver
                native_args_dependency = args_dependency
                if (
                    isinstance(called, (functools.partial, functools.partialmethod))
                    and called.args
                ):
                    native_receiver = environment_value_dependency(called.args[0])
                elif (
                    native_receiver is None
                    and inspect.ismethoddescriptor(identity_callable)
                    and args
                ):
                    native_receiver = args[0]
                    native_args_dependency = combine(args[1:])
                elif (
                    native_receiver is None
                    and inspect.isbuiltin(identity_callable)
                    and (bound_self := getattr(identity_callable, "__self__", None))
                    is not None
                    and not isinstance(bound_self, (type, types.ModuleType))
                ):
                    native_receiver = environment_value_dependency(bound_self)
                if (
                    native_receiver is not None
                    and (
                        inspect.ismethoddescriptor(identity_callable)
                        or inspect.isbuiltin(identity_callable)
                    )
                    and has_mutable_environment_reference(native_receiver)
                    and not is_readonly_native_method(
                        identity_callable, native_receiver
                    )
                ):
                    record_mutation()
                if (
                    native_receiver is not None
                    and (
                        result_dependency := native_method_result_dependency(
                            identity_callable, native_receiver
                        )
                    )
                    is not None
                ):
                    native_results.append(result_dependency)
                if (
                    native_receiver is not None
                    and (
                        inspect.ismethoddescriptor(identity_callable)
                        or inspect.isbuiltin(identity_callable)
                    )
                    and has_environment_reference(native_receiver)
                    and native_method_uses_argument_protocol(
                        identity_callable, native_receiver
                    )
                    and has_unsafe_native_arguments(native_args_dependency)
                ):
                    record_mutation()
                dynamic_attribute_analyzed = False
                if called in (getattr, hasattr) and len(args) >= 2:
                    attribute_names = set()
                    if args[1] is not None:
                        for path in args[1][2]:
                            attribute_name, resolved = resolve_environment_path(path)
                            if resolved and isinstance(attribute_name, str):
                                attribute_names.add(attribute_name)
                    if len(attribute_names) == 1 and args[0] is not None:
                        attribute_result = analyze_environment_descriptor(
                            args[0], next(iter(attribute_names))
                        )
                        if attribute_result is not None:
                            native_results.append(attribute_result)
                        dynamic_attribute_analyzed = True
                if (
                    not dynamic_attribute_analyzed
                    and not isinstance(called, types.FunctionType)
                    and callable(called)
                ):
                    if identity_callable in protocol_consuming_builtins:
                        if has_unsafe_native_arguments(args_dependency):
                            record_mutation()
                    elif (
                        native_receiver is None
                        and inspect.isbuiltin(identity_callable)
                        and identity_callable not in safe_native_free_callables
                        and identity_callable not in native_input_only_callables
                        and has_mutable_environment_reference(args_dependency)
                    ):
                        record_mutation()
                    elif (
                        native_receiver is None
                        and not is_library_module(
                            getattr(identity_callable, "__module__", None)
                        )
                        and getattr(identity_callable, "__module__", None) != "builtins"
                        and has_unsafe_native_arguments(
                            native_args_dependency
                            if native_receiver is not None
                            else args_dependency
                        )
                    ):
                        record_mutation()
                if identity_callable in (all, any):
                    native_results.append(
                        (
                            call_inputs,
                            False,
                            frozenset((("<input-value>",),))
                            if call_inputs
                            else frozenset(),
                        )
                    )
                if identity_callable in (operator.delitem, operator.setitem):
                    bound_environment = isinstance(
                        called, (functools.partial, functools.partialmethod)
                    ) and bool(called.args)
                    explicit_receiver = args[0] if args else None
                    if bound_environment or (
                        explicit_receiver is not None
                        and has_environment_reference(explicit_receiver)
                    ):
                        record_mutation()
                if identity_callable in inplace_operators:
                    bound_receiver = (
                        environment_value_dependency(called.args[0])
                        if isinstance(
                            called, (functools.partial, functools.partialmethod)
                        )
                        and called.args
                        else None
                    )
                    explicit_receiver = args[0] if args else None
                    if (
                        bound_receiver is not None
                        and has_mutable_environment_reference(bound_receiver)
                    ) or (
                        explicit_receiver is not None
                        and has_mutable_environment_reference(explicit_receiver)
                    ):
                        record_mutation()
                if called in (delattr, setattr):
                    explicit_receiver = args[0] if args else None
                    if explicit_receiver is not None and has_environment_reference(
                        explicit_receiver
                    ):
                        record_mutation()
                if (
                    getattr(called, "__module__", None) in ("_heapq", "heapq")
                    and getattr(called, "__name__", None)
                    in {
                        "_heapify_max",
                        "_heappop_max",
                        "_heapreplace_max",
                        "heapify",
                        "heappop",
                        "heappush",
                        "heappushpop",
                        "heapreplace",
                    }
                    and args
                    and args[0] is not None
                    and has_mutable_environment_reference(args[0])
                ):
                    record_mutation()
                identity_args = (
                    combine(
                        (
                            args_dependency,
                            *(
                                environment_value_dependency(value)
                                for value in (
                                    *called.args,
                                    *(called.keywords or {}).values(),
                                )
                            ),
                        )
                    )
                    if isinstance(called, (functools.partial, functools.partialmethod))
                    else args_dependency
                )
                if identity_callable in (
                    operator.contains,
                    operator.countOf,
                    operator.eq,
                    operator.getitem,
                    operator.indexOf,
                    operator.is_,
                    operator.is_not,
                    operator.ne,
                ):
                    identity_only = identity_callable in (
                        operator.is_,
                        operator.is_not,
                    )
                    operands = [
                        environment_value_dependency(value)
                        for value in (
                            called.args
                            if isinstance(
                                called,
                                (functools.partial, functools.partialmethod),
                            )
                            else ()
                        )
                    ]
                    operands.extend(value for value in args if value is not None)
                    if len(operands) >= 2:
                        left, right = operands[:2]
                        if identity_callable in (operator.eq, operator.ne):
                            analyze_call(
                                (environment_method_dependency(left, "__eq__"),),
                                (right,),
                            )
                            analyze_call(
                                (environment_method_dependency(right, "__eq__"),),
                                (left,),
                            )
                        elif identity_callable is not operator.is_ and (
                            identity_callable is not operator.is_not
                        ):
                            method_name = (
                                "__getitem__"
                                if identity_callable is operator.getitem
                                else (
                                    "__contains__"
                                    if identity_callable is operator.contains
                                    else (
                                        "count"
                                        if identity_callable is operator.countOf
                                        else "index"
                                    )
                                )
                            )
                            analyze_call(
                                (environment_method_dependency(left, method_name),),
                                (right,),
                            )
                        if identity_only:
                            found_identity |= (
                                (left[0] and has_identity_unstable_environment(right))
                                or (
                                    right[0] and has_identity_unstable_environment(left)
                                )
                                or may_have_unguarded_input_alias(left, right)
                            )
                        elif identity_callable in (operator.eq, operator.ne):
                            found_identity |= (
                                (
                                    left[0]
                                    and has_identity_sensitive_environment(right)
                                    and (
                                        has_input_container(left)
                                        or has_environment_container(right)
                                    )
                                )
                                or (
                                    right[0]
                                    and has_identity_sensitive_environment(left)
                                    and (
                                        has_input_container(right)
                                        or has_environment_container(left)
                                    )
                                )
                                or (
                                    left[0]
                                    and right[0]
                                    and (
                                        has_identity_sensitive_input_container(left)
                                        or has_identity_sensitive_input_container(right)
                                    )
                                )
                            )
                        else:
                            mapping_lookup = identity_callable is operator.getitem
                            input_container = (
                                has_input_mapping(left)
                                if mapping_lookup
                                else has_input_container(left)
                            )
                            environment_container = (
                                has_environment_mapping(left)
                                if mapping_lookup
                                else has_environment_container(left)
                            )
                            environment_container_sensitive = (
                                has_mapping_key_identity_sensitive_environment(left)
                                if mapping_lookup
                                else has_lookup_identity_sensitive_environment(left)
                            )
                            input_container_sensitive = (
                                has_identity_sensitive_input_mapping_keys(left)
                                if mapping_lookup
                                else has_lookup_identity_sensitive_input_container(left)
                            )
                            found_identity |= (
                                (
                                    input_container
                                    and has_identity_sensitive_environment(right)
                                )
                                or (
                                    right[0]
                                    and environment_container
                                    and environment_container_sensitive
                                )
                                or (left[0] and right[0] and input_container_sensitive)
                            )
                    elif identity_only:
                        found_identity |= identity_args[0] and (
                            has_identity_unstable_environment(identity_args)
                            or may_have_unguarded_input_alias(
                                identity_args, identity_args
                            )
                        )
                    continue
                nested_functions = python_callables(called, path, receiver)
                if isinstance(called, type) and isinstance(
                    inspect.getattr_static(type(called), "__call__", None),
                    types.FunctionType,
                ):
                    call_graph_incomplete = True
                if not resolved or (not nested_functions and not callable(called)):
                    call_graph_incomplete = True
                    continue
                if not nested_functions:
                    result = combine(
                        (
                            result,
                            (
                                False,
                                True,
                                frozenset(
                                    (
                                        ("<identity-sensitive>",),
                                        ("<identity-unstable>",),
                                    )
                                ),
                            ),
                        )
                    )
                    for callback_index, callback_dependency in enumerate(args):
                        if callback_dependency is None:
                            continue
                        callback_paths = [
                            callback_path
                            for callback_path in callback_dependency[2]
                            if callback_path[0] == "<local-function>"
                            or not is_synthetic_path(callback_path)
                        ]
                        if not any(
                            callback_path[0] == "<local-function>"
                            or python_callables(
                                resolve_called(callback_path)[0], callback_path, None
                            )
                            for callback_path in callback_paths
                        ):
                            continue
                        callback_args = [
                            argument
                            for index, argument in enumerate(args)
                            if index != callback_index
                        ]
                        callback_args = [
                            environment_container_contents_dependency(
                                argument, "iteration"
                            )
                            or argument
                            for argument in callback_args
                            if argument is not None
                        ]
                        callback_results.append(
                            analyze_call(
                                (callback_dependency,), callback_args, packed=True
                            )
                        )
                    if identity_callable in (map, functools.reduce):
                        native_results.extend(callback_results)
                    elif identity_callable is filter and len(args) >= 2:
                        filtered = args[1]
                        if filtered is not None:
                            native_results.append(
                                environment_container_contents_dependency(
                                    filtered, "iteration"
                                )
                                or filtered
                            )
                for (
                    nested_function,
                    receiver_dependency,
                    bound_positional,
                    bound_keywords,
                ) in nested_functions:
                    if is_library_defined_function(nested_function):
                        continue
                    if packed:
                        nested_arg_count = (
                            nested_function.__code__.co_argcount
                            + nested_function.__code__.co_kwonlyargcount
                        )
                        nested_parameters = dict.fromkeys(
                            nested_function.__code__.co_varnames[:nested_arg_count],
                            args_dependency,
                        )
                        if receiver_dependency is not None and nested_parameters:
                            first = next(iter(nested_parameters))
                            nested_parameters[first] = receiver_dependency
                        parameter_names = list(nested_parameters)
                        offset = int(receiver_dependency is not None)
                        for name, value in zip(
                            parameter_names[offset:], bound_positional
                        ):
                            nested_parameters[name] = environment_value_dependency(
                                value
                            )
                        for name, value in bound_keywords.items():
                            if name in nested_parameters:
                                nested_parameters[name] = environment_value_dependency(
                                    value
                                )
                    else:
                        nested_parameters = bind_call_dependencies(
                            nested_function,
                            args,
                            keyword_names,
                            receiver_dependency,
                            bound_positional,
                            bound_keywords,
                        )
                    (
                        nested_identity,
                        nested_calls,
                        nested_result,
                        nested_incomplete,
                        nested_mutates_environment,
                    ) = bytecode_dependencies(
                        nested_function.__code__,
                        nested_function.__globals__,
                        nested_parameters,
                        active,
                        dependency_scopes,
                    )
                    found_identity |= nested_identity
                    call_graph_incomplete |= nested_incomplete
                    mutates_environment |= nested_mutates_environment
                    called_globals.update(nested_calls)
                    nested_results.append(nested_result)
            call_results = [*nested_results, *native_results]
            if call_results:
                return combine(call_results)
            if retains_unverified_environment:
                return (
                    result[0],
                    result[1],
                    result[2] | frozenset((("<unverified-call-result>",),)),
                )
            return result

        def analyze_environment_descriptor(
            value: dependency, attribute: str
        ) -> dependency | None:
            if not has_mutable_environment_reference(value):
                return None
            resolved_any = False
            unverified = False
            seen = set()
            results = []
            for path in value[2]:
                item, resolved = resolve_environment_path(path)
                if not resolved:
                    unverified = (
                        unverified
                        or not is_synthetic_path(path)
                        or (path[0] == "<environment>")
                    )
                    continue
                if id(item) in seen:
                    continue
                seen.add(id(item))
                resolved_any = True
                item_dependency = environment_value_dependency(item)
                owner_type = item if isinstance(item, type) else type(item)
                getattribute = inspect.getattr_static(
                    owner_type, "__getattribute__", None
                )
                if isinstance(getattribute, types.FunctionType):
                    results.append(
                        analyze_call(
                            (environment_value_dependency(getattribute),),
                            (
                                item_dependency,
                                environment_value_dependency(attribute),
                            ),
                        )
                    )
                try:
                    descriptor = inspect.getattr_static(owner_type, attribute)
                except AttributeError:
                    getattr_fn = inspect.getattr_static(owner_type, "__getattr__", None)
                    if isinstance(getattr_fn, types.FunctionType):
                        results.append(
                            analyze_call(
                                (environment_value_dependency(getattr_fn),),
                                (
                                    item_dependency,
                                    environment_value_dependency(attribute),
                                ),
                            )
                        )
                    elif getattr_fn is not None:
                        unverified = True
                    continue
                if isinstance(descriptor, property):
                    if descriptor.fget is not None:
                        results.append(
                            analyze_call(
                                (environment_value_dependency(descriptor.fget),),
                                (item_dependency,),
                            )
                        )
                    continue
                if isinstance(
                    descriptor,
                    (
                        classmethod,
                        staticmethod,
                        types.FunctionType,
                        types.GetSetDescriptorType,
                        types.MemberDescriptorType,
                        types.MethodDescriptorType,
                        types.WrapperDescriptorType,
                    ),
                ):
                    continue
                descriptor_get = inspect.getattr_static(
                    type(descriptor), "__get__", None
                )
                if isinstance(descriptor_get, types.FunctionType):
                    results.append(
                        analyze_call(
                            (environment_value_dependency(descriptor_get),),
                            (
                                environment_value_dependency(descriptor),
                                item_dependency,
                                environment_value_dependency(owner_type),
                            ),
                        )
                    )
                elif descriptor_get is not None:
                    unverified = True
            if unverified or not resolved_any:
                record_mutation()
            return combine(results) if results else None

        arg_count = code.co_argcount + code.co_kwonlyargcount
        parameter_names = list(code.co_varnames[:arg_count])
        next_parameter = arg_count
        if code.co_flags & inspect.CO_VARARGS:
            parameter_names.append(code.co_varnames[next_parameter])
            next_parameter += 1
        if code.co_flags & inspect.CO_VARKEYWORDS:
            parameter_names.append(code.co_varnames[next_parameter])
        input_names = set(parameter_names)
        initial_locals: dict[str, dependency] = {
            name: (
                False,
                True,
                frozenset(
                    (
                        ("<environment-object>",),
                        ("<identity-sensitive>",),
                        ("<identity-unstable>",),
                    )
                ),
            )
            for name in code.co_freevars
        }
        if parameter_dependencies is not None:
            if isinstance(parameter_dependencies, Mapping):
                initial_locals.update(parameter_dependencies)
            else:
                initial_locals.update(zip(parameter_names, parameter_dependencies))
        for name in input_names - initial_locals.keys():
            initial_locals[name] = (
                True,
                False,
                frozenset((("<input-value>",),)),
            )
        instructions = list(dis.get_instructions(code))

        def dependency_pair(
            values: Sequence[stack_slot],
        ) -> tuple[dependency, dependency] | None:
            if len(values) != 2:
                return None
            left, right = values
            if left is None or right is None:
                return None
            return left, right

        instruction_indices = {
            instruction.offset: index for index, instruction in enumerate(instructions)
        }
        exception_successors: dict[int, list[tuple[int, int, int]]] = {}
        exception_entries = getattr(dis.Bytecode(code), "exception_entries", ())
        if exception_entries:
            for entry in exception_entries:
                target = instruction_indices.get(entry.target)
                if target is None:
                    continue
                for protected_index, instruction in enumerate(instructions):
                    if entry.start <= instruction.offset < entry.end:
                        exception_successors.setdefault(protected_index, []).append(
                            (target, entry.depth, 1 + int(entry.lasti))
                        )
        else:
            for setup_index, instruction in enumerate(instructions):
                if not instruction.opname.startswith("SETUP_"):
                    continue
                target = instruction_indices.get(cast("int", instruction.argval))
                if target is None:
                    continue
                for protected_index in range(setup_index + 1, target):
                    exception_successors.setdefault(protected_index, []).append(
                        (target, 0, 3)
                    )
        states: dict[int, analysis_state] = {0: ((), initial_locals)}
        pending = [0]
        steps = 0
        step_limit = max(1024, len(instructions) * 64)

        def enqueue(
            index: int, stack: list[stack_slot], locals_: dict[str, dependency]
        ) -> None:
            if index >= len(instructions):
                return
            if len(stack) > code.co_stacksize:
                merged_stack_value = combine(stack)
                stack = [merged_stack_value] * code.co_stacksize
            merged, changed = merge_state(
                states.get(index), (tuple(stack), dict(locals_))
            )
            if changed:
                states[index] = merged
                pending.append(index)

        while pending:
            steps += 1
            if steps > step_limit:
                call_graph_incomplete = True
                called_globals.update(loaded_global_paths(code))
                break
            index = pending.pop()
            stack_tuple, incoming_locals = states[index]
            stack = list(stack_tuple)
            locals_ = dict(incoming_locals)
            instruction = instructions[index]
            opname = instruction.opname

            for target, depth, extra_slots in exception_successors.get(index, ()):
                handler_stack = list(stack_tuple[:depth])
                handler_stack.extend(
                    (False, True, frozenset()) for _ in range(extra_slots)
                )
                enqueue(target, handler_stack, locals_)

            def pop(count: int) -> list[stack_slot]:
                if count > len(stack):
                    values = list(stack)
                    stack.clear()
                    return values
                values = stack[-count:] if count else []
                if count:
                    del stack[-count:]
                return values

            if opname == "FORMAT_VALUE":
                has_format_spec = bool(instruction.arg and instruction.arg & 0x04)
                value_index = -2 if has_format_spec else -1
                formatted = stack[value_index] if len(stack) >= -value_index else None
                if formatted is not None:
                    conversion = (instruction.arg or 0) & 0x03
                    method = (
                        "__format__"
                        if conversion == 0
                        else "__str__"
                        if conversion == 1
                        else "__repr__"
                    )
                    analyze_call(
                        (
                            combine(
                                (
                                    input_method_dependency(formatted, method),
                                    environment_method_dependency(formatted, method),
                                )
                            ),
                        ),
                        (stack[-1],) if has_format_spec else (),
                    )
            elif opname in ("BEFORE_WITH", "SETUP_WITH"):
                manager = stack[-1] if stack else None
                if manager is not None:
                    for method in ("__enter__", "__exit__"):
                        analyze_call(
                            (
                                combine(
                                    (
                                        input_method_dependency(manager, method),
                                        environment_method_dependency(manager, method),
                                    )
                                ),
                            ),
                            (),
                        )

            if opname in (
                "LOAD_FAST_LOAD_FAST",
                "LOAD_FAST_BORROW_LOAD_FAST_BORROW",
            ):
                names = cast("tuple[str, str]", instruction.argval)
                stack.extend(
                    locals_.get(name, (name in input_names, False, frozenset()))
                    for name in names
                )
                enqueue(index + 1, stack, locals_)
                continue
            elif opname == "STORE_FAST_STORE_FAST":
                names = cast("tuple[str, str]", instruction.argval)
                for name in names:
                    values = pop(1)
                    if values and values[0] is not None:
                        locals_[name] = values[0]
                enqueue(index + 1, stack, locals_)
                continue
            elif opname == "STORE_FAST_LOAD_FAST":
                store_name, load_name = cast("tuple[str, str]", instruction.argval)
                values = pop(1)
                if values and values[0] is not None:
                    locals_[store_name] = values[0]
                stack.append(
                    locals_.get(
                        load_name,
                        (load_name in input_names, False, frozenset()),
                    )
                )
                enqueue(index + 1, stack, locals_)
                continue
            elif opname in ("JUMP_IF_FALSE_OR_POP", "JUMP_IF_TRUE_OR_POP"):
                if (
                    stack
                    and stack[-1] is not None
                    and has_unsafe_environment_behavior(stack[-1])
                ):
                    record_mutation()
                target = instruction_indices.get(cast("int", instruction.argval))
                if target is not None:
                    enqueue(target, stack, locals_)
                pop(1)
                enqueue(index + 1, stack, locals_)
                continue
            if opname == "FOR_ITER":
                target = instruction_indices.get(cast("int", instruction.argval))
                exit_stack = list(stack)
                if exit_stack:
                    exit_stack.pop()
                if target is not None:
                    enqueue(target, exit_stack, locals_)
                iterator = stack[-1] if stack else (False, False, frozenset())
                item = (
                    environment_container_contents_dependency(iterator, "iteration")
                    if iterator is not None
                    else None
                )
                stack.append(item if item is not None else iterator)
                enqueue(index + 1, stack, locals_)
                continue
            if opname in ("LOAD_GLOBAL", "LOAD_NAME", "LOAD_FROM_DICT_OR_GLOBALS"):
                effect = dis.stack_effect(instruction.opcode, instruction.arg)
                stack.extend(None for _ in range(max(effect - 1, 0)))
                global_name = cast("str", instruction.argval)
                stack.append(
                    (
                        False,
                        True,
                        frozenset(
                            (
                                scoped_path((global_name,))
                                if global_name in globals_scope
                                else (global_name,),
                            )
                        ),
                    )
                )
            elif opname.startswith("LOAD_FAST") or opname == "LOAD_DEREF":
                name = cast("str", instruction.argval)
                stack.append(
                    locals_.get(name, (name in input_names, False, frozenset()))
                )
            elif opname in ("LOAD_ATTR", "LOAD_METHOD"):
                origins = pop(1)
                if not origins:
                    stack.append((False, False, frozenset()))
                else:
                    effect = dis.stack_effect(instruction.opcode, instruction.arg)
                    stack.extend(None for _ in range(max(effect, 0)))
                    value = combine(origins)
                    if any(path[0] == "<unverified-call-result>" for path in value[2]):
                        record_mutation()
                    descriptor_result = analyze_environment_descriptor(
                        value, cast("str", instruction.argval)
                    )
                    input_method_paths = frozenset(
                        (
                            "<input-method>",
                            path[1],
                            cast("str", instruction.argval),
                        )
                        for path in value[2]
                        if path[0] == "<input-object>"
                    )
                    stack.append(
                        descriptor_result
                        if descriptor_result is not None
                        else (
                            value[0],
                            value[1],
                            frozenset(
                                extend_path(path, cast("str", instruction.argval))
                                for path in value[2]
                            )
                            | (
                                frozenset(
                                    (("<input>", cast("str", instruction.argval)),)
                                )
                                if value[0]
                                else frozenset()
                            )
                            | input_method_paths,
                        )
                    )
            elif opname == "LOAD_SUPER_ATTR":
                prior = combine(stack)
                try:
                    effect = dis.stack_effect(instruction.opcode, instruction.arg)
                except ValueError:
                    effect = dis.stack_effect(instruction.opcode)
                pop(max(1, 1 - effect))
                stack.append(
                    (
                        prior[0],
                        True,
                        frozenset((("super", cast("str", instruction.argval)),)),
                    )
                )
            elif opname == "IMPORT_NAME":
                pop(2)
                stack.append(
                    (
                        False,
                        True,
                        frozenset(((cast("str", instruction.argval),),)),
                    )
                )
            elif opname == "IMPORT_FROM":
                origin = stack[-1] if stack else None
                value = combine((origin,))
                stack.append(
                    (
                        value[0],
                        value[1],
                        frozenset(
                            extend_path(path, cast("str", instruction.argval))
                            for path in value[2]
                        ),
                    )
                )
            elif opname == "LOAD_CONST":
                constant = instruction.argval
                if isinstance(constant, CodeType):
                    local_function_codes[id(constant)] = constant
                    stack.append(
                        (
                            False,
                            False,
                            frozenset((("<local-function>", str(id(constant))),)),
                        )
                    )
                elif type(constant) in (
                    int,
                    float,
                    complex,
                    str,
                    bytes,
                    tuple,
                    frozenset,
                ):
                    stack.append(environment_value_dependency(constant))
                else:
                    stack.append((False, False, frozenset()))
            elif opname == "MAKE_FUNCTION":
                try:
                    effect = dis.stack_effect(instruction.opcode, instruction.arg)
                except ValueError:
                    effect = dis.stack_effect(instruction.opcode)
                values = pop(max(1, 1 - effect))
                function_paths = {
                    path
                    for value in values
                    if value is not None
                    for path in value[2]
                    if path[0] == "<local-function>"
                }
                payload_values = []
                for value in values:
                    if value is None:
                        continue
                    payload_paths = frozenset(
                        path for path in value[2] if path[0] != "<local-function>"
                    )
                    if value[0] or value[1] or payload_paths:
                        payload_values.append((value[0], value[1], payload_paths))
                payload = combine(payload_values)
                for path in function_paths:
                    function_id = int(path[1])
                    local_function_payloads[function_id] = combine(
                        (local_function_payloads.get(function_id), payload)
                    )
                stack.append((False, False, frozenset(function_paths)))
            elif opname == "SET_FUNCTION_ATTRIBUTE":
                values = pop(2)
                attribute = values[0] if values else None
                function = values[1] if len(values) > 1 else None
                function_paths = (
                    frozenset(
                        path for path in function[2] if path[0] == "<local-function>"
                    )
                    if function is not None
                    else frozenset()
                )
                for path in function_paths:
                    function_id = int(path[1])
                    local_function_payloads[function_id] = combine(
                        (local_function_payloads.get(function_id), attribute)
                    )
                stack.append((False, False, function_paths))
            elif opname.startswith("LOAD_"):
                effect = dis.stack_effect(instruction.opcode, instruction.arg)
                stack.extend((False, False, frozenset()) for _ in range(max(effect, 0)))
            elif opname == "PUSH_NULL":
                stack.append(None)
            elif opname == "BINARY_SUBSCR":
                values = pop(2)
                selected = None
                if (pair := dependency_pair(values)) is not None:
                    left, right = pair
                    analyze_call(
                        (
                            combine(
                                (
                                    input_method_dependency(left, "__getitem__"),
                                    environment_method_dependency(left, "__getitem__"),
                                )
                            ),
                        ),
                        (right,),
                    )
                    if (
                        (
                            has_input_mapping(left)
                            and has_identity_sensitive_environment(right)
                        )
                        or (
                            has_identity_sensitive_input_mapping_keys(left) and right[0]
                        )
                        or (
                            right[0]
                            and has_environment_mapping(left)
                            and has_mapping_key_identity_sensitive_environment(left)
                        )
                    ):
                        found_identity = True
                    selected = environment_container_contents_dependency(left, "values")
                combined = combine(values)
                stack.append(
                    selected
                    if selected is not None
                    else (
                        combined[0],
                        combined[1],
                        frozenset(
                            path
                            for path in combined[2]
                            if path[0]
                            not in (
                                "<container>",
                                "<input-container>",
                                "<input-container-content>",
                            )
                        )
                        | (
                            frozenset((("<input-value>",),))
                            if combined[0]
                            else frozenset()
                        ),
                    )
                )
            elif opname == "IS_OP":
                values = pop(2)
                if (pair := dependency_pair(values)) is not None:
                    left, right = pair
                    if (
                        (left[0] and has_identity_unstable_environment(right))
                        or (right[0] and has_identity_unstable_environment(left))
                        or may_have_unguarded_input_alias(left, right)
                    ):
                        found_identity = True
                stack.append((False, False, frozenset()))
            elif opname == "CONTAINS_OP":
                values = pop(2)
                if (pair := dependency_pair(values)) is not None:
                    left, right = pair
                    analyze_call(
                        (
                            combine(
                                (
                                    input_method_dependency(right, "__contains__"),
                                    environment_method_dependency(
                                        right, "__contains__"
                                    ),
                                )
                            ),
                        ),
                        (left,),
                    )
                    if (
                        (
                            left[0]
                            and has_environment_container(right)
                            and has_lookup_identity_sensitive_environment(right)
                        )
                        or (
                            left[0]
                            and has_lookup_identity_sensitive_input_container(right)
                        )
                        or (
                            has_input_container(right)
                            and has_lookup_identity_sensitive_environment(left)
                        )
                    ):
                        found_identity = True
                stack.append((False, False, frozenset()))
            elif opname == "COMPARE_OP":
                values = pop(2)
                if any(
                    value is not None and has_unsafe_environment_behavior(value)
                    for value in values
                ):
                    record_mutation()
                if (
                    instruction.argval in ("==", "!=")
                    and (pair := dependency_pair(values)) is not None
                ):
                    left, right = pair
                    analyze_call(
                        (
                            combine(
                                (
                                    input_method_dependency(left, "__eq__"),
                                    environment_method_dependency(left, "__eq__"),
                                )
                            ),
                        ),
                        (right,),
                    )
                    analyze_call(
                        (
                            combine(
                                (
                                    input_method_dependency(right, "__eq__"),
                                    environment_method_dependency(right, "__eq__"),
                                )
                            ),
                        ),
                        (left,),
                    )
                    if (
                        (
                            has_input_container(left)
                            and has_identity_sensitive_environment(right)
                        )
                        or (
                            left[0]
                            and right[0]
                            and (
                                has_identity_sensitive_input_container(left)
                                or has_identity_sensitive_input_container(right)
                            )
                        )
                        or (
                            right[0]
                            and has_environment_container(left)
                            and has_identity_sensitive_environment(left)
                        )
                        or (
                            has_input_container(right)
                            and has_identity_sensitive_environment(left)
                        )
                        or (
                            left[0]
                            and has_environment_container(right)
                            and has_identity_sensitive_environment(right)
                        )
                    ):
                        found_identity = True
                stack.append(combine(values))
            elif opname.startswith("UNARY_"):
                values = pop(1)
                if any(
                    value is not None and has_unsafe_environment_behavior(value)
                    for value in values
                ):
                    record_mutation()
                stack.append(combine(values))
            elif opname.startswith("INPLACE_"):
                values = pop(2)
                if (
                    values
                    and values[0] is not None
                    and has_environment_reference(values[0])
                ):
                    record_mutation()
                stack.append(combine(values))
            elif opname.startswith("BINARY_"):
                values = pop(2)
                if (
                    opname == "BINARY_OP"
                    and isinstance(instruction.argrepr, str)
                    and instruction.argrepr.endswith("=")
                    and values
                    and values[0] is not None
                    and has_environment_reference(values[0])
                ) or (
                    not analyze_binary_behavior(
                        opname,
                        instruction.argrepr
                        if isinstance(instruction.argrepr, str)
                        else "",
                        values,
                    )
                    and any(
                        value is not None and has_unsafe_environment_behavior(value)
                        for value in values
                    )
                ):
                    record_mutation()
                stack.append(combine(values))
            elif opname in ("GET_ITER", "GET_YIELD_FROM_ITER"):
                if (
                    stack
                    and stack[-1] is not None
                    and has_unsafe_environment_behavior(stack[-1])
                ):
                    record_mutation()
            elif opname in ("UNPACK_SEQUENCE", "UNPACK_EX") and isinstance(
                instruction.arg, int
            ):
                values = pop(1)
                unpacked = combine(values)
                for method in ("__iter__", "__getitem__"):
                    analyze_call(
                        (
                            combine(
                                (
                                    input_method_dependency(unpacked, method),
                                    environment_method_dependency(unpacked, method),
                                )
                            ),
                        ),
                        (),
                    )
                count = (
                    instruction.arg
                    if opname == "UNPACK_SEQUENCE"
                    else (instruction.arg & 0xFF) + (instruction.arg >> 8) + 1
                )
                selected = environment_container_contents_dependency(
                    unpacked, "iteration"
                )
                stack.extend(
                    selected if selected is not None else unpacked for _ in range(count)
                )
            elif opname.startswith("BUILD_") and isinstance(instruction.arg, int):
                count = instruction.arg
                if opname == "BUILD_MAP":
                    count *= 2
                elif opname == "BUILD_CONST_KEY_MAP":
                    count += 1
                values = pop(count)
                protocol_values = (
                    values
                    if opname == "BUILD_SET"
                    else values[::2]
                    if opname == "BUILD_MAP"
                    else ()
                )
                if any(
                    value is not None and has_unsafe_environment_behavior(value)
                    for value in protocol_values
                ):
                    record_mutation()
                combined = combine(values)
                markers = {("<container>",)}
                if combined[0]:
                    markers.add(("<input-container-content>",))
                    if opname in ("BUILD_MAP", "BUILD_CONST_KEY_MAP"):
                        markers.add(("<input-mapping>",))
                if any(
                    path[0]
                    in (
                        "<input-identity-sensitive>",
                        "<input-identity-sensitive-container>",
                    )
                    for path in combined[2]
                ):
                    markers.add(("<input-identity-sensitive-container>",))
                if any(
                    path[0]
                    in (
                        "<input-identity-sensitive>",
                        "<input-lookup-identity-sensitive-container>",
                    )
                    for path in combined[2]
                ):
                    markers.add(("<input-lookup-identity-sensitive-container>",))
                    if opname in ("BUILD_MAP", "BUILD_CONST_KEY_MAP"):
                        markers.add(("<input-mapping-key-identity-sensitive>",))
                if has_identity_sensitive_environment(combined):
                    markers.add(("<identity-sensitive>",))
                stack.append(
                    (
                        combined[0],
                        combined[1],
                        frozenset(markers),
                    )
                )
            elif opname in (
                "DICT_MERGE",
                "DICT_UPDATE",
                "LIST_APPEND",
                "LIST_EXTEND",
                "MAP_ADD",
                "SET_ADD",
                "SET_UPDATE",
            ) and isinstance(instruction.arg, int):
                values = pop(2 if opname == "MAP_ADD" else 1)
                protocol_values = (
                    values[:1]
                    if opname == "MAP_ADD"
                    else values
                    if opname
                    in (
                        "DICT_MERGE",
                        "DICT_UPDATE",
                        "LIST_EXTEND",
                        "SET_ADD",
                        "SET_UPDATE",
                    )
                    else ()
                )
                if any(
                    value is not None
                    and (
                        has_unsafe_native_arguments(value)
                        if opname in ("DICT_MERGE", "DICT_UPDATE", "SET_UPDATE")
                        else has_unsafe_environment_behavior(value)
                    )
                    for value in protocol_values
                ):
                    record_mutation()
                destination = len(stack) - instruction.arg
                if destination >= 0:
                    stack[destination] = merge_dependency(
                        stack[destination], combine(values)
                    )
            elif opname == "LIST_TO_TUPLE":
                pass
            elif opname in (
                "CALL",
                "CALL_KW",
                "CALL_FUNCTION",
                "CALL_FUNCTION_KW",
                "CALL_METHOD",
            ) and isinstance(instruction.arg, int):
                keyword_names = keyword_names_for_call(index)
                if opname in ("CALL_FUNCTION_KW", "CALL_KW"):
                    pop(1)
                if opname == "CALL_METHOD":
                    args = pop(instruction.arg)
                    callable_values = pop(2)
                elif opname in ("CALL", "CALL_KW") and sys.version_info >= (3, 13):
                    args = pop(instruction.arg)
                    if stack and stack[-1] is None:
                        pop(1)
                    callable_values = pop(1)
                else:
                    args = pop(instruction.arg)
                    callable_values = pop(1)
                if stack and stack[-1] is None:
                    pop(1)
                stack.append(analyze_call(callable_values, args, keyword_names))
            elif opname == "CALL_FUNCTION_EX":
                if sys.version_info >= (3, 14):
                    kwargs = pop(1)
                else:
                    kwargs = pop(1) if instruction.arg else []
                args = pop(1)
                if sys.version_info >= (3, 13) and stack and stack[-1] is None:
                    pop(1)
                callable_values = pop(1)
                if stack and stack[-1] is None:
                    pop(1)
                stack.append(
                    analyze_call(callable_values, (*args, *kwargs), packed=True)
                )
            elif opname == "COPY" and isinstance(instruction.arg, int):
                index = len(stack) - instruction.arg
                stack.append(stack[index] if index >= 0 else None)
            elif opname == "SWAP" and isinstance(instruction.arg, int):
                index = len(stack) - instruction.arg
                if index >= 0 and stack:
                    stack[-1], stack[index] = stack[index], stack[-1]
            elif opname == "DUP_TOP":
                stack.append(stack[-1] if stack else None)
            elif opname == "DUP_TOP_TWO":
                stack.extend(stack[-2:])
            elif opname == "ROT_TWO" and len(stack) >= 2:
                stack[-2], stack[-1] = stack[-1], stack[-2]
            elif opname in ("ROT_THREE", "ROT_FOUR"):
                count = 3 if opname == "ROT_THREE" else 4
                if len(stack) >= count:
                    merged = combine(stack[-count:])
                    stack[-count:] = [merged] * count
            elif opname == "POP_TOP":
                pop(1)
            elif opname == "RETURN_VALUE":
                values = pop(1)
                if values and values[0] is not None:
                    return_dependencies.append(values[0])
                continue
            elif opname == "RETURN_CONST":
                return_dependencies.append(
                    environment_value_dependency(instruction.argval)
                )
                continue
            elif opname == "YIELD_VALUE":
                values = pop(1)
                if values and values[0] is not None:
                    return_dependencies.append(values[0])
            elif opname.startswith("STORE_FAST"):
                values = pop(1)
                if values and values[0] is not None:
                    locals_[cast("str", instruction.argval)] = values[0]
            elif opname == "STORE_DEREF":
                values = pop(1)
                name = cast("str", instruction.argval)
                if values and values[0] is not None:
                    locals_[name] = values[0]
                if name in code.co_freevars:
                    record_mutation()
            elif opname == "DELETE_DEREF":
                if cast("str", instruction.argval) in code.co_freevars:
                    record_mutation()
            elif opname == "STORE_ATTR":
                values = pop(2)
                if (
                    len(values) == 2
                    and values[1] is not None
                    and has_environment_reference(values[1])
                ):
                    record_mutation()
            elif opname == "STORE_SUBSCR":
                values = pop(3)
                if (
                    len(values) == 3
                    and values[1] is not None
                    and has_environment_reference(values[1])
                ):
                    record_mutation()
            elif opname == "DELETE_ATTR":
                values = pop(1)
                if (
                    values
                    and values[0] is not None
                    and has_environment_reference(values[0])
                ):
                    record_mutation()
            elif opname == "DELETE_SUBSCR":
                values = pop(2)
                if (
                    values
                    and values[0] is not None
                    and has_environment_reference(values[0])
                ):
                    record_mutation()
            elif opname.startswith("STORE_"):
                pop(1)
            elif opname.startswith("POP_JUMP"):
                pass
            elif opname in (
                "CACHE",
                "EXTENDED_ARG",
                "KW_NAMES",
                "NOP",
                "PRECALL",
                "RESUME",
            ):
                pass
            else:
                prior = combine(stack)
                try:
                    effect = dis.stack_effect(instruction.opcode, instruction.arg)
                except ValueError:
                    effect = dis.stack_effect(instruction.opcode)
                if effect < 0:
                    pop(-effect)
                    if stack:
                        stack[-1] = merge_dependency(stack[-1], prior)
                elif effect > 0:
                    stack.extend(prior for _ in range(effect))
                elif stack and (prior[0] or prior[1]):
                    stack[:] = [merge_dependency(value, prior) for value in stack]

            if opname in ("RAISE_VARARGS", "RERAISE"):
                continue
            if opname.startswith("POP_JUMP"):
                if stack and stack[-1] is not None:
                    if has_unsafe_environment_behavior(stack[-1]):
                        record_mutation()
                    analyze_call(
                        (
                            combine(
                                (
                                    input_method_dependency(stack[-1], "__bool__"),
                                    environment_method_dependency(
                                        stack[-1], "__bool__"
                                    ),
                                )
                            ),
                        ),
                        (),
                    )
                    analyze_call(
                        (
                            combine(
                                (
                                    input_method_dependency(stack[-1], "__len__"),
                                    environment_method_dependency(stack[-1], "__len__"),
                                )
                            ),
                        ),
                        (),
                    )
                pop(1)
                target = instruction_indices.get(cast("int", instruction.argval))
                if target is not None:
                    enqueue(target, stack, locals_)
                enqueue(index + 1, stack, locals_)
            elif opname.startswith("JUMP_IF"):
                target = instruction_indices.get(cast("int", instruction.argval))
                if target is not None:
                    enqueue(target, stack, locals_)
                enqueue(index + 1, stack, locals_)
            elif opname.startswith("JUMP_") or opname in (
                "JUMP_ABSOLUTE",
                "JUMP_FORWARD",
            ):
                target = instruction_indices.get(cast("int", instruction.argval))
                if target is not None:
                    enqueue(target, stack, locals_)
            else:
                enqueue(index + 1, stack, locals_)
        try:
            for const in code.co_consts:
                if isinstance(const, CodeType):
                    closure_dependencies = {
                        name: combine(
                            tuple(
                                state_locals.get(name)
                                for _, state_locals in states.values()
                            )
                        )
                        for name in const.co_freevars
                    }
                    (
                        nested_identity,
                        nested_calls,
                        _,
                        nested_incomplete,
                        nested_mutates_environment,
                    ) = bytecode_dependencies(
                        const,
                        globals_scope,
                        closure_dependencies,
                        active,
                        dependency_scopes,
                    )
                    found_identity |= nested_identity
                    call_graph_incomplete |= nested_incomplete
                    mutates_environment |= nested_mutates_environment
                    called_globals.update(nested_calls)
            return (
                found_identity,
                called_globals,
                combine(return_dependencies),
                call_graph_incomplete,
                mutates_environment,
            )
        finally:
            active.remove(id(code))

    def loaded_local_paths(code: CodeType, local_name: str) -> list[tuple[str, ...]]:
        paths = []
        instructions = list(dis.get_instructions(code))
        for index, instruction in enumerate(instructions):
            if not instruction.opname.startswith("LOAD_FAST") or (
                instruction.argval != local_name
            ):
                continue
            path = []
            for following in instructions[index + 1 :]:
                if following.opname not in ("LOAD_ATTR", "LOAD_METHOD"):
                    break
                if not isinstance(following.argval, str):
                    break
                path.append(following.argval)
            if path:
                paths.append(tuple(path))
        return paths

    def has_unmodeled_local_access(code: CodeType, local_name: str) -> bool:
        if local_name in code.co_cellvars:
            return True
        instructions = list(dis.get_instructions(code))
        for index, instruction in enumerate(instructions):
            if not instruction.opname.startswith("LOAD_FAST") or (
                instruction.argval != local_name
            ):
                continue
            if index + 1 == len(instructions) or instructions[index + 1].opname not in (
                "LOAD_ATTR",
                "LOAD_METHOD",
            ):
                return True
        return False

    def loaded_global_names(code: CodeType) -> set[str]:
        return {path[0] for path in loaded_global_paths(code)}

    def loaded_attribute_names(code: CodeType) -> set[str]:
        names = {
            instruction.argval
            for instruction in dis.get_instructions(code)
            if instruction.opname in ("LOAD_ATTR", "LOAD_METHOD")
            and isinstance(instruction.argval, str)
        }
        if any(
            instruction.opname == "BINARY_SUBSCR"
            for instruction in dis.get_instructions(code)
        ):
            names.add("__getitem__")
        for const in code.co_consts:
            if isinstance(const, CodeType):
                names.update(loaded_attribute_names(const))
        return names

    input_behavior_names = {"forward"}
    scan_all_input_behaviors = False
    explicit_input_ids = _dynamo_reachable_object_ids(
        [
            value
            for example in examples
            for value in (*example.args, *example.kwargs.values())
        ],
        skip_literals=True,
    )

    def is_library_module(module_name: str | None) -> bool:
        root = module_name.partition(".")[0] if isinstance(module_name, str) else ""
        return root == "torch" or root in sys.stdlib_module_names

    def is_library_defined_function(function: types.FunctionType) -> bool:
        code_path = os.path.realpath(function.__code__.co_filename)
        torch_root = os.path.realpath(os.path.dirname(torch.__file__))
        try:
            is_torch_file = os.path.commonpath((torch_root, code_path)) == torch_root
        except ValueError:
            is_torch_file = False
        if is_torch_file:
            return True
        if not is_library_module(function.__module__):
            return False
        module = sys.modules.get(function.__module__)
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            return False
        return os.path.realpath(module_file) == code_path

    def container_reaches_explicit_input(values: Sequence[object]) -> bool:
        pending = list(values)
        seen = set()
        while pending:
            value = pending.pop()
            value_id = id(value)
            if value_id in explicit_input_ids:
                return True
            if value_id in seen:
                continue
            seen.add(value_id)
            value_type = type(value)
            if issubclass(value_type, (dict, MappingProxyType)):
                mapping = cast("Mapping[object, object]", value)
                pending.extend(mapping.keys())
                pending.extend(mapping.values())
            elif issubclass(value_type, (tuple, list, set, frozenset)):
                pending.extend(cast("Iterable[object]", value))
        return False

    def function_module_reaches_explicit_input(
        function: types.FunctionType,
    ) -> bool:
        return container_reaches_explicit_input(list(function.__globals__.values()))

    def resolve_static_path(
        root: object,
        path: tuple[str, ...],
        ignored_input_ids: set[int] | None = None,
    ) -> tuple[object, object | None, bool, bool]:
        def has_custom_getattribute(value: object) -> bool:
            value_type = type(value)
            implementation = inspect.getattr_static(
                value_type, "__getattribute__", None
            )
            if isinstance(value, type):
                return implementation is not type.__getattribute__
            if isinstance(value, types.ModuleType):
                return implementation is not types.ModuleType.__getattribute__
            return implementation is not object.__getattribute__

        value = root
        receiver = None
        ignored = ignored_input_ids or set()
        aliases_input = id(value) in explicit_input_ids and id(value) not in ignored
        for attribute in path:
            receiver = value
            if isinstance(value, types.FunctionType) and attribute in (
                "__annotations__",
                "__closure__",
                "__defaults__",
                "__dict__",
                "__globals__",
                "__kwdefaults__",
            ):
                value = getattr(value, attribute)
                aliases_input |= (
                    id(value) in explicit_input_ids and id(value) not in ignored
                )
                continue
            dynamic_lookup = has_custom_getattribute(receiver)
            try:
                value = inspect.getattr_static(value, attribute)
            except AttributeError:
                return receiver, None, aliases_input, True
            aliases_input |= (
                id(value) in explicit_input_ids and id(value) not in ignored
            )
            if dynamic_lookup:
                return receiver, None, aliases_input, True
        return value, receiver, aliases_input, False

    def referenced_globals(
        function: types.FunctionType,
    ) -> list[tuple[str, object, object | None, bool, bool]]:
        values = []
        for path in loaded_global_paths(function.__code__):
            if path[0] not in function.__globals__:
                continue
            resolved = resolve_static_path(function.__globals__[path[0]], path[1:])
            values.append((".".join(path), *resolved))
        return values

    def referenced_python_functions(
        values: Sequence[object],
        *,
        traverse_containers: bool = True,
        traverse_constructors: bool = False,
    ) -> list[tuple[types.FunctionType, object | None]]:
        functions: list[tuple[types.FunctionType, object | None]] = []
        pending = list(values)
        seen = set()
        while pending:
            value = pending.pop()
            if id(value) in seen:
                continue
            seen.add(id(value))
            if isinstance(value, types.MethodType):
                functions.append((value.__func__, value.__self__))
                continue
            if isinstance(value, types.FunctionType):
                functions.append((value, None))
            elif isinstance(value, functools.partial):
                function = value.func
                if isinstance(function, types.MethodType):
                    functions.append((function.__func__, function.__self__))
                elif isinstance(function, types.FunctionType):
                    receiver = value.args[0] if value.args else None
                    functions.append((function, receiver))
                else:
                    pending.append(function)
                pending.extend(value.args)
                pending.extend((value.keywords or {}).values())
            elif isinstance(value, functools.partialmethod):
                pending.append(value.func)
                pending.extend(value.args)
                pending.extend((value.keywords or {}).values())
            elif isinstance(value, (classmethod, staticmethod)):
                pending.append(value.__func__)
            elif isinstance(value, property):
                pending.extend(
                    function
                    for function in (value.fget, value.fset, value.fdel)
                    if function is not None
                )
            elif isinstance(value, type) and traverse_constructors:
                pending.extend(
                    constructor
                    for name in ("__new__", "__init__")
                    if (constructor := inspect.getattr_static(value, name, None))
                    is not None
                )
            elif callable(value) and not isinstance(value, (type, types.ModuleType)):
                call = inspect.getattr_static(type(value), "__call__", None)
                if isinstance(call, types.FunctionType):
                    functions.append((call, value))
            elif traverse_containers and isinstance(value, (dict, MappingProxyType)):
                pending.extend(item for pair in value.items() for item in pair)
            elif traverse_containers and isinstance(
                value, (tuple, list, set, frozenset)
            ):
                pending.extend(value)
        return functions

    def library_value_leaves(values: Sequence[object]) -> list[object]:
        leaves = []
        pending = list(values)
        seen = set()
        while pending:
            value = pending.pop()
            if id(value) in seen:
                continue
            seen.add(id(value))
            if isinstance(value, (dict, MappingProxyType)):
                pending.extend(item for pair in value.items() for item in pair)
            elif isinstance(value, (tuple, list, set, frozenset)):
                pending.extend(value)
            elif isinstance(value, functools.partial):
                pending.append(value.func)
                pending.extend(value.args)
                pending.extend((value.keywords or {}).values())
            elif isinstance(value, functools.partialmethod):
                pending.append(value.func)
                pending.extend(value.args)
                pending.extend((value.keywords or {}).values())
            elif isinstance(value, types.MethodType):
                pending.extend((value.__func__, value.__self__))
            else:
                leaves.append(value)
        return leaves

    def mutates_globals(code: CodeType) -> bool:
        if any(
            instruction.opname in ("STORE_GLOBAL", "DELETE_GLOBAL")
            for instruction in dis.get_instructions(code)
        ):
            return True
        return any(
            mutates_globals(const)
            for const in code.co_consts
            if isinstance(const, CodeType)
        )

    def validate_no_global_mutation(
        functions: Sequence[types.FunctionType],
    ) -> None:
        mutation_pending = list(functions)
        mutation_seen = set()
        while mutation_pending:
            function = mutation_pending.pop()
            if id(function) in mutation_seen:
                continue
            mutation_seen.add(id(function))
            if mutates_globals(function.__code__):
                raise NotImplementedError(
                    "precompile tracer='dynamo' Python functions cannot mutate globals "
                    "or invoke unverified behavior on mutable environment objects."
                )
            _, called_paths, _, _, mutates_environment = bytecode_dependencies(
                function.__code__, function.__globals__
            )
            called_names = {".".join(path) for path in called_paths}
            for name, value, receiver, _, _ in referenced_globals(function):
                if (
                    name in called_names
                    and isinstance(value, type)
                    and isinstance(
                        inspect.getattr_static(type(value), "__call__", None),
                        types.FunctionType,
                    )
                ):
                    raise NotImplementedError(
                        "precompile tracer='dynamo' cannot verify an indirect "
                        "Python call through a user-defined metaclass."
                    )
                attribute = name.rsplit(".", 1)[-1]
                if isinstance(receiver, types.FunctionType) and attribute in (
                    "__closure__",
                    "__globals__",
                ):
                    raise PrecompileError(
                        "precompile tracer='dynamo' cannot preserve an input-derived "
                        "identity or mutation through dynamic function metadata "
                        f"access via {attribute}."
                    )
                for referenced, _ in referenced_python_functions(
                    [value], traverse_constructors=name in called_names
                ):
                    if not is_library_defined_function(referenced):
                        mutation_pending.append(referenced)
            if mutates_environment:
                raise NotImplementedError(
                    "precompile tracer='dynamo' Python functions cannot mutate globals "
                    "or invoke unverified behavior on mutable environment objects."
                )

    validate_no_global_mutation([target])

    pending_functions: list[tuple[types.FunctionType, object | None]] = [(target, None)]
    seen_functions = set()
    while pending_functions:
        function, _ = pending_functions.pop()
        if id(function) in seen_functions:
            continue
        seen_functions.add(id(function))
        input_behavior_names.update(loaded_attribute_names(function.__code__))
        for referenced, receiver in referenced_python_functions(
            [value for _, value, _, _, _ in referenced_globals(function)]
        ):
            if (
                not is_library_defined_function(referenced)
                or not is_library_module(function.__module__)
                or function_module_reaches_explicit_input(referenced)
            ):
                pending_functions.append((referenced, receiver))

    def loaded_bound_paths(code: CodeType, name: str) -> list[tuple[str, ...]]:
        paths = []
        instructions = list(dis.get_instructions(code))
        for index, instruction in enumerate(instructions):
            if (
                instruction.opname
                not in (
                    "LOAD_FAST",
                    "LOAD_FAST_CHECK",
                    "LOAD_DEREF",
                    "LOAD_NAME",
                    "LOAD_GLOBAL",
                )
                or instruction.argval != name
            ):
                continue
            path = []
            for following in instructions[index + 1 :]:
                if following.opname not in ("LOAD_ATTR", "LOAD_METHOD"):
                    break
                if not isinstance(following.argval, str):
                    break
                path.append(following.argval)
            paths.append(tuple(path))
        for constant in code.co_consts:
            if isinstance(constant, CodeType) and name in constant.co_freevars:
                paths.extend(loaded_bound_paths(constant, name))
        return paths

    def imported_bindings(
        code: CodeType, globals_scope: dict[str, object]
    ) -> list[tuple[CodeType, str | None, object, bool]]:
        bindings: list[tuple[CodeType, str | None, object, bool]] = []
        instructions = list(dis.get_instructions(code))
        for index, instruction in enumerate(instructions):
            if instruction.opname != "IMPORT_NAME" or not isinstance(
                instruction.argval, str
            ):
                continue
            level = (
                instructions[index - 2].argval
                if index >= 2
                and instructions[index - 2].opname == "LOAD_CONST"
                and isinstance(instructions[index - 2].argval, int)
                else 0
            )
            module_name = instruction.argval
            fromlist = (
                instructions[index - 1].argval
                if index >= 1 and instructions[index - 1].opname == "LOAD_CONST"
                else None
            )
            if level:
                package = globals_scope.get("__package__")
                if not isinstance(package, str):
                    continue
                try:
                    module_name = importlib.util.resolve_name(
                        "." * level + module_name, package
                    )
                except ImportError:
                    continue
            imported_module = sys.modules.get(module_name)
            if imported_module is None:
                root_name, _, remainder = module_name.partition(".")
                import_root = sys.modules.get(root_name)
                if import_root is not None and remainder:
                    resolved, _, _, unresolved = resolve_static_path(
                        import_root, tuple(remainder.split("."))
                    )
                    if not unresolved and isinstance(resolved, types.ModuleType):
                        imported_module = resolved
            if imported_module is None:
                continue
            following = instructions[index + 1 :]
            store_ops = {"STORE_DEREF", "STORE_FAST", "STORE_GLOBAL", "STORE_NAME"}
            if fromlist is None:
                store = next(
                    (
                        candidate
                        for candidate in following
                        if candidate.opname in store_ops
                        and isinstance(candidate.argval, str)
                    ),
                    None,
                )
                if store is None:
                    continue
                root = (
                    sys.modules.get(module_name.partition(".")[0], imported_module)
                    if following[0] is store
                    else imported_module
                )
                bindings.append((code, store.argval, root, False))
            elif following and following[0].opname == "IMPORT_FROM":
                import_root = (
                    sys.modules.get(module_name.partition(".")[0], imported_module)
                    if fromlist is None
                    else imported_module
                )
                offset = 0
                while offset < len(following):
                    imported = following[offset]
                    if imported.opname != "IMPORT_FROM" or not isinstance(
                        imported.argval, str
                    ):
                        break
                    if offset + 1 >= len(following):
                        break
                    store = following[offset + 1]
                    if store.opname not in store_ops or not isinstance(
                        store.argval, str
                    ):
                        break
                    value, _, _, unresolved = resolve_static_path(
                        import_root, (imported.argval,)
                    )
                    bindings.append((code, store.argval, value, unresolved))
                    offset += 2
            elif following and following[0].opname == "IMPORT_STAR":
                bindings.append((code, None, imported_module, True))
        for constant in code.co_consts:
            if isinstance(constant, CodeType):
                bindings.extend(imported_bindings(constant, globals_scope))
        return bindings

    import_scan_active: set[int] = set()

    def imported_modules_reach_explicit_input(
        function: types.FunctionType, *, reject_module_values: bool = True
    ) -> bool:
        function_id = id(function)
        if function_id in import_scan_active:
            return False
        import_scan_active.add(function_id)
        try:
            for code, local_name, root, unresolved in imported_bindings(
                function.__code__, function.__globals__
            ):
                if unresolved:
                    if receiver_reaches_explicit_input(root, set()):
                        return True
                    continue
                paths = (
                    [()] if local_name is None else loaded_bound_paths(code, local_name)
                )
                for path in paths:
                    if not path and isinstance(root, types.ModuleType):
                        if reject_module_values:
                            raise PrecompileError(
                                "precompile tracer='dynamo' cannot analyze a locally "
                                "imported module that is passed or aliased as a value; "
                                "access its attributes directly."
                            )
                        module_state = [
                            item
                            for item in vars(root).values()
                            if not isinstance(
                                item,
                                (
                                    CodeType,
                                    type,
                                    types.FunctionType,
                                    types.ModuleType,
                                ),
                            )
                        ]
                        if (
                            _dynamo_reachable_object_ids(
                                module_state, skip_literals=True
                            )
                            & explicit_input_ids
                        ):
                            return True
                        continue
                    value, path_receiver, aliases_input, path_unresolved = (
                        resolve_static_path(root, path)
                    )
                    if aliases_input:
                        return True
                    if path_unresolved:
                        if reject_module_values:
                            if receiver_reaches_explicit_input(value, set()):
                                return True
                            raise PrecompileError(
                                "precompile tracer='dynamo' cannot analyze dynamic "
                                "attribute access through a locally imported module."
                            )
                        if isinstance(value, types.ModuleType):
                            module_getattr = vars(value).get("__getattr__")
                            if isinstance(
                                module_getattr, types.FunctionType
                            ) and reaches_explicit_input(module_getattr, set()):
                                return True
                            if container_reaches_explicit_input(
                                list(vars(value).values())
                            ):
                                return True
                        elif (
                            _dynamo_reachable_object_ids([value], skip_literals=True)
                            & explicit_input_ids
                        ):
                            return True
                        continue
                    if isinstance(value, types.ModuleType):
                        if reject_module_values:
                            raise PrecompileError(
                                "precompile tracer='dynamo' cannot analyze a locally "
                                "imported module object passed or aliased as a value; "
                                "access its attributes directly."
                            )
                        if container_reaches_explicit_input(list(vars(value).values())):
                            return True
                    elif reject_module_values:
                        if reaches_explicit_input(value, set(), path_receiver):
                            return True
                    elif (
                        _dynamo_reachable_object_ids(
                            [value, path_receiver], skip_literals=True
                        )
                        & explicit_input_ids
                    ):
                        return True
            return False
        finally:
            import_scan_active.remove(function_id)

    def reachable_imports_reach_explicit_input(function: types.FunctionType) -> bool:
        pending = [function]
        seen = set()
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            current_is_library = is_library_module(current.__module__)
            if imported_modules_reach_explicit_input(
                current, reject_module_values=not current_is_library
            ):
                return True
            for value, _ in referenced_python_functions(
                [item for _, item, _, _, _ in referenced_globals(current)]
            ):
                pending.append(value)
        return False

    def module_values(value: object, seen: set[int]) -> list[types.ModuleType]:
        if isinstance(value, types.ModuleType):
            return [value]
        value_id = id(value)
        if value_id in seen:
            return []
        seen.add(value_id)
        if isinstance(value, (dict, MappingProxyType)):
            children = [item for pair in value.items() for item in pair]
        elif isinstance(value, (tuple, list, set, frozenset)):
            children = list(value)
        elif isinstance(value, types.MethodType):
            children = [value.__func__, value.__self__]
        elif isinstance(value, functools.partial):
            children = [value.func, *value.args, *(value.keywords or {}).values()]
        elif isinstance(value, types.FunctionType):
            children = [
                *(value.__defaults__ or ()),
                *((value.__kwdefaults__ or {}).values()),
                *value.__dict__.values(),
            ]
            for cell in value.__closure__ or ():
                try:
                    children.append(cell.cell_contents)
                except ValueError:
                    pass
        elif isinstance(value, (CodeType, type, torch.Tensor)):
            return []
        elif is_library_module(type(value).__module__) and not isinstance(
            value, types.SimpleNamespace
        ):
            return []
        else:
            children = _dynamo_object_state_values(value)
        return [module for child in children for module in module_values(child, seen)]

    def contains_module_value(value: object, seen: set[int]) -> bool:
        return bool(module_values(value, seen))

    def contains_tensor(value: object, seen: set[int]) -> bool:
        if isinstance(value, torch.Tensor):
            return True
        value_id = id(value)
        if value_id in seen:
            return False
        seen.add(value_id)
        if isinstance(value, (dict, MappingProxyType)):
            children = [item for pair in value.items() for item in pair]
        elif isinstance(value, (tuple, list, set, frozenset)):
            children = list(value)
        elif isinstance(value, types.MethodType):
            children = [value.__func__, value.__self__]
        elif isinstance(value, (classmethod, staticmethod)):
            children = [value.__func__]
        elif isinstance(value, property):
            children = [
                function
                for function in (value.fget, value.fset, value.fdel)
                if function is not None
            ]
        elif isinstance(value, types.FunctionType):
            if is_library_module(value.__module__):
                return False
            children = [
                *(value.__defaults__ or ()),
                *((value.__kwdefaults__ or {}).values()),
                *(item for _, item, _, _, _ in referenced_globals(value)),
            ]
            for imported_code, local_name, root, unresolved in imported_bindings(
                value.__code__, value.__globals__
            ):
                if unresolved or local_name is None:
                    continue
                paths = loaded_bound_paths(imported_code, local_name)
                if any(
                    instruction.opname.startswith("LOAD_")
                    and instruction.argval == local_name
                    for instruction in dis.get_instructions(imported_code)
                ):
                    paths.append(())
                for path in paths:
                    imported_value, _, _, path_unresolved = resolve_static_path(
                        root, path
                    )
                    if not path_unresolved:
                        children.append(imported_value)
            for cell in value.__closure__ or ():
                try:
                    children.append(cell.cell_contents)
                except ValueError:
                    pass
        elif isinstance(value, type):
            if is_library_module(value.__module__):
                return False
            children = [item for cls in value.__mro__ for item in vars(cls).values()]
        elif isinstance(value, (CodeType, types.ModuleType)):
            return False
        else:
            children = _dynamo_object_state_values(value)
        return any(contains_tensor(child, seen) for child in children)

    def accessed_receiver_contains_tensor(
        callable_value: object, receiver: object
    ) -> bool:
        for function, _ in referenced_python_functions([callable_value]):
            code = function.__code__
            if not code.co_argcount:
                continue
            for path in loaded_local_paths(code, code.co_varnames[0]):
                value = receiver
                try:
                    for attribute in path:
                        value = inspect.getattr_static(value, attribute)
                except AttributeError:
                    continue
                if contains_tensor(value, set()):
                    return True
        return False

    if contains_tensor((target.__defaults__, target.__kwdefaults__), set()):
        raise PrecompileError(
            "precompile tracer='dynamo' cannot serialize tensor-valued function "
            "defaults; remove the default and pass every tensor as an explicit input."
        )
    defaults = (
        *(target.__defaults__ or ()),
        *((target.__kwdefaults__ or {}).values()),
    )
    if not all(is_literal(value) for value in defaults):
        raise PrecompileError(
            "precompile tracer='dynamo' cannot serialize non-literal function defaults; "
            "remove the default and pass mutable or user-defined values explicitly."
        )

    def dynamic_globals_reach_explicit_input(
        globals_scope: dict[str, object],
        ignored_input_ids: set[int] | None = None,
    ) -> bool:
        values = list(globals_scope.values())
        leaves = []
        seen_modules = set()
        while values:
            value = values.pop()
            if isinstance(value, types.ModuleType) and not is_library_module(
                value.__name__
            ):
                if id(value) in seen_modules:
                    continue
                seen_modules.add(id(value))
                values.extend(vars(value).values())
            else:
                leaves.append(value)
        return bool(
            _dynamo_reachable_object_ids(leaves, skip_literals=True)
            & (explicit_input_ids - (ignored_input_ids or set()))
        )

    def dynamic_module_access_reaches_explicit_input(
        function: types.FunctionType,
        ignored_input_ids: set[int] | None = None,
    ) -> bool:
        paths = loaded_global_paths(function.__code__)
        dynamic_builtin_lookup = any(
            path[0] in ("getattr", "vars") and path[0] not in function.__globals__
            for path in paths
        )
        modules = []
        for path in paths:
            if path[0] not in function.__globals__:
                continue
            root = function.__globals__[path[0]]
            if not isinstance(root, types.ModuleType):
                continue
            if dynamic_builtin_lookup or any(
                name in ("__getattr__", "__getattribute__") for name in path[1:]
            ):
                modules.append(root)
        values = [
            value
            for module in modules
            for value in vars(module).values()
            if not isinstance(value, (CodeType, type, types.ModuleType))
        ]
        return bool(
            _dynamo_reachable_object_ids(values, skip_literals=True)
            & (explicit_input_ids - (ignored_input_ids or set()))
        )

    def receiver_reaches_explicit_input(
        receiver: object, seen: set[tuple[int, int | None, int, bool]]
    ) -> bool:
        return reaches_explicit_input(receiver, seen, scan_receiver=True)

    def reaches_explicit_input(
        value: object,
        seen: set[tuple[int, int | None, int, bool]],
        receiver: object | None = None,
        *,
        scan_receiver: bool = False,
        ignored_input_ids: set[int] | None = None,
        scan_dynamic_globals: bool = False,
    ) -> bool:
        nonlocal scan_all_input_behaviors
        ignored = ignored_input_ids or set()
        work = [(value, receiver, scan_receiver, target.__globals__, True)]
        scanned_receivers: set[tuple[int, int, bool]] = set()
        dynamic_globals_seen: set[str] = set()
        dynamic_global_builtins = (
            ("eval", eval),
            ("exec", exec),
            ("getattr", getattr),
            ("globals", globals),
            ("attrgetter", operator.attrgetter),
            ("itemgetter", operator.itemgetter),
            ("locals", locals),
            ("methodcaller", operator.methodcaller),
            ("vars", vars),
        )
        while work:
            value, receiver, scan_receiver, globals_scope, may_enter_external = (
                work.pop()
            )
            if scan_receiver:
                receiver_key = (id(value), id(globals_scope), may_enter_external)
                if receiver_key in scanned_receivers:
                    continue
                scanned_receivers.add(receiver_key)
                work.append((value, None, False, globals_scope, may_enter_external))
                if isinstance(value, types.ModuleType):
                    module_library = is_library_module(value.__name__)
                    receiver_values = (
                        (vars(value).get("__getattr__"),)
                        if module_library
                        else vars(value).values()
                    )
                    work.extend(
                        (
                            item,
                            None,
                            False,
                            globals_scope,
                            may_enter_external and not module_library,
                        )
                        for item in receiver_values
                        if item is not None
                    )
                receiver_type = value if isinstance(value, type) else type(value)
                receiver_library = is_library_module(receiver_type.__module__)
                work.extend(
                    (
                        item,
                        None,
                        False,
                        globals_scope,
                        may_enter_external and not receiver_library,
                    )
                    for cls in receiver_type.__mro__
                    for item in vars(cls).values()
                    if not is_library_module(cls.__module__)
                )
                continue

            value_id = id(value)
            receiver_id = None if receiver is None else id(receiver)
            if (value_id in explicit_input_ids and value_id not in ignored) or (
                receiver_id in explicit_input_ids and receiver_id not in ignored
            ):
                return True
            key = (value_id, receiver_id, id(globals_scope), may_enter_external)
            if key in seen:
                continue
            seen.add(key)
            dynamic_builtin_name = next(
                (name for name, builtin in dynamic_global_builtins if value is builtin),
                None,
            )
            bound_self = getattr(value, "__self__", None)
            dynamic_name = getattr(value, "__name__", None)
            if (
                dynamic_builtin_name is None
                and isinstance(bound_self, types.ModuleType)
                and dynamic_name in ("__getattr__", "__getattribute__")
            ):
                dynamic_builtin_name = dynamic_name
            if may_enter_external and dynamic_builtin_name is not None:
                if scan_dynamic_globals:
                    if dynamic_globals_reach_explicit_input(globals_scope, ignored):
                        return True
                else:
                    dynamic_globals_seen.add(dynamic_builtin_name)
                continue
            if isinstance(value, dict):
                items = [item for pair in value.items() for item in pair]
                if any(
                    id(item) in explicit_input_ids and id(item) not in ignored
                    for item in items
                ):
                    return True
                work.extend(
                    (item, None, False, globals_scope, may_enter_external)
                    for item in items
                )
            if isinstance(value, (tuple, list, set, frozenset)):
                if any(
                    id(item) in explicit_input_ids and id(item) not in ignored
                    for item in value
                ):
                    return True
                work.extend(
                    (item, None, False, globals_scope, may_enter_external)
                    for item in value
                )
            if isinstance(value, weakref.ProxyTypes):
                if may_enter_external:
                    return True
                continue
            if isinstance(value, weakref.ReferenceType):
                referent = value()
                callback = value.__callback__
                if referent is not None:
                    work.append(
                        (referent, None, False, globals_scope, may_enter_external)
                    )
                if callback is not None:
                    work.append(
                        (callback, None, False, globals_scope, may_enter_external)
                    )
                continue
            if isinstance(value, types.MethodType):
                work.append(
                    (
                        value.__func__,
                        value.__self__,
                        False,
                        globals_scope,
                        may_enter_external,
                    )
                )
                continue
            if isinstance(value, classmethod):
                bound = (
                    receiver
                    if isinstance(receiver, type) or receiver is None
                    else type(receiver)
                )
                work.append(
                    (value.__func__, bound, False, globals_scope, may_enter_external)
                )
                continue
            if isinstance(value, staticmethod):
                work.append(
                    (
                        value.__func__,
                        None,
                        False,
                        globals_scope,
                        may_enter_external,
                    )
                )
                continue
            if isinstance(value, property):
                if value.fget is not None:
                    work.append(
                        (
                            value.fget,
                            receiver,
                            False,
                            globals_scope,
                            may_enter_external,
                        )
                    )
                continue
            if isinstance(value, types.FunctionType):
                function_globals = value.__globals__
                next_scope = function_globals
                library_function = function_globals is not target.__globals__ and (
                    is_library_module(function_globals.get("__name__"))
                )
                if not library_function and imported_modules_reach_explicit_input(
                    value
                ):
                    return True
                if (
                    not library_function
                    and dynamic_module_access_reaches_explicit_input(value)
                ):
                    return True
                function_references = referenced_globals(value)
                if (
                    any(
                        aliases_input
                        for _, _, _, aliases_input, _ in function_references
                    )
                    or _dynamo_reachable_object_ids(
                        [item for _, item, _, _, _ in function_references],
                        skip_literals=True,
                    )
                    & explicit_input_ids
                ):
                    return True
                if not library_function:
                    referenced_modules = [
                        module
                        for _, item, _, _, unresolved in function_references
                        if not unresolved
                        for module in module_values(item, set())
                    ]
                    if any(
                        container_reaches_explicit_input(list(vars(module).values()))
                        for module in referenced_modules
                    ):
                        return True
                    if referenced_modules:
                        raise PrecompileError(
                            "precompile tracer='dynamo' cannot analyze a module object "
                            "passed or aliased as a value; access its attributes directly. "
                            f"Function: {value.__qualname__!r}; module: "
                            f"{referenced_modules[0].__name__!r}."
                        )
                function_behavior_names = loaded_attribute_names(value.__code__)
                if not library_function or function_module_reaches_explicit_input(
                    value
                ):
                    input_behavior_names.update(function_behavior_names)
                next_may_enter_external = (
                    may_enter_external and not library_function
                ) or function_globals is target.__globals__
                if (
                    not library_function
                    and "__globals__" in function_behavior_names
                    and _dynamo_reachable_object_ids(
                        list(function_globals.values()), skip_literals=True
                    )
                    & (explicit_input_ids - ignored)
                ):
                    return True
                captured = (
                    *(value.__defaults__ or ()),
                    *((value.__kwdefaults__ or {}).values()),
                    *value.__dict__.values(),
                    *(value.__annotations__.values() if not library_function else ()),
                )
                if (
                    _dynamo_reachable_object_ids(list(captured), skip_literals=True)
                    & explicit_input_ids
                ):
                    return True
                work.extend(
                    (
                        item,
                        None,
                        False,
                        next_scope,
                        next_may_enter_external,
                    )
                    for item in captured
                )
                closure_values = []
                for cell in value.__closure__ or ():
                    try:
                        closure_values.append(cell.cell_contents)
                        work.append(
                            (
                                cell.cell_contents,
                                None,
                                False,
                                next_scope,
                                next_may_enter_external,
                            )
                        )
                    except ValueError:
                        pass
                if (
                    _dynamo_reachable_object_ids(closure_values, skip_literals=True)
                    & explicit_input_ids
                ):
                    return True
                dynamic_globals = loaded_global_names(value.__code__) & {
                    "eval",
                    "exec",
                    "getattr",
                    "globals",
                    "locals",
                    "vars",
                }
                if dynamic_globals and (next_may_enter_external or library_function):
                    if library_function:
                        if dynamic_globals_reach_explicit_input(next_scope):
                            return True
                        scan_all_input_behaviors = True
                    elif scan_dynamic_globals:
                        if dynamic_globals_reach_explicit_input(next_scope, ignored):
                            return True
                    else:
                        dynamic_globals_seen.update(dynamic_globals)
                if receiver is not None and value.__code__.co_argcount:
                    local_name = value.__code__.co_varnames[0]
                    for path in loaded_local_paths(value.__code__, local_name):
                        item, item_receiver, aliases_input, unresolved = (
                            resolve_static_path(receiver, path, ignored)
                        )
                        if aliases_input:
                            return True
                        work.append(
                            (
                                item,
                                None,
                                True,
                                next_scope,
                                next_may_enter_external,
                            )
                            if unresolved
                            else (
                                item,
                                item_receiver,
                                False,
                                next_scope,
                                next_may_enter_external,
                            )
                        )
                    if has_unmodeled_local_access(value.__code__, local_name):
                        work.append(
                            (
                                receiver,
                                None,
                                True,
                                next_scope,
                                next_may_enter_external,
                            )
                        )
                if library_function:
                    library_values = [item for _, item, _, _, _ in function_references]
                    library_leaves = library_value_leaves(library_values)
                    for item in library_leaves:
                        if not isinstance(item, type):
                            continue
                        module = sys.modules.get(item.__module__)
                        if module is not None and container_reaches_explicit_input(
                            list(vars(module).values())
                        ):
                            return True
                    for _, item, _, _, unresolved in function_references:
                        if not unresolved or not isinstance(item, types.ModuleType):
                            continue
                        module_getattr = vars(item).get("__getattr__")
                        if isinstance(module_getattr, types.FunctionType):
                            work.append(
                                (
                                    module_getattr,
                                    None,
                                    False,
                                    module_getattr.__globals__,
                                    False,
                                )
                            )
                    referenced_modules = [
                        item
                        for _, value, _, _, unresolved in function_references
                        if not unresolved
                        for item in library_value_leaves([value])
                        if isinstance(item, types.ModuleType)
                    ]
                    if any(
                        container_reaches_explicit_input(list(vars(module).values()))
                        for module in referenced_modules
                    ):
                        return True
                    if any(
                        item is builtin
                        for item in library_values
                        for _, builtin in dynamic_global_builtins
                    ):
                        scan_all_input_behaviors = True
                    if (
                        _dynamo_reachable_object_ids(library_values, skip_literals=True)
                        & explicit_input_ids
                    ):
                        return True
                    for function, function_receiver in referenced_python_functions(
                        library_values
                    ):
                        if is_library_module(
                            function.__module__
                        ) and not function_module_reaches_explicit_input(function):
                            continue
                        work.append(
                            (
                                function,
                                function_receiver,
                                False,
                                function.__globals__,
                                False,
                            )
                        )
                    if not function_module_reaches_explicit_input(value):
                        continue
                    for (
                        _,
                        item,
                        item_receiver,
                        aliases_input,
                        unresolved,
                    ) in function_references:
                        if aliases_input:
                            return True
                        work.append(
                            (
                                item,
                                None,
                                True,
                                next_scope,
                                False,
                            )
                            if unresolved
                            else (
                                item,
                                item_receiver,
                                False,
                                next_scope,
                                False,
                            )
                        )
                    continue
                for (
                    _,
                    item,
                    item_receiver,
                    aliases_input,
                    unresolved,
                ) in function_references:
                    if aliases_input:
                        return True
                    work.append(
                        (
                            item,
                            None,
                            True,
                            next_scope,
                            next_may_enter_external,
                        )
                        if unresolved
                        else (
                            item,
                            item_receiver,
                            False,
                            next_scope,
                            next_may_enter_external,
                        )
                    )
                continue
            if receiver is not None:
                descriptor_get = inspect.getattr_static(type(value), "__get__", None)
                if descriptor_get is not None:
                    work.append(
                        (
                            descriptor_get,
                            value,
                            False,
                            globals_scope,
                            may_enter_external,
                        )
                    )
                    work.append(
                        (receiver, None, True, globals_scope, may_enter_external)
                    )
            if isinstance(value, torch.Tensor):
                continue
            if isinstance(value, types.ModuleType):
                continue
            elif isinstance(value, type):
                value_type = value
            else:
                work.extend(
                    (item, None, False, globals_scope, may_enter_external)
                    for item in _dynamo_object_state_values(value)
                )
                value_type = type(value)
            if isinstance(value, type):
                class_state = [
                    item for cls in value.__mro__ for item in vars(cls).values()
                ]
                if _dynamo_reachable_object_ids(class_state, skip_literals=True) & (
                    explicit_input_ids - ignored
                ):
                    return True
                work.extend(
                    (item, None, False, globals_scope, may_enter_external)
                    for cls in value.__mro__
                    for name, item in vars(cls).items()
                    if name in ("__class_getitem__", "__init__", "__new__")
                )
                work.extend(
                    (item, value, False, globals_scope, may_enter_external)
                    for cls in type(value).__mro__
                    for name, item in vars(cls).items()
                    if name.startswith("__")
                    and name.endswith("__")
                    and name not in ("__del__", "__init__", "__new__")
                )
                continue
            if value_type.__module__ != "builtins":
                behavior_library = is_library_module(value_type.__module__)
                behavior_type_is_importable = importable_global(value_type) is not None
                work.extend(
                    (
                        item,
                        None,
                        False,
                        globals_scope,
                        may_enter_external and not behavior_library,
                    )
                    for cls in value_type.__mro__
                    if not is_library_module(cls.__module__)
                    for item in vars(cls).values()
                )
                if behavior_library:
                    work.extend(
                        (item, value, False, globals_scope, may_enter_external)
                        for cls in value_type.__mro__
                        if is_library_module(cls.__module__)
                        for name, item in vars(cls).items()
                        if (
                            name in input_behavior_names
                            or (
                                not behavior_type_is_importable
                                and name.startswith("__")
                                and name.endswith("__")
                            )
                        )
                        and name not in ("__del__", "__init__", "__new__")
                    )
        if dynamic_globals_seen:
            names = ", ".join(sorted(dynamic_globals_seen))
            raise PrecompileError(
                "precompile tracer='dynamo' cannot preserve an input-derived "
                "identity that aliases the Python environment through dynamic "
                f"global access via {names}."
            )
        return False

    def target_environment_aliases() -> tuple[
        list[tuple[str, object, object | None, bool, bool]], list[str]
    ]:
        references = referenced_globals(target)
        aliases = [
            name
            for name, value, receiver, aliases_input, unresolved in references
            if aliases_input
            or (
                receiver_reaches_explicit_input(value, set())
                if unresolved
                else reaches_explicit_input(value, set(), receiver)
            )
        ]
        if imported_modules_reach_explicit_input(target):
            aliases.append("locally imported module")
        return references, aliases

    target_references, aliased_globals = target_environment_aliases()
    if aliased_globals:
        raise PrecompileError(
            "precompile tracer='dynamo' cannot preserve an input-derived identity "
            "relation to the Python environment; pass the value only as an input. "
            f"Aliased global: {aliased_globals[0]!r}."
        )
    if dynamic_module_access_reaches_explicit_input(target):
        raise PrecompileError(
            "precompile tracer='dynamo' cannot preserve an input-derived identity "
            "relation reached through dynamic module attribute access."
        )
    bare_module_globals = [
        name
        for name, value, _, _, unresolved in target_references
        if not unresolved and contains_module_value(value, set())
    ]
    if bare_module_globals:
        raise PrecompileError(
            "precompile tracer='dynamo' cannot analyze a module object passed or "
            "aliased as a value; access its attributes directly. Referenced module: "
            f"{bare_module_globals[0]!r}."
        )
    tensor_globals = [
        name
        for name, value, receiver, _, _ in target_references
        if contains_tensor(value, set())
        or (receiver is not None and accessed_receiver_contains_tensor(value, receiver))
    ]
    if tensor_globals or contains_tensor(target, set()):
        reference = tensor_globals[0] if tensor_globals else target.__qualname__
        raise PrecompileError(
            "precompile tracer='dynamo' cannot capture tensor-valued Python globals; "
            "pass every tensor as an explicit input. Referenced global: "
            f"{reference!r}."
        )

    input_objects: dict[int, object] = {}
    pending_inputs = [
        value
        for example in examples
        for value in (*example.args, *example.kwargs.values())
    ]
    if isinstance(fn, torch.nn.Module):
        pending_inputs.append(fn)
    seen_inputs = set()
    while pending_inputs:
        value = pending_inputs.pop()
        value_id = id(value)
        if value_id in seen_inputs:
            continue
        seen_inputs.add(value_id)
        if isinstance(value, torch.Tensor):
            continue
        if isinstance(value, (dict, MappingProxyType)):
            if type(value) not in (dict, MappingProxyType):
                input_objects[value_id] = value
            pending_inputs.extend(value.keys())
            pending_inputs.extend(value.values())
            continue
        if isinstance(value, (tuple, list, set, frozenset)):
            if type(value) not in (tuple, list, set, frozenset):
                input_objects[value_id] = value
            pending_inputs.extend(value)
            continue
        if isinstance(value, (CodeType, types.FunctionType, types.ModuleType)):
            continue
        if isinstance(value, types.GeneratorType):
            continue
        if isinstance(value, torch.nn.Module):
            pending_inputs.extend(value.modules())
            pending_inputs.extend(value.parameters())
            pending_inputs.extend(value.buffers())
        if isinstance(value, type) or type(value).__module__ != "builtins":
            input_objects[value_id] = value
        if not isinstance(value, type):
            pending_inputs.extend(_dynamo_object_state_values(value))

    input_behavior_object_ids = {
        object_id: _dynamo_reachable_object_ids([value], skip_literals=True)
        for object_id, value in input_objects.items()
    }

    def is_input_behavior(value: object) -> bool:
        if isinstance(value, (types.GetSetDescriptorType, types.MemberDescriptorType)):
            return False
        return (
            callable(value)
            or isinstance(value, (classmethod, property, staticmethod))
            or inspect.getattr_static(type(value), "__get__", None) is not None
        )

    selected_input_behaviors = [
        (behavior, value, input_behavior_object_ids[id(value)])
        for value in input_objects.values()
        for name in input_behavior_names | {"__call__"}
        if (behavior := inspect.getattr_static(type(value), name, None)) is not None
        and (
            name != "__call__"
            or (
                isinstance(behavior, (types.FunctionType, classmethod, staticmethod))
                and not is_library_module(
                    getattr(getattr(behavior, "__func__", behavior), "__module__", None)
                )
            )
        )
        and is_input_behavior(behavior)
    ]
    dunder_input_behaviors = [
        (behavior, value, input_behavior_object_ids[id(value)])
        for value in input_objects.values()
        for cls in type(value).__mro__
        if not is_library_module(cls.__module__)
        for name, behavior in vars(cls).items()
        if name.startswith("__")
        and name.endswith("__")
        and name not in ("__del__", "__init__", "__new__")
        and is_input_behavior(behavior)
    ]
    input_constructors = [
        constructor
        for value in input_objects.values()
        if isinstance(value, type)
        for name in ("__new__", "__init__")
        if (constructor := inspect.getattr_static(value, name, None)) is not None
    ]
    mutation_roots = [
        function
        for behavior, _, _ in (*selected_input_behaviors, *dunder_input_behaviors)
        for function, _ in referenced_python_functions([behavior])
        if not is_library_defined_function(function)
    ]
    mutation_roots.extend(
        function
        for constructor in input_constructors
        for function, _ in referenced_python_functions([constructor])
        if not is_library_defined_function(function)
    )
    validate_no_global_mutation(mutation_roots)

    input_behaviors = list(selected_input_behaviors)
    identity_input_behaviors = [*selected_input_behaviors, *dunder_input_behaviors]
    if scan_all_input_behaviors:
        all_input_behaviors = [
            (behavior, value, input_behavior_object_ids[id(value)])
            for value in input_objects.values()
            for cls in type(value).__mro__
            if not is_library_module(cls.__module__)
            for name, behavior in vars(cls).items()
            if name not in ("__del__", "__init__", "__new__")
            and is_input_behavior(behavior)
        ]
        validate_no_global_mutation(
            [
                function
                for behavior, _, _ in all_input_behaviors
                for function, _ in referenced_python_functions([behavior])
                if not is_library_defined_function(function)
            ]
        )
        input_behaviors.extend(all_input_behaviors)
        identity_input_behaviors.extend(all_input_behaviors)
    input_behavior_environment_identity = False
    input_dependency = (True, False, frozenset((("<input-value>",),)))
    for behavior, _, _ in identity_input_behaviors:
        for function, _ in referenced_python_functions([behavior]):
            if is_library_defined_function(function):
                continue
            function_code = function.__code__
            positional_names = list(
                function_code.co_varnames[: function_code.co_argcount]
            )
            parameter_count = (
                function_code.co_argcount + function_code.co_kwonlyargcount
            )
            parameter_dependencies = {
                name: (False, False, frozenset())
                for name in function_code.co_varnames[:parameter_count]
            }
            defaults = function.__defaults__ or ()
            for name, value in zip(
                positional_names[len(positional_names) - len(defaults) :], defaults
            ):
                parameter_dependencies[name] = environment_value_dependency(value)
            for name, value in (function.__kwdefaults__ or {}).items():
                parameter_dependencies[name] = environment_value_dependency(value)
            if positional_names:
                parameter_dependencies[positional_names[0]] = input_dependency
            input_behavior_environment_identity |= bytecode_dependencies(
                function_code,
                function.__globals__,
                parameter_dependencies,
            )[0]
    for function, receiver, receiver_input_ids in input_behaviors:
        behavior = getattr(function, "__func__", function)
        receiver_type = receiver if isinstance(receiver, type) else type(receiver)
        if is_library_module(getattr(behavior, "__module__", None)) and (
            importable_global(receiver_type) is not None
        ):
            continue
        if reaches_explicit_input(
            function,
            set(),
            receiver,
            ignored_input_ids=receiver_input_ids,
            scan_dynamic_globals=True,
        ):
            raise PrecompileError(
                "precompile tracer='dynamo' cannot preserve an input-derived identity "
                "relation between an input callable and the Python environment."
            )
    for behavior, behavior_receiver, _ in identity_input_behaviors:
        for function, receiver in referenced_python_functions([behavior]):
            if is_library_defined_function(function):
                continue
            behavior_owner = receiver if receiver is not None else behavior_receiver
            if contains_tensor(function, set()) or (
                behavior_owner is not None
                and accessed_receiver_contains_tensor(function, behavior_owner)
            ):
                raise PrecompileError(
                    "precompile tracer='dynamo' cannot serialize tensor-valued "
                    "input behavior state; pass every tensor as an explicit input."
                )

    def disabled_function_state(
        global_name: str, function: Callable[..., object]
    ) -> _DynamoDisabledFunction:
        original = getattr(function, "_torchdynamo_orig_callable", function)
        if not inspect.isfunction(original) or original.__closure__ is not None:
            raise NotImplementedError(
                "precompile tracer='dynamo' graph breaks currently require "
                f"{global_name!r} to wrap a closure-free Python function."
            )
        dynamic_globals = loaded_global_names(original.__code__) & {
            "eval",
            "exec",
            "globals",
        }
        if dynamic_globals:
            names = ", ".join(sorted(dynamic_globals))
            raise NotImplementedError(
                "precompile tracer='dynamo' disabled functions cannot use dynamic "
                f"global access through {names}."
            )
        validate_no_global_mutation([original])
        module_globals: dict[str, str] = {}
        value_globals: dict[str, object] = {}
        for name in loaded_global_names(original.__code__):
            if name not in original.__globals__:
                continue
            value = original.__globals__[name]
            if isinstance(value, types.ModuleType):
                binding = importable_global(value)
                if binding is None:
                    raise NotImplementedError(
                        "precompile tracer='dynamo' cannot serialize non-importable "
                        f"module global {name!r} used by disabled function "
                        f"{global_name!r}."
                    )
                module_globals[name] = binding[0]
            elif is_literal(value):
                value_globals[name] = value
            else:
                raise NotImplementedError(
                    "precompile tracer='dynamo' cannot serialize global "
                    f"{name!r} used by disabled function {global_name!r}; only "
                    "importable modules and recursive literal values are supported."
                )
        defaults = original.__defaults__
        kwdefaults = original.__kwdefaults__
        if not is_literal(defaults) or not is_literal(
            tuple((kwdefaults or {}).values())
        ):
            raise NotImplementedError(
                "precompile tracer='dynamo' disabled-function defaults must contain "
                "only recursive literal values."
            )
        return _DynamoDisabledFunction(
            code=SerializedCode.from_code_object(original.__code__),
            name=original.__name__,
            defaults=defaults,
            kwdefaults=kwdefaults,
            module_globals=module_globals,
            value_globals=value_globals,
        )

    runtime_examples = (
        [ExampleInput((fn, *example.args), example.kwargs) for example in examples]
        if isinstance(fn, torch.nn.Module)
        else examples
    )
    target_parameter_dependencies: dict[
        str, tuple[bool, bool, frozenset[tuple[str, ...]]]
    ] = {}
    supplied_parameter_counts: dict[str, int] = {}
    target_signature = inspect.signature(target)
    for example in runtime_examples:
        bound = target_signature.bind_partial(*example.args, **example.kwargs)
        for name, value in bound.arguments.items():
            supplied_parameter_counts[name] = supplied_parameter_counts.get(name, 0) + 1
            input_marker = (
                "<input-container>" if is_identity_container(value) else "<input-value>"
            )
            markers: set[tuple[str, ...]] = {
                (input_marker,),
                (
                    "<input-type>",
                    "tensor" if isinstance(value, torch.Tensor) else "value",
                    type(value).__module__,
                    type(value).__qualname__,
                ),
            }
            if not is_safely_reflexive(value):
                markers.add(("<input-identity-sensitive>",))
                if is_identity_container(value):
                    markers.add(("<input-identity-sensitive-container>",))
            if is_identity_container(
                value
            ) and not is_safely_reflexive_container_lookup(value):
                markers.add(("<input-lookup-identity-sensitive-container>",))
            if is_identity_mapping(value):
                markers.add(("<input-mapping>",))
                if not all(
                    is_safely_reflexive(item)
                    for item in cast("Mapping[object, object]", value)
                ):
                    markers.add(("<input-mapping-key-identity-sensitive>",))
            markers.update(
                ("<input-object>", str(object_id))
                for object_id in _dynamo_reachable_object_ids(
                    [value], skip_literals=True
                )
                if object_id in input_objects
            )
            previous = target_parameter_dependencies.get(name)
            paths = frozenset(markers)
            target_parameter_dependencies[name] = (
                True,
                False,
                paths if previous is None else previous[2] | paths,
            )
    for name, parameter in target_signature.parameters.items():
        if (
            parameter.default is inspect.Parameter.empty
            or supplied_parameter_counts.get(name, 0) == len(runtime_examples)
        ):
            continue
        default_dependency = environment_value_dependency(parameter.default)
        previous = target_parameter_dependencies.get(name)
        if previous is None:
            target_parameter_dependencies[name] = default_dependency
        else:
            target_parameter_dependencies[name] = (
                previous[0],
                True,
                previous[2] | default_dependency[2],
            )
    target_input_environment_identity = bytecode_dependencies(
        target.__code__,
        target.__globals__,
        target_parameter_dependencies,
    )[0]
    input_contract = _dynamo_input_contract(runtime_examples)

    _DYNAMO_COMPILE_LOCK.acquire()
    package: CompilePackage | None = None
    region = -1
    pgo_state = _new_code_state()
    capture_stack = contextlib.ExitStack()
    captured_guard_sets: dict[int, list[_DynamoCapturedGuardSet]] = {}
    current_example_index: int | None = None
    contract_dropped_guards: set[tuple[str, str]] = set()
    capture_errors: list[str] = []
    truncated: set[str] = set()
    generated_prefixes = ("__compiled_fn_", "__builtins_dict___", "__import_")
    existing_generated_globals = {
        name for name in target.__globals__ if name.startswith(generated_prefixes)
    }
    try:
        accumulated_limit = max(
            torch._dynamo.config.accumulated_recompile_limit, recompile_limit
        )
        capture_stack.enter_context(
            torch._dynamo.config.patch(
                accumulated_recompile_limit=accumulated_limit,
                fail_on_recompile_limit_hit=True,
                allow_empty_graphs=True,
                trace_autograd_ops=training,
            )
        )
        functorch_options = {"bundled_autograd_cache": True}
        if training:
            functorch_options["force_non_lazy_backward_lowering"] = True
        capture_stack.enter_context(functorch_config.patch(**functorch_options))
        capture_stack.enter_context(_use_code_state(pgo_state))
        capture_stack.enter_context(
            torch.enable_grad() if training else torch.no_grad()
        )

        def keep_portable_capture_guards(guards: Sequence[Any]) -> list[bool]:
            unsupported = CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
            current = package._current_entry if package is not None else None
            if current is None:
                raise AssertionError("Dynamo guard filter ran outside a package frame")
            if current_example_index is None:
                raise AssertionError("Dynamo guard filter ran outside an example call")
            input_environment_identity = (
                input_behavior_environment_identity or target_input_environment_identity
            )
            portable_guard_types = [
                guard.guard_type not in unsupported
                and not any(
                    derived in unsupported for derived in guard.derived_guard_types
                )
                for guard in guards
            ]
            # Explicit inputs, defaults, and values derived across graph breaks may
            # vary. Known process state, globals, and unsupported values outside that
            # input closure belong to the caller-promised invariant environment.
            environment_assumptions = [
                guard.guard_type in _DYNAMO_ENVIRONMENT_GUARD_TYPES
                or guard.source_root_is_import
                or (guard.has_value and guard.is_global and is_literal(guard.value))
                or (
                    guard.has_value
                    and guard.source_root_id is not None
                    and guard.source_root_id not in input_object_ids
                    and id(guard.value) not in input_object_ids
                    and importable_global(guard.value) is not None
                )
                or (
                    guard.source_root_id is not None
                    and guard.source_root_id not in input_object_ids
                    and (not guard.has_value or id(guard.value) not in input_object_ids)
                    and (guard.is_global or guard.source_has_unsupported_value)
                )
                for guard in guards
            ]
            chosen = (
                [True] * len(guards)
                if guard_filter_fn is None
                else list(guard_filter_fn(guards))
            )
            if len(chosen) != len(guards):
                raise ValueError(
                    f"guard_filter_fn returned {len(chosen)} decisions for "
                    f"{len(guards)} guards; it must return one per guard."
                )
            if not all(type(decision) is bool for decision in chosen):
                raise TypeError("guard_filter_fn decisions must all be bool values.")
            decisions = [
                guard_type_is_portable
                and not guard.source_has_unsupported_value
                and keep
                for guard, guard_type_is_portable, keep in zip(
                    guards, portable_guard_types, chosen, strict=True
                )
            ]
            if input_environment_identity:
                raise PrecompileError(
                    "precompile tracer='dynamo' cannot preserve an input-derived "
                    "identity relation to the Python environment."
                )
            dropped = set()
            risky = set()
            facts = []
            records = zip(
                guards,
                portable_guard_types,
                environment_assumptions,
                chosen,
                decisions,
                strict=True,
            )
            for (
                entry,
                guard_type_is_portable,
                environment_assumption,
                selected,
                keep,
            ) in records:
                slot = (
                    entry.guard_type,
                    _normalize_dynamo_guard_text(entry.name),
                )
                facts.append(_dynamo_guard_fact(entry, enforced=keep))
                if keep:
                    continue
                dropped.add(slot)
                if environment_assumption:
                    contract_dropped_guards.add(slot)
                    continue
                if not selected:
                    risky.add(slot)
                elif not guard_type_is_portable or entry.source_has_unsupported_value:
                    value = entry.value if entry.has_value else None
                    synthesized = entry.name.startswith(
                        ("__nested_resume_fns", "__nested_frame_values")
                    )
                    if not synthesized and not (
                        entry.is_global and importable_global(value) is not None
                    ):
                        risky.add(slot)
            captured_guard_sets.setdefault(id(current), []).append(
                _DynamoCapturedGuardSet(
                    example_index=current_example_index,
                    facts=tuple(facts),
                    dropped=frozenset(dropped),
                    risky_dropped=frozenset(risky),
                    environment=frozenset(
                        slot
                        for slot in {
                            (
                                entry.guard_type,
                                _normalize_dynamo_guard_text(entry.name),
                            )
                            for entry in guards
                        }
                        if all(
                            is_environment
                            for entry, is_environment in zip(
                                guards, environment_assumptions, strict=True
                            )
                            if (
                                entry.guard_type,
                                _normalize_dynamo_guard_text(entry.name),
                            )
                            == slot
                        )
                    ),
                )
            )
            return decisions

        package = CompilePackage(
            target_callable,
            serialization_guard_filter_fn=keep_portable_capture_guards,
        )
        context = torch._dynamo.optimize(
            backend=_dynamo_backend_compiler(backend, training),
            nopython=False,
            package=package,
            dynamic=dynamic,
            recompile_limit=recompile_limit,
            isolate_recompiles=True,
        )
        region = context._isolate_recompiles_id  # type: ignore[attr-defined]
        compiled = context(fn)
        saved_grads = _dynamo_example_grads(fn, examples)
        try:
            with torch.no_grad():
                for tensor, _ in saved_grads.values():
                    tensor.grad = None
            for example_index, example in enumerate(examples):
                current_example_index = example_index
                try:
                    compiled(*example.args, **example.kwargs)
                except (FailOnRecompileLimitHit, RecompileError) as e:
                    truncated.add(target.__qualname__)
                    capture_errors.append(f"{type(e).__name__}: {e}")
                    if require_complete:
                        raise
            current_example_index = None
            if contains_tensor(target, set()):
                raise PrecompileError(
                    "precompile tracer='dynamo' cannot capture tensor-valued Python "
                    "globals; pass every tensor as an explicit input."
                )
            if reachable_imports_reach_explicit_input(target):
                raise PrecompileError(
                    "precompile tracer='dynamo' cannot preserve an input-derived "
                    "identity relation to the Python environment; pass the value only "
                    "as an input. A locally imported module reaches that input."
                )
        finally:
            with torch.no_grad():
                for tensor, grad in saved_grads.values():
                    tensor.grad = grad

        for compiled_backend in package.cached_backends.values():
            if isinstance(compiled_backend, _DynamoPythonBackend):
                compiled_backend.finalize_training()

        cache_entry = package.cache_entry()
        code_entries = cache_entry.codes
        module_hint = (
            " Capture a Python function that calls the module and pass the module "
            "as an example argument."
            if isinstance(fn, torch.nn.Module)
            else ""
        )
        if not code_entries:
            raise PrecompileError(
                "precompile tracer='dynamo' did not capture a runnable entry frame."
                + module_hint
            )
        main_code = code_entries[0]
        if main_code.install_to_global or not main_code.guarded_codes:
            if main_code.bypassed:
                reason = getattr(main_code, "bypass_reason", None)
                detail = f" Dynamo reported: {reason}" if reason else ""
                raise PrecompileError(
                    "precompile tracer='dynamo' bypassed its entry frame during "
                    "capture, so there is no dispatchable artifact." + detail
                )
            raise PrecompileError(
                "precompile tracer='dynamo' did not capture a runnable entry frame."
                + module_hint
            )
        bypassed = [
            (
                f"{SerializedCode.to_code_object(code.python_code).co_name}: "
                f"{getattr(code, 'bypass_reason', None)}"
                if getattr(code, "bypass_reason", None)
                else SerializedCode.to_code_object(code.python_code).co_name
            )
            for code in code_entries
            if code.bypassed
        ]
        backend_ids = []
        for code in code_entries:
            for backend_id in code.backend_ids:
                if backend_id not in backend_ids:
                    backend_ids.append(backend_id)
        generated_function_names = {
            str(name)
            for code in code_entries
            if code.install_to_global
            for name in code.function_names
        }
        generated_global_names = {*backend_ids, *generated_function_names}
        compiled_backends = []
        for backend_id in backend_ids:
            compiled_backend = package.cached_backends.get(backend_id)
            if not isinstance(compiled_backend, _DynamoPythonBackend):
                raise PrecompileError(
                    "precompile tracer='dynamo' encountered a graph that could not be "
                    "represented as standalone Python source."
                )
            compiled_backends.append(compiled_backend)

        code_states: list[_DynamoCodeState] = []
        disabled_functions: dict[str, _DynamoDisabledFunction] = {}
        filtered_code_entries = []
        kept_by_entry: dict[int, list[frozenset[tuple[str, str]]]] = {}
        policy_dropped_guards = set(contract_dropped_guards)
        unportable_globals: set[tuple[str, str]] = set()
        for index, code in enumerate(code_entries):
            original_code = SerializedCode.to_code_object(code.python_code)
            runtime_globals = sys.modules[code.python_module].__dict__
            runtime_codes = [
                SerializedCode.to_code_object(guarded.dynamo_code)
                for guarded in code.guarded_codes
            ]
            if not runtime_codes:
                runtime_codes.append(original_code)
            global_bindings: dict[str, tuple[str, tuple[str, ...]]] = {}
            value_globals: dict[str, object] = {}
            for runtime_code in runtime_codes:
                for name in loaded_global_names(runtime_code):
                    if name in generated_global_names:
                        continue
                    if name not in runtime_globals:
                        continue
                    value = runtime_globals[name]
                    if getattr(value, "_torchdynamo_disable", False):
                        continue
                    if is_literal(value):
                        value_globals[name] = value
                        continue
                    binding = importable_global(value)
                    if binding is None:
                        unportable_globals.add((code.python_module, name))
                        continue
                    global_bindings[name] = binding
            finalized = _filter_dynamo_guards(
                original_code,
                runtime_globals,
                code.guarded_codes,
                captured_guard_sets.get(id(code), ()),
                package.live_guard_leaves(code),
            )
            filtered_guard_states = finalized.states
            kept_by_entry[id(code)] = list(finalized.kept_slots)
            policy_dropped_guards.update(finalized.policy_dropped)
            variants = tuple(
                _DynamoGuardedVariant(guards_state, guarded.dynamo_code)
                for guarded, guards_state in zip(
                    code.guarded_codes, filtered_guard_states
                )
            )
            filtered_code_entries.append(
                dataclasses.replace(
                    code,
                    guarded_codes=[
                        dataclasses.replace(guarded, guards_state=guards_state)
                        for guarded, guards_state in zip(
                            code.guarded_codes, filtered_guard_states
                        )
                    ],
                )
            )
            for runtime_code in runtime_codes:
                for name in loaded_global_names(runtime_code):
                    if name in generated_global_names:
                        continue
                    value = runtime_globals.get(name)
                    if not getattr(value, "_torchdynamo_disable", False):
                        continue
                    function_state = disabled_function_state(
                        name, cast("Callable[..., object]", value)
                    )
                    previous = disabled_functions.setdefault(name, function_state)
                    if previous != function_state:
                        raise PrecompileError(
                            "precompile tracer='dynamo' found conflicting disabled "
                            f"functions named {name!r}."
                        )
            code_states.append(
                _DynamoCodeState(
                    code=code.python_code,
                    python_module=code.python_module,
                    function_names=tuple(str(name) for name in code.function_names),
                    install_to_global=code.install_to_global,
                    code_source=code.code_source,
                    global_bindings=global_bindings,
                    value_globals=value_globals,
                    import_sources=dict(code.import_sources),
                    defaults=target.__defaults__ if index == 0 else None,
                    kwdefaults=target.__kwdefaults__ if index == 0 else None,
                    variants=variants,
                )
            )

        serving_mode = _dynamo_serving_mode(code_states)
        if serving_mode == "standalone" and unportable_globals:
            module_name, name = sorted(unportable_globals)[0]
            raise PrecompileError(
                "precompile tracer='dynamo' cannot make transformed global "
                f"{name!r} from {module_name!r} portable; use an importable value, "
                "a closure-free torch._dynamo.disable function, or a capture whose "
                "nested frame requires installed mode."
            )
        frame_invariants = _dynamo_frame_invariants(
            code_entries, captured_guard_sets, kept_by_entry
        )
        dropped_guards = {
            slot
            for records in captured_guard_sets.values()
            for record in records
            for slot in record.dropped
        }
        dropped_guards.update(policy_dropped_guards)
        risky_dropped_guards = {
            slot
            for records in captured_guard_sets.values()
            for record in records
            for slot in record.risky_dropped
        }
        kept_guards = {
            slot
            for variants in kept_by_entry.values()
            for slots in variants
            for slot in slots
        }
        uncovered = tuple(
            sorted(
                SerializedCode.to_code_object(code.python_code).co_name
                for code in code_entries
                if code.has_compile_id and not code.bypassed and not code.guarded_codes
            )
        )
        summary = PrecompileSummary(
            frames=len(code_entries),
            resume_functions=sum(code.install_to_global for code in code_entries),
            guarded_codes=sum(len(code.guarded_codes) for code in code_entries),
            backend_graphs=len(compiled_backends),
            bypassed=tuple(sorted(bypassed)),
            truncated=tuple(sorted(truncated)),
            uncovered_frames=uncovered,
            wont_generalize=_dynamo_wont_generalize(kept_guards),
            dropped_guards=tuple(sorted(dropped_guards)),
            kept_guards=tuple(sorted(kept_guards)),
            risky_dropped_guards=tuple(sorted(risky_dropped_guards)),
            policy_dropped_guards=tuple(sorted(policy_dropped_guards)),
            capture_errors=tuple(capture_errors),
            variant_examples=tuple(
                frame.variant_examples for frame in frame_invariants
            ),
        )
        if require_complete and not summary.complete:
            raise PrecompileError(
                "precompile tracer='dynamo' captured an incomplete artifact: "
                f"{summary}. Pass require_complete=False only after auditing the "
                "missing coverage."
            )
        if require_no_dropped_guards and summary.dropped_guards:
            raise PrecompileError(
                "precompile tracer='dynamo' dropped environment-contract, "
                "unserializable, or caller-filtered guards: "
                f"{list(summary.dropped_guards)}. Pass "
                "require_no_dropped_guards=False to accept them."
            )
        if require_no_risky_drops and summary.risky_dropped_guards:
            raise PrecompileError(
                "precompile tracer='dynamo' dropped guards that can affect "
                f"dispatch: {list(summary.risky_dropped_guards)}. Make the guarded "
                "value portable or pass require_no_risky_drops=False to accept the "
                "risk explicitly."
            )
        if summary.risky_dropped_guards:
            log.warning(
                "precompile: dropped guards can affect dispatch and are unchecked "
                "after load: %s",
                list(summary.risky_dropped_guards),
            )
        if summary.wont_generalize:
            log.warning(
                "precompile: values are pinned to the captured examples: %s",
                list(summary.wont_generalize),
            )
        if invariants is not None:
            _write_dynamo_invariants(invariants, target, frame_invariants)
        mutates_input_grads = any(
            _dynamo_code_writes_grad(SerializedCode.to_code_object(variant.dynamo_code))
            for code_state in code_states
            for variant in code_state.variants
        )
        state = _DynamoArtifactState(
            codes=tuple(code_states),
            disabled_functions=disabled_functions,
            input_contract=input_contract,
            serving_mode=serving_mode,
            entry_module=target.__module__,
            entry_qualname=target.__qualname__,
            entry_name=target.__code__.co_name,
            entry_firstlineno=target.__code__.co_firstlineno,
            device_type=cache_entry.device_type,
            system_info=cache_entry.system_info,
            mutates_input_grads=mutates_input_grads,
            recompile_limit=recompile_limit,
            dynamic=dynamic,
            summary=summary,
            package=(
                dataclasses.replace(cache_entry, codes=filtered_code_entries)
                if serving_mode == "installed"
                else None
            ),
        )
        python_code = _build_dynamo_python_source(
            backend=backend,
            training=training,
            state=state,
            backend_ids=[str(backend_id) for backend_id in backend_ids],
            compiled_backends=compiled_backends,
        )
        return python_code, _dynamo_cache_bytes(
            python_code,
            backend,
            [compiled.cache for compiled in compiled_backends],
        )
    except Unsupported as e:
        raise PrecompileError(
            "precompile tracer='dynamo' could not capture a resumable graph break. "
            f"Dynamo reported: {e}"
        ) from e
    except BackendCompilerFailed as e:
        if isinstance(e.inner_exception, PrecompileError):
            raise e.inner_exception from e
        raise
    except InternalTorchDynamoError as e:
        raise PrecompileError(
            f"precompile tracer='dynamo' failed during capture: {e}"
        ) from e
    except PackageError as e:
        raise PrecompileError(
            f"precompile tracer='dynamo' could not serialize the capture: {e}"
        ) from e
    except (FailOnRecompileLimitHit, RecompileError) as e:
        raise PrecompileError(
            "precompile tracer='dynamo' could not capture every example before "
            f"recompile_limit={recompile_limit}: {e}"
        ) from e
    finally:
        try:
            capture_stack.close()
        finally:
            try:
                if package is not None and region >= 0:
                    from torch._dynamo.eval_frame import _clear_cache_entries_for_region

                    for code in package.region_codes():
                        _clear_cache_entries_for_region(code, region)
                for name in list(target.__globals__):
                    if (
                        name.startswith(generated_prefixes)
                        and name not in existing_generated_globals
                    ):
                        target.__globals__.pop(name, None)
                pgo_state.clear()
            finally:
                _DYNAMO_COMPILE_LOCK.release()


class _PrecompileHandle:
    """Marker for loaded artifact handles that guard serialization must prune."""


class PrecompiledModule(_PrecompileHandle):
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
            raise AssertionError(
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

    def __call__(self, *args: object, **kwargs: object) -> object:
        # A PrecompiledModule is runnable only after load(); precompile() itself
        # returns (python_code, cache) rather than a runnable.
        if self._loaded_forward is None:
            raise PrecompileError(
                "this object is not runnable; build one with "
                "torch.compiler.precompile.load(python_code, cache)."
            )
        return self._loaded_forward(*args, **kwargs)

    def __enter__(self) -> Self:
        if self._loaded_forward is None:
            raise PrecompileError("precompile artifact has not been loaded")
        enter = getattr(self._loaded_forward, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._loaded_forward is not None:
            exit_fn = getattr(self._loaded_forward, "__exit__", None)
            if exit_fn is not None:
                exit_fn(*exc)

    def unload(self) -> None:
        if self._loaded_forward is not None:
            unload = getattr(self._loaded_forward, "unload", None)
            if unload is not None:
                unload()

    def serve_time_compiles(self) -> int:
        """Return the number of graphs compiled after loading (currently always zero)."""
        if self._loaded_forward is None:
            raise PrecompileError("precompile artifact has not been loaded")
        count = getattr(self._loaded_forward, "serve_time_compiles", None)
        return 0 if count is None else cast("int", count())

    @property
    def capture_summary(self) -> PrecompileSummary | None:
        if self._loaded_forward is None:
            return None
        return cast(
            "PrecompileSummary | None",
            getattr(self._loaded_forward, "capture_summary", None),
        )

    def to_python_code(self) -> str:
        """Return the executable Python artifact as a string.

        It needs no cache. Make-fx artifacts and standalone Dynamo artifacts run on
        their own; installed Dynamo artifacts also import their defining modules. For
        the inductor backend it embeds the composed graph module from
        aot_autograd.compile_to_python (kernels JIT-compile on first call;
        AOTAutograd's prelude/epilogue inlined), the calling-convention metadata, and a
        ``forward()`` that takes the same arguments the traced function took. For the
        eager backend it embeds the captured ATen graph and a driver that runs it
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
    """Fallback: execute the Python artifact string (JITs kernels).

    ``python_code`` needs no cache -- the kernels (inductor) or graph (eager) are
    inlined, so we just exec it and hand back its ``forward``. The returned
    ``forward`` takes the same args the traced fn took (model(s) plus runtime
    inputs)."""
    # python_code is untrusted EXECUTABLE input -- exec'ing it runs whatever it contains
    # (JIT-compiling inlined kernels or running the inlined graph). Warn per load (not
    # warning_once) before the exec so the inlined fallback is never silent about it.
    log.warning(
        "torch.compiler.precompile.load is about to EXEC python_code, which is untrusted "
        "executable input (it runs inlined kernels / graph code). Only exec python_code "
        "you produced or otherwise trust (Note [precompile programming model], "
        "invariant 7)."
    )
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
    ExampleInput = ExampleInput

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
        example_inputs: Sequence[tuple[object, ...] | ExampleInput] | None = None,
        backend: str = "inductor",
        tracer: str = "make_fx",
        decompositions: dict | None = None,
        training: bool = False,
        recompile_limit: int = 256,
        dynamic: bool | None = None,
        guard_filter_fn: Callable[[Sequence[Any]], Sequence[bool]] | None = None,
        invariants: str | None = None,
        require_complete: bool = True,
        require_no_risky_drops: bool = True,
        require_no_dropped_guards: bool = False,
    ) -> tuple[str, bytes]:
        """Ahead-of-time precompile ``fn`` against ``example_inputs``.

        .. note::

            ``torch.compiler.precompile`` is NOT
            ``torch._dynamo.config.caching_precompile`` (a ``torch.compile``
            guard-serialization caching mode); it captures ``fn`` ahead of time and
            lowers it to an executable Python source artifact.

        With the default ``make_fx`` tracer this is a non-strict trace with an explicit
        contract; read Note [precompile programming model] before using it. The artifact
        faithfully reproduces ``fn`` only for callers that uphold that contract.

        ``example_inputs`` is a sequence of positional-argument tuples or
        :class:`torch.compiler.ExampleInput` values for ``fn``. ``ExampleInput`` adds
        keyword arguments for the Dynamo tracer.
        For compatibility, positional arguments after ``fn`` describe one example call;
        they cannot be combined with ``example_inputs``.
        The outer sequence supports capture front-ends that can specialize one artifact
        from multiple calls. The ``make_fx`` tracer accepts exactly one tuple because it
        records only one execution; ``dynamo`` executes every tuple and records the
        guarded recompilations they trigger. The Dynamo artifact filters serialized guard
        records under its input-only recompilation contract while preserving how every
        example dispatches among those variants.

        THREADING: the inductor lowering step drives process-global compiler state
        and is serialized by an internal lock, so concurrent ``backend="inductor"``
        calls lower one at a time. Dynamo capture is also serialized because it uses
        process-global frame-evaluation and compilation state. The make_fx capture phase
        and its ``backend="eager"`` path are NOT serialized.

        ``backend`` selects how the captured graph is realized:

        - ``"inductor"`` (default): lower the graph through
          ``torch._functorch.aot_autograd.compile_to_python`` (the full AOTAutograd +
          Inductor pipeline, composed into one self-contained module). ``python_code``
          is the inlined Inductor output with AOTAutograd's prelude/epilogue; the cache
          holds the save_cache_artifacts bundle that primes the inductor cache on load.
        - ``"eager"``: do NOT lower -- keep the captured ATen graph and run it as-is
          (analogous to ``torch.compile(backend="eager")``). ``python_code`` inlines
          the readable captured graph. Higher-order graph bodies are readable Python;
          their FX structure is also embedded because eager HOP interpreters require a
          real ``Graph`` at runtime. Loading recompiles that structure to bytecode but
          never symbolically retraces it. The eager cache carries no compiled artifact
          (artifact=None) but is still a full integrity-tagged envelope -- with no
          kernels there is nothing to accelerate, so ``load`` runs the inlined graph.
          Useful for inspecting/debugging exactly what was traced without an Inductor
          dependency.

        ``tracer`` selects the capture front-end:

        - ``"make_fx"`` (default): a NON-STRICT make_fx trace -- it records the ATen ops
          that actually run when ``fn`` executes once on the sole example-input tuple
          and does not analyze your Python, so control flow and shapes are specialized
          to the example (the source of the programming-model contract).
        - ``"dynamo"``: analyze a Python function's bytecode and capture every guarded
          specialization/recompilation exercised by ``example_inputs``. The emitted
          artifact drops a serialized guard record only when doing so preserves every
          example's variant-match results, or when the guard only checks process-local
          state outside the explicit inputs. The Python environment -- including
          globals and context-manager state -- must be semantically identical between
          capture and runtime, and only explicit inputs may vary in a way that causes a
          recompile. Environment-only checks are therefore caller assumptions rather
          than dispatch predicates. By default, every portable input-derived guard is
          retained. Distinct tensor inputs must not share or overlap storage, and an
          explicit input must not also be reachable through the Python environment.
          Statically visible identity relations are rejected; dynamic native indirection
          that hides one is unsupported. Python functions that mutate globals or mutable
          objects reachable through the Python environment are rejected. Each guard is
          rebuilt independently from frozen capture state; an
          input-derived rebuild failure or changed predicate raises, while an
          environment-only failure is omitted with dependent attribute checks.
          Filtering is at guard-record granularity, so a retained composite record can
          still rebuild invariant leaf checks. Breaking an unchecked assumption can
          silently miscompute.
          A call that fails every retained guard set raises, including for an installed
          artifact. Graph breaks are preserved through their Dynamo resume frames;
          closure-free Python functions wrapped with ``torch._dynamo.disable`` are
          embedded and execute eagerly between graph segments. The top-level function
          must not have closure cells or nested functions that capture locals. Globals
          left in standalone transformed bytecode must be literal values or independently
          importable objects; installed frames may resolve them from their defining
          module. Disabled functions cannot assign globals or use
          ``globals()``, ``eval()``, or ``exec()``; their importable module globals are
          rebound at load, while recursive literal globals and defaults are captured by
          value. Top-level defaults must also be recursive literals; mutable or
          user-defined values must be passed explicitly rather than used as defaults.
          Tensor-valued defaults and tensor-valued globals referenced by user-defined
          code are rejected because user-owned tensors must be explicit inputs. Bound
          methods are unsupported; pass the unbound function and its receiver as an
          explicit input.
          ``nn.Module`` arguments are accepted and checked against the captured module
          type, training mode, parameter/buffer structure, aliasing, and tensor metadata.
          Frames reachable only through ordinary Python calls use an isolated installed
          artifact. Loading prepares its backends and guard trees without installing
          them; the first call installs. An uncovered call raises instead of compiling.
          ``unload()`` (or context-manager exit) removes the artifact's state.

        ``decompositions`` is an optional decomposition table (a dict mapping each
        ``OpOverload`` to a decomposition function) forwarded to ``make_fx`` as its
        ``decomposition_table`` during capture, so you can control how ATen ops are
        broken down in the captured graph. Defaults to ``None`` (make_fx's default) and
        is not yet supported with ``tracer="dynamo"``.

        ``recompile_limit`` bounds the number of variants captured per code object.
        ``dynamic`` has the same meaning as for :func:`torch.compile`; ``None`` enables
        automatic dynamic-shape promotion as dimensions vary across examples.

        ``guard_filter_fn`` receives each candidate Dynamo guard sequence and returns
        one bool per guard. It can narrow the portable default set but cannot restore a
        guard Dynamo cannot serialize. Input guards removed by this callback are risky
        drops and are rejected unless ``require_no_risky_drops=False``. After all examples
        run, precompile drops guards covered by the invariant-environment contract and
        verifies that retained frozen facts still distinguish the captured variants.
        ``invariants`` optionally names a text file receiving the resulting invariant,
        varying, and undetermined guard report.

        ``require_complete`` rejects captures with bypassed, truncated, uncovered, or
        failed frames. ``require_no_risky_drops`` rejects dropped guards that could alter
        dispatch and defaults to true. ``require_no_dropped_guards`` rejects every drop;
        it defaults to false because ordinary captures contain unserializable identity
        guards. The loaded callable exposes the final :class:`PrecompileSummary` as
        ``capture_summary``.

        ``training=True`` is supported with ``tracer="dynamo"`` on both backends.
        Capture runs with grad enabled and served outputs retain their ``grad_fn``.
        Serving pins grad mode to this option rather than inheriting the caller's mode.
        Inductor graphs carry readable AOTAutograd forward and backward source bridged
        by an emitted ``torch.autograd.Function``; eager graphs replay their captured
        differentiable operations. The input tensors that require gradients must do so
        in every example and at runtime.

        With ``tracer="dynamo"``, shape variation across ``example_inputs`` uses
        Dynamo's ordinary automatic dynamic-shape policy: for example, a static first
        graph can recompile into a symbolic graph when a later tuple changes a dimension.
        The symbolic graphs and contract-filtered dispatch guard records are retained in
        the artifact.

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
        so a mismatch is rejected. Any new deferred runtime shape constraint that the
        standalone driver cannot enforce, whether between independently marked input
        dims or on a derived data-dependent size, raises ``PrecompileError`` instead of
        baking the example relation.

        Returns ``(python_code, cache)`` -- an executable Python source string (the
        single source of truth for the calling convention) and a
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
        parameter/buffer structure (invariant 2). The ``make_fx`` tracer accepts only
        positional arguments. The Dynamo tracer also accepts
        ``torch.compiler.ExampleInput(args=..., kwargs=...)`` and reproduces keyword
        calls at runtime. With ``tracer="make_fx"``, if ``fn`` ran a
        backward, the resulting parameter gradients are scattered (accumulated) onto
        that runtime model's ``parameters()`` ``.grad`` fields, exactly like eager ``.backward()``,
        so a ``zero_grad()`` / ``optimizer.step()`` loop works unchanged; the artifact
        returns ``fn``'s own result (``None`` for a bare ``.backward()`` step), not the
        grads (invariant 5).

        With ``tracer="dynamo"``, when ``fn`` itself is a user-defined Python
        ``nn.Module`` whose ``forward`` method Dynamo can capture as a Python frame, the
        reloaded callable takes the runtime module as its first argument, followed by the
        arguments from each example. Built-in modules such as ``Linear`` and
        ``Sequential`` should instead be passed through a wrapper such as
        ``lambda model, x: model(x)``.

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
        if isinstance(example_inputs, Sequence) and len(example_inputs) == 0:
            raise ValueError(
                "precompile requires example_inputs=[(...), ...]: one tuple or "
                "torch.compiler.ExampleInput per call to capture."
            )
        if isinstance(
            example_inputs, (torch.Tensor, torch.nn.Module, str)
        ) or not isinstance(example_inputs, Sequence):
            raise TypeError(
                "precompile example_inputs takes a sequence of calls, not "
                f"{type(example_inputs).__name__}. Wrap one call as "
                "example_inputs=[(arg0, arg1)]."
            )
        if backend not in ("inductor", "eager"):
            raise ValueError(
                f"precompile backend must be 'inductor' or 'eager', got {backend!r}."
            )
        if tracer not in ("make_fx", "dynamo"):
            raise ValueError(
                f"precompile tracer must be 'make_fx' or 'dynamo', got {tracer!r}."
            )
        if training and tracer != "dynamo":
            raise NotImplementedError(
                "precompile training=True currently requires tracer='dynamo'."
            )
        if recompile_limit <= 0:
            raise ValueError("precompile recompile_limit must be positive")
        if tracer == "make_fx" and (
            recompile_limit != 256
            or dynamic is not None
            or guard_filter_fn is not None
            or invariants is not None
            or not require_complete
            or not require_no_risky_drops
            or require_no_dropped_guards
        ):
            raise ValueError(
                "precompile guard_filter_fn, recompile_limit, dynamic, invariants, "
                "and require_* options apply only to tracer='dynamo'."
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
                guard_filter_fn=guard_filter_fn,
                invariants=invariants,
                require_complete=require_complete,
                require_no_risky_drops=require_no_risky_drops,
                require_no_dropped_guards=require_no_dropped_guards,
            )
        make_fx_examples = []
        for example in example_inputs:
            if isinstance(example, ExampleInput):
                if example.kwargs:
                    raise NotImplementedError(
                        "precompile tracer='make_fx' does not support keyword example "
                        "inputs; use tracer='dynamo'."
                    )
                make_fx_examples.append(example.args)
            else:
                make_fx_examples.append(example)
        compiled = PrecompiledModule(
            fn, backend=backend, tracer=tracer, decompositions=decompositions
        )
        compiled._compile(make_fx_examples)
        # Build the (expensive) python_code ONCE and thread it into to_cache_bytes so
        # the full metadata + embedded kernel source is not rebuilt, and so code_hash is
        # sha256 over exactly the bytes returned to the caller (a matched pair loads).
        python_code = compiled.to_python_code()
        return python_code, compiled.to_cache_bytes(python_code)

    def load(
        self,
        python_code: str,
        cache: bytes,
        *,
        fn: Callable[..., object] | None = None,
    ) -> PrecompiledCallable:
        """Reconstruct a runnable from ``(python_code, cache)`` from precompile.

        The driver runs from ``python_code`` -- the single source of truth for the whole
        calling convention. ``load`` reads the cache's ``BACKEND`` (to check the pairing)
        and, for the inductor backend, primes the inductor kernel caches from its
        ``save_cache_artifacts`` bundle (via ``torch.compiler.load_cache_artifacts``) so a
        warm reload loads precompiled kernels instead of JIT-compiling; then it exec's
        ``python_code``. With no usable cache it degrades to JIT'ing from ``python_code``.

        Call the result with the SAME argument structure ``fn`` took -- the model(s) in
        their original positions plus the runtime inputs. The exception is a directly
        captured user-defined ``nn.Module`` with the Dynamo tracer: pass that runtime
        module before the arguments from the example. Per invariant 2 of Note
        [precompile programming model], the runtime model must match the example model's
        parameter/buffer structure; precompile re-derives the param/buffer list from it
        (same interning/order as capture).

        A Dynamo artifact containing nested frames that a source dispatcher cannot
        reach prepares its package while loading, then installs it lazily on first use.
        Pass ``fn=`` to bind that package to a live callable instead of reconstructing
        the entry function. Its defining modules must still be importable. Such an
        artifact supports ``unload()`` and the context-manager protocol; both remove
        only that artifact's isolated entries. An uncovered call raises rather than
        compiling a new variant, so ``serve_time_compiles()`` remains zero.

        Raises ``PrecompileError`` if ``python_code`` is malformed or is not a
        ``torch.compiler.precompile`` artifact (it fails to parse, or is missing the
        calling-convention metadata), if the cache's ``backend`` tag does not match
        ``python_code``, or if the cache's ``code_hash`` does not match
        ``sha256(python_code)`` -- i.e. the cache and python_code came from different
        ``precompile()`` calls. A cache whose ``format``/``version`` does not match (a
        foreign or different-build envelope) is NOT fatal: the cache is acceleration
        only, so ``load`` degrades to JIT'ing from ``python_code`` rather than crashing.
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
        if tracer == "dynamo":
            python_version = cast(tuple[int, int], meta["_DYNAMO_PYTHON_VERSION"])
            if python_version != sys.version_info[:2]:
                raise PrecompileError(
                    "precompile artifact was produced on Python "
                    f"{python_version[0]}.{python_version[1]}, but is being loaded on "
                    f"Python {sys.version_info[0]}.{sys.version_info[1]}."
                )
            torch_version = cast(str, meta["_DYNAMO_TORCH_VERSION"])
            if torch_version != torch.__version__:
                raise PrecompileError(
                    f"precompile artifact was produced by torch {torch_version}, but "
                    f"is being loaded by torch {torch.__version__}."
                )

        # weights_only=True is safe (plain str/int/bytes dict). The inner artifact bytes
        # are the inductor save_cache_artifacts bundle, used below to prime the kernel
        # caches. The cache is acceleration only, so an unreadable envelope or a FORMAT /
        # VERSION mismatch degrades to JIT'ing from python_code rather than crashing. A
        # BACKEND or CODE_HASH mismatch is different -- it signals a wrong (python_code,
        # cache) pairing -- so it hard-fails rather than running under foreign metadata.
        artifact = None
        try:
            blob = torch.load(io.BytesIO(cache), weights_only=True)
            if blob.get("format") != _CACHE_FORMAT or blob.get("version") != (
                _CACHE_VERSION
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
                if blob.get("backend") != backend:
                    raise PrecompileError(
                        f"cache backend {blob.get('backend')!r} does not match the "
                        f"python_code backend {backend!r}; the cache and python_code "
                        "came from different precompile() calls."
                    )
                # Reject a cache whose code_hash does not match this python_code (a
                # mismatched pairing); see Note [precompile programming model], invariant 7.
                expected_code_hash = hashlib.sha256(python_code.encode()).hexdigest()
                if blob.get("code_hash") != expected_code_hash:
                    raise PrecompileError(
                        "cache does not match python_code (its code_hash "
                        f"{blob.get('code_hash')!r} != sha256(python_code) "
                        f"{expected_code_hash!r}); the cache and python_code came from "
                        "different precompile() calls. Pair each cache with the "
                        "python_code from the same precompile() call."
                    )
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
        if fn is not None:
            rebind = getattr(forward, "_rebind", None)
            if rebind is not None:
                rebind(fn)
        prepare = getattr(forward, "_prepare", None)
        if prepare is not None:
            prepare()

        return PrecompiledModule._from_loaded(forward, backend=backend)


precompile = _PrecompileApi()
# ``torch.compiler.precompile`` is a callable instance, not a function, so give it the
# name/doc introspection (Sphinx autosummary, help(), IDEs) expects to find on a
# public callable; the rich usage docs live on ``__call__``.
precompile.__name__ = "precompile"  # type: ignore[attr-defined]
precompile.__qualname__ = "precompile"  # type: ignore[attr-defined]
precompile.__doc__ = _PrecompileApi.__call__.__doc__

# Both are public under torch.compiler.precompile, so report their module/qualname there
# (mirroring the singleton fixup above) -- otherwise Sphinx autoexception/autofunction
# would anchor them under this private module. load is a bound method; patch the
# underlying function so introspection on precompile.load reports torch.compiler too.
PrecompileError.__module__ = "torch.compiler"
PrecompileError.__qualname__ = "precompile.PrecompileError"
_PrecompileApi.load.__module__ = "torch.compiler"
_PrecompileApi.load.__qualname__ = "precompile.load"
