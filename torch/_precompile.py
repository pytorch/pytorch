"""Ahead-of-time precompilation. Capture is caller-driven: the caller invokes a
capture around their own execution rather than handing precompile example inputs to
run. Both ``precompile.capture(...)`` and ``precompile.accumulate(...)`` return a
capture that writes a ``(python_code, cache)`` artifact to disk; ``load`` reloads it
from those two files. ``capture`` writes the artifact once, when the ``with`` block
exits; ``accumulate`` rewrites it after every call, so a job that dies partway leaves
a working artifact for the batches it reached.

    with torch.compiler.precompile.capture(
        fn, artifact_path="model.py", cache_path="model.cache"
    ) as cap:
        out = cap(model, x)             # runs fn(model, x), returns its result
    f_c = torch.compiler.precompile.load("model.py", "model.cache")
    out = f_c(model, x)                 # pass the model again at runtime

    # Several guarded/recompiled variants, rewriting the artifact each call.
    with torch.compiler.precompile.accumulate(
        fn, artifact_path="model.py", cache_path="model.cache"
    ) as cap:
        for x in loader:
            out = cap(model, x)

The calls the caller makes ARE the capture: inputs flow through naturally and each
call returns what ``fn`` returned, so the capture drops into an ordinary
training/pipeline loop where intermediate values are needed. ``tracer`` picks the
capture front-end and carries its tracer-specific configuration -- ``DynamoTracer()``
(the default) takes as many calls as you make, ``MakeFxTracer()`` takes exactly one;
``accumulate`` is always dynamo. ``backend`` and ``training`` are shared across both
tracers.

``DynamoTracer`` (the default) analyzes the Python (bytecode) rather than tracing one
path. It inlines the TRANSFORMED BYTECODE Dynamo produces into ``python_code``
(marshalled, rehydrated at load) and lowers the compiled subgraphs through the chosen
backend; forward and training computations and ``mark_unbacked`` dynamic shapes work
with it. It records graph breaks and every guarded recompilation the calls exercise,
so make as many calls as you need to cover them.

``MakeFxTracer`` captures your computation with ``make_fx`` -- a NON-STRICT trace of
the ATen ops that run when ``fn`` executes once. It does not analyze your Python, so it
comes with an explicit contract (the programming model): stay inside it and the artifact
faithfully reproduces ``fn``; step outside it and you get an artifact that computes the
wrong thing. It produces one trace, so it captures exactly one call and refuses a second.
``MakeFxTracer.decompositions`` forwards a decomposition table (Dynamo lowers through the
backend instead, so it has no such knob). See the ``tracer`` note at the bottom of
Note [precompile programming model].

Multi-graph capture keeps all live guards while the calls run, then filters only the
serialized copy. The ``DynamoTracer`` ``require_*`` gates refuse known coverage gaps,
failed captures, and RISKY dropped guards by default; the stricter
``require_no_dropped_guards`` is off, since every model drops identity guards that cannot
be serialized. Coverage remains execution-driven: a complete summary describes the calls
that ran, not every possible input or unexecuted branch. The caller makes the calls, so
gradients and return values keep their normal eager/``torch.compile`` semantics: the
calls run in whatever grad mode the caller sets, and ``training=True`` lowers the
backward eagerly. Serve the resulting artifact under the mode it was captured in -- grad
mode is a ``GLOBAL_STATE`` guard, and it is checked.

The artifact is a self-contained, executable ``python_code`` string plus a
companion integrity-tagged ``cache``. With ``backend="inductor"`` (the default) the
captured graph is lowered through the AOT backend contract
(``torch._functorch.aot_autograd.compile_to_python``, AOTAutograd + Inductor);
``python_code`` JIT-compiles kernels on first call and the cache primes them so a warm
reload skips JIT. With ``backend="eager"`` ``python_code`` inlines the captured graph and
runs on its own. Reload with ``torch.compiler.precompile.load(artifact_path, cache_path)``.

The full contract, the calling convention, and the cache / code_hash design all live in
Note [precompile programming model] below; every public entry point and guard references
it.
"""

# Note [precompile programming model]
#
# ``fn`` is the WHOLE computation, e.g. ``lambda model, x: model(x)`` for inference
# or ``lambda model, x, t: loss_fn(model(x), t).backward()`` for a training step
# (captured with ``training=True`` so the backward is lowered eagerly; the calls run
# in whatever grad mode the caller sets). Among the positional args, the nn.Module arguments have their parameters and
# buffers lifted to explicit graph inputs (via functional reparametrization), so
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
#    over a static (Python ``int``) value and shape-dependent branching on a static size
#    are resolved at trace time and baked. Shapes are static BY DEFAULT (capture runs
#    make_fx in its "fake" mode, so each size is baked as a concrete constant). What is
#    NOT silently baked is a data-dependent op -- ``.item()``, ``.nonzero()``, a Python
#    ``if``/``for`` over a TENSOR VALUE: under fake tracing the value is unknown, so such
#    an op RAISES at capture (a GuardOnDataDependentSymNode / unbacked error surfaced as
#    PrecompileError) rather than freezing the example run's value into the artifact as a
#    real trace would. You can opt specific user-input
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
#    zero grads as usual. The dynamo tracer reaches the SAME observable behavior by a
#    different route (see the tracer note): a ``.backward()`` in ``fn`` graph-breaks
#    (Dynamo does not trace it while ``trace_autograd_ops`` is off, the default), so at
#    serve time the live autograd engine runs the compiled backward and does the
#    accumulate itself; there is no harvested-output list -- but which params get a
#    grad is still fixed at trace time, frozen params still keep ``.grad = None``, and
#    the accumulate still matches eager.
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
#    load() rejects a (code, cache) pair from different precompile captures (same
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
# (python_code, cache) pair that did not come from the same precompile capture.
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
# non-strict trace -- everything above (the invariants, the contract) describes its
# behavior. "dynamo" is a Dynamo-based front-end that analyzes the Python (bytecode)
# instead of specializing to one traced path.
#
# The "dynamo" tracer's TRICK, and how it differs from make_fx: Dynamo does not hand back
# a single graph we can render as source. It hands back (a) a TRANSFORMED bytecode -- a
# rewrite of fn that extracts the runtime model's params/buffers, calls a compiled
# subgraph, and reassembles fn's output -- plus (b) the subgraph (an fx GraphModule) for
# the backend to lower. So precompile INLINES the transformed bytecode into python_code
# (marshalled to a base64 blob, rehydrated by the driver via marshal.loads +
# types.FunctionType) and lowers the subgraph through the SAME backends as make_fx
# ("inductor" -> aot_autograd.compile_to_python source, "eager" -> the inlined subgraph),
# wiring the subgraph in under the backend id the bytecode calls. The transformed bytecode
# IS the calling convention: it reads params off the runtime model itself (given a
# structurally identical runtime model it reads the right weights, invariant 2), which is
# why the dynamo driver is thin (rehydrate + wire) and carries none of the make_fx
# PARAM_NAMES / OUT_SPEC metadata.
#
# TRAINING works here too, by a different mechanism than make_fx. make_fx traces THROUGH
# .backward(), so its artifact is one flat graph of fwd+bwd ATen ops with the grads as
# extra outputs and a Python-level scatter in the driver (invariant 5). The dynamo tracer
# handles a training step the way torch.compile does: the forward subgraph lowers to a
# differentiable autograd.Function whose compiled backward AOTAutograd would normally
# produce lazily on the first .backward() call -- capture forces that lowering eagerly
# (force_non_lazy_backward_lowering), so the artifact carries the compiled backward and
# serving never compiles. A .backward() inside the captured fn graph-breaks like any
# other side effect and re-runs at serve time through the live autograd engine, which is
# also what accumulates .grad on the runtime model's params: there is no in-graph
# autograd.grad rewrite and no grad-scatter metadata.
#
# Dynamic shapes work here too, by a different mechanism than make_fx: mark_unbacked is
# Dynamo's OWN decorator, so Dynamo captures the marked dim as an UNBACKED symint
# directly -- unguardable, so a graph that needs to guard on it fails loudly at capture
# (the same PrecompileError the make_fx tracer raises) instead of baking a size. Dynamo
# emits the ShapeEnv's runtime asserts (mark_unbacked's min/max, a shared shape_id's
# equality) into the subgraph itself, so they hold on BOTH backends -- unlike the make_fx
# tracer, whose eager backend has no such asserts and therefore rejects dynamic dims
# outright. The STRICT variant is NOT rejected here: Dynamo reads it as a
# RelaxedUnspecConstraint -- a BACKED dynamic dim that errors at capture only if the
# trace specializes it to a constant -- and any guards taken on it ride in the artifact's
# serialized guard state like every other guard. Decompositions do NOT work here: Dynamo
# captures torch-level IR and never consults a decomposition table, so passing
# ``decompositions`` with DynamoTracer is rejected up front (ValueError) rather than
# silently ignored.
#
# Scope and differences from make_fx: the capture is execution-driven and multi-frame --
# it preserves every graph-break continuation, guard, and recompiled variant of the
# example calls, one transformed bytecode per captured frame.
# This path does not
# reproduce the make_fx drivers' upfront runtime validation (the param/buffer structural
# check, invariant 2, and the per-input shape/dtype/device checks, invariants 3/6): safety
# comes from the SERIALIZED GUARDS the driver rebuilds and evaluates per variant (minus
# the unserializable ones that were dropped -- see the require_* gates), from the same
# specialization contract as make_fx (control flow and unmarked shapes are specialized to
# the example), and from the captured graph's own asserts -- on the INDUCTOR backend the
# baked assert_size_stride (which catches a runtime input/weight whose SHAPE or STRIDE
# differs from the example, but not its DTYPE) and, for a dynamic capture, the ShapeEnv
# range / equality asserts on both backends. A call no surviving guard set covers is a
# loud miss rather than a silent wrong answer, but a contract violation the DROPPED
# guards would have caught can still reach a raw kernel error, and on the EAGER backend
# (no assert_size_stride) a broadcast-compatible shape mismatch can silently miscompute
# -- pass inputs and a model matching the example, as the contract requires. Because Dynamo bakes the trace-time environment (e.g. the current accelerator
# stream) into the bytecode, the artifact is environment-specialized like the make_fx one.
# This artifact renders its compiled subgraphs as source like the make_fx tracer does,
# but it ALSO inlines MARSHALLED CPython bytecode plus a PICKLED guard-state blob (which
# have no source form), so it is LOCKED to the producing Python version:
# loading it under a different CPython (3.10-3.14) fails with a clean PrecompileError
# (see the driver's version gate). It is ALSO locked to a compatible torch build, because its import
# aliases can reference private torch._dynamo runtime modules (also surfaced as a clean
# PrecompileError). Regenerate per Python version / torch build, or use make_fx for portable
# source (backend='eager' for torch-build portability -- the default make_fx inductor artifact
# itself inlines private torch._inductor modules, so it too is torch-build-locked; the
# Python-version portability holds for either make_fx backend).

from __future__ import annotations

import base64
import dataclasses
import functools
import hashlib
import inspect
import io
import logging
import os
import pickle
import sys
import threading
import types
import uuid
from collections.abc import Callable, Mapping, Sequence  # noqa: TC003
from types import MappingProxyType
from typing import Any, cast, NewType, TYPE_CHECKING
from typing_extensions import Self

import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


log = logging.getLogger(__name__)

# Installing records backend keys into the process-global PrecompileContext
# and unload takes them back. Serialize the snapshot/record and the take-back
# so two handles on one artifact cannot each record, or take, the other's keys.
_RECORD_LOCK = threading.Lock()


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from torch._functorch._aot_autograd.codegen import PySourceBuilder
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.compiler._precompile_types import FrameInvariants, PrecompileSummary


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

    ``result`` carries what the call that raised returned, when it ran before the
    refusal -- for ``precompile(...)`` the list of example results; ``None`` otherwise.
    """

    # Re-exported in torch.compiler.__all__, so pickle and test_public_bindings
    # resolve it there.
    __module__ = "torch.compiler"

    #: What the call that raised this returned, when it ran before the refusal:
    #: an accumulating capture's step has already executed by the time its
    #: artifact gate refuses, so the result rides on the error. ``None`` otherwise.
    result: object = None


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


_MULTIGRAPH_GENERATED_HEADER = """\
# Generated by torch.compiler.precompile (multi-graph) -- do not edit.
#
# Self-contained, executable artifact for a computation with GRAPH BREAKS or several
# guarded variants. Where the single-graph forms inline one graph, this inlines every
# frame Dynamo compiled -- the entry frame plus each continuation -- with one guard tree
# per captured variant, and dispatches among them at call time:
#
#     ns = {}
#     exec(open("this_file.py").read(), ns)
#     out = ns["forward"](model, my_input)      # same args as the captured callable
#
# Sections below are labelled. What is OPAQUE is base64 of pickled Dynamo state --
# the guard trees and the transformed bytecode -- because those have no readable
# source form. The compiled subgraphs DO have one and are emitted as source; only a
# subgraph the backend could not render (an eager fx graph, a training graph) falls
# back to the blob. Everything else is meant to be read and reviewed.
"""


# What the artifact does to the process, which differs by serving mode and was
# previously described only for the standalone one -- in a banner emitted on both.
_SERVING_NOTES = {
    "standalone": """\
# Nothing is installed onto your code objects and no frame evaluator is involved, so
# loading this mutates no global state. The flip side is that there is no compiler
# behind it: a call no captured variant covers RAISES rather than compiling a new one.
""",
    "installed": """\
# This artifact SERVES BY INSTALLING onto the live code objects, so loading and then
# entering it mutates global state, which unload() and __exit__ take back out. And
# there IS a compiler behind it: a call no captured variant covers is compiled fresh
# at serve time rather than refused. That is what makes a graph-breaking model
# servable at all, but it means the artifact can quietly serve less and less of
# itself -- watch for the "serving compiled a NEW graph" warning, or read
# serve_time_compiles() on the loaded object.
""",
}


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


def _make_inlined_forward(
    python_code: str, *, warn: bool = True
) -> Callable[..., object]:
    """Fallback: execute the self-contained python string (JITs kernels).

    ``python_code`` needs no cache -- the kernels (inductor) or graph (eager) are
    inlined, so we just exec it and hand back its ``forward``. The returned
    ``forward`` takes the same args the traced fn took (model(s) plus runtime
    inputs). ``warn`` is off only for the capture-time self-load, where the source
    was just produced in-process and so is not untrusted input."""
    # python_code is untrusted EXECUTABLE input -- exec'ing it runs whatever it contains
    # (JIT-compiling inlined kernels or running the inlined graph). Warn per load (not
    # warning_once) before the exec so the inlined fallback is never silent about it.
    if warn:
        log.warning(
            "torch.compiler.precompile.load is about to EXEC python_code, which is "
            "untrusted executable input (it runs inlined kernels / graph code). Only "
            "exec python_code you produced or otherwise trust (Note [precompile "
            "programming model], invariant 7)."
        )
    module_ns: dict[str, object] = {"__name__": "_precompiled_artifact"}
    exec(compile(python_code, "<precompile>", "exec"), module_ns)
    return cast("Callable[..., object]", module_ns["forward"])
