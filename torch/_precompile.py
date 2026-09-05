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


@dataclasses.dataclass(frozen=True)
class MakeFxTracer:
    """The ``make_fx`` capture front-end, passed as ``tracer=`` to
    :func:`torch.compiler.precompile.capture`.

    A NON-STRICT single make_fx trace: it records the ATen ops of ONE execution of
    ``fn``, so a ``capture`` with this tracer takes exactly one call and refuses a
    second, and control flow and shapes are specialized to that call (the source of
    the programming-model contract). Part of the prototype ``torch.compiler.precompile``
    API, so it may change without a deprecation cycle.

    ``decompositions`` is an optional decomposition table (a dict mapping each
    ``OpOverload`` to a decomposition function) forwarded to ``make_fx`` as its
    ``decomposition_table``; it is specific to this tracer (Dynamo lowers through the
    backend instead).
    """

    decompositions: dict | None = None


@dataclasses.dataclass(frozen=True)
class DynamoTracer:
    """The ``dynamo`` capture front-end (the default), passed as ``tracer=`` to
    :func:`torch.compiler.precompile.capture` or :func:`torch.compiler.precompile.accumulate`.

    An execution-driven multi-graph capture that analyzes the Python (bytecode) rather
    than tracing one path: it records graph-break continuations and every guarded
    recompilation the calls exercise, so a capture with this tracer takes as many calls
    as you make. Part of the prototype ``torch.compiler.precompile`` API, so it may change
    without a deprecation cycle.

    The fields configure the multi-variant capture: ``guard_filter_fn`` filters the guards
    kept in the SERIALIZED artifact (runtime capture guards are always retained);
    ``recompile_limit`` caps recompilations; ``dynamic`` forces dynamic shapes;
    ``invariants`` selects the guard-drop policy; and the ``require_*`` gates refuse known
    coverage gaps and risky dropped guards (``require_no_dropped_guards`` is off by default,
    since every model drops identity guards that cannot be serialized).
    """

    guard_filter_fn: Callable[[Sequence[Any]], Sequence[bool]] | None = None
    recompile_limit: int = 256
    dynamic: bool | None = None
    invariants: str | None = None
    require_complete: bool = True
    require_no_risky_drops: bool = True
    require_no_dropped_guards: bool = False


class PrecompiledRunnable:
    """What :func:`torch.compiler.precompile.load` returns.

    A callable with the captured ``fn``'s calling convention that can also be
    entered as a context manager and unloaded. A standalone artifact installs
    nothing, so for it ``__enter__``/``__exit__``/:meth:`unload` are no-ops;
    :class:`PrecompiledCallable` is the shape that installs. ``installed`` tells
    them apart. Part of the prototype ``torch.compiler.precompile`` API, so it
    may change without a deprecation cycle.
    """

    __module__ = "torch.compiler"

    installed: bool = False
    """Whether calling this handle installs onto the captured code objects."""

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.unload()

    def unload(self) -> None:
        """Remove whatever this loaded artifact installed; a no-op when it installed nothing."""


class PrecompiledCallable(PrecompiledRunnable):
    """Callable handle for one loaded multi-graph precompile artifact.

    Returned by :func:`torch.compiler.precompile.load` for an artifact that
    serves by installing; it is not constructed directly. Part of the prototype
    ``torch.compiler.precompile`` API, so it may change without a deprecation
    cycle.
    """

    __module__ = "torch.compiler"

    def __init__(self, compiled: Any) -> None:
        self._compiled = compiled

    def _call(self, method: Callable[..., Any], *args: object, **kwargs: object) -> Any:
        from torch._dynamo.exc import PackageError, RecompileError

        try:
            return method(*args, **kwargs)
        except (PackageError, RecompileError) as e:
            raise PrecompileError(str(e)) from e

    installed: bool = True
    """Serves by installing onto the captured code objects; see :meth:`unload`."""

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self._call(self._compiled, *args, **kwargs)

    def __enter__(self) -> Self:
        self._call(self._compiled.__enter__)
        return self

    def unload(self) -> None:
        """Remove everything this loaded artifact installed.

        The precompiled entries come off the code objects and the globals the
        artifact wrote come out of their modules, so the model recompiles
        normally afterwards. Exiting this object as a context manager does the
        same thing; call it directly when the artifact's lifetime is not
        lexically scoped. Unloading twice is harmless, and a call already in
        flight on another thread is allowed to finish first.
        """
        self._call(self._compiled.unload)

    def serve_time_compiles(self) -> int:
        """Graphs this artifact compiled while SERVING, rather than serving.

        An installed artifact answers a guard miss by compiling, so a climbing
        count means it is covering less of the workload than the capture
        measured. Zero is the number to gate a job on.
        """
        return self._compiled.serve_time_compiles()


class _InstalledArtifact:
    """Handle for a multi-graph artifact that serves by installing.

    ``load`` builds this without touching any code object; entering it or the
    first call installs the captured frames, and ``unload``/exit takes them back
    out. After an unload a bare call raises rather than silently re-installing;
    re-entering as a context manager is the explicit way to install again.
    """

    def __init__(
        self,
        serve: Callable[..., Any],
        entry_factory: Callable[[], Callable[..., object]],
        *,
        check_fn: Callable[[Callable[..., object]], None] | None = None,
        backend_keys: Sequence[str] = (),
    ) -> None:
        self._serve = serve
        self._entry_factory = entry_factory
        # Refuses a load(fn=...) target the artifact was not captured from.
        self._check_fn = check_fn
        # PrecompileContext keys serve() records; unload() takes back the ones
        # this install added, by identity (see _ensure).
        self._backend_keys = backend_keys
        self._recorded: dict[str, Any] = {}
        self._fn: Callable[..., object] | None = None
        self._inner: Any = None
        self._prepared: Any = None
        # Serve-time compiles survive unload/exit: a job reads this after the
        # scope, when the live inner is already gone.
        self._serve_time_compiles = 0
        # unload() retires the handle for good. Without this flag a call after
        # unload would silently re-run _serve() -- re-mutating process-global
        # code objects with no paired unload, exactly the attributable-install
        # contract this class exists to keep -- and a call racing unload()
        # could reinstall before unload() returned.
        self._unloaded = False
        # Installing mutates process-global code objects, so racing first calls
        # must not both install.
        self._install_lock = threading.Lock()

    def _rebind(self, fn: Callable[..., object]) -> None:
        from torch._dynamo.exc import PackageError

        with self._install_lock:
            if self._inner is not None:
                raise PrecompileError(
                    "precompile: this artifact is already installed; pass fn= to load() "
                    "before the first call."
                )
            if self._check_fn is not None:
                try:
                    self._check_fn(fn)
                except (PackageError, TypeError) as e:
                    raise PrecompileError(str(e)) from e
            self._fn = fn

    def _prepare(self, package_blob: str) -> None:
        """Build what serving needs, at load rather than at the first call."""
        import base64

        from torch._dynamo.precompile_package import prepare_cache_entry

        # Exactly the resolution _ensure performs, or this prepares a different
        # entry frame than the install will use.
        fn = self._entry_factory() if self._fn is None else self._fn
        try:
            self._prepared = prepare_cache_entry(
                fn, pickle.loads(base64.b64decode(package_blob))
            )
        except PrecompileError:
            raise
        except Exception as e:
            raise PrecompileError(str(e)) from e

    def _ensure(self) -> Any:
        inner = self._inner
        if inner is None:
            with self._install_lock:
                if self._unloaded:
                    raise PrecompileError(
                        "precompile: this artifact has been unloaded; enter it "
                        "as a context manager to install it again, or load() a "
                        "new handle."
                    )
                if self._inner is None:
                    from torch._dynamo.precompile_context import PrecompileContext

                    fn = self._entry_factory() if self._fn is None else self._fn
                    # Backend keys are per-capture uuids, so only another handle
                    # on this very artifact (or the same entry loaded through
                    # DynamoStore) can already hold them, with identical content.
                    # _serve records only absent keys; remember which those were
                    # so unload takes back exactly what this install added.
                    # Consume the prepared entry first, so a failed serve
                    # does not reuse it on a later call.
                    prepared, self._prepared = self._prepared, None
                    with _RECORD_LOCK:
                        present = {
                            k
                            for k in self._backend_keys
                            if PrecompileContext.serialize_artifact_by_key(k)
                            is not None
                        }
                        self._inner = self._serve(fn, prepared=prepared)
                        self._recorded = {
                            k: PrecompileContext.serialize_artifact_by_key(k)
                            for k in self._backend_keys
                            if k not in present
                        }
                inner = self._inner
        return inner

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self._ensure()(*args, **kwargs)

    def __enter__(self) -> Self:
        # Entering IS a point in the caller's control flow, so it explicitly
        # re-arms a handle a prior unload() retired for bare calls.
        with self._install_lock:
            self._unloaded = False
        self._ensure()
        return self

    def __exit__(self, *exc: object) -> None:
        self.unload()

    def serve_time_compiles(self) -> int:
        inner = self._inner
        live = inner.serve_time_compiles() if inner is not None else 0
        return self._serve_time_compiles + live

    def unload(self) -> None:
        with self._install_lock:
            self._unloaded = True
            inner, self._inner = self._inner, None
            self._prepared = None
        if inner is not None:
            # Fold the retired install's serve-time count into the running
            # total so serve_time_compiles() survives unload/exit.
            self._serve_time_compiles += inner.serve_time_compiles()
            inner.unload()
            from torch._dynamo.precompile_context import PrecompileContext

            with _RECORD_LOCK:
                recorded, self._recorded = self._recorded, {}
                for key, artifact in recorded.items():
                    # Only the object this install filed: a same-key artifact
                    # filed since (an ambient run on the same graph) is not ours
                    # to take.
                    if PrecompileContext.serialize_artifact_by_key(key) is artifact:
                        PrecompileContext.take_artifact(key)


class Capture:
    r"""The caller-driven capture :func:`torch.compiler.precompile.capture` returns.

    Part of the prototype ``torch.compiler.precompile`` API, so it may change
    without a deprecation cycle. Enter it as a context manager to arm the
    capture, call it exactly as you would ``fn`` inside the block -- each call
    runs for real, folds what it exercised into the capture, and returns what
    ``fn`` returned -- and the artifact is written to the ``artifact_path`` /
    ``cache_path`` files when the block exits. It is the render-once counterpart
    of :class:`AccumulatingCapture`, which rewrites the same files on every call.
    """

    __module__ = "torch.compiler.precompile"

    def __enter__(self) -> Self:
        raise NotImplementedError

    def __exit__(self, *exc: object) -> None:
        raise NotImplementedError

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError


class _MakeFxCapture(Capture):
    r"""Single-shot capture: the :class:`MakeFxTracer` front-end.

    A make_fx trace records the ATen ops of ONE execution of ``fn``, so this
    captures exactly one call and refuses a second -- there is no notion of
    guards or recompiled variants here, and thus nothing a further call could
    add. Pass ``tracer=DynamoTracer()`` to capture several calls with the graph
    breaks and recompilations between them.
    """

    def __init__(
        self,
        fn: Callable[..., object],
        artifact_path: str | os.PathLike[str],
        cache_path: str | os.PathLike[str],
        *,
        backend: str,
        decompositions: dict | None,
        training: bool,
    ) -> None:
        if isinstance(fn, functools.partial):
            raise PrecompileError(
                "precompile cannot capture a partial. Pass the underlying function "
                "and give its bound arguments as call arguments."
            )
        self._module = PrecompiledModule(
            fn, backend=backend, tracer="make_fx", decompositions=decompositions
        )
        self._artifact_path = artifact_path
        self._cache_path = cache_path
        self._training = training
        self._traced = False
        self._rendered: tuple[str, bytes] | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        # Write the rendered artifact only on a clean exit that captured a call;
        # a block that raised, or one that never called the capture, leaves the
        # files untouched.
        if exc[0] is not None:
            return
        if self._rendered is None:
            raise PrecompileError(
                "nothing was captured: call the capture with your example "
                "arguments inside the `with` block."
            )
        _write_artifact(self._artifact_path, self._cache_path, *self._rendered)

    def __call__(self, *args: object, **kwargs: object) -> object:
        if kwargs:
            raise ValueError(
                "MakeFxTracer takes positional arguments only; pass "
                "tracer=DynamoTracer() to capture calls with keyword arguments."
            )
        if self._traced:
            raise PrecompileError(
                "MakeFxTracer captures a single call and one has already been "
                "traced. Use tracer=DynamoTracer() to capture several calls, with "
                "the graph breaks and recompilations between them."
            )
        self._traced = True
        # make_fx traces one execution of fn and lowers it to the artifact; we then
        # serve that artifact on the real args through the SAME load() path a caller
        # would take, so the value handed back is exactly what serving produces
        # (invariants checked, grads scattered onto the model) rather than a bare
        # trace with no result. Single-shot: a make_fx trace records one path, so a
        # second call has nothing new to add. Build python_code ONCE and thread it
        # into to_cache_bytes (the metadata + embedded kernel source is not rebuilt,
        # and code_hash is sha256 over exactly the bytes written on exit). The
        # caller drives the grad mode, for both the trace and the serve.
        with torch.enable_grad() if self._training else torch.no_grad():
            self._module._compile(args)
            python_code = self._module.to_python_code()
            self._rendered = (python_code, self._module.to_cache_bytes(python_code))
            return _runnable_from_pair(*self._rendered, _trusted=True)(*args)


class _DynamoCapture(Capture):
    r"""Multi-call capture: the :class:`DynamoTracer` front-end.

    Enter the ``with`` block, call it as many times as you need to exercise the
    graph breaks and recompiled variants you want captured; the artifact is
    rendered and written to disk once, when the block exits. The same
    execution-driven model as :class:`AccumulatingCapture`, without the per-call
    disk rewrite.
    """

    def __init__(
        self,
        session: Any,
        artifact_path: str | os.PathLike[str],
        cache_path: str | os.PathLike[str],
        *,
        backend: str,
        require_complete: bool,
        require_no_risky_drops: bool,
        require_no_dropped_guards: bool,
    ) -> None:
        self._session = session
        self._artifact_path = artifact_path
        self._cache_path = cache_path
        self._backend = backend
        self._require_complete = require_complete
        self._require_no_risky_drops = require_no_risky_drops
        self._require_no_dropped_guards = require_no_dropped_guards
        self._call: Callable[..., object] | None = None
        self._fresh_cache: Any = None
        self._exited = False
        self._calls = 0
        self._rendered: tuple[str, bytes] | None = None
        self._render_error: BaseException | None = None
        self._lock = threading.RLock()

    def _map(self, method: Callable[..., Any], *args: object, **kwargs: object) -> Any:
        from torch._dynamo.exc import PackageError, RecompileError

        try:
            return method(*args, **kwargs)
        except (PackageError, RecompileError) as e:
            raise PrecompileError(str(e)) from e

    def __enter__(self) -> Self:
        from torch.compiler._cache import CacheArtifactManager

        with self._lock:
            if self._call is not None or self._exited:
                raise PrecompileError(
                    "this capture has already been entered; capture() returns a "
                    "fresh capture per call."
                )
            # The capture's compiles record into the process-global cache-artifact
            # list, which result() serializes. A fresh one so the bundle holds
            # only this capture rather than unrelated pending compiles; entered
            # here and left in __exit__, spanning every captured call.
            self._fresh_cache = CacheArtifactManager.with_fresh_cache()
            self._fresh_cache.__enter__()
            try:
                self._call = self._map(self._session.__enter__)
            except BaseException:
                self._fresh_cache.__exit__(*sys.exc_info())
                self._fresh_cache = None
                raise
        return self

    def __call__(self, *args: object, **kwargs: object) -> object:
        with self._lock:
            if self._call is None or self._exited:
                raise PrecompileError(
                    "capture is not active: enter it with a `with` block before "
                    "calling it."
                )
            call = self._call
        result = self._map(call, *args, **kwargs)
        with self._lock:
            self._calls += 1
        return result

    def __exit__(self, *exc: object) -> None:
        with self._lock:
            self._exited = True
            try:
                self._map(self._session.__exit__, *exc)
                if exc[0] is None:
                    # Render now, while the fresh cache still holds this capture's
                    # compiles, then write the pair to disk. A gate refusal (or a
                    # write failure) is held and re-raised below, after the fresh
                    # cache is released, so summary()/invariants() stay readable.
                    try:
                        self._rendered = self._map(
                            self._session.artifact,
                            require_complete=self._require_complete,
                            require_no_risky_drops=self._require_no_risky_drops,
                            require_no_dropped_guards=self._require_no_dropped_guards,
                        )
                        _write_artifact(
                            self._artifact_path, self._cache_path, *self._rendered
                        )
                    except BaseException as e:
                        self._render_error = e
            finally:
                if self._fresh_cache is not None:
                    self._fresh_cache.__exit__(*sys.exc_info())
                    self._fresh_cache = None
        # A clean block whose render/write failed surfaces the failure here; a
        # block that itself raised propagates its own error and we stay quiet.
        if exc[0] is None and self._render_error is not None:
            raise self._render_error

    def summary(self) -> PrecompileSummary:
        r"""summary() -> PrecompileSummary

        Coverage, recompilation, failure and guard information for everything
        captured so far.
        """
        return self._map(self._session.summary)

    def invariants(self) -> tuple[FrameInvariants, ...]:
        r"""invariants() -> tuple

        The guards that held across every captured variant of each frame.
        """
        return self._map(self._session.invariants)

    def calls(self) -> int:
        r"""calls() -> int

        How many calls have been folded into this capture.
        """
        return self._calls


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


def _unbacked_guard_error(e: BaseException) -> PrecompileError:
    """The shared capture-time error for a guard on a mark_unbacked dim (both tracers).

    A mark_unbacked dim is captured as an unbacked symint (no hint), so a computation that
    needs to guard on / specialize its size (a shape-dependent branch, a reshape that pins
    it) cannot be captured. Unbacked dims cannot be guarded, so rather than bake a
    silently-wrong artifact, fail here.
    """
    return PrecompileError(
        "precompile: fn needs to guard on a dim marked with mark_unbacked "
        "(it branches on or specializes that size), which is not allowed for "
        "an unbacked dynamic dim. Do not mark that dim (capture it static), "
        "or restructure fn to avoid the size-dependent operation. Underlying: "
        f"{(str(e).splitlines() or [''])[0]}"
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
    that run for THIS example. Static (Python ``int``) control flow and shapes are
    specialized to ``args`` and baked; a data-dependent op (``.item()``, a branch
    over a tensor value) instead raises at capture, since this traces under fake
    mode where the value is unknown. The interning/order established here for params
    then buffers is the calling convention the runtime model must reproduce
    (invariant 2).
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
    from torch._subclasses.fake_tensor import FakeTensorMode

    # ``fake_mode`` is the DYNAMIC (symbolic) fake mode -- set only on the unbacked path,
    # where it threads its ShapeEnv to the lowering (and turns on scalar_asserts). The
    # static path also traces on fakes, but with a plain (non-symbolic) fake mode kept in
    # ``capture_cm`` only, so ``_Capture.fake_mode`` stays None and the lowering treats
    # the capture as static. Either way ``capture_cm`` is the FakeTensorMode we trace in.
    fake_mode = None
    capture_cm: FakeTensorMode
    if any(marks):
        flat_args, fake_mode = _fakeify_with_unbacked(pb_flat, user_flat, marks)
        capture_cm = fake_mode
        user_input_shapes = [
            None
            if base is None
            else tuple(None if i in per else s for i, s in enumerate(base))
            for base, per in zip(user_input_shapes, marks)
        ]
    else:
        # Static capture: fakeify every input so the trace runs no real compute (no
        # in-place input mutation, no grad on the example model). allow_non_fake_inputs
        # lets a real tensor that fn closes over (an unregistered attr, a global, a
        # captured constant -- invariant 1) flow through as a baked constant, so
        # _check_no_constant_tensors below rejects it with the same clean PrecompileError
        # a real trace gave, rather than a raw mixed-fake AssertionError.
        capture_cm = FakeTensorMode(allow_non_fake_inputs=True)
        with capture_cm:
            flat_args = [
                capture_cm.from_tensor(a, static_shapes=True)
                if isinstance(a, torch.Tensor)
                else a
                for a in flat_args
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

    # The caller picks the grad mode (no_grad unless training=True), so a backward
    # in ``fn`` is built as graph ops only under training. Restore .grad in finally
    # so a make_fx failure (e.g. fn raising after running a backward) does not
    # leave the user's example model with clobbered .grad fields.
    from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode

    # Trace on FAKE tensors either way (flat_args were fakeified above), so the trace
    # runs NO real compute and has no real side effects (no in-place input mutation, no
    # grad accumulation on the example model): the single real execution is the graph run
    # on the real inputs, which the caller-driven capture does by serving the built
    # artifact and handing back its result. The unbacked path keeps its symbolic ShapeEnv
    # ("symbolic"); a static capture uses concrete fake shapes ("fake"). A data-dependent
    # op (.item(), .nonzero(), a tensor-value branch) that a real trace would silently
    # specialize now raises at capture rather than baking an unsound constant.
    tracing_mode = "symbolic" if fake_mode is not None else "fake"
    try:
        with capture_cm:
            try:
                gm = make_fx(
                    flat_fn,
                    decomposition_table=decompositions,
                    tracing_mode=tracing_mode,
                )(flat_args)
            except GuardOnDataDependentSymNode as e:
                raise _unbacked_guard_error(e) from e
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


def _build_metadata_section(buf: PySourceBuilder, compiled: PrecompiledModule) -> None:
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
    buf.writeline("# " + "=" * 70)
    buf.writeline("# 2. Calling-convention metadata")
    buf.writeline("# " + "=" * 70)
    buf.writeline("import torch as _torch")
    buf.writeline("import torch.utils._pytree as _pytree")
    buf.writeline("")
    # python_code is the single source of truth for the calling convention; the
    # cache holds ONLY the compiled/captured artifact. load() reads these
    # constants back out of python_code (see _parse_artifact_metadata).
    buf.writeline(f"BACKEND = {compiled._backend!r}")
    buf.writeline(f"MODULE_POSITIONS = {compiled._module_positions!r}")
    # Number of positional args the traced fn took (modules + runtime inputs); the
    # driver checks the runtime call passes the same count up front, so a wrong
    # arity raises a clear PrecompileError instead of a raw IndexError.
    buf.writeline(f"NUM_POSITIONAL_ARGS = {compiled._num_positional_args}")
    buf.writeline(f"PARAM_NAMES = {compiled._param_names!r}")
    buf.writeline(f"BUFFER_NAMES = {compiled._buffer_names!r}")
    # Per interned param / buffer example shape / dtype / device (aligned to
    # PARAM_NAMES / BUFFER_NAMES); the driver checks each runtime param/buffer against
    # these for the structural contract (invariant 2).
    buf.writeline(f"PARAM_SHAPES = {compiled._param_shapes!r}")
    buf.writeline(f"BUFFER_SHAPES = {compiled._buffer_shapes!r}")
    buf.writeline(f"PARAM_DTYPES = {compiled._param_dtypes!r}")
    buf.writeline(f"BUFFER_DTYPES = {compiled._buffer_dtypes!r}")
    buf.writeline(f"PARAM_DEVICES = {compiled._param_devices!r}")
    buf.writeline(f"BUFFER_DEVICES = {compiled._buffer_devices!r}")
    # Which unique-param index each trailing grad output belongs to (see invariant 5);
    # the driver scatters grad k onto params[GRAD_PARAM_INDICES[k]].
    buf.writeline(f"GRAD_PARAM_INDICES = {compiled._grad_param_indices!r}")
    # The pytree structure of the runtime inputs, or None if not serializable (the
    # driver validates against it when present, else skips the structure check).
    buf.writeline(f"IN_SPEC = {in_spec_str!r}")
    buf.writeline(f"OUT_SPEC = {out_spec_str!r}")
    # Per user-input-leaf example shape / dtype / device (None for a non-tensor /
    # subclass leaf); the drivers reject a runtime mismatch (invariants 3 and 6).
    # Memory-format mismatches are caught by the inductor artifact's own
    # assert_size_stride (pinned on at capture).
    # Every device type the captured graph dispatches on, from the GRAPH
    # rather than from the runtime tensors: the drivers neutralize ambient
    # autocast on these, and a graph can reach a device none of its inputs
    # live on (an explicit .to("cuda") inside fn) or have no tensor inputs
    # at all. Artifacts written before this field carry their own, older
    # driver, which never reads it.
    buf.writeline(
        f"GRAPH_DEVICES = {_graph_device_types(compiled._gm) if compiled._gm is not None else ()!r}"
    )
    buf.writeline(f"USER_INPUT_SHAPES = {compiled._user_input_shapes!r}")
    buf.writeline(f"USER_INPUT_DTYPES = {compiled._user_input_dtypes!r}")
    buf.writeline(f"USER_INPUT_DEVICES = {compiled._user_input_devices!r}")
    # Per user-input-leaf mark_unbacked min/max bounds: None for a leaf with no bounded
    # marked dim, else {dim: (lo, hi)} (either may be None). The drivers reject a
    # runtime size outside the declared range (invariant 3); see the inlined drivers.
    buf.writeline(f"USER_INPUT_BOUNDS = {compiled._user_input_bounds!r}")
    buf.writeline("")


def _read_literal(tree: object, name: str) -> object:
    """One top-level ``NAME = <literal>`` out of a parsed artifact, else None."""
    import ast

    for node in cast("ast.Module", tree).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            try:
                return ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                return None
    return None


def _parse_artifact_metadata(python_code: str) -> dict[str, object]:
    """Read the calling-convention constants back out of ``python_code`` WITHOUT
    executing it (exec'ing the inlined Inductor output would JIT the kernels, the
    very work the cache exists to skip).

    python_code is the single source of truth: the metadata builders emit the constants
    below as top-level literal assignments, so an AST walk + literal_eval recovers them
    safely. The cache then only needs to carry the compiled artifact.

    The required constant set is tracer-dependent: the make_fx tracer emits the full
    calling-convention set the inlined driver reads (PARAM_NAMES, OUT_SPEC, ...), while
    the dynamo tracer's driver reads its own (BACKEND_ID, IMPORT_SOURCES) and rehydrates
    the rest from opaque blobs. TRACER is absent on artifacts predating the dynamo tracer,
    so its absence means make_fx.
    """
    import ast

    try:
        tree = ast.parse(python_code)
    except SyntaxError as e:
        raise PrecompileError(
            "python_code is not valid Python; it does not look like a "
            "torch.compiler.precompile artifact."
        ) from e
    # The make_fx and dynamo drivers read different calling-convention literals,
    # so TRACER picks the required set. It is absent on artifacts predating the
    # dynamo tracer, which are all make_fx.
    if _read_literal(tree, "TRACER") == "dynamo":
        # The multi-graph driver rehydrates every frame from _FRAMES and the
        # subgraphs from _BACKENDS; the readable literals beside them describe
        # what is in those blobs and are validated so a truncated artifact fails
        # here rather than deep inside the driver.
        wanted = {
            "BACKEND",
            "TRACER",
            "FN_NAME",
            "FRAMES",
            "DROPPED_GUARDS",
            "RISKY_DROPPED_GUARDS",
            "WONT_GENERALIZE",
            "_FRAMES",
            "_BACKENDS",
            "_DYNAMO_PYTHON_VERSION",
        }
        # An installed artifact carries the whole package in one blob instead of
        # the per-frame records, so it reads a different set. SERVING_MODE is
        # absent on artifacts predating it, which were all standalone.
        if _read_literal(tree, "SERVING_MODE") == "installed":
            wanted -= {"_FRAMES", "_BACKENDS"}
            wanted |= {"SERVING_MODE", "_PACKAGE", "UNREACHABLE_WITHOUT_INSTALL"}
        wanted |= {"_ENTRY_BINDING"}
    else:
        wanted = {
            "BACKEND",
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
    found: dict[str, object] = {}
    # Parsed when present, never required: the guard-audit sections are
    # reporting, and artifacts predating them load unchanged. An auditor
    # reading a shipped artifact wants them back as data rather than by
    # grepping the source.
    optional = {"POLICY_DROPPED_GUARDS", "DROPPED_GUARD_CODE"}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id in wanted or target.id in optional:
            try:
                found[target.id] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError) as e:
                raise PrecompileError(
                    f"python_code {target.id!r} calling-convention metadata is "
                    f"malformed; it must be a Python literal."
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
    missing = wanted - found.keys()
    if missing:
        raise PrecompileError(
            f"python_code is missing calling-convention metadata {sorted(missing)}; "
            "it does not look like a torch.compiler.precompile artifact."
        )
    # Reported but not required: artifacts predating the installed serving mode
    # carry no SERVING_MODE, and they were all standalone.
    found.setdefault(
        "SERVING_MODE", _read_literal(tree, "SERVING_MODE") or "standalone"
    )
    return found


def _build_python_source(
    compiled: PrecompiledModule,
    graph_python: str,
) -> str:
    from torch._functorch._aot_autograd.codegen import PySourceBuilder

    buf = PySourceBuilder()
    buf.writeline(_GENERATED_HEADER)
    buf.writeline("")
    buf.writeline("# " + "=" * 70)
    buf.writeline("# 1. Compiled graph (AOTAutograd + Inductor): exposes ``call``")
    buf.writeline("# " + "=" * 70)
    # The composed graph module from aot_autograd.compile_to_python: the inlined
    # Inductor kernels plus AOTAutograd's codegen'd prelude/epilogue, exposing
    # ``call(flat_inputs) -> outputs`` (subclass + mutation handled inside).
    buf.writeline(graph_python)
    buf.writeline("")
    _build_metadata_section(buf, compiled)
    buf.writeline("# " + "=" * 70)
    buf.writeline(
        "# 3. Driver: module params/buffers + grad scatter + calling convention"
    )
    buf.writeline("# " + "=" * 70)
    buf.writeline(_emit_driver_source("_inductor_forward"))
    return buf.getvalue()


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
    from torch._functorch._aot_autograd.codegen import PySourceBuilder

    buf = PySourceBuilder()
    buf.writeline(_EAGER_GENERATED_HEADER)
    buf.writeline("")
    buf.writeline("# " + "=" * 70)
    buf.writeline("# 1. Captured ATen graph (eager backend) -- executable and readable")
    buf.writeline("# " + "=" * 70)
    # gm.code relies on fx's custom builtins (torch, device, inf, nan, NoneType,
    # fx_pytree, pytree) being in scope -- fx injects them when a real GraphModule
    # runs. Reproduce the FULL set (not just torch/pytree) so a graph that bakes a
    # device / inf / nan constant (e.g. BatchNorm, masked_fill to -inf) runs
    # standalone instead of raising NameError. Sourced from fx so it stays correct.
    from torch.fx.graph import _custom_builtins

    for _cb in _custom_builtins.values():
        buf.writeline(_cb.import_str)
    buf.writeline(graph_src)
    buf.writeline("")
    buf.writeline("class _GraphSelf:")
    buf.writeline(f"    _in_spec = pytree.treespec_loads({in_spec_str!r})")
    buf.writeline(f"    _out_spec = pytree.treespec_loads({out_spec_str!r})")
    buf.writeline("")
    buf.writeline("")
    buf.writeline("def call(args):")
    buf.writeline("    out = _graph_forward(_GraphSelf(), list(args))")
    buf.writeline("    return list(out) if isinstance(out, (list, tuple)) else [out]")
    buf.writeline("")
    _build_metadata_section(buf, compiled)
    buf.writeline("# " + "=" * 70)
    buf.writeline("# 3. Driver: run the inlined captured graph eagerly")
    buf.writeline("# " + "=" * 70)
    buf.writeline(_emit_driver_source("_eager_forward"))
    return buf.getvalue()


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


def _graph_device_types(gm: torch.fx.GraphModule) -> tuple[str, ...]:
    """Every device type the graph dispatches on, from its node metadata.

    Derived from the GRAPH, not from the runtime params and inputs: a graph can
    reach a device none of its inputs live on (an explicit ``.to("cuda")`` in
    the middle of fn), and a graph built only from factory ops has no input
    device at all. Both cases leave a runtime scan blind exactly where an
    ambient-state leak needs closing.
    """
    from torch._dynamo.graph_utils import _graph_device_types as _scan

    return tuple(
        sorted(d for d in _scan(gm.graph) if torch.amp.is_autocast_available(d))
    )


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


def _b64(payload: object) -> str:
    return base64.b64encode(pickle.dumps(payload)).decode("ascii")


def _multigraph_frames(entry: Any) -> list[dict[str, Any]]:
    """One record per Dynamo frame, in the form the driver rebuilds from.

    A standalone artifact can reach exactly two kinds of frame: the ENTRY frame,
    which the caller invokes, and a continuation, which the frame ahead of it
    reaches by LOAD_GLOBAL on the name capture minted. Anything else Dynamo
    compiled is entered by an ordinary Python call that only the frame evaluator
    intercepts, so it has no place in a source artifact -- see
    _reject_unreachable_frames.
    """
    from torch._dynamo.package import SerializedCode

    entry_name = str(entry.fn_name).rsplit(".", 1)[-1]
    frames = []
    seen_entry = False
    for code in entry.codes:
        if code.bypassed:
            continue
        name = SerializedCode.to_code_object(code.python_code).co_name
        is_entry = not code.install_to_global and name == entry_name and not seen_entry
        seen_entry = seen_entry or is_entry
        frames.append(
            {
                "is_entry": is_entry,
                "code": code.python_code,
                "python_module": code.python_module,
                "import_sources": dict(code.import_sources),
                "resume_names": (
                    list(code.function_names) if code.install_to_global else []
                ),
                "variants": [
                    {
                        "guards_state": guarded.guards_state,
                        "dynamo_code": guarded.dynamo_code,
                    }
                    for guarded in code.guarded_codes
                ],
            }
        )
    return frames


def _build_multigraph_python_source(
    entry: Any,
    backends: Mapping[str, Any],
    summary: Any,
    backend: str,
    serving_mode: str = "standalone",
    package_entry: object = None,
    entry_binding: dict[str, Any] | None = None,
    rendered: Mapping[str, str] | None = None,
    refused: Mapping[str, str] | None = None,
) -> str:
    """Render a multi-graph capture as ``python_code``.

    The readable half -- what was captured, which frames, how many variants, what
    guards were dropped -- is emitted as literals so a reviewer can diff it, and
    the compiled subgraphs are rendered as source beside them. Only the guard
    trees and the transformed bytecode, which have no source form, go into the
    clearly banners' opaque blobs.
    """
    from torch._dynamo.package import SerializedCode
    from torch._functorch._aot_autograd.codegen import PySourceBuilder

    frames = _multigraph_frames(entry)
    buf = PySourceBuilder()
    buf.writeline(_MULTIGRAPH_GENERATED_HEADER)
    buf.writeline(_SERVING_NOTES[serving_mode])
    buf.writeline("")
    buf.writeline("# " + "=" * 70)
    buf.writeline("# 1. What was captured (readable)")
    buf.writeline("# " + "=" * 70)
    buf.writeline(f"BACKEND = {backend!r}")
    buf.writeline('TRACER = "dynamo"')
    buf.writeline(f"FN_NAME = {entry.fn_name!r}")
    buf.writeline(f"FN_FIRST_LINENO = {entry.fn_first_lineno!r}")
    buf.writeline(f"_DYNAMO_PYTHON_VERSION = {tuple(sys.version_info[:2])!r}")
    buf.writeline(f"TORCH_VERSION = {torch.__version__!r}")
    buf.writeline("")
    buf.writeline(
        "# frame name -> number of captured variants. A frame with one variant"
    )
    buf.writeline("# serves one specialization; the artifact covers no other.")
    buf.writeline("FRAMES = [")
    for frame, code in zip(frames, [c for c in entry.codes if not c.bypassed]):
        name = SerializedCode.to_code_object(code.python_code).co_name
        buf.writeline(f"    ({name!r}, {len(frame['variants'])}),")
    buf.writeline("]")
    buf.writeline("")
    buf.writeline(
        "# Guards that could NOT be serialized and are therefore not checked at"
    )
    buf.writeline("# serve time. Rebinding any of these between capture and load can")
    buf.writeline(
        "# silently select the wrong graph -- audit them against your deployment."
    )
    buf.writeline(f"DROPPED_GUARDS = {[list(g) for g in summary.dropped_guards]!r}")
    buf.writeline(
        f"RISKY_DROPPED_GUARDS = {[list(g) for g in summary.risky_dropped_guards]!r}"
    )
    buf.writeline("")
    buf.writeline("# Guards that COULD be serialized and were not, because they held")
    buf.writeline("# identically in every captured variant so they discriminated")
    buf.writeline("# nothing. Not checked at serve time either: a call outside the")
    buf.writeline("# captured domain along one of these is served, not refused.")
    buf.writeline(
        f"POLICY_DROPPED_GUARDS = {[list(g) for g in summary.policy_dropped_guards]!r}"
    )
    buf.writeline("")
    buf.writeline("# What a dropped slot above actually checked, where it renders one.")
    buf.writeline("# A slot is named by")
    buf.writeline("# its type and SOURCE, which for some types does not say enough to")
    buf.writeline("# judge the drop: a dropped HASATTR on a source may be the benign")
    buf.writeline("# companion of a kept TENSOR_MATCH on the same source, or the only")
    buf.writeline("# thing pinning an optional attribute. The rendered check names the")
    buf.writeline("# attribute and tells the two apart.")
    buf.writeline(
        f"DROPPED_GUARD_CODE = "
        f"{[list(g) for g in getattr(summary, 'dropped_guard_code', ())]!r}"
    )
    buf.writeline("")
    buf.writeline(
        "# Values pinned to exactly what capture saw; any other value misses."
    )
    buf.writeline(f"WONT_GENERALIZE = {tuple(summary.wont_generalize)!r}")
    buf.writeline("")
    if serving_mode == "installed":
        reachable = _reachable_frames(frames)
        dead = sorted(
            SerializedCode.to_code_object(frame["code"]).co_name
            for i, frame in enumerate(frames)
            if frame["variants"] and i not in reachable
        )
        buf.writeline("# " + "=" * 70)
        buf.writeline("# 2. How this artifact serves: INSTALLED (readable)")
        buf.writeline("#")
        buf.writeline("# A source artifact dispatches only the entry frame and the")
        buf.writeline(
            "# continuations its bytecode names. These frames are entered by an"
        )
        buf.writeline(
            "# ordinary call, so nothing in the entry reaches them and they would"
        )
        buf.writeline(
            "# run eager. This artifact therefore installs onto the live code"
        )
        buf.writeline("# objects instead, which reaches every frame.")
        buf.writeline("#")
        buf.writeline("# load() installs NOTHING; entering the result -- or calling")
        buf.writeline("# it -- installs, and unload()/exit removes it again.")
        buf.writeline("# " + "=" * 70)
        buf.writeline('SERVING_MODE = "installed"')
        buf.writeline(f"UNREACHABLE_WITHOUT_INSTALL = {tuple(dead)!r}")
        buf.writeline("")
        buf.writeline("# " + "=" * 70)
        buf.writeline("# 3. The captured package -- OPAQUE")
        buf.writeline("#")
        buf.writeline(
            "# base64(pickle) of the serialized Dynamo state (per-frame guard"
        )
        buf.writeline(
            "# trees and transformed bytecode) plus the compiled subgraphs. Guard"
        )
        buf.writeline(
            "# trees are a spec for a C++ GuardManager and have no readable form;"
        )
        buf.writeline("# the counts and names above describe what is in here.")
        buf.writeline("# " + "=" * 70)
        buf.writeline(f"_PACKAGE = {_b64(package_entry)!r}")
        buf.writeline("")
        buf.writeline("# The entry's default arguments; see the standalone note")
        buf.writeline("# above -- the installed driver rebuilds an entry too.")
        buf.writeline(f"_ENTRY_BINDING = {_b64(entry_binding or {})!r}")
        buf.writeline("")
        buf.writeline("# " + "=" * 70)
        buf.writeline("# 4. Driver: rebuild the package, install, dispatch (readable)")
        buf.writeline("# " + "=" * 70)
        buf.writeline(_emit_installed_driver_source())
        return buf.getvalue()

    buf.writeline('SERVING_MODE = "standalone"')
    buf.writeline("")
    buf.writeline(
        "# Every function Dynamo INLINED into a captured graph, so the driver"
    )
    buf.writeline("# can tell that the source it is about to trust still says what it")
    buf.writeline(
        "# said at capture. The installed mode gets this from CompilePackage;"
    )
    buf.writeline(
        "# a standalone artifact builds no package, so it carries the records."
    )
    buf.writeline("# __main__ is skipped: it names the LOADER's script on another")
    buf.writeline("# machine, which is exactly what a portable artifact is for.")
    buf.writeline(
        f"INLINED_SOURCES = "
        f"{sorted((s.module, s.firstlineno, s.lastlineno, s.checksum) for s in entry.source_info.inlined_sources if s.module != '__main__')!r}"
    )
    buf.writeline("")
    buf.writeline("# The entry's defaults and closure values: a code object carries")
    buf.writeline("# neither, and the driver rebuilds the entry from one.")
    try:
        binding_blob = _b64(entry_binding or {})
    except Exception as e:
        raise PrecompileError(
            f"precompile cannot carry {entry.fn_name!r}'s default arguments in the "
            f"artifact; defaults must be picklable ({type(e).__name__}: {e})."
        ) from e
    buf.writeline(f"_ENTRY_BINDING = {binding_blob!r}")
    buf.writeline("")
    buf.writeline("# " + "=" * 70)
    buf.writeline("# 2. Guard trees and transformed bytecode -- OPAQUE")
    buf.writeline("#")
    buf.writeline(
        "# base64(pickle) of one record per frame: the frame's code object, its"
    )
    buf.writeline(
        "# variants' serialized guard state and Dynamo bytecode, and the globals"
    )
    buf.writeline(
        "# it reads. Guard trees are a spec for a C++ GuardManager and have no"
    )
    buf.writeline(
        "# readable form; the counts and names above describe what is in here."
    )
    buf.writeline("# " + "=" * 70)
    buf.writeline(f"_FRAMES = {_b64(frames)!r}")
    buf.writeline("")
    # A compiled subgraph is Inductor output, which HAS a source form -- unlike
    # the guard trees and bytecode above. Emit that source where the backend can
    # produce it, and fall back to the pickle only for the rest (eager graphs,
    # anything the lowering refused). rendered_backends already suffixed every
    # top-level name each block defines per slot (namespace_module_names), so the
    # blocks splice sequentially into ONE namespace and resolve siblings late
    # without a variant silently running another variant's code.
    rendered = dict(rendered or {})
    buf.writeline("# " + "=" * 70)
    if rendered:
        buf.writeline(f"# 3. Compiled subgraphs -- {len(rendered)} READABLE below")
        if len(rendered) < len(backends):
            buf.writeline(
                f"#    ({len(backends) - len(rendered)} could not be rendered and "
                f"stay in _BACKENDS)"
            )
    else:
        buf.writeline("# 3. Compiled subgraphs -- OPAQUE")
    # Why a subgraph stayed pickled, so the fallback is visible in the artifact
    # itself and not only in a warning at capture time.
    for backend_id, reason in (refused or {}).items():
        buf.writeline(f"#    {backend_id} stays pickled: {reason}")
    buf.writeline("#")
    buf.writeline("# base64(pickle) of the backend artifacts the frames call by name.")
    buf.writeline("# " + "=" * 70)
    buf.writeline(
        f"_BACKENDS = {_b64({k: v for k, v in backends.items() if k not in rendered})!r}"
    )
    if rendered:
        buf.writeline("")
        buf.writeline("_SUBGRAPHS = {}")
        _seen_slots: list[str] = []
        for backend_id, source in rendered.items():
            buf.writeline("")
            buf.writeline("# " + "-" * 70)
            buf.writeline(f"# subgraph {backend_id}")
            buf.writeline("# " + "-" * 70)
            buf.writeline(source)
            # the block's entry was renamed along with everything else it defines
            entry_name = f"call_s{len(_seen_slots)}"
            _seen_slots.append(backend_id)
            buf.writeline(f"_SUBGRAPHS[{backend_id!r}] = {entry_name}")
    buf.writeline("")
    buf.writeline("# " + "=" * 70)
    buf.writeline(
        "# 4. Driver: rebuild the guards, wire the names, dispatch (readable)"
    )
    buf.writeline("# " + "=" * 70)
    buf.writeline(_emit_multigraph_driver_source())
    return buf.getvalue()


def _reachable_frames(frames: list[dict[str, Any]]) -> set[int]:
    """Indices of the frames a standalone driver can actually dispatch.

    The entry is reachable, and a continuation is reachable only once some
    ALREADY reachable frame's bytecode names it. Asking merely whether a frame
    carries a resume name is not the same question: a continuation whose parent
    is itself unreachable is just as dead, and counting it as covered
    under-reports how much of the artifact will run eager.
    """
    from torch._dynamo.package import SerializedCode

    def named_globals(frame: dict[str, Any]) -> set[str]:
        out: set[str] = set()
        stack = [
            SerializedCode.to_code_object(variant["dynamo_code"])
            for variant in frame["variants"]
        ]
        seen: set[int] = set()
        while stack:
            code = stack.pop()
            if id(code) in seen:
                continue
            seen.add(id(code))
            out.update(code.co_names)
            stack.extend(c for c in code.co_consts if isinstance(c, types.CodeType))
        return out

    reachable = {i for i, frame in enumerate(frames) if frame["is_entry"]}
    while True:
        named: set[str] = set()
        for i in reachable:
            named |= named_globals(frames[i])
        grew = {
            i
            for i, frame in enumerate(frames)
            if i not in reachable and any(n in named for n in frame["resume_names"])
        }
        if not grew:
            return reachable
        reachable |= grew


def _serving_mode(frames: list[dict[str, Any]]) -> str:
    """``"standalone"`` when a source artifact covers every captured frame.

    Anything it cannot reach would run eager, silently giving up the compiled
    variant, so those captures are served by installing instead.
    """
    reachable = _reachable_frames(frames)
    unreachable = [
        i for i, frame in enumerate(frames) if frame["variants"] and i not in reachable
    ]
    return "installed" if unreachable else "standalone"


def _reject_uninstallable_entry(frames: list[dict[str, Any]], entry: Any) -> None:
    """Refuse a capture whose entry an installed artifact could not rebuild.

    The installed driver rebuilds the entry from its code object, and
    ``types.FunctionType`` cannot supply a closure or defaults it was not given:
    a closure entry fails to build at all, and a defaulted argument silently
    goes missing on the first served call. Both are capture-time facts, so they
    are refused here rather than at load on the serving machine.
    """
    from torch._dynamo.package import SerializedCode

    entry_frames = [f for f in frames if f["is_entry"]]
    if not entry_frames or not any(f["variants"] for f in entry_frames):
        # A missing or variant-less entry frame has very different causes, and
        # guessing the wrong one sends the caller restructuring code that was
        # never the problem. If Dynamo BYPASSED the entry frame, it recorded
        # why -- and _multigraph_frames DROPS bypassed codes entirely, so the
        # bypassed-entry case arrives here as "no entry frame at all", not as
        # an entry with no variants. Say the recorded reason, because the
        # thin-wrapper advice below is then simply wrong. Only the ENTRY's own
        # bypassed codes count (matched by the same heuristic that picks entry
        # frames): an unrelated bypassed helper frame must not relabel a
        # thin-wrapper entry as a bypass.
        entry_name = str(entry.fn_name).rsplit(".", 1)[-1]
        bypassed = [
            code
            for code in entry.codes
            if code.bypassed
            and code.bypass_reason
            and not code.install_to_global
            and SerializedCode.to_code_object(code.python_code).co_name == entry_name
        ]
        if bypassed:
            reasons = ", ".join(sorted({str(c.bypass_reason) for c in bypassed}))
            raise PrecompileError(
                f"precompile captured no dispatchable graph for {entry.fn_name!r}: "
                f"{len(bypassed)} entry frame(s) were BYPASSED during capture, so "
                f"their guards were never written. Reason: {reasons}. Fix that "
                f"rather than restructuring the captured callable."
            )
        if not entry_frames:
            # Not bypassed and not compiled at all: nothing here can say why,
            # and both diagnostics below would be guesses.
            return
        # Handing precompile a bare nn.Module compiles Dynamo's own wrapper
        # frame (external_utils.wrap_inline's `inner`) rather than the module:
        # every graph lands there, closing over the module, and the entry frame
        # itself holds nothing. Load cannot rebuild that closure, and `inner`'s
        # code object is shared by every wrap_inline in the process, so serving
        # it would let an unrelated frame hit these guards.
        raise PrecompileError(
            f"precompile captured no dispatchable graph for {entry.fn_name!r}. The "
            f"entry frame produced no guarded code, so the artifact would serve "
            f"nothing. This happens when the captured callable is a thin wrapper -- "
            f"an nn.Module, or a forward that immediately delegates -- where Dynamo "
            f"compiles the wrapper's inner frame instead. Capture the function that "
            f"CALLS the model, e.g. "
            f"precompile.capture(lambda m, x: m(x), ...) and calling cap(model, x)."
        )
    code = SerializedCode.to_code_object(entry_frames[0]["code"])
    if code.co_freevars:
        raise PrecompileError(
            f"precompile cannot build a self-contained artifact for {entry.fn_name!r}: "
            f"it closes over {list(code.co_freevars)!r}, and this capture has to rebuild "
            f"the entry from its code object, which cannot restore a closure. Capture a "
            f"module-level function that takes what it needs as arguments, e.g. "
            f"precompile.capture(step, ...), calling cap(model, x), with "
            f"'def step(model, x): return model(x)'."
        )


def _reject_unreachable_frames(frames: list[dict[str, Any]], entry: Any) -> None:
    """Check what a standalone artifact will actually be able to dispatch.

    Dispatch here is explicit: the caller invokes the entry frame, and a
    continuation is reached by LOAD_GLOBAL from the frame ahead of it. A frame
    Dynamo compiled that is entered by an ordinary Python call -- an inner
    ``nn.Module.__call__`` wrapper, an un-inlinable helper -- is reachable only
    through the frame evaluator, which a source artifact does not use.

    Such a frame runs EAGER when served. That is a coverage and performance
    gap, not a correctness one: eager is what the graph was traced from, so the
    answer is the same. It is therefore a warning.
    """
    from torch._dynamo.package import SerializedCode

    reachable = _reachable_frames(frames)
    unreachable = [
        f for i, f in enumerate(frames) if f["variants"] and i not in reachable
    ]
    if unreachable:
        # Correct but under-compiled. A frame Dynamo compiled that is entered by
        # an ORDINARY call -- an un-inlinable helper, a separately-compiled
        # callee -- is reached only through the frame evaluator, which a source
        # artifact does not use. It runs eager instead, so the answer stays
        # right and the compiled variant simply goes unused. Say so, rather than
        # let summary() imply coverage the artifact will not deliver.
        names = sorted(
            SerializedCode.to_code_object(f["code"]).co_name for f in unreachable
        )
        log.warning(
            "precompile: %d captured frame(s) will run EAGER when served, because a "
            "self-contained artifact reaches only the entry frame and the "
            "continuations its bytecode names: %s. The result stays correct; those "
            "graphs are simply unused. Inline them into the captured callable to "
            "get their compiled versions.",
            len(names),
            names,
        )


def _entry_binding(fn: object) -> dict[str, Any]:
    """The default arguments an entry's code object does not carry.

    The artifact rebuilds its entry from that entry's code object, which holds
    no default arguments. Without them a defaulted parameter is simply absent at
    the served call -- which the guard check then cannot bind, so every variant
    misses. Closure cells are not carried: an entry that closes over free
    variables is refused before we reach here (a rebuilt cell is a new object
    that Dynamo's identity guard would miss), so a valid entry has none.
    """
    return {
        "defaults": getattr(fn, "__defaults__", None),
        "kwdefaults": getattr(fn, "__kwdefaults__", None),
    }


def _reject_uninstallable_entry_defaults(fn: object) -> None:
    """Refuse an entry whose default arguments cannot be carried in the artifact.

    The artifact rebuilds the entry from its code object and re-attaches the
    entry's defaults, pickled into the source. A tensor default would bake a
    weight in (precompile embeds none), and any unpicklable default would fail
    only at write time, after the capture has already run. Refuse both up front.
    """
    from torch._dynamo.precompile_package import _entry_fn_of

    try:
        entry = _entry_fn_of(fn)
    except TypeError:
        # Not a plain function or nn.Module entry (e.g. a partial): it carries
        # no defaults to check, and the capture path rejects it with its own
        # message (see _capture_session).
        return
    name = getattr(entry, "__name__", repr(entry))
    defaults = list(getattr(entry, "__defaults__", None) or ())
    defaults += list((getattr(entry, "__kwdefaults__", None) or {}).values())
    for value in defaults:
        if isinstance(value, torch.Tensor):
            raise PrecompileError(
                f"precompile cannot capture {name!r}: it has a tensor default "
                f"argument, which the artifact would bake in as a weight. "
                f"precompile embeds no weights -- pass the tensor as an argument."
            )
        try:
            pickle.dumps(value)
        except Exception as e:
            raise PrecompileError(
                f"precompile cannot capture {name!r}: its default argument "
                f"{value!r} cannot be pickled into the artifact ({e}). Give the "
                f"parameter a picklable default, or pass the value as an argument."
            ) from e


def _build_multigraph_artifact(
    entry: Any,
    backends: Mapping[str, Any],
    summary: Any,
    backend: str,
    entry_fn: object = None,
    rendered: Mapping[str, str] | None = None,
    refused: Mapping[str, str] | None = None,
) -> tuple[str, bytes]:
    """``(python_code, cache)`` for a multi-graph capture.

    python_code is self-contained -- it carries the frames, the guard trees and
    the compiled subgraphs -- so cache is the same acceleration it is for the
    single-graph forms: the inductor bundle that primes the kernel caches, and
    the tag binding it to this python_code (invariant 7).
    """
    frames = _multigraph_frames(entry)
    serving_mode = _serving_mode(frames)
    package_entry = None
    if serving_mode == "installed":
        # Frames a source artifact cannot reach would run eager, so serve this
        # capture by installing instead. That needs the package itself, not the
        # per-frame records the standalone driver rebuilds from.
        from torch._dynamo.package import PrecompileCacheEntry

        package_entry = PrecompileCacheEntry(entry, cast("dict[Any, Any]", backends))
    else:
        _reject_unreachable_frames(frames, entry)
    # Defaults the artifact can carry; closure cells it cannot. Dynamo guards a
    # cell by identity, and a rebuilt cell holding the same value is a different
    # object, so a closure entry would load and then miss every variant.
    _reject_uninstallable_entry(frames, entry)

    python_code = _build_multigraph_python_source(
        entry,
        backends,
        summary,
        backend,
        serving_mode,
        package_entry,
        _entry_binding(entry_fn),
        dict(rendered or {}),
        refused,
    )
    inductor_bundle = None
    if backend != "eager":
        # The dynamo capture runs under with_fresh_cache(), so this is
        # exactly what the capture's compiles recorded.
        from torch.compiler._cache import CacheArtifactManager

        saved = CacheArtifactManager.serialize()
        inductor_bundle = None if saved is None else saved[0]
    buf = io.BytesIO()
    torch.save(
        {
            "format": _CACHE_FORMAT,
            "version": _CACHE_VERSION,
            "backend": backend,
            "tracer": "dynamo",
            "code_hash": hashlib.sha256(python_code.encode()).hexdigest(),
            "artifact": inductor_bundle,
        },
        buf,
    )
    return python_code, buf.getvalue()


def _emit_multigraph_driver_source() -> str:
    """Emit the multi-graph driver as text, the same getsource path the others use."""

    from torch import _precompile_driver as driver

    body = inspect.getsource(driver._build_multigraph_forward).rstrip()
    return "\n" + body + "\n\n\nforward = _build_multigraph_forward()\n"


def _emit_installed_driver_source() -> str:
    """Emit the installing driver as text, the same getsource path the others use."""

    from torch import _precompile_driver as driver

    body = inspect.getsource(driver._build_installed_forward).rstrip()
    return "\n" + body + "\n\n\nforward = _build_installed_forward()\n"


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


class PrecompiledModule(PrecompiledRunnable):
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

    def _compile(self, args: tuple[object, ...]) -> None:
        # PrecompiledModule is the make_fx path only: the DynamoTracer front-end is routed
        # to the execution-driven capture before it gets here.
        if self._backend == "eager" and _has_unbacked_marks(args):
            raise NotImplementedError(
                "precompile: mark_unbacked (dynamic shapes) with MakeFxTracer is only "
                "supported with backend='inductor'; make_fx + eager + unbacked is not "
                "supported (DynamoTracer supports either backend)."
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
        options: dict[str, object] = {"size_asserts": True}
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
        # A PrecompiledModule is runnable only after load(); a capture instead
        # renders (python_code, cache) rather than a runnable.
        if self._loaded_forward is None:
            raise PrecompileError(
                "this object is not runnable; build one with "
                "torch.compiler.precompile.load(python_code, cache)."
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
                "tracer": self._tracer,
                "code_hash": code_hash,
                "artifact": self._artifact_bytes,
            },
            buf,
        )
        return buf.getvalue()


def _capture_session(fn, **kwargs):
    """Start an internal multi-graph capture, mapping package errors to ours.

    precompile.capture() and precompile.accumulate() drive the capture through
    the caller-driven capture objects, so this exists to keep the error
    translation of starting the underlying session in one place.
    """
    from torch._dynamo.exc import PackageError
    from torch._dynamo.precompile_package import precompile_capture

    if isinstance(fn, functools.partial):
        raise PrecompileError(
            "precompile cannot capture a partial. Pass the underlying function "
            "and give its bound arguments as call arguments."
        )
    try:
        return precompile_capture(fn, **kwargs)
    except PackageError as e:
        raise PrecompileError(str(e)) from e


def _artifact_paths(
    artifact_path: str | os.PathLike[str] | None,
    cache_path: str | os.PathLike[str] | None,
    *,
    who: str,
    neither: str,
) -> tuple[str | os.PathLike[str], str | os.PathLike[str]] | None:
    """Validate the on-disk form's path pair, or ``None`` if it was not requested.

    The two files only load as a matched pair -- the cache carries a sha256 of
    exactly the python_code bytes it was emitted with -- so accepting one path
    without the other would name half an artifact that can never be loaded.
    """
    if (artifact_path is None) != (cache_path is None):
        given, missing = (
            ("artifact_path", "cache_path")
            if artifact_path is not None
            else ("cache_path", "artifact_path")
        )
        raise ValueError(
            f"{who} got {given} without {missing}. The artifact and its cache "
            f"are a matched pair. {neither}"
        )
    if artifact_path is None or cache_path is None:
        return None
    if os.path.normcase(os.path.abspath(artifact_path)) == os.path.normcase(
        os.path.abspath(cache_path)
    ):
        raise ValueError(
            f"{who} got the same file for artifact_path and cache_path "
            f"({os.fspath(artifact_path)!r}); the two halves are separate files."
        )
    return artifact_path, cache_path


def _write_artifact(
    artifact_path: str | os.PathLike[str],
    cache_path: str | os.PathLike[str],
    python_code: str,
    cache: bytes,
) -> None:
    """Write the matched (python_code, cache) pair, creating parent directories.

    Both halves are written beside their targets and renamed into place, rather
    than truncated where they lie. The pair only loads together -- the cache
    carries a sha256 of exactly the python_code it was emitted with -- so a
    process that dies mid-write would otherwise leave a new artifact paired with
    the previous cache, which refuses to load. An accumulating capture rewrites
    on every call and its whole promise is that the files on disk are always a
    working artifact, so at hundreds of megabytes that window is the failure it
    is meant to protect against. Two renames are not one atomic step, so the
    previous source is hard-linked to a backup first and put back if the second
    rename fails; the named source path therefore always holds the previous or
    the new artifact, and only a crash in the gap between the two renames leaves
    a mismatched pair, which load refuses on the cache's sha256 rather than
    serving stale code. The containing directory is fsync'd after, so a rename that returned
    is durable.
    """
    written = []
    try:
        for path, payload in ((artifact_path, python_code), (cache_path, cache)):
            parent = os.path.dirname(os.fspath(path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            # A unique name per writer: two captures targeting one path must
            # not share a scratch file, or one renames the other's half-written
            # bytes into place. Beside the target, so the rename stays on one
            # filesystem. A plain open rather than mkstemp: mkstemp creates the
            # file 0600 and the rename carries that mode onto the artifact,
            # which nobody else on a shared directory can then read; open()
            # honours the umask.
            tmp = f"{os.fspath(path)}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            written.append((tmp, path))
            mode, encoding = (
                ("wb", None) if isinstance(payload, bytes) else ("w", "utf-8")
            )
            with open(tmp, mode, encoding=encoding) as f:
                f.write(payload)  # type: ignore[arg-type]
                f.flush()
                os.fsync(f.fileno())
    except BaseException:
        for tmp, _ in written:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise
    (artifact_tmp, _), (cache_tmp, _) = written
    backup = f"{os.fspath(artifact_path)}.{os.getpid()}.{uuid.uuid4().hex}.bak"
    previous = None
    installed = False
    try:
        # A hard link, not a move: the named path must resolve to the previous
        # or the new source at every instant, for a reader racing this write
        # and for a crash between the two renames below.
        try:
            os.link(artifact_path, backup)
            previous = backup
        except FileNotFoundError:
            pass
        except OSError:
            # No hard links on this filesystem: fall back to moving aside.
            os.replace(artifact_path, backup)
            previous = backup
        os.replace(artifact_tmp, artifact_path)
        installed = True
        os.replace(cache_tmp, cache_path)
    except BaseException:
        # Put the previous source back (or remove the new one on a first
        # write), best effort, so the named files stay a loadable artifact;
        # then drop every temp and re-raise.
        if previous is not None:
            undo = [(previous, artifact_path)]
        elif installed:
            undo = [(artifact_path, artifact_tmp)]
        else:
            undo = []
        for src, dst in undo:
            try:
                os.replace(src, dst)
            except OSError:
                pass
        for tmp, _ in written:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise
    if previous is not None:
        try:
            os.unlink(previous)
        except OSError:
            pass
    parents = {os.path.dirname(os.fspath(path)) or "." for _, path in written}
    # Durably record the renames: without an fsync of the containing directory
    # a crash just after os.replace returns can still lose the new directory
    # entry and resurrect the previous artifact.
    for parent in parents:
        try:
            fd = os.open(parent, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _read_artifact(
    artifact_path: str | os.PathLike[str],
    cache_path: str | os.PathLike[str],
) -> tuple[str, bytes]:
    """Read back a pair written by :func:`_write_artifact`."""
    with open(artifact_path, encoding="utf-8") as f:
        python_code = f.read()
    with open(cache_path, "rb") as f:
        cache = f.read()
    return python_code, cache


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


def _runnable_from_pair(
    python_code: str,
    cache: bytes,
    *,
    fn: Callable[..., object] | None = None,
    _trusted: bool = False,
) -> PrecompiledRunnable:
    """Reconstruct a runnable from an in-memory ``(python_code, cache)`` pair.

    The shared core of :func:`load` (which reads the pair off disk first) and the
    capture-time self-load in :class:`_MakeFxCapture`. ``_trusted`` is set only for
    that self-load, where the source was just produced in-process, to suppress the
    exec warning.
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
    # TRACER is absent on artifacts predating the dynamo tracer, so treat its absence
    # as make_fx (matching _parse_artifact_metadata and the cache-envelope default);
    # this keeps the pairing check below correct for older make_fx python_code.
    tracer = cast(str, meta.get("TRACER", "make_fx"))

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
                    "came from different precompile captures."
                )
            # A tracer tag was added alongside the dynamo tracer; treat its absence as
            # make_fx so an older make_fx cache still pairs with its python_code. A
            # differing tag means a wrong (code, cache) pairing, so hard-fail.
            if blob.get("tracer", "make_fx") != tracer:
                raise PrecompileError(
                    f"cache tracer {blob.get('tracer', 'make_fx')!r} does not match "
                    f"the python_code tracer {tracer!r}; the cache and python_code "
                    "came from different precompile captures."
                )
            # Reject a cache whose code_hash does not match this python_code (a
            # mismatched pairing); see Note [precompile programming model], invariant 7.
            expected_code_hash = hashlib.sha256(python_code.encode()).hexdigest()
            if blob.get("code_hash") != expected_code_hash:
                raise PrecompileError(
                    "cache does not match python_code (its code_hash "
                    f"{blob.get('code_hash')!r} != sha256(python_code) "
                    f"{expected_code_hash!r}); the cache and python_code came from "
                    "different precompile captures. Pair each cache with the "
                    "python_code from the same precompile capture."
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
        try:
            torch.compiler.load_cache_artifacts(artifact)
        except Exception as e:
            log.warning(
                "torch.compiler.precompile.load could not prime the cache from the "
                "artifact bundle (%s: %s); it is likely stale or from a different "
                "torch build. Falling back to JIT from python_code.",
                type(e).__name__,
                e,
            )
    # Run the driver inlined in python_code. It carries the full calling convention and
    # runtime safety checks (subclass wrap/unwrap, param/buffer lifting, grad harvest,
    # input/model validation) and JITs the kernels -- which hit the primed cache when
    # the bundle above loaded, so the "cache" path is exec-with-warm-kernels rather than
    # a separate runtime.
    forward = _make_inlined_forward(python_code, warn=not _trusted)

    if meta.get("SERVING_MODE") == "installed":
        # The exec above only built the handle; entering or calling it installs.
        if not isinstance(forward, _InstalledArtifact):
            raise PrecompileError(
                "python_code declares SERVING_MODE='installed' but its driver "
                "did not produce an installable handle; the artifact is "
                "malformed."
            )
        if fn is not None:
            forward._rebind(fn)
        forward._prepare(cast(str, meta["_PACKAGE"]))
        return PrecompiledCallable(forward)
    if fn is not None:
        raise PrecompileError(
            "fn= applies only to an artifact with SERVING_MODE='installed'; a "
            "standalone artifact carries its own entry and takes the captured "
            "arguments directly."
        )
    return PrecompiledModule._from_loaded(forward, backend=backend)


def load(
    artifact_path: str | os.PathLike[str],
    cache_path: str | os.PathLike[str],
    *,
    fn: Callable[..., object] | None = None,
) -> PrecompiledRunnable:
    """Reconstruct a runnable from the two files a precompile capture wrote.

    .. warning::

        This is a prototype API. Its signature, error types and artifact
        format may change between releases without a deprecation cycle.

    Name the two files :func:`capture` or :func:`accumulate` wrote -- the
    ``python_code`` artifact and its ``cache``. They load only as a matched pair
    (the cache carries a sha256 of exactly the python_code bytes it was emitted
    with).

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

    The returned object comes in two shapes, decided by the CAPTURE, not by a
    load-time choice. A dynamo artifact with captured frames the entry bytecode
    cannot reach on its own -- e.g. a graph break inside a child module's frame
    -- serves by INSTALLING onto the captured code objects: the returned callable
    mutates process state on first call (or on ``__enter__``) and supports
    ``with`` / ``unload()`` to take that back out. An artifact whose frames are
    all reachable from the entry -- including one that graph-broke or recompiled
    only within the entry frame -- is standalone: it installs nothing, and its
    ``with`` / ``unload()`` are no-ops. Both shapes are a
    :class:`torch.compiler.PrecompiledRunnable`, so a caller can enter and unload
    every loaded artifact uniformly; ``installed`` (``True`` for the installing
    shape, ``False`` for standalone) tells them apart.
    ``fn=`` applies to the
    installing shape only -- pass the function object to install onto when it is
    not importable from where it was captured (defined in ``__main__``, a
    notebook, or a REPL); it must be passed before the first call, and a
    standalone artifact rejects it with ``PrecompileError``.

    Raises ``PrecompileError`` if ``python_code`` is malformed or is not a
    ``torch.compiler.precompile`` artifact (it fails to parse, or is missing the
    calling-convention metadata), if the cache's ``backend`` or ``tracer`` tag does
    not match ``python_code``, or if the cache's ``code_hash`` does not match
    ``sha256(python_code)`` -- i.e. the cache and python_code came from different
    precompile captures. A cache whose ``format``/``version`` does not match (a
    foreign or different-build envelope) is NOT fatal: the cache is acceleration
    only, so ``load`` degrades to JIT'ing from ``python_code`` rather than crashing.
    """
    torch._C._log_api_usage_once("torch.compiler.precompile.load")
    python_code, cache = _read_artifact(artifact_path, cache_path)
    return _runnable_from_pair(python_code, cache, fn=fn)
