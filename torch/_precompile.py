"""Ahead-of-time precompilation: the internals behind the public
``torch.compiler.precompile`` module.

``torch/compiler/precompile.py`` re-exports the public types defined here --
``Capture``, ``DynamoTracer``, ``MakeFxTracer``, ``PrecompiledRunnable`` -- beside
``PrecompileSummary``, and the two caller-driven entry points defined here:
``capture``, which writes the pair from the calls the caller makes, and ``load``,
which reconstructs a runnable from it. Either front-end writes the same pair: a
self-contained, executable ``python_code`` string plus a companion
integrity-tagged ``cache``. ``_runnable_from_pair`` is the loader core that turns
a pair back into a runnable.

``DynamoTracer`` (the default) drives a ``PrecompileSession``
(``torch/_dynamo/precompile_package.py``) across the caller's calls, and the
multi-graph frame records and driver emitter at the end of this module render
every frame Dynamo compiled: the entry, its graph-break continuations and its
recompiled variants. ``MakeFxTracer`` drives ``PrecompiledModule``, a NON-STRICT
make_fx trace of one execution of ``fn``: with ``backend="inductor"`` (the
default) the captured graph is lowered through the AOT backend contract
(``torch._functorch.aot_autograd.compile_to_python``, AOTAutograd + Inductor), and
``python_code`` JIT-compiles kernels on first call while the cache primes them so a
warm reload skips JIT; with ``backend="eager"`` ``python_code`` inlines the captured
graph and runs on its own.

The full contract, the calling convention, and the cache / code_hash design all live in
Note [precompile programming model] below; every public entry point and guard references
it.
"""

# Note [precompile programming model]
#
# ``fn`` is the WHOLE computation, e.g. a module-level ``def step(model, x): return
# model(x)`` for inference or ``def step(model, x, t): loss_fn(model(x),
# t).backward()`` for a training step (only a MakeFxTracer capture also takes the
# ``lambda`` forms).
# Among the positional args, the nn.Module arguments have their parameters and
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
#    over tensor values, ``.item()``, and shape-dependent branching are resolved at
#    trace time and baked. Shapes are static BY DEFAULT (capture uses make_fx in its
#    "real" mode, so each size is baked as a constant).
#    Capture also constrains the example INPUTS: a NESTED tensor (either layout) is
#    refused on BOTH capture paths, as a restriction rather than a claim that it cannot be
#    fakeified -- nothing downstream of the trace has a nested representation (the recorded
#    dense shape/dtype/device the driver checks against is None for one). That refusal
#    applies to a traceable wrapper subclass's INNER tensors too, since a wrapper reports
#    non-nested whatever it wraps.
#    Make such a value a plain dense tensor or a supported subclass (e.g. DTensor).
#    An UNBACKED capture traces under a fake mode IT built, so THAT path -- and only that
#    path -- also refuses to run inside another trace: an ambient fake mode (a
#    torch.compile / export / AOTAutograd trace, or an enclosing ``with FakeTensorMode()``)
#    outranks the mode capture built, and no enclosing-trace mode passes
#    ``allow_fallback_kernels=False``, so a meta-less op in an allowlisted namespace would
#    be run for real again. A mode built under DEFAULT config (an AOTAutograd / inductor
#    one) lacks the data-ptr snapshot as well, so a ``.data_ptr()`` read would bake 0
#    rather than raise; a torch.compile / export mode does build under that patch, so for
#    those two only the fallback setting is lost. A STATIC capture has no mode of its own to
#    lose (it traces on the real example tensors), so the refusal does not apply to it: with
#    ``backend="eager"`` it captures inside another trace as it does outside one, while the
#    inductor lowering itself refuses a foreign fake mode ("Mixing fake modes NYI"), a
#    limitation that predates this note. Call precompile outside the enclosing trace.
#
#    You can opt specific user-input dims into being dynamic by marking them with
#    ``torch._dynamo.decorators.mark_unbacked`` before calling: those dims are
#    captured as UNBACKED symints (symbolic capture), which CANNOT be guarded on -- so
#    the artifact is valid for any runtime size of those dims, and a graph that needs to
#    guard on / specialize a marked dim fails LOUDLY at capture (PrecompileError) instead
#    of baking a silently-wrong result. Dims that MUST be equal at runtime (e.g. two
#    inputs combined by a broadcast that requires equal sizes, ``model(a) + model(b)``)
#    MUST share a ``shape_id``: marking them INDEPENDENTLY bakes a SILENT equal-size
#    assumption, and a runtime mismatch does not raise the loud failure eager gives. The
#    capture ShapeEnv records the equality as a deferred runtime assert (e.g.
#    ``Eq(u0, u1)``), but precompile does not yet harvest those relational asserts --
#    only mark_unbacked's min/max feed the runtime bound checks -- so a shared
#    ``shape_id`` is the way to get the check today.
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
# python_code you JIT -- same results either way. The eager backend has no kernels
# to accelerate, so the eager cache carries no compiled artifact (artifact=None) but is still a full integrity-tagged envelope, and load()
# always runs the graph inlined in python_code. The metadata
# lives in one place (python_code); the envelope carries a code_hash (sha256 of
# python_code) alongside the format/version + backend tag, so load() rejects a
# (python_code, cache) pair that did not come from the same precompile capture.
# Custom ops: an artifact whose graph calls them first imports the modules recorded in
# CUSTOM_OP_MODULES (a Dynamo artifact after its version checks). A failed import only
# warns, unless the module is in CUSTOM_OP_OVERRIDE_MODULES (its kernel overrode another
# for the same key at capture, or it called register_kernel on another module's
# custom_op) or, for a Dynamo artifact, in SERVED_MODULES (it serves
# the artifact's frames); then the load fails. The load then refuses unless every op in
# CUSTOM_OPS and every kernel key CUSTOM_OP_KERNELS lists for it is registered, and
# every overriding kernel is still the active one, registered from the same file or
# by the same recorded calling module. make_fx case: before load, the caller must
# import C++ extension ops, ops and kernels registered from __main__ or outside a
# module's import, and kernels registered through another module's Library. Dynamo
# case: if an op or kernel has no recorded module or a recorded module is outside
# torch, the artifact also imports its served frames' modules, all in the order
# their imports finished, so the caller must import only those registered outside
# every served frame's module.
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
# THREADING: the inductor lowering drives process-global compiler state and is
# serialized by _COMPILE_LOCK (torch/_functorch/_aot_autograd/to_standalone_python.py),
# so concurrent backend="inductor" captures lower one at a time. The make_fx trace
# and the backend="eager" path are NOT serialized.
#
# tracer: the capture front-end, orthogonal to backend. MakeFxTracer is the non-strict
# trace everything above describes (the invariants, the contract). DynamoTracer (the
# default) analyzes the Python instead of specializing to one traced path: it records
# every frame Dynamo compiles while the caller's calls run -- the entry, the graph-break
# continuations, the recompiled variants -- with one guard tree per variant, and the
# artifact dispatches among them at call time. Its artifact is standalone (it installs
# nothing) unless a frame is reachable only by an ordinary call, in which case it is
# installed (see load()); either way it is locked to the producing Python version and
# torch build, since it inlines serialized bytecode and pickled guard state. Invariants 1, 2, 3 and 6 above
# are the make_fx contract; under Dynamo a call outside the captured variants is
# REFUSED by the guards rather than silently served, except in a frame Dynamo gave up
# on during the capture (PrecompileSummary.eager_fallbacks), which runs that call
# eagerly as the capture did; and the guards that could not be serialized are
# reported (PrecompileSummary.dropped_guards) rather than checked.

from __future__ import annotations

import ast
import base64
import contextlib
import dataclasses
import errno
import functools
import hashlib
import inspect
import io
import logging
import operator
import os
import pickle
import re
import stat
import sys
import threading
import types
import uuid
from collections.abc import (
    Callable,  # noqa: TC003
    Iterable,  # noqa: TC003
    Sequence,  # noqa: TC003
)
from types import MappingProxyType
from typing import (
    Any,
    cast,
    Generic,
    Literal,
    NamedTuple,
    NewType,
    ParamSpec,
    TYPE_CHECKING,
    TypeVar,
)

import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._guards import detect_fake_mode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


log = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing_extensions import Self

    from torch._dynamo.precompile_package import PrecompileSession
    from torch._functorch._aot_autograd.codegen import PySourceBuilder
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.compiler._precompile_types import PrecompileSummary


# The public surface (``capture``, ``load`` and the types they take and return) is
# re-exported by torch/compiler/precompile.py, and ``PrecompileError`` by
# torch/compiler/__init__.py; both re-home ``__module__`` there. This private module
# therefore exports nothing, so test_public_bindings sees one public location.
__all__: list[str] = []


# Integrity tag baked into the cache envelope and verified by load() (with the
# code_hash) to reject a foreign / mismatched cache; see Note [precompile programming
# model], invariant 7.
_CACHE_FORMAT = "torch.compiler.precompile"
_CACHE_VERSION = 1

# What PrecompiledModule._compile pins over a make_fx capture's options (see there).
_MAKE_FX_ASSERT_PINS = {"size_asserts": True, "scalar_asserts": True}


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
    effectful op, a nested example input, which capture does not support on either path
    (invariant 3), an UNBACKED capture attempted inside another trace (invariant 3), a
    non-tensor output the inductor backend cannot lower, or a runtime input whose shape or
    memory format differs from the example (invariants 3 and 6).
    See Note [precompile programming model] in this module for the full contract.
    """

    # Re-exported in torch.compiler.__all__, so pickle and test_public_bindings
    # resolve it there. Not in torch/compiler/precompile.py's re-homing loop, so its
    # string annotations resolve against torch.compiler: keep them to builtins.
    __module__ = "torch.compiler"


@dataclasses.dataclass(frozen=True)
class MakeFxTracer:
    """The ``make_fx`` capture front-end, passed as ``tracer=`` to a precompile capture.

    A NON-STRICT single make_fx trace: it records the ATen ops of ONE execution of
    ``fn``, so a ``capture`` with this tracer takes exactly one call and refuses a
    second, and control flow and shapes are specialized to that call (the source of
    the programming-model contract). Part of the prototype ``torch.compiler.precompile``
    API, so it may change without a deprecation cycle.

    The contract in brief (Note [precompile programming model] in
    ``torch/_precompile.py`` has the full list): every tensor ``fn`` reads is an
    argument or a registered parameter/buffer of an ``nn.Module`` argument, and
    shapes are static unless a dim is marked with ``mark_unbacked``
    (``torch._dynamo.decorators``, inductor backend only) before the call: a
    marked dim is captured as an unbacked symint, so one artifact serves any
    runtime size of it, and a graph that needs to guard on it fails at capture.
    Dims that must be equal at runtime must share a ``shape_id``: marked
    independently they bake a silent equal-size assumption, and a mismatch at
    runtime is not caught. Each input's dtype and device are specialized too (a
    runtime mismatch is rejected), the inductor backend additionally specializes
    on memory format, and a nested-tensor example input is refused.

    ``decompositions`` is an optional decomposition table (a dict mapping each
    ``OpOverload`` to a decomposition function) forwarded to ``make_fx`` as its
    ``decomposition_table``; it is specific to this tracer (Dynamo lowers through the
    backend instead).
    """

    decompositions: dict | None = None


@dataclasses.dataclass(frozen=True)
class DynamoTracer:
    """The ``dynamo`` capture front-end (the default), passed as ``tracer=`` to
    :func:`torch.compiler.precompile.capture`.

    An execution-driven multi-graph capture that analyzes the Python (bytecode)
    rather than tracing one path: it records graph-break continuations and every
    guarded recompilation the calls exercise, so a capture with this tracer takes
    as many calls as you make, keyword arguments included. Part of the prototype
    ``torch.compiler.precompile`` API, so it may change without a deprecation cycle.

    ``guard_filter_fn`` takes the ``GuardFilterEntry`` records (``guard_type``,
    ``name``, ...) and returns one keep flag per entry. It filters the runtime
    guards during capture as well as the serialized ones (the default drops only
    what cannot be serialized), so a dropped guard never triggers a recompile during
    capture; ``recompile_limit`` caps recompilations per frame, and a call past it
    runs eager and is absent from the artifact without any gate refusing it;
    ``dynamic`` forces dynamic shapes as ``torch.compile(dynamic=)`` does. The
    ``require_*`` gates refuse, at write time, an artifact with a coverage gap
    (``require_complete``: a bypassed or uncovered frame, a call that raised) or one
    whose dropped guards told captured variants apart or hang off a
    configuration-chosen slot (``require_no_risky_drops``). Other dropped guards,
    such as the identity guards every model drops because they cannot be serialized,
    are listed in the artifact rather than refused.
    """

    guard_filter_fn: Callable[[Sequence[Any]], Sequence[bool]] | None = None
    recompile_limit: int = 256
    dynamic: bool | None = None
    require_complete: bool = True
    require_no_risky_drops: bool = True


class PrecompiledRunnable:
    """What ``torch.compiler.precompile.load`` returns.

    A callable with the captured ``fn``'s calling convention that can also be
    entered as a context manager and unloaded. A standalone artifact installs
    nothing, so for it ``__enter__``/``__exit__``/:meth:`unload` are no-ops.
    Removing an installed artifact's entries is not implemented yet either, so
    they stay installed until the process exits: serve an installed artifact
    from a process you can throw away.
    Part of the prototype ``torch.compiler.precompile`` API, so it may change
    without a deprecation cycle.

    Attributes:
        installed: Whether loading this artifact installed onto the captured
            code objects (``SERVING_MODE = "installed"``); ``False`` for a
            standalone artifact.
    """

    installed: bool = False

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.unload()

    def unload(self) -> None:
        """Remove whatever this loaded artifact installed; currently a no-op in every mode."""


_P = ParamSpec("_P")
_R = TypeVar("_R")


class Capture(Generic[_P, _R]):
    r"""The caller-driven capture ``torch.compiler.precompile.capture`` returns.

    Part of the prototype ``torch.compiler.precompile`` API, so it may change
    without a deprecation cycle. Enter it as a context manager to arm the
    capture, call it exactly as you would ``fn`` inside the block -- each call
    runs for real, is folded into the capture, and returns that run's result
    -- and the artifact is written to the ``artifact_path`` / ``cache_path``
    files when the block exits. How many calls a capture takes is the tracer's:
    :class:`MakeFxTracer` takes exactly one. Call :meth:`save` inside the block
    to write what has been captured so far to those same files without ending
    the capture. A block that raises writes nothing it has not already saved,
    and a block that made no call raises ``PrecompileError`` on exit rather
    than writing an empty artifact. A capture is single-use: once its block
    exits it cannot be entered again, so call ``capture()`` again to retry.
    That includes a failed write at exit: the spent capture does not keep its
    pair, so fix the path and capture again.
    """

    def __init__(self) -> None:
        self._state: Literal["new", "active", "spent"] = "new"
        # A Dynamo capture's calls all feed one PrecompileSession: serialize
        # the block's calls, saves and exit across threads.
        self._lock = threading.RLock()

    def __enter__(self) -> Self:
        with self._lock:
            if self._state == "spent":
                raise PrecompileError(_SPENT_CAPTURE)
            if self._state == "active":
                raise PrecompileError(
                    "this capture has already been entered; capture() returns a "
                    "fresh capture per call."
                )
            self._start()
            self._state = "active"
        return self

    def __exit__(self, *exc: object) -> None:
        with self._lock:
            # The block is over either way: a later call must not run a trace
            # whose result nothing would write.
            self._state = "spent"
            self._finish(exc)

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        with self._lock:
            self._check_active("calling it, or nothing is written when the block exits")
            return self._run(*args, **kwargs)

    def save(self) -> None:
        """Write everything captured so far to the artifact files without ending the capture.

        Each call re-renders and rewrites both files, so a job that dies between
        saves leaves the last checkpoint loadable. A refusal (such as a
        :class:`DynamoTracer` ``require_*`` gate) or a failed write raises but
        writes nothing partial, and the capture stays open. Raises
        ``PrecompileError`` outside the capture's ``with`` block.
        """
        with self._lock:
            self._check_active("calling save()")
            self._save()

    def _check_active(self, before: str) -> None:
        if self._state == "spent":
            raise PrecompileError(_SPENT_CAPTURE)
        if self._state == "new":
            raise PrecompileError(
                f"capture is not active: enter it with a `with` block before {before}."
            )

    # The tracer's half, each run under the lock: arm on entry, run one call,
    # write what has been captured, and end the block with its exc_info.
    def _start(self) -> None:
        pass

    def _run(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        raise NotImplementedError

    def _save(self) -> None:
        raise NotImplementedError

    def _finish(self, exc: tuple[object, ...]) -> None:
        raise NotImplementedError


_SPENT_CAPTURE = (
    "capture is not active: its block has already exited, and a capture is "
    "single-use; call capture() again for a fresh one."
)


class _MakeFxCapture(Capture[_P, _R]):
    r"""Single-shot capture: the :class:`MakeFxTracer` front-end.

    A make_fx trace records the ATen ops of ONE execution of ``fn``, so this
    captures exactly one call and refuses a second -- there is no notion of
    guards or recompiled variants here, and thus nothing a further call could
    add. Pass ``tracer=DynamoTracer()`` to capture several calls with the graph
    breaks and recompilations between them.
    """

    def __init__(
        self,
        fn: Callable[_P, _R],
        artifact_path: str | os.PathLike[str],
        cache_path: str | os.PathLike[str],
        *,
        backend: str,
        decompositions: dict | None,
        inductor_options: dict[str, Any],
    ) -> None:
        super().__init__()
        self._module = PrecompiledModule(
            fn,
            backend=backend,
            tracer="make_fx",
            decompositions=decompositions,
            inductor_options=inductor_options,
        )
        self._artifact_path = artifact_path
        self._cache_path = cache_path
        self._rendered: tuple[str, bytes] | None = None
        self._called = False

    def _finish(self, exc: tuple[object, ...]) -> None:
        # Write only on a clean exit that captured a call; a block that raised,
        # or one that never called the capture, leaves the files untouched. The
        # pair is dropped either way, so the spent capture cannot write it again.
        try:
            if exc[0] is None:
                self._save()
        finally:
            self._rendered = None

    def _save(self) -> None:
        # A make_fx capture records a single call, so there is nothing further to
        # fold in; save() and block exit write the same files.
        if self._rendered is None and self._called:
            raise PrecompileError(
                "nothing was captured: the capture's call raised, so there is no "
                "pair to write."
            )
        if self._rendered is None:
            raise PrecompileError(
                "nothing was captured: call the capture with your example "
                "arguments inside the `with` block."
            )
        try:
            _write_artifact(self._artifact_path, self._cache_path, *self._rendered)
        except OSError as e:
            # _write_artifact restores the previous pair on a best-effort basis.
            raise PrecompileError(
                f"precompile could not write the artifact: {e}"
            ) from e

    def _run(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        if kwargs:
            raise ValueError(
                "MakeFxTracer takes positional arguments only; pass "
                "tracer=DynamoTracer() to capture calls with keyword arguments."
            )
        if self._rendered is not None:
            raise PrecompileError(
                "MakeFxTracer captures a single call and one has already been "
                "traced. Use tracer=DynamoTracer() to capture several calls, with "
                "the graph breaks and recompilations between them."
            )
        self._called = True
        # make_fx traces one execution of fn and lowers it to the artifact; we then
        # serve that artifact on the real args through the SAME load() path a caller
        # would take, so the value handed back is exactly what serving produces
        # (invariants checked, grads scattered onto the model) rather than a bare
        # trace with no result. Single-shot: a make_fx trace records one path, so a
        # second call has nothing new to add. Build python_code ONCE and thread it
        # into to_cache_bytes (the metadata + embedded kernel source is not rebuilt,
        # and code_hash is sha256 over exactly the bytes written on exit). The pair is
        # recorded only once the serve has returned: a serve that raised leaves the
        # capture retryable and nothing for exit or save() to write. The trace runs
        # with grad enabled and the drivers pin their own grad mode, so the caller's
        # ambient mode is not consulted.
        self._trace_without_side_effects(args)
        python_code = self._module.to_python_code()
        rendered = (python_code, self._module.to_cache_bytes(python_code))
        result = _runnable_from_pair(*rendered, _trusted=True)(*args)
        self._rendered = rendered
        return cast(_R, result)

    def _trace_without_side_effects(self, args: tuple[object, ...]) -> None:
        # A static trace runs fn on the real example tensors, so an in-place update in
        # fn (a BatchNorm's running stats, ``x.add_(1)``) would land once in the trace
        # and again in the serve that follows. Put the buffers and tensor inputs the
        # trace mutated back, so the serve is the one execution the caller sees
        # (``.grad`` gets the same treatment inside _capture). A parameter updated in
        # place, e.g. by an optimizer step inside fn, is refused rather than every
        # parameter being cloned up front.
        if _has_unbacked_marks(args):
            # An unbacked capture traces on fakes; only the serve touches the reals.
            self._module._compile(args)
            return
        mods = [a for a in args if isinstance(a, torch.nn.Module)]
        # A parameter also passed as a plain argument is traced as that user input,
        # so it is snapshotted with the inputs rather than left to the refusal below.
        leaves = [t for t in pytree.tree_leaves(args) if isinstance(t, torch.Tensor)]
        params = {id(p) for m in mods for p in m.parameters()} - {id(t) for t in leaves}
        tensors = [*(b for m in mods for b in m.buffers()), *leaves]
        saved = {id(t): (t, t.detach().clone()) for t in tensors if id(t) not in params}
        try:
            self._module._compile(args)
        finally:
            # Compared by value: under make_fx a kernel's in-place running-stat
            # update does not bump the buffer's version counter.
            with torch.no_grad():
                for t, value in saved.values():
                    if not torch.equal(t, value):
                        t.copy_(value)
        gm = self._module._gm
        if gm is None:
            raise AssertionError("_compile left no traced graph")
        if _writes_a_parameter(gm, len(self._module._param_names)):
            raise PrecompileError(
                "MakeFxTracer cannot capture a fn that updates a parameter in place "
                "(e.g. an optimizer step): serving the capture would apply the update "
                "a second time, and the trace may already have applied it once. Step "
                "the optimizer outside the captured fn."
            )


class _DynamoCapture(Capture[_P, _R]):
    r"""Multi-call capture: the :class:`DynamoTracer` front-end.

    Enter the ``with`` block, call it as many times as you need to exercise the
    graph breaks and recompiled variants you want captured; the artifact is
    rendered and written to disk when the block exits. Call :meth:`save` inside
    the block to checkpoint everything captured so far to the same files without
    ending the capture. Calls and saves are serialized under one lock, so a
    save() never snapshots the capture mid-compile.
    """

    def __init__(
        self,
        session: PrecompileSession[_P, _R],
        artifact_path: str | os.PathLike[str],
        cache_path: str | os.PathLike[str],
        *,
        require_complete: bool,
        require_no_risky_drops: bool,
    ) -> None:
        super().__init__()
        self._session = session
        self._artifact_path = artifact_path
        self._cache_path = cache_path
        self._gates = {
            "require_complete": require_complete,
            "require_no_risky_drops": require_no_risky_drops,
        }
        self._call: Callable[..., object] | None = None
        self._block = contextlib.ExitStack()
        self._in_call = False
        self._calls = 0
        # The call count the last save() wrote. -1, not 0, so a block that never
        # called the capture is still "dirty" at exit and raises the
        # nothing-captured error rather than writing an empty artifact.
        self._saved_calls = -1

    def _map(self, method: Callable[..., Any], *args: object, **kwargs: object) -> Any:
        from torch._dynamo.exc import PackageError, RecompileError

        try:
            return method(*args, **kwargs)
        except (PackageError, RecompileError) as e:
            raise PrecompileError(str(e)) from e

    def _start(self) -> None:
        from torch.compiler._cache import CacheArtifactManager
        from torch.compiler._runtime_cache import capture_scope

        # The capture's compiles record into the process-global cache-artifact
        # list, which the cache half of the artifact bundles to prime the
        # kernel caches on load. A fresh one so the bundle holds only this
        # capture's compiles; left in _finish. The swap spans the whole
        # block, so anything else compiled inside the block records here too,
        # as it does into the runtime scope whose dependencies _write freezes.
        # Captures open at once share that scope, so each also freezes the
        # kernels the others compiled.
        try:
            self._block.enter_context(capture_scope())
            self._block.enter_context(CacheArtifactManager.with_fresh_cache())
            self._call = self._map(self._session.__enter__)
        except BaseException:
            self._block.close()
            raise

    def _run(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        if self._call is None:
            raise AssertionError("an active capture has no session call")
        if self._in_call:
            raise PrecompileError(
                "this capture is being re-entered recursively: fn called the "
                "capture it is being captured through. Call fn itself from "
                "inside fn."
            )
        self._in_call = True
        # Counted before the call: a call that raised is for the session's
        # require_complete gate to report, not "nothing was captured".
        self._calls += 1
        try:
            return self._map(self._call, *args, **kwargs)
        finally:
            self._in_call = False

    def _write(self) -> None:
        from torch.compiler._runtime_cache import freeze_runtime_cache

        python_code = ""
        blob: dict[str, Any] = {}

        # Called by freeze_runtime_cache under its freeze lock.
        def snapshot() -> bytes | None:
            nonlocal python_code, blob
            python_code, cache = self._map(
                self._session.snapshot_artifact, **self._gates
            )
            blob = torch.load(io.BytesIO(cache), weights_only=True)
            if blob["backend"] != "eager" and blob["artifact"] is None:
                raise RuntimeError(
                    "precompile.capture requires the cache artifact of a compiled "
                    "capture, and this capture saved none"
                )
            return blob["artifact"]

        def write(artifact: bytes | None) -> None:
            buf = io.BytesIO()
            torch.save({**blob, "artifact": artifact}, buf)
            try:
                _write_artifact(
                    self._artifact_path, self._cache_path, python_code, buf.getvalue()
                )
            except OSError as e:
                # Only the on-disk rewrite failed; _write_artifact leaves
                # whatever pair was already at these paths in place.
                raise PrecompileError(
                    f"precompile could not write the artifact: {e}"
                ) from e

        try:
            freeze_runtime_cache(snapshot, write)
        except PrecompileError:
            raise
        except Exception as e:
            raise PrecompileError(
                "precompile could not freeze the runtime dependencies of "
                f"cache_path={os.fspath(self._cache_path)!r}: {e}"
            ) from e
        self._saved_calls = self._calls

    def _save(self) -> None:
        if self._in_call:
            raise PrecompileError(
                "save() was called from inside fn while the capture is "
                "running it; save after the call returns."
            )
        if self._calls == 0:
            raise PrecompileError(
                "nothing was captured: call the capture with your example "
                "arguments before calling save()."
            )
        self._write()

    def _finish(self, exc: tuple[object, ...]) -> None:
        error: BaseException | None = None
        try:
            # A clean block with calls the last save() did not cover writes a
            # final checkpoint while the compiles are still recorded; a block
            # that raised, or whose last save() covered every call, writes
            # nothing here. A gate refusal is held and re-raised after
            # teardown so summary() stays readable.
            if exc[0] is None and self._calls > self._saved_calls:
                try:
                    if self._calls == 0:
                        raise PrecompileError(
                            "nothing was captured: call the capture with your "
                            "example arguments inside the `with` block."
                        )
                    self._write()
                except BaseException as e:
                    error = e
            try:
                self._map(self._session.__exit__, *exc)
            except BaseException as e:
                if error is None:
                    raise
                raise e from error
        finally:
            self._block.close()
        if exc[0] is None and error is not None:
            raise error

    def summary(self) -> PrecompileSummary:
        r"""Coverage, recompilation, failure and guard information for everything
        captured so far.
        """
        with self._lock:
            return self._map(self._session.summary)


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


def _aliased_input(node: torch.fx.Node) -> object:
    """The argument a view, alias or in-place result aliases (the ``out`` of an
    ``out=`` overload, the list behind a ``split`` piece), or None."""
    if node.target is operator.getitem:
        return node.args[0]
    schema = getattr(node.target, "_schema", None)
    if schema is None or not schema.returns:
        return None
    ret = schema.returns[0].alias_info
    if ret is None:
        return None
    for i, arg in enumerate(schema.arguments):
        # A list-of-views return (split, unbind) keeps its alias set on the elements.
        if arg.alias_info is not None and (
            not ret.before_set or arg.alias_info.before_set & ret.before_set
        ):
            return node.args[i] if i < len(node.args) else node.kwargs.get(arg.name)
    return None


def _writes_a_parameter(gm: torch.fx.GraphModule, num_params: int) -> bool:
    """Whether a traced op writes into one of the first ``num_params`` placeholders
    (the lifted parameters), directly or through a view or alias of one such as
    ``p.data`` or a ``chunk`` piece. Read off the graph because neither the version
    counter nor a view's own counter records a write through ``.data``."""
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    param_nodes = set(placeholders[:num_params])
    for node in gm.graph.nodes:
        schema = getattr(node.target, "_schema", None)
        if schema is None:
            continue
        for i, arg in enumerate(schema.arguments):
            if arg.alias_info is None or not arg.alias_info.is_write:
                continue
            value = node.args[i] if i < len(node.args) else node.kwargs.get(arg.name)
            # A Tensor(a!)[] argument (_foreach_add_, _fused_adam_) writes each element.
            for base in value if isinstance(value, (list, tuple)) else [value]:
                while isinstance(base, torch.fx.Node) and base not in param_nodes:
                    base = _aliased_input(base)
                if base in param_nodes:
                    return True
    return False


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
    """The capture-time error for a guard on a mark_unbacked dim, shared by the capture paths.

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
    import torch._functorch.config as functorch_config
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    # FakeTensorMode SNAPSHOTS this config at construction, and that snapshot is what
    # every fake tensor it makes consults, so the patch must wrap the CONSTRUCTION (as
    # make_fx does around its own mode): with it off, a .data_ptr() read in fn raises
    # instead of returning a meaningless value. allow_fallback_kernels=False keeps a
    # meta-less op in an allowlisted namespace (aten, prims, quantized, ...) from having
    # its real kernel run on zero-filled substitutes and whatever shape that produced
    # baked.
    with functorch_config.patch(fake_tensor_allow_unsafe_data_ptr_access=False):
        fake_mode = FakeTensorMode(
            shape_env=shape_env,
            allow_non_fake_inputs=True,
            allow_fallback_kernels=False,
        )
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


def _reject_unfakeifiable_input(label: str, a: Tensor) -> None:
    """Refuse an example tensor capture cannot trace on.

    ``label`` is one of ``_capture``'s ``input_labels`` ("parameter w", "buffer nt", "user
    input 0"), so each refusal names what the caller has to change. A NESTED input is
    refused on BOTH capture paths, and worded as a restriction rather than as a claim about
    either fake mode: the unbacked path's ShapeEnv could mint the symbolic nested int a
    jagged tensor's ragged dim needs (a static capture has none, so fakeifying one there
    dies on a raw internal assertion), but nothing downstream of the trace has a nested
    representation either way -- the recorded dense shape/dtype/device the driver checks
    against is None for one. Running ahead of every shape read also gets the STRIDED
    layout, whose ``t.shape`` read raises inside NestedTensorImpl, this same named refusal.
    """
    if a.is_nested:
        raise PrecompileError(
            f"precompile: example {label} is a nested tensor ({a.layout} layout), "
            "which capture does not support. Make it a plain dense tensor (or a "
            "supported subclass) -- on the model for a parameter/buffer, at the call "
            "site for a user input."
        )
    # The metadata read above sees the OUTER tensor only, and a traceable wrapper subclass
    # reports non-nested whatever it wraps, so recurse into its inner tensors.
    # Unwrapping rather than refusing wrappers wholesale keeps DTensor working, and it is
    # the same flattening the fake conversion performs.
    if is_traceable_wrapper_subclass(a):
        attrs, _ = a.__tensor_flatten__()
        for name in attrs:
            inner = getattr(a, name)
            # Not every listed attribute is a tensor: DTensor lists its device_mesh.
            if isinstance(inner, torch.Tensor):
                _reject_unfakeifiable_input(
                    f"{label} (inner tensor {name!r} of a {type(a).__name__} subclass)",
                    inner,
                )


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
    # Labels for the per-input refusal below, aligned with flat_args. Each names its KIND,
    # because the model half is not an argument the caller can swap out: a refusal saying
    # "buffer nt" sends them to the model, "user input 0" to the call site.
    input_labels = [
        *(f"parameter {n}" for n in param_names),
        *(f"buffer {n}" for n in buffer_names),
        *(f"user input {i}" for i in range(len(user_flat))),
    ]
    # Example inputs capture cannot trace on are refused capture-wide, on BOTH paths and
    # before either traces. This runs ahead of EVERY shape read below -- the param/buffer
    # records and the user-input _dense_shape records alike -- which is what gets a STRIDED
    # nested parameter, buffer or user input the same named refusal rather than the raw
    # "NestedTensorImpl doesn't support sizes" that reading t.shape raises.
    for label, a in zip(input_labels, flat_args):
        if isinstance(a, torch.Tensor):
            _reject_unfakeifiable_input(label, a)
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
    # An UNBACKED capture traces under a fake mode IT built, and an ambient one outranks
    # that, carrying neither of the two settings _fakeify_with_unbacked gives it (invariant
    # 3 in the Note has the details), so refuse rather than trace under someone else's
    # contract. The STATIC path has no mode to lose: make_fx's "real" mode resolves none. Ask
    # detect_fake_mode -- what make_fx itself resolves through -- so both sources it sees
    # without arguments (an ambient TracingContext, the dispatch-mode stack) are refused by
    # name here instead of reaching its own mode-mismatch assertion once capture enters its
    # mode; a fake tensor handed in as an example input, its third source, is not covered.
    if any(marks) and detect_fake_mode() is not None:
        raise PrecompileError(
            "precompile: unbacked capture cannot run inside another trace -- a "
            "FakeTensorMode is already active (e.g. precompile called from inside "
            "torch.compile or an export/AOTAutograd trace). make_fx would adopt that mode "
            "and its ShapeEnv, so capture's own safety settings would not apply. Capture "
            "outside the enclosing trace."
        )
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
#     import runpy
#     ns = runpy.run_path("this_file.py")
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
#   0. Imports of the modules registering the custom ops the graph calls, and a check
#      that those ops are registered (only when the graph calls custom ops).
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
    buf.writeline(f"USER_INPUT_SHAPES = {compiled._user_input_shapes!r}")
    buf.writeline(f"USER_INPUT_DTYPES = {compiled._user_input_dtypes!r}")
    buf.writeline(f"USER_INPUT_DEVICES = {compiled._user_input_devices!r}")
    # Per user-input-leaf mark_unbacked min/max bounds: None for a leaf with no bounded
    # marked dim, else {dim: (lo, hi)} (either may be None). The drivers reject a
    # runtime size outside the declared range (invariant 3); see the inlined drivers.
    buf.writeline(f"USER_INPUT_BOUNDS = {compiled._user_input_bounds!r}")
    buf.writeline("")


def _parse_artifact_metadata(python_code: str) -> dict[str, object]:
    """Read the calling-convention constants back out of ``python_code`` WITHOUT
    executing it (exec'ing the inlined Inductor output would JIT the kernels, the
    very work the cache exists to skip).

    python_code is the single source of truth: the metadata builders emit the constants
    below as top-level literal assignments, so an AST walk + literal_eval recovers them
    safely. The cache then only needs to carry the compiled artifact.

    The required set follows TRACER: absent (artifacts predating the dynamo tracer) or
    anything but "dynamo" means the make_fx set the inlined driver reads. A dynamo
    artifact instead carries the multi-graph driver's blobs (_FRAMES, _BACKENDS and
    _ENTRY_BINDING, or _PACKAGE and UNREACHABLE_WITHOUT_INSTALL once SERVING_MODE is
    "installed"), the readable frame report beside them and the two versions that
    lock them, _DYNAMO_PYTHON_VERSION for the marshalled bytecode and TORCH_VERSION
    for the pickled guard state.
    """
    try:
        tree = ast.parse(python_code)
    except SyntaxError as e:
        raise PrecompileError(
            "python_code is not valid Python; it does not look like a "
            "torch.compiler.precompile artifact."
        ) from e
    # The last top-level assignment to a name wins, as it would under exec, so the
    # set selection below and the reported values read the same assignment.
    assigns: dict[str, ast.expr] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            assigns[node.targets[0].id] = node.value

    def literal(name: str) -> object:
        try:
            return ast.literal_eval(assigns[name])
        except (ValueError, TypeError) as e:
            raise PrecompileError(
                f"python_code {name!r} calling-convention metadata is malformed; "
                "it must be a Python literal."
            ) from e

    tracer = literal("TRACER") if "TRACER" in assigns else None
    if tracer == "dynamo":
        wanted = {
            "BACKEND",
            "FN_NAME",
            "FRAMES",
            "DROPPED_GUARDS",
            "RISKY_DROPPED_GUARDS",
            "WONT_GENERALIZE",
            "_FRAMES",
            "_BACKENDS",
            "_DYNAMO_PYTHON_VERSION",
            "_ENTRY_BINDING",
            "TORCH_VERSION",
        }
        mode = literal("SERVING_MODE") if "SERVING_MODE" in assigns else None
        if mode == "installed":
            wanted -= {"_FRAMES", "_BACKENDS", "_ENTRY_BINDING"}
            wanted |= {"_PACKAGE", "UNREACHABLE_WITHOUT_INSTALL"}
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
    # Parsed when present but never required, so older artifacts load unchanged:
    # TRACER and SERVING_MODE already selected the required set above, and
    # the guard-audit sections come back as data.
    optional = {
        "TRACER",
        "SERVING_MODE",
        "_ALLOW_SERVE_TIME_COMPILES",
        "CUDAGRAPH_CONFIG",
        "INDUCTOR_OPTIONS",
        "POLICY_DROPPED_GUARDS",
        "DROPPED_GUARD_CODE",
        "CUSTOM_OP_MODULES",
        "CUSTOM_OPS",
        "CUSTOM_OP_KERNELS",
        "CUSTOM_OP_OVERRIDE_MODULES",
        "SERVED_MODULES",
        "ENTRY_MODULE",
    }
    found = {name: literal(name) for name in assigns if name in wanted | optional}
    for name in assigns.keys() - wanted - optional - {"forward"}:
        # Not a metadata name we consume (``forward = ...`` is the multi-graph
        # driver's own binding, emitted later in this stack). Skipped by design,
        # but log it at debug so a malformed / renamed artifact is diagnosable
        # rather than silently dropped.
        log.debug(
            "precompile: ignoring unrecognized top-level assignment %r while "
            "parsing artifact calling-convention metadata",
            name,
        )
    missing = wanted - found.keys()
    if missing:
        raise PrecompileError(
            f"python_code is missing calling-convention metadata {sorted(missing)}; "
            "it does not look like a torch.compiler.precompile artifact."
        )
    # Artifacts predating the installed serving mode carry no SERVING_MODE, and
    # they were all standalone.
    found.setdefault("SERVING_MODE", "standalone")
    if found["BACKEND"] not in ("inductor", "eager"):
        raise PrecompileError(
            f"python_code names an unknown backend {found['BACKEND']!r}; a "
            "torch.compiler.precompile artifact is 'inductor' or 'eager'."
        )
    return found


def _overload_names(op: object) -> set[tuple[str, str]]:
    """The ``(ns::op, overload)`` names of ``op`` if it is an operator: its own,
    or every overload's for an ``OpOverloadPacket`` but its TorchScript-only ones."""
    if isinstance(op, torch._ops.OpOverloadPacket):
        ops = [getattr(op, name) for name in op.overloads()]
        ops = [o for o in ops if torch._C._dispatch_has_kernel(o._name)]
    else:
        ops = [op]
    return {
        (o._schema.name, o._schema.overload_name)
        for o in ops
        if isinstance(o, torch._ops.OpOverload)
    }


def _op_names(gm: torch.fx.GraphModule) -> set[tuple[str, str]]:
    """The ``(ns::op, overload)`` names of the operators ``gm`` and its subgraphs
    call, directly or passed to a higher-order op."""
    names: set[tuple[str, str]] = set()
    for module in gm.modules():
        graph = getattr(module, "graph", None)
        if not isinstance(graph, torch.fx.Graph):
            continue
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            for op in (node.target, *pytree.tree_leaves((node.args, node.kwargs))):
                names |= _overload_names(op)
    return names


def _called_op_names(python_code: str) -> set[tuple[str, str]]:
    """The ``(ns::op, overload)`` names of the ``torch.ops.<ns>.<op>.<overload>``
    calls in Inductor's generated code. Unlike the captured graph, this omits ops
    Inductor decomposed or lowered and includes ops a post-grad pass inserted."""
    names: set[tuple[str, str]] = set()
    for node in ast.walk(ast.parse(python_code)):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name) and node.id == "torch" and len(parts) == 4:
            overload, op, ns, ops = parts
            if ops == "ops":
                names.add((f"{ns}::{op}", "" if overload == "default" else overload))
    return names


def _importable_module(name: str) -> bool:
    """Whether the loaded module ``name`` can be imported by name in another
    process. Under ``python -m``, ``__main__`` (and a spawn worker's
    ``__mp_main__``) carries the spec of the module it ran; running a directory
    or zipapp names that spec ``__main__``."""
    spec = getattr(sys.modules.get(name), "__spec__", None)
    return name != "__main__" and spec is not None and spec.name == name


def _custom_op_module(op_name: str, overload: str) -> str | None:
    """The importable module that registers ``op_name``, if one is known: the one
    defining the function ``torch.library.custom_op`` wrapped, or the one a C++ op
    names with ``m.set_python_module``."""
    from torch._library.custom_ops import OPDEFS

    qualname = f"{op_name}.{overload}" if overload else op_name
    if (opdef := OPDEFS.get(qualname)) is not None:
        name = getattr(opdef._init_fn, "__module__", None)
    else:
        pystub = torch._C._dispatch_pystub(op_name, overload)
        name = pystub[0] if pystub is not None else None
    return name if name is not None and _importable_module(name) else None


def _registration_site(op_name: str, overload: str) -> str:
    """The ``file:line`` that defined ``op_name``, as recorded at build time ("" if
    none)."""
    try:
        debug = torch._C._dispatch_find_schema_or_throw(op_name, overload).debug()
    except RuntimeError:
        return ""
    return debug.removeprefix("registered at ")


def _registration_path(op_name: str, overload: str) -> str:
    """The file that defined ``op_name``, as recorded at build time ("" if none)."""
    return _registration_site(op_name, overload).rpartition(":")[0]


@functools.cache
def _torch_source_root() -> str | None:
    """The source root torch's C++ op registrations recorded at build time (the CI
    checkout in a wheel), from a generated and a hand-written aten schema."""
    paths = [
        _registration_path("aten::add", "Tensor"),
        _registration_path("aten::quantized_lstm", "input"),
    ]
    try:
        root = os.path.commonpath(paths) if all(paths) else None
    except ValueError:
        return None
    # Paths sharing only a filesystem root (an out-of-tree build) match any path.
    return root if root and root != os.path.dirname(root) else None


def _torch_cpp_op(op_name: str, overload: str) -> bool:
    """Whether ``import torch`` registers ``op_name`` from C++: its namespace is one
    of torch's, and it is not an extension's that adds ops to that namespace from a
    C++ source outside torch's root. Paths alone cannot decide: an extension built in
    a checkout, or under a directory the root shares, lies inside the root. One that
    is built in torch's checkout and adds ops to a torch namespace is taken for
    torch's."""
    if op_name.split("::")[0] not in _TORCH_OP_NAMESPACES:
        return False
    root = _torch_source_root()
    path = _registration_path(op_name, overload)
    if path.endswith(".py"):
        # A torch helper's Library records itself for any caller.
        torch_file = path.startswith(os.path.join(_TORCH_DIR, ""))
        return torch_file and not path.startswith(_TORCH_OP_HELPERS)
    if root is None or not path:
        return True
    path = os.path.normcase(os.path.normpath(path))
    return path.startswith(os.path.normcase(os.path.join(root, "")))


# Torch's namespaces defined from C++.
_TORCH_OP_NAMESPACES = frozenset(
    "aten prim inductor inductor_prims profiler quantized _quantized quantization"  # noqa: SIM905
    " sparse c10d _c10d_functional _c10d_functional_autograd _dtensor symm_mem fsdp"
    " mkl mkldnn mkldnn_prepacked onednn static_runtime xnnpack prepacked vulkan"
    " vulkan_prepack metal metal_prepack vulkan_quantized _nnapi _native_aot".split()
)
_TORCH_DIR = os.path.dirname(torch.__file__)
# torch's helpers that build a Library for the caller's namespace record themselves;
# torch.library.define and impl record their caller in REGISTRATION_CALLERS instead.
_TORCH_OP_HELPERS = tuple(
    os.path.join(_TORCH_DIR, name) for name in ("library.py", "_library", "_custom_op")
)
# The modules defining prims with torch._prims._make_prim, which records itself.
_TORCH_PRIM_MODULES = (
    "torch._prims",
    "torch._inductor.inductor_prims",
    "torch._dynamo.tensor_version_op",
)
# Torch ops defined only when a function runs, not by a module import, and what
# must run on the serving host before the artifact loads.
_CUDA_INIT = "CUDA initialization (torch.cuda.init()) with the triton package installed"
_TORCH_LAZY_OPS = MappingProxyType(
    {
        "_inductor_debug::check_memory_step": "torch._inductor.runtime.debug_utils.register_check_mem_op()",
        "semi_structured::cusparselt_mm": "torch.sparse.semi_structured._ensure_cusparselt_mm_registered()",
        "semi_structured::cutlass_mm": "torch.sparse.semi_structured._ensure_cutlass_mm_registered()",
    }
)
# The op names come from a process-global counter, so only the same call sequence
# reproduces them.
_TORCH_LAZY_OP_NAMESPACES = MappingProxyType(
    {"_native": "a torch._native register_op_override call"}
)


def _file_modules() -> dict[str, str]:
    """The importable modules in ``sys.modules`` by the real path of their file."""
    modules: dict[str, str] = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if isinstance(path, str) and _importable_module(name):
            modules.setdefault(os.path.realpath(path), name)
    return modules


def _library_op_module(
    op_name: str, overload: str, file_modules: Callable[[], dict[str, str]]
) -> str | None:
    """The module whose import registers ``op_name``, if a loaded module defines it:
    the one creating the ``torch.library.Library`` that defines it, or whose body
    calls ``torch.library.define``. ``file_modules`` is ``_file_modules``."""
    from torch._library.utils import REGISTRATION_CALLERS

    path = _registration_path(op_name, overload)
    if path.startswith(_TORCH_OP_HELPERS):
        qualname = f"{op_name}.{overload}" if overload else op_name
        module, site = REGISTRATION_CALLERS.get((qualname, ""), ("__main__", ""))
        defined = site == _registration_site(op_name, overload)
        return module if defined and _importable_module(module) else None
    if not path.endswith(".py"):
        return None
    return file_modules().get(os.path.realpath(path))


def _kernel_paths(qualname: str) -> dict[str, str]:
    """The files registering the active kernels of ``qualname``, by key."""
    pattern = r"^(\w+)(?:\[alias\])?: registered at (.+):\d+ ::"
    return dict(re.findall(pattern, torch._C._dispatch_dump(qualname), re.MULTILINE))


def _kernel_modules(
    qualname: str,
    paths: dict[str, str],
    callers: dict[str, str],
    keys: tuple[str, ...],
    file_modules: Callable[[], dict[str, str]],
    shadowing: bool = False,
) -> dict[str, str]:
    """The importable modules outside torch registering the active kernel of
    ``qualname`` for each of ``keys``, by key: the one whose body called
    ``torch.library.impl`` or ``register_kernel`` for it, else the one that
    constructed its ``torch.library.Library``. Often not the module defining the op,
    and only for the keys where that kernel replaced another if ``shadowing``.
    ``paths`` is ``_kernel_paths(qualname)`` and ``callers`` is ``kernel_callers``'s.
    A kernel registered through another module's Library records that module, not
    the one calling ``impl``."""
    if shadowing:
        # A replaced kernel's line reads "CPU (inactive): registered at ...".
        inactive = r"^(\w+)(?:\[alias\])? \(inactive\):"
        replaced = re.findall(inactive, torch._C._dispatch_dump(qualname), re.MULTILINE)
        keys = tuple(k for k in keys if k in replaced)
    modules = {
        k: callers.get(k) or file_modules().get(os.path.realpath(p))
        for k, p in paths.items()
        if k in keys and p.endswith(".py")
    }
    return {
        k: m
        for k, m in modules.items()
        if m and m.partition(".")[0] != "torch" and _importable_module(m)
    }


def _op_registration(
    op_name: str, overload: str, file_modules: Callable[[], dict[str, str]]
) -> tuple[Literal["lazy", "module", "torch", "unrecorded"], str]:
    """What registers ``op_name`` on a fresh host: ``("lazy", setup)`` for a torch op
    defined only once ``setup`` runs, ``("module", name)`` for one an importable
    module registers, ``("torch", "")`` for one ``import torch`` registers from C++,
    and ``("unrecorded", "")`` otherwise."""
    ns, _, name = op_name.partition("::")
    if op_name in _TORCH_LAZY_OPS:
        return "lazy", _TORCH_LAZY_OPS[op_name]
    if ns == "triton" and any(k == name for k, _ in torch._TritonLibrary.ops_table):
        return "lazy", _CUDA_INIT
    if ns in _TORCH_LAZY_OP_NAMESPACES:
        return "lazy", _TORCH_LAZY_OP_NAMESPACES[ns]
    module = None
    if ns == "prims":
        # torch._prims._make_prim defines every prim from its own function.
        op = getattr(getattr(torch.ops.prims, name), overload or "default")
        modules = (sys.modules.get(m) for m in _TORCH_PRIM_MODULES)
        owners = (m for m in modules if m and any(v is op for v in vars(m).values()))
        module = next((m.__name__ for m in owners), None)
    if module is None:
        module = _custom_op_module(op_name, overload) or _library_op_module(
            op_name, overload, file_modules
        )
        # A prim another module makes with _make_prim is registered from torch._prims.
        if ns == "prims" and module in _TORCH_PRIM_MODULES:
            module = None
    if module is not None:
        return "module", module
    return ("torch" if _torch_cpp_op(op_name, overload) else "unrecorded"), ""


class _CustomOpModules(NamedTuple):
    """What registers a capture's custom ops, from ``_custom_op_modules``."""

    # The modules registering the ops, in the order their imports finished, which
    # load replays with the modules of overriding kernels last.
    modules: tuple[str, ...]
    # The custom ops no importable module is known to register.
    unrecorded: tuple[tuple[str, str], ...]
    # The ops torch defines only at runtime, by the setup that defines them.
    lazy: tuple[tuple[str, tuple[str, ...]], ...]
    # The custom ops with a kernel for the keys no importable module is known to
    # register (a torch helper's, a C++ extension's or __main__'s).
    unrecorded_kernels: tuple[tuple[str, str], ...] = ()
    # The ``(op, key)`` of those kernels registered from __main__, by a C++
    # extension, or by a torch.library helper (register_kernel, impl) called from
    # __main__ or a module that is not importable.
    untracked_kernels: tuple[tuple[str, str], ...] = ()


def _custom_op_modules(
    op_keys: Mapping[tuple[str, str], tuple[str, ...]],
) -> _CustomOpModules:
    """What registers the ops in ``op_keys`` and their active kernels for the op's
    keys; pass it to ``_warn_unregistered_ops``."""
    from torch._library.utils import kernel_callers

    has_kernel = torch._C._dispatch_has_kernel_for_dispatch_key
    modules: set[str] = set()
    unrecorded: set[tuple[str, str]] = set()
    unrecorded_kernels: set[tuple[str, str]] = set()
    untracked_kernels: set[tuple[str, str]] = set()
    torch_dir = os.path.join(os.path.realpath(_TORCH_DIR), "")
    lazy: dict[str, set[str]] = {}
    file_modules = functools.cache(_file_modules)
    for (op_name, overload), keys in op_keys.items():
        kind, value = _op_registration(op_name, overload, file_modules)
        if kind == "lazy":
            lazy.setdefault(value, set()).add(op_name)
            continue
        qualname = f"{op_name}.{overload}" if overload else op_name
        paths = _kernel_paths(qualname)
        callers = kernel_callers(qualname)
        kernels = _kernel_modules(qualname, paths, callers, keys, file_modules)
        modules.update(kernels.values())
        if kind == "module":
            modules.add(value)
        elif kind == "unrecorded":
            unrecorded.add((op_name, overload))
        # torch registers the kernels of its own ops.
        if kind == "torch" or value.partition(".")[0] == "torch":
            continue
        for k in keys:
            if k in kernels or not has_kernel(qualname, k):
                continue
            caller = callers.get(k)
            if k not in paths:
                unrecorded_kernels.add((op_name, overload))
            elif kind == "module" and not paths[k].endswith(".py"):
                # Taken as registered by the C++ extension the op's module loads.
                unrecorded_kernels.add((op_name, overload))
            elif not os.path.realpath(paths[k]).startswith(torch_dir):
                untracked_kernels.add((qualname, k))
                unrecorded_kernels.add((op_name, overload))
            elif caller is None:
                # Through a Library torch made, as _scoped_library's, and not by
                # the last register_kernel or torch.library.impl call recorded.
                unrecorded_kernels.add((op_name, overload))
            elif caller.partition(".")[0] != "torch":
                untracked_kernels.add((qualname, k))
                unrecorded_kernels.add((op_name, overload))
    lazy_ops = tuple((s, tuple(sorted(ops))) for s, ops in sorted(lazy.items()))
    ordered = tuple(m for m in sys.modules if m in modules)
    return _CustomOpModules(
        ordered,
        tuple(sorted(unrecorded)),
        lazy_ops,
        tuple(sorted(unrecorded_kernels)),
        tuple(sorted(untracked_kernels)),
    )


def _warn_unregistered_ops(
    found: _CustomOpModules, fallback_modules: Sequence[str] = ()
) -> None:
    """Warns about the ops ``found`` says torch defines only at runtime, or that
    no module registers or registers a kernel of, naming the ``fallback_modules``
    the artifact imports in case they register the latter."""
    for setup, ops in found.lazy:
        log.warning(
            "torch.compiler.precompile: the captured graphs call %s, which torch "
            "defines only inside %s, so serving this artifact is unsupported: it "
            "fails to load on a host where that has not run.",
            ", ".join(ops),
            setup,
        )
    names = sorted({f"{n}.{o}" if o else n for n, o in found.unrecorded})
    kernels = [f"{n} ({k})" for n, k in found.untracked_kernels if n not in names]
    if not names and not kernels:
        return
    serves = (
        f"The artifact imports {', '.join(fallback_modules)}, the modules its "
        "frames run in, as a best-effort fallback, and serves only where those or "
        "something else register them before it is loaded."
        if fallback_modules
        else "The artifact serves only where something else registers them before "
        "it is loaded."
    )
    if names:
        log.warning(
            "torch.compiler.precompile: the captured graphs call %s, which no "
            "importable module is known to register: defined in __main__ or outside "
            "a module's import, with torch._custom_ops.custom_op, or a C++ op "
            "without set_python_module. %s "
            "Define a Python op with torch.library.custom_op in an importable "
            "module, and give a C++ op m.set_python_module; for a library's ops, "
            "import that library before loading the artifact.",
            ", ".join(names),
            serves,
        )
    if kernels:
        log.warning(
            "torch.compiler.precompile: the captured graphs run the kernels %s, "
            "which no recorded module is known to register: registered from "
            "__main__ or a module that is not importable, or by a C++ extension. "
            "%s Register the kernel in an importable module, or load the extension "
            "in the op's module, before loading the artifact.",
            ", ".join(kernels),
            serves,
        )


def _custom_ops(op_names: Iterable[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    """The ops among ``op_names`` outside aten, which the artifact checks are
    registered once its imports ran. Prims are kept: inductor's and dynamo's are
    registered only once their modules are imported."""
    return tuple(sorted(n for n in op_names if not n[0].startswith("aten::")))


_COMPOSITE_KERNEL_KEYS = (
    "CompositeImplicitAutograd",
    "CompositeExplicitAutograd",
    "CompositeExplicitAutogradNonFunctional",
)
# The dispatch keys whose kernels an artifact checks are still registered at load,
# when its graph's tensors are unknown.
_KERNEL_KEYS = ("CPU", "CUDA", "XPU", "MPS", "PrivateUse1", *_COMPOSITE_KERNEL_KEYS)


def _graph_kernel_keys(
    gm: torch.fx.GraphModule, example_inputs: Sequence[object]
) -> tuple[str, ...]:
    """The dispatch keys an artifact of ``gm`` runs kernels under: the composite
    keys and the backend keys of its tensors and ``example_inputs`` (``SparseCPU``
    for a sparse CPU one). A real-mode trace leaves no ``meta["val"]`` for a sparse
    tensor, so one the graph creates adds no key."""
    # A tensor's backend key is its highest key below BackendSelect.
    below = torch._C._dispatch_keyset_full_after(torch._C.DispatchKey.BackendSelect)
    vals = [
        node.meta.get("val")
        for module in gm.modules()
        if isinstance(graph := getattr(module, "graph", None), torch.fx.Graph)
        for node in graph.nodes
    ]
    backends = {
        (torch._C._dispatch_keys(leaf) & below).highestPriorityTypeId().name
        for leaf in pytree.tree_leaves((vals, list(example_inputs)))
        if isinstance(leaf, torch.Tensor)
    }
    # Many per-backend keys, e.g. QuantizedMeta, have names the dispatcher cannot
    # parse back, so no kernel can be checked under them.
    parseable = {k for k in backends if torch._C._parse_dispatch_key(k) is not None}
    return (*sorted(parseable - {"Undefined"}), *_COMPOSITE_KERNEL_KEYS)


def _device_kernel_keys(device_types: Iterable[str]) -> tuple[str, ...]:
    """The composite keys and the backend keys of ``device_types``."""
    backends = {torch._C._dispatch_key_for_device(t) for t in device_types}
    return (*sorted(backends), *_COMPOSITE_KERNEL_KEYS)


def _write_custom_op_imports(
    buf: PySourceBuilder,
    modules: tuple[str, ...],
    ops: tuple[tuple[str, str], ...],
    keys: Mapping[tuple[str, str], tuple[str, ...]],
    call: bool = True,
) -> None:
    # A compiled graph calls torch.ops.<ns>.<op> directly, so nothing on the serving
    # path imports the module that registers the op. The recorded modules are the
    # ones that constructed each Library used for the op's define and impls, or whose
    # body called torch.library.define, impl or register_kernel. That misses a module
    # calling impl on another module's Library (and an op defined from __main__ has
    # none), so the imports are best effort and the ops and their kernels are checked.
    if not ops:
        return
    buf.writeline("# " + "=" * 70)
    buf.writeline("# 0. The modules registering the custom ops the graphs call")
    buf.writeline("# " + "=" * 70)
    buf.writeline(f"CUSTOM_OP_MODULES = {modules!r}")
    buf.writeline(f"CUSTOM_OPS = {ops!r}")
    # A kernel is often registered by a module other than the op's own, so the
    # schema alone does not show the op can run.
    has_kernel = torch._C._dispatch_has_kernel_for_dispatch_key
    kernels = {}
    file_modules = functools.cache(_file_modules)
    from torch._library import custom_ops
    from torch._library.utils import kernel_callers

    custom_ops_file = os.path.realpath(custom_ops.__file__)
    overrides = {}
    for name, overload in ops:
        qualname = f"{name}.{overload}" if overload else name
        # Despite its name, _dispatch_has_kernel only tests that the op exists,
        # which _dispatch_has_kernel_for_dispatch_key requires (it raises otherwise).
        op_exists = torch._C._dispatch_has_kernel(qualname)
        op_keys = keys[name, overload]
        if op_exists and (
            found := tuple(k for k in op_keys if has_kernel(qualname, k))
        ):
            kernels[qualname] = found
            paths = _kernel_paths(qualname)
            # register_kernel from another module replaces only the kernel the
            # custom_op's own dispatcher kernel calls, so its caller is recorded.
            callers = kernel_callers(qualname)
            own = _custom_op_module(name, overload)
            owners = {
                k: c
                for k in found
                if (c := callers.get(k)) is not None
                and c != own
                and c.partition(".")[0] != "torch"
                and _importable_module(c)
                and os.path.realpath(paths.get(k, "")) == custom_ops_file
            }
            owners |= _kernel_modules(
                qualname, paths, callers, op_keys, file_modules, True
            )
            if owners:
                overrides[qualname] = owners
    buf.writeline(f"CUSTOM_OP_KERNELS = {kernels!r}")
    # By op and key, the module registering a kernel the graph ran over another one
    # for the same key, or a custom_op's kernel through register_kernel from a module
    # other than the op's. The replaced kernel passes the check above, and would run
    # in its place if the module failed to import, registered it only when called,
    # or was imported before the replaced kernel's module.
    buf.writeline(f"CUSTOM_OP_OVERRIDE_MODULES = {overrides!r}")
    from torch import _precompile_driver as driver

    buf.writeline(inspect.getsource(driver._import_custom_op_modules).rstrip())
    if call:
        buf.writeline("_import_custom_op_modules()")
        buf.writeline("del _import_custom_op_modules")
    buf.writeline("")


def _build_python_source(
    compiled: PrecompiledModule,
    graph_python: str,
) -> str:
    from torch._functorch._aot_autograd.codegen import PySourceBuilder

    buf = PySourceBuilder()
    buf.writeline(_GENERATED_HEADER)
    buf.writeline("")
    op_names = _called_op_names(graph_python)
    if ops := _custom_ops(op_names):
        keys = dict.fromkeys(ops, compiled._kernel_keys)
        found = _custom_op_modules(keys)
        _warn_unregistered_ops(found)
        _write_custom_op_imports(buf, found.modules, ops, keys)
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
#     import runpy
#     ns = runpy.run_path("this_file.py")
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
    op_names = _op_names(gm)
    if ops := _custom_ops(op_names):
        keys = dict.fromkeys(ops, compiled._kernel_keys)
        found = _custom_op_modules(keys)
        _warn_unregistered_ops(found)
        _write_custom_op_imports(buf, found.modules, ops, keys)
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
# guarded variants. Where the single-graph forms inline one graph, this carries every
# frame Dynamo compiled -- the entry frame plus each continuation -- with one guard tree
# per captured variant (how it serves them is said below):
#
#     import runpy
#     ns = runpy.run_path("this_file.py")
#     out = ns["forward"](model, my_input)      # same args as the captured callable
#
# Sections below are labelled. What is OPAQUE is base64 of pickled Dynamo state,
# which has no readable source form here. Everything else is meant to be read and
# reviewed.
"""


# What the artifact does to the process, which differs by serving mode.
_SERVING_NOTES = {
    "standalone": """\
# This artifact dispatches among the captured variants itself at call time; the opaque
# sections hold the guard trees, the transformed bytecode and the compiled subgraphs.
# Nothing is installed onto your code objects and no frame evaluator is involved; the
# only global state loading touches is the captured module, where the names Dynamo
# minted while tracing (import aliases, the builtins dict key, the compiled subgraphs
# under their backend ids) are re-seeded and stay. The flip side is that there is no
# compiler behind it: a call no captured variant covers RAISES rather than compiling a
# new one, except in a frame Dynamo gave up on during capture (marked "eager
# fallback" in FRAMES below), which runs it eagerly.
""",
    "installed": """\
# This artifact SERVES BY INSTALLING the Dynamo package in _PACKAGE onto the live code
# objects of the captured modules, so loading it mutates global state: a frame the
# entry reaches by an ordinary call (a graph break inside a child module's forward)
# can only be served through Dynamo's frame evaluator. Its calls refuse to compile,
# as under the fail_on_recompile stance (but without setting that process-wide
# stance), so a call no captured variant covers raises rather than compiling a new
# graph, under every process-wide stance that would compile it (force_backend,
# "eager_then_compile", ...) too; a frame Dynamo gave up on during capture (marked
# "eager fallback" in FRAMES below) runs such a call eagerly instead. Two stances you set
# yourself do change serving:
# "force_eager" runs every call eagerly, bypassing the installed entries, and
# "eager_on_recompile" serves the captured variants and runs any other call eagerly.
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
    intercepts, so it has no place in a source artifact.

    The entry is ``codes[0]``: CompilePackage records the captured callable's
    code first and reads it back by position (``main, *codes``). A bypassed code
    keeps its record, flagged and without variants, the way install() keeps its
    global binding but none of its guarded codes: a continuation the frame ahead
    of it names must stay bound, and a bypassed code's guarded codes are dead.
    A code that is neither bypassed nor has variants is ``trivial``: it ran as
    plain Python during capture, and the driver rebuilds it as plain Python when
    a resume name reaches it. An unnamed trivial code (a callback the
    continuation called) gets a refusal stub that nothing binds.
    """
    return [
        {
            "is_entry": i == 0,
            "bypassed": code.bypassed,
            # No variant and not bypassed: Dynamo skipped the frame (no tensor
            # reached it) or traced no ops in it, e.g. the continuation after a
            # trailing .backward(). Either way it ran eager during capture, so
            # the driver runs it eager too.
            "trivial": not code.guarded_codes and not code.bypassed,
            # Dynamo gave up on the frame during capture and skipped it from
            # then on; a load still serves its variants and runs a miss eager.
            "eager_fallback": code.eager_fallback and not code.bypassed,
            "code": code.python_code,
            "python_module": code.python_module,
            "import_sources": dict(code.import_sources),
            "resume_names": list(code.function_names) if code.install_to_global else [],
            "variants": [
                {
                    "guards_state": guarded.guards_state,
                    "dynamo_code": guarded.dynamo_code,
                }
                for guarded in ([] if code.bypassed else code.guarded_codes)
            ],
        }
        for i, code in enumerate(entry.codes)
    ]


def _opens_scope(frame: dict[str, Any]) -> bool:
    """Whether ``_build_multigraph_forward`` opens ``frame``'s module for its globals.
    That driver inlines this test, as it imports nothing from torch._precompile."""
    return bool(frame["variants"] or (frame["trivial"] and frame["resume_names"]))


def _served_modules(frames: list[dict[str, Any]]) -> set[str]:
    """The modules of the entry frame and of the frames ``_build_multigraph_forward``
    opens a scope for, which that driver imports. Never a dead record's. The
    installed driver's ``CompilePackage.install`` also binds the modules of the
    codes this leaves out, which its pre-check does not cover."""
    return {f["python_module"] for f in frames if f["is_entry"] or _opens_scope(f)}


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
        # co_names of every variant's bytecode, nested code objects included.
        # co_consts is acyclic (a code object cannot contain itself).
        out: set[str] = set()
        stack = [
            SerializedCode.to_code_object(variant["dynamo_code"])
            for variant in frame["variants"]
        ]
        while stack:
            code = stack.pop()
            out.update(code.co_names)
            stack.extend(c for c in code.co_consts if isinstance(c, types.CodeType))
        return out

    reachable = {i for i, frame in enumerate(frames) if frame["is_entry"]}
    named: set[str] = set()
    pending = list(reachable)
    while pending:
        named |= named_globals(frames[pending.pop()])
        grew = {
            i
            for i, frame in enumerate(frames)
            if i not in reachable and any(n in named for n in frame["resume_names"])
        }
        reachable |= grew
        pending.extend(grew)
    return reachable


def _unreachable_frames(frames: list[dict[str, Any]]) -> list[str]:
    """The compiled frames the entry cannot reach through a continuation, by fqn.

    The standalone dispatcher would run such a frame eager, silently giving up
    its compiled variant, so any of them makes the capture serve by installing.
    A reachable frame with no variant keeps the capture standalone: the driver
    runs a trivial one eager, as capture did, and refuses a bypassed one when a
    call reaches it.
    """
    reachable = _reachable_frames(frames)
    return sorted(
        f"{frame['python_module']}.{frame['code'].co_name}"
        for i, frame in enumerate(frames)
        if frame["variants"] and i not in reachable
    )


def _reject_uninstallable_entry(frames: list[dict[str, Any]], entry: Any) -> None:
    """Refuse a capture whose entry the artifact could not rebuild or serve.

    The multi-graph driver rebuilds the entry from its code object, which
    carries no closure, so a closure entry cannot be built at all. That is a
    capture-time fact and is refused here rather than at load on the serving
    machine. (Defaults, which the code object does not carry either, are
    recorded into the artifact rather than refused.) An entry with no
    dispatchable variant is refused too, naming which of its causes applies:
    Dynamo bypassed it, Dynamo gave up on it at its first call, or it produced
    no guarded code (a thin wrapper).
    """
    from torch._dynamo.package import SerializedCode
    from torch._dynamo.precompile_package import _variantless_entry_fallback_message

    entry_frame = next((f for f in frames if f["is_entry"]), None)
    if entry_frame is None:
        return
    if entry_frame["bypassed"]:
        # Dynamo logged why when it bypassed the frame; the thin-wrapper advice
        # below would send the caller restructuring code that was never the
        # problem.
        raise PrecompileError(
            f"precompile captured no dispatchable graph for {entry.fn_name!r}: the "
            f"entry frame was BYPASSED during capture (its guards could not be "
            f"serialized, or a backend artifact was missing when the package was "
            f"saved), so no variant of it can be served. Dynamo logged the reason: "
            f"look for the package-bypass warning, or the precompile_cache_bypass / "
            f"dynamo_cache_bypass artifacts under tlparse. Fix that rather than "
            f"restructuring the captured callable."
        )
    if entry_frame["eager_fallback"] and not entry_frame["variants"]:
        name = entry_frame["code"].co_name
        raise PrecompileError(_variantless_entry_fallback_message(name))
    if not entry_frame["variants"]:
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
            f"CALLS the model, e.g. a module-level def step(m, x): return m(x), "
            f"with precompile.capture(step, ...) and calling cap(model, x)."
        )
    code = SerializedCode.to_code_object(entry_frame["code"])
    if code.co_freevars:
        raise PrecompileError(
            f"precompile cannot build a self-contained artifact for {entry.fn_name!r}: "
            f"it closes over {list(code.co_freevars)!r}, and this capture has to rebuild "
            f"the entry from its code object, which cannot restore a closure. Capture a "
            f"module-level function that takes what it needs as arguments, e.g. "
            f"precompile.capture(step, ...), calling cap(model, x), with "
            f"'def step(model, x): return model(x)'."
        )


def _entry_binding(fn: object) -> dict[str, Any]:
    """The default arguments an entry's code object does not carry.

    The artifact rebuilds its entry from that entry's code object, which holds
    no default arguments. Without them a defaulted parameter is simply absent at
    the served call -- which the guard check then cannot bind, so every variant
    misses. Closure cells are not carried: an entry that closes over free
    variables is refused before we reach here (a rebuilt cell is a new object
    that Dynamo's identity guard would miss), so a valid entry has none, and
    the driver refuses one that arrives anyway rather than rebuilding it over
    empty cells.
    """
    return {
        "defaults": getattr(fn, "__defaults__", None),
        "kwdefaults": getattr(fn, "__kwdefaults__", None),
    }


def _build_multigraph_python_source(
    entry: Any,
    frames: list[dict[str, Any]],
    backends: Mapping[str, Any],
    summary: PrecompileSummary,
    backend: str,
    entry_binding: dict[str, Any] | None,
    mode: str,
    unreachable: list[str],
    custom_op_modules: tuple[str, ...],
    custom_ops: tuple[tuple[str, str], ...],
    custom_op_keys: Mapping[tuple[str, str], tuple[str, ...]],
    cudagraph_config: dict[str, bool],
    inductor_options: Mapping[str, object] | None,
) -> str:
    """Render a multi-graph capture as ``python_code``.

    An ``"installed"`` artifact carries the whole Dynamo package and serves by
    installing it (see _build_installed_forward); ``unreachable`` names the
    frames that made it so. The readable half -- what was captured, which
    frames, how many variants, what guards were dropped -- is emitted as literals
    so a reviewer can diff it. The guard trees, the transformed bytecode and the
    compiled subgraphs have no source form here, so they go into the clearly
    bannered opaque blobs the inlined driver rebuilds from.
    """
    from torch._functorch._aot_autograd.codegen import PySourceBuilder

    buf = PySourceBuilder()
    buf.writeline(_MULTIGRAPH_GENERATED_HEADER)
    buf.writeline(_SERVING_NOTES[mode])
    buf.writeline("")
    # The driver imports them after its version checks.
    _write_custom_op_imports(
        buf, custom_op_modules, custom_ops, custom_op_keys, call=False
    )
    buf.writeline("# " + "=" * 70)
    buf.writeline("# 1. What was captured (readable)")
    buf.writeline("# " + "=" * 70)
    buf.writeline(f"BACKEND = {backend!r}")
    buf.writeline('TRACER = "dynamo"')
    buf.writeline(f'SERVING_MODE = "{mode}"')
    if mode == "installed":
        # load(allow_serve_time_compiles=True) flips this to True before exec.
        buf.writeline("_ALLOW_SERVE_TIME_COMPILES = False")
        # What a serve-time compile is configured with: the capture's options, or
        # None when they have no literal form (a custom pass), which load() refuses.
        options = dict(inductor_options or {})
        try:
            recordable = ast.literal_eval(repr(options)) == options
        except (ValueError, SyntaxError):
            recordable = False
        buf.writeline(f"INDUCTOR_OPTIONS = {options if recordable else None!r}")
    buf.writeline(f"FN_NAME = {entry.fn_name!r}")
    buf.writeline(f"ENTRY_MODULE = {entry.codes[0].python_module!r}")
    buf.writeline(
        "# The inductor CUDA graph config the capture compiled under. Load wraps"
    )
    buf.writeline("# the subgraphs by it, not by the loading process's config.")
    buf.writeline(f"CUDAGRAPH_CONFIG = {cudagraph_config!r}")
    buf.writeline(f"_DYNAMO_PYTHON_VERSION = {tuple(sys.version_info[:2])!r}")
    buf.writeline(f"TORCH_VERSION = {torch.__version__!r}")
    buf.writeline("# Frame modules: section 0 imports those CUSTOM_OP_MODULES lists,")
    buf.writeline("# probes the rest; the driver (installed: install()) imports them.")
    buf.writeline(f"SERVED_MODULES = {tuple(sorted(_served_modules(frames)))!r}")
    buf.writeline("")
    buf.writeline(
        "# frame name -> number of captured variants. A frame with one variant"
    )
    buf.writeline("# serves one specialization; the artifact covers no other, except")
    buf.writeline("# that an eager fallback runs a call its variants miss eagerly.")
    buf.writeline("FRAMES = [")
    for frame in frames:
        row = f"    ({frame['code'].co_name!r}, {len(frame['variants'])}),"
        buf.writeline(row + ("  # eager fallback" if frame["eager_fallback"] else ""))
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
    buf.writeline(
        "# Kept by the filter but discarded by the invariance policy; likewise not"
    )
    buf.writeline("# checked at serve time.")
    buf.writeline(
        f"POLICY_DROPPED_GUARDS = {[list(g) for g in summary.policy_dropped_guards]!r}"
    )
    buf.writeline("")
    buf.writeline("# What a dropped slot above actually checked, where it renders one.")
    buf.writeline(
        f"DROPPED_GUARD_CODE = {[list(g) for g in summary.dropped_guard_code]!r}"
    )
    buf.writeline("")
    buf.writeline(
        "# Values pinned to exactly what capture saw; any other value misses."
    )
    buf.writeline(f"WONT_GENERALIZE = {tuple(summary.wont_generalize)!r}")
    buf.writeline("")
    if mode == "installed":
        buf.writeline("# Frames entered by an ordinary call; served by installing.")
        buf.writeline(f"UNREACHABLE_WITHOUT_INSTALL = {unreachable!r}")
        buf.writeline("")
        buf.writeline("# " + "=" * 70)
        buf.writeline("# 2. Dynamo package and compiled subgraphs -- OPAQUE")
        buf.writeline("#")
        buf.writeline(
            "# base64(pickle) of the Dynamo cache entry (guards, transformed bytecode)"
        )
        buf.writeline("# and the backend artifacts it calls by id.")
        buf.writeline("# " + "=" * 70)
        package = {"dynamo": entry, "backends": dict(backends)}
        buf.writeline(f"_PACKAGE = {_b64(package)!r}")
        buf.writeline("")
        buf.writeline("# " + "=" * 70)
        buf.writeline("# 3. Driver: install the package and serve (readable)")
        buf.writeline("# " + "=" * 70)
        buf.writeline(_emit_multigraph_driver_source("_build_installed_forward"))
        return buf.getvalue()
    buf.writeline(
        "# The entry's default arguments: a code object carries none, and the"
    )
    buf.writeline("# driver rebuilds the entry from one.")
    try:
        binding_blob = _b64(entry_binding)
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
    buf.writeline("# " + "=" * 70)
    buf.writeline("# 3. Compiled subgraphs -- OPAQUE")
    buf.writeline("#")
    buf.writeline(
        "# base64(pickle) of the backend artifacts the frames call by name: the"
    )
    buf.writeline(
        "# bundled AOTAutograd entries (inductor) or the captured fx graphs (eager)."
    )
    buf.writeline("# " + "=" * 70)
    buf.writeline(f"_BACKENDS = {_b64(dict(backends))!r}")
    buf.writeline("")
    buf.writeline("# " + "=" * 70)
    buf.writeline(
        "# 4. Driver: rebuild the guards, wire the names, dispatch (readable)"
    )
    buf.writeline("# " + "=" * 70)
    buf.writeline(_emit_multigraph_driver_source("_build_multigraph_forward"))
    return buf.getvalue()


def _build_multigraph_artifact(
    entry: Any,
    backends: Mapping[str, Any],
    summary: PrecompileSummary,
    backend: str,
    entry_fn: object,
    cudagraph_config: dict[str, bool],
    *,
    custom_op_modules: tuple[str, ...],
    custom_ops: tuple[tuple[str, str], ...],
    custom_op_keys: Mapping[tuple[str, str], tuple[str, ...]],
    inductor_options: Mapping[str, object] | None = None,
) -> tuple[str, bytes]:
    """``(python_code, cache)`` for a multi-graph capture.

    python_code is self-contained -- it carries the frames, the guard trees and
    the compiled subgraphs, or for an installed artifact the pickled
    ``CompilePackage`` state in ``_PACKAGE`` -- so cache is the same acceleration it is for the
    make_fx form: the inductor bundle that primes the kernel caches, and the tag
    binding it to this python_code (invariant 7). The bundle is whatever
    ``CacheArtifactManager`` recorded, so the caller must have run the capture
    under ``with_fresh_cache()``.
    """
    frames = _multigraph_frames(entry)
    # First: an entry with no variant reaches nothing, which the unreachable
    # check below would misread as frames that need installing.
    _reject_uninstallable_entry(frames, entry)
    unreachable = _unreachable_frames(frames)
    mode = "installed" if unreachable else "standalone"
    binding = _entry_binding(entry_fn) if mode == "standalone" else None
    imports = (custom_op_modules, custom_ops, custom_op_keys)
    python_code = _build_multigraph_python_source(
        entry,
        frames,
        backends,
        summary,
        backend,
        binding,
        mode,
        unreachable,
        *imports,
        cudagraph_config,
        inductor_options,
    )
    inductor_bundle = None
    if backend != "eager":
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


def _emit_multigraph_driver_source(builder: str) -> str:
    """Emit a multi-graph driver as text, the same getsource path the others use.

    The driver is a builder, so the section binds ``forward`` itself and then ends
    in the same ``__main__`` hint as the single-graph drivers."""

    from torch import _precompile_driver as driver

    body = inspect.getsource(getattr(driver, builder)).rstrip()
    forward = f"forward = {builder}()"
    return "\n" + body + "\n\n\n" + forward + "\n\n\n" + _DRIVER_MAIN


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
        inductor_options: dict[str, Any] | None = None,
    ) -> None:
        # ``fn`` is the whole computation: an nn.Module, or a callable that closes
        # over the module(s) it uses (e.g. ``lambda x: model(x)``, or a training
        # step that computes a loss and torch.autograd.grad).
        if backend not in ("inductor", "eager"):
            raise ValueError(
                f"precompile backend must be 'inductor' or 'eager', got {backend!r}."
            )
        if tracer not in ("make_fx", "dynamo"):
            raise ValueError(
                f"precompile tracer must be 'make_fx' or 'dynamo', got {tracer!r}."
            )
        from torch._dynamo.precompile_package import FROZEN_KERNEL_PINS
        from torch._inductor.standalone_compile import STANDALONE_PYTHON_PINS

        options = inductor_options or {}
        pins = {**_MAKE_FX_ASSERT_PINS, **STANDALONE_PYTHON_PINS, **FROZEN_KERNEL_PINS}
        overridden = {k: v for k, v in pins.items() if options.get(k, v) != v}
        if overridden:
            raise PrecompileError(
                f"a MakeFxTracer capture always compiles with {overridden}: its "
                "standalone module is a plain Python wrapper whose kernels are tuned "
                "at compile time, with no multi-kernel, and its size_asserts and "
                "scalar_asserts check at load that a call's inputs have the layout and "
                "ranges its graph was compiled for. It compiles unfrozen: freezing's "
                "compiler would rewrite the graph the module is built from."
            )
        self._fn = fn
        self._backend = backend
        self._tracer = tracer
        self._decompositions = decompositions
        self._inductor_options = inductor_options or {}
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
        # The dispatch keys the graph runs custom op kernels under.
        self._kernel_keys: tuple[str, ...] = _KERNEL_KEYS
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
        installed: bool = False,
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
        obj.installed = installed
        return obj

    def _compile(self, args: tuple[object, ...]) -> None:
        # This holder only runs make_fx; a DynamoTracer capture goes through
        # _DynamoCapture. Reject "dynamo" here (the single capture-dispatch point)
        # before running fn, so the failure is clear rather than a wrong fallback.
        if self._tracer != "make_fx":
            raise NotImplementedError(
                f"precompile tracer={self._tracer!r} is not implemented here; use "
                "tracer='make_fx', or capture(fn, tracer=DynamoTracer())."
            )
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
        self._kernel_keys = _graph_kernel_keys(capture.gm, capture.flat_args)

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
        from torch._dynamo.precompile_package import _frozen_kernel_config
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
        # cannot recompile to specialize a new layout the way torch.compile would.
        # scalar_asserts keeps the unbacked-symbol range asserts in the artifact. A
        # static capture can have them too: compile_to_python re-traces under a fresh
        # ShapeEnv, so a data-dependent op (nonzero, masked_select) becomes unbacked.
        #
        # These are inductor config keys, so they ride in as ``options`` (aot_autograd.
        # compile_to_python merges them into the inductor config.patch it wraps the
        # compile in) rather than being patched around the call. The AOT layer detects
        # dynamic (symbolic) shapes off the captured graph and threads the make_fx
        # ShapeEnv through automatically, so there is no dynamic_shapes knob to pass and
        # no manual TracingContext to install: a static capture specializes to the
        # example shapes, an unbacked capture keeps the symbols.
        #
        # _frozen_kernel_config() is the Triton config shared with the Dynamo capture
        # (and may log its "precompile capture overrides ..." warnings here). Its
        # autotune_at_compile_time is redundant (inductor's compile_to_python already
        # patches it) and is passed only so the two tracers cannot drift.
        options = {
            **self._inductor_options,
            **_MAKE_FX_ASSERT_PINS,
            **_frozen_kernel_config(),
        }
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
                "this object is not runnable; serve a captured artifact with "
                "torch.compiler.precompile.load(artifact_path, cache_path)."
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


def _make_inlined_forward(
    python_code: str, *, warn: bool = True, allow_serve_time_compiles: bool = False
) -> Callable[..., object]:
    """Fallback: execute the self-contained python string (JITs kernels).

    ``python_code`` needs no cache -- the kernels (inductor) or graph (eager) are
    inlined, so we just exec it and hand back its ``forward``. The returned
    ``forward`` takes the same args the traced fn took (model(s) plus runtime
    inputs). ``warn`` is off only for the capture-time self-load, where the source
    was just produced in-process and so is not untrusted input."""
    # python_code is untrusted EXECUTABLE input -- exec'ing it runs whatever it contains
    # (JIT-compiling inlined kernels or running the inlined graph). Warn per load (not
    # warning_once) before the exec so a load is never silent about it; only the
    # capture-time self-load of source this process just rendered passes warn=False.
    if warn:
        log.warning(
            "torch.compiler.precompile.load is about to EXEC python_code, which is "
            "untrusted executable input (it runs inlined kernels / graph code). Only "
            "exec python_code you produced or otherwise trust (Note [precompile "
            "programming model], invariant 7)."
        )
    if allow_serve_time_compiles:
        default = "\n_ALLOW_SERVE_TIME_COMPILES = False\n"
        if default not in python_code:
            raise PrecompileError(
                "allow_serve_time_compiles=True: the artifact's "
                "_ALLOW_SERVE_TIME_COMPILES line is not in the form capture() writes, "
                "so the flag cannot be applied. Recapture it under this torch."
            )
        python_code = python_code.replace(
            default, "\n_ALLOW_SERVE_TIME_COMPILES = True\n", 1
        )
    from torch._inductor.codecache import exec_from_cache_file

    module_ns = exec_from_cache_file(python_code, "_precompiled_artifact")
    return cast("Callable[..., object]", module_ns["forward"])


# The os.link failures that mean "this filesystem does not do hard links" (a
# FAT/exFAT mount, a container overlay, a cross-device target), as opposed to one
# about the path itself, which must not be papered over with a move. EPERM is in the
# set only for a source the CALLER owns (the ownership test at its use site).
_NO_HARD_LINK_ERRNOS = frozenset(
    {errno.EPERM, errno.EOPNOTSUPP, errno.EMLINK, errno.EXDEV}
)


def _same_inode(
    path: str | os.PathLike[str], other: str | os.PathLike[str] | os.stat_result
) -> bool:
    """True when ``path`` resolves now to ``other`` (a path or a recorded stat).

    False whenever either st_ino is 0: an inode number identifies a file only if it is
    non-zero (a FAT/exFAT mount, CIFS mounted noserverino, Windows without
    FILE_ID_INFO), and every name the writer compares lives in one directory, so
    st_dev alone would call two DIFFERENT files the same one. Unknown identity must
    never read as "same file".
    """
    try:
        st = other if isinstance(other, os.stat_result) else os.stat(other)
        here = os.stat(path)
        if not st.st_ino or not here.st_ino:
            return False
        return os.path.samestat(here, st)
    except OSError:
        return False


def _unlink_quietly(path: str | os.PathLike[str]) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _write_artifact(
    artifact_path: str | os.PathLike[str],
    cache_path: str | os.PathLike[str],
    python_code: str,
    cache: bytes,
) -> None:
    """Write the matched (python_code, cache) pair, creating parent directories.

    Each half is written beside its target and renamed into place, so neither named file
    is ever truncated or half-written. The two renames are not one atomic step: the
    previous source is hard-linked to a backup first and put back if the second rename
    raises, so a Python exception (a full disk, a permission error) leaves the previous
    pair intact. Without hard links the previous source is MOVED aside instead, so an
    interrupt before the undo's probes have run leaves the artifact NAME empty with that
    source only in the ``.bak``, unreported and one rename from recovered. Which undo
    runs, and whether it is reported, is read off the DISK, not from flags (the comment
    on the undo has the reasoning). Process death between the renames is not covered, nor
    a reader or a second writer racing them: that can leave one source beside the other's
    cache, which ``load`` refuses on the cache's sha256, and can cost the previous source.
    Nor is a SYMLINKED artifact name: ``os.link`` follows it, so the backup holds the
    link's target while ``os.replace`` replaces the link itself, and an undo therefore
    restores a regular file where the symlink stood.
    The parent directory is fsync'd after, best effort.
    """
    written = []
    new_stats: list[os.stat_result] = []
    try:
        for path, payload in ((artifact_path, python_code), (cache_path, cache)):
            parent = os.path.dirname(os.fspath(path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            # A unique name per writer: two captures targeting one path must not share a
            # scratch file. Beside the target, so the rename stays on one filesystem.
            tmp = f"{os.fspath(path)}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            written.append((tmp, path))
            # The temp is CREATED with the mode of the file its rename replaces: chmod'ing
            # it down only after the write would publish the whole new payload at the umask
            # mode, in a directory the caller chose. Permission bits only: setuid/setgid on a
            # new inode owned by the WRITING user name a different principal. O_BINARY:
            # os.open on Windows translates the newlines code_hash covers.
            try:
                mode: int | None = stat.S_IMODE(os.stat(path).st_mode) & 0o777
            except OSError:
                mode = None
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
            perm = 0o666 if mode is None else mode
            # opener=, not a bare os.open: an fd is unowned until open() wraps it.
            with open(tmp, "wb", opener=lambda p, _: os.open(p, flags, perm)) as f:
                f.write(payload.encode() if isinstance(payload, str) else payload)
                f.flush()
                os.fsync(f.fileno())
            if mode is not None:
                # O_CREAT's mode is umask-masked, so the exact bits need this too; best
                # effort, a filesystem that drops modes must not fail the write.
                try:
                    os.chmod(tmp, mode)
                except OSError:
                    pass
            # Name the bytes about to be renamed in by inode, for the undo below.
            new_stats.append(os.stat(tmp))
    except BaseException:
        for tmp, _ in written:
            _unlink_quietly(tmp)
        raise
    (artifact_tmp, _), (cache_tmp, _) = written
    backup = f"{os.fspath(artifact_path)}.{os.getpid()}.{uuid.uuid4().hex}.bak"
    try:
        # A hard link, not a move: the named path must resolve to the previous or the new
        # source at every instant.
        try:
            os.link(artifact_path, backup)
        except FileNotFoundError:
            pass
        except OSError as e:
            # No hard links here: fall back to moving aside, only for the errnos that mean
            # unsupported and only for a regular file (os.link on a DIRECTORY also fails EPERM).
            if e.errno not in _NO_HARD_LINK_ERRNOS or not os.path.isfile(artifact_path):
                raise
            # EPERM is also what fs.protected_hardlinks=1 (a Linux default) raises for a
            # source the caller does not OWN, and chattr +i for one it cannot write -- on a
            # filesystem that does have hard links. Moving such a file out from under its
            # name is what the fallback must not do to someone else's artifact, so on POSIX
            # that errno means "no hard links" only for a source the caller owns.
            if (
                e.errno == errno.EPERM
                and hasattr(os, "geteuid")
                and os.stat(artifact_path).st_uid != os.geteuid()
            ):
                raise
            os.replace(artifact_path, backup)
        os.replace(artifact_tmp, artifact_path)
        os.replace(cache_tmp, cache_path)
        # Any backup taken above is superseded; inside the try, so an interrupt here is handled.
        _unlink_quietly(backup)
    except BaseException:
        # Put the previous source back (or remove the new one on a first write) so the named
        # files stay loadable, best effort; then drop every temp and re-raise. Every predicate
        # that CHOOSES the undo is read off the DISK, never from a flag set after its own
        # syscall, and ``probed`` says all four reads RAN: an interrupt among them leaves the
        # rest half-set, so neither the report nor the drop below may consult one. ``kept``:
        # the backup name carries this call's pid and a uuid, so its existing is this call's.
        # ``complete`` (both names are this call's temps) means written even though this block
        # ran; ``aside`` (a backup with the artifact NAME gone) is the move-aside fallback
        # holding the previous source's only copy; ``landed`` says the FIRST rename happened.
        # ``undone`` says the undo RETURNED, the finally's unlink safe under it.
        complete = kept = landed = aside = probed = undone = False
        try:
            # Where st_ino is 0 no inode read can say whose source a name holds, so the
            # temps stand in for the renames: os.replace CONSUMED the one it renamed, so a
            # temp that is GONE is its rename having run. Only consulted there, since a
            # temp says nothing about a SECOND writer repointing the name afterwards, which
            # is what the inode read sees and what the restore below must not overwrite.
            # Per HALF: the two names are independent arguments, so one can be on a
            # filesystem with inodes and the other not.
            blind_artifact = not new_stats[0].st_ino
            blind_cache = not new_stats[1].st_ino
            landed = _same_inode(artifact_path, new_stats[0]) or (
                blind_artifact and not os.path.lexists(artifact_tmp)
            )
            complete = landed and (
                _same_inode(cache_path, new_stats[1])
                or (blind_cache and not os.path.lexists(cache_tmp))
            )
            kept = os.path.lexists(backup)
            aside = kept and not os.path.lexists(artifact_path)
            probed = True
            try:
                if complete or not (landed or kept):
                    undone = True
                elif kept and (landed or aside):
                    os.replace(backup, artifact_path)
                    undone = True
                elif kept:
                    # The artifact name is neither this call's new source nor gone, so it
                    # cannot tell the previous source still under the hard link (a failed FIRST
                    # rename) from one a second WRITER repointed here. Neither wants a restore:
                    # the first already IS the previous pair, the second a THIRD, older source.
                    undone = True
                elif landed:
                    # A first write, so there is no previous pair to restore: drop the new
                    # source rather than leave it named with no cache beside it. An unlink, not
                    # a rename: a rename needs a directory entry, which ENOSPC just exhausted.
                    os.unlink(artifact_path)
                    undone = True
            except OSError:
                pass
        finally:
            # Keyed on that same on-disk outcome: a report NAMES a file, so it fires only
            # while that file is there and ``probed`` says the reads that chose it ran.
            named = backup if kept else artifact_path
            if probed:
                drop_backup = kept
            else:
                # An interrupt among the reads never set ``kept``, so the drop reads the
                # name itself and fires only while it resolves to the backup's inode.
                drop_backup = _same_inode(artifact_path, backup)
            if probed and not undone and os.path.lexists(named):
                if kept:
                    # Reached from both shapes the undo's rename serves, so the wording
                    # names only what holds in both: the .bak, and the one rename back.
                    log.warning(
                        "precompile could not put the previous artifact back at %s; that "
                        "previous source is kept at %s, and the pair does not load until "
                        "that file is moved back over the first path (the cache at %s is "
                        "the one that matches it).",
                        os.fspath(artifact_path),
                        backup,
                        os.fspath(cache_path),
                    )
                else:
                    log.warning(
                        "precompile wrote the artifact at %s and then failed to write "
                        "its cache at %s, and could not remove the artifact again; no "
                        "cache beside it matches that file, so the pair does not load.",
                        os.fspath(artifact_path),
                        os.fspath(cache_path),
                    )
            elif drop_backup:
                # Reached only where the named pair came out loadable, so the backup is not a
                # copy anyone still needs: an undo rename consumed it, or nothing needed undoing
                # because the previous source is still under its own name (a failed FIRST rename).
                _unlink_quietly(backup)
            for tmp, _ in written:
                _unlink_quietly(tmp)
        raise
    parents = {os.path.dirname(os.fspath(path)) or "." for _, path in written}
    # Durably record the renames: without an fsync of the containing directory a crash
    # just after os.replace returns can still lose the new entry and resurrect the old.
    for parent in parents:
        try:
            fd = os.open(parent, os.O_RDONLY)
        except OSError:
            continue
        try:
            # Best effort like the os.open above, close included: by here both renames have
            # returned, so an error out of either would fail an already loadable pair.
            os.fsync(fd)
        except OSError:
            pass
        finally:
            try:
                os.close(fd)
            except OSError:
                pass


def _read_artifact(
    artifact_path: str | os.PathLike[str],
    cache_path: str | os.PathLike[str],
) -> tuple[str, bytes]:
    r"""Read back a pair written by :func:`_write_artifact`.

    Bytes, then decoded: the exact inverse of the byte-mode write, so a ``\r`` in
    python_code survives the round trip instead of being translated to ``\n`` by a
    text-mode read and failing the cache's code_hash. An ``OSError`` from either open (a
    missing half, a path the filesystem cannot open) or a ``UnicodeDecodeError`` from the
    decode (a readable file that is not the source half -- transposed arguments, say) is
    a ``PrecompileError`` naming both paths, with the original as its ``__cause__``.
    """
    try:
        with open(artifact_path, "rb") as f:
            python_code = f.read().decode()
        with open(cache_path, "rb") as f:
            cache = f.read()
    except (OSError, UnicodeDecodeError) as e:
        raise PrecompileError(
            f"precompile could not read the artifact pair (artifact_path="
            f"{artifact_path!r}, cache_path={cache_path!r}): {e}"
        ) from e
    return python_code, cache


def _verified_cache_envelope(
    cache: bytes,
    *,
    backend: str,
    tracer: str,
    code_hash: str,
    strict: bool,
) -> dict[str, Any] | None:
    # weights_only=True is safe (plain str/int/bytes dict). Outside strict mode the
    # cache is acceleration only, so an unreadable envelope or a FORMAT / VERSION
    # mismatch degrades to JIT'ing from python_code. Under no_compilation() that JIT
    # is forbidden, so the same conditions raise instead. A BACKEND, TRACER or
    # CODE_HASH mismatch signals a wrong (python_code, cache) pairing and always
    # raises; see Note [precompile programming model], invariant 7.
    try:
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        if blob.get("format") != _CACHE_FORMAT or blob.get("version") != _CACHE_VERSION:
            if strict:
                raise PrecompileError(
                    "strict precompile.load requires a compatible cache envelope; "
                    f"got format={blob.get('format')!r}, "
                    f"version={blob.get('version')!r}, "
                    f"expected {_CACHE_FORMAT!r}, {_CACHE_VERSION!r}."
                )
            log.warning(
                "torch.compiler.precompile got a cache with format=%r "
                "version=%r, expected %r / %r; it is likely from a different torch "
                "build. Falling back to JIT from python_code.",
                blob.get("format"),
                blob.get("version"),
                _CACHE_FORMAT,
                _CACHE_VERSION,
            )
            return None
        if blob.get("backend") != backend:
            raise PrecompileError(
                f"cache backend {blob.get('backend')!r} does not match the "
                f"python_code backend {backend!r}; the cache and python_code "
                "came from different precompile captures."
            )
        if blob.get("tracer", "make_fx") != tracer:
            raise PrecompileError(
                f"cache tracer {blob.get('tracer', 'make_fx')!r} does not match "
                f"the python_code tracer {tracer!r}; the cache and python_code "
                "came from different precompile captures."
            )
        if blob.get("code_hash") != code_hash:
            raise PrecompileError(
                "cache does not match python_code (its code_hash "
                f"{blob.get('code_hash')!r} != sha256(python_code) "
                f"{code_hash!r}); the cache and python_code came from "
                "different precompile captures. Pair each cache with the "
                "python_code from the same capture."
            )
        return blob
    except PrecompileError:
        raise
    except Exception as e:
        if strict:
            raise PrecompileError(
                "strict precompile.load could not read the cache envelope"
            ) from e
        log.warning(
            "torch.compiler.precompile could not read the cache envelope (%s: %s); the "
            "cache is likely corrupt or from a different torch build. Falling back "
            "to JIT from python_code.",
            type(e).__name__,
            e,
        )
        return None


def no_compilation() -> contextlib.AbstractContextManager[None]:
    """
    Forbid graph/kernel compilation and autotuning on every thread of this process.

    Use it around a serving worker's lifetime to prove that a loaded artifact
    and its kernel/autotune cache cover every execution: anything that would
    trace a graph, compile a kernel, benchmark or autotune raises
    :class:`~torch.compiler.PrecompileError` instead of silently compiling::

        runnable = torch.compiler.precompile.load(artifact_path, cache_path)
        with torch.compiler.precompile.no_compilation():
            serve(runnable)  # raises rather than compiling on a cache miss

    Only kernels already in the in-process kernel cache may run.

    Unlike ``precompile.serving()``, which is thread-local, the policy is
    process-wide. Nested and overlapping uses are depth-counted, so it stays
    active until the last owner exits; callers must drain work they dispatched
    (e.g. to compile workers) before that exit. While active,
    ``TORCH_PRECOMPILE_NO_COMPILATION=1`` is exported so child processes
    inherit it, and setting exactly ``"1"`` before startup enables the policy
    for the whole process.
    """
    from torch.compiler._no_compile import no_compilation as _no_compilation

    return _no_compilation()


def _runnable_from_pair(
    python_code: str,
    cache: bytes,
    *,
    allow_serve_time_compiles: bool = False,
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
    # TRACER is absent on make_fx artifacts predating the tag; the cache envelope
    # defaults the same way, so an older pair still matches.
    tracer = cast(str, meta.get("TRACER", "make_fx"))
    installed = meta["SERVING_MODE"] == "installed"
    if allow_serve_time_compiles and not installed:
        raise PrecompileError(
            "allow_serve_time_compiles=True applies only to an installed artifact "
            '(SERVING_MODE = "installed"), which serves through Dynamo; this '
            "artifact is standalone and has no compiler behind it. Load it without "
            "the flag, or recapture to cover the calls it misses."
        )
    if allow_serve_time_compiles and "_ALLOW_SERVE_TIME_COMPILES" not in meta:
        raise PrecompileError(
            "allow_serve_time_compiles=True: this installed artifact predates the "
            "flag and would still refuse every uncovered call. Recapture it under "
            "this torch."
        )
    if allow_serve_time_compiles and meta.get("INDUCTOR_OPTIONS", {}) is None:
        raise PrecompileError(
            "allow_serve_time_compiles=True: this artifact's capture took options "
            "that are not all Python literals (such as a custom pass), so a "
            "serve-time compile could not use them. Load it without the flag, or "
            "recapture with literal options."
        )
    if meta.get("_ALLOW_SERVE_TIME_COMPILES", False) is not False:
        raise PrecompileError(
            "this artifact sets _ALLOW_SERVE_TIME_COMPILES to something other than "
            "False; capture() always writes False, and serve-time compiles are "
            "enabled only by load(..., allow_serve_time_compiles=True)."
        )

    from torch.compiler._no_compile import is_compilation_forbidden

    strict = is_compilation_forbidden()
    blob = _verified_cache_envelope(
        cache,
        backend=backend,
        tracer=tracer,
        code_hash=hashlib.sha256(python_code.encode()).hexdigest(),
        strict=strict,
    )
    artifact = blob.get("artifact") if blob is not None else None
    # Only inductor's python_code JITs kernels at load, so other backends run without
    # the artifact. An inductor capture of an uncacheable graph saves no artifact and
    # cannot be served strictly.
    if strict and backend == "inductor" and not artifact:
        raise PrecompileError(
            "strict precompile.load requires the compiled cache artifact"
        )
    if artifact is not None:
        # Prime the inductor kernel caches from the bundle so the exec of python_code
        # below loads the precompiled kernels (Triton binaries / autotune results)
        # instead of recompiling them. The composed python_code runs its inlined
        # kernels directly (no compile_fx re-entry, so no FxGraphCache lookup); the
        # acceleration is the warm kernel cache. Outside strict mode this is a pure
        # acceleration: a stale / cross-torch-version / corrupt bundle that fails to
        # load just leaves the caches cold, and python_code JITs -- same result, no
        # crash. Under no_compilation() that JIT is forbidden, so hydration must
        # succeed.
        try:
            cache_info = torch.compiler.load_cache_artifacts(artifact)
        except Exception as e:
            if strict:
                raise PrecompileError(
                    "strict precompile.load could not hydrate the compiled cache"
                ) from e
            log.warning(
                "torch.compiler.precompile could not prime the cache from the "
                "artifact bundle (%s: %s); it is likely stale or from a different "
                "torch build. Falling back to JIT from python_code.",
                type(e).__name__,
                e,
            )
        else:
            if strict and (cache_info is None or cache_info.empty()):
                raise PrecompileError(
                    "strict precompile.load could not hydrate the compiled cache"
                )
    # Run the driver inlined in python_code. It carries the full calling convention and
    # runtime safety checks (subclass wrap/unwrap, param/buffer lifting, grad harvest,
    # input/model validation) and JITs the kernels -- which hit the primed cache when
    # the bundle above loaded, so the "cache" path is exec-with-warm-kernels rather than
    # a separate runtime.
    forward = _make_inlined_forward(
        python_code,
        warn=not _trusted,
        allow_serve_time_compiles=allow_serve_time_compiles,
    )
    return PrecompiledModule._from_loaded(forward, backend=backend, installed=installed)


def capture(
    fn: Callable[_P, _R],
    /,
    *,
    artifact_path: str | os.PathLike[str],
    cache_path: str | os.PathLike[str],
    tracer: MakeFxTracer | DynamoTracer = DynamoTracer(),
    backend: str = "inductor",
    mode: str | None = None,
    options: dict[str, Any] | None = None,
) -> Capture[_P, _R]:
    """Capture ``fn`` across the calls YOUR loop makes, writing the artifact on exit.

    .. warning::

        This is a prototype API. Its signature, error types and artifact
        format may change between releases without a deprecation cycle.

    Capture is caller-driven: this returns a capture object rather than running
    anything. Enter it as a context manager, call it exactly as you would ``fn``
    inside the block (a :class:`MakeFxTracer` capture takes exactly one call) --
    each call runs for real, folds what it exercised into the capture, and
    returns that run's result -- and the ``(python_code, cache)`` artifact is
    written to ``artifact_path`` / ``cache_path`` when the block exits::

        with torch.compiler.precompile.capture(
            fn, artifact_path="m.py", cache_path="m.cache"
        ) as cap:
            y = cap(model, x)
        f = torch.compiler.precompile.load("m.py", "m.cache")

    Because the caller makes the calls, inputs flow through naturally and return
    values stay available, so the capture drops into an ordinary pipeline loop.
    To write the artifact before the block ends, call ``cap.save()`` inside it.
    A block that raises writes nothing it has not already saved, and a block
    that made no call raises ``PrecompileError`` on exit. A capture is single-use:
    to capture again, call ``capture()`` again. A :class:`DynamoTracer` capture
    records the kernel-cache artifacts of everything compiled inside the block,
    so compile nothing else there, and do not overlap two such captures. Only
    what runs inside the block is carried: run every kernel the serving process
    needs there, since one compiled or launched only before the block is not in
    the cache. Each write also freezes the compiled kernels the serving process
    needs, and raises :class:`~torch.compiler.PrecompileError` for one it cannot
    freeze. That includes kernels launched directly through ``@triton.jit`` and
    ``@triton.autotune`` inside the block: Triton compiles into a fresh cache
    directory there, with its autotuning cache on, so a ``TRITON_CACHE_DIR``
    you set is not used inside the block, and a JIT kernel or autotuner already
    used before the block compiles and tunes again on its first launch inside
    it. That retune benchmarks on the block's real inputs, and the config it
    picks, which is what ships and what the process keeps using afterwards, may
    differ from the one picked before the block. Compiled JIT kernel
    specializations and autotuning decisions the block does not produce again
    are kept afterwards.
    Triton does not cache the decision of an autotuner with ``pre_hook``
    configs, so one that does not prune its configs keeps its earlier decision
    in the block and is benchmarked again on first use in the serving process.
    No other thread should launch Triton kernels while the block opens or
    closes: the in-memory caches are dropped and restored then without
    synchronizing with launches, so a concurrent autotuner launch can fail.
    Triton kernels another thread compiles or tunes while the block is open
    go into the block's cache directory and ship with it.

    ``tracer`` picks the capture front-end and carries its tracer-specific
    configuration. :class:`DynamoTracer` (the default) is an execution-driven
    multi-graph capture: it analyzes the Python rather than tracing one path,
    records every graph-break continuation and guarded recompilation the calls
    exercise, takes as many calls as you make (keyword arguments included), and
    the artifact dispatches among the captured variants by their guards -- a
    call no variant covers RAISES rather than silently serving, since there is
    no compiler behind a source artifact, unless Dynamo gave up on that frame
    during the capture (``PrecompileSummary.eager_fallbacks``): the capture ran
    that frame eagerly, so the call runs eagerly too. :class:`MakeFxTracer` is
    one non-strict ATen trace: the capture takes exactly ONE positional call,
    refuses a second, and specializes control flow and shapes to that call, with
    the contract of Note [precompile programming model] in
    ``torch/_precompile.py``. ``fn`` is the whole computation, e.g. a
    module-level ``def step(model, x): return model(x)`` (only a
    :class:`MakeFxTracer` capture also takes a lambda): the ``nn.Module``
    arguments have their params/buffers lifted to graph inputs (no weights are
    baked in) and the rest are the runtime inputs; the reloaded callable is
    invoked with the same argument structure, and the runtime model must match
    the captured model's parameter/buffer structure.

    A Dynamo artifact is STANDALONE: it rebuilds the entry from its code object
    and installs nothing, so ``fn`` must be a module-level function of an
    importable module (not a closure, not defined in ``__main__``). The
    exception is a capture that compiled a frame the entry cannot reach through
    a graph-break continuation -- a graph break inside a child module's forward:
    its artifact has ``SERVING_MODE = "installed"`` and serves by installing the
    captured entries onto the live code objects when loaded. It is locked
    to the producing Python version and torch build. The guards that could not
    be serialized are reported in the capture's ``summary()`` rather than
    checked, and the :class:`DynamoTracer` ``require_*`` gates refuse the risky
    ones by default.

    ``backend`` selects how the captured graph is realized: ``"inductor"``
    (default) lowers through AOTAutograd + Inductor into one self-contained
    module, and the cache holds the bundle that primes the inductor kernel caches
    on load; ``"eager"`` keeps the captured ATen graph and runs it as-is (no
    kernels, so the cache carries no artifact). A make_fx ``fn`` that runs a
    backward captures it: the trace runs with grad enabled, the resulting
    parameter gradients are scattered onto the runtime model exactly like eager
    ``.backward()``, and the artifact returns ``fn``'s own result. A Dynamo
    capture runs every call in the caller's own grad mode and lowers a backward
    whenever grad is enabled (a ``.backward()`` in ``fn`` graph-breaks and
    re-runs at serve time through the live autograd engine, accumulating
    ``.grad`` like eager); serve the artifact under the grad mode it was
    captured in.

    An ``"inductor"`` capture tunes every kernel while its graph lowers, since
    the artifact replays each kernel with one frozen config: Inductor's
    kernels, forward included, are tuned on synthetic inputs at the size
    hints, not on the block's real calls. A ``@triton.autotune`` kernel traced
    into a captured graph is likewise tuned on random tensors, and the config
    it picks ships in the artifact. A nonzero ``triton.multi_kernel`` is turned
    off, with a warning, while each captured graph lowers (user code between
    captured calls keeps the ambient config): a multi-kernel picks its
    sub-kernel when the loaded module runs, by benchmarking at first launch or
    from the serving process's own ``triton.multi_kernel``, which the artifact
    cannot record. A :class:`DynamoTracer` capture with grad enabled lowers
    each backward while the forward's tensors are still live, so peak device
    memory during the capture grows by roughly the backward's working set.

    ``mode`` and ``options`` configure Inductor as they do for
    :func:`torch.compile`: at most one of them may be given, and only with
    ``backend="inductor"``. They apply to every graph the capture compiles.
    ``mode="reduce-overhead"`` (or ``"max-autotune"``, or
    ``options={"triton.cudagraphs": True}``) makes a :class:`DynamoTracer`
    capture serve its loaded graphs as CUDA graphs, as the capture ran them:
    the artifact records ``triton.cudagraphs``, ``triton.cudagraph_trees``
    and ``graph_partition`` (which picks how a partitioned graph is wrapped)
    as the capture compiled under them (``mode`` and ``options`` over the
    process's ``torch._inductor.config`` when ``capture()`` is called) in its
    ``CUDAGRAPH_CONFIG``, and the loading process records its CUDA graphs
    afresh on its first calls. Unlike under ``torch.compile``, whose options
    reach only Inductor, the capture's calls are traced under these three
    keys too, so with CUDA graphs on Dynamo specializes every Python float
    argument (each distinct value is its own captured variant) rather than
    turning one that changes into a CPU scalar tensor, which would keep the
    graph out of CUDA graphs. Whether a graph can run as a CUDA graph at all
    (``triton.cudagraph_support_input_mutation``,
    ``force_disable_cudagraph_TESTING_ONLY``) is decided when it compiles, and
    that decision ships with it. The settings CUDA graph trees read while the
    graphs run (``triton.cudagraph_capture_sizes``,
    ``triton.cudagraph_dynamic_shape_rerecord_limit``,
    ``triton.cudagraph_or_error``, ...) are the serving process's, as under
    ``torch.compile``, whose options also cover only the compile.
    ``cudagraph_policy``, which wraps the graphs again at load and is a Python
    object, is refused in ``options``. As under ``torch.compile``, a
    graph's outputs are overwritten by its next replay, so clone what you keep
    across calls or call :func:`torch.compiler.cudagraph_mark_step_begin`
    between iterations; a graph Inductor cannot capture (CPU tensors, mutated
    inputs, ...) runs without one and logs a cudagraph skip; and a frame served
    eagerly (``PrecompileSummary.eager_fallbacks``) runs no CUDA graph. With
    ``backend="inductor"``, a capture is refused when a
    :func:`torch._dynamo.override_cudagraphs` changes whether a graph runs as a
    CUDA graph from what the capture's ``triton.cudagraphs`` decides (``fwd``
    differing from it, or ``bwd=False`` under a CUDA-graphed forward): the
    artifact records the capture's CUDA graph config, not each graph's. A
    :class:`DynamoTracer` capture refuses ``freezing`` and
    ``force_disable_caches``, whose compiles file no AOTAutograd cache entry
    for the artifact to ship, ``fx_wrapper``, whose graphs Inductor does not
    cache, and ``bundled_autotune_remote_cache`` on, which the capture turns
    off (its eager backward lowering fails on a remote autotune cache miss);
    its ``cpp_wrapper`` graphs capture on CPU only. A :class:`MakeFxTracer`
    capture refuses a ``mode`` or ``options`` that turns on
    ``triton.cudagraphs``, since its standalone module runs outside Inductor's
    CUDA graph wrapper (it ignores the ambient flag, as before), and options
    that module pins: ``size_asserts`` or ``scalar_asserts`` off,
    ``cpp_wrapper``, ``fx_wrapper``, ``benchmark_harness`` or ``freezing`` on
    (freezing's compiler would rewrite the graph the module is built from).
    Either tracer refuses ``triton.autotune_at_compile_time`` off and a nonzero
    ``triton.multi_kernel``, which the paragraph above turns off. Other options
    pass through unchecked, as under :func:`torch.compile`.

    .. note::

        ``torch.compiler.precompile`` is NOT
        ``torch._dynamo.config.caching_precompile`` (a ``torch.compile``
        guard-serialization caching mode); it captures ``fn`` ahead of time and
        lowers it to a self-contained Python source artifact.
    """
    torch._C._log_api_usage_once("torch.compiler.precompile.capture")
    if isinstance(fn, functools.partial):
        raise TypeError(
            "precompile.capture takes the function itself, not a functools.partial: "
            "the artifact rebuilds the call from the function's own parameters, so a "
            "pre-bound model or argument would not be one of them. Pass them as call "
            "arguments instead."
        )
    if not isinstance(tracer, (MakeFxTracer, DynamoTracer)):
        raise TypeError(
            "precompile.capture tracer must be a MakeFxTracer or DynamoTracer, "
            f"got {type(tracer).__name__}."
        )
    # Without mode or options the backend string compiles, as it always has.
    inductor: torch._TorchCompileInductorWrapper | None = None
    if mode is not None or options is not None:
        if backend != "inductor":
            raise ValueError(
                f"precompile mode= and options= configure Inductor; backend={backend!r} "
                "takes neither."
            )
        if mode is not None and options is not None:
            raise RuntimeError(
                "Either mode or options can be specified, but both can't be "
                "specified at the same time."
            )
        # Keys normalized as the wrapper does; it cannot take this one at all
        # (its type check fails on the string annotation).
        keys = {k.replace("-", "_") for k in options or ()}
        if "cudagraph_policy" in keys:
            raise PrecompileError(
                "precompile cannot record options={'cudagraph_policy': ...}: the "
                "policy wraps the graphs again at load, and an artifact holds no "
                "Python objects. Set torch._inductor.config.cudagraph_policy in the "
                "loading process instead."
            )
        if isinstance(tracer, MakeFxTracer):
            # Refused before the wrapper is built, which sets CUPTI environment
            # variables for a CUDA graph config. A value that is not a bool is
            # left to the wrapper's type check.
            from torch._inductor import list_mode_options

            requested = list_mode_options(mode) if mode is not None else options or {}
            if requested.get("triton.cudagraphs") is True:
                remedy = (
                    "use mode='max-autotune-no-cudagraphs'"
                    if mode is not None
                    else "drop triton.cudagraphs from options"
                )
                raise PrecompileError(
                    "a MakeFxTracer capture cannot serve CUDA graphs: its standalone "
                    "module runs outside Inductor's CUDA graph wrapper. Capture with "
                    f"tracer=DynamoTracer(), or {remedy}."
                )
        # torch.compile's own parsing: mode expands to its options, and an unknown
        # option or a value of the wrong type raises here rather than mid-capture.
        dynamic = tracer.dynamic if isinstance(tracer, DynamoTracer) else None
        inductor = torch._TorchCompileInductorWrapper(mode, options, dynamic)
    if isinstance(tracer, MakeFxTracer):
        inductor_options = inductor.config if inductor is not None else {}
        return _MakeFxCapture(
            fn,
            artifact_path,
            cache_path,
            backend=backend,
            decompositions=tracer.decompositions,
            inductor_options=inductor_options,
        )
    # The make_fx front-end validates backend when it builds its PrecompiledModule;
    # the Dynamo front-end builds no PrecompiledModule, so it is checked here.
    if backend not in ("inductor", "eager"):
        raise ValueError(
            f"precompile backend must be 'inductor' or 'eager', got {backend!r}."
        )
    from torch._dynamo.exc import PackageError
    from torch._dynamo.precompile_package import precompile_capture

    try:
        session = precompile_capture(
            fn,
            backend=inductor or backend,
            guard_filter_fn=tracer.guard_filter_fn,
            recompile_limit=tracer.recompile_limit,
            dynamic=tracer.dynamic,
        )
    except PackageError as e:
        raise PrecompileError(str(e)) from e
    return _DynamoCapture(
        session,
        artifact_path,
        cache_path,
        require_complete=tracer.require_complete,
        require_no_risky_drops=tracer.require_no_risky_drops,
    )


def load(
    artifact_path: str | os.PathLike[str],
    cache_path: str | os.PathLike[str],
    *,
    allow_serve_time_compiles: bool = False,
) -> PrecompiledRunnable:
    """Reconstruct a runnable from the two files a precompile capture wrote.

    .. warning::

        This is a prototype API. Its signature, error types and artifact
        format may change between releases without a deprecation cycle.

    Name the two files :func:`capture` wrote -- the ``python_code`` artifact and
    its ``cache``. A readable, current-format cache loads only
    against the ``python_code`` it was emitted with (it carries a sha256 of exactly
    those bytes); an unreadable or other-format cache is ignored with a warning.

    The driver runs from ``python_code`` -- the single source of truth for the whole
    calling convention. ``load`` reads the source's ``BACKEND``, checks the cache's
    ``backend`` tag against it, primes the inductor kernel caches when the cache
    carries a bundle so a warm reload loads precompiled kernels instead of
    JIT-compiling, and then exec's ``python_code``. With no usable cache a
    :class:`MakeFxTracer` artifact degrades to JIT'ing from ``python_code``; a
    :class:`DynamoTracer` artifact carries its compiled subgraphs inside
    ``python_code``, with their kernels under inductor's default bundling, so it
    serves without JIT either way. Both files are trusted, EXECUTABLE input: load
    only artifacts you produced or otherwise trust.

    Call the result with the SAME argument structure ``fn`` took -- the model(s)
    in their original positions plus the runtime inputs (keyword arguments too,
    for a :class:`DynamoTracer` artifact). The runtime model must match the
    captured model's parameter/buffer structure. The result is a
    :class:`torch.compiler.precompile.PrecompiledRunnable`; a Dynamo artifact
    is standalone, so entering and unloading it are no-ops, and it re-seeds the
    names Dynamo minted into the captured module. The exception is a capture that
    compiled a frame the entry reaches only by an ordinary call (a graph break
    inside a child module's forward): its artifact has ``SERVING_MODE =
    "installed"`` and serves by installing the captured entries onto the live
    code objects; its calls refuse to compile, as under the ``fail_on_recompile``
    stance, without setting that process-wide stance, and keep refusing under
    every process-wide stance that would compile (``force_backend``,
    ``"eager_then_compile"``, ...). Under ``"force_eager"`` every call runs
    eagerly, bypassing the installed entries; under ``"eager_on_recompile"`` the
    captured variants are served and any other call runs eagerly. An installed
    artifact refuses to load with ``torch._dynamo.config.compiled_autograd``
    enabled, which would compile backward graphs outside it, and with Dynamo
    disabled (``TORCHDYNAMO_DISABLE=1`` or the ``enable_dynamo`` killswitch),
    which it serves through. A Dynamo artifact whose
    capture graph-broke cannot load beside the live compile that captured it
    (the continuation names collide), and an installed artifact is refused
    wherever any frame it installs onto already has live Dynamo cache entries:
    load it in a fresh process. That check runs only at load: compiling those
    frames after load adds live entries that serve the calls the installed
    ones miss.

    Either kind of artifact runs a call eagerly instead of refusing it when the
    missed frame is one Dynamo gave up on during the capture (listed in
    ``PrecompileSummary.eager_fallbacks`` and marked ``# eager fallback`` in the
    artifact's ``FRAMES``), since the capture ran that frame eagerly too. A
    frame that kept variants logs a warning the first time a call misses them;
    one that compiled none is served as plain Python from the start, as a
    trivial frame is, with no warning, since there is no compiled code to lose.
    Under an installed artifact only that frame runs eagerly: the frames it
    calls still dispatch to their own variants. A standalone artifact installs
    nothing, so the rest of the call from that frame on, its callees included,
    runs as plain Python; whatever ran before it, such as the entry's variant
    up to a graph break whose continuation fell back, still ran compiled.

    The graphs a :class:`DynamoTracer` artifact ships are wrapped in CUDA graphs
    by its capture's ``triton.cudagraphs``, ``triton.cudagraph_trees`` and
    ``graph_partition`` (``CUDAGRAPH_CONFIG``), whatever the loading process's
    ``torch._inductor.config`` says, unless the loading process sets
    ``cudagraph_policy``, which wraps them again; the rest of the CUDA graph
    config is the loading process's. A serve-time compile
    (``allow_serve_time_compiles``) is lowered by Inductor under the capture's
    ``mode`` or ``options`` and ``CUDAGRAPH_CONFIG`` over the loading process's
    config, but Dynamo traces it under the loading process's own
    ``triton.cudagraphs``, which decides whether a Python float that changed
    since capture is specialized; set it as the capture did to trace alike.

    Frozen C++ kernels are keyed by their build command with the PyTorch install
    taken out, so they serve a host that installed the same PyTorch build
    elsewhere. The Python, compiler, CUDA and other SDK paths must still match the
    producer's. The key also covers each kernel's source, so a kernel whose source
    or command names a path in the Inductor cache dir (a linked Halide runtime,
    CUTLASS or CK library, or a GPU ``cpp_wrapper`` wrapper's cubin path) keeps
    that path: a host whose cache dir is at the producer's path restores the frozen
    binary, which needs those files at the same path, and any other host rebuilds
    it. Rebuilding needs a C++ compiler on the serving host: :func:`no_compilation`
    does not gate C++ builds.

    ``allow_serve_time_compiles=True`` opts an installed artifact out of the
    refusal: a call no captured variant covers compiles with the artifact's
    backend and the capture's ``mode`` or ``options`` (recorded in the
    artifact's ``INDUCTOR_OPTIONS``), as ``torch.compile`` would, under the
    process-wide stance like any other compile. An artifact whose options are
    not all Python literals (such as a custom pass) refuses the flag. The
    captured variants are still served first and do not count toward
    ``torch._dynamo.config.recompile_limit``, so each frame gets
    that many serve-time compiles on top of them; a missed call past the limit
    runs eagerly instead of compiling.
    Serve-time compiles are recorded nowhere, but they are built under the
    process's Dynamo config: with ``torch._dynamo.config.caching_precompile`` on
    they drop identity guards (``ID_MATCH``, ``CLOSURE_MATCH``, ...) exactly as
    any other compile in that process does.
    The flag does not lift the ``compiled_autograd`` refusal above. A standalone
    artifact has no compiler behind it, and an installed artifact captured
    before the flag existed cannot honor it, so passing the flag for either
    raises ``PrecompileError``.

    Before its graphs load, the artifact imports the modules in its
    ``CUSTOM_OP_MODULES`` (the ones registering the custom ops its graphs call)
    and then checks that every op in ``CUSTOM_OPS`` is registered, with a kernel
    for each dispatch key ``CUSTOM_OP_KERNELS`` lists for it (the composite keys,
    and the backend keys of the graph's tensors, that it had a kernel for at
    capture). A module in ``CUSTOM_OP_OVERRIDE_MODULES``, one whose kernel the
    graph ran at capture overrode another kernel for the same dispatch key, must
    import, and that kernel must still be the active one, registered from the same
    module's file: otherwise the replaced kernel would run in its place. The same
    holds for a module that called ``register_kernel`` on another module's custom
    op, whose kernel must still be the one that call registered. So such a module
    is refused, not warned about, when it raises on import (even after
    registering), when the server imported it under another name, or when
    another module registers the same kernel. Any other module not in
    ``SERVED_MODULES`` only warns when it fails to import and the ops and kernels
    are registered anyway, e.g. by a module the server imported under another
    name.

    Raises ``PrecompileError`` if either file cannot be read, if ``python_code`` is
    not a ``torch.compiler.precompile`` artifact, or if the cache's ``backend``,
    ``tracer`` or ``code_hash`` does not match ``python_code`` -- the pair came from different
    captures -- or if an installed artifact cannot install onto the serving process (for
    example, a captured source changed since capture), or if a custom op the
    artifact calls, or one of its captured kernels, is not registered once its
    imports ran, or if a captured overriding kernel is no longer the active one. A cache whose ``format``/``version`` does not match (a foreign or
    different-build envelope) is NOT fatal: the cache is acceleration only, so
    ``load`` degrades to JIT'ing from ``python_code`` rather than crashing.
    """
    torch._C._log_api_usage_once("torch.compiler.precompile.load")
    python_code, cache = _read_artifact(artifact_path, cache_path)
    return _runnable_from_pair(
        python_code, cache, allow_serve_time_compiles=allow_serve_time_compiles
    )
