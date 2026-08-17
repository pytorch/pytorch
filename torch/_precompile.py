"""Ahead-of-time precompilation (``make_fx`` tracer by default; ``dynamo`` also available).

    python_code, cache = torch.compiler.precompile(fn, model, *example_inputs)
    f_c = torch.compiler.precompile.load(python_code, cache)
    out = f_c(model, *example_inputs)   # pass the model again at runtime

    session = torch.compiler.precompile(
        fn, example_inputs=[(model, x1), (model, x2)]
    )
    session.save("model.pt")

precompile captures your computation with ``make_fx`` (the default ``tracer``) -- a
NON-STRICT trace of the ATen ops that run when ``fn`` executes once on the example
inputs. It does not analyze your Python, so it comes with an explicit contract (the
programming model): stay inside it and the artifact faithfully reproduces ``fn``; step
outside it and you get an artifact that computes the wrong thing.

The ``dynamo`` tracer is an alternative capture front-end that analyzes the Python
(bytecode) rather than tracing one path. It inlines the TRANSFORMED BYTECODE Dynamo
produces into ``python_code`` (marshalled, rehydrated at load) and lowers the compiled
subgraph through the same backends; forward and training computations, ``mark_unbacked``
dynamic shapes, and a ``decompositions`` table all work with it. See the ``tracer`` note at
the bottom of Note [precompile programming model].

For a computation with graph breaks, or to retain several guarded/recompiled variants,
pass a sequence of calls through ``example_inputs=[(...,), (...,)]``. That form returns
an execution-driven session whose ``save()`` writes a package reloaded with
``torch.compiler.precompile.load_package``. ``torch.compiler.precompile.capture`` remains
available when the calls must be made manually, such as a training capture. The positional
form above remains the self-contained source-artifact path and requires one full Dynamo
graph.

Multi-graph capture keeps all live guards while running examples, then filters only the
serialized copy. ``save()`` refuses known coverage gaps, failed captures, and RISKY
dropped guards by default; the stricter ``require_no_dropped_guards`` is off, since
every model drops identity guards that cannot be serialized. Coverage remains
execution-driven: a complete summary describes the calls
that ran, not every possible input or unexecuted branch. Automatic examples run under
ordinary ``torch.no_grad()``, so serve that inference artifact under the same grad mode;
automatic inputs and module parameters/buffers created as inference tensors are rejected.

The positional form returns a self-contained, executable ``python_code`` string plus a
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
#    zero grads as usual. The dynamo tracer reaches the SAME observable behavior by a
#    different route (see the tracer note): the backward stays a traced autograd call
#    inside the graph and Dynamo's own bytecode does the accumulate, so there is no
#    harvested-output list -- but which params get a grad is still fixed at trace time,
#    frozen params still keep ``.grad = None``, and the accumulate still matches eager.
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
# extra outputs and a Python-level scatter in the driver (invariant 5). Dynamo instead
# rewrites the backward into an in-graph torch.autograd.grad plus the .grad updates around
# it -- so precompile pins trace_autograd_ops (otherwise Dynamo graph-breaks on the
# backward), runs the artifact under enable_grad (the traced call differentiates a live
# autograd graph the same call builds; a forward capture keeps no_grad), and lowers with
# compile_to_python(grad_enabled=True) for the same reason. No grad-scatter metadata is
# needed: the transformed bytecode does the .grad update itself, on the runtime model.
# The ONE thing precompile has to correct is that Dynamo's rewrite SPECIALIZES on whether
# p.grad is None at trace time -- a guarded choice in torch.compile, and this artifact
# checks no guards -- so precompile always bakes the ACCUMULATING form and the driver
# materializes a zero .grad where the runtime model has none; see Note [precompile dynamo
# training grad accumulation]. Only params get a harvested grad, as on the make_fx path
# (invariant 5): a user input that requires grad is rejected.
#
# Dynamic shapes and decompositions work here too, by different mechanisms than make_fx.
# Dynamic shapes: mark_unbacked is Dynamo's OWN decorator, so Dynamo captures the marked
# dim as an UNBACKED symint directly -- unguardable, so a graph that needs to guard on it
# fails loudly at capture (the same PrecompileError the make_fx tracer raises) instead of
# baking a size, which is what makes a guard-free artifact sound. Dynamo emits the
# ShapeEnv's runtime asserts (mark_unbacked's min/max, a shared shape_id's equality) into
# the subgraph itself, so they hold on BOTH backends -- unlike the make_fx tracer, whose
# eager backend has no such asserts and therefore rejects dynamic dims outright. The
# STRICT variant is rejected here: Dynamo reads it as a RelaxedUnspecConstraint, i.e. a
# BACKED dynamic dim it may guard on, and this artifact does not check guards (see
# _reject_strict_unbacked_marks); mark_dynamic / specialize_on are rejected as on the
# make_fx path. Decompositions: Dynamo captures torch-level IR and never consults a
# decomposition table, so precompile applies it by re-tracing the captured subgraph with
# make_fx (see _decompose_subgraph) -- the same table shaping the same ATen graph, just
# applied one step later than the make_fx tracer applies it.
#
# Scope and differences from make_fx: the source-artifact call needs one transformed
# bytecode and therefore still requires a single Dynamo graph. A graph-breaking fn raises
# a PrecompileError pointing at ``torch.compiler.precompile.capture``, whose execution-
# driven package preserves every graph-break continuation, guard, and recompiled variant.
# The source-artifact path does NOT check Dynamo's guards at
# runtime, NOR does
# it reproduce the make_fx drivers' upfront runtime validation (the param/buffer structural
# check, invariant 2, and the per-input shape/dtype/device checks, invariants 3/6): safety
# comes from the same specialization contract as make_fx (control flow and unmarked shapes
# are specialized to the example) plus the captured graph's own asserts -- on the INDUCTOR
# backend the baked assert_size_stride (which catches a runtime input/weight whose SHAPE or
# STRIDE differs from the example, but not its DTYPE) and, for a dynamic capture, the
# ShapeEnv range / equality asserts on both backends -- not from a reconstructed guard
# manager. So a contract-violating runtime
# input/model may fail with a raw kernel error rather than a clean PrecompileError, and on
# the EAGER backend (no assert_size_stride) a broadcast-compatible shape mismatch can
# silently miscompute -- pass inputs and a model matching the example, as the contract
# requires. Because Dynamo bakes the trace-time environment (e.g. the current accelerator
# stream) into the bytecode, the artifact is environment-specialized like the make_fx one.
# Unlike the make_fx tracer's rendered source, this artifact inlines MARSHALLED CPython
# bytecode plus a PICKLED state blob, so it is LOCKED to the producing Python version:
# loading it under a different CPython (3.10-3.14) fails with a clean PrecompileError (see
# _build_dynamo_forward). It is ALSO locked to a compatible torch build, because its import
# aliases can reference private torch._dynamo runtime modules (also surfaced as a clean
# PrecompileError). Regenerate per Python version / torch build, or use make_fx for portable
# source (backend='eager' for torch-build portability -- the default make_fx inductor artifact
# itself inlines private torch._inductor modules, so it too is torch-build-locked; the
# Python-version portability holds for either make_fx backend). A tensor closed over by fn (a
# global, captured local, or DEFAULT ARGUMENT value,
# including one nested in a container, an nn.Module, a plain/__slots__ object attribute, or
# a functools.partial / bound method) is rejected (invariant 1) here too -- Dynamo surfaces
# it as a used-global / closure content / argdef, not a graph get_attr constant, so the
# check scans those by replaying pickle's own traversal (see _baked_tensors and
# _reject_baked_tensors).

from __future__ import annotations

import base64
import hashlib
import io
import logging
import marshal
import pickle
import sys
import types
from collections.abc import Callable, Iterator, Mapping, Sequence  # noqa: TC003
from contextlib import AbstractContextManager  # noqa: TC003
from types import MappingProxyType
from typing import Any, cast, NewType, TYPE_CHECKING
from typing_extensions import Self

import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch.compiler._precompile_types import (
    FrameInvariants,
    GuardFact,
    PrecompileSummary,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


log = logging.getLogger(__name__)


if TYPE_CHECKING:
    from torch._dynamo import convert_frame
    from torch._dynamo.source import Source
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


class PrecompiledCallable:
    """Callable handle for one loaded multi-graph precompile artifact."""

    def __init__(self, compiled: Any) -> None:
        self._compiled = compiled

    def _call(self, method: Callable[..., Any], *args: object, **kwargs: object) -> Any:
        from torch._dynamo.exc import PackageError, RecompileError

        try:
            return method(*args, **kwargs)
        except (PackageError, RecompileError) as e:
            raise PrecompileError(str(e)) from e

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self._call(self._compiled, *args, **kwargs)

    def __enter__(self) -> Self:
        self._call(self._compiled.__enter__)
        return self

    def __exit__(self, *exc: object) -> None:
        self._call(self._compiled.__exit__, *exc)

    def unload(self) -> None:
        """Remove everything :func:`load_package` installed for this artifact.

        The precompiled entries come off the code objects and the globals the
        artifact wrote come out of their modules, so the model recompiles
        normally afterwards. Exiting this object as a context manager does the
        same thing; call it directly when the artifact's lifetime is not
        lexically scoped. Unloading twice is harmless, and a call already in
        flight on another thread is allowed to finish first.
        """
        self._call(self._compiled.unload)

    @property
    def _package(self) -> Any:
        return self._compiled._package


class PrecompileSession:
    r"""Execution-driven multi-graph capture returned by :func:`precompile`."""

    def __init__(self, session: Any) -> None:
        self._session = session

    def _call(self, method: Callable[..., Any], *args: object, **kwargs: object) -> Any:
        from torch._dynamo.exc import PackageError, RecompileError

        try:
            return method(*args, **kwargs)
        except (PackageError, RecompileError) as e:
            raise PrecompileError(str(e)) from e

    def __enter__(self) -> Callable[..., object]:
        compiled = self._call(self._session.__enter__)

        def call(*args: object, **kwargs: object) -> object:
            return self._call(compiled, *args, **kwargs)

        return call

    def __exit__(self, *exc: object) -> None:
        self._call(self._session.__exit__, *exc)

    def invariants(self) -> tuple[FrameInvariants, ...]:
        r"""invariants() -> tuple

        Return the guards that held across every captured variant of each frame.
        """
        return self._call(self._session.invariants)

    def write_invariants(self, path: str) -> None:
        r"""write_invariants(path) -> None

        Write :meth:`invariants` to ``path`` in a stable, human-readable form.

        Args:
            path (str): destination file. Parent directories are created as needed.
        """
        self._call(self._session.write_invariants, path)

    def summary(self) -> PrecompileSummary:
        r"""summary() -> PrecompileSummary

        Return observed capture coverage, recompilation, failure, and guard information.

        ``complete`` covers only calls that ran during capture; it cannot account for an
        unexecuted path through the callable.
        """
        return self._call(self._session.summary)

    def save(
        self,
        path: str,
        *,
        require_complete: bool = True,
        require_no_risky_drops: bool = True,
        require_no_dropped_guards: bool = False,
    ) -> PrecompileSummary:
        r"""save(path, *, require_complete=True, require_no_risky_drops=True, require_no_dropped_guards=False) -> PrecompileSummary

        Write the captured package to ``path`` and return its summary.

        Args:
            path (str): destination artifact file.
            require_complete (bool, optional): reject uncovered, bypassed, or truncated
              frames and captures that raised. Default: ``True``.
            require_no_risky_drops (bool, optional): reject dropped guards from a custom
              filter, guards observed to distinguish variants, and identity guards on
              configurable slots. Default: ``True``.
            require_no_dropped_guards (bool, optional): reject EVERY guard omitted
              from the serialized artifact, not just the ones the lint calls
              load-bearing. Default: ``False``, because every model drops the
              identity guards precompile cannot serialize -- a plain
              ``nn.Module`` capture drops torch-owned ``MODULE_MATCH`` and
              ``BUILTIN_MATCH`` guards -- so ``True`` refuses essentially every
              real artifact. ``require_no_risky_drops`` is the rail that is on
              by default; set this one as well only if you want the strictest
              possible check and are prepared for it to refuse.

        Returns:
            PrecompileSummary: capture coverage and guard information.

        .. warning::
            Only ``require_no_risky_drops`` is on by default, and the risky subset is a
            lint, not a proof. Audit :meth:`summary` before relying on that default, and
            before relaxing either requirement.
        """
        return self._call(
            self._session.save,
            path,
            require_complete=require_complete,
            require_no_risky_drops=require_no_risky_drops,
            require_no_dropped_guards=require_no_dropped_guards,
        )


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


def _reject_strict_unbacked_marks(user_flat: list[object]) -> None:
    """Reject ``mark_unbacked(..., strict=True)`` for the dynamo tracer (invariant 3).

    The strict variant means something DIFFERENT to Dynamo than it does to the make_fx
    tracer. make_fx capture is precompile's own (_fakeify_with_unbacked), so it makes a
    strict dim unbacked exactly like a plain one; Dynamo instead records only a
    RelaxedUnspecConstraint, making the dim a BACKED dynamic symbol -- one Dynamo may guard
    on (a shape-dependent branch just adds ``s0 > 4``) and errors on only if the dim gets
    fully specialized. The dynamo artifact does not check Dynamo's guards at runtime, so
    such a guard would be silently dropped and a different runtime size would take the
    wrong path. Plain mark_unbacked gives a genuinely unguardable dim, which is what the
    guard-free artifact needs, so point there.
    """
    if any(
        isinstance(t, torch.Tensor)
        and getattr(t, "_dynamo_strict_unbacked_indices", None)
        for t in user_flat
    ):
        raise PrecompileError(
            "precompile: an input has a mark_unbacked(strict=True) dim, which "
            "tracer='dynamo' cannot honor: Dynamo captures a strict mark as a BACKED "
            "dynamic dim it may guard on, and the dynamo artifact does not check Dynamo's "
            "guards at runtime. Use plain mark_unbacked (strict=False), which captures a "
            "genuinely unguardable dim, or use tracer='make_fx'."
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


def _reject_nonliftable_get_attrs(gm: torch.fx.GraphModule) -> None:
    """Reject a captured dynamo subgraph get_attr whose target is neither a Tensor nor a
    GraphModule (a symbolic size, a torchbind script object, or a bare nn.Module). The
    eager backend inlines gm.code against an EMPTY _GraphSelf(), so such a get_attr would
    resolve to nothing and raise a raw AttributeError at runtime; reject it at capture
    with a concrete reason instead. Tensor and GraphModule get_attrs are already handled
    by _check_no_constant_tensors / _assert_no_control_flow_subgraphs.
    """
    offending = [
        (target, type(attr).__name__)
        for target, attr in _resolved_get_attrs(gm)
        if not isinstance(attr, (torch.Tensor, torch.fx.GraphModule))
    ]
    if offending:
        raise PrecompileError(
            "precompile tracer='dynamo' captured a subgraph get_attr that is not a "
            "liftable graph input (e.g. a symbolic size, a torchbind script object, or a "
            "bare nn.Module); the eager backend inlines the subgraph and cannot resolve "
            f"it. Offending (target, type): {offending}. Pass such state as an explicit "
            "argument, or use tracer='make_fx'."
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


def _baked_tensors(value: object) -> list[torch.Tensor]:
    """Return the tensors pickle would embed BY VALUE when serializing ``value`` -- exactly
    the tensors a captured global / closure / default holding ``value`` bakes into the
    artifact's _DYNAMO_STATE. Rather than reimplement pickle's traversal (and mis-handle
    __slots__, functools.partial, bound methods, custom __reduce__, dict keys, protocol
    quirks, ...), this RUNS pickle's own traversal with a persistent_id hook that records
    every Tensor pickle is about to serialize by value and short-circuits it (so its storage
    is never written). Consequences fall out for free: a tensor in a container, an nn.Module,
    a plain / __slots__ / custom-__reduce__ object, or a bound method's ``__self__`` is
    caught, while a by-reference object (a function, class, or module -- re-imported at load,
    nothing baked) contributes nothing. The hook fires on isinstance(Tensor) BEFORE the
    object's own reducer, so a tensor SUBCLASS that opts into a by-reference __reduce__ is
    conservatively (and harmlessly) rejected too -- an accepted tradeoff for not executing
    arbitrary reducer code. If ``value`` is not fully picklable, returns the tensors found
    BEFORE the failure (so a tensor preceding an unpicklable object still yields the precise
    invariant-1 error; an all-unpicklable value returns [] and the real pickle of the
    artifact state fails loudly on its own). Mirrors the make_fx get_attr constant scan.
    """
    found: list[torch.Tensor] = []

    class _TensorFinder(pickle.Pickler):
        def persistent_id(self, obj: object) -> object:
            if isinstance(obj, torch.Tensor):
                found.append(obj)
                return len(found)
            return None

    try:
        _TensorFinder(io.BytesIO(), protocol=pickle.DEFAULT_PROTOCOL).dump(value)
    except Exception:
        pass
    return found


def _reject_baked_tensors(
    used_globals: Mapping[str, object],
    closure_contents: list[object],
    defaults: Sequence[object] = (),
) -> None:
    """Enforce invariant 1 for the dynamo tracer: reject a live tensor referenced from a
    module global, a closure cell, or a default argument value, which would be embedded
    by VALUE into the artifact rather than passed at runtime.

    This is the dynamo tracer's analogue of _check_no_constant_tensors (the make_fx
    tracer's get_attr scan): Dynamo surfaces a closed-over tensor as a used-global or a
    closure content (a global tensor becomes a graph input sourced from its GlobalSource,
    which get_runtime_env records in used_globals -- as the whole container when the
    tensor is nested), and fn's argdefs + kwdefaults are pickled into _DYNAMO_STATE too, so
    a tensor default (``def f(m, x, w=W)``) is baked as well. Each source is scanned with
    _baked_tensors, which uses pickle's own traversal to find exactly the tensors that get
    baked (see _baked_tensors for what that covers).
    """
    offending = []
    for name, v in used_globals.items():
        offending += [(name, tuple(t.shape), str(t.dtype)) for t in _baked_tensors(v)]
    for v in closure_contents:
        offending += [
            ("<closure>", tuple(t.shape), str(t.dtype)) for t in _baked_tensors(v)
        ]
    for v in defaults:
        offending += [
            ("<default arg>", tuple(t.shape), str(t.dtype)) for t in _baked_tensors(v)
        ]
    if offending:
        raise PrecompileError(
            "precompile traced a tensor that is neither a graph input (module "
            "parameter/buffer or user input) nor an intermediate: it is closed over by "
            "fn (a global, captured local, or default argument value, including one "
            "nested in a container, an nn.Module, or a plain object attribute) and would "
            "be hard-coded into the artifact. Offending constants (name, shape, dtype): "
            f"{offending}. Fix by passing the tensor as an explicit argument; for module "
            "state register it as a parameter/buffer, and pass the owning nn.Module as an "
            "explicit argument rather than closing over it."
        )


class _DynamoCapture:
    """The pieces the dynamo tracer produces: the transformed bytecode + the names it
    references (import aliases, plain globals, closure, arg/kw defaults, the compiled
    subgraph's backend id), plus the subgraph GraphModule + the example inputs for the
    backend to lower. ``backend_id`` / ``gm`` are None when fn produced no tensor compute
    (the transformed bytecode is then the whole artifact). ``dynamic`` records whether the
    subgraph was captured with dynamic (mark_unbacked) dims, in which case
    ``example_inputs`` are Dynamo's own symbolic FAKE tensors rather than the real ones.
    ``trains`` marks a capture that performs autograd (the graph carries a traced
    ``torch.autograd.grad``), and ``grad_accum_params`` then names every param the graph
    accumulates a gradient into, as ``(positional arg index of its module, param name)``;
    see _seed_grad_targets and the emitted driver."""

    def __init__(
        self,
        *,
        bytecode: types.CodeType,
        import_sources: dict[str, str],
        used_globals: dict[str, object],
        closure_contents: list[object],
        argdefs: tuple[object, ...] | None,
        kwdefaults: dict[str, object] | None,
        backend_id: str | None,
        gm: torch.fx.GraphModule | None,
        example_inputs: Sequence[object],
        dynamic: bool = False,
        trains: bool = False,
        grad_accum_params: list[tuple[int, list[tuple[str, object]], str]]
        | None = None,
    ) -> None:
        self.bytecode = bytecode
        self.import_sources = import_sources
        self.used_globals = used_globals
        self.closure_contents = closure_contents
        self.argdefs = argdefs
        self.kwdefaults = kwdefaults
        self.backend_id = backend_id
        self.gm = gm
        self.example_inputs = example_inputs
        self.dynamic = dynamic
        self.trains = trains
        self.grad_accum_params = grad_accum_params or []


def _graph_traces_autograd(gm: torch.fx.GraphModule) -> bool:
    """True if the captured dynamo subgraph performs autograd itself -- i.e. fn ran a
    backward and Dynamo (with trace_autograd_ops) rewrote it into an in-graph
    ``torch.autograd.grad`` call plus the ``.grad`` updates around it.

    Both ``Tensor.backward()`` and a direct ``torch.autograd.grad(...)`` funnel through
    that one target, so this single check identifies a training capture. Such a graph is
    the reason the training path needs grad ENABLED where an inference capture runs under
    no_grad (invariant 5): the traced call differentiates a live autograd graph.
    """
    return any(
        node.op == "call_function" and node.target is torch.autograd.grad
        for node in gm.graph.nodes
    )


# How deep to look for an nn.Module inside a frame argument. A path is a tuple of
# ("index"|"key"|"attr", accessor) steps; the driver replays exactly these.
_MODULE_SEARCH_DEPTH = 6
# Objects walked PER FRAME ARGUMENT, so an argument that happens to reference a
# large object graph cannot dominate capture time. Per argument rather than one
# counter shared by the frame: a shared one let a big EARLIER argument (a vocab,
# an in-memory dataset) exhaust the walk before it reached the model in a later
# argument, so whether the artifact was correct depended on the argument ORDER.
# The walk below stops expanding as soon as the budget is gone, so the cost is
# O(budget) rather than O(size of the argument), which is what makes a budget
# this large cheaper on a pathological argument than the old 2000 was.
_MODULE_SEARCH_BUDGET = 20000
_ModulePath = tuple[tuple[str, object], ...]


def _walk_for_modules(
    value: object, budget: list[int]
) -> list[tuple[_ModulePath, torch.nn.Module]]:
    """Find every nn.Module reachable from ``value``, each with the PATH to it.

    A path rather than just a position, because two same-shaped modules in one
    container are indistinguishable by parameter name: re-searching by name
    found the first one for both, left the second unseeded, and stamped the
    first one's grads twice.

    The walk starts over for each ARGUMENT (a fresh ``seen`` set and a fresh
    budget, see _frame_modules), never once for the whole frame: one module
    reachable from two arguments has to be recorded at BOTH paths, because the
    runtime call may put a different object at each and the driver replays each
    path independently. Sharing the set recorded only the first and then crashed
    on the second.

    A module found is not descended into: its own parameters come from
    named_parameters(), and its children are reached through it.

    BREADTH first, so the budget cuts off the deepest layer rather than whatever
    the first attribute happened to lead into. A trainer holding both a big
    ``self.dataset`` and ``self.model`` is the ordinary shape, and depth first
    opened the dataset and never reached the model -- silently, since a training
    capture with nothing to seed bakes the overwriting form of the backward.
    """
    found: list[tuple[_ModulePath, torch.nn.Module]] = []
    seen: set[int] = {id(value)}
    frontier: list[tuple[_ModulePath, object]] = [((), value)]
    for depth in range(_MODULE_SEARCH_DEPTH + 1):
        if not frontier:
            break
        nxt: list[tuple[_ModulePath, object]] = []
        for path, obj in frontier:
            if budget[0] <= 0:
                break
            budget[0] -= 1
            if isinstance(obj, torch.nn.Module):
                found.append((path, obj))
                continue
            # Never walk into a module object: it is a namespace, not model
            # state, and `torch` alone reaches ~48k objects.
            if isinstance(obj, (types.ModuleType, type)):
                continue
            if depth == _MODULE_SEARCH_DEPTH:
                continue
            edges: Iterator[tuple[tuple[str, object], object]]
            if isinstance(obj, (list, tuple)):
                edges = ((("index", i), v) for i, v in enumerate(obj))
            elif isinstance(obj, dict):
                # Any hashable key, not just str: {0: model} is an ordinary
                # shape. A key the artifact cannot write down is refused later,
                # by _param_grad_inputs, and only if a model is actually found
                # behind it (_representable_as_source).
                edges = ((("key", k), v) for k, v in obj.items())
            else:
                edges = ((("attr", n), v) for n, v in _attribute_edges(obj))
            for step, inner in edges:
                # Stop EXPANDING once the frontier already covers the remaining
                # budget: without this the walk still pays a full iteration of a
                # 300k-element argument to enqueue objects it will never look at.
                if len(nxt) >= budget[0]:
                    break
                if id(inner) in seen:
                    continue
                seen.add(id(inner))
                nxt.append((path + (step,), inner))
        frontier = nxt
    return found


def _attribute_edges(value: object) -> Iterator[tuple[str, object]]:
    """(name, value) for each instance attribute, lazily -- the walk stops
    consuming this as soon as its budget is gone."""
    for name in _attribute_names(value):
        try:
            yield name, getattr(value, name)
        except Exception:
            # A property can raise; it is not somewhere a model lives.
            continue


def _attribute_names(value: object) -> list[str]:
    """Instance attribute names, covering __slots__ as well as __dict__.

    A __slots__ holder has no __dict__ at all, so a vars()-only walk missed the
    model entirely -- and missing it is silent, because a training capture with
    nothing to seed bakes the assign form.
    """
    names = list(getattr(value, "__dict__", {}) or {})
    for klass in type(value).__mro__:
        slots = klass.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        names.extend(s for s in slots if s != "__dict__")
    return names


def _frame_modules(
    fn: Callable[..., object], args: tuple[object, ...]
) -> list[tuple[int, _ModulePath, torch.nn.Module]]:
    """Every nn.Module the artifact's ``forward`` will receive, positioned.

    Positions are over the FRAME args, not the caller's ``args``: Dynamo traces
    ``get_traced_fn(fn)`` and ``convert_frame._get_frame`` PREPENDS the bound
    self, so for an nn.Module ``fn`` -- or a bound method -- the module carrying
    the parameters is argument 0 and is absent from ``args`` entirely.
    """
    from torch._dynamo.convert_frame import get_traced_fn

    _traced, bound_self = get_traced_fn(fn)
    frame_args: list[object] = ([bound_self] if bound_self is not None else []) + list(
        args
    )
    found = []
    truncated = False
    for pos, a in enumerate(frame_args):
        budget = [_MODULE_SEARCH_BUDGET]
        for path, module in _walk_for_modules(a, budget):
            found.append((pos, path, module))
        truncated = truncated or budget[0] <= 0
    if truncated and not found:
        # Only when the search came back EMPTY and we know it was cut short:
        # that is the combination that silently bakes the overwriting form of a
        # backward (this walk runs only for a training capture). Having found
        # some module but missed another is not detectable here, and is what the
        # per-tensor attribution check at the end of _capture_dynamo is for.
        log.warning(
            "precompile: no nn.Module was found in fn's arguments, and the search "
            "for one stopped early after %d objects in at least one argument. A "
            "training capture that finds no model cannot bake the accumulating form "
            "of its backward. Pass the model as its own argument, or hold it in a "
            "shallower container.",
            _MODULE_SEARCH_BUDGET,
        )
    return found


def _module_import_name(module: types.ModuleType) -> str | None:
    """The sys.modules KEY for ``module``, which is what re-imports to it.

    ``__name__`` is not: ``_collections_abc`` sets its own to "collections.abc",
    which imports to the three-line shim instead. Real models inline through
    such modules, so resolve the key and refuse rather than record a name that
    binds something else at load.
    """
    name = getattr(module, "__name__", None)
    if name is not None and sys.modules.get(name) is module:
        return name
    for key, candidate in list(sys.modules.items()):
        if candidate is module:
            return key
    return None


def _representable_as_source(key: object) -> bool:
    """Whether ``key`` survives the artifact, which stores it as SOURCE TEXT.

    GRAD_ACCUM_PARAMS is emitted with repr() and read back by ast.literal_eval
    (load) and exec (the driver), so a key is usable exactly when its repr is a
    literal that reads back EQUAL -- equality (and matching hash) rather than
    identity because that is all the driver's ``obj[key]`` lookup needs. An enum
    member reprs as ``<K.A: 1>``, which is not even parseable: the capture
    succeeded and load() then rejected the whole file as "not a precompile
    artifact".
    """
    import ast

    try:
        rebuilt = ast.literal_eval(repr(key))
    except Exception:
        return False
    try:
        return bool(rebuilt == key) and hash(rebuilt) == hash(key)
    except Exception:
        # __eq__ / __hash__ are user code and can raise or return a non-bool
        # (a Tensor key); either way the key is not one we can write down.
        return False


def _grad_target_inputs(gm: torch.fx.GraphModule, n_inputs: int) -> set[int]:
    """Graph-input indices of the tensors the traced backward differentiates.

    Dynamo lowers both ``Tensor.backward()`` and ``torch.autograd.grad`` to ONE in-graph
    ``torch.autograd.grad(loss, inputs)`` node, so that node's ``inputs`` list is exactly
    the set of leaves that can receive a ``.grad``. An unrecognized shape falls back to
    every input: seeding more can only turn a silent wrong answer into a refusal.
    """
    phs = list(gm.graph.find_nodes(op="placeholder"))
    if len(phs) != n_inputs:
        raise PrecompileError(
            f"internal: the captured dynamo subgraph has {len(phs)} placeholders but "
            f"{n_inputs} example inputs, so precompile cannot tell which tensors its "
            "backward differentiates."
        )
    idx = {id(n): i for i, n in enumerate(phs)}
    targets: set[int] = set()
    saw = False
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target is torch.autograd.grad:
            saw = True
            inputs = node.args[1] if len(node.args) > 1 else node.kwargs.get("inputs")
            if not isinstance(inputs, (list, tuple)):
                return set(range(len(phs)))
            targets.update(idx[id(v)] for v in inputs if id(v) in idx)
    return targets if saw else set()


def _seed_grad_targets(
    gm: torch.fx.GraphModule, example_inputs: Sequence[object]
) -> list[torch.Tensor]:
    """Give every tensor the traced backward differentiates a zero ``.grad`` if it has
    none, so a re-capture bakes the ACCUMULATING form. Returns what was seeded.

    Note [precompile dynamo training grad accumulation]
    Dynamo's backward rewrite SPECIALIZES on whether ``p.grad`` is None at trace time: with
    no grad it emits ``new = empty_like(p); new.copy_(g)`` and the bytecode ASSIGNS it; with
    a grad present it additionally emits ``p.grad.add_(new)``. torch.compile protects that
    choice with a guard -- the precompile artifact checks no guards, so baking the assign
    form silently OVERWRITES on the second call of a training loop where eager accumulates
    (``p.grad += g``). So precompile always bakes the ACCUMULATE form and has the driver
    materialize a zero ``.grad`` for a param the runtime model left at None; zero + accum
    equals eager's assign on the first step, and equals eager on every step after.

    Seeding is driven by the GRAPH, not by the module walk, because the walk is the thing
    that can miss a param -- and a param the walk missed is precisely the one that would
    silently bake the assign form. Seeding it makes its ``.grad`` a graph INPUT, which is
    what the attribution check in _param_grad_inputs can see and refuse. The seed is
    therefore not just a fixup, it is the ORACLE: whether the re-capture still bakes the
    assign form is exactly how precompile learns that fn nulls that ``.grad`` itself
    (``zero_grad(set_to_none=True)``, ``p.grad = None``), in which case the assign form is
    what eager does too. Keyed on requires_grad rather than isinstance(Parameter): a
    requires_grad BUFFER or a bare tensor attribute is a legal autograd target that
    named_parameters() can never name, and it has to be caught, not skipped.
    """
    seeded: list[torch.Tensor] = []
    try:
        for i in sorted(_grad_target_inputs(gm, len(example_inputs))):
            t = example_inputs[i]
            if (
                isinstance(t, torch.Tensor)
                and t.requires_grad
                and t.is_leaf
                and t.grad is None
            ):
                t.grad = torch.zeros_like(t)
                seeded.append(t)
    except BaseException:
        # Seeding mutates the CALLER's model, so a failure part way through --
        # an OOM on zeros_like for a large model -- must not leave grads behind.
        for t in seeded:
            t.grad = None
        raise
    return seeded


def _dot_grad_graph_inputs(
    example_inputs: Sequence[object], sources: Mapping[int, Source]
) -> dict[int, str]:
    """``{graph-input index: access path}`` for every input that is some tensor's ``.grad``.

    Two detectors, unioned, because each fails open on its own and refusing is the safe
    direction: the SOURCE label is precise and names a path for the error message but is
    Dynamo's spelling and could be renamed; IDENTITY cannot be renamed but cannot see a
    ``.grad`` whose owner the forward never read.
    """
    from torch._dynamo.source import AttrSource, GradSource

    found: dict[int, str] = {}
    for i, src in (sources or {}).items():
        if isinstance(src, GradSource) or (
            isinstance(src, AttrSource) and src.member == "grad"
        ):
            found[i] = src.name
    owners = {
        id(t.grad)
        for t in example_inputs
        if isinstance(t, torch.Tensor) and t.grad is not None
    }
    for i, t in enumerate(example_inputs):
        if isinstance(t, torch.Tensor) and id(t) in owners and i not in found:
            src = (sources or {}).get(i)
            found[i] = src.name if src is not None else f"graph input #{i}"
    return found


def _param_grad_inputs(
    fn: Callable[..., object],
    args: tuple[object, ...],
    capture_output: convert_frame.CaptureOutput,
) -> tuple[list[tuple[int, list[tuple[str, object]], str]], list[str], bool]:
    """``(grad_accum_params, unattributed, any_grad_inputs)``.

    ``grad_accum_params`` names every param whose ``.grad`` the graph took as an INPUT, by
    tensor IDENTITY (not by parsing Dynamo's mangled placeholder names), as ``(frame arg
    index, path from that arg to the owning module, param name)``. ``unattributed`` is
    every OTHER ``.grad`` graph input: the artifact will pass it to the graph, the only
    mechanism for guaranteeing it exists at runtime is a GRAD_ACCUM_PARAMS entry, and we
    could not write one -- so the caller refuses. Same loop builds both, so the check and
    its remedy cannot drift apart.
    """
    bi = capture_output.backend_input
    example_inputs = list(bi.example_inputs) if bi is not None else []
    sources = getattr(
        capture_output.graph_capture_output.output_graph.export_metadata,
        "graph_input_idx_to_local_source",
        {},
    )
    by_id = {id(t) for t in example_inputs if isinstance(t, torch.Tensor)}
    found = []
    covered = set()
    for pos, path, module in _frame_modules(fn, args):
        for name, p in module.named_parameters(remove_duplicate=False):
            if p.grad is not None and id(p.grad) in by_id:
                # Checked here, on the paths actually EMITTED, rather than in
                # the walk: a dict keyed by something unwritable is only a
                # problem when a trained model turns out to live behind it.
                for kind, acc in path:
                    if kind == "key" and not _representable_as_source(acc):
                        raise PrecompileError(
                            "precompile tracer='dynamo': fn's argument "
                            f"{pos} reaches the nn.Module owning parameter "
                            f"{name!r} through a dict keyed by {acc!r} (a "
                            f"{type(acc).__name__}). The artifact records that "
                            "path as source text, so the key has to be one whose "
                            "repr() is a Python literal that reads back equal -- "
                            "a str, int, bool, float, bytes, None, or a tuple of "
                            "those. Key the dict by the module's name instead, or "
                            "hold the modules in a list or an attribute."
                        )
                found.append((pos, list(path), name))
                covered.add(id(p.grad))
    grad_inputs = _dot_grad_graph_inputs(example_inputs, sources)
    unattributed = [
        # the OWNER is what the user has to make reachable, so drop the ".grad" tail
        path.removesuffix(".grad")
        for i, path in sorted(grad_inputs.items())
        if id(example_inputs[i]) not in covered
    ]
    return found, unattributed, bool(grad_inputs)


def _decompose_subgraph(
    gm: torch.fx.GraphModule,
    fake_inputs: list[object],
    decompositions: dict,
    trains: bool = False,
) -> torch.fx.GraphModule:
    """Apply ``decompositions`` to a captured dynamo subgraph by re-tracing it with make_fx.

    Dynamo captures torch-level IR and never consults a decomposition table, so the dynamo
    tracer honors ``decompositions`` the only way it can: make_fx re-traces the captured
    subgraph WITH the table, which is the same thing the make_fx tracer does during its own
    capture (the table shapes the captured ATen graph, and the backend then lowers /
    inlines that graph). The re-trace runs on Dynamo's own FAKE placeholder values, never
    the real example tensors, so it costs no real compute, cannot mutate the caller's
    model, and preserves the (possibly unbacked-symbolic) input shapes. Placeholder count
    and order are unchanged (fx's DCE never drops a placeholder), so the transformed
    bytecode still calls the subgraph with exactly the same arguments.
    """
    from torch._dispatch.python import enable_python_dispatcher
    from torch._dynamo.utils import detect_fake_mode
    from torch._dynamo.variables.torch_function import (
        torch_function_mode_stack_state_mgr,
    )

    # Take the mode off the fakes THEMSELVES rather than from BackendInput.fake_mode: the
    # two are distinct FakeTensorMode objects (they share a ShapeEnv), and running the ops
    # under a mode other than the one that made the tensors raises "Mixing fake modes NYI".
    fake_mode = detect_fake_mode(fake_inputs)
    if fake_mode is None:
        raise PrecompileError(
            "internal: the captured dynamo subgraph has no fake placeholder values, so "
            "the decompositions table cannot be applied by re-tracing it."
        )
    # tracing_mode="real" hands the fake inputs to the trace UNTOUCHED -- the "symbolic" /
    # "fake" modes would refakeify them under a fresh source and allocate new symbols,
    # losing Dynamo's unbacked ones. The ambient fake mode is what actually executes the
    # ops and the python dispatcher is what lets a symbolic-shaped trace reach the meta
    # kernels; this is how AOTAutograd itself re-traces a graph with a decomposition table.
    #
    # A TRAINING graph is re-traced under enable_grad: its ``torch.autograd.grad`` needs a
    # live autograd graph to differentiate. Re-tracing then RUNS the autograd engine on the
    # fakes, so the backward lands in the new graph as ordinary ATen ops (the traced call
    # itself disappears) -- which is exactly the shape the make_fx tracer produces, and why
    # the caller re-derives ``trains`` from the returned graph rather than reusing its own.
    grad_cm = torch.enable_grad() if trains else torch.no_grad()
    # With the caller's torch_function modes CLEARED. gm is torch-level Python that
    # Dynamo already produced WITH those modes applied symbolically, so re-tracing it
    # while they are live applies each of them a second time and bakes a
    # doubly-transformed graph -- the same trap as the inductor lowering below, and
    # just as silent, since the artifact then needs no mode to reproduce the wrong
    # number.
    with (
        torch_function_mode_stack_state_mgr,
        grad_cm,
        fake_mode,
        enable_python_dispatcher(),
    ):
        return make_fx(gm, decomposition_table=decompositions)(*fake_inputs)


def _capture_dynamo(
    fn: Callable[..., object],
    args: tuple[object, ...],
    decompositions: dict | None = None,
) -> _DynamoCapture:
    """Capture ``fn(*args)`` with Dynamo (tracer="dynamo").

    Unlike the make_fx tracer (a non-strict trace of the single path the example takes),
    Dynamo analyzes the Python bytecode and emits a TRANSFORMED bytecode: it extracts the
    model's params/buffers, calls a compiled subgraph, and reassembles fn's output, plus
    the subgraph (an fx GraphModule) for the backend to lower. precompile inlines that
    transformed bytecode into the artifact (marshalled) and lowers the subgraph exactly
    the way the make_fx inductor path does. See Note [precompile programming model].

    Dynamic shapes are opt-in via ``mark_unbacked``, which Dynamo honors natively: a marked
    dim becomes an UNBACKED symint it cannot guard on, so the subgraph stays symbolic and
    the artifact serves any runtime size of that dim. ``decompositions`` is applied by
    re-tracing the captured subgraph (see _decompose_subgraph).

    A TRAINING step (fn runs ``.backward()`` or ``torch.autograd.grad``) is captured too:
    precompile pins ``trace_autograd_ops`` so Dynamo rewrites the backward into an in-graph
    ``torch.autograd.grad`` rather than graph-breaking, then CAPTURES TWICE when needed so
    the accumulating form of the ``.grad`` update is the one baked (Note [precompile dynamo
    training grad accumulation]). Other graph-breaking Python still raises a
    PrecompileError pointing at ``torch.compiler.precompile.capture``.
    """
    # get_traced_fn refuses anything that is not a function, bound method or
    # nn.Module -- a functools.partial, a callable object -- and Dynamo reaches
    # it deep inside fullgraph_capture, where the raw RuntimeError escapes the
    # handler below. Check up front so the caller gets a clean error naming the
    # shapes that do work.
    from torch._dynamo.convert_frame import get_traced_fn as _get_traced_fn

    try:
        _get_traced_fn(fn)
    except RuntimeError as e:
        raise PrecompileError(
            f"precompile tracer='dynamo' cannot capture a {type(fn).__name__}: Dynamo "
            f"traces a function, a bound method or an nn.Module. Pass partial.func with "
            f"its arguments as explicit arguments, or the object's __call__, or use "
            f"tracer='make_fx'. Underlying: {e}"
        ) from e

    from torch._dynamo import config as dynamo_config, convert_frame
    from torch._dynamo.exc import UncapturedHigherOrderOpError, Unsupported, UserError
    from torch._dynamo.utils import dynamo_timed, get_metrics_context
    from torch._dynamo.variables.torch_function import (
        torch_function_mode_stack_state_mgr,
    )
    from torch._functorch._aot_autograd.to_standalone_python import (
        _graph_has_dynamic_shapes,
    )
    from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode

    args = tuple(args)

    def _run_capture() -> convert_frame.CaptureOutput:
        with (
            # Pin static-shape capture so an UNMARKED dim specializes to the example like
            # the make_fx tracer (invariant 3), independent of the ambient torch._dynamo
            # config AND of per-code-object shape history: assume_static_by_default guards the
            # config default, and automatic_dynamic_shapes=False stops Dynamo promoting a dim
            # to dynamic when the SAME fn was already precompiled at another shape this process
            # (both otherwise yield an out-of-contract DYNAMIC artifact here). Neither affects
            # a mark_unbacked dim: an explicit mark takes precedence and still captures
            # unbacked (the marks precompile cannot honor are rejected upstream). The same
            # automatic_dynamic pin is what makes the training RE-capture below safe: a
            # second capture of the same code object cannot promote a dim to dynamic.
            #
            # trace_autograd_ops lets Dynamo trace a backward (rewriting it into an in-graph
            # torch.autograd.grad) instead of graph-breaking on it, which is what extends the
            # dynamo tracer to training steps. It is inert for an inference fn -- it gates
            # only the Tensor.backward / autograd.grad handlers -- so it is pinned ON here
            # rather than left to the caller to discover.
            dynamo_config.patch(
                assume_static_by_default=True,
                automatic_dynamic_shapes=False,
                trace_autograd_ops=True,
            ),
            get_metrics_context(),
            dynamo_timed("precompile_dynamo_capture"),
            torch_function_mode_stack_state_mgr,
        ):
            return convert_frame.fullgraph_capture(fn, args, {})

    grad_accum_params: list[tuple[int, list[tuple[str, object]], str]] = []
    unattributed_grads: list[str] = []
    try:
        capture_output = _run_capture()
        bi = capture_output.backend_input
        trains = bi is not None and _graph_traces_autograd(bi.graph_module)
        if trains:
            # Training: re-capture with a zero .grad on every tensor the traced backward
            # differentiates that lacks one, so the ``.grad`` update is baked in its
            # ACCUMULATING form (Note [precompile dynamo training grad accumulation]).
            # Seeding needs the first capture's graph to know WHICH tensors those are,
            # hence two passes; a model whose params already carry grads (a warm training
            # loop) needs no seeding and skips the second pass. The seeds are recorded
            # while still attached (that is what _param_grad_inputs matches on) and dropped
            # in the finally -- the captured graph keeps its own reference, so the lowering
            # below still sees them, and the caller's model is left exactly as it was.
            if bi is None:
                raise AssertionError("a training capture always has a backend input")
            first_capture = capture_output
            seeded = _seed_grad_targets(bi.graph_module, bi.example_inputs)
            try:
                if seeded:
                    capture_output = _run_capture()
                grad_accum_params, unattributed_grads, any_grad_inputs = (
                    _param_grad_inputs(fn, args, capture_output)
                )
                if seeded and not any_grad_inputs:
                    # The re-capture bakes no accumulate at all -- fn nulls .grad itself,
                    # or only returns grads -- so the seeds bought nothing and only changed
                    # what fn OBSERVED about .grad. Ship the unseeded capture.
                    capture_output = first_capture
            finally:
                for p in seeded:
                    p.grad = None
    except (Unsupported, UncapturedHigherOrderOpError, UserError) as e:
        # Dynamo could not capture fn as one full graph (a graph break / uncaptured HOP /
        # user error) -- e.g. a Tensor.backward() call. These are exactly the exceptions
        # fullgraph_capture re-raises. Surface a clear PrecompileError instead of the raw
        # dynamo error, and point at the make_fx tracer (which handles training). Guard the
        # first-line extraction: ``"".splitlines()`` is [], so an empty dynamo message would
        # otherwise raise IndexError from inside this very handler.
        #
        # A guard on a mark_unbacked dim arrives wrapped in one of these too (Dynamo
        # converts the GuardOnDataDependentSymNode as it unwinds), and "could not capture
        # fn as a single full graph" would misdiagnose it -- the graph is fine, the MARK is
        # what fn cannot honor. Detect it in the cause chain and use the same message the
        # make_fx tracer raises for the same situation.
        cause: BaseException | None = e
        while cause is not None:
            if isinstance(cause, GuardOnDataDependentSymNode):
                raise _unbacked_guard_error(cause) from e
            cause = cause.__cause__ or cause.__context__
        reason = (str(e).splitlines() or [""])[0]
        raise PrecompileError(
            "precompile tracer='dynamo' could not capture fn as a single full graph "
            f"({reason}). For graph-breaking code, pass example_inputs=[(...,), ...] "
            "to torch.compiler.precompile, or use torch.compiler.precompile.capture "
            "when the calls must be made manually; otherwise use tracer='make_fx' for "
            "a single non-strict trace."
        ) from e
    except GuardOnDataDependentSymNode as e:
        raise _unbacked_guard_error(e) from e

    gco = capture_output.graph_capture_output
    bi = capture_output.backend_input
    runtime_env = gco.get_runtime_env()

    # get_runtime_env records only the globals the transformed bytecode reads as GRAPH
    # INPUTS (plus a builtins fallback). A global the bytecode references elsewhere -- e.g.
    # a plain constant folded straight into the output (``return model(x), SCALE``) -- is in
    # external_refs but NOT (correctly) in used_globals, so without this it would NameError
    # at runtime. Resolve those uncovered refs from the TRACED code's globals and carry them
    # along (baking a module constant is consistent with the specialization contract); a
    # tensor among them is caught by _reject_baked_tensors below (invariant 1). Use
    # gco.f_globals, NOT fn.__globals__: Dynamo traces get_traced_fn(fn) -- fn.forward for an
    # nn.Module, the unwrapped target for a functools.partial -- whose globals live in
    # gco.f_globals (the same dict get_runtime_env built used_globals from). fn.__globals__
    # is {} for a raw nn.Module fn, which would drop the folded-in global and NameError.
    #
    # A real module global always WINS over get_runtime_env's builtin fallback: when a global
    # shadows a builtin name (``sum = [...]``; ``id``, ``type``, ...) get_runtime_env
    # pre-seeds used_globals[ref] with the BUILTIN, so override it with fn_globals[ref] rather
    # than skip on ``ref in used_globals`` -- else the artifact bakes the builtin and silently
    # miscomputes (and a shadowing tensor global would slip past invariant 1). The override is
    # a no-op for a ref already recorded as a graph-input global (used_globals[ref] already
    # equals fn_globals[ref]); skip __builtins_dict__ refs, which get_runtime_env deliberately
    # stores as the _safe_builtins_dict-filtered (picklable) dict rather than the raw one.
    used_globals = dict(runtime_env.used_globals)
    import_sources = dict(runtime_env.import_sources)
    fn_globals = gco.f_globals
    for ref in runtime_env.external_refs:
        if (
            ref in import_sources
            or ref == (bi.backend_id if bi is not None else None)
            or ref.startswith("__builtins_dict__")
        ):
            continue
        if ref not in fn_globals:
            continue
        value = fn_globals[ref]
        if isinstance(value, types.ModuleType):
            # A module is not picklable, so putting it in used_globals fails the
            # capture outright with an unactionable error -- `torch` and `math`
            # both land here whenever the residual bytecode loads the bare name
            # rather than Dynamo's mangled __import_ alias. It is by-reference
            # state, so record it the way the aliases are recorded and let the
            # driver re-import it at load.
            #
            # By sys.modules KEY, not __name__: _collections_abc sets its own
            # __name__ to "collections.abc", which re-imports to a DIFFERENT
            # module. Same trap _defining_module_name documents.
            import_name = _module_import_name(value)
            if import_name is None:
                raise PrecompileError(
                    f"precompile: fn reads the global {ref!r}, which holds module "
                    f"{getattr(value, '__name__', value)!r} that is not in sys.modules, "
                    f"so the artifact cannot re-import it at load. Pass what you need "
                    f"from it as an explicit argument instead."
                )
            import_sources[ref] = import_name
            # get_runtime_env pre-seeds used_globals[ref] with a BUILTIN of the
            # same name when one exists (`import torch as vars`). The driver
            # applies used_globals AFTER IMPORT_SOURCES, so leaving it would let
            # the builtin win and produce a wrong artifact rather than an error.
            used_globals.pop(ref, None)
            continue
        used_globals[ref] = value

    # Invariant 1: reject a tensor closed over by fn (global, captured local, or default
    # argument value, including one nested in a container or nn.Module); Dynamo surfaces
    # it in used_globals / closure / argdefs+kwdefaults rather than as a graph get_attr
    # constant. argdefs/kwdefaults are pickled into _DYNAMO_STATE and restored, so a tensor
    # default is baked too and must be scanned alongside globals/closure.
    closure_contents = [c.cell_contents for c in (runtime_env.closure or ())]
    defaults = [
        *(runtime_env.argdefs or ()),
        *((runtime_env.kwdefaults or {}).values()),
    ]
    _reject_baked_tensors(used_globals, closure_contents, defaults)

    if trains:
        # Only PARAMETERS get a scattered gradient (invariant 5), mirroring the make_fx
        # tracer, which rejects a user input that received one. Here the whole ``.grad``
        # update is inside Dynamo's rewrite, so precompile cannot re-point it -- and a
        # non-parameter leaf would carry the same trace-time ``.grad is None``
        # specialization with no place to correct it. Reject up front rather than bake it.
        offending = [
            tuple(t.shape)
            for t in pytree.tree_leaves(
                tuple(a for a in args if not isinstance(a, torch.nn.Module))
            )
            if isinstance(t, torch.Tensor) and t.requires_grad
        ]
        if offending:
            raise PrecompileError(
                "precompile tracer='dynamo': fn runs a backward and a user input requires "
                f"grad (shapes {offending}); precompile only harvests gradients for module "
                "parameters, so that input's gradient would be baked with a trace-time "
                "assumption about its .grad. Pass the tensor as a module parameter, detach "
                "it (or set requires_grad=False) before the call, or use tracer='make_fx'."
            )

        if unattributed_grads:
            raise PrecompileError(
                "precompile tracer='dynamo': fn's backward accumulates a gradient into "
                f"{len(unattributed_grads)} tensor(s) that precompile cannot re-create at "
                "runtime, because it could not find the nn.Module that owns them among "
                f"fn's arguments: {unattributed_grads}. The artifact bakes "
                "``p.grad.add_(new)``, so it must materialize a zero .grad for each one "
                "before the call, and it can only do that for a parameter it can name; "
                "baking the assign form instead would match eager on the first call and "
                "silently OVERWRITE on every call after (Note [precompile dynamo training "
                "grad accumulation]). precompile searches each argument (including the "
                "bound self of a method or nn.Module fn) through attributes, lists, tuples "
                f"and dicts, at most {_MODULE_SEARCH_DEPTH} steps deep and "
                f"{_MODULE_SEARCH_BUDGET} objects wide, then names parameters with "
                "named_parameters(). Fix by passing the owning module as its own argument; "
                "by moving the model ahead of the large unrelated attribute that exhausted "
                "the search; by registering submodules with nn.ModuleList / nn.ModuleDict "
                "rather than a plain list or dict; and by registering a trainable tensor "
                "with register_parameter rather than as a requires_grad buffer or a bare "
                "attribute. tracer='make_fx' does not help here -- it cannot reach the "
                "tensor either and refuses too."
            )

    gm = bi.graph_module if bi is not None else None
    example_inputs: Sequence[object] = bi.example_inputs if bi is not None else []
    dynamic = False
    if gm is not None:
        # _graph_has_dynamic_shapes is the SAME predicate compile_to_python uses to pick its
        # shapes mode, so reading it here keeps the example inputs we hand it consistent
        # with how it will treat the graph.
        dynamic = _graph_has_dynamic_shapes(gm)
        # Dynamo stashes each placeholder's fake under "example_value". For a DYNAMIC
        # capture those fakes (not bi.example_inputs, which are the REAL tensors) are what
        # the backend must lower: they carry Dynamo's unbacked symbols, whereas refakeifying
        # the reals under the lowering's own TracingContext -- which does not carry Dynamo's
        # symbol cache -- would allocate fresh symbols and silently specialize the artifact
        # to the example sizes. A static capture keeps the real inputs (they refakeify to
        # exactly the same static fakes, and the lowering asks for a fresh ShapeEnv there).
        fake_inputs = [
            n.meta.get("example_value") for n in gm.graph.find_nodes(op="placeholder")
        ]
        if dynamic:
            example_inputs = fake_inputs
        if decompositions is not None:
            gm = _decompose_subgraph(gm, fake_inputs, decompositions, trains)
            # Re-derive: re-tracing a training graph inlines its backward as plain ATen ops,
            # so the result no longer performs autograd itself and must be lowered / run the
            # inference way (grad_accum_params still stands -- the .grad accumulation the
            # inlined backward feeds is unchanged).
            trains = _graph_traces_autograd(gm)
            if dynamic:
                # The re-trace records its placeholders' fakes under make_fx's "val" key.
                example_inputs = [
                    n.meta["val"] for n in gm.graph.find_nodes(op="placeholder")
                ]
        # A get_attr tensor in the subgraph is a baked constant (invariant 1), and a
        # captured control-flow subgraph is not lowerable to standalone source -- reject
        # both, reusing the make_fx tracer's guards on the captured graph. Checked AFTER
        # the decomposition re-trace, so a constant a decomposition introduces is caught too.
        _check_no_constant_tensors(gm)
        _assert_no_control_flow_subgraphs(gm)

    return _DynamoCapture(
        bytecode=runtime_env.bytecode,
        import_sources=import_sources,
        used_globals=used_globals,
        closure_contents=closure_contents,
        argdefs=runtime_env.argdefs,
        kwdefaults=runtime_env.kwdefaults,
        backend_id=bi.backend_id if bi is not None else None,
        gm=gm,
        example_inputs=example_inputs,
        dynamic=dynamic,
        trains=trains,
        grad_accum_params=grad_accum_params,
    )


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
    # Read TRACER first (a top-level literal assignment) to pick the required constant set
    # below: the make_fx and dynamo drivers read different calling-convention literals, so
    # the presence check has to know which set applies. TRACER is absent on artifacts
    # predating the dynamo tracer, so its absence means make_fx.
    tracer = "make_fx"
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "TRACER"
        ):
            try:
                tracer = cast(str, ast.literal_eval(node.value))
            except (ValueError, SyntaxError):
                pass
            break
    if tracer == "dynamo":
        # Validate every top-level literal the dynamo driver reads, including the two opaque
        # blobs, so a truncated / edited artifact missing one fails here with the clean
        # "missing calling-convention metadata" error rather than surfacing a misleading
        # Python-version error from _build_dynamo_forward at exec time.
        wanted = {
            "BACKEND",
            "TRACER",
            "BACKEND_ID",
            "IMPORT_SOURCES",
            "GRAD_ACCUM_PARAMS",
            "_DYNAMO_CODE",
            "_DYNAMO_STATE",
        }
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
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id in wanted:
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


_DYNAMO_GENERATED_HEADER = """\
# Generated by torch.compiler.precompile (tracer="dynamo") -- do not edit.
#
# Self-contained, executable artifact. Unlike the make_fx tracer (which inlines a single
# traced graph), the dynamo tracer analyzes fn's Python and inlines the TRANSFORMED
# BYTECODE Dynamo produced (marshalled, rehydrated by the driver below): it extracts the
# model's params/buffers, calls the compiled subgraph, and reassembles fn's output.
# Provide the model(s) at runtime exactly as fn took them:
#
#     ns = {}
#     exec(open("this_file.py").read(), ns)
#     out = ns["forward"](model, my_input)      # same args as the traced fn
#
# The runtime model must be structurally identical to the traced one (same
# parameter/buffer names, order, and weight tying); only the weight VALUES may differ.
# Dynamo specializes to the example (shapes, control flow), and the environment it was
# traced in (e.g. the accelerator stream) is baked into the bytecode, so this artifact
# is environment-specialized like the make_fx one. See Note [precompile programming
# model] in torch/_precompile.py.
"""


def _build_dynamo_metadata_section(compiled: PrecompiledModule) -> list[str]:
    """Emit the dynamo tracer's calling-convention metadata: the inspectable literals
    (BACKEND / TRACER / BACKEND_ID / IMPORT_SOURCES) plus two opaque blobs -- the
    transformed bytecode (marshalled) and the reconstruction state (the globals it reads,
    its closure contents, and arg/kw defaults, pickled). The driver rehydrates them."""
    capture = compiled._dynamo_capture
    if capture is None:
        raise PrecompileError(
            "internal: cannot build dynamo metadata before _compile()"
        )
    # marshal serializes code objects + basic constants, but NOT arbitrary objects baked
    # as bytecode constants -- e.g. the bound method / type Dynamo emits to reconstruct a
    # collections.namedtuple (or other custom class) output. Surface that as a clear
    # PrecompileError rather than a raw "unmarshallable object" ValueError.
    try:
        code_blob = base64.b64encode(marshal.dumps(capture.bytecode)).decode("ascii")
    except ValueError as e:
        raise PrecompileError(
            "precompile tracer='dynamo' could not serialize fn's transformed bytecode "
            "(it references an unmarshallable constant). This fires when fn returns a "
            "value Dynamo reconstructs via a baked type or bound method, e.g. a "
            "collections.namedtuple or a custom class. Return a plain tuple / list / dict "
            f"of tensors, or use tracer='make_fx'. Underlying: {e}"
        ) from e
    # Only the genuinely non-literal reconstruction state is pickled (the inspectable
    # literals above stay as readable source). A tensor reachable from any of these was
    # already rejected at capture (invariant 1), so what remains is plain globals / closure
    # contents / arg-kw defaults.
    try:
        state_blob = base64.b64encode(
            pickle.dumps(
                {
                    "used_globals": capture.used_globals,
                    "closure": capture.closure_contents,
                    "argdefs": capture.argdefs,
                    "kwdefaults": capture.kwdefaults,
                }
            )
        ).decode("ascii")
    except Exception as e:
        # pickle invokes arbitrary user __reduce__ / __getstate__ code and can raise any
        # exception type (RuntimeError, RecursionError on deep nesting, ...), so catch
        # broadly and relabel -- matching _baked_tensors, which replays the same traversal
        # under an equally broad except. Exception (not BaseException) still lets
        # KeyboardInterrupt / SystemExit propagate.
        raise PrecompileError(
            "precompile tracer='dynamo' could not serialize fn's captured globals / "
            "closure into the artifact (they are not picklable). This fires when fn "
            "closes over an unpicklable object (e.g. a module handle or a local lambda); "
            f"refer to such state through explicit arguments instead. Underlying: {e}"
        ) from e
    return [
        "# " + "=" * 70,
        "# 2. Dynamo calling-convention metadata",
        "# " + "=" * 70,
        # python_code is the single source of truth for the calling convention; load()
        # reads BACKEND / TRACER back out of it (see _parse_artifact_metadata). The two
        # blobs are consumed by the inlined driver, not by load().
        f"BACKEND = {compiled._backend!r}",
        "TRACER = 'dynamo'",
        f"BACKEND_ID = {capture.backend_id!r}",
        f"IMPORT_SOURCES = {capture.import_sources!r}",
        # Training only: (positional arg index of the owning module, param name) for every
        # param the captured backward accumulates a gradient into. Empty for a forward
        # capture. The driver materializes a zero .grad for any of these the runtime model
        # left at None, which is what makes the baked accumulate form match eager on the
        # first step too (Note [precompile dynamo training grad accumulation]).
        f"GRAD_ACCUM_PARAMS = {capture.grad_accum_params!r}",
        # The marshalled bytecode below is CPython-version specific. marshal only
        # REJECTS a foreign blob across the 3.10/3.11 layout change; between 3.11
        # and 3.14 it loads happily and the resulting code object segfaults when
        # called, so the version has to be written down and checked explicitly.
        f"_DYNAMO_PYTHON_VERSION = {tuple(sys.version_info[:2])!r}",
        f"_DYNAMO_CODE = {code_blob!r}",
        f"_DYNAMO_STATE = {state_blob!r}",
        "",
    ]


def _emit_dynamo_eager_subgraph(
    gm: torch.fx.GraphModule, trains: bool = False
) -> list[str]:
    """Inline the captured dynamo subgraph (eager backend) and wrap it in a ``call``
    with the same (boxed) contract the inductor path emits, so the driver is backend-
    agnostic. The dynamo subgraph's ``forward`` takes the graph inputs positionally
    (one per placeholder) and returns a tuple; ``call(flat_inputs)`` splats the list
    into it and normalizes the result to a list.

    A forward capture runs under no_grad. A TRAINING capture runs under enable_grad
    instead: its traced ``torch.autograd.grad`` differentiates an autograd graph that the
    same call has to build, so no_grad would fail it outright ("element 0 of tensors does
    not require grad"). enable_grad rather than "inherit the caller's mode" because the
    artifact captured a training step -- it cannot do anything useful under an ambient
    no_grad, so it pins what it needs (the graph itself disables grad again around the
    ``.grad`` update, exactly as Dynamo traced it).
    """
    from torch.fx.graph import _custom_builtins

    graph_src = gm.code.replace("def forward(", "def _graph_forward(", 1)
    parts = ["import torch as _torch"]
    # gm.code relies on fx's custom builtins (torch, device, inf, nan, NoneType, fx_pytree,
    # pytree) being in scope -- fx injects them when a real GraphModule runs. Reproduce the
    # FULL set (not just torch) so a graph that bakes a device / inf / nan constant (e.g.
    # BatchNorm, masked_fill to -inf) runs standalone instead of raising NameError. Sourced
    # from fx so it stays correct.
    for _cb in _custom_builtins.values():
        parts.append(_cb.import_str)
    parts.append(graph_src)
    parts.append("")
    parts.append("class _GraphSelf:")
    parts.append("    pass")
    parts.append("")
    parts.append("")
    parts.append("def call(args):")
    grad_cm = "enable_grad" if trains else "no_grad"
    parts.append(f"    with _torch.{grad_cm}():")
    parts.append("        out = _graph_forward(_GraphSelf(), *args)")
    parts.append("    return list(out) if isinstance(out, (list, tuple)) else [out]")
    parts.append("")
    return parts


def _emit_dynamo_driver_source() -> str:
    """Emit the dynamo driver (rehydrate the inlined bytecode) as text for inlining, the
    same getsource path _emit_driver_source uses. The last statement binds the public
    ``forward`` to the reconstructed function."""
    import inspect

    from torch import _precompile_driver as driver

    blocks = [
        inspect.getsource(driver._rebuild_cell),
        inspect.getsource(driver._build_dynamo_forward),
    ]
    body = "\n\n".join(block.rstrip() for block in blocks)
    return "\n" + body + "\n\n\nforward = _build_dynamo_forward()\n\n\n" + _DRIVER_MAIN


def _build_dynamo_python_source(compiled: PrecompiledModule) -> str:
    capture = compiled._dynamo_capture
    if capture is None:
        raise PrecompileError("internal: not compiled; call _compile() first")
    parts = [_DYNAMO_GENERATED_HEADER, ""]
    parts.append("# " + "=" * 70)
    if capture.gm is None:
        # fn produced no tensor compute: the transformed bytecode is the whole artifact
        # (it references no compiled subgraph), so there is nothing to inline here; the
        # driver rehydrates that bytecode below with BACKEND_ID None (no subgraph).
        parts.append("# 1. (no compiled subgraph: fn produced no tensor compute)")
        parts.append("# " + "=" * 70)
    elif compiled._backend == "inductor":
        parts.append("# 1. Compiled subgraph (AOTAutograd + Inductor): exposes call")
        parts.append("# " + "=" * 70)
        parts.append(compiled._graph_python)
    else:
        parts.append("# 1. Captured dynamo subgraph (eager backend): exposes ``call``")
        parts.append("# " + "=" * 70)
        parts.extend(_emit_dynamo_eager_subgraph(capture.gm, capture.trains))
    parts.append("")
    parts.extend(_build_dynamo_metadata_section(compiled))
    parts.append("# " + "=" * 70)
    parts.append("# 3. Driver: rehydrate the inlined transformed bytecode")
    parts.append("# " + "=" * 70)
    parts.append(_emit_dynamo_driver_source())
    return "\n".join(parts)


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
        # Dynamo tracer (tracer="dynamo"): the transformed bytecode + the names it
        # references, populated by _compile() when tracer == "dynamo". The bytecode is
        # marshalled and the state pickled into python_code; the subgraph is lowered like
        # the make_fx inductor path (backend="inductor", via self._graph_python /
        # self._artifact_bytes) or inlined as eager graph source (backend="eager",
        # self._gm). None on the make_fx path and on the load() path.
        self._dynamo_capture: _DynamoCapture | None = None
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
        # tracer selects the capture front-end (orthogonal to backend); dispatch here,
        # the single capture-dispatch point, before running fn -- so a dynamo capture
        # never falls through to the make_fx path below with the wrong front-end.
        if self._tracer == "dynamo":
            self._compile_dynamo(args)
            return
        if self._backend == "eager" and _has_unbacked_marks(args):
            raise NotImplementedError(
                "precompile: mark_unbacked (dynamic shapes) with tracer='make_fx' is only "
                "supported with backend='inductor'; make_fx + eager + unbacked is not "
                "supported (tracer='dynamo' supports either backend)."
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

    def _compile_dynamo(self, args: tuple[object, ...]) -> None:
        # The dynamo tracer inlines the transformed bytecode; the subgraph is realized by
        # the SAME backends as the make_fx tracer, and dynamic shapes are opt-in via
        # mark_unbacked exactly as there -- on BOTH backends here, since Dynamo emits the
        # ShapeEnv's runtime asserts into the subgraph itself (the make_fx tracer's eager
        # backend has no such asserts, hence its inductor-only restriction). Dynamo honors
        # the mark itself, so the dim is captured UNBACKED -- unguardable, hence sound
        # without a runtime guard check. The marks precompile cannot honor at all
        # (mark_dynamic / specialize_on) are rejected by _reject_unsupported_marks, and
        # strict marks -- which Dynamo captures BACKED -- by _reject_strict_unbacked_marks,
        # rather than silently baking a wrong artifact.
        user_flat = pytree.tree_leaves(
            tuple(a for a in args if not isinstance(a, torch.nn.Module))
        )
        _reject_unsupported_marks(user_flat)
        _reject_strict_unbacked_marks(user_flat)
        capture = _capture_dynamo(self._fn, tuple(args), self._decompositions)
        self._dynamo_capture = capture

        if self._backend == "eager":
            # eager: keep the captured subgraph and run it as-is (no inductor lowering).
            # gm is None when fn had no tensor op -- the bytecode is the whole artifact.
            # The eager emit inlines the subgraph against an empty _GraphSelf(), so reject
            # any get_attr that is not a liftable graph input (symint / torchbind / module).
            if capture.gm is not None:
                _reject_nonliftable_get_attrs(capture.gm)
            self._gm = capture.gm
            return

        if capture.gm is None:
            # inductor + no tensor compute: there is no subgraph to lower (fn returns its
            # inputs / Python constants). Mirror the make_fx inductor path and reject with
            # a clear error rather than emitting a backend='inductor' artifact with no
            # kernels; the eager backend handles these.
            raise PrecompileError(
                "the inductor backend cannot lower a computation with no tensor compute "
                "-- the traced fn returns its inputs or Python constants unchanged. "
                "Return a computed tensor, or use backend='eager'."
            )

        if not capture.example_inputs:
            # inductor + a subgraph that HAS compute but no graph inputs (fn's tensor
            # compute depends on no lifted param/buffer or user input, e.g.
            # ``torch.ones(3) * 2``). AOTAutograd's standalone lowering detects its fake
            # mode off the graph's placeholders -- of which there are none -- and raises a
            # raw RuntimeError deep in compile_to_python (neither NoRunnableInductorModule
            # nor InductorError, so the handlers below miss it). Reject up front with a
            # clean PrecompileError pointing at eager (which runs it fine), like the
            # no-compute case above.
            raise PrecompileError(
                "the inductor backend cannot lower a subgraph with no graph inputs -- the "
                "traced fn's tensor compute depends on no model parameter/buffer or user "
                "input (e.g. it builds a constant tensor). Make the compute depend on an "
                "input, or use backend='eager'."
            )

        # inductor: lower the subgraph to self-contained source exposing
        # call(flat_inputs), reusing the make_fx inductor path. For a STATIC capture
        # capture.example_inputs are REAL (the model's params/buffers + user inputs), so
        # AOTAutograd re-fakeifies them under its own fake mode -- do NOT install the dynamo
        # tracing context, which would mix fake modes. For a DYNAMIC (mark_unbacked) capture
        # they are Dynamo's own symbolic fakes instead, which the lowering passes straight
        # through (it recovers their fake mode off the graph), so the unbacked symbols
        # survive. size_asserts is pinned on for the same memory-format contract
        # (invariant 6) the make_fx inductor path enforces, and a dynamic capture also pins
        # scalar_asserts so the ShapeEnv's runtime range / shape_id-equality asserts (the
        # only enforcement of mark_unbacked's min/max here -- the thin dynamo driver has no
        # bounds check of its own) survive into the artifact.
        import torch._inductor.config as _ind_config
        from torch._dynamo.variables.torch_function import (
            torch_function_mode_stack_state_mgr,
        )
        from torch._functorch import aot_autograd
        from torch._inductor.exc import InductorError
        from torch._inductor.standalone_compile import NoRunnableInductorModuleError

        options: dict[str, object] = {"size_asserts": True}
        if capture.dynamic and hasattr(_ind_config, "scalar_asserts"):
            options["scalar_asserts"] = True
        try:
            # With the caller's torch_function modes CLEARED, the way convert_frame
            # holds them across a backend. Dynamo already applied them symbolically
            # while capturing, and capture.gm is torch-level Python, so lowering it
            # with the modes live re-traces through every one of them a SECOND time
            # and bakes a doubly-transformed kernel -- silently, since the artifact
            # then needs no mode at all to reproduce the wrong number.
            with torch_function_mode_stack_state_mgr:
                self._graph_python, self._artifact_bytes = (
                    aot_autograd.compile_to_python(
                        capture.gm,
                        capture.example_inputs,
                        options=options,
                        # A training capture differentiates INSIDE the graph, so the AOT
                        # capture pass has to run with grad on; see compile_to_python's
                        # grad_enabled.
                        grad_enabled=capture.trains,
                    )
                )
        except NoRunnableInductorModuleError as e:
            # The subgraph has no lowerable compute (e.g. fn returns an input unchanged,
            # so the graph is a pass-through). Mirror the make_fx inductor path's clean
            # error rather than leaking the raw lowering error.
            raise PrecompileError(
                "the inductor backend cannot lower a subgraph with no compute -- the "
                "traced fn returns its inputs or Python constants unchanged, producing no "
                "Inductor kernel. Return a computed tensor, or use backend='eager'."
            ) from e
        except InductorError as e:
            # Dynamo puts non-tensor outputs in the bytecode (not the subgraph), so this
            # is unlikely, but relabel it clearly if it does fire (mirrors the make_fx
            # inductor path).
            if "Unexpected output types" in str(e):
                raise PrecompileError(
                    "the inductor backend cannot lower the captured subgraph whose "
                    "output mixes a non-tensor Python value with computed tensors. Use "
                    "backend='eager'."
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
        if self._tracer == "dynamo":
            if self._dynamo_capture is None:
                raise PrecompileError("internal: not compiled; call _compile() first")
            return _build_dynamo_python_source(self)
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


def _make_inlined_forward(python_code: str) -> Callable[..., object]:
    """Fallback: execute the self-contained python string (JITs kernels).

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
    """Callable namespace implementing ``torch.compiler.precompile`` and its loaders.

    A single instance is exposed as ``torch.compiler.precompile``; calling it precompiles a
    computation and ``torch.compiler.precompile.load`` reloads the resulting source
    artifacts. ``capture`` / ``load_package`` provide the guarded multi-graph path. It is a
    class (rather than a function with attached attributes) so these operations and the
    error type are explicit members.

    The contract for both ``__call__`` and ``load`` is Note [precompile programming
    model] in this module.
    """

    # Reported so test_public_bindings / introspection see this as ``torch.compiler``.
    __module__ = "torch.compiler"

    # The error type raised by precompile, reachable as
    # ``torch.compiler.precompile.PrecompileError``.
    PrecompileError = PrecompileError
    PrecompileSession = PrecompileSession
    PrecompiledCallable = PrecompiledCallable

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
        backend: str = "inductor",
        # Sentinel rather than "make_fx": the multi-graph path below has to
        # REJECT an explicit tracer=, and its real default is a valid value, so
        # "was it passed" cannot be recovered from the value alone.
        tracer: str | None = None,
        decompositions: dict | None = None,
        example_inputs: Sequence[tuple[object, ...]] | None = None,
        guard_filter_fn: Callable[[Sequence[Any]], Sequence[bool]] | None = None,
        recompile_limit: int = 256,
        dynamic: bool | None = None,
        invariants: str | None = None,
    ) -> tuple[str, bytes] | PrecompileSession:
        """Ahead-of-time precompile ``fn`` against example inputs.

        Passing positional example arguments keeps the self-contained source-artifact
        path and returns ``(python_code, cache)``. Passing ``example_inputs`` as a
        sequence of positional-argument tuples instead runs an execution-driven
        multi-graph capture and returns a completed session. That form records graph-break
        continuations and every guarded recompilation exercised by the supplied calls.
        Live capture keeps all guards so one example cannot silently reuse another's graph;
        ``guard_filter_fn`` applies only to the serialized artifact. Automatic calls run
        under ordinary ``torch.no_grad()`` even when the caller is in inference mode;
        serve the resulting artifact under that same grad mode. Automatic inputs and module
        parameters/buffers created inside inference mode are rejected because they remain
        inference tensors after the ambient mode is disabled. Inspect ``summary()`` and call
        ``save(path)`` on the result. Capture sessions are one-shot and cannot be re-entered.

        Capture is execution-driven, not an exhaustive analysis of ``fn``. A complete
        summary covers the calls that ran successfully; unexecuted paths and values are not
        present. ``save()`` refuses known gaps and risky dropped guards by default (the
        stricter ``require_no_dropped_guards`` is off), and :meth:`serving` turns an
        uncovered runtime call into an error.

        Do not combine the two input forms. ``tracer`` and ``decompositions`` apply only
        to the positional source-artifact path; the multi-graph path uses Dynamo and
        accepts ``guard_filter_fn``, ``recompile_limit``, ``dynamic``, and ``invariants``.
        The guard filter controls serialization only; runtime capture guards are retained.

        .. note::

            ``torch.compiler.precompile`` is NOT
            ``torch._dynamo.config.caching_precompile`` (a ``torch.compile``
            guard-serialization caching mode); it captures ``fn`` ahead of time and
            lowers it to a self-contained Python source artifact.

        With the default ``make_fx`` tracer this is a non-strict trace with an explicit
        contract; read Note [precompile programming model] before using it. The artifact
        faithfully reproduces ``fn`` only for callers that uphold that contract.

        For the positional source-artifact form, the inductor lowering step drives
        process-global compiler state and is serialized by an internal lock, so concurrent
        ``backend="inductor"`` calls lower one at a time. The make_fx capture phase and
        the ``backend="eager"`` path are not serialized.

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

        ``tracer`` selects the capture front-end (orthogonal to ``backend``):

        - ``"make_fx"`` (default): a NON-STRICT make_fx trace -- it records the ATen ops
          that actually run when ``fn`` executes once on the example inputs and does not
          analyze your Python, so control flow and shapes are specialized to the example
          (the source of the programming-model contract).
        - ``"dynamo"``: a Dynamo-based front-end that analyzes the Python (bytecode)
          rather than tracing one path. The TRANSFORMED bytecode Dynamo produces (which
          extracts the model's params/buffers, calls the compiled subgraph, and
          reassembles the output) is inlined into ``python_code`` (marshalled); the
          subgraph is lowered through the same ``backend`` choices, and ``mark_unbacked``
          dynamic shapes and ``decompositions`` are honored (see below). Scoped to
          TRAINING steps too: a ``.backward()`` / ``torch.autograd.grad`` is traced INTO
          the graph (precompile pins Dynamo's ``trace_autograd_ops`` so it does not
          graph-break), and the artifact accumulates the resulting parameter gradients onto
          the runtime model exactly like eager. This source-artifact path requires one full
          graph; for graph breaks and multiple guarded/recompiled variants, pass
          ``example_inputs=[(...,), ...]`` or use ``torch.compiler.precompile.capture``
          when the calls must be made manually.
          Dynamo's runtime guards are not embedded, and -- UNLIKE the ``make_fx`` tracer --
          the dynamo driver does NOT re-validate the runtime model/inputs at load: it does
          not reproduce the ``make_fx`` driver's param/buffer structural check (invariant 2)
          or per-input shape/dtype/device checks (invariants 3/6). Safety comes from the
          same specialization contract plus, on the inductor backend, the baked
          ``assert_size_stride`` (which catches a runtime input/weight whose SHAPE or STRIDE
          differs, but not its DTYPE); on the EAGER backend nothing is re-checked, so a
          drifted runtime model (broken weight tying, a retyped/reshaped weight) or a
          broadcast-compatible input-shape mismatch can SILENTLY miscompute where
          ``make_fx`` would raise. Pass a model and inputs matching the example, as the
          contract requires. See the ``tracer`` note in Note [precompile programming model].
          The dynamo artifact inlines marshalled bytecode plus a pickled state blob, so it
          is locked to the producing Python version (unlike the portable ``make_fx`` source)
          AND, because its import aliases can reference private ``torch._dynamo`` runtime
          modules, to a compatible torch build; load it under the same CPython and a
          matching torch build (or use ``make_fx`` with ``backend='eager'`` for portable
          source; the default ``make_fx`` inductor artifact itself inlines private
          ``torch._inductor`` modules, so it is also torch-build-locked -- only the
          Python-version portability holds for either ``make_fx`` backend).

        ``decompositions`` is an optional decomposition table (a dict mapping each
        ``OpOverload`` to a decomposition function) that controls how ATen ops are broken
        down in the captured graph. Defaults to ``None`` (make_fx's default). The
        ``make_fx`` tracer forwards it to ``make_fx`` as its ``decomposition_table``
        during capture; the ``dynamo`` tracer applies the same table by re-tracing
        Dynamo's captured subgraph with it (Dynamo itself never consults a decomposition
        table), so the resulting graph is decomposed the same way on either tracer.

        Dynamic shapes are opt-in via ``torch._dynamo.decorators.mark_unbacked``, NOT a
        precompile kwarg: mark dims on the inputs before calling, e.g.
        ``mark_unbacked(x, 0); precompile(fn, model, x)`` frees ``x``'s batch dim. Marked
        dims are captured as UNBACKED symints, which cannot be guarded on, so one artifact
        serves any runtime size of them (invariant 3); a graph that needs to guard on /
        specialize a marked dim fails at capture with a ``PrecompileError``. Dims sharing a
        ``shape_id`` reuse one symbol (equal by construction); ``min``/``max`` become
        runtime asserts. Other dims stay static. With ``tracer="make_fx"`` this requires
        ``backend="inductor"``; with ``tracer="dynamo"`` both backends work (Dynamo emits
        the runtime asserts into the subgraph itself), and ``mark_unbacked(strict=True)``
        is rejected there because Dynamo reads it as a BACKED (guardable) dim.

        One ``tracer="make_fx"`` caveat: dims that MUST be equal at runtime (e.g. two
        inputs combined by a broadcast that requires equal sizes, ``model(a) + model(b)``)
        MUST be given a SHARED ``shape_id`` so a mismatch is rejected; marking two such
        dims INDEPENDENTLY bakes a SILENT equal-size assumption there and a runtime
        mismatch does NOT raise the loud failure eager gives (invariant 3). This is a
        harvesting gap, not an inherent limit of the standalone artifact: the capture
        ShapeEnv DOES record the equality (as a deferred runtime assert, e.g.
        ``Eq(u0, u1)``), but the make_fx path does not yet harvest/enforce those relational
        asserts in its driver -- only the decorator's declared min/max feed its runtime
        bound checks. A shared ``shape_id`` is the way to get the check there;
        ``tracer="dynamo"`` enforces it either way, since the asserts ride in the graph.

        The positional form returns ``(python_code, cache)`` -- a self-contained,
        executable Python source string (the single source of truth for the calling
        convention) and a binary cache holding ONLY the backend artifact (NO metadata,
        NO weights). Reload it with
        ``torch.compiler.precompile.load(python_code, cache)``. The keyword
        ``example_inputs`` form returns a completed ``PrecompileSession``; save it to a
        package and reload with ``torch.compiler.precompile.load_package``.

        ``fn`` is the whole computation, e.g.::

            python_code, cache = torch.compiler.precompile(
                lambda model, x: model(x), model, x
            )


            def train_step(model, x, t):
                loss_fn(model(x), t).backward()  # or return autograd.grad(...)


            python_code, cache = torch.compiler.precompile(train_step, model, x, t)

        Among the positional example arguments, the ``nn.Module`` arguments have their
        params/buffers lifted to graph inputs (no weights are baked into the artifact --
        invariant 1); the rest are the runtime inputs. The reloaded callable is invoked
        with the SAME argument structure -- pass the model(s) again at runtime, e.g.
        ``f_c(model, x)``, and that runtime model must match the example model's
        parameter/buffer structure (invariant 2). Arguments are matched POSITIONALLY:
        pass the model(s) and inputs positionally both here and at load time; keyword-
        argument calling conventions are not supported (a fn that relies on them would
        surface as a raw arity error). If ``fn`` ran a backward, the
        resulting parameter gradients are scattered (accumulated) onto that runtime
        model's ``parameters()`` ``.grad`` fields, exactly like eager ``.backward()``,
        so a ``zero_grad()`` / ``optimizer.step()`` loop works unchanged; the artifact
        returns ``fn``'s own result (``None`` for a bare ``.backward()`` step), not the
        grads (invariant 5). This holds for either ``tracer`` -- only the mechanism
        differs (``make_fx`` harvests the grads as graph outputs and scatters them in the
        driver; ``dynamo`` traces the backward as an in-graph autograd call whose own
        bytecode does the accumulate).

        Input mutation (incl. module buffers, e.g. BatchNorm running stats in
        training mode), tensor subclasses (e.g. DTensor), and outputs aliasing inputs
        are supported -- AOTAutograd's prelude/epilogue is composed into the artifact
        (invariant 4), as is functionalized RNG. Caller responsibilities NOT checked
        here (see the Note): the runtime model must be structurally identical to the
        example, and control flow / shapes are specialized to those positional examples
        (invariants 2 and 3). Violations that ARE checked raise ``PrecompileError``: a
        tensor baked
        as a constant (invariant 1), effectful ops (invariant 4), and -- for the
        inductor backend -- a runtime input whose stride / memory format differs from
        the example's (invariant 6).
        """
        torch._C._log_api_usage_once("torch.compiler.precompile")
        if backend not in ("inductor", "eager"):
            raise ValueError(
                f"precompile backend must be 'inductor' or 'eager', got {backend!r}."
            )
        if tracer is not None and tracer not in ("make_fx", "dynamo"):
            raise ValueError(
                f"precompile tracer must be 'make_fx' or 'dynamo', got {tracer!r}."
            )
        if example_inputs is not None:
            if example_args:
                raise ValueError(
                    "pass either positional example arguments or example_inputs=[...], "
                    "not both"
                )
            if decompositions is not None:
                raise ValueError(
                    "example_inputs=[...] selects multi-graph Dynamo capture; "
                    "decompositions apply only to the positional source-artifact path"
                )
            if tracer is not None:
                raise ValueError(
                    "example_inputs=[...] selects multi-graph Dynamo capture, which "
                    f"has no tracer choice; drop tracer={tracer!r}. It applies only to "
                    "the positional source-artifact path."
                )
            session = self.capture(
                fn,
                backend=backend,
                guard_filter_fn=guard_filter_fn,
                recompile_limit=recompile_limit,
                dynamic=dynamic,
                example_inputs=example_inputs,
                invariants=invariants,
            )
            with session:
                pass
            return session
        if (
            guard_filter_fn is not None
            or recompile_limit != 256
            or dynamic is not None
            or invariants is not None
        ):
            raise ValueError(
                "guard_filter_fn, recompile_limit, dynamic, and invariants require "
                "example_inputs=[...]"
            )
        compiled = PrecompiledModule(
            fn,
            backend=backend,
            tracer="make_fx" if tracer is None else tracer,
            decompositions=decompositions,
        )
        compiled._compile(example_args)
        # Build the (expensive) python_code ONCE and thread it into to_cache_bytes so
        # the full metadata + embedded kernel source is not rebuilt, and so code_hash is
        # sha256 over exactly the bytes returned to the caller (a matched pair loads).
        python_code = compiled.to_python_code()
        return python_code, compiled.to_cache_bytes(python_code)

    def capture(
        self,
        fn: Callable[..., object],
        *,
        backend: str = "inductor",
        guard_filter_fn: Callable[[Sequence[Any]], Sequence[bool]] | None = None,
        recompile_limit: int = 256,
        dynamic: bool | None = None,
        example_inputs: Sequence[tuple[object, ...]] | None = None,
        invariants: str | None = None,
    ) -> PrecompileSession:
        r"""capture(fn, *, backend="inductor", guard_filter_fn=None, recompile_limit=256, dynamic=None, example_inputs=None, invariants=None) -> PrecompileSession

        Begin an execution-driven multi-graph precompile capture.

        Use this form when calls must be made manually. The shorter
        ``precompile(fn, example_inputs=[...])`` form runs known inference examples for you.
        Both preserve Dynamo graph breaks, resume continuations, and every recompiled
        variant exercised by the supplied calls. The yielded callable is valid only inside
        the capture block, and the session cannot be re-entered after that block exits.

        Capture is by execution: call the yielded function on every path and specialization
        the artifact must serve. ``example_inputs`` can make positional-only inference
        calls automatically under ordinary ``torch.no_grad()``; calls in the block use the
        ambient grad mode, so wrap forward-only inference calls in ``torch.no_grad()``.
        Automatic inputs and an ``nn.Module``'s parameters/buffers must not be inference
        tensors. Live capture retains all guards. The filter controls only which guards are
        serialized, and ``save()`` rejects the risky ones by default. A complete
        summary covers only successful calls that actually ran; unexercised paths remain
        absent. Any captured call that raises marks the session incomplete, even when the
        exception is caught inside the block.

        Args:
            fn (Callable): callable to capture.
            backend (str, optional): ``torch.compile`` backend. Default: ``"inductor"``.
            guard_filter_fn (Callable, optional): receives a sequence of guard entries and
              returns one boolean per entry, where ``True`` serializes the guard. Live
              capture always retains it. The default drops identity guards that cannot be
              serialized. ``save()`` refuses only the RISKY drops by default, and every
              custom-filter drop counts as risky; ``require_no_dropped_guards=True``
              refuses all of them.
              Default: ``None``.
            recompile_limit (int, optional): maximum variants captured per frame. Default:
              ``256``. This overrides a lower ambient accumulated-recompile limit for the
              capture.
            dynamic (bool, optional): dynamic-shape policy forwarded to ``torch.compile``.
              Default: ``None``.
            example_inputs (Sequence[tuple], optional): positional-argument tuples run
              automatically under ``torch.no_grad()``. Inference tensors are rejected;
              create automatic inputs outside inference mode. Default: ``None``.
            invariants (str, optional): file receiving the invariant report after a
              successful capture. Default: ``None``.

        Returns:
            PrecompileSession: session whose context manager yields the callable to
            exercise and whose ``save()`` method writes the artifact.
        """
        from torch._dynamo.exc import PackageError
        from torch._dynamo.precompile_package import precompile_capture

        try:
            session = precompile_capture(
                fn,
                backend=backend,
                guard_filter_fn=guard_filter_fn,
                recompile_limit=recompile_limit,
                dynamic=dynamic,
                example_inputs=example_inputs,
                invariants=invariants,
            )
        except PackageError as e:
            raise PrecompileError(str(e)) from e
        return PrecompileSession(session)

    def load_package(
        self,
        fn: Callable[..., object],
        path: str,
        *,
        backend: str = "inductor",
        guard_filter_fn: Callable[[Sequence[Any]], Sequence[bool]] | None = None,
        recompile_limit: int = 256,
        dynamic: bool | None = None,
    ) -> PrecompiledCallable:
        r"""load_package(fn, path, *, backend="inductor", guard_filter_fn=None, recompile_limit=256, dynamic=None) -> Callable

        Load a multi-graph artifact saved by :meth:`capture`.

        Loading installs compiled backends and resume globals, while guarded dispatch is
        scoped to the returned callable's isolated compile region. The result is therefore
        also a context manager; exiting it, or calling ``unload()``, removes that region
        and its owned globals. The artifact is executable pickle data; only load a package
        you trust.

        Args:
            fn (Callable): callable that the artifact was captured from.
            path (str): artifact file written by :meth:`capture`.
            backend (str, optional): ``torch.compile`` backend. Default: ``"inductor"``.
            guard_filter_fn (Callable, optional): serialization filter for an uncovered call
              allowed to compile outside :meth:`serving`; it returns one boolean per guard
              entry. Runtime guards remain intact. Default: ``None``.
            recompile_limit (int, optional): recompilation limit outside :meth:`serving`.
              Default: ``256``. This overrides a lower ambient accumulated-recompile limit.
            dynamic (bool, optional): dynamic-shape policy forwarded to ``torch.compile``.
              Default: ``None``.

        Returns:
            Callable: loaded callable and context manager whose ``unload()`` method
            removes the package.
        """
        from torch._dynamo.precompile_package import precompile_load

        log.warning(
            "torch.compiler.precompile.load_package is about to unpickle and install "
            "an executable artifact. Only load a package you produced or otherwise "
            "trust."
        )
        try:
            return PrecompiledCallable(
                precompile_load(
                    fn,
                    path,
                    backend=backend,
                    guard_filter_fn=guard_filter_fn,
                    recompile_limit=recompile_limit,
                    dynamic=dynamic,
                )
            )
        except PrecompileError:
            raise
        except RuntimeError as e:
            raise PrecompileError(str(e)) from e

    def serving(self) -> AbstractContextManager[None]:
        """Forbid compilation while serving a loaded multi-graph artifact.

        An input or path missing from the capture then raises instead of silently compiling
        a new variant. Scoped to the CALLING THREAD for the duration of the
        context, so work handed to other threads -- a thread pool over a batch
        -- is NOT covered by it.
        """
        from torch._dynamo.precompile_package import serving

        return serving()

    def load(self, python_code: str, cache: bytes) -> Callable[..., object]:
        """Reconstruct a runnable from ``(python_code, cache)`` from precompile.

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
        calling-convention metadata), if the cache's ``backend`` or ``tracer`` tag does
        not match ``python_code``, or if the cache's ``code_hash`` does not match
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
                        "came from different precompile() calls."
                    )
                # A tracer tag was added alongside the dynamo tracer; treat its absence as
                # make_fx so an older make_fx cache still pairs with its python_code. A
                # differing tag means a wrong (code, cache) pairing, so hard-fail.
                if blob.get("tracer", "make_fx") != tracer:
                    raise PrecompileError(
                        f"cache tracer {blob.get('tracer', 'make_fx')!r} does not match "
                        f"the python_code tracer {tracer!r}; the cache and python_code "
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
# (mirroring the singleton fixup above) -- otherwise Sphinx and IDE introspection would
# anchor them under this private module. The operations are bound methods, so patch their
# underlying functions.
PrecompileError.__module__ = "torch.compiler"
PrecompileError.__qualname__ = "precompile.PrecompileError"
PrecompileSession.__module__ = "torch.compiler"
PrecompiledCallable.__module__ = "torch.compiler"
_PrecompileApi.FrameInvariants = FrameInvariants
_PrecompileApi.GuardFact = GuardFact
_PrecompileApi.PrecompileSummary = PrecompileSummary
_PrecompileApi.load.__module__ = "torch.compiler"
_PrecompileApi.load.__qualname__ = "precompile.load"
_PrecompileApi.capture.__module__ = "torch.compiler"
_PrecompileApi.capture.__qualname__ = "precompile.capture"
_PrecompileApi.load_package.__module__ = "torch.compiler"
_PrecompileApi.load_package.__qualname__ = "precompile.load_package"
_PrecompileApi.serving.__module__ = "torch.compiler"
_PrecompileApi.serving.__qualname__ = "precompile.serving"
