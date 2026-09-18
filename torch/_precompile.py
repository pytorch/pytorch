"""Ahead-of-time precompilation (``make_fx`` tracer by default; Dynamo planned).

    python_code, cache = torch.compiler.precompile(fn, model, *example_inputs)
    f_c = torch.compiler.precompile.load(python_code, cache)
    out = f_c(model, *example_inputs)   # pass the model again at runtime

precompile captures your computation with ``make_fx`` -- a NON-STRICT trace of the ATen
ops that run when ``fn`` executes once on FAKE tensors derived from the example inputs
(the example tensors themselves are never computed on). It does not analyze your Python,
so it comes with an explicit contract (the programming model): stay inside it and the
artifact faithfully reproduces ``fn``; step outside it and you get an artifact that
computes the wrong thing.

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
# 3. PYTHON control flow (and, by default, shapes) is specialized to the example. A
#    non-strict trace follows the single path taken for the example inputs: Python
#    ``if``/``for`` over a static (Python ``int``) value and shape-dependent branching on
#    a static size are resolved at trace time and baked. A control-flow HOP is the
#    exception: ``torch.cond`` / ``torch.while_loop`` is REFUSED outright rather than
#    specialized, because neither backend can lower the captured subgraph it traces into.
#    Shapes are static BY DEFAULT (capture runs make_fx in its "fake" mode, so each size is
#    baked as a concrete constant). What is NOT silently baked is a data-dependent op --
#    ``.item()``, ``.nonzero()``, a Python ``if`` over a TENSOR VALUE: under fake tracing
#    the value is unknown. (A ``for`` over a tensor is not data-dependent -- it iterates
#    dim 0 at its static size, so it unrolls and bakes like the static-``int`` case above.)
#    A STATIC capture REFUSES such a data-dependent op: the fake trace raises
#    DataDependentOutputException or DynamicOutputShapeException, surfaced as a
#    PrecompileError. The op need not be in ``fn``'s own code -- in ``train()`` mode
#    ``nn.BatchNorm*(momentum=None)`` reads ``float(num_batches_tracked)``, so the refusal
#    names the underlying ATen op. What a real (non-fake) trace did with these was not one
#    thing: it RAISED on ``.item()`` and on a branch over a tensor value, so for those the
#    refusal is only an error-quality win; it CAPTURED ``.nonzero()`` / masked_select into
#    a graph whose output size varies with the data, so refusing that one is a deliberate
#    narrowing of the STATIC path. It has to be: minting a size the fake kernel cannot know
#    needs a ShapeEnv, which a static capture deliberately does not have. An UNBACKED
#    capture (mark_unbacked, below) is the only path with a ShapeEnv, so it can instead
#    CAPTURE ``.item()`` as an unbacked value -- and likewise a ``.nonzero()``-sized
#    intermediate, which captures and serves at any runtime size -- and fails if the
#    computation must GUARD on one, or if the op is one no ShapeEnv can fake at all (e.g.
#    ``aten.equal``), which BOTH paths refuse.
#    Tracing on fake tensors also needs a meta/fake kernel for every op ``fn`` calls: an
#    op without one (a custom op missing ``torch.library.register_fake``, but also a
#    built-in one -- capture turns FakeTensorMode's unsafe fallback OFF, so no op is ever
#    run for real on zero-filled substitutes) is refused with a PrecompileError, UNLESS
#    one can be synthesized for it -- a ``torch.library.custom_op`` tagged
#    ``torch.Tag.inplace`` or ``torch.Tag.out``, or one that only mutates its arguments and
#    returns nothing, gets a trivial fake impl and captures without a ``register_fake``, so
#    declare the mutation (``mutates_args``) and the tags accurately. Also refused
#    is a ``fn`` that reads a tensor's DATA -- ``.data_ptr()``, or a NumPy conversion
#    (``.numpy()``, ``np.asarray(t)``, ``t.__array__()``) -- since a fake tensor has no
#    real memory behind it, where a real trace would have run the kernel. Fake tracing also
#    constrains the example INPUTS themselves: every example tensor (user input, param or
#    buffer) must be representable as a fake tensor, so one the meta converter cannot
#    represent -- a quantized tensor, a lazy-device tensor, a legacy batched tensor, a
#    view out of a sparse tensor -- is refused at capture. A real trace accepted a view out
#    of a sparse tensor; it did NOT accept a quantized input, it crashed on one with a raw
#    NotImplementedError out of make_fx's own placeholder fakeification. Three kinds of
#    input are refused for the opposite reason -- fakeification SUCCEEDS but silently drops
#    metadata the trace reads at Python level and bakes, which a real trace baked correctly:
#    an MKLDNN tensor (its fake is strided, so ``to_dense()`` / ``to_mkldnn()`` record no
#    node), a SPARSE tensor (its fake has 0 nnz, so ``.values()`` is annotated empty and a
#    Python read of the nnz bakes 0) and a PINNED one (its fake reads ``is_pinned() ==
#    False``, so a branch on it bakes the unpinned side). Those three are deliberate
#    narrowings of both paths, taken because the alternative is a silently wrong artifact
#    on any backend, and they are applied to a traceable wrapper subclass's INNER tensors
#    too (a strided-looking wrapper over sparse data would otherwise bake that 0 nnz);
#    ``fn`` calling ``.pin_memory()`` is refused as well, as an op with no fake
#    kernel. A NESTED tensor is refused too, on BOTH capture paths, but as a
#    restriction rather than a claim about fakeification (the unbacked path's ShapeEnv
#    could fakeify a jagged one; nothing downstream of the trace has a nested
#    representation) -- and a real trace did not accept a STRIDED nested input either, it
#    crashed on it with a raw internal error.
#    Make such a value a plain dense tensor or a supported subclass (e.g. DTensor).
#    Every refusal above rests on capture tracing under a fake mode IT built, so capture
#    also refuses to run inside another trace: an ambient ``TracingContext.fake_mode`` (a
#    torch.compile / export / AOTAutograd trace) outranks capture's own mode, and no foreign
#    mode passes ``allow_fallback_kernels=False``, so a meta-less op in an allowlisted
#    namespace would be run for real again. A mode built under DEFAULT config (an
#    AOTAutograd / inductor one) lacks the data-ptr snapshot as well, so a ``.data_ptr()``
#    read would bake 0 rather than raise; a torch.compile / export mode does build under
#    that patch, so for those two only the fallback setting is lost.
#    Call precompile outside the enclosing trace.
#
#    You can opt specific user-input dims into being dynamic by marking them with
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
# (python_code, cache) pair from different precompile captures.
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
# non-strict trace and is the only tracer implemented today -- everything above (the
# invariants, the contract) describes its behavior. "dynamo" is planned (a Dynamo-based
# front-end that analyzes Python rather than specializing to one traced path) and
# currently raises NotImplementedError.

from __future__ import annotations

import dataclasses
import errno
import functools
import hashlib
import io
import logging
import os
import stat
import uuid
from types import MappingProxyType
from typing import Any, cast, NewType, TYPE_CHECKING
from typing_extensions import Self

import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._guards import TracingContext
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


log = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from torch._functorch._aot_autograd.codegen import PySourceBuilder
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
    effectful op, a data-dependent op the fake-tensor capture cannot know (``.item()``,
    ``.nonzero()``, a branch over a tensor value), an op with no meta/fake kernel and none
    that can be synthesized (a ``torch.library.custom_op`` tagged ``torch.Tag.inplace`` /
    ``torch.Tag.out``, or one that only mutates its arguments and returns nothing, gets a
    trivial one, so it needs no ``register_fake``), a read of a traced tensor's data
    (``.data_ptr()``, ``.numpy()`` / ``np.asarray()``), an
    example input that cannot be represented as a fake tensor (a quantized tensor) or whose
    metadata a fake tensor drops (a pinned, mkldnn or sparse tensor), a nested example
    input, none of which capture supports on either path (invariant 3), a control-flow HOP
    (``torch.cond`` / ``torch.while_loop``), whose captured subgraph neither backend can
    lower, a capture attempted inside another trace (an ambient ``TracingContext``
    fake mode, which does not carry capture's fallback-off setting), a non-tensor output the
    inductor backend cannot lower, or a runtime input whose shape or memory format differs
    from the example (invariants 3 and 6).
    See Note [precompile programming model] in this module for the full contract.
    """


@dataclasses.dataclass(frozen=True)
class MakeFxTracer:
    """The ``make_fx`` capture front-end, passed as ``tracer=`` to :func:`capture`.

    A NON-STRICT single make_fx trace: it records the ATen ops of ONE execution of
    ``fn``, so a ``capture`` with this tracer takes exactly one call and refuses a
    second, and control flow and shapes are specialized to that call (the source of
    the programming-model contract). Part of the prototype ``torch.compiler.precompile``
    API, so it may change without a deprecation cycle.

    ``decompositions`` is an optional decomposition table (a dict mapping each
    ``OpOverload`` to a decomposition function) forwarded to ``make_fx`` as its
    ``decomposition_table``; it is specific to this tracer (Dynamo lowers through the
    backend).
    """

    decompositions: dict | None = None


class PrecompiledRunnable:
    """What :func:`load` returns.

    A callable with the captured ``fn``'s calling convention that can also be entered as
    a context manager and unloaded. A standalone artifact installs nothing, so for it
    ``__enter__``/``__exit__``/:meth:`unload` are no-ops; the shape that installs onto its
    captured code objects arrives with the dynamo tracer, and ``installed`` tells them
    apart. Part of the prototype ``torch.compiler.precompile`` API, so it may change
    without a deprecation cycle.
    """

    installed: bool = False
    """Whether calling this handle installs onto the captured code objects.

    ``False``, the base-class value, is the contract for a STANDALONE artifact --
    every artifact :func:`load` returns today: it serves by being called and has
    nothing to take back out. A subclass that installs sets it ``True``.
    """

    def __call__(self, *args: object) -> object:
        raise NotImplementedError

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.unload()

    def unload(self) -> None:
        """Remove whatever this loaded artifact installed.

        The base implementation does nothing, which is the whole contract while
        ``installed`` is ``False``: unloading a standalone artifact leaves the handle
        exactly as callable. Calling it twice, or on a handle never entered, is fine.
        """


class Capture:
    r"""The caller-driven capture :func:`capture` returns.

    Part of the prototype ``torch.compiler.precompile`` API, so it may change
    without a deprecation cycle. Enter it as a context manager to arm the
    capture, call it with the positional arguments ``fn`` takes inside the block
    (keyword arguments are refused) -- each call runs for real, is folded into the
    capture, and returns what serving the artifact produces (:func:`capture` has
    the ``requires_grad`` contract of that served value) -- and the artifact is
    written to the ``artifact_path`` / ``cache_path`` files when the block exits
    CLEANLY: a block that RAISED writes nothing, and a clean exit that never called
    the capture raises instead of writing. Call :meth:`save` inside the block to
    checkpoint everything captured so far to those same files without ending the
    capture. The object is single-shot: the block is entered once, calling outside
    it is refused, and so is saving -- except to retry a WRITE that failed.
    """

    def __enter__(self) -> Self:
        raise NotImplementedError

    def __exit__(self, *exc: object) -> None:
        raise NotImplementedError

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise NotImplementedError

    def save(self) -> None:
        raise NotImplementedError


_SPENT = (
    "capture is spent: its `with` block already ran. Call capture() again for "
    "another artifact."
)
# The message for the one spent state with something left to offer, selected by
# _write_failed: the render is in hand, so the caller wants save(), not a re-trace.
_SPENT_RETRY = (
    "capture is spent for tracing, but its WRITE is what failed: call save() to "
    "retry it."
)


class _MakeFxCapture(Capture):
    r"""Single-shot capture: the :class:`MakeFxTracer` front-end.

    A make_fx trace records the ATen ops of ONE execution of ``fn``, so this captures
    exactly one call and refuses a second: there are no guards or recompiled variants
    here, so a further call could add nothing.
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
        # The module scan and the param/buffer lifting both work off the CALL arguments, so a
        # model reached any OTHER way -- fn itself, fn's __self__, a partial's bound argument
        # or the (nestable) callable a partial wraps -- is invisible to them and would bake
        # into the graph as constants (invariant 1), crashing in _check_no_constant_tensors.
        target, held = fn, []
        while isinstance(target, functools.partial):
            held += pytree.tree_leaves((target.args, target.keywords))
            target = target.func
        held.append(getattr(target, "__self__", target))
        if any(isinstance(a, (torch.Tensor, torch.nn.Module)) for a in held):
            raise PrecompileError(
                "precompile cannot capture a model itself, or a callable that HOLDS a "
                "tensor or an nn.Module (a bound method, a partial that holds one): "
                "precompile discovers the model(s) and the runtime inputs among the CALL "
                "arguments, so a held one would be baked into the graph as a constant "
                "(invariant 1). Pass a function taking them and give them as call "
                "arguments; a partial binding only non-tensor arguments is fine."
            )
        self._module = PrecompiledModule(
            fn, backend=backend, tracer="make_fx", decompositions=decompositions
        )
        self._artifact_path = artifact_path
        self._cache_path = cache_path
        self._training = training
        self._entered = False
        self._exited = False
        self._traced = False
        self._trace_failed = False
        self._serve_failed = False
        self._write_failed = False
        self._rendered: tuple[str, bytes] | None = None

    def __enter__(self) -> Self:
        # Single-shot in both directions: a nested `with cap:` would deactivate the outer
        # block on its own exit, a second one rewrite both files from the first's render.
        if self._entered:
            raise PrecompileError(
                "capture is already active: it is not re-entrant, use one `with` block."
            )
        if self._exited:
            raise PrecompileError(_SPENT_RETRY if self._write_failed else _SPENT)
        self._entered = True
        return self

    def __exit__(self, *exc: object) -> None:
        # The capture goes inactive however the block ends, including the nothing-was-captured
        # raise below: a call after the block would trace, lower and serve and write nothing.
        self._entered = False
        self._exited = True
        # Only a clean exit that captured a call writes (see both docstrings).
        if exc[0] is not None:
            return
        if self._rendered is None:
            raise self._nothing_captured("inside the `with` block.")
        # A write that RAISED (ENOSPC, a read-only directory) leaves the previous pair
        # intact and this render in memory, so it must not strand a finished capture:
        # the flag keeps save() open as a retry of the WRITE, no re-trace needed.
        self._write_failed = True
        _write_artifact(self._artifact_path, self._cache_path, *self._rendered)
        self._write_failed = False

    def _nothing_captured(self, how: str) -> PrecompileError:
        """The refusal for a write with no render: the call failed, or never happened."""
        if self._trace_failed or self._serve_failed:
            stage = (
                "while serving the artifact it rendered"
                if self._serve_failed
                else "before it rendered an artifact"
            )
            return PrecompileError(
                f"nothing was captured: the capture call raised {stage}, so there is "
                "nothing to write."
            )
        return PrecompileError(
            f"nothing was captured: call the capture with your example arguments {how}"
        )

    def save(self) -> None:
        """Write the captured artifact to disk.

        A make_fx capture records a single call, so there is nothing further to fold in;
        save() and block exit write the same files. Callable inside the block, and after
        a block whose WRITE failed -- the exit's or an in-block save()'s -- to retry it.
        """
        # Gated on the block being live like __call__ is, EXCEPT after a WRITE that raised,
        # this method's or the exit's: after a block that raised, the render is still here and
        # a save() outside would otherwise write the pair __exit__ refused to write.
        if not self._entered and not self._write_failed:
            # After the block, the fix is a fresh capture(): re-entering is refused too.
            if self._exited:
                raise PrecompileError(_SPENT)
            raise PrecompileError(
                "capture is not active: call save() inside its `with` block."
            )
        if self._rendered is None:
            raise self._nothing_captured("before calling save().")
        # Armed before the write like __exit__'s, so a write that RAISED keeps save() open as
        # a retry from OUTSIDE the block too: that exception leaves the block unwritten.
        self._write_failed = True
        _write_artifact(self._artifact_path, self._cache_path, *self._rendered)
        self._write_failed = False

    def __call__(self, *args: object, **kwargs: object) -> object:
        # Only the block exit writes the files, so a call made outside it would
        # trace, lower and serve and then write nothing at all.
        if not self._entered:
            if self._exited:
                raise PrecompileError(_SPENT_RETRY if self._write_failed else _SPENT)
            raise PrecompileError(
                "capture is not active: call it inside its `with` block."
            )
        if kwargs:
            raise TypeError(
                "MakeFxTracer takes positional arguments only; pass fn's arguments "
                "positionally. Capturing a call made with keyword arguments is not "
                "available in this build."
            )
        if self._trace_failed or self._serve_failed:
            raise PrecompileError(
                "the capture's single call already ran and raised, so nothing was "
                "captured and a retry would trace the same failure. Fix what it "
                "raised, then run a fresh capture()."
            )
        if self._traced:
            raise PrecompileError(
                "MakeFxTracer captures a single call and one has already been "
                "traced. Capturing several calls, with the graph breaks and "
                "recompilations between them, is not available in this build."
            )
        # make_fx traces one execution of fn and lowers it to the artifact; we then serve that
        # artifact on the real args through the SAME load() path a caller would take, so the
        # value handed back is what serving produces (invariants checked, grads scattered onto
        # the model) rather than a bare trace. python_code is built ONCE and threaded into
        # to_cache_bytes, so code_hash is sha256 over exactly the bytes written on exit;
        # ``training`` sets the grad mode of trace and serve alike.
        with torch.enable_grad() if self._training else torch.no_grad():
            # The single-call flag counts a RENDER, not an attempt: a trace that raised (a
            # data-dependent op, a missing fake impl) captured nothing, so the retry and the
            # nothing-was-captured refusals must say the trace failed, not that a call was.
            try:
                self._module._compile(args)
                python_code = self._module.to_python_code()
                self._rendered = (python_code, self._module.to_cache_bytes(python_code))
            except RuntimeError as e:
                self._trace_failed = True
                # Precompile's own refusals are already explained, and one IS a
                # RuntimeError: re-raise before the substring clause below re-wraps it.
                if isinstance(e, PrecompileError):
                    raise
                # A backward under training=False: nothing the no_grad trace produced has a
                # grad_fn, so autograd's own message comes back with no hint that ``training``
                # is the switch that fixes it. It raises that same string when nothing fn
                # differentiates requires grad at all (frozen params, a detached loss), which
                # training=True does not fix, so only name it as the cause when a tensor or a
                # param/buffer passed in DOES require grad.
                if not self._training and "does not require grad" in str(e):
                    leaves = pytree.tree_leaves(args)
                    mods = [a for a in leaves if isinstance(a, torch.nn.Module)]
                    tensors = [t for m in mods for t in (*m.parameters(), *m.buffers())]
                    tensors += [a for a in leaves if isinstance(a, torch.Tensor)]
                    if any(t.requires_grad for t in tensors):
                        raise PrecompileError(
                            "precompile: fn ran a backward but the capture traced "
                            "under no_grad, so nothing it produced has a grad_fn. "
                            "Pass training=True to capture the backward (the artifact "
                            "then scatters the gradients onto the model), unless fn "
                            "itself blocks the gradient -- it detaches what it "
                            "differentiates, runs under its own torch.no_grad(), or "
                            "backwards a tensor that does not require grad. "
                            f"Underlying: {e}"
                        ) from e
                raise
            except BaseException:
                self._trace_failed = True
                raise
            self._traced = True
            try:
                return _runnable_from_pair(
                    *self._rendered, who="torch._precompile.capture", _trusted=True
                )(*args)
            except BaseException:
                # The serve is where the driver's own runtime checks run, so a serve that
                # raised is a call that did not work: drop the render, or a CAUGHT serve
                # error would leave the exit writing a pair whose serve never worked.
                self._serve_failed, self._rendered = True, None
                raise


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
    pb_flat: list[Tensor],
    user_flat: list[object],
    marks: list[dict[int, _MarkSpec]],
    labels: list[str],
) -> tuple[list[object], FakeTensorMode]:
    """Fakeify the flat capture inputs for an unbacked dynamic-shape capture.

    Params/buffers and unmarked dims become static fakes; each mark_unbacked dim becomes
    an UNBACKED SymInt (unguardable, so the artifact is valid for any runtime size and a
    graph that needs to guard on it fails at capture). Dims sharing a ``shape_id`` reuse
    one symbol; ``min``/``max`` add runtime asserts. Returns ``(flat_fake, fake_mode)``;
    the fake_mode (ShapeEnv) is threaded to the lowering via from_tracing_context.
    ``labels`` (aligned to ``[*pb_flat, *user_flat]``) names an input in a refusal.
    """
    import torch._functorch.config as functorch_config
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    # FakeTensorMode SNAPSHOTS this config at construction, and that snapshot is what
    # every fake tensor it makes consults, so the patch must wrap the CONSTRUCTION (as
    # make_fx does around its own mode): with it off, a .data_ptr() read in fn raises
    # instead of returning a meaningless value. allow_fallback_kernels=False for the
    # reason _capture's static mode gives (no real kernel on zero-filled substitutes).
    with functorch_config.patch(fake_tensor_allow_unsafe_data_ptr_access=False):
        fake_mode = FakeTensorMode(
            shape_env=shape_env,
            allow_non_fake_inputs=True,
            allow_fallback_kernels=False,
        )
    # shape_id -> unbacked symint (a dynamic SymInt); untyped so grouped dims share one symbol.
    shared: dict[object, Any] = {}
    user_labels = labels[len(pb_flat) :]
    with fake_mode:
        fake_pb = [_fakeify_input(fake_mode, t, lbl) for t, lbl in zip(pb_flat, labels)]
        fake_user: list[object] = []
        for leaf, per, label in zip(user_flat, marks, user_labels):
            if not isinstance(leaf, torch.Tensor):
                fake_user.append(leaf)
            elif not per:
                fake_user.append(_fakeify_input(fake_mode, leaf, label))
            else:
                # Validate the marked leaf through the same helper -- the fake it returns
                # is discarded, since the leaf is rebuilt at its unbacked sizes below --
                # so an unfakeifiable input is named whether or not it is the marked one.
                # Without this it escapes as a raw meta-kernel error (for a marked
                # quantized input: "SymIntArrayRef expected to contain only concrete
                # integers"), because the rebuild never consults the meta converter.
                _fakeify_input(fake_mode, leaf, label)
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


def _fakeify_input(fake_mode: FakeTensorMode, t: Tensor, label: str) -> Tensor:
    """from_tensor for one static example tensor, naming it if it cannot be fakeified.

    ``label`` is one of ``_capture``'s ``input_labels`` ("parameter w", "buffer nt",
    "user input 0"), so the refusal names what the caller has to change.

    Shared by BOTH capture paths, so the same input is refused the same way whether or not
    any dim is marked: the unbacked path fakeifies its params/buffers and its unmarked
    inputs through this helper, and calls it on a MARKED leaf too (discarding the result)
    to validate a leaf it then rebuilds at unbacked sizes.
    An input the meta converter cannot represent -- a quantized tensor, a lazy-device
    tensor, a legacy batched tensor or a view out of a sparse tensor (fake_tensor.py
    raises this for the first, and meta_utils returns NotImplemented on the rest) --
    cannot be traced on either path; this buys error quality, a PrecompileError that names
    the input instead of a raw converter error. Narrow on purpose: any other failure in
    from_tensor is an internal bug and must surface as itself. The inputs from_tensor
    accepts but cannot faithfully REPRESENT (nested, mkldnn, sparse, pinned) are refused
    before this, by ``_reject_unfakeifiable_input`` over every example input.
    """
    from torch._subclasses.fake_tensor import UnsupportedFakeTensorException

    try:
        return fake_mode.from_tensor(t, static_shapes=True)
    except UnsupportedFakeTensorException as e:
        raise PrecompileError(
            f"precompile: example {label} cannot be represented as a fake tensor, which "
            "capture traces on. Make it a plain dense tensor (or a supported subclass) -- "
            "on the model for a parameter/buffer, at the call site for a user input. "
            f"Underlying: {(str(e).splitlines() or [''])[0]}"
        ) from e


def _reject_unfakeifiable_input(label: str, a: Tensor) -> None:
    """Refuse an example tensor a fake tensor cannot faithfully stand in for.

    ``label`` is one of ``_capture``'s ``input_labels`` ("parameter w", "buffer nt", "user
    input 0"), so each refusal names what the caller has to change. A NESTED input cannot be
    fakeified by a static capture at all (minting the symbolic nested int for a jagged
    tensor's ragged dim needs a ShapeEnv, which a static capture deliberately does not have,
    so from_tensor dies on a raw internal assertion), and while the unbacked path's ShapeEnv
    could, nothing downstream of the trace has a nested representation (the recorded dense
    shape/dtype/device the driver checks against is None for one), so it is a capture-wide
    restriction rather than a claim about either fake mode. The other three are worse than
    unfakeifiable: from_tensor SUCCEEDS and silently DROPS metadata the trace then reads at
    Python level and bakes -- an mkldnn tensor comes back strided, a sparse one comes back
    with 0 nnz, a pinned one comes back unpinned -- where the real-tensor trace this commit
    replaces baked the right thing. None of the three can be repaired on the fake side, so
    they are refused rather than left to bake a wrong artifact with nothing downstream able
    to notice. The clause order is for diagnosis quality: mkldnn precedes the generic-layout
    test so an mkldnn input gets the layout-conversion diagnosis rather than the nnz one, and
    the dispatched is_pinned() probe comes last.
    """
    if a.is_nested:
        raise PrecompileError(
            f"precompile: example {label} is a nested tensor ({a.layout} layout), "
            "which capture does not support. Make it a plain dense tensor (or a "
            "supported subclass) -- on the model for a parameter/buffer, at the call "
            "site for a user input."
        )
    if a.is_mkldnn:
        raise PrecompileError(
            f"precompile: example {label} has {a.layout} layout, which capture does "
            "not support: its fake stand-in comes back STRIDED, so the layout "
            "conversions (to_dense / to_mkldnn) are traced away and the graph would "
            "run dense kernels on mkldnn data. Pass a strided tensor -- on the model "
            "for a parameter/buffer (do not torch.utils.mkldnn.to_mkldnn the example "
            "module), at the call site for a user input."
        )
    if a.layout is not torch.strided:
        raise PrecompileError(
            f"precompile: example {label} has {a.layout} layout, which capture does "
            "not support: its fake stand-in reports 0 nnz, so the trace annotates "
            ".values() empty and bakes a Python read of the nnz (._nnz(), len(values)) "
            "as 0 -- a wrong result rather than an error. Pass a strided (dense) "
            "tensor -- on the model for a parameter/buffer, at the call site for a "
            "user input."
        )
    # is_pinned() DISPATCHES, unlike the three metadata reads above, so unlike them it can
    # raise. The types caught are the ways seen so far, not a closed set: no batching rule
    # for aten::is_pinned on a functorch-batched tensor (RuntimeError), the dispatcher's
    # TypeError for a __torch_dispatch__ subclass that declines the op the documented way
    # (every handler returning NotImplemented, as torch.masked.MaskedTensor does), and
    # UninitializedTensorMixin's ValueError for a lazy module's uninitialized parameter
    # (its allowlist of tolerated methods omits is_pinned). It stays last so a tensor a
    # clause above describes better gets that diagnosis, and the raise is swallowed so
    # none of them escapes the public API raw.
    # Swallowing is sound but not because a declining tensor is unpinned (a vmap-batched
    # view of pinned memory declines the op AND is pinned): such an input cannot be SHOWN
    # to be pinned here, and it is refused elsewhere on its own terms -- a batched one by
    # invariant 1, once its compute is traced.
    try:
        is_pinned = a.is_pinned()
    except (RuntimeError, TypeError, ValueError):
        is_pinned = False
    if is_pinned:
        raise PrecompileError(
            f"precompile: example {label} is in pinned memory, which a fake tensor "
            "cannot represent: is_pinned() reads False while tracing, so a branch on "
            "it bakes the unpinned side. Pass an unpinned tensor -- pinning is a "
            "runtime staging concern the artifact does not encode, so pin outside the "
            "captured region (a .pin_memory() call INSIDE fn is refused too, as an op "
            "with no fake kernel)."
        )
    # The three metadata reads above see the OUTER tensor only, and a traceable wrapper
    # subclass reports strided, non-nested, non-mkldnn whatever it wraps, so recurse into
    # its inner tensors (the is_pinned() probe already sees through one, because it
    # dispatches). Without this a wrapper over sparse data passes every clause above and
    # then has that inner metadata dropped by the fake conversion exactly as the messages
    # describe, baking a wrong artifact where this function exists to leave none.
    # Unwrapping rather than refusing wrappers wholesale keeps DTensor working, and it is
    # the same flattening the fake conversion performs. It is NOT what keeps MaskedTensor
    # working: that is not a traceable wrapper subclass, so it never reaches this branch --
    # the except clause on the is_pinned() probe above is what lets it through.
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


def _control_flow_refusal(detail: str) -> PrecompileError:
    return PrecompileError(
        "precompile cannot lower a captured control-flow subgraph (e.g. from "
        f"torch.cond / torch.while_loop); not supported yet. {detail}"
    )


def _missing_fake_kernel_refusal(detail: str) -> PrecompileError:
    return PrecompileError(
        "precompile: fn calls an operator that has no meta/fake kernel, which the "
        "fake-tensor trace needs to compute output shapes. For a custom op, register "
        "one with torch.library.register_fake. A built-in ATen op has no fake kernel you "
        "are expected to register -- registering one overrides it process-wide -- so avoid "
        f"the op inside fn instead. {detail}"
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
        raise _control_flow_refusal(f"Offending get_attr targets: {offending}.")


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

    The trace runs in the AMBIENT grad mode, so the caller must invoke ``_capture``
    with grad ENABLED for a training ``fn``'s backward to be captured. The callable
    entry point always wraps its call in ``torch.enable_grad()``; :func:`capture`
    instead lets ``training=`` pick the mode (``enable_grad`` for ``True``,
    ``no_grad`` for ``False``). Under ``no_grad`` nothing the trace produces has a
    ``grad_fn``, so ``fn``'s ``.backward()`` raises ("element 0 of tensors does not
    require grad and does not have a grad_fn"), which :func:`capture` re-reports as
    a ``PrecompileError`` naming ``training=True``.

    This is a NON-STRICT trace (invariant 3): make_fx records only the ATen ops
    that run for THIS example. Static (Python ``int``) control flow and shapes are
    specialized to ``args`` and baked; a data-dependent op (``.item()``, a branch
    over a tensor value) instead raises at capture on the STATIC path, which traces
    under a fake mode with no ShapeEnv, so the value is unknown and unguardable (the
    unbacked path has one, and can capture such a value symbolically). The interning/order
    established here for params then buffers is the calling convention the runtime model
    must reproduce (invariant 2).
    """
    import contextlib

    args = tuple(args)
    # An ambient TracingContext.fake_mode OUTRANKS both mode sources this capture offers
    # make_fx (see the trace site below), and no foreign mode passes
    # allow_fallback_kernels=False, so a meta-less op in an allowlisted namespace would run
    # for real on zero-filled substitutes again. A mode built under DEFAULT config (an
    # AOTAutograd / inductor one) also lacks the
    # fake_tensor_allow_unsafe_data_ptr_access=False snapshot, so a .data_ptr() read bakes 0
    # instead of raising; dynamo and export do build theirs inside that patch, so there only
    # the fallback setting is lost. Refuse rather than trace under someone else's contract.
    tc = TracingContext.try_get()
    if tc is not None and tc.fake_mode is not None:
        raise PrecompileError(
            "precompile: capture cannot run inside another trace -- a TracingContext with "
            "a FakeTensorMode is active (e.g. precompile called from inside torch.compile "
            "or an export/AOTAutograd trace). make_fx would adopt that mode and its "
            "ShapeEnv, so capture's own safety settings would not apply. Capture outside "
            "the enclosing trace."
        )
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
    # to FAKE tensors below (both paths fakeify), but the saved-grad snapshot/clear/restore
    # block must protect the real example model's .grad fields (those are what the user
    # owns and what a backward in fn populates), not the throwaway fakes. list() snapshots
    # the real tensors here, so the later flat_args rebind does not affect real_flat.
    real_flat = list(flat_args)
    # Labels for the per-input refusals here and below, aligned with flat_args. Each names
    # its KIND, because the model half is not an argument the caller can swap out: a
    # refusal saying "buffer nt" sends them to the model, "user input 0" to the call site.
    input_labels = [
        *(f"parameter {n}" for n in param_names),
        *(f"buffer {n}" for n in buffer_names),
        *(f"user input {i}" for i in range(len(user_flat))),
    ]
    # Example inputs a fake tensor cannot faithfully stand in for are refused capture-wide,
    # on BOTH paths and before either fakeifies. This runs ahead of EVERY shape read below
    # -- the param/buffer records and the user-input _dense_shape records alike -- which is
    # what gets a STRIDED nested parameter, buffer or user input the same named refusal
    # rather than the raw "NestedTensorImpl doesn't support sizes" that reading t.shape
    # raises.
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
    # Record each marked dim's declared min/max so the driver enforces them at runtime;
    # the capture-time torch._check on an unbacked symint never becomes a runtime guard,
    # so without this the documented mark_unbacked min/max check would be a silent no-op.
    user_input_bounds = _read_input_bounds(marks)
    # Snapshot and clear the REAL example tensors' .grad BEFORE fakeifying and tracing.
    # A backward in fn accumulates (``p.grad = p.grad + new``), so a live pre-existing
    # grad would be read into the graph and baked by make_fx as a get_attr constant --
    # tripping the invariant-1 guard with a misleading "tensor closed over by fn" error on
    # the common warmup-step-then-precompile flow. The clear MUST precede BOTH fakeify
    # sites (the static loop below and _fakeify_with_unbacked): from_tensor copies .grad
    # onto the fakes we trace on, so clearing the reals first keeps the fakes grad-free
    # too. Restored in finally; precompile does not mutate the user's example .grad
    # (params/buffers AND user inputs).
    # Snapshot the ORIGINAL .grad object (no clone) and restore that SAME object below, so
    # grad IDENTITY is preserved -- a caller holding a prior p.grad reference, or optimizer
    # state keyed on grad identity, is not invalidated. Both paths trace on fakes, so a
    # backward in fn writes only the fakes' .grad and the reals' .grad is never touched by
    # the trace: the sole clobber is our own clear just above, which the finally-restore
    # below undoes, putting the snapshotted object back (identity and value preserved).
    saved_grads = [a.grad if isinstance(a, torch.Tensor) else None for a in real_flat]
    for a in real_flat:
        if isinstance(a, torch.Tensor):
            a.grad = None
    from torch._subclasses.fake_tensor import (
        DataDependentOutputException,
        DynamicOutputShapeException,
        FakeTensorMode,
        UnsupportedOperatorException,
    )

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

    # The trace runs in the ambient grad mode, which the entry point sets: the
    # callable API keeps grad enabled around it, as it always did, while capture()
    # lets ``training=`` select it (grad on for a backward in ``fn``, off for an
    # inference capture) -- so a capture can reach here with grad OFF.
    from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode

    # ``fake_mode`` is the DYNAMIC (symbolic) fake mode -- set only on the unbacked path,
    # where it threads its ShapeEnv to the lowering (and turns on scalar_asserts). The
    # static path also traces on fakes, but with a plain (non-symbolic) fake mode kept in
    # ``capture_cm`` only, so ``_Capture.fake_mode`` stays None and the lowering treats
    # the capture as static. Either way ``capture_cm`` is the FakeTensorMode we trace in.
    fake_mode = None
    capture_cm: FakeTensorMode
    import torch._functorch.config as functorch_config

    # Fakeify INSIDE the try: from_tensor can refuse an input (e.g. a quantized tensor,
    # which has no meta representation), and the finally must still put the caller's
    # example .grad back. A fake-mode backward cannot reach the real .grad at all, so
    # that clear (just above) is the only clobber left to undo.
    try:
        if any(marks):
            flat_args, fake_mode = _fakeify_with_unbacked(
                pb_flat, user_flat, marks, input_labels
            )
            capture_cm = fake_mode
            user_input_shapes = [
                None
                if base is None
                else tuple(None if i in per else s for i, s in enumerate(base))
                for base, per in zip(user_input_shapes, marks)
            ]
        else:
            # Static capture: fakeify every input so the trace runs no real compute
            # (no in-place input mutation, no grad on the example model).
            # allow_non_fake_inputs lets a real tensor that fn closes over (an
            # unregistered attr, a global, a captured constant -- invariant 1) flow
            # through as a baked constant, so _check_no_constant_tensors below
            # rejects it with the same clean PrecompileError a real trace gave,
            # rather than a raw mixed-fake AssertionError. That holds for a
            # REPRESENTABLE closed-over tensor: a quantized or STRIDED-nested one still
            # dies inside the trace with its own raw error (as it did on a real trace), in
            # the meta CONVERSION rather than in an op: make_fx fakeifies that constant for
            # node metadata and the converter represents neither layout.
            # allow_fallback_kernels defaults to True, which would run a meta-less op in
            # an allowlisted namespace (aten, prims, quantized, ...) for real on zero-filled
            # substitutes and bake whatever shape that produced; off, a meta-less op takes
            # the UnsupportedOperatorException path refused below -- unless it is
            # trivially fakeifiable (a mutating op with no returns, for which
            # can_generate_trivial_fake_impl synthesizes one). The mode is built inside
            # the config patch for the reason _fakeify_with_unbacked gives (the config is
            # snapshotted at construction).
            with functorch_config.patch(fake_tensor_allow_unsafe_data_ptr_access=False):
                capture_cm = FakeTensorMode(
                    allow_non_fake_inputs=True, allow_fallback_kernels=False
                )
            with capture_cm:
                flat_args = [
                    _fakeify_input(capture_cm, a, label)
                    if isinstance(a, torch.Tensor)
                    else a
                    for a, label in zip(flat_args, input_labels)
                ]

        # Trace on FAKE tensors either way, so no real compute runs on the flattened
        # inputs -- no in-place mutation of an example input, no grad accumulation on the
        # example model. That is scoped to the flattened inputs and the model's
        # params/buffers: a REAL tensor fn closes over is deliberately NOT converted
        # (allow_non_fake_inputs), so an in-place op on it does execute for real, before
        # invariant 1 refuses the program below. make_fx re-fakeifies the leaves itself (it
        # passes a source, so from_tensor's identity fast-path does not apply), so the
        # graph's placeholder metas are NOT the fakes built above. What the trace must adopt
        # is a shape_env-less fake mode, and there are TWO sources for one here:
        # detect_fake_mode takes the ACTIVE dispatch-stack mode (the "with capture_cm"
        # below) first and falls back to a mode it finds on the arguments (the flat_args
        # pre-fakeified above). Dropping either alone still refuses everything below;
        # dropping BOTH (handing make_fx real tensors with no "with") is what silently gives
        # it its own mode with a fresh ShapeEnv and allow_fallback_kernels back on, turning
        # the value- and shape-producing refusals below (.item(), .nonzero()) into unbacked
        # symints; a tensor-value branch instead surfaces as GuardOnDataDependentSymNode, so
        # that one stays refused, with the unbacked-guard wording. Both are kept, with
        # the "with" as the primary: it also covers a call whose flat_args hold no tensor at
        # all, where there is nothing to detect a mode from. Neither source outranks an
        # ambient TracingContext.fake_mode, which detect_fake_mode takes authoritatively, so
        # a capture inside another trace would run under that mode and none of the settings
        # below -- which is why _capture refuses one up front. What the
        # pre-fakeify loop buys on top is the named per-input refusal above (which reports
        # WHICH example input cannot be fakeified) and the fake flat_args handed to the
        # lowering. The unbacked path keeps its symbolic ShapeEnv ("symbolic"); a static
        # capture uses concrete fake shapes ("fake"). A data-dependent op the fake trace
        # cannot know is refused below rather than leaking the fake exception: a real trace
        # RAISED on .item() and on a tensor-value branch, and CAPTURED .nonzero() into a
        # dynamically-sized graph that a shape_env-less capture cannot express at all
        # (mark_unbacked is the path that can). The conditional is load-bearing the other
        # way round: only the unbacked mode has a ShapeEnv, and make_fx asserts one for
        # "symbolic". The static literal is a declaration rather than a switch -- with the
        # two mode sources above, "real" traces on the same fakes and produces the same
        # artifact; "fake" is what says so, and is what makes make_fx re-fakeify the leaves
        # rather than bake a real one as a constant if the pre-fakeify loop ever goes away.
        tracing_mode = "symbolic" if fake_mode is not None else "fake"
        with capture_cm:
            try:
                gm = make_fx(
                    flat_fn,
                    decomposition_table=decompositions,
                    tracing_mode=tracing_mode,
                )(flat_args)
            except PrecompileError:
                # precompile's own refusals (invariant-5 checks inside flat_fn) are
                # already explained; re-raise them before the clauses below, so the
                # RuntimeError clause cannot re-wrap one on a substring accident
                # (PrecompileError subclasses RuntimeError).
                raise
            except GuardOnDataDependentSymNode as e:
                # An unbacked capture holds both a mark_unbacked dim and a
                # data-dependent value (.item(), .nonzero()) as an unbacked symbol with
                # no hint, so a computation that must guard on one (a branch, a reshape
                # that pins a size) cannot be captured. Unbacked symbols cannot be
                # guarded, so rather than bake a silently-wrong artifact, fail here.
                raise PrecompileError(
                    "precompile: fn needs to guard on a value this capture holds as "
                    "unbacked -- a dim marked with mark_unbacked, or a data-dependent "
                    "value (.item(), .nonzero()) -- which is not allowed. Do not mark "
                    "that dim (capture it static), or restructure fn to avoid the "
                    f"branch. Underlying: {(str(e).splitlines() or [''])[0]}"
                ) from e
            except (DataDependentOutputException, DynamicOutputShapeException) as e:
                # A static capture has no ShapeEnv, so a value the fake trace cannot
                # know surfaces as one of these rather than as
                # GuardOnDataDependentSymNode: .item() and a branch over a tensor
                # raise the first, .nonzero()/masked_select and other shape-producing
                # ops the second. Refuse cleanly instead of leaking either. The
                # mark_unbacked advice is for a STATIC capture only: an unbacked capture
                # already has the ShapeEnv, so the op that reached here (e.g. aten.equal)
                # is one no ShapeEnv can help with, and telling the caller to mark a dim
                # they may already have marked would be a dead end.
                mark_hint = (
                    " A shape-producing op (.nonzero(), masked_select) can be captured by "
                    "marking a user-input dim with torch._dynamo.decorators.mark_unbacked, "
                    "which gives capture the ShapeEnv it needs."
                    if fake_mode is None
                    else ""
                )
                raise PrecompileError(
                    "precompile: fn performs a data-dependent operation (.item(), "
                    ".nonzero(), masked_select, a Python branch over a tensor value) "
                    "whose result cannot be known while tracing on fake tensors, so it "
                    "cannot be captured; make_fx specializes only static (Python int) "
                    "control flow. The op may be inside a module fn calls rather than in "
                    "fn's own code -- nn.BatchNorm*(momentum=None) in train() mode reads "
                    "float(num_batches_tracked) -- so go by the underlying op named below."
                    f"{mark_hint} Underlying: {(str(e).splitlines() or [''])[0]}"
                ) from e
            except AttributeError as e:
                # torch.while_loop's fake kernel unconditionally enters
                # mode.shape_env.ignore_fresh_unbacked_symbols(), so on a static capture
                # (whose fake mode deliberately has no ShapeEnv) it dies here instead of
                # reaching _assert_no_control_flow_subgraphs below, which is what refuses
                # the torch.cond form. Give it that same refusal; any other AttributeError
                # comes from fn and is not ours to explain.
                if "ignore_fresh_unbacked_symbols" not in str(e):
                    raise
                raise _control_flow_refusal(
                    f"Underlying: {(str(e).splitlines() or [''])[0]}"
                ) from e
            except AssertionError as e:
                # The control-flow HOPs assert a ShapeEnv on the mode they trace under,
                # BEFORE the fake kernel the clause above relabels: an int while_loop carry
                # (the spelling in its own docstring) takes the proxy path, which
                # unspecializes it into an unbacked symint, and torch.cond merges differing
                # int values or output sizes across its branches the same way. Same refusal
                # as the get_attr check below; only while_loop/cond raise these messages.
                first = (str(e).splitlines() or [""])[0]
                if first in (
                    "Must provide a fake_mode with shape_env.",
                    "mode.shape_env is None",
                ):
                    raise _control_flow_refusal(f"Underlying: {first}") from e
                # FakeTensorMode declines a small set of device/pinning ops
                # (aten._pin_memory, aten._resize_output) with a bare
                # AssertionError("NYI: <op>") instead of UnsupportedOperatorException, so
                # e.g. the usual "pin if not pinned" idiom in fn escaped raw. It is the
                # missing-fake-kernel condition wearing a different exception type; give it
                # that refusal. The prefix match is exact (one raise site reachable inside
                # make_fx, in fake_impls), so an AssertionError matching neither this
                # prefix nor the two messages above is an internal bug or fn's own and is
                # re-raised; an fn that itself raises AssertionError("NYI: ...") is
                # relabeled, an accepted collision.
                if not first.startswith("NYI: "):
                    raise
                raise _missing_fake_kernel_refusal(f"Underlying: {first}") from e
            except RuntimeError as e:
                # Tracing on fake tensors needs a meta/fake kernel for every op, and no
                # op may read a fake tensor's data pointer. A library op with no kernel
                # raises UnsupportedOperatorException (a RuntimeError subclass, so this
                # clause catches it); a torch.library.custom_op without
                # register_fake raises a RuntimeError naming the missing fake impl; a
                # .data_ptr() read raises one of the two FAKE data-pointer messages matched
                # below (that is what the config patch above buys us). Anything else is not
                # ours to explain.
                first = (str(e).splitlines() or [""])[0]
                # Match the fake-specific texts, not the generic "Cannot access data
                # pointer" prefix: a REAL tensor with no storage (e.g. a sparse tensor fn
                # closes over) raises "...of Tensor that doesn't have storage" from the same
                # c10 code and must reach the caller unrelabeled. The match stops at
                # StorageImpl's "(e.g." rather than spelling out its example list, which is
                # already unambiguous against every sibling message and does not silently
                # stop matching if that list is reordered or extended. Two sites report a fake
                # pointer read: StorageImpl names FakeTensor, while TensorImpl's typed
                # data_ptr_impl (reached by a kernel that dereferences a fake tensor -- e.g.
                # tensor_split with tensor indices) reports uninitialized storage instead.
                # A NumPy conversion (t.numpy(), np.asarray(t), t.__array__(); everyday
                # logging/metric code) is the same read, but tensor_numpy.cpp rejects it
                # earlier and blames "tensor subclasses", which under capture is usually
                # precompile's own FakeTensor and not anything the caller wrote -- so match
                # that text too rather than send them after a subclass that does not exist.
                # The blamed subclass CAN be the caller's (the check is is_python_dispatch(),
                # so any python-dispatch subclass fn builds trips it), which is why the
                # refusal points at Underlying: rather than asserting whose it is. That
                # string has one raise site.
                reads_fake_data = (
                    "Cannot access data pointer of Tensor (e.g." in str(e)
                    or "its data is not allocated yet" in str(e)
                    or ".numpy() is not supported for tensor subclasses" in str(e)
                )
                if reads_fake_data:
                    raise PrecompileError(
                        "precompile: fn reads a tensor's data -- its data pointer "
                        "(.data_ptr(), or a kernel that dereferences one) or a NumPy "
                        "conversion (.numpy(), np.asarray(), __array__) -- which the "
                        "fake-tensor trace cannot provide: the traced tensors have no data "
                        "behind them, and the tensor subclass a NumPy conversion blames may "
                        "be capture's own FakeTensor rather than one you wrote -- check "
                        "Underlying:. Move the read out of fn, or wrap that kernel in a "
                        f"custom op with a registered fake impl. Underlying: {first}"
                    ) from e
                no_fake_impl = "no fake impl registered" in str(e)
                if not isinstance(e, UnsupportedOperatorException) and not no_fake_impl:
                    raise
                raise _missing_fake_kernel_refusal(f"Underlying: {first}") from e
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

    python_code is the single source of truth: ``_build_metadata_section`` emits the
    constants below as top-level literal assignments, so an AST walk + literal_eval
    recovers them safely. The cache then only needs to carry the compiled artifact.
    """
    import ast

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
            except (ValueError, TypeError) as e:
                raise PrecompileError(
                    f"python_code {target.id!r} calling-convention metadata is "
                    f"malformed; it must be a Python literal."
                ) from e
        else:
            # Not a metadata name we consume: the inlined graph section declares module-level
            # names of its own (aten, async_compile, the kernel handles, call), so this fires
            # many times on an inductor artifact. Skipped by design, but logged at debug so a
            # malformed / renamed artifact is diagnosable rather than silently lost.
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
        # make_fx is the only implemented tracer; "dynamo" is a planned alternative
        # capture front-end. Reject it here (the single capture-dispatch point) before
        # running fn, so the failure is clear rather than a wrong default.
        if self._tracer != "make_fx":
            raise NotImplementedError(
                f"precompile tracer={self._tracer!r} is not implemented yet; use "
                "tracer='make_fx' (the default)."
            )
        if self._backend == "eager" and _has_unbacked_marks(args):
            raise NotImplementedError(
                "precompile: mark_unbacked (dynamic shapes) is only supported with "
                "backend='inductor'; eager + unbacked is not supported."
            )
        captured = _capture(self._fn, args, self._decompositions)
        self._module_positions = captured.module_positions
        self._num_positional_args = captured.num_positional_args
        self._param_names = captured.param_names
        self._buffer_names = captured.buffer_names
        self._param_shapes = captured.param_shapes
        self._buffer_shapes = captured.buffer_shapes
        self._param_dtypes = captured.param_dtypes
        self._buffer_dtypes = captured.buffer_dtypes
        self._param_devices = captured.param_devices
        self._buffer_devices = captured.buffer_devices
        self._user_input_shapes = captured.user_input_shapes
        self._user_input_dtypes = captured.user_input_dtypes
        self._user_input_devices = captured.user_input_devices
        self._user_input_bounds = captured.user_input_bounds
        self._in_spec = captured.in_spec
        self._out_spec = captured.out_spec
        self._grad_param_indices = captured.grad_param_indices
        self._gm = captured.gm

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
        if captured.fake_mode is not None and hasattr(_ind_config, "scalar_asserts"):
            options["scalar_asserts"] = True
        try:
            self._graph_python, self._artifact_bytes = aot_autograd.compile_to_python(
                captured.gm, captured.flat_args, options=options
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
        # A PrecompiledModule is runnable only after load(); a capture instead
        # renders (python_code, cache) rather than a runnable.
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


def _make_inlined_forward(
    python_code: str, who: str, *, warn: bool = True
) -> Callable[..., object]:
    """Fallback: execute the self-contained python string (JITs kernels).

    ``python_code`` needs no cache -- the kernels (inductor) or graph (eager) are
    inlined, so we just exec it and hand back its ``forward``. The returned
    ``forward`` takes the same args the traced fn took (model(s) plus runtime
    inputs). ``who`` names the entry point the caller actually called; ``warn`` is off
    only for the capture-time self-load, whose source was produced in-process."""
    # python_code is untrusted EXECUTABLE input -- exec'ing it runs whatever it contains
    # (JIT-compiling inlined kernels or running the inlined graph). Warn per load (not
    # warning_once) before the exec so the inlined fallback is never silent about it.
    if warn:
        log.warning(
            "%s is about to EXEC python_code, which is untrusted executable input (it "
            "runs inlined kernels / graph code). Only exec python_code you produced or "
            "otherwise trust (Note [precompile programming model], invariant 7).",
            who,
        )
    module_ns: dict[str, object] = {"__name__": "_precompiled_artifact"}
    exec(compile(python_code, "<precompile>", "exec"), module_ns)
    return cast("Callable[..., object]", module_ns["forward"])


def _check_path_pair(
    who: str,
    artifact_path: str | os.PathLike[str],
    cache_path: str | os.PathLike[str],
) -> None:
    """Refuse an artifact_path / cache_path pair no entry point can use.

    Three ways it can be unusable: half a pair (the two files only load together), one
    file named for both halves after resolving links (the write would clobber the
    source), and a path that exists but is not a regular file. The ``None`` checks are
    defensive (both parameters are typed) but turn ``os.path.realpath(None)``'s bare
    ``TypeError`` into one naming the pair.
    """
    if artifact_path is None or cache_path is None:
        if artifact_path is None and cache_path is None:
            raise ValueError(
                f"{who} got neither artifact_path nor cache_path; the artifact and "
                f"its cache are a matched pair, pass both."
            )
        given, missing = (
            ("artifact_path", "cache_path")
            if cache_path is None
            else ("cache_path", "artifact_path")
        )
        raise ValueError(
            f"{who} got {given} without {missing}; the artifact and its cache are "
            f"a matched pair, pass both."
        )
    if os.path.normcase(os.path.realpath(artifact_path)) == os.path.normcase(
        os.path.realpath(cache_path)
    ):
        raise ValueError(
            f"{who} got the same file for artifact_path and cache_path "
            f"({os.fspath(artifact_path)!r}); the two halves are separate files."
        )
    for name, path in (("artifact_path", artifact_path), ("cache_path", cache_path)):
        if os.path.exists(path) and not os.path.isfile(path):
            raise ValueError(
                f"{who} got {name}={os.fspath(path)!r}, which is not a regular "
                f"file; each half of the pair is a plain file."
            )


# The os.link failures that mean "this filesystem does not do hard links" (a
# FAT/exFAT mount, a container overlay, a cross-device target), as opposed to one
# about the path itself, which must not be papered over with a move.
_NO_HARD_LINK_ERRNOS = frozenset(
    {errno.EPERM, errno.EOPNOTSUPP, errno.EMLINK, errno.EXDEV}
)


def _same_inode(
    path: str | os.PathLike[str], other: str | os.PathLike[str] | os.stat_result
) -> bool:
    """True when ``path`` resolves now to ``other`` (a path or a recorded stat)."""
    try:
        st = other if isinstance(other, os.stat_result) else os.stat(other)
        return os.path.samestat(os.stat(path), st)
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
    pair intact. Which undo runs, and whether it is reported, is read off the DISK, not
    from flags (the comment on the undo has the reasoning). Process death between the
    renames is not covered, nor a reader or a second writer racing them: that can leave
    one source beside the other's cache, which ``load`` refuses on the cache's sha256,
    and can cost the previous source. The parent directory is fsync'd after, best effort.
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
            # The rename repoints the name at the temp's inode and carries its mode over, so
            # the temp is CREATED with the mode of the file it replaces: chmod'ing it down
            # only after the write would publish the whole new payload at the umask mode, in
            # a directory the caller chose. Permission bits only: setuid/setgid on a new inode
            # owned by the WRITING user name a different principal. No previous file means that
            # default. O_BINARY: os.open on Windows translates the newlines code_hash covers.
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
        # source at every instant, for a racing reader or a crash between the renames.
        try:
            os.link(artifact_path, backup)
        except FileNotFoundError:
            pass
        except OSError as e:
            # No hard links here: fall back to moving aside, only for the errnos that mean
            # unsupported and only for a regular file (os.link on a DIRECTORY also fails EPERM).
            if e.errno not in _NO_HARD_LINK_ERRNOS or not os.path.isfile(artifact_path):
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
            complete = all(map(_same_inode, (artifact_path, cache_path), new_stats))
            kept = os.path.lexists(backup)
            landed = _same_inode(artifact_path, new_stats[0])
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
            if probed and not undone and os.path.lexists(named):
                if kept:
                    log.warning(
                        "precompile could not put the previous artifact back at %s; its "
                        "previous source is kept at %s and the cache at %s does not match "
                        "that file, so the pair does not load until it is rewritten.",
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
            elif kept if probed else _same_inode(artifact_path, backup):
                # Reached only where the named pair came out loadable, so the backup is not a
                # copy anyone still needs: an undo rename consumed it, or nothing needed undoing
                # because the previous source is still under its own name -- a failed FIRST
                # rename, or an interrupt among the reads, which never consults ``kept`` and
                # drops only while that name still resolves to the backup's inode.
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
            # Best effort like the os.open above, close included: fsync on a DIRECTORY fd is
            # not supported everywhere, and by here both renames have returned, so an error
            # out of either would fail a write whose pair is already loadable.
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


def _runnable_from_pair(
    python_code: str,
    cache: bytes,
    *,
    who: str,
    _trusted: bool = False,
) -> PrecompiledRunnable:
    """Reconstruct a runnable from an in-memory ``(python_code, cache)`` pair.

    The shared core of :func:`load`, ``_PrecompileApi.load`` and the capture-time
    self-load in :class:`_MakeFxCapture`, so ``who`` names the entry point the caller
    actually called rather than one of the other two. ``_trusted`` suppresses the exec
    warning, and is set only for that self-load: its source was produced in-process.
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
                "%s got a cache with format=%r version=%r, expected %r / %r; it is "
                "likely from a different torch build. Falling back to JIT from "
                "python_code.",
                who,
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
            "%s could not read the cache envelope (%s: %s); the cache is likely "
            "corrupt or from a different torch build. Falling back to JIT from "
            "python_code.",
            who,
            type(e).__name__,
            e,
        )
    if artifact is not None:
        # Prime the inductor kernel caches from the bundle so the exec of python_code
        # below loads the precompiled kernels (Triton binaries / autotune results) instead
        # of recompiling them: the composed python_code runs its inlined kernels directly,
        # with no compile_fx re-entry and so no FxGraphCache lookup.
        try:
            torch.compiler.load_cache_artifacts(artifact)
        except Exception as e:
            log.warning(
                "%s could not prime the cache from the artifact bundle (%s: %s); it is "
                "likely stale or from a different torch build. Falling back to JIT "
                "from python_code.",
                who,
                type(e).__name__,
                e,
            )
    # Run the driver inlined in python_code. It carries the full calling convention and
    # runtime safety checks (subclass wrap/unwrap, param/buffer lifting, grad harvest,
    # input/model validation) and JITs the kernels, which hit the cache primed above.
    forward = _make_inlined_forward(python_code, who, warn=not _trusted)
    return PrecompiledModule._from_loaded(forward, backend=backend)


def capture(
    fn: Callable[..., object],
    /,
    *,
    artifact_path: str | os.PathLike[str],
    cache_path: str | os.PathLike[str],
    tracer: MakeFxTracer = MakeFxTracer(),
    backend: str = "inductor",
    training: bool = False,
) -> Capture:
    """Capture ``fn`` across YOUR loop's calls; the pair is written on block exit.

    .. warning::

        This is a prototype API. Its signature, error types and artifact
        format may change between releases without a deprecation cycle.

    Capture is caller-driven: this returns a capture object rather than running
    anything. Enter it as a context manager, call it with the positional arguments
    ``fn`` takes inside the block (keyword arguments are refused) -- the call runs
    for real, is folded into the capture, and returns what serving the artifact
    produces -- and the ``(python_code, cache)`` artifact is written to
    ``artifact_path`` / ``cache_path`` when the block exits CLEANLY: a block that
    RAISED writes nothing, and a clean exit that never called the capture raises
    instead of writing. These names will be exported from
    ``torch.compiler.precompile``; until then import them from here::

        from torch._precompile import capture, load

        with capture(fn, artifact_path="m.py", cache_path="m.cache") as cap:
            y = cap(model, x)
        f = load("m.py", "m.cache")

    :class:`MakeFxTracer` is one non-strict ATen trace, so the capture takes
    exactly ONE call and refuses a second; ``MakeFxTracer.decompositions``
    forwards a decomposition table to ``make_fx``. ``training`` selects the grad
    mode of the captured call, whatever the ambient mode: ``training=True`` runs
    it under ``torch.enable_grad()`` so a ``.backward()`` in ``fn`` is captured
    and the artifact scatters the gradients onto the model's ``.grad`` fields;
    ``training=False`` runs it under ``torch.no_grad()``. Either way the value
    ``cap(...)`` returns is the served result: a computed output does not require
    grad, and an output that IS an input or parameter comes back as that same tensor,
    hence with its own ``requires_grad``. An output that ALIASES an INPUT -- a view of
    it, ``t.detach()`` included, which the trace records as a view -- is instead REBUILT
    at serve time and its ``requires_grad`` is not part of the contract (eager takes it
    from the runtime input, inductor from the capture), so set it yourself if you depend
    on it. ``backend`` picks ``"inductor"`` (lower through AOTAutograd + Inductor into
    self-contained source plus an acceleration cache) or ``"eager"`` (inline the captured
    ATen graph as readable source). Call ``cap.save()`` inside the block to write the
    files before it exits; after a WRITE that failed -- the exit's or a ``save()``'s --
    call it again to retry that write, from outside the block too. The contract is Note
    [precompile programming model] in this module; see :func:`load` for reading the pair
    back.

    Raises ``ValueError`` for a ``backend`` outside ``{"inductor", "eager"}`` and for a
    path pair no entry point can use (half a pair, one file named for both halves, a path
    that is not a regular file); ``TypeError`` for a non-:class:`MakeFxTracer` ``tracer``;
    ``PrecompileError`` for an ``fn`` that HOLDS a model or a tensor instead of taking it
    as a call argument (a bound method, a partial that holds one).
    """
    # Keyed on the module these live in TODAY; the module switch moves both keys to
    # the public torch.compiler.precompile.* spelling.
    torch._C._log_api_usage_once("torch._precompile.capture")
    if backend not in ("inductor", "eager"):
        raise ValueError(
            f"precompile backend must be 'inductor' or 'eager', got {backend!r}."
        )
    _check_path_pair("torch._precompile.capture", artifact_path, cache_path)
    if isinstance(tracer, MakeFxTracer):
        return _MakeFxCapture(
            fn,
            artifact_path,
            cache_path,
            backend=backend,
            decompositions=tracer.decompositions,
            training=bool(training),
        )
    raise TypeError(
        f"precompile.capture tracer must be a MakeFxTracer, got {type(tracer).__name__}."
    )


# This ``load`` takes two PATHS; ``_PrecompileApi.load`` (the one reachable today as
# torch.compiler.precompile.load) takes the in-memory (python_code, cache) pair -- same
# name, same arity. The in-memory one retires with the callable API at the module switch.
def load(
    artifact_path: str | os.PathLike[str],
    cache_path: str | os.PathLike[str],
    /,
) -> PrecompiledRunnable:
    """Reconstruct a runnable from the two files a precompile capture wrote.

    .. warning::

        This is a prototype API. Its signature, error types and artifact
        format may change between releases without a deprecation cycle.

    Name the two files :func:`capture` wrote -- the ``python_code`` artifact and its
    ``cache``. They load only as a matched pair (the cache carries a sha256 of exactly
    the python_code bytes it was emitted with).

    The driver runs from ``python_code`` -- the single source of truth for the whole
    calling convention. ``load`` reads ``BACKEND`` out of ``python_code``'s metadata
    (the cache's ``backend`` tag is compared against it, and its ``code_hash`` against
    the source, so a cache from another capture is refused) and, for the inductor
    backend, primes the inductor kernel caches from the cache's ``save_cache_artifacts``
    bundle (via ``torch.compiler.load_cache_artifacts``) so a warm reload loads
    precompiled kernels instead of JIT-compiling; then it exec's ``python_code``. With
    no usable cache it degrades to JIT'ing from ``python_code``.

    Call the result with the SAME argument structure ``fn`` took -- the model(s) in their
    original positions plus the runtime inputs. Per invariant 2 of Note [precompile
    programming model], the runtime model must match the example model's parameter/buffer
    structure; precompile re-derives the param/buffer list from it (same order).

    The result is a :class:`PrecompiledRunnable`: a make_fx artifact is standalone, so it
    installs nothing, its ``with`` / ``unload()`` are no-ops and ``installed`` is
    ``False``.

    Raises ``PrecompileError`` if ``python_code`` is malformed or is not a
    ``torch.compiler.precompile`` artifact (it fails to parse, or is missing the
    calling-convention metadata), if the cache's ``backend`` tag does not match
    ``python_code``, or if the cache's ``code_hash`` does not match
    ``sha256(python_code)`` -- i.e. the cache and python_code came from different
    precompile captures. A cache whose ``format``/``version`` does not match (a
    foreign or different-build envelope) is NOT fatal: the cache is acceleration
    only, so ``load`` degrades to JIT'ing from ``python_code`` rather than crashing.
    A half that cannot be READ or decoded -- a missing file, the two paths passed the
    wrong way round -- is a ``PrecompileError`` too, with the original error as its
    ``__cause__``. A pair no entry point can use raises ``ValueError`` instead: half a
    pair, one file named for both halves, or a path that is not a regular file.
    """
    torch._C._log_api_usage_once("torch._precompile.load")
    _check_path_pair("torch._precompile.load", artifact_path, cache_path)
    python_code, cache = _read_artifact(artifact_path, cache_path)
    return _runnable_from_pair(python_code, cache, who="torch._precompile.load")


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
        *example_inputs: object,
        backend: str = "inductor",
        tracer: str = "make_fx",
        decompositions: dict | None = None,
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

        THREADING: the inductor lowering step drives process-global compiler state
        and is serialized by an internal lock, so concurrent ``backend="inductor"``
        calls lower one at a time. The make_fx capture phase and the ``backend="eager"``
        path are NOT serialized.

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
          that actually run when ``fn`` executes once on FAKE tensors derived from the
          example inputs (the example tensors themselves are never computed on) and does
          not analyze your Python, so Python control flow and shapes are specialized to
          the example (the source of the programming-model contract). The only tracer
          implemented today.
        - ``"dynamo"``: planned (a Dynamo-based front-end that analyzes the Python);
          raises ``NotImplementedError`` for now.

        ``decompositions`` is an optional decomposition table (a dict mapping each
        ``OpOverload`` to a decomposition function) forwarded to ``make_fx`` as its
        ``decomposition_table`` during capture, so you can control how ATen ops are
        broken down in the captured graph. Defaults to ``None`` (make_fx's default).

        Dynamic shapes are opt-in via ``torch._dynamo.decorators.mark_unbacked``
        (inductor backend only), NOT a precompile kwarg: mark dims on the inputs before
        calling, e.g. ``mark_unbacked(x, 0); precompile(fn, model, x)`` frees ``x``'s
        batch dim. Marked dims are captured as UNBACKED symints, which cannot be guarded
        on, so one artifact serves any runtime size of them (invariant 3); a graph that
        needs to guard on / specialize a marked dim fails at capture with a
        ``PrecompileError``. Dims sharing a ``shape_id`` reuse one symbol (equal by
        construction); ``min``/``max`` become runtime asserts. Other dims stay static.
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
                lambda model, x: model(x), model, x
            )


            def train_step(model, x, t):
                loss_fn(model(x), t).backward()  # or return autograd.grad(...)


            python_code, cache = torch.compiler.precompile(train_step, model, x, t)

        Among ``example_inputs``, the ``nn.Module`` arguments have their params/buffers
        lifted to graph inputs (no weights are baked into the artifact -- invariant 1);
        the rest are the runtime inputs. The reloaded callable is invoked with the SAME
        argument structure -- pass the model(s) again at runtime, e.g.
        ``f_c(model, x)``, and that runtime model must match the example model's
        parameter/buffer structure (invariant 2). Arguments are matched POSITIONALLY:
        pass the model(s) and inputs positionally both here and at load time; keyword-
        argument calling conventions are not supported (a fn that relies on them would
        surface as a raw arity error). If ``fn`` ran a backward, the
        resulting parameter gradients are scattered (accumulated) onto that runtime
        model's ``parameters()`` ``.grad`` fields, exactly like eager ``.backward()``,
        so a ``zero_grad()`` / ``optimizer.step()`` loop works unchanged; the artifact
        returns ``fn``'s own result (``None`` for a bare ``.backward()`` step), not the
        grads (invariant 5).

        Input mutation (incl. module buffers, e.g. BatchNorm running stats in
        training mode), tensor subclasses (e.g. DTensor), and outputs aliasing inputs
        are supported -- AOTAutograd's prelude/epilogue is composed into the artifact
        (invariant 4), as is functionalized RNG. Caller responsibilities NOT checked
        here (see the Note): the runtime model must be structurally identical to the
        example, and Python control flow / shapes are specialized to ``example_inputs``
        (invariants 2 and 3). Violations that ARE checked raise ``PrecompileError``: a
        tensor baked as a constant (invariant 1), effectful ops (invariant 4), a
        data-dependent op a static (fake-tensor) capture cannot know -- ``.item()``,
        ``.nonzero()``, a Python branch over a tensor value -- an op with no meta/fake
        kernel and none that can be synthesized (a ``torch.library.custom_op`` tagged
        ``torch.Tag.inplace`` / ``torch.Tag.out``, or one that only mutates its arguments
        and returns nothing, gets a trivial one, so it captures without a
        ``register_fake``), a read of a traced tensor's data (``.data_ptr()``,
        ``.numpy()`` / ``np.asarray()``), an example input the fake trace cannot represent
        (a quantized tensor) or whose metadata it silently drops (a pinned, mkldnn or
        sparse tensor), a nested example input, none of which capture supports on either
        path (invariant 3), a control-flow HOP (``torch.cond`` / ``torch.while_loop``),
        which is refused rather than specialized because neither backend can lower the
        subgraph it captures, a capture attempted inside another trace (an ambient
        ``TracingContext`` fake mode outranks capture's own and does not carry its
        fallback-off setting), and -- for the inductor backend -- a runtime input whose
        stride / memory format differs from the example's (invariant 6).
        """
        torch._C._log_api_usage_once("torch.compiler.precompile")
        if backend not in ("inductor", "eager"):
            raise ValueError(
                f"precompile backend must be 'inductor' or 'eager', got {backend!r}."
            )
        if tracer not in ("make_fx", "dynamo"):
            raise ValueError(
                f"precompile tracer must be 'make_fx' or 'dynamo', got {tracer!r}."
            )
        compiled = PrecompiledModule(
            fn, backend=backend, tracer=tracer, decompositions=decompositions
        )
        # Grad stays enabled around the trace whatever the ambient mode, as it always
        # was for this entry point, so a ``.backward()`` in ``fn`` builds graph ops
        # and the artifact does not depend on where the call happened.
        with torch.enable_grad():
            compiled._compile(example_inputs)
        # Build the (expensive) python_code ONCE and thread it into to_cache_bytes so
        # the full metadata + embedded kernel source is not rebuilt, and so code_hash is
        # sha256 over exactly the bytes returned to the caller (a matched pair loads).
        python_code = compiled.to_python_code()
        return python_code, compiled.to_cache_bytes(python_code)

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
        calling-convention metadata), if the cache's ``backend`` tag does not match
        ``python_code``, or if the cache's ``code_hash`` does not match
        ``sha256(python_code)`` -- i.e. the cache and python_code came from different
        precompile captures. A cache whose ``format``/``version`` does not match (a
        foreign or different-build envelope) is NOT fatal: the cache is acceleration
        only, so ``load`` degrades to JIT'ing from ``python_code`` rather than crashing.
        """
        return _runnable_from_pair(
            python_code, cache, who="torch.compiler.precompile.load"
        )


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
