"""Ahead-of-time precompilation (``make_fx`` tracer by default; Dynamo planned).

    python_code, cache = torch.compiler.precompile(fn, model, *example_inputs)
    f_c = torch.compiler.precompile.load(python_code, cache)
    out = f_c(model, *example_inputs)   # pass the model again at runtime

precompile captures your computation with ``make_fx`` -- a NON-STRICT trace of the ATen
ops that run when ``fn`` executes once on the example inputs. It does not analyze your
Python, so it comes with an explicit contract (the programming model): stay inside it
and the artifact faithfully reproduces ``fn``; step outside it and you get an artifact
that computes the wrong thing.

This module implements the ``backend="eager"`` path: ``precompile`` returns a
self-contained, executable ``python_code`` string that inlines the captured graph, plus
a companion integrity-tagged ``cache``. Reload with
``torch.compiler.precompile.load(python_code, cache)``.

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
#    the same shapes/dtypes/devices are the intended use (the eager backend is
#    layout-flexible).
#
# 3. Control flow and shapes are specialized to the example. A non-strict trace follows
#    the single path taken for the example inputs: Python ``if``/``for`` over tensor
#    values, ``.item()``, and shape-dependent branching are resolved at trace time and
#    baked. Shapes are STATIC for now (capture uses make_fx in its "real" mode, so each
#    size is baked as a constant); inputs that would take a different path, or a different
#    shape, yield a wrong result (a static-shape mismatch is rejected up front by the
#    driver). Dynamic-shape support (symbolic sizes that need not be retraced per shape)
#    is planned in a follow-up later in this stack.
#    Each dense user-input leaf's DTYPE and DEVICE are also baked at capture: a runtime
#    input whose dtype or device differs from the example is rejected up front with a
#    PrecompileError, since the graph is specialized to them. Control flow is NOT enforced
#    -- this is the defining property of a non-strict trace. Capture also EXECUTES ``fn``
#    once on the example inputs, so any in-place mutation of an input or other side effect
#    ``fn`` performs (e.g. ``x.add_(1)``, printing, RNG advancement) happens to the
#    example inputs / external state at capture time; pass throwaway example inputs if
#    that matters.
#
# 4. Boundary effects. The eager backend runs the captured ATen graph as-is on the
#    (subclass-level) inputs, so tensor subclasses (e.g. DTensor) pass through unchanged.
#    Effectful ops are not supported yet and raise at capture time (_assert_supported)
#    with a concrete reason; this is an implementation gap, not a fundamental limit.
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
# 6. Each dense user-input leaf's shape, dtype, and device are recorded at capture and
#    checked at runtime: a shape / dtype / device-mismatched input is rejected with a
#    PrecompileError rather than crashing deep in a kernel or reading a wrong value. The
#    graph is specialized to the example input shapes (invariant 3); a different runtime
#    shape is undefined and rejected. The eager backend is layout-flexible (it does not
#    specialize on stride / memory format).
#
# 7. Both python_code and the cache are trusted, EXECUTABLE input to load(). The cache
#    outer envelope is a plain {"artifact": bytes, ...} dict (read with
#    weights_only=True) carrying a format/version + backend tag AND a code_hash
#    (sha256 of the python_code it accompanies) that load() verifies (raising
#    PrecompileError on mismatch). load() then EXECs python_code -- treat it like code
#    you are about to run. The code_hash binds the cache to its python_code: load()
#    rejects a (code, cache) pair from different precompile() calls (same backend)
#    rather than running under foreign metadata.
#
# self-contained (eager backend): ``python_code`` runs on its own -- it inlines the
# captured graph plus all calling-convention metadata, and at RUNTIME it never reads the
# cache (the eager backend has no kernels, so artifact=None and the driver just runs the
# inlined ATen graph). This is an eager-backend property, NOT a general one: the inductor
# backend's generated python_code will carry @triton_heuristics-decorated kernels that
# consult a kernel cache and JIT on a miss, so there python_code is coupled to the cache
# (a miss recompiles rather than failing). Regardless of backend, python_code is the
# SINGLE SOURCE OF TRUTH for the calling convention: that metadata lives in one place
# (python_code), and the envelope carries a code_hash (sha256 of python_code) alongside
# the format/version + backend tag, so load() rejects a (python_code, cache) pair that
# did not come from the same precompile() call. The eager cache is still a full
# integrity-tagged envelope (just with artifact=None), and load() reads the metadata back
# out of python_code (parsing, not exec'ing) rather than from the cache.
#
# backend: "eager" (the only backend today) skips lowering and runs the captured ATen
# graph as-is (analogous to torch.compile(backend="eager")), for inspecting or debugging
# exactly what was traced. An inductor backend is added in a follow-up.
#
# tracer: the capture front-end, orthogonal to backend. "make_fx" (default) is a
# non-strict trace and is the only tracer implemented today -- everything above (the
# invariants, the contract) describes its behavior. "dynamo" is planned (a Dynamo-based
# front-end that analyzes Python rather than specializing to one traced path) and
# currently raises NotImplementedError.

from __future__ import annotations

import hashlib
import io
import logging
from typing import cast, NewType, TYPE_CHECKING

import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


log = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Callable


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


class PrecompileError(RuntimeError):
    """The error type raised by ``torch.compiler.precompile`` and its artifacts.

    Raised when capture, ``load``, or a runtime call violates the precompile contract --
    e.g. a tensor baked as a constant (invariant 1), an unsupported / effectful op, or a
    runtime input whose shape / dtype / device differs from the example (invariants 3 and
    6). See Note [precompile programming model] in this module for the full contract.
    """


def _dense_shape(t: object) -> tuple[int, ...] | None:
    """Return the shape of a plain dense tensor, else ``None`` (non-tensor / subclass).

    Tensor subclasses (e.g. DTensor) carry an outer shape that is not a plain dense
    shape; record ``None`` and skip them in the shape check.
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
    not on the standalone ``_GraphSelf`` holder), so the artifact would crash at
    runtime. Fail at capture with a concrete reason instead, like ``_assert_supported``.
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
    here is load-bearing and is reproduced VERBATIM by the embedded
    ``_extract_param_buffers`` in _EAGER_DRIVER_SOURCE (the inlined eager load path), so
    the two must stay in sync.
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
    flat_args = [*pb_flat, *user_flat]
    # The REAL example tensors (params/buffers and user inputs). The saved-grad
    # snapshot/clear/restore block below protects the real example model's .grad fields
    # (those are what the user owns and what a backward in fn populates).
    real_flat = list(flat_args)
    # Record the example user inputs' dense shapes/dtypes/devices so the driver can
    # reject a shape (invariant 3) or dtype/device mismatch up front; see the inlined
    # _EAGER_DRIVER_SOURCE checks. Subclasses -> None.
    user_input_shapes = [_dense_shape(t) for t in user_flat]
    user_input_dtypes = [_dense_dtype(t) for t in user_flat]
    user_input_devices = [_dense_device(t) for t in user_flat]

    # Snapshot and clear the REAL example tensors' .grad BEFORE tracing. A backward in fn
    # accumulates (``p.grad = p.grad + new``), so a live pre-existing grad would be read
    # into the graph and baked by make_fx as a get_attr constant -- tripping the
    # invariant-1 guard with a misleading "tensor closed over by fn" error on the common
    # warmup-step-then-precompile flow. Restored in finally; precompile does not mutate the
    # user's example .grad (params/buffers AND user inputs). Snapshot the ORIGINAL .grad
    # object (no clone) and restore that SAME object below, so grad IDENTITY is preserved --
    # a caller holding a prior p.grad reference, or optimizer state keyed on grad identity,
    # is not invalidated. Tracing runs on the real interned params, so a backward in fn DOES
    # write .grad in place -- but onto a fresh grad object, since .grad was snapshotted and
    # cleared to None just above; the finally-restore puts the snapshotted object back.
    saved_grads = [a.grad if isinstance(a, torch.Tensor) else None for a in real_flat]
    for a in real_flat:
        if isinstance(a, torch.Tensor):
            a.grad = None

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
    try:
        with torch.enable_grad():
            gm = make_fx(
                flat_fn,
                decomposition_table=decompositions,
                tracing_mode="real",
            )(flat_args)
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
        user_input_shapes: list[tuple[int, ...] | None],
        user_input_dtypes: list[str | None],
        user_input_devices: list[str | None],
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
        # subclass leaf); the driver rejects a runtime mismatch (invariants 3 and 6).
        f"USER_INPUT_SHAPES = {compiled._user_input_shapes!r}",
        f"USER_INPUT_DTYPES = {compiled._user_input_dtypes!r}",
        f"USER_INPUT_DEVICES = {compiled._user_input_devices!r}",
        "",
    ]
    return parts


def _parse_artifact_metadata(python_code: str) -> dict[str, object]:
    """Read the calling-convention constants back out of ``python_code`` WITHOUT
    executing it (exec'ing the inlined graph would run it, not just read metadata).

    python_code is the single source of truth: ``_build_metadata_section`` emits the
    constants below as top-level literal assignments, so an AST walk + literal_eval
    recovers them safely.
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
            found[target.id] = ast.literal_eval(node.value)
        else:
            # Not a metadata name we consume (e.g. a driver-internal top-level like
            # _MODULE_POSITIONS_SET). Skipped by design, but log it at debug so a
            # malformed / renamed artifact is diagnosable rather than silently dropped.
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
    parts.append(_EAGER_DRIVER_SOURCE)
    return "\n".join(parts)


_EAGER_DRIVER_SOURCE = '''
def _extract_param_buffers(mods):
    """Lift the runtime modules' params then buffers, interning by identity, in the
    same order as capture, so the list lines up with the captured graph. Returns
    (pb, names) where names mirrors PARAM_NAMES + BUFFER_NAMES. This ordering AND the
    naming must match torch._precompile._intern_param_buffers verbatim (its INVARIANT)."""
    multi = len(mods) > 1
    seen = set()
    pb = []
    names = []
    def intern(mi, n, t):
        if id(t) not in seen:
            seen.add(id(t))
            pb.append(t)
            names.append(("m%d.%s" % (mi, n)) if multi else n)
    for mi, m in enumerate(mods):
        for n, p in m.named_parameters(remove_duplicate=False):
            intern(mi, n, p)
    for mi, m in enumerate(mods):
        for n, b in m.named_buffers(remove_duplicate=False):
            intern(mi, n, b)
    return pb, names


def _fail(msg):
    # Imported lazily (only when a guard fails) so a normal run does not couple the
    # standalone artifact to torch._precompile's import surface.
    from torch._precompile import PrecompileError as _PrecompileError

    raise _PrecompileError(msg)


def _check_structure(pb, names):
    # Verify the runtime model's extracted param/buffer NAMES match the baked
    # PARAM_NAMES + BUFFER_NAMES (count AND order/identity), so a reordered or
    # structurally-drifted same-count model is caught precisely (invariant 2) rather
    # than scattering grads onto the wrong slot. Then check each tensor's SHAPE, DTYPE and
    # DEVICE against the baked example: the graph is specialized to the example shapes and
    # can bake a device literal, so a same-named but differently shaped/typed/placed
    # runtime tensor would silently miscompute or fail deep in a kernel.
    expected = list(PARAM_NAMES) + list(BUFFER_NAMES)  # noqa: F821
    if names != expected:
        _fail(
            "precompile: the runtime model's param/buffer names %r do not match the "
            "traced model's %r; the runtime model must be structurally identical to the "
            "traced model (invariant 2)." % (names, expected)
        )
    expected_shapes = list(PARAM_SHAPES) + list(BUFFER_SHAPES)  # noqa: F821
    expected_dtypes = list(PARAM_DTYPES) + list(BUFFER_DTYPES)  # noqa: F821
    expected_devices = list(PARAM_DEVICES) + list(BUFFER_DEVICES)  # noqa: F821
    for _nm, _t, _shp, _dt, _dev in zip(
        names, pb, expected_shapes, expected_dtypes, expected_devices
    ):
        if tuple(_t.shape) != tuple(_shp):
            _fail(
                "precompile: the runtime param/buffer %r has shape %s but the traced "
                "model's was %s; the runtime model must be structurally identical to the "
                "traced model (invariant 2)." % (_nm, tuple(_t.shape), tuple(_shp))
            )
        if str(_t.dtype) != _dt:
            _fail(
                "precompile: the runtime param/buffer %r has dtype %s but the traced "
                "model's was %s; the runtime model must be structurally identical to the "
                "traced model (invariant 2)." % (_nm, str(_t.dtype), _dt)
            )
        if str(_t.device) != _dev:
            _fail(
                "precompile: the runtime param/buffer %r is on device %s but the traced "
                "model's was %s; the runtime model must be structurally identical to the "
                "traced model (invariant 2)." % (_nm, str(_t.device), _dev)
            )


_MODULE_POSITIONS_SET = set(MODULE_POSITIONS)  # noqa: F821


def forward(*args):
    """Run the captured ATen graph eagerly. Pass the same args the traced fn took --
    the module(s) in the same positions plus the runtime inputs. The module(s) must
    be structurally identical to the ones precompile traced (same param/buffer order
    and tying); only the weight values may differ.

    The eager backend runs the graph as captured: inputs (including tensor
    subclasses) are passed through unchanged (no dense flatten/unflatten), and the
    graph's flat outputs are reassembled into fn's output structure. If fn ran a
    backward, the trailing grad outputs (one per GRAD_PARAM_INDICES entry) are
    parameter grads, scattered (accumulated) onto the params that received one like
    eager .backward() -- frozen / non-contributing params keep .grad = None."""
    if len(args) != NUM_POSITIONAL_ARGS:  # noqa: F821
        _fail(
            "precompile: expected %d positional args (the same as the traced fn), got "
            "%d (invariant 2)." % (NUM_POSITIONAL_ARGS, len(args))  # noqa: F821
        )
    mods = []
    for _i in MODULE_POSITIONS:  # noqa: F821
        if not isinstance(args[_i], _torch.nn.Module):
            _fail(
                "precompile: argument at position %d must be the nn.Module the traced "
                "fn took (invariant 2), got %s." % (_i, type(args[_i]).__name__)
            )
        mods.append(args[_i])
    user_inputs = [a for i, a in enumerate(args) if i not in _MODULE_POSITIONS_SET]  # noqa: F821
    user_flat, _runtime_in_spec = _pytree.tree_flatten(tuple(user_inputs))
    if IN_SPEC is not None and _runtime_in_spec != _pytree.treespec_loads(IN_SPEC):  # noqa: F821
        _fail(
            "precompile: runtime inputs have a different structure than the traced "
            "example inputs (invariant 3); they must match in nesting and count."
        )
    # Reject a SHAPE / DTYPE / DEVICE mismatch (invariants 3 and 6) up front. The eager
    # backend is layout-flexible, so stride / memory format is not checked.
    if len(user_flat) != len(USER_INPUT_SHAPES):  # noqa: F821
        _fail(
            "precompile: runtime inputs flattened to a different number of leaves than "
            "the traced example (invariant 3); they must match the traced structure."
        )
    for _t, _shp, _dt, _dev in zip(
        user_flat, USER_INPUT_SHAPES, USER_INPUT_DTYPES, USER_INPUT_DEVICES  # noqa: F821
    ):
        if _shp is None or not isinstance(_t, _torch.Tensor):
            continue
        _act = tuple(_t.shape)
        if len(_act) != len(_shp) or any(a != e for a, e in zip(_act, _shp)):
            _fail(
                "precompile: a runtime input has shape %s but the artifact was traced "
                "with shape %s; the graph is specialized to the static dims (invariant "
                "3). Retrace for this shape, or use backend='eager'." % (_act, tuple(_shp))
            )
        if _dt is not None and str(_t.dtype) != _dt:
            _fail(
                "precompile: a runtime input has dtype %s but the artifact was traced "
                "with dtype %s; the graph is specialized to the example dtype "
                "(invariant 6). Cast the input to the traced dtype, or retrace."
                % (str(_t.dtype), _dt)
            )
        if _dev is not None and str(_t.device) != _dev:
            _fail(
                "precompile: a runtime input is on device %s but the artifact was traced "
                "on device %s; the graph is specialized to the example device "
                "(invariant 6). Move the input to the traced device, or retrace."
                % (str(_t.device), _dev)
            )
    pb, _names = _extract_param_buffers(mods)
    _check_structure(pb, _names)
    with _torch.no_grad():
        out = list(call([*pb, *user_flat]))  # noqa: F821
    if GRAD_PARAM_INDICES:  # noqa: F821
        n = len(GRAD_PARAM_INDICES)  # noqa: F821
        grads = out[len(out) - n:]
        out = out[:len(out) - n]
        for idx, g in zip(GRAD_PARAM_INDICES, grads):  # noqa: F821
            p = pb[idx]
            if p.grad is None:
                p.grad = g
            else:
                p.grad.add_(g)
    return _pytree.tree_unflatten(out, _pytree.treespec_loads(OUT_SPEC))  # noqa: F821


if __name__ == "__main__":
    print("forward() is ready; call it with the model(s) and inputs the traced")
    print("fn took, e.g. forward(model, x).")
'''


def _assert_supported(gm: torch.fx.GraphModule) -> None:
    """Enforce invariant 4 of Note [precompile programming model]: reject boundary
    effects the standalone eager artifact does not handle. Detected directly from the
    captured graph.

    Effectful ops are not supported yet (an implementation gap, not a fundamental
    limit), so raise here with a concrete reason rather than let the failure surface
    later. See _unsupported for the mechanical cause.
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
        "effectful op, which is not supported yet: its with_effects HOP cannot be "
        "lowered to standalone source."
    )


class PrecompiledModule:
    """Internal holder for a precompiled computation / a loaded runnable."""

    def __init__(
        self,
        fn: Callable[..., object],
        *,
        backend: str = "eager",
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
        # The eager backend has no kernels to accelerate, so the cache carries no compiled
        # artifact (always None here); it is still a full integrity-tagged envelope.
        self._artifact_bytes: bytes | None = None
        # Which unique-param index each emitted (trailing) grad output belongs to; its
        # length is the number of grad outputs. Lets the driver scatter grads onto
        # exactly the params that received one, leaving frozen / non-contributing
        # params' .grad as None.
        self._grad_param_indices: list[int] = []
        # Per user-input-leaf example shape, dtype, and device (None for a subclass /
        # non-tensor leaf); the driver rejects a runtime mismatch (invariants 3 and 6).
        # Populated by _compile().
        self._user_input_shapes: list[tuple[int, ...] | None] = []
        self._user_input_dtypes: list[str | None] = []
        self._user_input_devices: list[str | None] = []
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
        self._in_spec = capture.in_spec
        self._out_spec = capture.out_spec
        self._grad_param_indices = capture.grad_param_indices
        self._gm = capture.gm
        # The eager backend does no Inductor lowering: the captured ATen graph IS the
        # artifact. It is run directly on the (subclass-level) inputs, so there is no
        # inductor ``call`` to inline and no dense flatten/unflatten -- the graph runs
        # exactly as captured (see Note [precompile programming model]).

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
        "self-contained"). For the eager backend it embeds the captured ATen graph
        (both readable and executable) plus a driver that runs it eagerly and a
        ``forward()`` that takes the same args the traced fn took (the model(s) plus
        runtime inputs). No weights are embedded.
        """
        if self._loaded_forward is not None:
            raise PrecompileError(
                "this object was produced by torch.compiler.precompile.load(); the "
                "python_code you passed in is the source artifact (load() does not "
                "re-capture, so there is no python_code to re-emit from this object)."
            )
        if self._gm is None:
            raise PrecompileError("internal: not compiled; call _compile() first")
        return _build_eager_python_source(self)

    def to_cache_bytes(self, python_code: str | None = None) -> bytes:
        """Return the binary cache as bytes -- an ACCELERATION, not required to run.

        ``python_code`` is the single source of truth for the calling convention, so the
        cache holds only the compiled artifact (None for the eager backend) plus the
        integrity tag and code_hash. See Note [precompile programming model], invariant 7.

        ``python_code`` defaults to what ``to_python_code()`` would emit; ``__call__``
        threads in the exact string it already built so code_hash matches the bytes
        returned to the user and the metadata is not rebuilt.
        """
        # Eager has no kernels, so _artifact_bytes is always None; the envelope is a plain
        # str/int/bytes dict (weights_only-safe) carrying the tag + code_hash that binds it
        # to python_code (invariant 7).
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
    """Execute the self-contained python string and hand back its ``forward``.

    ``python_code`` needs no cache -- the captured graph (eager) is inlined, so we just
    exec it and hand back its ``forward``. The returned ``forward`` takes the same args
    the traced fn took (model(s) plus runtime inputs)."""
    # python_code is untrusted EXECUTABLE input -- exec'ing it runs whatever it contains
    # (running the inlined graph). Warn per load (not warning_once) before the exec so the
    # load path is never silent about it.
    log.warning(
        "torch.compiler.precompile.load is about to EXEC python_code, which is untrusted "
        "executable input (it runs inlined graph code). Only exec python_code "
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
        backend: str = "eager",
        tracer: str = "make_fx",
        decompositions: dict | None = None,
    ) -> tuple[str, bytes]:
        """Ahead-of-time precompile ``fn`` against ``example_inputs``.

        .. note::

            ``torch.compiler.precompile`` is NOT
            ``torch._dynamo.config.caching_precompile`` (a ``torch.compile`` caching
            mode); it captures ``fn`` ahead of time to a self-contained source artifact.

        With the default ``make_fx`` tracer this is a non-strict trace with an explicit
        contract; read Note [precompile programming model] before using it. The artifact
        faithfully reproduces ``fn`` only for callers that uphold that contract.

        ``backend`` selects how the captured graph is realized:

        - ``"eager"`` (default): keep the captured ATen graph and run it as-is
          (analogous to ``torch.compile(backend="eager")``). ``python_code`` inlines
          the readable captured graph (both the inspectable rendering and the executable
          artifact); the eager cache carries no compiled artifact (artifact=None) but is
          still a full integrity-tagged envelope. The inductor backend is added in a
          follow-up.

        ``tracer`` selects the capture front-end:

        - ``"make_fx"`` (default): a NON-STRICT make_fx trace -- it records the ATen ops
          that actually run when ``fn`` executes once on the example inputs and does not
          analyze your Python, so control flow and shapes are specialized to the example
          (the source of the programming-model contract). The only tracer implemented
          today.
        - ``"dynamo"``: planned (a Dynamo-based front-end that analyzes the Python);
          raises ``NotImplementedError`` for now.

        ``decompositions`` is an optional decomposition table (a dict mapping each
        ``OpOverload`` to a decomposition function) forwarded to ``make_fx`` as its
        ``decomposition_table`` during capture, so you can control how ATen ops are
        broken down in the captured graph. Defaults to ``None`` (make_fx's default).

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
        grads (invariant 5). Caller responsibilities NOT checked here, and violations
        that ARE checked (a baked constant, effectful ops), are in Note [precompile
        programming model].
        """
        torch._C._log_api_usage_once("torch.compiler.precompile")
        if backend not in ("eager", "inductor"):
            raise ValueError(
                f"precompile backend must be 'eager' or 'inductor', got {backend!r}."
            )
        if backend == "inductor":
            raise NotImplementedError(
                f"precompile backend={backend!r} is not implemented yet; use "
                "backend='eager' (the default)."
            )
        if tracer not in ("make_fx", "dynamo"):
            raise ValueError(
                f"precompile tracer must be 'make_fx' or 'dynamo', got {tracer!r}."
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

    def load(self, python_code: str, cache: bytes) -> Callable[..., object]:
        """Reconstruct a runnable from ``(python_code, cache)`` from precompile.

        The driver runs from ``python_code`` -- the single source of truth for the whole
        calling convention. The eager ``cache`` carries no compiled artifact, so ``load``
        only reads its ``BACKEND`` (to check the pairing) and runs the graph inlined in
        ``python_code``.

        Call the result with the SAME argument structure ``fn`` took -- the
        model(s) in their original positions plus the runtime inputs. Per invariant
        2 of Note [precompile programming model], the runtime model must match the
        example model's parameter/buffer structure; precompile re-derives the
        param/buffer list from it (same interning/order as capture).

        Raises ``PrecompileError`` if ``python_code`` is malformed / not a
        ``torch.compiler.precompile`` artifact, or if the cache's ``backend`` /
        ``code_hash`` does not match ``python_code`` (a cache and python_code from
        different ``precompile()`` calls). A ``format``/``version`` mismatch (a foreign
        or different-build envelope) is NOT fatal: the cache is acceleration only, so
        ``load`` degrades rather than crashing.
        """
        # The whole calling convention (MODULE_POSITIONS, NUM_POSITIONAL_ARGS, OUT_SPEC,
        # GRAD_PARAM_INDICES, USER_INPUT_*, PARAM_*, BUFFER_*, IN_SPEC) is consumed by the
        # driver INLINED in python_code (_EAGER_DRIVER_SOURCE), so the loaded object needs
        # none of it. _parse_artifact_metadata still runs to validate python_code is a
        # precompile artifact and to read BACKEND for the cache-pairing check below.
        meta = _parse_artifact_metadata(python_code)
        backend = cast(str, meta["BACKEND"])

        # weights_only=True is safe (plain str/int/bytes dict). The cache is acceleration
        # only, so an unreadable envelope or a FORMAT / VERSION mismatch degrades to
        # running python_code alone rather than crashing. A BACKEND or CODE_HASH mismatch
        # is different -- it signals a wrong (python_code, cache) pairing -- so it
        # hard-fails rather than running under foreign metadata.
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
        # The eager envelope carries no compiled artifact to prime. Run the driver inlined
        # in python_code: it carries the full calling convention and runtime safety checks
        # (subclass pass-through, param/buffer lifting, grad harvest, input/model
        # validation) and runs the inlined captured graph.
        forward = _make_inlined_forward(python_code)

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
