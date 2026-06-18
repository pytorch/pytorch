"""make_fx-based ahead-of-time precompilation.

    python_code, cache = torch.precompile(fn, model, *example_inputs)
    f_c = torch.precompile.load(python_code, cache)
    out = f_c(model, *example_inputs)   # pass the model again at runtime

precompile captures your computation with ``make_fx``, which is a NON-STRICT
trace: it records the ATen ops that actually run when ``fn`` executes once on the
example inputs. It does not analyze your Python. There is therefore a small,
explicit contract -- the programming model -- that the caller must follow. Stay
inside it and the artifact faithfully reproduces ``fn``; step outside it and the
trace silently bakes assumptions, and you get a fast artifact that computes the
wrong thing. This is by design: a precise contract instead of best-effort magic.

The captured graph is lowered through the AOT backend contract
(``torch._functorch.aot_autograd.compile_to_python``, which drives AOTAutograd +
Inductor and composes the runtime prelude/epilogue into the emitted source).
``precompile`` returns a self-contained, executable Python source string -- which
runs on its own, JIT-compiling kernels -- and a binary cache that is purely an
acceleration (the real compiled artifact, so ``load`` skips JIT; NO model weights --
you pass the model again at runtime). Reload with
``torch.precompile.load(python_code, cache)``.

The full contract is in Note [precompile programming model] below; every public
entry point and guard references it.

The cache carries ONLY the compiled/captured artifact -- no calling-convention
metadata and no weights. ``python_code`` is the single source of truth for the
calling convention, and ``load`` reads it back from there (parsing, not exec'ing),
so the cache cannot drift from the code it accelerates.

SECURITY: ``load`` reads the cache as a plain ``{"artifact": bytes}`` envelope with
``weights_only=True`` (it carries no Python objects), then reconstructs the compiled
artifact from those inner bytes via ``CompiledArtifact.load`` -- which unpickles the
inductor/AOTAutograd compiled artifact and therefore EXECUTES code embedded in it.
Loading a cache runs that embedded code, exactly like unpickling. Only load caches
you produced or otherwise trust (invariant 7).
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
# 1. Everything live is an input. Every tensor the computation reads must reach the
#    graph as a parameter/buffer of a module argument or as an explicit tensor
#    argument. For an nn.Module argument you do NOT enumerate its tensors yourself:
#    precompile lifts every registered parameter and buffer (recursively, including
#    submodules, tied weights collapsed by identity) to explicit graph inputs for you
#    via functional reparametrization, and re-derives the same list from the runtime
#    model you pass to load(). Passing the module is enough -- that is the whole point
#    of accepting modules as arguments. What is NOT lifted is anything not reachable
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
#    not change what the artifact computes. PARTIALLY enforced: the driver checks the
#    param/buffer count, but a same-count-different-structure model is not detected.
#
# 3. Control flow and shapes are specialized to the example. A non-strict trace
#    follows the single path taken for the example inputs: Python ``if``/``for``
#    over tensor values, ``.item()``, and shape-dependent branching are resolved at
#    trace time and baked. Shapes are static unless captured as dynamic. Inputs that
#    would take a different path, or different shapes that change control flow,
#    yield a wrong result. NOT enforced -- this is the defining property of a
#    non-strict trace. Capture also EXECUTES ``fn`` once on the example inputs, so any
#    in-place mutation of an input or other side effect ``fn`` performs (e.g.
#    ``x.add_(1)``, printing, RNG advancement) happens to the example inputs / external
#    state at capture time; pass throwaway example inputs if that matters.
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
# 5. Backward is part of the computation. If ``fn`` runs a backward, the parameter
#    gradients are harvested inside the (functional) graph as extra outputs, and the
#    driver scatters them back onto the runtime model's ``parameters()`` ``.grad``
#    fields -- accumulating (``p.grad += g``) exactly like eager ``.backward()``, so
#    a ``zero_grad()`` / ``optimizer.step()`` loop works unchanged. Only params that
#    actually received a gradient at trace time are harvested (recorded by index); a
#    frozen (``requires_grad=False``) or non-contributing param keeps ``.grad = None``,
#    exactly as eager leaves it -- precompile does NOT zero-fill such params. The
#    artifact therefore returns ``fn``'s own result (``None`` for a bare ``.backward()``
#    step), not the grads. The grad scatter is the ONLY mutation precompile performs,
#    and it happens in Python outside the graph, so the graph stays functional
#    (invariant 4 is about in-graph mutation and is unaffected). precompile does not
#    own optimizer state; bring your own optimizer and zero grads as usual.
#
# 6. Shapes are static. The graph is specialized to the example input shapes
#    (invariant 3); tensor-subclass outputs in particular are rebuilt with constant
#    outer sizes/strides, so a different runtime shape is undefined.
#
# 7. The cache is trusted, executable state. The outer envelope is a plain
#    {"artifact": bytes} dict (read with weights_only=True), but load()
#    reconstructs the inductor artifact from those bytes via CompiledArtifact.load,
#    which unpickles (and so runs) code embedded in the artifact; treat a cache like
#    code you are about to run.
#
# self-contained: ``python_code`` runs on its own -- it inlines the composed graph
# module (inductor: kernels JIT-compiled on first call, plus AOTAutograd's codegen'd
# prelude/epilogue) or the captured graph (eager), plus all calling-convention
# metadata. It NEVER reads the cache, and it is the SINGLE SOURCE OF TRUTH for the
# calling convention. The ``cache`` holds ONLY the compiled INDUCTOR artifact and is
# purely an ACCELERATION consumed only by load(): load reads the calling convention
# back out of python_code (an AST scrape, not an exec) and uses the cached artifact
# to reconstruct the real compiled artifact (FxGraphCache hit, no JIT). With the
# cache you skip JIT; with only python_code you JIT -- same results either way. The
# eager backend has no kernels to accelerate, so it caches nothing (an empty
# artifact) and load() always runs the graph inlined in python_code. The metadata
# lives in one place: no drift.
#
# backend: "inductor" (default) lowers the captured graph through
# torch._inductor.standalone_compile. "eager" skips lowering and runs the captured
# ATen graph as-is (analogous to torch.compile(backend="eager")), for inspecting or
# debugging exactly what was traced. The contract above is identical for both
# backends -- the same graph is captured; only its realization differs. Two
# mechanical consequences: the eager backend runs the graph directly on the
# (subclass-level) inputs, so it does not exercise the dense subclass
# flatten/unflatten path that the inductor backend's calling convention requires;
# and because there are no kernels, the eager cache is empty (python_code is the
# whole artifact).

from __future__ import annotations

import io
import logging
import os
import tempfile
from typing import Any, TYPE_CHECKING

import torch
import torch.utils._pytree as pytree
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless


log = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Callable


# ``precompile`` and ``PrecompileError`` are exposed at the top level as
# ``torch.precompile`` / ``torch.precompile.PrecompileError`` and registered in
# ``torch.__all__``; they are deliberately kept out of this private module's
# ``__all__`` so test_public_bindings sees a consistent single public location.
__all__: list[str] = []


class PrecompileError(RuntimeError):
    """Raised when precompile tracing would bake a tensor into the graph."""


def _scatter_grads(params: list[Any], grads: list[Any]) -> None:
    """Accumulate harvested grads onto the runtime params' ``.grad`` fields, exactly
    like eager ``.backward()`` (invariant 5 of Note [precompile programming model]).

    ``params`` is the subset of interned params that actually received a gradient
    (selected by the caller via ``grad_param_indices``), aligned with ``grads`` so
    ``grads[i]`` belongs to ``params[i]``. Params that received no gradient are not in
    this list and keep ``.grad = None``, exactly like eager. Accumulating rather than
    overwriting means a standard ``zero_grad()`` / ``optimizer.step()`` loop behaves as
    it does in eager.
    """
    for p, g in zip(params, grads):
        p.grad = g if p.grad is None else p.grad + g


def _check_no_constant_tensors(gm: torch.fx.GraphModule) -> None:
    """Enforce invariant 1 of Note [precompile programming model]: everything live
    is an input.

    Every legitimate tensor in a non-strict capture is a placeholder (a lifted
    parameter/buffer or user input) or the result of a ``call_function`` node.
    A ``get_attr`` pointing at a tensor therefore means some tensor was closed
    over (a global, captured local, or non-registered module attribute) and would
    be baked into the graph as a constant, which we forbid.
    """
    offending = []
    for node in gm.graph.nodes:
        if node.op != "get_attr":
            continue
        attr = gm
        for part in node.target.split("."):
            attr = getattr(attr, part, None)
        if isinstance(attr, torch.Tensor):
            offending.append((node.target, tuple(attr.shape), str(attr.dtype)))
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


def _intern_param_buffers(
    mods: list[torch.nn.Module],
) -> tuple[list[Any], list[str], list[str], list[tuple[int, str, int]], int]:
    """Lift each module's parameters then buffers to a flat list, interning by
    tensor identity so a tied weight becomes a single entry (one optimizer step,
    accumulated gradient -- not one per name).

    Returns ``(pb_flat, param_names, buffer_names, alias_entries, num_params)``,
    where ``alias_entries`` maps each ``(module_index, name)`` to its index in
    ``pb_flat`` (used to reparametrize during capture). This same params-then-
    buffers, intern-by-identity order is reproduced at runtime against the
    user-supplied modules, so the dense list lines up with the compiled graph.
    """
    multi = len(mods) > 1

    def _name(mi: int, n: str) -> str:
        return f"m{mi}.{n}" if multi else n

    unique: list[Any] = []
    id_to_uidx: dict[int, int] = {}
    alias_entries: list[tuple[int, str, int]] = []

    def _intern(mi: int, n: str, t: Any, names_out: list[str]) -> None:
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
            _intern(mi, n, p, param_names)
    num_params = len(unique)
    buffer_names: list[str] = []
    for mi, m in enumerate(mods):
        for n, b in m.named_buffers(remove_duplicate=False):
            _intern(mi, n, b, buffer_names)
    return unique, param_names, buffer_names, alias_entries, num_params


def _capture(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
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
    mods = [args[i] for i in module_positions]
    user_inputs = tuple(a for i, a in enumerate(args) if i not in module_pos_set)

    # Lift the example modules' params/buffers for tracing only. Their VALUES are
    # never stored in the cache -- the user passes the model(s) again at runtime
    # (mirroring fn's signature), and the same interning is reproduced there.
    pb_flat, param_names, buffer_names, alias_entries, num_params = (
        _intern_param_buffers(mods)
    )
    num_pb = len(pb_flat)

    user_flat, in_spec = pytree.tree_flatten(user_inputs)
    flat_args = [*pb_flat, *user_flat]

    out_spec_holder: dict[str, Any] = {}

    def flat_fn(flat: list[Any]) -> list[Any]:
        pb = flat[:num_pb]
        runtime_inputs = pytree.tree_unflatten(flat[num_pb:], in_spec)
        with contextlib.ExitStack() as stack:
            for mi, m in enumerate(mods):
                reparam = {n: pb[uidx] for emi, n, uidx in alias_entries if emi == mi}
                stack.enter_context(
                    stateless._reparametrize_module(m, reparam, tie_weights=True)
                )
            # Reconstruct fn's full positional args: reparametrized modules at
            # their original positions, runtime inputs at theirs.
            full: list[Any] = []
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
        out_spec_holder["spec"] = result_spec
        out_spec_holder["num_grad_outputs"] = len(grad_flat)
        out_spec_holder["grad_param_indices"] = grad_param_indices
        return [*result_flat, *grad_flat]

    # Tracing runs fn (and any backward in it) once on the real example tensors, so a
    # backward populates .grad on the lifted example params. We harvest those grads as
    # graph outputs, but must not leave them on the user's example model -- precompile
    # does not mutate the example model's params or .grad (in-place mutation that fn
    # itself performs on its inputs is a separate matter; see invariant 3), and the
    # runtime driver accumulates onto whatever .grad it finds. Snapshot and restore the
    # example params' grads around the trace.
    saved_grads = [p.grad if isinstance(p, torch.Tensor) else None for p in pb_flat]
    # Trace with grad enabled so any backward in ``fn`` is built as graph ops; the
    # forward graph is the same as under no_grad.
    with torch.enable_grad():
        gm = make_fx(flat_fn, decomposition_table=decompositions)(flat_args)
    for p, g in zip(pb_flat, saved_grads):
        if isinstance(p, torch.Tensor):
            p.grad = g
    _check_no_constant_tensors(gm)
    _assert_supported(gm, flat_args)

    return _Capture(
        gm=gm,
        flat_args=flat_args,
        module_positions=module_positions,
        param_names=param_names,
        buffer_names=buffer_names,
        param_buffer_flat=pb_flat,
        num_params_buffers=num_pb,
        in_spec=in_spec,
        out_spec=out_spec_holder["spec"],
        num_grad_outputs=out_spec_holder["num_grad_outputs"],
        grad_param_indices=out_spec_holder["grad_param_indices"],
    )


class _Capture:
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        flat_args: list[Any],
        module_positions: list[int],
        param_names: list[str],
        buffer_names: list[str],
        param_buffer_flat: list[Any],
        num_params_buffers: int,
        in_spec: pytree.TreeSpec,
        out_spec: pytree.TreeSpec,
        num_grad_outputs: int,
        grad_param_indices: list[int],
    ) -> None:
        self.gm = gm
        self.flat_args = flat_args
        self.module_positions = module_positions
        self.param_names = param_names
        self.buffer_names = buffer_names
        self.param_buffer_flat = param_buffer_flat
        self.num_params_buffers = num_params_buffers
        self.in_spec = in_spec
        self.out_spec = out_spec
        self.num_grad_outputs = num_grad_outputs
        self.grad_param_indices = grad_param_indices


_GENERATED_HEADER = """\
# Generated by torch.precompile -- do not edit.
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
# make_fx trace, so control flow and shapes are specialized to the example inputs.
# See Note [precompile programming model] in torch/_precompile.py.
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
# torch.precompile.load: it reconstructs the real compiled artifact (FxGraphCache
# hit, no JIT). This file does not read it; running this file alone just JITs.
"""


def _build_metadata_section(compiled: PrecompiledModule) -> list[str]:
    if compiled._out_spec is None:
        raise PrecompileError("internal: cannot build metadata before _compile()")
    out_spec_str = pytree.treespec_dumps(compiled._out_spec)
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
        f"PARAM_NAMES = {compiled._param_names!r}",
        f"BUFFER_NAMES = {compiled._buffer_names!r}",
        f"NUM_PARAMS_BUFFERS = {compiled._num_params_buffers}",
        f"NUM_GRAD_OUTPUTS = {compiled._num_grad_outputs}",
        # Which unique-param index each grad output belongs to; the driver scatters
        # grad k onto params/buffers[GRAD_PARAM_INDICES[k]] so frozen / non-
        # contributing params keep .grad = None, exactly like eager.
        f"GRAD_PARAM_INDICES = {compiled._grad_param_indices!r}",
        f"OUT_SPEC = {out_spec_str!r}",
        "",
    ]
    return parts


def _parse_artifact_metadata(python_code: str) -> dict[str, Any]:
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
        "NUM_PARAMS_BUFFERS",
        "NUM_GRAD_OUTPUTS",
        "GRAD_PARAM_INDICES",
        "OUT_SPEC",
    }
    found: dict[str, Any] = {}
    for node in ast.parse(python_code).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            found[target.id] = ast.literal_eval(node.value)
    missing = wanted - found.keys()
    if missing:
        raise PrecompileError(
            f"python_code is missing calling-convention metadata {sorted(missing)}; "
            "it does not look like a torch.precompile artifact."
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
    parts.append(_DRIVER_SOURCE)
    return "\n".join(parts)


_EAGER_GENERATED_HEADER = """\
# Generated by torch.precompile (backend="eager") -- do not edit.
#
# This is the EAGER backend: the captured ATen graph is NOT lowered through
# Inductor. It is a SELF-CONTAINED, EXECUTABLE artifact: the graph is inlined below
# and runs on its own, needing no companion cache. You provide the model(s) at
# runtime, exactly as the original fn took them, e.g.:
#
#     ns = {}
#     exec(open("this_file.py").read(), ns)
#     out = ns["forward"](model, my_input)      # same args as the traced fn
#
# The runtime model must be STRUCTURALLY IDENTICAL to the one precompile traced
# (same parameter/buffer names, order, and weight tying); only the weight VALUES
# may differ. This artifact was produced by a non-strict make_fx trace, so control
# flow and shapes are specialized to the example inputs. See Note [precompile
# programming model] in torch/_precompile.py.
#
# Section 1 below is the captured graph -- it is both the human-readable rendering
# and the executable code. The eager backend has no kernels to accelerate, so the
# companion ``cache`` returned by precompile is empty; torch.precompile.load runs
# this inlined graph, and this file is the whole artifact.
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
    same order as capture, so the list lines up with the captured graph."""
    seen = set()
    pb = []
    def intern(t):
        if id(t) not in seen:
            seen.add(id(t))
            pb.append(t)
    for m in mods:
        for _, p in m.named_parameters(remove_duplicate=False):
            intern(p)
    for m in mods:
        for _, b in m.named_buffers(remove_duplicate=False):
            intern(b)
    return pb


def forward(*args):
    """Run the captured ATen graph eagerly. Pass the same args the traced fn took --
    the module(s) in the same positions plus the runtime inputs. The module(s) must
    be structurally identical to the ones precompile traced (same param/buffer order
    and tying); only the weight values may differ.

    The eager backend runs the graph as captured: inputs (including tensor
    subclasses) are passed through unchanged (no dense flatten/unflatten), and the
    graph's flat outputs are reassembled into fn's output structure. If fn ran a
    backward, the trailing NUM_GRAD_OUTPUTS outputs are parameter grads, scattered
    (accumulated) onto the params that received one (GRAD_PARAM_INDICES) like eager
    .backward() -- frozen / non-contributing params keep .grad = None."""
    mods = [args[i] for i in MODULE_POSITIONS]
    user_inputs = [a for i, a in enumerate(args) if i not in set(MODULE_POSITIONS)]
    user_flat, _ = _pytree.tree_flatten(tuple(user_inputs))
    pb = _extract_param_buffers(mods)
    if len(pb) != NUM_PARAMS_BUFFERS:  # noqa: F821
        # Imported lazily (only on this error path) so a normal run does not couple
        # the standalone artifact to torch._precompile's import surface.
        from torch._precompile import PrecompileError as _PrecompileError

        raise _PrecompileError(
            "precompile: runtime model has %d params/buffers but this artifact was "
            "traced with %d; the runtime model must be structurally identical to "
            "the traced model (invariant 2)." % (len(pb), NUM_PARAMS_BUFFERS)  # noqa: F821
        )
    with _torch.no_grad():
        out = list(call([*pb, *user_flat]))  # noqa: F821
    if NUM_GRAD_OUTPUTS:  # noqa: F821
        grads = out[len(out) - NUM_GRAD_OUTPUTS:]  # noqa: F821
        out = out[:len(out) - NUM_GRAD_OUTPUTS]  # noqa: F821
        for idx, g in zip(GRAD_PARAM_INDICES, grads):  # noqa: F821
            p = pb[idx]
            p.grad = g if p.grad is None else p.grad + g
    return _pytree.tree_unflatten(out, _pytree.treespec_loads(OUT_SPEC))  # noqa: F821


if __name__ == "__main__":
    print("forward() is ready; call it with the model(s) and inputs the traced")
    print("fn took, e.g. forward(model, x).")
'''


_DRIVER_SOURCE = '''
def _extract_param_buffers(mods):
    """Lift the runtime modules' params then buffers, interning by identity, in the
    same order as capture, so the dense list lines up with the compiled graph."""
    seen = set()
    pb = []
    def intern(t):
        if id(t) not in seen:
            seen.add(id(t))
            pb.append(t)
    for m in mods:
        for _, p in m.named_parameters(remove_duplicate=False):
            intern(p)
    for m in mods:
        for _, b in m.named_buffers(remove_duplicate=False):
            intern(b)
    return pb


def forward(*args):
    """Run the compiled computation. Pass the same args the traced fn took -- the
    module(s) in the same positions plus the runtime inputs. The module(s) must be
    structurally identical to the ones precompile traced (same param/buffer order
    and tying); only the weight values may differ.

    Module params/buffers are extracted (no weights are baked into the artifact) and,
    together with the runtime inputs, passed to the composed ``call`` -- which is the
    AOTAutograd+Inductor graph with its own prelude/epilogue, so it handles tensor-
    subclass wrap/unwrap and input mutation (e.g. BatchNorm running stats) internally
    and disables grad itself. If fn ran a backward, the trailing NUM_GRAD_OUTPUTS
    outputs are parameter grads: they are scattered (accumulated) onto the params that
    received one (GRAD_PARAM_INDICES), mirroring eager .backward() (frozen /
    non-contributing params keep .grad = None), and the artifact returns fn's own
    result. Nothing here reads an external cache: the kernels JIT-compile from the
    inlined source on first call."""
    mods = [args[i] for i in MODULE_POSITIONS]
    user_inputs = [a for i, a in enumerate(args) if i not in set(MODULE_POSITIONS)]
    user_flat, _ = _pytree.tree_flatten(tuple(user_inputs))
    pb = _extract_param_buffers(mods)
    if len(pb) != NUM_PARAMS_BUFFERS:  # noqa: F821
        # Imported lazily (only on this error path) so a normal run does not couple
        # the standalone artifact to torch._precompile's import surface.
        from torch._precompile import PrecompileError as _PrecompileError

        raise _PrecompileError(
            "precompile: runtime model has %d params/buffers but this artifact was "
            "traced with %d; the runtime model must be structurally identical to "
            "the traced model (invariant 2)." % (len(pb), NUM_PARAMS_BUFFERS)  # noqa: F821
        )
    out = list(call([*pb, *user_flat]))  # noqa: F821 (inlined composed entry point)
    if NUM_GRAD_OUTPUTS:  # noqa: F821
        grads = out[len(out) - NUM_GRAD_OUTPUTS:]  # noqa: F821
        out = out[:len(out) - NUM_GRAD_OUTPUTS]  # noqa: F821
        for idx, g in zip(GRAD_PARAM_INDICES, grads):  # noqa: F821
            p = pb[idx]
            p.grad = g if p.grad is None else p.grad + g
    return _pytree.tree_unflatten(out, _pytree.treespec_loads(OUT_SPEC))  # noqa: F821


if __name__ == "__main__":
    print("forward() is ready; call it with the model(s) and inputs the traced")
    print("fn took, e.g. forward(model, x).")
'''


def _assert_supported(gm: torch.fx.GraphModule, flat_args: list[Any]) -> None:
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
        fn: Callable[..., Any],
        *,
        backend: str = "inductor",
        decompositions: dict | None = None,
    ) -> None:
        # ``fn`` is the whole computation: an nn.Module, or a callable that closes
        # over the module(s) it uses (e.g. ``lambda x: model(x)``, or a training
        # step that computes a loss and torch.autograd.grad).
        self._fn = fn
        # "inductor" (default) lowers the captured graph through
        # standalone_compile; "eager" keeps the captured ATen graph and runs it
        # as-is (see Note [precompile programming model], "backend").
        self._backend = backend
        self._decompositions = decompositions
        self._artifact: Any = None
        self._module_positions: list[int] = []
        self._param_names: list[str] = []
        self._buffer_names: list[str] = []
        self._num_params_buffers: int = 0
        self._in_spec: pytree.TreeSpec | None = None
        self._out_spec: pytree.TreeSpec | None = None
        self._gm: torch.fx.GraphModule | None = None
        # Inductor backend: the composed self-contained graph module (from
        # aot_autograd.compile_to_python, exposing ``call(flat_inputs)``) and the
        # opaque artifact-cache bytes (None if uncacheable), populated by _compile().
        self._graph_python: str = ""
        self._artifact_bytes: bytes | None = None
        self._num_grad_outputs: int = 0
        # Which unique-param index each emitted grad output belongs to (length ==
        # _num_grad_outputs). Lets the driver scatter grads onto exactly the params
        # that received one, leaving frozen / non-contributing params' .grad as None.
        self._grad_param_indices: list[int] = []
        # Set only on the load() path, where we wrap a reconstructed callable.
        self._loaded_forward: Callable[..., Any] | None = None

    @classmethod
    def _from_loaded(
        cls,
        forward: Callable[..., Any],
        *,
        backend: str,
        module_positions: list[int],
        out_spec: pytree.TreeSpec,
        num_grad_outputs: int,
        grad_param_indices: list[int],
        num_params_buffers: int,
    ) -> PrecompiledModule:
        """Build a runnable from load()'s reconstructed forward.

        load() does not re-run capture/_compile, so reuse ``__init__`` for all the
        defaults (the single definition of this object's state) and override only the
        calling-convention fields recovered from python_code plus the reconstructed
        forward. The capture-only fields (``_fn``, ``_gm``, ``_param_names``, ...)
        stay at their ``__init__`` defaults; inspect the artifact via python_code.
        """
        obj = cls(None, backend=backend)  # type: ignore[arg-type]
        obj._module_positions = module_positions
        obj._out_spec = out_spec
        obj._num_grad_outputs = num_grad_outputs
        obj._grad_param_indices = grad_param_indices
        obj._num_params_buffers = num_params_buffers
        obj._loaded_forward = forward
        return obj

    def _compile(self, args: tuple[Any, ...]) -> None:
        capture = _capture(self._fn, args, self._decompositions)
        self._module_positions = capture.module_positions
        self._param_names = capture.param_names
        self._buffer_names = capture.buffer_names
        self._num_params_buffers = capture.num_params_buffers
        self._in_spec = capture.in_spec
        self._out_spec = capture.out_spec
        self._num_grad_outputs = capture.num_grad_outputs
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
        # serialized artifact, or None for uncacheable graphs).
        from torch._functorch import aot_autograd

        self._graph_python, self._artifact_bytes = aot_autograd.compile_to_python(
            capture.gm, capture.flat_args, dynamic_shapes="from_example_inputs"
        )

    def __call__(self, *args: Any) -> Any:
        # A PrecompiledModule is runnable only after load(); precompile() itself
        # returns (python_code, cache) rather than a runnable.
        if self._loaded_forward is None:
            raise PrecompileError(
                "this object is not runnable; build one with "
                "torch.precompile.load(python_code, cache)."
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
        if self._backend == "eager":
            if self._gm is None:
                raise PrecompileError("internal: not compiled; call _compile() first")
            return _build_eager_python_source(self)
        if not self._graph_python:
            raise PrecompileError("internal: not compiled; call _compile() first")
        return _build_python_source(self, self._graph_python)

    def to_cache_bytes(self) -> bytes:
        """Return the binary cache as bytes -- an ACCELERATION, not required to run.

        ``python_code`` already runs standalone AND is the single source of truth for
        the calling convention, so the cache holds ONLY the compiled inductor artifact
        -- no calling-convention metadata, no model weights. load() recovers the
        calling convention by parsing python_code (``_parse_artifact_metadata``) and
        uses this artifact solely to skip JIT/recompile. For the inductor backend the
        artifact is the real Inductor/AOTAutograd compiled-artifact bytes (load primes
        the inductor cache via a FxGraphCache hit, so reload does not re-trace/re-lower
        and, on GPU, restores bundled Triton kernels); ``None`` if the graph is
        uncacheable, in which case load() falls back to the standalone python. The
        eager backend has no kernels to accelerate, so its artifact is always ``None``
        and load() runs the graph inlined in python_code.
        """
        # The opaque artifact-cache bytes from aot_autograd.compile_to_python (None
        # for an uncacheable inductor graph, and always None for the eager backend,
        # which has no kernels to cache); load() then falls back to executing the
        # self-contained python_code.
        buf = io.BytesIO()
        torch.save({"artifact": self._artifact_bytes}, buf)
        return buf.getvalue()


def _make_cached_forward(
    artifact_bytes: bytes,
    module_positions: list[int],
    out_spec: pytree.TreeSpec,
    num_grad_outputs: int,
    grad_param_indices: list[int],
    num_params_buffers: int,
) -> Callable[..., Any]:
    """Reconstruct the compiled artifact from the cache and drive it.

    ``CompiledArtifact.load`` primes the inductor cache from ``artifact_bytes``
    (so the graph is not re-traced/re-lowered and bundled kernels are restored)
    and rebuilds the full AOTAutograd runtime (subclass aware), so it takes the
    subclass-level params/inputs directly. The runtime model(s) are supplied by the
    caller (mirroring fn's signature); the graph is functional, so it runs under
    no_grad.
    """
    from torch._inductor import CompiledArtifact

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
        tmp = tf.name
        tf.write(artifact_bytes)
    try:
        artifact = CompiledArtifact.load(path=tmp, format="binary")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    pos = set(module_positions)

    def forward(*args: Any) -> Any:
        mods = [args[i] for i in module_positions]
        user_inputs = tuple(a for i, a in enumerate(args) if i not in pos)
        pb_flat = _intern_param_buffers(mods)[0]
        if len(pb_flat) != num_params_buffers:
            raise PrecompileError(
                f"runtime model has {len(pb_flat)} params/buffers but this artifact "
                f"was traced with {num_params_buffers}; the runtime model must be "
                "structurally identical to the traced model (Note [precompile "
                "programming model], invariant 2)."
            )
        user_flat, _ = pytree.tree_flatten(user_inputs)
        with torch.no_grad():
            out_flat = list(artifact(*[*pb_flat, *user_flat]))
        if num_grad_outputs:
            grads = out_flat[len(out_flat) - num_grad_outputs :]
            out_flat = out_flat[: len(out_flat) - num_grad_outputs]
            _scatter_grads([pb_flat[i] for i in grad_param_indices], grads)
        return pytree.tree_unflatten(out_flat, out_spec)

    return forward


def _make_inlined_forward(python_code: str) -> Callable[..., Any]:
    """Fallback: execute the self-contained python string (JITs kernels).

    ``python_code`` needs no cache -- the kernels (inductor) or graph (eager) are
    inlined, so we just exec it and hand back its ``forward``. The returned
    ``forward`` takes the same args the traced fn took (model(s) plus runtime
    inputs)."""
    module_ns: dict[str, Any] = {"__name__": "_precompiled_artifact"}
    exec(compile(python_code, "<precompile>", "exec"), module_ns)
    return module_ns["forward"]


class _PrecompileApi:
    """Callable namespace implementing ``torch.precompile`` and ``.load``.

    A single instance is exposed as ``torch.precompile``; calling it precompiles a
    computation and ``torch.precompile.load`` reloads the resulting artifacts. It
    is a class (rather than a function with attached attributes) so the call, the
    loader, and the error type are explicit members.

    The contract for both ``__call__`` and ``load`` is Note [precompile programming
    model] in this module.
    """

    # Reported so test_public_bindings / introspection see this as ``torch``.
    __module__ = "torch"

    # The error type raised by precompile, reachable as
    # ``torch.precompile.PrecompileError``.
    PrecompileError = PrecompileError

    def __call__(
        self,
        fn: Callable[..., Any],
        *args: Any,
        backend: str = "inductor",
        decompositions: dict | None = None,
    ) -> tuple[str, bytes]:
        """Ahead-of-time precompile ``fn`` against example ``args`` via make_fx.

        This is a NON-STRICT trace with an explicit contract; read Note [precompile
        programming model] before using it. The artifact faithfully reproduces
        ``fn`` only for callers that uphold that contract.

        THREADING: precompilation drives process-global compiler state and is
        serialized by an internal lock, so concurrent ``torch.precompile`` calls run
        one at a time rather than in parallel.

        ``backend`` selects how the captured graph is realized:

        - ``"inductor"`` (default): lower the graph through
          ``torch._inductor.standalone_compile`` (the full AOTAutograd + Inductor
          pipeline). ``python_code`` is the inlined Inductor output; the cache holds
          the real compiled artifact.
        - ``"eager"``: do NOT lower -- keep the captured ATen graph and run it as-is
          (analogous to ``torch.compile(backend="eager")``). ``python_code`` inlines
          the readable captured graph (both the inspectable rendering and the
          executable artifact); the cache is empty -- with no kernels there is
          nothing to accelerate, so ``load`` runs the inlined graph. Useful for
          inspecting/debugging exactly what was traced without an Inductor dependency.

        Returns ``(python_code, cache)`` -- a self-contained, executable Python
        source string (the single source of truth for the calling convention) and a
        binary cache holding ONLY the backend artifact (NO metadata, NO weights).
        Reload a runnable with ``torch.precompile.load(python_code, cache)``.

        ``fn`` is the whole computation, e.g.::

            python_code, cache = torch.precompile(lambda model, x: model(x), model, x)


            def train_step(model, x, t):
                loss_fn(model(x), t).backward()  # or return autograd.grad(...)


            python_code, cache = torch.precompile(train_step, model, x, t)

        Among ``args``, the ``nn.Module`` arguments have their params/buffers lifted
        to graph inputs (no weights are baked into the artifact -- invariant 1); the
        rest are the runtime inputs. The reloaded callable is invoked with the SAME
        argument structure -- pass the model(s) again at runtime, e.g.
        ``f_c(model, x)``, and that runtime model must match the example model's
        parameter/buffer structure (invariant 2). If ``fn`` ran a backward, the
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
        example, and control flow / shapes are specialized to ``args`` (invariants 2
        and 3). Violations that ARE checked raise ``PrecompileError``: a tensor baked
        as a constant (invariant 1) and effectful ops (invariant 4).
        """
        torch._C._log_api_usage_once("torch.precompile")
        if backend not in ("inductor", "eager"):
            raise ValueError(
                f"precompile backend must be 'inductor' or 'eager', got {backend!r}."
            )
        compiled = PrecompiledModule(fn, backend=backend, decompositions=decompositions)
        compiled._compile(args)
        return compiled.to_python_code(), compiled.to_cache_bytes()

    def load(self, python_code: str, cache: bytes) -> PrecompiledModule:
        """Reconstruct a runnable from ``(python_code, cache)`` from precompile.

        The calling convention is read from ``python_code`` -- the single source of
        truth -- by ``_parse_artifact_metadata`` (an AST scrape, NOT an exec, so the
        inlined kernels are not JIT'd). The ``cache`` carries ONLY the compiled
        inductor artifact and is a pure acceleration (Note [precompile programming
        model], "self-contained"): for the inductor backend it is the serialized
        artifact, rebuilt via ``CompiledArtifact.load`` -- priming the inductor cache
        (FxGraphCache hit, no re-lowering; restores bundled kernels) and the full
        AOTAutograd runtime, so tensor subclasses (DTensor) work. The eager backend
        caches nothing (no kernels to accelerate). With no artifact in the cache,
        exec'ing ``python_code`` JITs the kernels (inductor) or runs the inlined graph
        (eager) -- same result, just without the cache's speedup.

        Call the result with the SAME argument structure ``fn`` took -- the
        model(s) in their original positions plus the runtime inputs. Per invariant
        2 of Note [precompile programming model], the runtime model must match the
        example model's parameter/buffer structure; precompile re-derives the
        param/buffer list from it (same interning/order as capture).

        SECURITY: the cache envelope is read with ``weights_only=True``, but
        reconstructing the artifact (``CompiledArtifact.load``) unpickles the compiled
        inductor artifact and so executes code embedded in it. Only load trusted
        caches (invariant 7).
        """
        # Unpickling the cache references classes in AOTAutograd's runtime; import
        # dynamo first so that import completes in a non-circular order (otherwise
        # a cold load can hit a runtime_wrappers <-> _dynamo circular import).
        import torch._dynamo

        meta = _parse_artifact_metadata(python_code)
        backend = meta["BACKEND"]
        module_positions = meta["MODULE_POSITIONS"]
        num_params_buffers = meta["NUM_PARAMS_BUFFERS"]
        num_grad_outputs = meta["NUM_GRAD_OUTPUTS"]
        grad_param_indices = meta["GRAD_PARAM_INDICES"]
        out_spec = pytree.treespec_loads(meta["OUT_SPEC"])

        # The envelope is a plain {"artifact": bytes} dict, so weights_only=True is
        # safe here; the executable artifact is the inner bytes, reconstructed by
        # CompiledArtifact.load below. Use it for the fast reconstruction, else fall
        # back to exec'ing the self-contained python_code.
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        artifact = blob.get("artifact")
        if artifact is not None:
            # Reconstructing the artifact unpickles (and runs) code embedded in it --
            # the step weights_only/add_safe_globals cannot make safe -- so warn that
            # loading a cache runs embedded code; only load trusted caches.
            from torch._logging import warning_once

            warning_once(
                log,
                "torch.precompile.load reconstructs the compiled artifact from the "
                "cache (CompiledArtifact.load), which unpickles and runs code embedded "
                "in it. Only load caches you produced or otherwise trust (Note "
                "[precompile programming model], invariant 7).",
            )
            forward = _make_cached_forward(
                artifact,
                module_positions,
                out_spec,
                num_grad_outputs,
                grad_param_indices,
                num_params_buffers,
            )
        else:
            # No serialized artifact: an uncacheable inductor graph, or the eager
            # backend, which caches nothing. Run the graph inlined in python_code.
            forward = _make_inlined_forward(python_code)

        return PrecompiledModule._from_loaded(
            forward,
            backend=backend,
            module_positions=module_positions,
            out_spec=out_spec,
            num_grad_outputs=num_grad_outputs,
            grad_param_indices=grad_param_indices,
            num_params_buffers=num_params_buffers,
        )


precompile = _PrecompileApi()
# ``torch.precompile`` is a callable instance, not a function, so give it the
# name/doc introspection (Sphinx autosummary, help(), IDEs) expects to find on a
# top-level callable; the rich usage docs live on ``__call__``.
precompile.__name__ = "precompile"  # type: ignore[attr-defined]
precompile.__qualname__ = "precompile"  # type: ignore[attr-defined]
precompile.__doc__ = _PrecompileApi.__call__.__doc__
