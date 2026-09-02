"""Lower a GraphModule to a self-contained Python module via AOTAutograd + Inductor.

This is the AOT half of the backend contract behind ``torch.compiler.precompile``:

    python_code, cache = compile_to_python(gm, example_inputs)

``inductor.compile_to_python`` produces only the inner Inductor ``call`` (kernels
for the post-AOTAutograd dense graph). This module wraps that with the prelude /
epilogue (input-mutation reflection, output-alias regen, subclass wrap/unwrap,
...) by COMPOSING AOTAutograd's own codegen'd runtime-wrapper source -- captured
during compile -- rather than reimplementing it. Every wrapper is spliced verbatim as a
real top-level ``def``, with its closed-over globals hoisted to module-scope assignments,
so the module reads as ordinary code. These nest in the SAME order the runtime builds:
the INNER chain wrappers (subclass / functionalized-RNG) wrap the inner ``call`` and are
composed inside the orchestration, while the OUTER ``CompilerWrapper``s (dedup / synthetic
base) wrap the orchestration itself (via a single-arg adapter over it). Cross-wrapper references, the inner ``call`` chain,
public helpers, and baked metadata objects (reconstructed as source -- see
``_emit_value``) are wired by name; a guard rejects the rare case where a wrapper def
name or hoisted global would collide with another top-level name (a sibling wrapper or
an inner-module binding) rather than silently rebinding one.
The result is a standalone module exposing ``call(flat_inputs) -> outputs`` that
runs on its own (JIT-compiling kernels); ``cache`` is an opaque acceleration (or
None).

TRAINING ARTIFACTS (``grad_enabled=True`` with inputs that require grad). AOTAutograd
emits a joint forward+backward, and the module carries BOTH Inductor lowerings (their
top-level names suffixed ``_s0`` / ``_s1``... so they share one namespace, see
``namespace_module_names``) bridged by an emitted ``torch.autograd.Function``,
``_CompiledFunction``, whose forward/backward bodies are AOTAutograd's own codegen'd
``_compiled_forward`` / ``_compiled_backward`` (spliced like every other wrapper) and
whose remaining glue is spliced from ``standalone_training_glue.py``. Calling
convention, also written into the artifact header (``_TRAINING_MODULE_HEADER``):

- ``call`` runs the forward under ``torch.enable_grad`` like AOTAutograd's runtime
  wrapper, so freshly created outputs carry a ``_CompiledFunction`` ``grad_fn`` and
  ``.backward()`` runs the precompiled backward. Detaching under an ambient
  ``torch.no_grad()`` is the caller's job (the ``torch.compiler.precompile`` driver
  does it); this module never inspects the ambient grad mode.
- Every input's ``requires_grad`` is baked from capture (AOTAutograd's joint is
  specialized on it); the module does not guard it, ``precompile`` does upstream.
- Grads reach ``_CompiledFunction.backward`` as ONE boxed list (``boxed_grads_call``),
  ``None`` marking an undefined output tangent.
- ``_AOT_BACKWARD_VARIANTS`` maps a canonical undefined-tangent bitmask (bit ``i`` set
  when specializable user output ``i`` received no grad) to the ``_BackwardVariant``
  serving it: the Inductor backward to run, the saved-arg indices it keeps, the outputs
  it prunes to None (``None`` = decide at runtime from the baked dependency table) and
  the tangents it leaves unmaterialized. Mask 0 (all tangents defined) is ALWAYS
  present; an unseen pattern is served by it, materializing the missing tangents from
  the prototypes saved at forward time -- mainline AOTAutograd's own fallback -- so the
  artifact never compiles at serve time.
- The undefined-tangent handling (mainline's ``aot_autograd_prune_unused_outputs``) is
  baked ON at emission; the artifact reads no functorch / dynamo / inductor config at
  load or call time.

Capturing further tangent patterns is the job of ``_CompileToPythonState`` (returned by
``_compile_to_python_with_state``): it owns every piece of the capture protocol the
artifact exposes (``install_capture`` / ``uninstall_capture`` / ``observed_masks`` /
``compiled_masks`` / ``finalize``) so callers never touch the module's globals.

Baked metadata is emitted as plain Python source (no pickle / base64 blobs), so the
generated module is fully auditable and exec'ing it never invokes ``pickle.loads``.
A leaf that cannot be expressed as source raises NotImplementedError rather than
falling back to an opaque embedding.

Contract note: the standalone ``call`` deliberately substitutes ``nullcontext`` / a
no-op for the runtime's first-invocation context and profiler prologue, dropping the
cold-start custom-op aliasing analysis and the profiler prologue -- both diagnostics
with no effect on numerics (see the generated-call emission site). One caveat: that
dropped first-invocation custom-op aliasing analysis can itself RAISE under
``config.error_on_custom_op_aliasing`` (default on in CI), so a graph whose custom op
violates the aliasing contract runs SILENTLY in the standalone artifact where the
eager / compiled path would error -- an intentional trade-off, not a numerics bug.
"""

from __future__ import annotations

import ast
import itertools
import re
import threading
from dataclasses import dataclass, field
from typing import Any, cast, TYPE_CHECKING

from .codegen import capture_generated_sources, GeneratedSource
from .source_emit import _REBUILD_HELPER, emit_value


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from torch.fx import GraphModule


# Serializes compile_to_python: the wrapper-source capture is thread-local, but the
# AOTAutograd capture pass and the inner inductor compile swap process-global cache state
# (see the THREADING note on compile_to_python), so concurrent compiles must not overlap.
# An RLock (not a
# plain Lock) because this entry point is re-entrant on a single thread: a custom
# backend or inductor pass invoked while the lock is held may call back into this
# lowering to compile a subgraph on the SAME thread, and a plain Lock would
# self-deadlock on that re-entry. On-thread re-entry is safe here: the capture sink and
# the cache-state swap (CacheArtifactManager.with_fresh_cache) are each self-contained
# per call, so a nested compile neither corrupts the outer capture nor the outer cache
# scope.
_COMPILE_LOCK = threading.RLock()


# ======================================================================
# WHAT IS GOING ON HERE: composing AOTAutograd's runtime wrappers as source
# ======================================================================
#
# AOTAutograd does not hand back a single flat function. The dense graph Inductor
# compiles (the inner ``call``) is only the arithmetic core; around it AOTAutograd
# wraps a prelude/epilogue that does what the core cannot express: reflecting input
# mutations back onto the caller's tensors, regenerating outputs that alias an input
# or each other, wrapping/unwrapping tensor subclasses, de-duplicating aliased
# inputs, and threading functionalized RNG state. At runtime AOTAutograd emits each
# wrapper as Python *source*, exec's it (the chokepoint is _compile_and_exec_source
# in codegen.py), and the resulting function runs while closing over a
# globals dict supplied in-process -- i.e. a wrapper is (source text) +
# ({local_name: live_object}).
#
# The objects a wrapper closes over come in a few kinds:
#   - public runtime helpers the codegen'd source references (e.g. increment_version,
#     gen_alias_from_base, _unwrap_tensoralias, mark_dynamo_propagated_dynamic_indices,
#     the CUDARngStateHelper staticmethods) -- ordinary importable objects;
#   - the inner Inductor ``call`` that the chain ultimately invokes;
#   - sibling captured wrappers -- the next link of the runtime chain (an inner subclass /
#     functionalized-RNG wrapper, or an outer dedup / synthetic-base wrapper, whose body
#     calls the link it wraps -- the orchestration's outer closure for the innermost outer
#     wrapper), plus the orchestration's output-alias and mutation epilogue helpers, which
#     it closes over directly by reference;
#   - per-graph metadata baked at compile time (e.g. a ViewMetaSequence for alias
#     regen, tensor-subclass metadata) -- live objects with no import path.
#
# Concretely, the module emitted for ``x.view(-1)`` (an output aliasing its input) has
# one of each (heavily trimmed; the (1)-(4) tags match the kinds listed above):
#
#     from ...standalone_runtime import gen_alias_from_base    # (1) public helper
#     <inner Inductor kernels + ``def call(args): ...``>
#     _inner_call = call                                       # (2) the inner call
#     _vms_0 = ViewMetaSequence._from_parts(...)               # (4) metadata as source
#     def _alias_fn(orig_inputs, fw_outs):                     # (3) sibling wrapper
#         return [gen_alias_from_base(orig_inputs[0], fw_outs[0], False, _vms_0, ...)]
#     _replay_aliases_ = _alias_fn                             # (3) orchestration's ref
#     def _runtime_wrapper(_compiled_fn_, _first_ctx_, _on_before_call_, args):
#         all_outs = _compiled_fn_(args)                       # (2) inner call invoked
#         return _replay_aliases_(orig_inputs, all_outs)       # (3) sibling invoked
#     def call(flat_inputs):
#         return _runtime_wrapper(
#             _inner_call, contextlib.nullcontext, lambda: None, list(flat_inputs))
#
# We do NOT reimplement any of this. We CAPTURE AOTAutograd's exact codegen'd wrapper
# source together with the (pre-exec) globals dict each wrapper closed over: a
# thread-local sink in codegen.py records one GeneratedSource per wrapper.
# To trigger the capture we run AOTAutograd ourselves (grad mode selects the inference
# path or a joint forward/backward; see ``compile_to_python.grad_enabled``) with a
# capture-only inner compiler: it grabs the dense inner graph and returns a placeholder
# callable, so AOTAutograd still codegen's the runtime-wrapper chain AROUND that
# placeholder -- which is what the sink records. Inductor does not run in that pass;
# it runs once afterward on the captured dense graph (via inductor.compile_to_python), and
# the composer swaps the placeholder for the inner inductor ``call`` by object identity.
#
# THE COMPOSITION PROBLEM. To turn a captured wrapper into a real top-level ``def`` in
# the standalone module we splice its source verbatim. But that source refers to each
# global by the LOCAL ALIAS AOTAutograd happened to choose (e.g. ``compiled_fn`` for
# the inner call), not by any importable name. So for each ``{name: obj}`` in the
# captured globals dict we emit a top-level binding ``name = <source for obj>`` (see
# _emit_inline) -- except when the resolved expression already IS that module-scope
# name (an import, or ``torch``), which needs no binding. The hard part is the right-
# hand side: given only a live object, produce source that reproduces it. That requires
# RECOGNIZING what the object is, the job of _resolve_global and the id-keyed
# structures it consults, in order:
#   - inner_call_id         -> the inner ``call`` becomes ``_inner_call``
#   - fn_id_to_name         -> a sibling wrapper's fn becomes that wrapper's own name
#   - _known_helper_table() -> an importable helper becomes (import, expr)
#   - anything else: reconstruct field-by-field as source (_emit_value), or raise.
# Recognition is by id() (object identity), not value-equality: for "is this the EXACT
# object the wrapper closed over," == is the wrong tool (functions don't compare by
# value, and an equal-but-different object would mis-resolve). Value-equality IS used,
# but later and for a different job -- _emit_value round-trip-checks reconstructed
# metadata (rebuilt == obj) before trusting it. Every hoisted name is _reserve'd: a
# collision with a sibling wrapper's name or an inner-module binding fails loudly
# rather than silently rebinding.
#
# WHY THIS IS SAFE ACROSS PROCESSES AND MACHINES. Every id() above is consulted ONLY
# here, during composition, in the process that just ran the compile -- where all the
# candidate objects (helpers, the inner call, sibling wrappers) are simultaneously
# alive, held by the GeneratedSource records and the captured globals dicts, so no
# address can be freed and reused mid-pass. Nothing the composer emits carries an id()
# value, a live object, or any thread-local capture state: it emits only import lines,
# ``name = expr`` bindings, and verbatim def source (the inner Inductor source is
# likewise spliced as text). Grep the GENERATED module and there is no ``id(`` to
# find. By the time the user holds the Python, all the process/thread-local state the
# composer leaned on is gone; what ships is imports + name bindings + verbatim code.
# Load it on another machine with the same torch version and the imports resolve by
# name and the bindings reconstruct -- identity was a compile-time recognition device,
# never a serialized artifact (live tensors / pickle / base64 blobs are rejected
# outright rather than embedded).
#
# So the one genuine cross-machine contract is not id() but that the *names* the
# artifact imports still resolve on load. Helpers are emitted as either ``import
# torch`` (the torch module and stable public paths, e.g.
# torch.autograd.graph.increment_version) or an import from the single small surface
# standalone_runtime.py (for the AOTAutograd-area internals -- plus CUDARngStateHelper,
# re-exported there for import-ordering -- whose locations are not themselves a stable
# contract). That file's IDENTITY CONTRACT -- re-exports must preserve object id --
# exists purely so the COMPOSER's id-lookup keeps matching; it is a compile-time
# requirement, and the runtime artifact has no id dependency of its own.
# ======================================================================


# Generated artifacts import runtime helpers from the single stable surface
# ``standalone_runtime`` (not scattered AOTAutograd internals).
_SURFACE_MODULE = "torch._functorch._aot_autograd.standalone_runtime"
_SURFACE_IMPORT = f"from {_SURFACE_MODULE} import"


# Global objects the codegen'd wrappers close over that are reproducible as an
# import in the standalone module (rather than reconstructed field-by-field). Maps
# object id -> (import_statement, expression). Built lazily to avoid import cycles.
def _known_helper_table() -> dict[int, tuple[str, str]]:
    import torch

    from . import standalone_runtime as rt

    def surface(name: str) -> tuple[str, str]:
        return (f"{_SURFACE_IMPORT} {name}", name)

    table: dict[int, tuple[str, str]] = {
        id(torch): ("import torch", "torch"),
        id(torch.autograd.graph.increment_version): (
            "import torch",
            "torch.autograd.graph.increment_version",
        ),
        id(rt.CUDARngStateHelper.get_torch_state_as_tuple): (
            f"{_SURFACE_IMPORT} CUDARngStateHelper",
            "CUDARngStateHelper.get_torch_state_as_tuple",
        ),
        id(rt.CUDARngStateHelper.set_new_offset): (
            f"{_SURFACE_IMPORT} CUDARngStateHelper",
            "CUDARngStateHelper.set_new_offset",
        ),
        id(rt.AOTDispatchAutograd.process_runtime_tangent): (
            f"{_SURFACE_IMPORT} AOTDispatchAutograd",
            "AOTDispatchAutograd.process_runtime_tangent",
        ),
    }
    for name in (
        "normalize_as_list",
        "mark_dynamo_propagated_dynamic_indices",
        "gen_alias_from_base",
        "_unwrap_tensoralias",
        # Closed over by the codegen'd training wrappers (backward prologue,
        # compiled_fn_wrapper, compiled_function_forward).
        "_materialize_missing_grad_outputs",
        "_process_runtime_or_materialized_tangent",
        "_unwrap_no_symints",
        "_dealias_marked_returns",
        "wrap_tensor_subclasses",
        "_wrap_pruned_subclass_grad",
        "TensorAlias",
        "BackwardState",
    ):
        table[id(getattr(rt, name))] = surface(name)
    return table


# Bound builtin methods have no stable identity (``itertools.chain.from_iterable is
# itertools.chain.from_iterable`` is False) and no import path emit_value can derive,
# so they are recognized by (owner id, method name) instead of object id.
def _bound_builtin_table() -> dict[tuple[int, str], tuple[str, str]]:
    return {
        (id(itertools.chain), "from_iterable"): (
            "import itertools",
            "itertools.chain.from_iterable",
        ),
    }


def _emit_via_surface(obj: object, imports: set[str]) -> str:
    """``emit_value`` whose ``module.Name`` references to torch-internal definitions
    (a metadata dataclass, an enum member, a NamedTuple type) are redirected to the
    ``standalone_runtime`` surface whenever it re-exports that exact object, so a
    baked value pulls in ``from ...standalone_runtime import Name`` rather than the
    defining module. A name the surface does not re-export (e.g. the ViewMeta C++
    bindings) keeps its defining-module import, exactly as emit_value produced it."""
    import importlib

    from . import standalone_runtime as rt

    emitted: set[str] = set()
    expr = emit_value(obj, emitted)
    for stmt in sorted(emitted):
        module_name = stmt.removeprefix("import ")
        if not module_name.startswith("torch."):
            imports.add(stmt)
            continue
        module = importlib.import_module(module_name)
        keep_module_import = False

        def route(match: re.Match[str]) -> str:
            nonlocal keep_module_import
            name = match.group(1)
            if name in rt.__all__ and getattr(rt, name) is getattr(module, name, None):
                imports.add(f"{_SURFACE_IMPORT} {name}")
                return name
            keep_module_import = True
            return match.group(0)

        expr = re.sub(rf"\b{re.escape(module_name)}\.([A-Za-z_]\w*)", route, expr)
        if keep_module_import:
            imports.add(stmt)
    return expr


_MODULE_HEADER = """\
# Generated by torch._functorch.aot_autograd.compile_to_python -- do not edit.
#
# Self-contained, executable module exposing ``call(flat_inputs) -> outputs`` for
# the post-AOTAutograd graph. The Inductor kernels JIT-compile from the inlined
# source on first call (no cache needed). The prelude/epilogue is AOTAutograd's own
# codegen'd runtime wrappers, not reimplemented: each (the orchestration and any chain
# wrappers) is spliced as a real top-level ``def`` with its closed-over globals (inner
# ``call``, sibling wrappers, public helpers, baked metadata reconstructed as source)
# hoisted to module-scope assignments -- so results match eager. The companion opaque
# cache is only an acceleration; this module never reads it.
"""

_TRAINING_MODULE_HEADER = """\
# Generated by torch._functorch.aot_autograd.compile_to_python (training) -- do not edit.
#
# Self-contained module exposing ``call(flat_inputs) -> outputs`` for a differentiable
# post-AOTAutograd graph. The Inductor FORWARD and BACKWARD lowerings are inlined (each
# module's top-level names carry a ``_sN`` suffix so the two share one namespace) and
# bridged by the ``_CompiledFunction`` autograd Function below, whose forward/backward
# bodies are AOTAutograd's own codegen'd wrappers. Calling convention:
#
# - ``call`` runs the forward under ``torch.enable_grad`` like AOTAutograd's runtime
#   wrapper; freshly created outputs carry a ``_CompiledFunction`` grad_fn, so their
#   ``.backward()`` runs the inlined backward kernels. Detaching under an ambient
#   ``torch.no_grad()`` is the caller's job (the torch.compiler.precompile driver does
#   it); this module never inspects the ambient grad mode.
# - Every input's ``requires_grad`` is baked from capture and not guarded here (the
#   precompile driver guards it); a flipped input yields a wrong autograd graph.
# - Grads reach ``_CompiledFunction.backward`` as ONE boxed list (``boxed_grads_call``),
#   ``None`` marking an undefined output tangent.
# - ``_AOT_BACKWARD_VARIANTS`` maps a canonical undefined-tangent bitmask (bit i set when
#   specializable user output i received no grad) to the ``_BackwardVariant`` serving
#   it: the Inductor backward to run, the saved-arg indices it keeps, the outputs it
#   prunes to None (``None`` = decided at runtime from ``_BACKWARD_OUTPUT_DEPENDENCIES``)
#   and the tangents it leaves unmaterialized. Mask 0 (all tangents defined) is always
#   present and serves any pattern not in the table by materializing the missing
#   tangents from the prototypes saved at forward time; nothing compiles at serve time.
# - Undefined-tangent handling (mainline's aot_autograd_prune_unused_outputs) is baked
#   on; the module reads no functorch / dynamo / inductor config at load or call time.
"""

# The codegen'd signatures the standalone module invokes positionally (see
# runtime_wrappers.py: _codegen_runtime_wrapper / _codegen_compiled_forward /
# _codegen_compiled_backward / the compiled_fn_wrapper builder).
_ORCHESTRATION_PARAMS = ["_compiled_fn_", "_first_ctx_", "_on_before_call_", "args"]
_TRAINING_CODEGEN_PARAMS = {
    "compiled_function_forward": [
        "ctx",
        "args",
        "_rng_add_",
        "_save_",
        "_finalize_",
        "_compiled_fw_",
    ],
    "compiled_function_backward": [
        "_flat_args_",
        "_ctx_",
        "_prologue_",
        "_rng_add_",
        "_impl_",
        "_epilogue_",
        "_double_bw_",
    ],
    "compiled_fn_wrapper": ["raw_returns"],
}


def _resolve_global(
    obj: object,
    helper_table: dict[int, tuple[str, str]],
    inner_call_id: int | None,
    fn_id_to_name: dict[int, str],
    imports: set[str],
    orch_closure_id: int | None = None,
    orch_entry_name: str | None = None,
    inner_call_name: str = "_inner_call",
) -> str:
    """Return a Python expression (valid in the generated module) that reproduces
    ``obj``, recording any needed import. Raises NotImplementedError if ``obj`` is
    neither the inner call, a sibling wrapper, a known helper, nor source-
    reconstructible (see ``_emit_value``)."""
    if inner_call_id is not None and id(obj) == inner_call_id:
        return inner_call_name
    # An OUTER wrapper (dedup / synthetic base) closes over the orchestration's outer
    # closure as its inner. That closure is not a captured wrapper and has no import
    # path, so wire it to the single-arg orchestration entry adapter the composer emits.
    if orch_closure_id is not None and id(obj) == orch_closure_id:
        if orch_entry_name is None:
            raise AssertionError("expected orch_entry_name to be not None")
        return orch_entry_name
    if id(obj) in fn_id_to_name:
        return fn_id_to_name[id(obj)]
    if id(obj) in helper_table:
        import_stmt, expr = helper_table[id(obj)]
        imports.add(import_stmt)
        return expr
    owner = getattr(obj, "__self__", None)
    if owner is not None:
        bound = _bound_builtin_table().get((id(owner), getattr(obj, "__name__", "")))
        if bound is not None:
            import_stmt, expr = bound
            imports.add(import_stmt)
            return expr
    # Not a wired reference (inner call / sibling wrapper / helper): emit ``obj`` as
    # plain reconstruction source. Raises if it is not source-expressible.
    return _emit_via_surface(obj, imports)


def _check_signature(gen: GeneratedSource, expected: list[str], what: str) -> None:
    """Reject a captured wrapper whose codegen'd signature drifted from the positional
    call the standalone module makes. The FULL signature is compared, not just the
    positional params: the emitted call is purely positional, so an added keyword-only
    / *args / **kwargs param would otherwise be silently dropped."""
    fn_def = next(
        (
            n
            for n in ast.walk(ast.parse(gen.source))
            if isinstance(n, ast.FunctionDef) and n.name == gen.fn_name
        ),
        None,
    )
    params = None
    if fn_def is not None:
        a = fn_def.args
        params = [p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)]
        if a.vararg is not None:
            params.append("*" + a.vararg.arg)
        if a.kwarg is not None:
            params.append("**" + a.kwarg.arg)
    if params != expected:
        raise NotImplementedError(
            f"aot_autograd.compile_to_python: the {what} signature changed (expected "
            f"{expected}, got {params}); the standalone module invokes it positionally "
            "and must be updated to match."
        )


def _module_level_names(tree: ast.Module) -> set[str]:
    """Names bound at module scope by a parsed module. Used to seed ``_reserve`` so an
    inlined wrapper's def name or hoisted global (chain wrapper or orchestration) cannot
    silently shadow a top-level name the inner Inductor module already binds."""
    names: set[str] = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            # Walk each target so tuple/list/starred unpacking (``a, b = ...`` /
            # ``first, *rest = ...``) is covered, not just bare-name targets.
            for t in n.targets:
                names.update(x.id for x in ast.walk(t) if isinstance(x, ast.Name))
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            names.add(n.target.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.Delete):
            # Inductor's inner module does ``async_compile = AsyncCompile()`` then
            # ``del async_compile`` at module scope; a del'd name does not survive, so it
            # must not be reserved (else a hoisted wrapper global of the same name would
            # trip a spurious collision). Body order is assign-then-del, so removing here
            # leaves only names that actually persist.
            for t in n.targets:
                names.difference_update(
                    x.id for x in ast.walk(t) if isinstance(x, ast.Name)
                )
    return names


def _compose_standalone_module(
    inner_python: str, captured: list[GeneratedSource], inner_call_obj: Any
) -> str:
    """Compose the inner Inductor ``call`` with AOTAutograd's captured runtime
    wrappers into one standalone module exposing ``call(flat_inputs) -> outputs``.

    Every wrapper (chain wrappers and the orchestration) is spliced as a real top-level
    ``def`` with its closed-over globals hoisted to module-scope assignments (resolved
    here). They are chained by name in the SAME nesting the runtime builds: the
    orchestration is invoked with the INNER InductorWrapper chain head (subclass / RNG)
    as its inner, and any OUTER ``CompilerWrapper`` (dedup / synthetic base, applied
    AROUND the orchestration in ``graph_compile._aot_stage2c_make_inference_function``)
    wraps a single-arg adapter over the orchestration.

    ``inner_call_obj`` is the placeholder the capture pass returned for the inner call;
    its identity is authoritative (see the note at the inner-call site below).
    """
    # The capture sink is duration-scoped over the inner inductor compile, with no
    # originating-graph id at install time, so a re-entrant on-thread AOTAutograd /
    # inductor lowering during that window that codegen's wrappers into THIS sink would
    # append ITS wrappers here too. (A nested compile_to_python installs its OWN sink via
    # capture_generated_sources, so its wrappers go there, not here -- it is not this
    # case.) Each captured wrapper is tagged at append time with its TracingContext
    # identity (origin_id), which separates such a foreign lowering when it ran under a
    # DISTINCT TracingContext. A same-context re-entrant lowering reuses the ambient
    # TracingContext via try_get() and so shares this origin_id; that case is instead
    # caught by the orchestration count/wiring guards below (a second orchestration trips
    # the len() != 1 check). Filter to the target graph's origin before composing. The
    # target is the origin of the LAST captured orchestration wrapper: a foreign lowering
    # appends its orchestration before the outer one finishes, so the final orchestration
    # is always the outer/target one.
    orchestrations = [
        g for g in captured if g.artifact_name == "runtime_wrapper_orchestration"
    ]
    if orchestrations:
        # target_origin is None only without an ambient TracingContext, which the real
        # precompile path never hits (capture always runs under one); in that defensive
        # case the filter keeps every None-origin wrapper and the count/wiring guards
        # below remain the backstop.
        target_origin = orchestrations[-1].origin_id
        captured = [g for g in captured if g.origin_id == target_origin]

    # Backward wrappers are out of scope for forward lowering; reject them up front.
    # Every other wrapper that can appear in a composable (cacheable) forward graph is
    # codegen'd as source and captured here. The one non-codegen'd wrapper,
    # FakifiedOutWrapper, only activates under fakify_first_call, which makes the graph
    # non-cacheable -- so such a graph is rejected before it ever reaches composition.
    unsupported = [g.artifact_name for g in captured if "backward" in g.artifact_name]
    if unsupported:
        raise NotImplementedError(
            "aot_autograd.compile_to_python cannot yet compose these runtime "
            f"wrappers into standalone source: {sorted(set(unsupported))}."
        )

    orchestration = [
        g for g in captured if g.artifact_name == "runtime_wrapper_orchestration"
    ]
    if len(orchestration) != 1:
        raise NotImplementedError(
            "aot_autograd.compile_to_python expected exactly one forward "
            f"orchestration wrapper, captured {len(orchestration)}."
        )
    orch = orchestration[0]
    non_orch = [g for g in captured if g is not orch]

    # The generated ``call`` invokes the orchestration POSITIONALLY by its own name (see
    # the bottom of this function): _runtime_wrapper(chain_head, contextlib.nullcontext,
    # lambda: None, flat_inputs). That mapping is hardcoded to the codegen'd signature in
    # runtime_wrappers.py; verify the captured signature still matches so a future
    # rename/reorder fails loudly here instead of silently passing wrong arguments.
    _check_signature(orch, _ORCHESTRATION_PARAMS, "orchestration wrapper")

    helper_table = _known_helper_table()
    # Every wrapper is inlined (below) as a real def at module scope under its OWN codegen'd
    # name, so references resolve to that name. Note these names are NOT distinct in general
    # -- the subclass, dedup, and debug-assert chain wrappers all codegen ``inner_fn``; what
    # holds today is that at most one chain wrapper appears per composable forward graph (see
    # the test note on multi-link chains), so the names don't actually clash. ``_reserve``
    # fails loudly if that ever stops holding (the old ``_wrapper_{i}`` scheme could carry
    # multiple same-named wrappers in private exec namespaces; inlining deliberately cannot).
    fn_id_to_name = {id(g.fn): g.fn_name for g in non_orch}

    # A chain wrapper references the inner it wraps via one of these globals
    # (subclass/dedup use ``compiled_fn``; the functionalized-RNG wrapper uses
    # ``_compiled_fn_``). The orchestration takes its inner as a call-time arg, not a
    # global, so it is never a chain wrapper. MAINTAINERS: if AOTAutograd adds a
    # forward chain wrapper that names its inner via a new global, add that name here,
    # otherwise inner-call/chain-head detection silently bypasses it.
    _INNER_NAMES = ("compiled_fn", "_compiled_fn_")

    def _inner_ref(g: GeneratedSource) -> Any:
        for nm in _INNER_NAMES:
            if nm in g.globals_dict:
                return g.globals_dict[nm]
        return None

    # The inner Inductor call is AUTHORITATIVE: it is the placeholder object the capture
    # pass returned (threaded in as ``inner_call_obj``), NOT inferred from capture order.
    # This is what lets the composer tell the inner call apart from the orchestration's
    # own outer closure -- both surface as some wrapper's inner-ref yet neither is a
    # captured wrapper fn -- which is precisely how INNER wrappers (subclass / RNG,
    # wrapping the inner call, composed INSIDE the orchestration) are distinguished from
    # OUTER wrappers (dedup / synthetic base, ``CompilerWrapper``s applied AROUND the
    # orchestration in graph_compile._aot_stage2c_make_inference_function).
    inner_call_id: int = id(inner_call_obj)

    # Name of the single-arg adapter emitted over the orchestration; the innermost outer
    # wrapper closes over the orchestration's outer closure and is wired to this name.
    _ORCH_ENTRY = "_orchestration_entry"

    # The orchestration's outer closure is the object outer wrappers wrap. The composer
    # never captures it directly (the captured orchestration ``fn`` is the inner
    # ``_codegen_runtime_wrapper``), so recognize it structurally: an inner-ref that is
    # neither the inner call nor any captured wrapper's fn can only be that closure.
    orch_closure_ids = {
        id(_inner_ref(g))
        for g in non_orch
        if _inner_ref(g) is not None
        and id(_inner_ref(g)) != inner_call_id
        and id(_inner_ref(g)) not in fn_id_to_name
    }
    if len(orch_closure_ids) > 1:
        raise NotImplementedError(
            "aot_autograd.compile_to_python: captured multiple wrappers whose inner "
            "reference is neither the inner call nor a captured wrapper; cannot tell "
            "which wraps the orchestration."
        )
    orch_closure_id: int | None = next(iter(orch_closure_ids), None)

    # Walk the OUTER chain outward from the orchestration closure: the innermost outer
    # wrapper wraps the closure, the next wraps that wrapper's fn, and so on. Everything
    # else is inner-side -- the subclass / RNG chain wrappers plus the alias / mutation
    # epilogue helpers the orchestration closes over (which reference no inner at all).
    outer_wrappers: list[GeneratedSource] = []
    if orch_closure_id is not None:
        target_id: int | None = orch_closure_id
        while target_id is not None:
            nxt = next(
                (
                    g
                    for g in non_orch
                    if g not in outer_wrappers
                    and _inner_ref(g) is not None
                    and id(_inner_ref(g)) == target_id
                ),
                None,
            )
            if nxt is None:
                break
            outer_wrappers.append(nxt)
            target_id = id(nxt.fn)
    outer_ids = {id(g) for g in outer_wrappers}
    inner_side = [g for g in non_orch if id(g) not in outer_ids]

    # Chain head passed to the orchestration: the outermost INNER InductorWrapper (last
    # inner-side wrapper that wraps via an inner reference), else the inner call. Computed
    # up front (a pure capture-order check) so the order-inversion guard below fires
    # before the later name-uniqueness guard -- a mis-ordered chain is the more specific
    # diagnosis.
    chain_head = "_inner_call"
    chain_head_g: GeneratedSource | None = None
    for g in inner_side:
        if _inner_ref(g) is not None:
            chain_head = fn_id_to_name[id(g.fn)]
            chain_head_g = g

    # "Last with an inner-ref == outermost" holds only when INNER capture order is
    # innermost-to-outermost (it is today: subclass before functionalized-RNG). Back that
    # assumption with a guard: the true outermost inner wrapper is the one NO other inner
    # wrapper wraps, i.e. whose fn is not referenced as another inner wrapper's inner. If
    # the chosen head is itself wrapped, capture order inverted and the chain would be
    # mis-ordered -- reject rather than silently emit a wrong chain (the wiring guard
    # below would not catch this, since every wrapper is still referenced somewhere).
    referenced_inner_ids = {
        id(_inner_ref(g)) for g in inner_side if _inner_ref(g) is not None
    }
    if chain_head_g is not None and id(chain_head_g.fn) in referenced_inner_ids:
        raise NotImplementedError(
            "aot_autograd.compile_to_python: the selected chain head is itself wrapped "
            "by another captured wrapper, so capture order is not innermost-to-outermost "
            "as assumed; refusing to emit a mis-ordered runtime-wrapper chain."
        )

    imports: set[str] = set()

    # Parse the inner module once: to verify it binds a module-level ``call`` (the
    # inner-call contract, checked below) and to collect its top-level names so no inlined
    # wrapper's def name or hoisted global can silently shadow one.
    inner_tree = ast.parse(inner_python)
    inner_module_names = _module_level_names(inner_tree)

    # Every runtime wrapper is inlined: its def is spliced at module scope and its
    # closed-over globals hoisted to top-level assignments (no exec / private namespace).
    # So every emitted top-level name -- each wrapper's def name and each hoisted global --
    # must be unique and must not shadow a name the inner Inductor module binds. This holds
    # in practice because at most one chain wrapper appears per composable forward graph, so
    # its def name (``inner_fn`` is shared across subclass/dedup/debug-assert wrappers) and
    # the inner-ref global ``compiled_fn`` each occur once, and metadata globals are
    # per-wrapper suffixed. ``_reserve`` guards it: a collision fails loudly (rename/namespace
    # needed) rather than silently rebinding a name.
    emitted_names = set(inner_module_names) | {
        "call",
        "_inner_call",
        "_rebuild",
        "contextlib",
        _ORCH_ENTRY,
    }

    def _reserve(name: str) -> None:
        if name in emitted_names:
            raise NotImplementedError(
                "aot_autograd.compile_to_python: generated top-level name "
                f"{name!r} collides with another top-level name in the composed module; "
                "inlining the runtime wrappers would shadow a binding."
            )
        emitted_names.add(name)

    # Reserve every wrapper def name up front (before hoists) so a hoisted global cannot
    # shadow a def and two wrappers cannot share a name.
    for _g in (*non_orch, orch):
        _reserve(_g.fn_name)

    def _resolve_globals(globals_dict: dict[str, object]) -> list[tuple[str, str]]:
        # Resolve each global a wrapper closes over to a standalone source expression.
        # ``globals_dict`` is the pre-exec snapshot from codegen.py, so the
        # interpreter ``__builtins__`` is absent; the skip is kept defensively in case a
        # future caller hands us a post-exec live dict.
        out: list[tuple[str, str]] = []
        for gname, gobj in globals_dict.items():
            if gname == "__builtins__":
                continue
            expr = _resolve_global(
                gobj,
                helper_table,
                inner_call_id,
                fn_id_to_name,
                imports,
                orch_closure_id,
                _ORCH_ENTRY,
            )
            out.append((gname, expr))
        return out

    def _emit_inline(source: str, globals_dict: dict[str, object]) -> str:
        # Splice the wrapper's def verbatim at module scope, hoisting each closed-over global
        # to a top-level assignment (skipping a name already module-available -- an imported
        # helper or ``torch`` -- detected as gname == its resolved expr). No exec / private
        # namespace: the def reads as ordinary code and is referenced by its own name. Each
        # hoisted name is ``_reserve``'d so a collision fails loudly rather than rebinding.
        hoists: list[str] = []
        for gname, expr in _resolve_globals(globals_dict):
            if gname == expr:
                continue  # already at module scope (an import / ``torch``)
            _reserve(gname)
            hoists.append(f"{gname} = {expr}")
        return "\n".join(hoists + [source, ""])

    # Inner-side wrappers first (subclass / RNG chain wrappers innermost-to-outermost,
    # plus the epilogue helpers), then the orchestration, then the outer wrappers -- all
    # spliced as real defs. An inner wrapper's hoisted inner-ref (``compiled_fn``)
    # references ``_inner_call`` / a sibling emitted earlier; an outer wrapper's inner-ref
    # references the orchestration entry adapter / a sibling emitted earlier -- so order
    # is satisfied.
    inner_blocks = [_emit_inline(g.source, g.globals_dict) for g in inner_side]
    orch_block = _emit_inline(orch.source, orch.globals_dict)
    outer_blocks = [_emit_inline(g.source, g.globals_dict) for g in outer_wrappers]

    # Imports (helper table + whatever emit_value added) are emitted BEFORE the inner
    # module and the wrapper blocks, so a later top-level binding of the same name -- an
    # inner-module binding or a hoisted wrapper global -- would shadow the import. And
    # _emit_inline skips hoisting a global whose resolved expr already equals its own name
    # (an imported helper referenced as ``gname == expr``), so such a wrapper would then
    # silently bind to the shadowing object instead of the helper. ``from X import Y``
    # names bind a specific (non-module) object and are the ones at risk; plain ``import
    # mod`` names resolve to the same singleton module no matter who imports them, so a
    # duplicate binding is benign and is left unchecked (else ``import torch`` -- which the
    # inner module also emits -- would trip a spurious collision). ``_reserve`` fails loudly
    # if an inner-module name or a hoisted global collides with one of these import names.
    for stmt in sorted(imports):
        node = ast.parse(stmt).body[0]
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                _reserve(alias.asname or alias.name)

    # The final ``call`` invokes the outermost outer wrapper (else the orchestration
    # directly). Build that line now so the wiring guard's corpus includes it -- the
    # outermost outer wrapper is referenced ONLY there.
    if outer_wrappers:
        outermost_name = fn_id_to_name[id(outer_wrappers[-1].fn)]
        final_invoke = f"    return {outermost_name}(list(flat_inputs))"
    else:
        final_invoke = (
            f"    return {orch.fn_name}(\n"
            f"        {chain_head}, contextlib.nullcontext, lambda: None, "
            "list(flat_inputs)\n    )"
        )

    # The single-arg adapter over the orchestration (emitted only when outer wrappers
    # wrap it). The outer wrappers call their inner as ``fn(args)``; this adapts the
    # orchestration's positional (_compiled_fn_, _first_ctx_, _on_before_call_, args)
    # signature to that. When there are no outer wrappers the orchestration is invoked
    # directly in ``call`` (see final_invoke).
    orch_invoke_comment = [
        "    # The 2nd/3rd positional args INTENTIONALLY substitute contextlib.nullcontext",
        "    # for the runtime's first-invocation context (_FirstInvocationContext) and a",
        "    # no-op for the profiler-prologue exit. This drops two cold-start diagnostics:",
        "    # the first-call custom-op aliasing analysis (_AnalyzeCustomOpInputOutputMode,",
        "    # active when check_custom_op_aliasing is set, which can even RAISE under",
        "    # error_on_custom_op_aliasing) and the profiler prologue. Neither affects",
        "    # numerics, so this is not a bug -- the standalone artifact deliberately omits",
        "    # them. (See the positional-mapping note in _compose_standalone_module.)",
    ]
    entry_block: list[str] = []
    if outer_wrappers:
        entry_block = [
            f"def {_ORCH_ENTRY}(args):",
            "    # Single-arg adapter so the CompilerWrappers applied AROUND the",
            "    # orchestration (dedup / synthetic base) can invoke it as ``fn(args)``.",
            *orch_invoke_comment,
            f"    return {orch.fn_name}(",
            f"        {chain_head}, contextlib.nullcontext, lambda: None, args",
            "    )",
            "",
        ]

    # _INNER_NAMES detection is a hardcoded allowlist (see above). If AOTAutograd adds
    # a forward wrapper that names its inner via an unrecognized global, that wrapper
    # is captured but may never be wired into the module -- silently composing a
    # structurally-wrong result. Enforce that every captured non-orch wrapper is
    # actually referenced somewhere: as the inner chain head, in the final ``call``
    # (the outermost outer wrapper), or by name in another block (another wrapper's
    # globals, the orchestration's epilogue helpers -- e.g. ``_alias_fn`` /
    # ``_apply_mutations`` -- or the entry adapter). A wrapper whose name appears in no
    # other block went unwired, so reject rather than emit a wrong module.
    block_of = {id(g): b for g, b in zip(inner_side, inner_blocks)}
    block_of.update({id(g): b for g, b in zip(outer_wrappers, outer_blocks)})
    other_text = "\n".join(
        inner_blocks + [orch_block] + entry_block + outer_blocks + [final_invoke]
    )
    for g in non_orch:
        name = fn_id_to_name[id(g.fn)]
        own = block_of[id(g)]
        elsewhere = other_text.replace(own, "", 1)
        # Whole-token match: ``name`` is a wrapper def name (e.g. ``inner_fn``); a raw
        # substring test would treat ``inner_fn`` as wired whenever a longer token like
        # ``inner_fn2`` is referenced, silently defeating this guard.
        wired = re.search(r"\b" + re.escape(name) + r"\b", elsewhere) is not None
        if name != chain_head and not wired:
            raise NotImplementedError(
                "aot_autograd.compile_to_python could not wire captured runtime "
                f"wrapper {g.fn_name!r} into the module (an inner-call global may be "
                "unrecognized; see _INNER_NAMES)."
            )

    # The module splices ``_inner_call = call`` below, relying on inner_python binding a
    # module-level ``call`` entry point. Inductor emits this in one of two forms: the
    # flat path defines ``def call(args):`` (FunctionDef) while the graph_partition Runner
    # path binds ``call = runner.call`` (Assign with a Name target). Verify one is present
    # so a future inductor codegen drift fails loudly here -- like the orchestration /
    # chain / wiring guards above -- instead of surfacing as a bare NameError at exec of
    # the generated module. (``inner_tree`` was parsed once up front.)
    binds_call = any(
        (isinstance(n, ast.FunctionDef) and n.name == "call")
        or (
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "call" for t in n.targets)
        )
        for n in inner_tree.body
    )
    if not binds_call:
        raise NotImplementedError(
            "compile_to_python: inner Inductor module does not bind a module-level "
            "'call' entry point (the inner-call contract); the standalone module "
            "splices ``_inner_call = call`` and must be updated to match."
        )

    # Only emit the _rebuild helper if a baked value actually reconstructs through it.
    needs_rebuild = any(
        "_rebuild(" in b for b in (*inner_blocks, orch_block, *outer_blocks)
    )

    # ``call`` invokes the orchestration directly when nothing wraps it; when outer
    # wrappers do, its body just calls the outermost of them (which drives the entry
    # adapter, then the orchestration) -- so the orchestration-substitution comment lives
    # on whichever site actually invokes the orchestration.
    if outer_wrappers:
        call_body = [
            "    # Invokes the outer CompilerWrapper chain (dedup / synthetic base), which",
            f"    # wraps {_ORCH_ENTRY} (the orchestration adapter).",
            final_invoke,
        ]
    else:
        call_body = [
            "    # AOTAutograd orchestration: disables grad, invokes the inner chain,",
            "    # bumps mutated-input versions, applies the output epilogue.",
            "    #",
            *orch_invoke_comment,
            final_invoke,
        ]

    parts = [
        _MODULE_HEADER,
        "import contextlib",
        *sorted(imports),
        "",
        "",
        *(_REBUILD_HELPER if needs_rebuild else []),
        "# " + "=" * 70,
        "# Inner Inductor output code (kernels + ``call``)",
        "# " + "=" * 70,
        inner_python,
        "_inner_call = call",
        "",
        "# " + "=" * 70,
        "# AOTAutograd runtime wrappers (codegen'd): each inlined as a real def with its",
        "# closed-over globals hoisted to module scope -- inner chain wrappers first, then",
        "# the orchestration, then any outer wrappers (dedup / synthetic base) that wrap it",
        "# " + "=" * 70,
        *inner_blocks,
        orch_block,
        *entry_block,
        *outer_blocks,
        "",
        "def call(flat_inputs):  # noqa: F811",
        *call_body,
        "",
    ]
    return "\n".join(parts)


def _find_effectful_op(gm: GraphModule, get_effect: Any) -> Any:
    """Return the first effectful op target reachable from ``gm``, or None.

    The target may be an ``OpOverload`` (e.g. ``aten::_print``) or a
    ``HigherOrderOperator`` -- ``call_torchbind`` / ``hop_print`` /
    ``invoke_leaf_function`` are registered ``_EffectType.ORDERED`` HOPs, not
    OpOverloads -- so both are checked (``get_effect`` returns None for a
    non-effectful HOP like ``cond``).

    Walks the graph and descends into any child GraphModule a node references -- a
    HOP (cond/while_loop/scan) holds its body as a get_attr'd submodule or passes it
    directly as a node arg -- so an effect nested inside a HOP subgraph is caught, not
    just effects at the top level."""
    import torch
    from torch.fx import GraphModule as _GraphModule

    seen: set[int] = set()

    def _scan(g: _GraphModule) -> Any:
        if id(g) in seen:
            return None
        seen.add(id(g))
        for node in g.graph.nodes:
            if (
                node.op == "call_function"
                and isinstance(
                    node.target,
                    (torch._ops.OpOverload, torch._ops.HigherOrderOperator),
                )
                and get_effect(node.target) is not None
            ):
                return node.target
            for sub in _iter_subgraphs(g, node):
                found = _scan(sub)
                if found is not None:
                    return found
        return None

    def _walk_values(value: Any) -> Iterator[_GraphModule]:
        # A GraphModule can appear as a direct node arg OR nested one level (or more)
        # inside a list/tuple/dict arg -- some HOPs pass their branch/body callables
        # inside a container -- so descend into containers before the isinstance check,
        # otherwise the recursive effect scan would never enter that nested subgraph.
        if isinstance(value, _GraphModule):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from _walk_values(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from _walk_values(item)

    def _iter_subgraphs(g: _GraphModule, node: Any) -> Iterator[_GraphModule]:
        # A child graph reaches a node either as an attribute fetched by get_attr or
        # as a (possibly container-nested) argument (the form HOPs use for their
        # branch/body callables). make_fx emits FLAT get_attr targets (e.g.
        # true_graph_0) since _scan recurses per-GraphModule, so a plain getattr
        # suffices (no dotted walk needed here).
        if node.op == "get_attr":
            attr = getattr(g, node.target, None)
            if isinstance(attr, _GraphModule):
                yield attr
        for arg in (*node.args, *node.kwargs.values()):
            yield from _walk_values(arg)

    return _scan(gm)


def _graph_has_dynamic_shapes(gm: GraphModule) -> bool:
    """True if any placeholder is itself a SymInt or carries symbolic (SymInt) size,
    stride, or storage-offset metadata -- i.e. the graph was traced with dynamic dims.
    Drives the shapes mode for the capture pass: dynamic graphs stay dynamic, static
    graphs specialize so the composer can bake their (static) view metadata. Strides and
    storage offset are checked too, not just sizes: a graph dynamic solely via symbolic
    strides has static sizes, and treating it as static would silently specialize the
    artifact to the example strides. (Unbacked symints appearing only in intermediates,
    not on any placeholder, are still missed here, but such a graph fails loudly
    downstream when emit_value rejects the still-symbolic metadata.)

    Both metadata keys are checked, like ``_resolve_fake_mode``: make_fx stashes the fake
    under "val", while a Dynamo graph (which torch.compiler.precompile's dynamo tracer
    feeds here) stashes it under "example_value" -- reading only "val" would call a
    dynamic Dynamo graph static and silently specialize it to the example sizes."""
    import torch

    def _is_symbolic(v: Any) -> bool:
        return isinstance(v, torch.SymInt)

    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        for key in ("val", "example_value"):
            val = node.meta.get(key)
            if _is_symbolic(val):
                return True
            if isinstance(val, torch.Tensor) and (
                any(_is_symbolic(s) for s in val.shape)
                or any(_is_symbolic(s) for s in val.stride())
                or _is_symbolic(val.storage_offset())
            ):
                return True
    return False


def namespace_module_names(sources: Sequence[str]) -> list[str]:
    """Suffix every top-level name each module DEFINES, per module.

    Splicing several Inductor modules into ONE namespace is only safe if their
    top-level names are disjoint, because the code inside a module resolves its
    siblings as late-bound globals: a module's ``call`` looks up its kernels and
    its ``_runtime_wrapper`` when INVOKED, not when defined. Two modules of the
    same computation -- a forward and its backward, or two shape variants of one
    frame -- otherwise define the same names, and the first silently runs the
    second's code. Snapshotting each module's entry is not enough for the same
    reason.

    Rewriting is driven by AST positions rather than a text match, which is what
    keeps three lookalikes out of it: an attribute (``runner.call`` is an
    ``Attribute``, not a ``Name``), a nested binding (``def call`` inside
    ``class Runner`` is not module-level), and an import (``async_compile`` is
    both a local binding and part of ``torch._inductor.async_compile``). AST
    column offsets count UTF-8 BYTES, so the splice runs on the encoded line (a
    non-ASCII string literal earlier on the line would otherwise shift every later
    edit); a ``global`` / ``nonlocal`` statement names a binding without a Name
    node and is rewritten as a whole statement.
    """
    out: list[str] = []
    for slot, source in enumerate(sources):
        tree = ast.parse(source)
        imported: set[str] = set()
        defined: set[str] = set()
        headers: list[tuple[int, str]] = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    imported.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                defined.add(node.name)
                headers.append((node.lineno, node.name))
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                defined.add(node.target.id)
        targets = defined - imported
        if not targets:
            out.append(source)
            continue
        suffix = f"_s{slot}"
        lines = [line.encode() for line in source.split("\n")]
        edits: dict[int, list[tuple[int, int, bytes]]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id in targets and node.end_col_offset is not None:
                    renamed = (node.id + suffix).encode()
                    edits.setdefault(node.lineno, []).append(
                        (node.col_offset, node.end_col_offset, renamed)
                    )
            elif isinstance(node, (ast.Global, ast.Nonlocal)) and targets.intersection(
                node.names
            ):
                if node.end_lineno != node.lineno or node.end_col_offset is None:
                    raise NotImplementedError(
                        "namespace_module_names cannot rewrite a global/nonlocal "
                        "statement spanning several lines."
                    )
                keyword = "global" if isinstance(node, ast.Global) else "nonlocal"
                names = ", ".join(n + suffix if n in targets else n for n in node.names)
                statement = f"{keyword} {names}".encode()
                edits.setdefault(node.lineno, []).append(
                    (node.col_offset, node.end_col_offset, statement)
                )
        for lineno, name in headers:
            if name not in targets:
                continue
            header = re.match(
                rb"\s*(?:async\s+def|def|class)\s+("
                + re.escape(name.encode())
                + rb")\b",
                lines[lineno - 1],
            )
            if header is not None:
                edits.setdefault(lineno, []).append(
                    (header.start(1), header.end(1), (name + suffix).encode())
                )
        for lineno, spans in edits.items():
            line = lines[lineno - 1]
            for begin, finish, text in sorted(spans, reverse=True):
                line = line[:begin] + text + line[finish:]
            lines[lineno - 1] = line
        out.append("\n".join(line.decode("utf-8") for line in lines))
    return out


@dataclass(frozen=True)
class _CompiledBackwardVariant:
    """One lowered backward and how the emitted backward runs it (field semantics as
    in ``standalone_runtime._BackwardVariant``). ``call`` is the loaded Inductor entry
    of a specialized (retraced / structurally pruned) variant, needed because the
    capturing module runs it right away; None means the variant runs the base
    all-tangents-defined backward, whose live entry the compile state learns from the
    capture namespace (``_BASE_BACKWARD_CALL``)."""

    python_code: str
    cache: bytes | None
    call: Callable[[list[Any]], Any] | None = None
    kept_arg_indices: tuple[int, ...] | None = None
    pruned_output_indices: tuple[int, ...] | None = None
    skip_materialize_indices: tuple[int, ...] = ()

    def runtime_variant(self, base_call: Callable[[list[Any]], Any]) -> Any:
        from .standalone_runtime import _BackwardVariant

        return _BackwardVariant(
            self.call if self.call is not None else base_call,
            self.kept_arg_indices,
            self.pruned_output_indices,
            self.skip_materialize_indices,
        )


# The two names of the emitted training module that the capture protocol touches;
# owned by _CompileToPythonState and the composer, never by callers.
_VARIANT_COMPILER_GLOBAL = "_AOT_BACKWARD_VARIANT_COMPILER"
_BASE_BACKWARD_CALL = "_inner_call_bw_0"


@dataclass
class _CompileToPythonState:
    """Capture-side owner of a training artifact's backward variants.

    ``_compile_to_python_with_state`` returns one per training compile. It holds the
    lowered forward, the base (all-tangents-defined) backward and everything needed to
    lower further backward variants later. Masks are canonical throughout -- restricted
    to the specializable user-output indices, like the live runtime's
    ``_specializable_user_grad_output_mask`` -- so a non-differentiable output's
    always-undefined tangent cannot fork spurious variants; every entry point
    re-canonicalizes defensively because ``finalize`` may be handed masks observed on a
    sibling backend (a stateful automatic-dynamic recompile) whose canonical form was
    computed against ITS metadata.

    Protocol. ``install_capture(namespace)`` binds ``compile_mask`` into the loaded
    module's namespace; from then on every backward the module runs reports its
    canonical mask here (``observed_masks``) and receives its variant -- lowered on
    first sight (``compiled_masks``) or reused -- while the module's own variant table
    stays untouched. ``uninstall_capture`` unbinds the hook, returning the module to
    serving from its table (an unseen mask falls back to mask 0). ``finalize(masks)``
    emits a fresh artifact covering mask 0 plus the given masks, lowering any it has
    not seen; it never touches the capture namespace, so a stateful capture keeps
    running the live module between snapshots. Callers never read or write the emitted
    module's globals themselves.
    """

    forward_python: str
    forward_cache: bytes | None
    forward_inner_call: Callable[..., Any]
    base_backward: _CompiledBackwardVariant
    captured: list[GeneratedSource]
    spec: Any
    backward_graph: GraphModule
    backward_inputs: list[Any]
    options: dict[str, Any] | None
    disable_saved_tensors_hooks: bool
    _namespace: dict[str, Any] | None = field(default=None, init=False)
    _base_call: Callable[[list[Any]], Any] | None = field(default=None, init=False)
    _observed: set[int] = field(default_factory=set, init=False)
    _variants: dict[int, _CompiledBackwardVariant] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        self._variants[0] = self.base_backward

    def install_capture(self, namespace: dict[str, Any]) -> None:
        self._namespace = namespace
        self._base_call = namespace[_BASE_BACKWARD_CALL]
        namespace[_VARIANT_COMPILER_GLOBAL] = self.compile_mask

    def uninstall_capture(self) -> None:
        if self._namespace is not None:
            self._namespace[_VARIANT_COMPILER_GLOBAL] = None
        self._namespace = None
        self._base_call = None

    def observed_masks(self) -> set[int]:
        with _COMPILE_LOCK:
            return set(self._observed)

    def compiled_masks(self) -> set[int]:
        with _COMPILE_LOCK:
            return set(self._variants)

    def canonical_mask(self, mask: int) -> int:
        from . import runtime_wrappers as _rw

        return _rw._specializable_user_grad_output_mask(
            self.spec.fw_metadata, _rw._bitmask_to_indices(mask)
        )

    def compile_mask(self, mask: int) -> Any:
        """The hook the capturing module calls from every backward (install_capture)."""
        with _COMPILE_LOCK:
            if self._base_call is None:
                raise AssertionError("compile_mask requires install_capture first")
            mask = self.canonical_mask(mask)
            # Recorded before lowering so a failing compile still leaves the
            # pattern observed (a later finalize retries or drops it, but the
            # caller can tell it was seen).
            self._observed.add(mask)
            return self._compile_mask(mask).runtime_variant(self._base_call)

    def _compile_mask(self, mask: int) -> _CompiledBackwardVariant:
        import dataclasses

        from torch._guards import compile_context, tracing
        from torch._inductor import (
            compile_to_python as _inductor_compile_to_python,
            load_from_python as _inductor_load_from_python,
        )

        from . import runtime_wrappers as _rw
        from .graph_compile import _retrace_backward_for_undefined_grad_outputs

        mask = self.canonical_mask(mask)
        compiled = self._variants.get(mask)
        if compiled is not None:
            return compiled

        specialization_indices = _rw._bitmask_to_indices(mask)
        result = None
        lazy_info = self.spec.lazy_backward_info
        if (
            isinstance(lazy_info, _rw.AutogradLazyBackwardCompileInfo)
            and lazy_info.autograd_trace_info is not None
        ):
            with (
                tracing(lazy_info.saved_context),
                compile_context(lazy_info.saved_compile_context),
            ):
                result = _retrace_backward_for_undefined_grad_outputs(
                    lazy_info.autograd_trace_info,
                    self.spec.aot_config,
                    specialization_indices,
                    self.backward_graph,
                    self.backward_inputs,
                )
        if result is None:
            # The retrace can decline for reasons unrelated to the ABI (e.g. the
            # ambient autocast state differs between forward and backward, the
            # standard AMP pattern); structural specialization preserves the
            # saved-activation ABI and stays available, as on the live path.
            result = _rw._specialize_bw_module_for_undefined_grad_outputs(
                self.backward_graph,
                self.backward_inputs,
                self.spec.fw_metadata,
                specialization_indices,
                list(self.backward_inputs),
            )
        if result is None:
            # Neither applies: run the base backward with the undefined tangents
            # materialized as zeros and its provably-zero outputs pruned -- the
            # live runtime's own fallback (_AutogradBackwardCompiler.get_or_compile).
            pruned = _rw._pruned_backward_output_indices_for_undefined_grad_outputs(
                self.backward_graph, self.spec.fw_metadata, specialization_indices
            )
            compiled = dataclasses.replace(
                self.base_backward, pruned_output_indices=pruned
            )
        else:
            graph, inputs, kept_arg_indices = result
            python_code, cache = _inductor_compile_to_python(
                graph,
                inputs,
                options=self.options,
                is_inference=False,
                is_backward=True,
            )
            compiled = _CompiledBackwardVariant(
                python_code=python_code,
                cache=cache,
                call=_inductor_load_from_python(python_code, cache),
                kept_arg_indices=kept_arg_indices,
                skip_materialize_indices=specialization_indices,
            )
        self._variants[mask] = compiled
        return compiled

    def finalize(self, tangent_masks: Sequence[int]) -> tuple[str, bytes | None]:
        from torch.compiler._cache import CacheArtifactManager

        with _COMPILE_LOCK:
            masks = sorted({0, *(self.canonical_mask(mask) for mask in tangent_masks)})
            variants = {mask: self._compile_mask(mask) for mask in masks}
            source = _compose_training_module(
                self.forward_python,
                variants,
                self.captured,
                self.spec,
                self.backward_graph,
                self.forward_inner_call,
                self.disable_saved_tensors_hooks,
            )
            caches = (self.forward_cache, *(v.cache for v in variants.values()))
            return source, CacheArtifactManager.merge(caches)


def _emit_fw_metadata(meta: Any, imports: set[str]) -> list[str]:
    """Bake ``_fw_metadata`` as its constructor call plus an explicit assignment for
    every attribute the constructor does not take. ``ViewAndMutationMeta.__post_init__``
    derives state from the AMBIENT config (``is_rng_op_functionalized`` and what
    follows from it, which the loading process' config must not change) and from
    fields ``make_runtime_safe`` later clears (``output_types``), and the compile
    pipeline assigns more afterwards (``dynamic_saved_tensors_idxs``). emit_value's
    round-trip check sees none of that -- the class's ``__eq__`` compares eight fields
    -- so every non-init attribute is restored explicitly, whatever it is."""
    import dataclasses

    init_fields = {f.name for f in dataclasses.fields(meta) if f.init}
    lines = [f"_fw_metadata = {_emit_via_surface(meta, imports)}"]
    for name, value in vars(meta).items():
        if name not in init_fields:
            lines.append(f"_fw_metadata.{name} = {_emit_via_surface(value, imports)}")
    return lines


def _training_glue() -> tuple[list[str], set[str], set[str]]:
    """The autograd-bridge source spliced into a training artifact, read off
    ``standalone_training_glue`` (real, linted code rather than an f-string):
    ``(blocks, imports, declared)``. ``imports`` are the module's runtime imports,
    which the artifact must carry too; ``declared`` are the ``_BOUND_BY_ARTIFACT``
    placeholders naming the module globals the template expects the artifact to bind,
    which the composer checks it binds exactly."""
    import inspect

    from . import standalone_training_glue as glue

    imports: set[str] = set()
    declared: set[str] = set()
    for node in ast.parse(inspect.getsource(glue)).body:
        if isinstance(node, ast.Import):
            imports.update(f"import {alias.name}" for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module not in (
            # Annotations are lazy in the artifact (from __future__ import
            # annotations), so typing-only imports are not carried over.
            "__future__",
            "typing",
        ):
            imports.update(f"from {node.module} import {a.name}" for a in node.names)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(node.value, ast.Name)
            and node.value.id == "_BOUND_BY_ARTIFACT"
        ):
            declared.add(node.target.id)
    return [inspect.getsource(obj) for obj in glue.SPLICED], imports, declared


def _closes_over(fn: Any, target: object) -> bool:
    """Whether ``fn`` -- following functools.wraps' ``__wrapped__`` chain -- is a
    closure with a cell bound to ``target``. This is how the orchestration's live
    closure is recognized: ``create_runtime_wrapper``'s ``runtime_wrapper`` closes over
    the codegen'd orchestration fn the capture recorded, and its saved-tensors-hooks
    variant wraps that closure."""
    seen: set[int] = set()
    while fn is not None and id(fn) not in seen:
        seen.add(id(fn))
        for cell in getattr(fn, "__closure__", None) or ():
            try:
                contents = cell.cell_contents
            except ValueError:
                continue
            if contents is target:
                return True
        fn = getattr(fn, "__wrapped__", None)
    return False


_TRAINING_CORE_ARTIFACTS = (
    "backward_prologue",
    "backward_epilogue",
    "compiled_fn_wrapper",
    "compiled_function_forward",
    "compiled_function_backward",
    "runtime_wrapper_orchestration",
)


def _compose_training_module(
    fw_python: str,
    backward_variants: Mapping[int, _CompiledBackwardVariant],
    captured: list[GeneratedSource],
    spec: Any,
    bw_gm: GraphModule,
    forward_inner_call: Callable[..., Any],
    disable_saved_tensors_hooks: bool,
) -> str:
    """Compose a FORWARD and its BACKWARD variants into one standalone training module.

    The inference composer nests wrappers around a single ``call``. Training is a
    different shape: AOTAutograd's bridge is a ``torch.autograd.Function`` whose
    forward/backward bodies ARE codegen'd source (``compiled_function_forward`` /
    ``compiled_function_backward``, with ``backward_prologue`` / ``backward_epilogue``
    / ``compiled_fn_wrapper`` beside them), but the class holding them is ordinary
    Python, so it is spliced from ``standalone_training_glue`` and wired to the
    per-artifact bindings emitted just before it. The Inductor modules -- the forward
    plus one per distinct backward lowering -- are spliced at module level after
    ``namespace_module_names`` suffixes their top-level names, because each module's
    code resolves its kernels and ``call`` as late-bound globals and the modules
    otherwise define the same names.

    Wrapper nesting mirrors the live runtime (``_aot_stage2c_make_autograd_function``):
    the inner chain (subclass / functionalized RNG) wraps the inner forward call and is
    what ``_CompiledFunction.forward`` invokes; the orchestration wraps the autograd
    Function; ``DebugAssertWrapper`` and the dedup / synthetic-base wrappers wrap the
    orchestration (through a single-arg entry adapter), and the saved-tensors-hooks
    disable ``create_runtime_wrapper`` adds for a joint sits inside that adapter. Each
    captured wrapper is placed by the IDENTITY of what it wraps; one wrapping anything
    else is rejected rather than guessed. ``backward_variants`` maps canonical tangent
    masks to lowered backwards and must contain mask 0, the universal fallback.
    """
    from . import runtime_wrappers as _rw

    if spec.backward_state_indices:
        raise NotImplementedError(
            "aot_autograd.compile_to_python cannot compose a training graph that "
            "carries a BackwardState into standalone source yet."
        )
    if 0 not in backward_variants:
        raise AssertionError(
            "the training compose requires the mask-0 backward variant"
        )

    # Filter to the target graph's capture origin (see _compose_standalone_module).
    orchestrations = [
        gen for gen in captured if gen.artifact_name == "runtime_wrapper_orchestration"
    ]
    if orchestrations:
        target_origin = orchestrations[-1].origin_id
        captured = [gen for gen in captured if gen.origin_id == target_origin]

    by_name: dict[str, list[GeneratedSource]] = {}
    for gen in captured:
        by_name.setdefault(gen.artifact_name, []).append(gen)
    for name in _TRAINING_CORE_ARTIFACTS:
        if len(by_name.get(name, [])) != 1:
            raise NotImplementedError(
                "aot_autograd.compile_to_python: the training compose expects "
                f"AOTAutograd to codegen exactly one {name!r} wrapper, captured "
                f"{len(by_name.get(name, []))}."
            )
    core = {name: by_name[name][0] for name in _TRAINING_CORE_ARTIFACTS}
    orch = core["runtime_wrapper_orchestration"]
    # Every codegen'd function the emitted module (or the glue) calls positionally.
    _check_signature(orch, _ORCHESTRATION_PARAMS, "orchestration wrapper")
    for artifact_name, expected in _TRAINING_CODEGEN_PARAMS.items():
        _check_signature(core[artifact_name], expected, f"codegen'd {artifact_name}")

    auxiliary = [gen for gen in captured if gen.artifact_name not in core]
    fn_id_to_name = {id(gen.fn): gen.fn_name for gen in auxiliary}

    # A wrapper names the callable it wraps via one of these globals (subclass /
    # dedup / debug-assert use ``compiled_fn``, the functionalized-RNG wrapper
    # ``_compiled_fn_``); the epilogue helpers the orchestration closes over
    # (alias regen, mutation copy-back) wrap nothing.
    def inner_ref(gen: GeneratedSource) -> Any:
        for name in ("compiled_fn", "_compiled_fn_"):
            if name in gen.globals_dict:
                return gen.globals_dict[name]
        return None

    # Place each wrapper by the identity of what it wraps: the inner forward call or
    # a sibling wrapper (an inner-chain link, or a wrapper around another outer
    # wrapper, wired by name below), or the orchestration's live closure -- the
    # innermost OUTER wrapper. Anything else cannot be wired and is rejected.
    roots: list[GeneratedSource] = []
    for gen in auxiliary:
        ref = inner_ref(gen)
        if ref is None or ref is forward_inner_call or id(ref) in fn_id_to_name:
            continue
        if not _closes_over(ref, orch.fn):
            raise NotImplementedError(
                f"aot_autograd.compile_to_python: captured training wrapper "
                f"{gen.fn_name!r} ({gen.artifact_name}) wraps a callable that is "
                "neither the inner forward call, a sibling wrapper nor the "
                "orchestration; refusing to guess its place in the chain."
            )
        roots.append(gen)
    if len(roots) > 1:
        raise NotImplementedError(
            "aot_autograd.compile_to_python: several captured training wrappers "
            f"({[gen.fn_name for gen in roots]}) wrap the orchestration directly."
        )
    orch_closure_id: int | None = None
    outer_wrappers: list[GeneratedSource] = []
    if roots:
        orch_closure_id = id(inner_ref(roots[0]))
        target_id: int | None = orch_closure_id
        while target_id is not None:
            wrapper = next(
                (
                    gen
                    for gen in auxiliary
                    if gen not in outer_wrappers
                    and inner_ref(gen) is not None
                    and id(inner_ref(gen)) == target_id
                ),
                None,
            )
            if wrapper is None:
                break
            outer_wrappers.append(wrapper)
            target_id = id(wrapper.fn)
    outer_ids = {id(gen) for gen in outer_wrappers}
    inner_auxiliary = [gen for gen in auxiliary if id(gen) not in outer_ids]

    # What _CompiledFunction.forward invokes: the inner forward call, or the
    # outermost inner-chain wrapper AOTAutograd built around it.
    forward_call_name = "_inner_call_fw"
    if spec.compiled_fw_func is not forward_inner_call:
        forward_wrapper = fn_id_to_name.get(id(spec.compiled_fw_func))
        if forward_wrapper is None or id(spec.compiled_fw_func) in {
            id(gen.fn) for gen in outer_wrappers
        }:
            raise NotImplementedError(
                "aot_autograd.compile_to_python could not identify the captured "
                "training forward wrapper chain."
            )
        forward_call_name = forward_wrapper

    # One Inductor module per distinct backward lowering; variants that run the
    # base backward with different pruning share its source. Mask 0 sorts first,
    # so the base backward is always ``_inner_call_bw_0`` (_BASE_BACKWARD_CALL).
    masks = sorted(backward_variants)
    backward_sources: list[str] = []
    source_index: dict[str, int] = {}
    for mask in masks:
        code = backward_variants[mask].python_code
        if code not in source_index:
            source_index[code] = len(backward_sources)
            backward_sources.append(code)
    fw_ns, *bw_ns = namespace_module_names([fw_python, *backward_sources])
    backward_call_names = [f"_inner_call_bw_{i}" for i in range(len(bw_ns))]
    if backward_call_names[0] != _BASE_BACKWARD_CALL:
        raise AssertionError("the base backward must be the first backward lowering")

    glue_blocks, glue_imports, glue_declared = _training_glue()
    imports: set[str] = set(glue_imports)
    helper_table = _known_helper_table()
    entry_name = "_autograd_orchestration_entry"

    # Every emitted top-level name -- wrapper defs, hoisted globals, the glue, the
    # bindings, from-imports -- must be unique and must not shadow a name either
    # Inductor module binds; a collision fails loudly rather than rebinding.
    emitted_names: set[str] = {
        "call",
        "contextlib",
        "weakref",
        "_rebuild",
        "_inner_call_fw",
        entry_name,
        *backward_call_names,
    }
    for module_source in (fw_ns, *bw_ns):
        emitted_names |= _module_level_names(ast.parse(module_source))

    def _reserve(name: str) -> None:
        if name in emitted_names:
            raise NotImplementedError(
                "aot_autograd.compile_to_python: generated top-level name "
                f"{name!r} collides with another top-level name in the composed "
                "training module; inlining would shadow a binding."
            )
        emitted_names.add(name)

    for gen in (*auxiliary, *core.values()):
        _reserve(gen.fn_name)
    for block in glue_blocks:
        for node in ast.parse(block).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                _reserve(node.name)

    global_bindings: dict[str, str] = {}

    def splice(gen: GeneratedSource) -> str:
        hoists = []
        for name, obj in gen.globals_dict.items():
            if name == "__builtins__":
                continue
            expr = _resolve_global(
                obj,
                helper_table,
                id(forward_inner_call),
                fn_id_to_name,
                imports,
                orch_closure_id,
                entry_name,
                inner_call_name="_inner_call_fw",
            )
            if name == expr:
                continue
            previous = global_bindings.get(name)
            if previous is not None:
                if previous != expr:
                    raise NotImplementedError(
                        "aot_autograd.compile_to_python cannot inline training "
                        f"wrappers that bind {name!r} to different values."
                    )
                continue
            _reserve(name)
            global_bindings[name] = expr
            hoists.append(f"{name} = {expr}")
        return "\n".join(hoists + [gen.source, ""])

    inner_blocks = [splice(gen) for gen in inner_auxiliary]
    core_blocks = [
        splice(core[name])
        for name in _TRAINING_CORE_ARTIFACTS
        if name != "runtime_wrapper_orchestration"
    ]
    orchestration_block = splice(orch)
    outer_blocks = [splice(gen) for gen in outer_wrappers]

    meta = spec.fw_metadata
    # The grad-output indices a backward variant may specialize on (surviving
    # user outputs); the emitted backward canonicalizes its scanned mask over
    # these, like the live _specializable_user_grad_output_mask.
    specializable = _rw._specializable_user_grad_output_indices(
        meta, range(meta.num_forward_returns)
    )
    # Provably-zero bit per backward output, computed once with EVERY
    # specializable tangent pruned: a dependency-eligible output's cone contains
    # exactly the visible tangents its dependency entry lists, so one pass
    # answers "is output i zero once the tangents it depends on are undefined"
    # for all of them. The mask-0 variant's runtime pruning masks only outputs
    # that are BOTH dependency-implied AND in this set -- dependency alone is
    # wrong for a backward that is not linear in its tangents (a custom Function
    # returning g + 1 yields a nonzero grad from an undefined tangent, which
    # eager materializes zeros for). Mirrors the runtime
    # _pruned_backward_output_indices_for_undefined_grad_outputs.
    tangent_placeholders = [
        node
        for node in bw_gm.graph.find_nodes(op="placeholder")
        if _rw._is_grad_tangent(node)
    ]
    expected_tangents = sum(
        _rw._tangent_meta_arg_count(m) for m in meta.subclass_tangent_meta
    )
    provably_zero: frozenset[int] = frozenset()
    if len(tangent_placeholders) == expected_tangents:
        flat_indices = _rw._undefined_tangent_flat_indices(meta, specializable)
        provably_zero = _rw._provably_zero_backward_output_indices(
            bw_gm, {tangent_placeholders[i] for i in flat_indices}
        )
    dependencies = _rw._backward_output_tangent_dependencies(bw_gm, meta)

    for name in ("_AutogradSavedState", "_AutogradRngStateTracker", "_BackwardVariant"):
        imports.add(f"{_SURFACE_IMPORT} {name}")
    variant_lines = ["_AOT_BACKWARD_VARIANTS = {"]
    for mask in masks:
        variant = backward_variants[mask]
        variant_lines.append(
            f"    {mask:#b}: _BackwardVariant("
            f"inner_call={backward_call_names[source_index[variant.python_code]]}, "
            f"kept_arg_indices={variant.kept_arg_indices!r}, "
            f"pruned_output_indices={variant.pruned_output_indices!r}, "
            f"skip_materialize_indices={variant.skip_materialize_indices!r}),"
        )
    variant_lines.append("}")
    bindings = [
        *_emit_fw_metadata(meta, imports),
        "_saved_state = _AutogradSavedState(metadata=_fw_metadata)",
        # A fresh tracker: its RNG generators and iteration counter belong to the
        # serving process, not to the capture that emitted the artifact.
        "_rng_state = _AutogradRngStateTracker("
        f"num_rng={meta.num_graphsafe_rng_states!r}, "
        f"graphsafe_idx={meta.graphsafe_rng_state_index!r}, "
        f"device={_emit_via_surface(meta.graphsafe_rng_device, imports)})",
        f"_BACKWARD_OUTPUT_DEPENDENCIES = {_emit_via_surface(dependencies, imports)}",
        f"_BACKWARD_OUTPUT_PROVABLY_ZERO = frozenset({tuple(sorted(provably_zero))!r})",
        f"_AOT_SPECIALIZABLE_GRAD_OUT_INDICES = frozenset({tuple(specializable)!r})",
        f"_NUM_FORWARD_RETURNS = {meta.num_forward_returns!r}",
        f"_DISABLE_AMP = {spec.disable_amp!r}",
        f"_FORWARD_CALL = {forward_call_name}",
        "\n".join(variant_lines),
        f"{_VARIANT_COMPILER_GLOBAL} = None",
    ]
    bound = {
        target.id
        for node in ast.parse("\n".join(bindings)).body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    codegen_names = {gen.fn_name for gen in core.values() if gen is not orch}
    if glue_declared != bound | codegen_names:
        raise AssertionError(
            f"standalone_training_glue declares {sorted(glue_declared)} but the "
            f"composer binds {sorted(bound | codegen_names)}"
        )
    for name in sorted(bound):
        _reserve(name)
    for stmt in sorted(imports):
        node = ast.parse(stmt).body[0]
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                _reserve(alias.asname or alias.name)

    # The autograd Function is the orchestration's inner call; the outer wrappers
    # (debug-assert / dedup / synthetic base) invoke the orchestration as fn(args)
    # through this adapter. The 2nd/3rd args substitute nullcontext / a no-op for
    # the first-invocation context and profiler prologue (diagnostics only; see the
    # positional-mapping note in _compose_standalone_module).
    invoke = (
        f"{orch.fn_name}(_boxed_autograd_apply, contextlib.nullcontext, "
        "lambda: None, args)"
    )
    if disable_saved_tensors_hooks:
        imports.add(f"{_SURFACE_IMPORT} _disable_saved_tensors_hooks")
        _reserve("_disable_saved_tensors_hooks")
        entry_body = [
            "    # The joint inlined the saved-tensors hooks active at capture, so",
            "    # hooks are disabled around the compiled region as the live",
            "    # runtime wrapper does (create_runtime_wrapper, trace_joint).",
            "    with _disable_saved_tensors_hooks():",
            f"        return {invoke}",
        ]
    else:
        entry_body = [f"    return {invoke}"]
    entry_block = [f"def {entry_name}(args):", *entry_body, ""]
    outermost = (
        fn_id_to_name[id(outer_wrappers[-1].fn)] if outer_wrappers else entry_name
    )
    call_block = [
        "def call(flat_inputs):  # noqa: F811",
        f"    return {outermost}(list(flat_inputs))",
        "",
    ]

    # Every captured auxiliary wrapper must be referenced by name outside its own
    # block -- as _FORWARD_CALL, by a sibling's hoisted global, or by ``call`` --
    # otherwise it went unwired and the module would be structurally wrong.
    block_of = dict(zip((id(g) for g in inner_auxiliary), inner_blocks))
    block_of.update(zip((id(g) for g in outer_wrappers), outer_blocks))
    corpus = "\n".join(
        [
            *inner_blocks,
            *core_blocks,
            orchestration_block,
            *outer_blocks,
            *bindings,
            *entry_block,
            *call_block,
        ]
    )
    for gen in auxiliary:
        elsewhere = corpus.replace(block_of[id(gen)], "", 1)
        if re.search(rf"\b{re.escape(gen.fn_name)}\b", elsewhere) is None:
            raise NotImplementedError(
                "aot_autograd.compile_to_python could not wire captured runtime "
                f"wrapper {gen.fn_name!r} ({gen.artifact_name}) into the training "
                "module."
            )

    needs_rebuild = any(
        "_rebuild(" in text
        for text in (
            *inner_blocks,
            *core_blocks,
            orchestration_block,
            *outer_blocks,
            *bindings,
        )
    )
    backward_sections: list[str] = []
    for index, module_source in enumerate(bw_ns):
        served = [
            f"{mask:#b}"
            for mask in masks
            if source_index[backward_variants[mask].python_code] == index
        ]
        backward_sections += [
            "# " + "=" * 70,
            f"# Inner Inductor output code: BACKWARD lowering {index} "
            f"(tangent masks {', '.join(served)})",
            "# " + "=" * 70,
            module_source,
            f"{backward_call_names[index]} = call_s{index + 1}",
            "",
        ]
    parts = [
        _TRAINING_MODULE_HEADER,
        "from __future__ import annotations",
        "",
        *sorted(imports),
        "",
        "",
        *(_REBUILD_HELPER if needs_rebuild else []),
        "# " + "=" * 70,
        "# Inner Inductor output code: FORWARD",
        "# " + "=" * 70,
        fw_ns,
        "_inner_call_fw = call_s0",
        "",
        *backward_sections,
        "# " + "=" * 70,
        "# AOTAutograd runtime wrappers (codegen'd): the inner forward chain and",
        "# epilogue helpers, the autograd Function's prologue / epilogue / forward /",
        "# backward bodies, then the orchestration",
        "# " + "=" * 70,
        *inner_blocks,
        *core_blocks,
        orchestration_block,
        "# " + "=" * 70,
        "# Autograd bridge: baked metadata, the backward variant table, and the",
        "# _CompiledFunction glue (spliced from standalone_training_glue.py)",
        "# " + "=" * 70,
        *bindings,
        "",
        "",
        *glue_blocks,
        *entry_block,
        *outer_blocks,
        *call_block,
    ]
    return "\n".join(parts)


def _restride_backward_placeholders(
    bw_gm: GraphModule,
    fwd_output_strides: Sequence[tuple[str, ...] | None],
    spec: Any,
) -> None:
    """Restride the backward's saved-activation inputs to what the forward chose.

    Layout optimization lets the compiled forward hand back e.g. channels-last
    saved activations, so the backward must be lowered against those strides
    rather than the eager ones its joint trace carries. torch.compile does this
    in ``_aot_stage2b_bw_compile`` with strides reported out of the forward's
    lowering; this is the same restride against the strides
    ``_compile_to_python_impl`` reports.

    The rewrite lands on ``node.meta["val"]``, NOT on an example-inputs list:
    ``inductor.compile_to_python`` rebuilds its fakes from the placeholders'
    metadata and ignores the inputs it is handed, so restriding a list is a
    silent no-op through that entry point.
    """
    import torch
    from torch._inductor.utils import shape_env_from_inputs
    from torch.fx.experimental.symbolic_shapes import statically_known_true

    from .graph_compile import _get_inner_meta

    if not fwd_output_strides:
        return
    # Which of the forward's outputs are the saved activations, and where they
    # sit among the backward's inputs, are both recorded -- guessing positionally
    # restrides the wrong placeholder and asserts (or miscomputes) at runtime.
    # The dense forward returns one output per INNER tensor of a subclass, so its
    # layout is described by the inner metadata, not the user-facing one
    # (mirrors _aot_stage2b_bw_compile).
    meta = _get_inner_meta(spec.maybe_subclass_meta, spec.fw_metadata)
    saved = list(fwd_output_strides[meta.tensors_saved_for_backwards_slice])
    num_symints = spec.num_symints_saved_for_bw
    placeholders = [n for n in bw_gm.graph.nodes if n.op == "placeholder"]
    shape_env = shape_env_from_inputs([node.meta.get("val") for node in placeholders])
    for index, node in enumerate(placeholders):
        val = node.meta.get("val")
        if not isinstance(val, torch.Tensor):
            continue
        offset = index - num_symints
        if not (0 <= offset < len(saved)) or not saved[offset]:
            continue
        real = tuple(
            cast("int | torch.SymInt", shape_env.deserialize_symexpr(s))
            if shape_env is not None and isinstance(s, str)
            else int(s)
            for s in saved[offset]
        )
        if len(real) == val.dim() and not all(
            statically_known_true(actual == expected)
            for actual, expected in zip(val.stride(), real)
        ):
            node.meta["val"] = val.as_strided(val.size(), real)


def _compile_to_python_with_state(
    gm: GraphModule,
    example_inputs: Sequence[Any],
    *,
    options: dict[str, Any] | None = None,
    grad_enabled: bool = False,
) -> tuple[str, bytes | None, _CompileToPythonState | None]:
    """Compile ``gm`` to ``(python_code, cache, state)``; see the module docstring.

    ``grad_enabled`` runs the AOTAutograd capture pass under ``enable_grad`` instead of the
    default ``no_grad``, and covers two different graphs. One performs autograd INTERNALLY
    -- a Dynamo graph captured with ``trace_autograd_ops``, whose traced
    ``torch.autograd.grad`` call needs a live autograd graph to differentiate and otherwise
    fails the capture pass with "element 0 of tensors does not require grad"; that is still
    an INFERENCE graph at the AOT boundary, since its backward lives inside the traced call.
    The other has INPUTS that require grad, and AOTAutograd then emits a joint
    forward+backward: two dense graphs, composed into a module whose ``call`` returns
    outputs carrying ``grad_fn``, so the caller's ``.backward()`` runs the compiled
    backward; ``state`` (None for an inference graph) captures further backward variants
    (see ``_CompileToPythonState``). Leaving it off pins the inference path, which is what
    you want for a forward you will never differentiate.

    THREADING: serialized by a process-global lock (``_COMPILE_LOCK``). The wrapper-source
    capture is thread-local, but the AOTAutograd pass and the inner inductor compile both
    swap process-global cache state (``CacheArtifactManager.with_fresh_cache()``); a
    concurrent compile on another thread would corrupt the captured wrappers or cache
    artifacts, so concurrent calls (including via ``torch.compiler.precompile``) are
    serialized rather than run in parallel.
    """
    import copy

    import torch
    from torch._functorch import config as functorch_config
    from torch._higher_order_ops.effects import _get_effect
    from torch._inductor import compile_to_python as _inductor_compile_to_python
    from torch._inductor.compile_fx import compile_fx
    from torch._inductor.standalone_compile import (
        _compile_to_python_impl,
        _resolve_ignore_shape_env,
        _standalone_context,
    )
    from torch.fx.graph_module import _share_torchbind_and_process_group_on_deepcopy

    from .utils import make_boxed_func

    # Validate up front: this layer dereferences ``gm.graph`` (the effectful-op scan
    # below) before reaching inductor's own type-check, so a non-GraphModule would
    # otherwise surface as an opaque AttributeError instead of this clear contract error.
    if not isinstance(gm, torch.fx.GraphModule):
        raise TypeError(
            "aot_autograd.compile_to_python expects a post-AOTAutograd "
            f"torch.fx.GraphModule, got {type(gm)}."
        )

    # Effectful ops thread effect tokens through a calling convention the standalone
    # composition does not reproduce (and their with_effects HOP is non-cacheable);
    # reject them up front with a concrete reason. Not supported yet. (Detected here
    # too, not only in torch.compiler.precompile's capture-time guard, so direct callers of
    # this lowering get the same clear failure rather than a silently-wrong artifact.)
    # Scan recursively: a HOP (cond/while_loop/scan) carries its body as a child
    # GraphModule referenced by a get_attr node (or passed directly as a node arg), so
    # effects nested in such a subgraph would be missed by a top-level-only scan.
    effectful = _find_effectful_op(gm, _get_effect)
    if effectful is not None:
        raise NotImplementedError(
            "aot_autograd.compile_to_python cannot lower this graph to standalone "
            f"source: it contains an effectful op ({effectful}), which is not "
            "supported yet."
        )

    with _COMPILE_LOCK:
        # Run AOTAutograd ONCE to do two things: (1) produce the dense, decomposed,
        # functionalized inner graph, and (2) codegen its runtime-wrapper chain, which the
        # thread-local ``capture_generated_sources`` sink records. A capture-only inner
        # compiler grabs the dense graph and returns a placeholder boxed callable, so
        # AOTAutograd still builds and codegen's the wrappers AROUND it -- that codegen is
        # what we capture. Inductor is NOT run in this pass; it runs exactly once below, on
        # the captured dense graph, via the ``_compile_to_python_impl`` call (which
        # drives inductor's ``compile_fx_inner`` directly, not a re-entry into AOTAutograd).
        # The composer swaps the placeholder (the wrappers' inner reference) for the
        # inner inductor ``call`` by object identity, so the placeholder is only a
        # compile-time token and never runs.
        captured: list[GeneratedSource] = []
        dense: dict[str, Any] = {}
        # The training compose needs AOTAutograd's own spec (fw_metadata, RNG
        # state, disable_amp, the saved/symint counts). It is built during the
        # capture pass and not otherwise reachable from out here.
        from . import runtime_wrappers as _rw

        specs: list[Any] = []

        def _capture_inner_compile(dense_gm, dense_inputs, **kwargs):
            # A training graph reaches this TWICE -- once for the forward, once
            # for the backward -- and inductor says which via is_backward. Keep
            # the "only one of each" guard per slot; a third call is still
            # something this layer does not model.
            slot = "bw" if kwargs.get("is_backward") else "gm"
            if slot in dense:
                raise NotImplementedError(
                    "aot_autograd.compile_to_python does not support a graph whose "
                    f"AOTAutograd lowering emits more than one inner {slot} graph."
                )
            dense[slot] = dense_gm
            dense[f"{slot}_inputs"] = list(dense_inputs)
            # Retain the placeholder's IDENTITY: it is the authoritative inner call the
            # runtime-wrapper chain closes over. The composer needs it to tell the inner
            # call apart from the orchestration's own outer closure (both surface as a
            # wrapper's inner-ref yet neither is a captured wrapper fn), which is what
            # separates INNER wrappers from the OUTER dedup / synthetic-base wrappers.
            placeholder = make_boxed_func(dense_gm.forward)
            if slot == "gm":
                dense["placeholder"] = placeholder
                # AOTAutograd's own dispatch decision, threaded through by inductor:
                # False for a joint's forward, True for aot_dispatch_base -- so the
                # forward is lowered under exactly the flag torch.compile would use.
                dense["is_inference"] = bool(kwargs.get("is_inference", False))
            return placeholder

        # Drive inductor's own ``compile_fx`` (i.e. its exact AOTAutograd invocation --
        # decomposition table + aot config) but swap in the capture inner compiler so the
        # dense graph is intercepted before codegen and no inductor compile happens. Using
        # compile_fx (rather than calling aot_autograd directly) guarantees the dense graph
        # matches what the step-2 inductor compile below expects. Pick the shapes mode from
        # the graph (there is no dynamic_shapes knob): a symbolically-traced graph uses
        # ``"from_graph"`` to stay dynamic, a static one ``"from_example_inputs"`` to
        # specialize -- matching what the composer can bake (symbolic view metadata is
        # rejected downstream). grad mode selects an inference forward or a joint
        # forward/backward.
        # Deepcopy first so compile_fx cannot mutate the caller's gm (torchbind
        # ProcessGroups smuggled through as shared references). Note: the raw-collective /
        # torchbind rewrites are inductor-lowering prereqs and belong to the step-2 inductor
        # compile, which applies them to the dense graph -- not duplicated here.
        shapes_mode = (
            "from_tracing_context"
            if torch._guards.TracingContext.try_get() is not None
            else (
                "from_graph" if _graph_has_dynamic_shapes(gm) else "from_example_inputs"
            )
        )
        with (
            torch.enable_grad() if grad_enabled else torch.no_grad(),
            _standalone_context(gm, shapes_mode, aot=False),
            functorch_config.patch(
                enable_autograd_cache=False,
                enable_remote_autograd_cache=False,
            ),
            capture_generated_sources(captured),
            _rw.capture_aot_dispatch_autograd_specs(specs),
        ):
            with _share_torchbind_and_process_group_on_deepcopy():
                gm_owned = copy.deepcopy(gm)
            compile_fx(
                gm_owned,
                example_inputs,
                # Placeholder returns a boxed callable, not a full OutputCode;
                # AOTAutograd only wraps it (never inductor-post-compiles it), so
                # this is fine at runtime.
                inner_compile=_capture_inner_compile,  # pyrefly: ignore[bad-argument-type]
                ignore_shape_env=_resolve_ignore_shape_env(shapes_mode),
            )
            # Same ambient state create_runtime_wrapper just consulted: a joint
            # traced with inlineable saved-tensors hooks gets them disabled around
            # the compiled region, and the artifact must do the same.
            has_joint = "bw" in dense
            disable_saved_tensors_hooks = (
                has_joint and _rw._should_disable_saved_tensors_hooks()
            )
        if "gm" not in dense:
            raise RuntimeError(
                "aot_autograd.compile_to_python: AOTAutograd never reached the inner "
                "forward compiler, so no dense graph was captured."
            )

        # Lower the FORWARD first and take the strides inductor actually chose,
        # then restride the backward's placeholders to match before lowering it.
        # This is the fw->bw coupling torch.compile gets from
        # TracingContext.report_output_strides (graph_compile.py) and feeds
        # to _aot_stage2b_bw_compile; a capture pass that never lowers cannot
        # observe it, and two independently-lowered graphs then disagree about
        # layout -- loudly on a conv net, silently if size asserts are off.
        inner_python, forward_cache, fwd_output_strides = _compile_to_python_impl(
            dense["gm"],
            example_inputs,
            options=options,
            is_inference=dense["is_inference"],
            is_backward=False,
        )
        if not has_joint:
            source = _compose_standalone_module(
                inner_python, captured, dense["placeholder"]
            )
            return source, forward_cache, None

        if len(specs) != 1:
            raise NotImplementedError(
                "aot_autograd.compile_to_python expected exactly one training "
                f"autograd-function spec, captured {len(specs)}."
            )
        spec = specs[0]

        _restride_backward_placeholders(dense["bw"], fwd_output_strides, spec)
        bw_python, backward_cache = _inductor_compile_to_python(
            dense["bw"],
            [],
            options=options,
            is_inference=False,
            is_backward=True,
        )
        state = _CompileToPythonState(
            forward_python=inner_python,
            forward_cache=forward_cache,
            forward_inner_call=dense["placeholder"],
            # pruned_output_indices=None: the base backward decides what to prune
            # at runtime, which is what lets it serve every unseen tangent mask.
            base_backward=_CompiledBackwardVariant(
                python_code=bw_python, cache=backward_cache
            ),
            captured=captured,
            spec=spec,
            backward_graph=dense["bw"],
            backward_inputs=dense["bw_inputs"],
            options=options,
            disable_saved_tensors_hooks=disable_saved_tensors_hooks,
        )
        source, cache = state.finalize(())
    return source, cache, state


def compile_to_python(
    gm: GraphModule,
    example_inputs: Sequence[Any],
    *,
    options: dict[str, Any] | None = None,
    grad_enabled: bool = False,
) -> tuple[str, bytes | None]:
    """Compile ``gm`` to standalone Python source and an optional cache bundle.

    The module docstring describes the generated module's calling convention: the
    ``call(flat_inputs) -> outputs`` inference module, and -- when ``grad_enabled=True``
    and an input requires grad -- the training module whose outputs carry ``grad_fn``
    (see the TRAINING ARTIFACTS section there and ``_compile_to_python_with_state`` for
    what ``grad_enabled`` selects).
    """
    source, cache, _ = _compile_to_python_with_state(
        gm,
        example_inputs,
        options=options,
        grad_enabled=grad_enabled,
    )
    return source, cache


def load_from_python(
    python_code: str, cache: bytes | None = None
) -> Callable[..., Any]:
    """Load the module emitted by ``compile_to_python`` into a runnable ``call`` -- the
    inverse of ``compile_to_python``: (python_code, cache) in, runnable ``call`` out.

    The composed module is self-contained -- the inductor kernels and the pure-Python
    runtime wrappers are inlined -- so this delegates straight to the inductor loader:
    ``python_code`` runs standalone (kernels JIT-compile on first use), and ``cache`` (the
    inductor accelerator bundle this layer forwards) warms the kernel caches so exec loads
    the precompiled binaries instead of recompiling. There is no separate aot-level load
    step: exec'ing the module yields the aot-composed top-level ``call`` directly, and the
    wrappers carry no kernels to load.
    """
    from torch._inductor import load_from_python as _inductor_load_from_python

    return _inductor_load_from_python(python_code, cache)
