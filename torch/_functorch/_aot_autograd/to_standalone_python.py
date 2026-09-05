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

Not covered: a training artifact's ``.backward()`` cannot run under COMPILED AUTOGRAD.
Compiled autograd inlines an AOT backward from the fx ``bw_module`` the forward class
carries as ``_lazy_backward_info``, and a source artifact has none to carry (its
backward is already lowered to source). The emitted ``_CompiledFunction`` therefore
refuses with ``NotImplementedError`` the moment compiled autograd reads that attribute,
rather than letting it blame AOTAutogradCache; run the served backward under the
ordinary autograd engine. The runtime's real CompiledFunction does not have this
limitation -- it is the price of the source rendering.
"""

from __future__ import annotations

import ast
import contextlib
import re
import threading
from typing import Any, TYPE_CHECKING

from .codegen import capture_generated_sources, GeneratedSource
from .source_emit import _REBUILD_HELPER, emit_value


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

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
#     gen_alias_from_base, _unwrap_tensoralias, _dealias_marked_returns,
#     mark_dynamo_propagated_dynamic_indices,
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
# To trigger the capture we run AOTAutograd ourselves (under no_grad, the inference path --
# compile_to_python's ``grad_enabled`` is what lets a training graph emit a joint instead)
# with a capture-only inner compiler: it grabs the dense inner graph and returns a
# placeholder callable, so AOTAutograd still codegen's the runtime-wrapper chain AROUND
# that placeholder -- which is what the sink records. Inductor does not run in that pass;
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


# Global objects the codegen'd wrappers close over that are reproducible as an
# import in the standalone module (rather than reconstructed field-by-field). Maps
# object id -> (import_statement, expression). Built lazily to avoid import cycles.
def _known_helper_table() -> dict[int, tuple[str, str]]:
    # Generated artifacts import runtime helpers from the single stable surface
    # ``standalone_runtime`` (not scattered AOTAutograd internals).
    import torch

    from . import standalone_runtime as rt

    _RT = "from torch._functorch._aot_autograd.standalone_runtime import"
    table: dict[int, tuple[str, str]] = {
        id(torch): ("import torch", "torch"),
        id(rt.normalize_as_list): (f"{_RT} normalize_as_list", "normalize_as_list"),
        id(rt.mark_dynamo_propagated_dynamic_indices): (
            f"{_RT} mark_dynamo_propagated_dynamic_indices",
            "mark_dynamo_propagated_dynamic_indices",
        ),
        id(torch.autograd.graph.increment_version): (
            "import torch",
            "torch.autograd.graph.increment_version",
        ),
        id(rt.gen_alias_from_base): (
            f"{_RT} gen_alias_from_base",
            "gen_alias_from_base",
        ),
        id(rt._unwrap_tensoralias): (
            f"{_RT} _unwrap_tensoralias",
            "_unwrap_tensoralias",
        ),
        id(rt._dealias_marked_returns): (
            f"{_RT} _dealias_marked_returns",
            "_dealias_marked_returns",
        ),
        id(rt._replay_input_mutation): (
            f"{_RT} _replay_input_mutation",
            "_replay_input_mutation",
        ),
        id(rt.CUDARngStateHelper.get_torch_state_as_tuple): (
            f"{_RT} CUDARngStateHelper",
            "CUDARngStateHelper.get_torch_state_as_tuple",
        ),
        id(rt.CUDARngStateHelper.set_new_offset): (
            f"{_RT} CUDARngStateHelper",
            "CUDARngStateHelper.set_new_offset",
        ),
        id(rt.AOTDispatchAutograd.process_runtime_tangent): (
            f"{_RT} AOTDispatchAutograd",
            "AOTDispatchAutograd.process_runtime_tangent",
        ),
        id(rt.TensorAlias): (f"{_RT} TensorAlias", "TensorAlias"),
    }
    return table


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


def _resolve_global(
    obj: object,
    helper_table: dict[int, tuple[str, str]],
    inner_call_id: int | None,
    fn_id_to_name: dict[int, str],
    imports: set[str],
    orch_closure_id: int | None = None,
    orch_entry_name: str | None = None,
) -> str:
    """Return a Python expression (valid in the generated module) that reproduces
    ``obj``, recording any needed import. Raises NotImplementedError if ``obj`` is
    neither the inner call, a sibling wrapper, a known helper, nor source-
    reconstructible (see ``_emit_value``)."""
    if inner_call_id is not None and id(obj) == inner_call_id:
        return "_inner_call"
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
        if import_stmt:
            imports.add(import_stmt)
        return expr
    # Not a wired reference (inner call / sibling wrapper / helper): emit ``obj`` as
    # plain reconstruction source. Raises if it is not source-expressible.
    return emit_value(obj, imports)


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


def _check_runtime_wrapper_signature(orch: GeneratedSource) -> None:
    """The composed ``call`` invokes the orchestration POSITIONALLY:
    _runtime_wrapper(inner, contextlib.nullcontext, lambda: None, flat_inputs),
    hardcoded to the codegen'd signature in runtime_wrappers.py (``def
    _runtime_wrapper(_compiled_fn_, _first_ctx_, _on_before_call_, args)``).
    Verify the captured signature still matches so a future rename/reorder
    fails loudly here instead of silently passing wrong arguments. The FULL
    signature is compared, not just positional params: the standalone call is
    purely positional, so a keyword-only / *args / **kwargs param would be
    silently dropped.
    """
    expected = ["_compiled_fn_", "_first_ctx_", "_on_before_call_", "args"]
    orch_def = next(
        (
            n
            for n in ast.walk(ast.parse(orch.source))
            if isinstance(n, ast.FunctionDef) and n.name == orch.fn_name
        ),
        None,
    )
    args_node = orch_def.args if orch_def is not None else None
    if args_node is None:
        params = None
    else:
        params = [a.arg for a in (*args_node.posonlyargs, *args_node.args)]
        params += [a.arg for a in args_node.kwonlyargs]
        if args_node.vararg is not None:
            params.append("*" + args_node.vararg.arg)
        if args_node.kwarg is not None:
            params.append("**" + args_node.kwarg.arg)
    if params != expected:
        raise NotImplementedError(
            "aot_autograd.compile_to_python: the orchestration wrapper signature "
            f"changed (expected {expected}, got {params}); the "
            "standalone module invokes it positionally and must be updated to match."
        )


def _check_binds_call(inner_tree: ast.Module) -> None:
    """The composers splice ``_inner_call = call``, so the inner Inductor module must
    bind a module-level ``call``: ``def call(args)`` on the flat path or ``call =
    runner.call`` on the graph_partition Runner path. Fail here, like the other
    guards, instead of with a bare NameError at exec of the generated module."""
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

    _check_runtime_wrapper_signature(orch)

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

    _check_binds_call(inner_tree)

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
    both a local binding and part of ``torch._inductor.async_compile``).
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
        edits: dict[int, list[tuple[int, int, str]]] = {}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Name)
                and node.id in targets
                and node.end_col_offset is not None
            ):
                edits.setdefault(node.lineno, []).append(
                    (node.col_offset, node.end_col_offset, node.id + suffix)
                )
        lines = source.split("\n")
        for lineno, name in headers:
            if name not in targets:
                continue
            found = re.search(rf"\b{re.escape(name)}\b", lines[lineno - 1])
            if found is not None:
                edits.setdefault(lineno, []).append(
                    (found.start(), found.end(), name + suffix)
                )
        for lineno, spans in edits.items():
            line = lines[lineno - 1]
            for begin, finish, text in sorted(spans, reverse=True):
                line = line[:begin] + text + line[finish:]
            lines[lineno - 1] = line
        out.append("\n".join(lines))
    return out


class _AutogradSpecCapture:
    """State of the ``_AOTDispatchAutogradFunctionFactory.build`` hook: installed
    process-wide while any capture is active (``depth`` counts nested and concurrent
    captures), recording into a THREAD-LOCAL sink so a compile on another thread is
    neither recorded here nor disturbed -- the same contract as
    ``codegen.capture_generated_sources``."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.depth = 0
        self.orig_build: Callable[..., Any] | None = None
        self.tls = threading.local()


_SPEC_CAPTURE = _AutogradSpecCapture()


def _recording_build(factory: Any) -> Any:
    import torch

    sink = getattr(_SPEC_CAPTURE.tls, "sink", None)
    if sink is not None:
        ctx = torch._guards.TracingContext.try_get()
        sink.append((id(ctx) if ctx is not None else None, factory.spec))
    orig_build = _SPEC_CAPTURE.orig_build
    if orig_build is None:
        raise AssertionError("autograd spec capture hook ran before it was installed")
    return orig_build(factory)


@contextlib.contextmanager
def _capture_autograd_specs(into: list[tuple[int | None, Any]]) -> Iterator[None]:
    """Record every AOTDispatchAutogradCompileSpec AOTAutograd builds ON THIS THREAD
    while active.

    The training compose needs the spec (fw_metadata, RNG state, disable_amp,
    the saved/symint counts), which is built during the capture pass and not
    otherwise reachable. Each one is tagged with the TracingContext identity the
    codegen'd wrappers carry (``GeneratedSource.origin_id``), so a re-entrant
    lowering's spec is told apart the same way its wrappers are.
    """
    from . import runtime_wrappers as _rw

    factory = _rw._AOTDispatchAutogradFunctionFactory
    cap = _SPEC_CAPTURE
    with cap.lock:
        if cap.depth == 0:
            cap.orig_build = factory.build
            factory.build = _recording_build  # type: ignore[method-assign]
        cap.depth += 1
    prev = getattr(cap.tls, "sink", None)
    cap.tls.sink = into
    try:
        yield
    finally:
        cap.tls.sink = prev
        with cap.lock:
            cap.depth -= 1
            if cap.depth == 0:
                factory.build = cap.orig_build  # type: ignore[method-assign]


def _select_training_spec(
    specs: list[tuple[int | None, Any]], captured: list[GeneratedSource]
) -> Any:
    """The spec built under the same TracingContext as the captured orchestration."""
    orchestrations = [
        g for g in captured if g.artifact_name == "runtime_wrapper_orchestration"
    ]
    target_origin = orchestrations[-1].origin_id if orchestrations else None
    matching = [spec for origin, spec in specs if origin == target_origin]
    if len(matching) != 1:
        raise NotImplementedError(
            "aot_autograd.compile_to_python expected exactly one autograd spec "
            f"built for the training graph, captured {len(matching)}."
        )
    return matching[0]


def _compose_training_module(
    fw_python: str,
    bw_python: str,
    captured: list[GeneratedSource],
    spec: Any,
    fw_placeholder: Any,
    bw_placeholder: Any,
) -> str:
    """Compose a FORWARD and BACKWARD lowering into one standalone module.

    The inference composer nests wrappers around a single ``call``. Training is a
    different shape: AOTAutograd's own bridge is a ``torch.autograd.Function``
    whose ``forward``/``backward`` bodies ARE codegen'd source
    (``compiled_function_forward`` / ``compiled_function_backward``, with
    ``backward_prologue`` / ``backward_epilogue`` beside them), but the class
    holding them is ordinary Python, so the composer emits it.

    Both inductor modules are spliced at module level, and each one's ``call`` /
    ``Runner`` / kernels are renamed with a per-module suffix
    (``namespace_module_names``) because a module resolves those as late-bound
    globals when invoked, so the second block would otherwise rebind the names
    the first block's code looks up.

    ``fw_placeholder`` / ``bw_placeholder`` are the callables the capture pass
    returned for the two inner compiles; the emitted class wires ``compiled_fw`` /
    ``compiled_bw`` straight to the inner Inductor calls, which is only right if
    AOTAutograd wrapped nothing around them.
    """
    # The dynamo backend wraps every compiled backward in one torch._dynamo.disable
    # layer (Note [Wrapping bw_compiler in disable] in _dynamo/backends/common.py).
    compiled_bw = spec.compiled_bw_func
    if getattr(compiled_bw, "_torchdynamo_disable", False):
        compiled_bw = getattr(compiled_bw, "__wrapped__", compiled_bw)
    if spec.compiled_fw_func is not fw_placeholder or compiled_bw is not bw_placeholder:
        raise NotImplementedError(
            "aot_autograd.compile_to_python: the training compose wires the "
            "autograd Function straight to the inner Inductor calls, but AOTAutograd "
            "wrapped the captured forward/backward in an InductorWrapper it cannot "
            "yet splice (e.g. FakifiedOutWrapper under fakify_first_call)."
        )
    if spec.maybe_subclass_meta is not None:
        raise NotImplementedError(
            "aot_autograd.compile_to_python cannot compose a training graph with "
            "tensor subclasses into standalone source yet."
        )
    if spec.backward_state_indices:
        raise NotImplementedError(
            "aot_autograd.compile_to_python cannot compose a training graph that "
            "carries a BackwardState into standalone source yet."
        )
    from . import runtime_wrappers as _rw

    if _rw._should_disable_saved_tensors_hooks():
        # The runtime wraps a trace_joint call in _disable_saved_tensors_hooks
        # when inlineable ambient hooks were traced INTO the joint graph. That
        # wrapper is plain Python, never a captured GeneratedSource, so the
        # compose cannot splice it -- and without it the ambient hooks would
        # fire a SECOND time on _CompiledFunction's ctx.save_for_backward at
        # serve time (pack applied to already-packed activations: silently
        # corrupted gradients). Refuse, never mis-render.
        raise NotImplementedError(
            "aot_autograd.compile_to_python cannot compose a training graph "
            "captured under inlineable saved_tensors_hooks into standalone "
            "source yet: the runtime disables the ambient hooks around the "
            "compiled call and the composed artifact cannot."
        )

    # Same hazards as the inference composer above: a re-entrant lowering
    # under a DISTINCT TracingContext appends ITS wrappers to this sink, so
    # filter to the target orchestration's origin; a same-context re-entry is
    # caught by the exactly-one check below instead.
    orchestrations = [
        g for g in captured if g.artifact_name == "runtime_wrapper_orchestration"
    ]
    if orchestrations:
        target_origin = orchestrations[-1].origin_id
        captured = [g for g in captured if g.origin_id == target_origin]

    by_name: dict[str, list[GeneratedSource]] = {}
    for gen in captured:
        by_name.setdefault(gen.artifact_name, []).append(gen)
    required = (
        "backward_prologue",
        "backward_epilogue",
        "compiled_fn_wrapper",
        "compiled_function_forward",
        "compiled_function_backward",
        "runtime_wrapper_orchestration",
    )
    # The orchestration closes over these two epilogue helpers by reference
    # (``_replay_aliases_`` / ``_apply_mutations_``), each present only when the
    # graph aliases an output or mutates an input; they are spliced like the
    # inference composer's inner-side wrappers.
    epilogue_helpers = ("output_alias_wrapper", "mutation_epilogue")
    # All six required names: AOTAutograd codegens compiled_fn_wrapper
    # unconditionally alongside the forward/backward artifacts, and the
    # emitted glue calls _transform_raw_returns unguarded.
    missing = [name for name in required if name not in by_name]
    if missing:
        raise NotImplementedError(
            f"aot_autograd.compile_to_python: the training compose expects "
            f"AOTAutograd to codegen {missing}, which this graph did not produce."
        )
    # Refuse, never mis-render: a captured wrapper this compose does not
    # splice (synthetic base, dedup, functionalized RNG, ...) changes the
    # inner calling convention or the runtime behaviour, so silently dropping
    # it ships an artifact that only fails at serve time, while a raise here
    # keeps the caller's working pickled-bundle fallback. debug_assert_wrapper
    # is the deliberate exception: it is a metadata DIAGNOSTIC with no effect
    # on numerics, dropped like the profiler prologue -- and it is captured
    # whenever functorch's debug_assert is on, which the CI test environment
    # enables, so refusing it would turn every CI training render into a blob.
    modeled = (*required, *epilogue_helpers, "debug_assert_wrapper")
    unmodeled = sorted(name for name in by_name if name not in modeled)
    if unmodeled:
        raise NotImplementedError(
            "aot_autograd.compile_to_python: the training compose cannot yet "
            f"model these captured runtime wrappers: {unmodeled}."
        )
    if len(by_name["runtime_wrapper_orchestration"]) != 1:
        raise NotImplementedError(
            "aot_autograd.compile_to_python expected exactly one training "
            "orchestration wrapper, captured "
            f"{len(by_name['runtime_wrapper_orchestration'])}."
        )
    orch = by_name["runtime_wrapper_orchestration"][0]
    _check_runtime_wrapper_signature(orch)
    helpers = [gen for name in epilogue_helpers for gen in by_name.get(name, [])]
    orch_refs = {id(obj) for obj in orch.globals_dict.values()}
    unwired = [gen.artifact_name for gen in helpers if id(gen.fn) not in orch_refs]
    if unwired:
        raise NotImplementedError(
            "aot_autograd.compile_to_python: the training orchestration does not "
            f"close over the captured epilogue helper(s) {unwired}; refusing to "
            "emit a module that would never invoke them."
        )
    fn_id_to_name = {id(gen.fn): gen.fn_name for gen in helpers}

    imports: set[str] = set()
    helper_table = _known_helper_table()
    # Hoisted globals land in one flat namespace; the same name resolving to two
    # different expressions would silently rebind one wrapper's global.
    hoisted: dict[str, str] = {}

    def splice(gen: GeneratedSource) -> str:
        hoists = []
        for name, obj in gen.globals_dict.items():
            if name == "__builtins__":
                continue
            expr = _resolve_global(obj, helper_table, None, fn_id_to_name, imports)
            if name == expr:
                continue
            if hoisted.setdefault(name, expr) != expr:
                raise NotImplementedError(
                    "aot_autograd.compile_to_python: generated top-level name "
                    f"{name!r} is bound to two different values in the composed "
                    "training module."
                )
            hoists.append(f"{name} = {expr}")
        return "\n".join(hoists + [gen.source, ""])

    blocks = [
        splice(gen)
        for name in (
            "backward_prologue",
            "backward_epilogue",
            "compiled_fn_wrapper",
            "compiled_function_forward",
            "compiled_function_backward",
            *epilogue_helpers,
        )
        for gen in by_name.get(name, [])
    ]
    orchestration = splice(orch)

    fw_metadata_src = emit_value(spec.fw_metadata, imports)
    # dynamic_saved_tensors_idxs is set in __post_init__, so emit_value's
    # constructor-field reduction drops it (and ViewAndMutationMeta.__eq__ is
    # blind to it); reassign it after construction so the symbolic backward
    # keeps the strides it needs to restride saved tensors.
    dsi_src = emit_value(spec.fw_metadata.dynamic_saved_tensors_idxs, imports)
    # Emitted as a CONSTRUCTOR CALL, never by value-reduction: the tracker's
    # default runtime-state fields include an itertools.count, whose pickle
    # support was removed in Python 3.12+ (gone in 3.14), so reducing a live
    # instance fails there -- and that state is per-process anyway, so a load
    # must start it fresh.
    rng_device_src = emit_value(spec.fw_metadata.graphsafe_rng_device, imports)
    rng_src = (
        "_AutogradRngStateTracker("
        f"num_rng={spec.fw_metadata.num_graphsafe_rng_states!r}, "
        f"graphsafe_idx={spec.fw_metadata.graphsafe_rng_state_index!r}, "
        f"device={rng_device_src})"
    )
    imports |= {
        "import contextlib",
        "import torch",
        "import weakref",
        "from torch._functorch._aot_autograd.standalone_runtime import "
        "_AutogradRngStateTracker, _AutogradSavedState, "
        "_snapshot_external_objects, index_to_external_object_weakref, "
        "normalize_as_list",
    }

    glue = f"""
_fw_metadata = {fw_metadata_src}
_fw_metadata.dynamic_saved_tensors_idxs = {dsi_src}
_saved_state = _AutogradSavedState(metadata=_fw_metadata)
_rng_state = {rng_src}
_NUM_FORWARD_RETURNS = {spec.fw_metadata.num_forward_returns}
_DISABLE_AMP = {spec.disable_amp!r}


def _finalize(ctx, fw_outs):
    raw_returns = list(fw_outs[:_NUM_FORWARD_RETURNS])
    ctx.mark_non_differentiable(*_transform_raw_returns(raw_returns))
    ctx._materialize_non_diff_grads = False
    _snapshot_external_objects(ctx)
    return tuple(raw_returns)


class _NoCompiledAutogradError(NotImplementedError, AttributeError):
    # Also an AttributeError so hasattr/getattr-with-default read the attribute
    # as absent; compiled autograd reads it directly and gets the reason.
    pass


class _NoCompiledAutograd:
    # Compiled autograd reads _lazy_backward_info first and blames AOTAutogradCache
    # on None; this artifact has no fx bw_module to hand over, so refuse up front.
    def __get__(self, obj, objtype=None):
        raise _NoCompiledAutogradError(
            "compiled autograd is not supported for a standalone training artifact"
        )


class _CompiledFunction(torch.autograd.Function):
    # Same class attributes as AOTAutograd's CompiledFunction (runtime_wrappers.py).
    compiled_fw = _inner_call_fw
    compiled_bw = _inner_call_bw
    metadata = _fw_metadata
    # The compose refuses tensor subclasses, so this is exactly None here.
    maybe_subclass_metadata = None
    num_symints_saved_for_bw = {spec.num_symints_saved_for_bw!r}
    _aot_id = {spec.aot_config.aot_id!r}
    _lazy_backward_info = _NoCompiledAutograd()
    _bw_prologue_fn = _backward_prologue
    _bw_epilogue_fn = _backward_epilogue
    _fwd_fn = _compiled_forward
    _bwd_fn = _compiled_backward
    boxed_grads_call = True

    @staticmethod
    def _compiled_autograd_key(ctx):
        return (ctx._autograd_function_id, *ctx.symints)

    @staticmethod
    def forward(ctx, *deduped_flat_tensor_args):
        return _CompiledFunction._fwd_fn(
            ctx,
            deduped_flat_tensor_args,
            _rng_state.add_forward_args,
            _saved_state.save_from_forward,
            _finalize,
            _CompiledFunction.compiled_fw,
        )

    @staticmethod
    def backward(ctx, *flat_args):
        return _CompiledFunction._bwd_fn(
            flat_args,
            ctx,
            _CompiledFunction._bw_prologue_fn,
            _rng_state.add_backward_args,
            _CompiledFunction._backward_impl,
            _CompiledFunction._bw_epilogue_fn,
            _CompiledFunction._double_backward,
        )

    @staticmethod
    def _double_backward(ctx, impl_fn, all_args):
        class _CompiledFunctionBackward(torch.autograd.Function):
            _aot_id = _CompiledFunction._aot_id

            @staticmethod
            def forward(double_ctx, *unused_args):
                return impl_fn(double_ctx)

            @staticmethod
            def backward(ctx, *args):
                raise RuntimeError(
                    "torch.compile with aot_autograd does not currently support "
                    "double backward"
                )

        _CompiledFunctionBackward._compiled_autograd_key = (
            _CompiledFunction._compiled_autograd_key
        )
        if not any(
            t.requires_grad for t in all_args if isinstance(t, torch.Tensor)
        ):
            all_args = [torch.empty(0, requires_grad=True)] + all_args
        return _CompiledFunctionBackward.apply(*all_args)

    @staticmethod
    def _backward_impl(ctx, all_args):
        ctx.maybe_clear_saved_tensors()
        for idx, obj in getattr(ctx, "_external_objects", {{}}).items():
            index_to_external_object_weakref[idx] = weakref.ref(obj)
        amp = torch._C._DisableAutocast if _DISABLE_AMP else contextlib.nullcontext
        with amp():
            out = _CompiledFunction.compiled_bw(all_args)
        return normalize_as_list(out)


def _boxed_autograd_apply(args):
    return _CompiledFunction.apply(*args)


def call(flat_inputs):  # noqa: F811
    return _runtime_wrapper(
        _boxed_autograd_apply, contextlib.nullcontext, lambda: None, list(flat_inputs)
    )
"""

    # Both modules land in this one namespace and each defines ``call``,
    # ``Runner`` and its kernels; without renaming, the forward's ``call``
    # resolves the BACKWARD's kernels at invocation time.
    for inner_python in (fw_python, bw_python):
        _check_binds_call(ast.parse(inner_python))
    fw_ns, bw_ns = namespace_module_names([fw_python, bw_python])
    needs_rebuild = any(
        "_rebuild(" in b for b in (*blocks, orchestration, fw_metadata_src, rng_src)
    )
    # ``import torch`` first: in a fresh process the standalone_runtime import
    # must not be the one that pulls torch in.
    parts = [
        "# Generated by aot_autograd.compile_to_python (training) -- do not edit.",
        "",
        "import torch",
        *sorted(imports - {"import torch"}),
        "",
        "",
        *(_REBUILD_HELPER if needs_rebuild else []),
        "# === Inner Inductor output code: FORWARD ===",
        fw_ns,
        "_inner_call_fw = call_s0",
        "",
        "# === Inner Inductor output code: BACKWARD ===",
        bw_ns,
        "_inner_call_bw = call_s1",
        "",
        "# === AOTAutograd runtime wrappers ===",
        *blocks,
        orchestration,
        glue,
    ]
    return "\n".join(parts)


def _graph_differentiates(gm: GraphModule) -> bool:
    """Whether a dense graph carries an INLINED backward (a make_fx trace through
    ``.backward()`` has no joint). Inductor branches on is_inference in
    ``decide_layout_opt`` and in the post-grad passes (reorder_for_locality); the
    layout branch acts only on convolutions, whose backward is a named
    ``convolution_backward`` call, so matching the op name of call nodes suffices."""
    return any(
        node.op == "call_function" and "convolution_backward" in str(node.target)
        for node in gm.graph.nodes
    )


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
    ``_compile_to_python_result`` reports.

    The rewrite lands on ``node.meta["val"]``, NOT on an example-inputs list:
    ``inductor.compile_to_python`` rebuilds its fakes from the placeholders'
    metadata and ignores the inputs it is handed, so restriding a list is a
    silent no-op through that entry point.
    """
    import torch
    from torch.fx.experimental.symbolic_shapes import statically_known_true

    if not fwd_output_strides:
        return
    # Which of the forward's outputs are the saved activations, and where they
    # sit among the backward's inputs, are both recorded -- guessing positionally
    # restrides the wrong placeholder and asserts (or miscomputes) at runtime.
    meta = spec.fw_metadata
    saved = list(fwd_output_strides[meta.tensors_saved_for_backwards_slice])
    num_symints = spec.num_symints_saved_for_bw
    placeholders = [n for n in bw_gm.graph.nodes if n.op == "placeholder"]
    for index, node in enumerate(placeholders):
        val = node.meta.get("val")
        if not isinstance(val, torch.Tensor):
            continue
        offset = index - num_symints
        if not (0 <= offset < len(saved)) or not saved[offset]:
            continue
        # CompiledFxGraph.output_strides entries are PRINTED stride expressions.
        # A symbolic one is rebuilt as a SymInt over the placeholder's shape
        # symbols and kept symbolic, exactly as _aot_stage2b_bw_compile does:
        # resolving it to a hint would bake one call's sizes into the backward.
        fake_mode = getattr(val, "fake_mode", None)
        shape_env = fake_mode.shape_env if fake_mode is not None else None
        try:
            inductor_stride = tuple(
                int(s) if shape_env is None else shape_env.deserialize_symexpr(s)
                for s in saved[offset]
            )
        # deserialize_symexpr asserts on a symbol this shape env never created.
        except (AssertionError, NameError, TypeError, ValueError) as e:
            raise NotImplementedError(
                "aot_autograd.compile_to_python cannot restride the backward for "
                f"unevaluable forward output strides {saved[offset]!r}."
            ) from e
        if len(inductor_stride) != val.dim():
            raise NotImplementedError(
                "aot_autograd.compile_to_python got a rank-"
                f"{len(inductor_stride)} forward output stride for a rank-{val.dim()} "
                "backward placeholder; the saved-activation mapping does not "
                "line up, so restriding here would target the wrong input."
            )
        # Proof-only comparison: no guard is installed and a hint match is not
        # mistaken for equality (see _aot_stage2b_bw_compile).
        different = any(
            not statically_known_true(ph_stride == inductor_stride[k])
            for k, ph_stride in enumerate(val.stride())
        )
        if different:
            node.meta["val"] = val.as_strided(val.size(), inductor_stride)


def compile_to_python(
    gm: GraphModule,
    example_inputs: Sequence[Any],
    *,
    options: dict[str, Any] | None = None,
    grad_enabled: bool = False,
) -> tuple[str, bytes | None]:
    """Compile ``gm`` to ``(python_code, cache)``; see the module docstring.

    ``grad_enabled`` runs the AOTAutograd capture pass under ``enable_grad`` instead of the
    default ``no_grad``. Pass it for a TRAINING graph: when an input requires grad,
    AOTAutograd then emits a joint forward+backward, and the two dense graphs are composed
    into a module whose ``call`` returns outputs carrying ``grad_fn``, so the caller's
    ``.backward()`` runs the compiled backward. When no input requires grad the graph
    composes as inference exactly as it would under ``no_grad``. Leaving it off pins the
    inference path regardless of the inputs, which is what an artifact that will never be
    differentiated wants (``torch.compiler.precompile`` passes ``training``).

    THREADING: serialized by a process-global lock (``_COMPILE_LOCK``). The wrapper-source
    capture is thread-local, but the AOTAutograd pass and the inner inductor compile both
    swap process-global cache state (``CacheArtifactManager.with_fresh_cache()``); a
    concurrent compile on another thread would corrupt the captured wrappers or cache
    artifacts, so concurrent calls (including via ``torch.compiler.precompile``) are
    serialized rather than run in parallel.
    """
    import copy

    import torch
    from torch._higher_order_ops.effects import _get_effect
    from torch._inductor.compile_fx import compile_fx
    from torch._inductor.standalone_compile import (
        _compile_to_python_result,
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
        # the captured dense graph, via the ``_compile_to_python_result`` call (which
        # drives inductor's ``compile_fx_inner`` directly, not a re-entry into AOTAutograd).
        # The composer swaps the placeholder (the wrappers' inner reference) for the
        # inner inductor ``call`` by object identity, so the placeholder is only a
        # compile-time token and never runs.
        captured: list[GeneratedSource] = []
        specs: list[tuple[int | None, Any]] = []
        dense: dict[str, Any] = {}

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
            # Retain the placeholder's IDENTITY: it is the authoritative inner call the
            # runtime-wrapper chain closes over. The composer needs it to tell the inner
            # call apart from the orchestration's own outer closure (both surface as a
            # wrapper's inner-ref yet neither is a captured wrapper fn), which is what
            # separates INNER wrappers from the OUTER dedup / synthetic-base wrappers.
            placeholder = make_boxed_func(dense_gm.forward)
            dense["placeholder" if slot == "gm" else "bw_placeholder"] = placeholder
            return placeholder

        # Drive inductor's own ``compile_fx`` (i.e. its exact AOTAutograd invocation --
        # decomposition table + aot config) but swap in the capture inner compiler so the
        # dense graph is intercepted before codegen and no inductor compile happens. Using
        # compile_fx (rather than calling aot_autograd directly) guarantees the dense graph
        # matches what the step-2 inductor compile below expects. Pick the shapes mode from
        # the graph (there is no dynamic_shapes knob): a symbolically-traced graph uses
        # ``"from_graph"`` to stay dynamic, a static one ``"from_example_inputs"`` to
        # specialize -- matching what the composer can bake (symbolic view metadata is
        # rejected downstream). Grad mode decides the shape of the capture: no_grad pins
        # the inference path (one forward module), enable_grad lets inputs that require
        # grad produce a joint (a forward and a backward module).
        # Deepcopy first so compile_fx cannot mutate the caller's gm (torchbind
        # ProcessGroups smuggled through as shared references). Note: the raw-collective /
        # torchbind rewrites are inductor-lowering prereqs and belong to the step-2 inductor
        # compile, which applies them to the dense graph -- not duplicated here.
        shapes_mode = (
            "from_graph" if _graph_has_dynamic_shapes(gm) else "from_example_inputs"
        )
        with (
            torch.enable_grad() if grad_enabled else torch.no_grad(),
            _standalone_context(gm, shapes_mode, aot=False),
            capture_generated_sources(captured),
            _capture_autograd_specs(specs),
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
        if "gm" not in dense:
            raise RuntimeError(
                "aot_autograd.compile_to_python: AOTAutograd never reached the inner "
                "forward compiler, so no dense graph was captured."
            )

        # Lower the forward first, then restride the backward to the strides it
        # chose (torch.compile's report_output_strides coupling). A JOINT decides
        # what to compose; DIFFERENTIATING decides inductor's is_inference layout branch.
        has_joint = "bw" in dense
        differentiates = has_joint or _graph_differentiates(dense["gm"])
        fw = _compile_to_python_result(
            dense["gm"],
            example_inputs,
            options=options,
            is_inference=not differentiates,
        )
        if not has_joint:
            source = _compose_standalone_module(
                fw.inner_python, captured, dense["placeholder"]
            )
            return source, fw.cache

        spec = _select_training_spec(specs, captured)
        _restride_backward_placeholders(dense["bw"], fw.output_strides, spec)
        # is_backward=True matches torch.compile's backward compile: it gates
        # GraphLowering's backward-only require_contiguous safeguard for
        # untagged implicit-fallback aten ops (#140452), without which such a
        # fallback can silently miscompute on a non-contiguous input.
        bw = _compile_to_python_result(
            dense["bw"], [], options=options, is_inference=False, is_backward=True
        )
        source = _compose_training_module(
            fw.inner_python,
            bw.inner_python,
            captured,
            spec,
            dense["placeholder"],
            dense["bw_placeholder"],
        )
    return source, fw.cache


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
