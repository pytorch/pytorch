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
# To trigger the capture we run AOTAutograd ourselves, using grad mode to select the
# inference path or a joint forward/backward; see ``compile_to_python.grad_enabled``.
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
        id(rt.CUDARngStateHelper.get_torch_state_as_tuple): (
            f"{_RT} CUDARngStateHelper",
            "CUDARngStateHelper.get_torch_state_as_tuple",
        ),
        id(rt.CUDARngStateHelper.set_new_offset): (
            f"{_RT} CUDARngStateHelper",
            "CUDARngStateHelper.set_new_offset",
        ),
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
    if (
        getattr(obj, "__self__", None) is itertools.chain
        and getattr(obj, "__name__", None) == "from_iterable"
    ):
        imports.add("import itertools")
        return "itertools.chain.from_iterable"
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
    # runtime_wrappers.py (``def _runtime_wrapper(_compiled_fn_, _first_ctx_,
    # _on_before_call_, args)``). Verify the captured signature still matches so a future
    # rename/reorder fails loudly here instead of silently passing wrong arguments.
    expected_orch_params = ["_compiled_fn_", "_first_ctx_", "_on_before_call_", "args"]
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
        orch_params = None
    else:
        # Compare the FULL signature, not just positional params: the standalone call is
        # purely positional, so a keyword-only / *args / **kwargs param (e.g. an added
        # kw-only-with-default) would be silently dropped. Surface any such param so it
        # trips this guard rather than passing.
        orch_params = [a.arg for a in (*args_node.posonlyargs, *args_node.args)]
        orch_params += [a.arg for a in args_node.kwonlyargs]
        if args_node.vararg is not None:
            orch_params.append("*" + args_node.vararg.arg)
        if args_node.kwarg is not None:
            orch_params.append("**" + args_node.kwarg.arg)
    if orch_params != expected_orch_params:
        raise NotImplementedError(
            "aot_autograd.compile_to_python: the orchestration wrapper signature "
            f"changed (expected {expected_orch_params}, got {orch_params}); the "
            "standalone module invokes it positionally and must be updated to match."
        )

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


@dataclass(frozen=True)
class _StandaloneBackwardVariant:
    undefined_grad_out_mask: int | None
    python_code: str
    kept_arg_indices: tuple[int, ...] | None = None
    pruned_output_indices: tuple[int, ...] | None = ()
    skip_materialize_grad_output_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class _CompiledBackwardSpecialization:
    python_code: str
    call: Callable[[list[Any]], Any]
    cache: bytes | None
    kept_arg_indices: tuple[int, ...] | None = None
    pruned_output_indices: tuple[int, ...] | None = ()
    skip_materialize_grad_output_indices: tuple[int, ...] = ()


@dataclass
class _CompileToPythonState:
    """Capture-only state for specializing standalone backwards.

    ``install_capture`` connects the generated module's private compiler hook to
    ``compile_mask``. Bit ``i`` is set when grad output ``i`` is undefined. Masks are
    kept CANONICAL throughout -- restricted to the specializable user-output indices
    (``_specializable_user_grad_output_mask``), exactly like the live runtime path --
    so a non-differentiable output's always-undefined tangent cannot fork spurious
    variants or make the auto-covered mask 0 unreachable; every entry point
    re-canonicalizes defensively. The first
    real backward with a new mask compiles and runs that specialization immediately,
    then records its source and cache. After running its examples, the CALLER reads
    the observed masks out of ``call.__globals__["_AOT_OBSERVED_UNDEFINED_TANGENT_MASKS"]``
    and passes them to ``finalize``, which emits only those recorded variants.
    ``finalize`` does not touch the module globals: the caller is responsible for
    setting ``_AOT_BACKWARD_VARIANT_COMPILER`` back to ``None`` and clearing
    ``_AOT_BACKWARD_VARIANTS`` in that globals dict (see
    ``torch._precompile._DynamoPythonBackend.finalize_training``).
    """

    forward_python: str
    backward_python: str
    captured: list[GeneratedSource]
    spec: Any
    backward_graph: GraphModule
    backward_inputs: list[Any]
    options: dict[str, Any] | None
    forward_cache: bytes | None
    backward_cache: bytes | None
    forward_inner_call: Callable[..., Any]
    _capture_backward_call: Callable[[list[Any]], Any] | None = field(
        default=None, init=False
    )
    _specializations: dict[int, _CompiledBackwardSpecialization] = field(
        default_factory=dict, init=False
    )
    _observed_variants: dict[int, _StandaloneBackwardVariant] = field(
        default_factory=dict, init=False
    )

    def install_capture(self, globals_dict: dict[str, Any]) -> None:
        self._capture_backward_call = globals_dict["_inner_call_bw_0"]
        globals_dict["_AOT_BACKWARD_VARIANT_COMPILER"] = self.compile_mask

    def canonical_mask(self, mask: int) -> int:
        from . import runtime_wrappers as _rw

        return _rw._specializable_user_grad_output_mask(
            self.spec.fw_metadata, _rw._bitmask_to_indices(mask)
        )

    def compile_mask(
        self, mask: int
    ) -> tuple[
        Callable[[list[Any]], Any],
        tuple[int, ...] | None,
        tuple[int, ...] | None,
        tuple[int, ...],
    ]:
        with _COMPILE_LOCK:
            return self._compile_mask(mask)

    def _compile_mask(
        self, mask: int
    ) -> tuple[
        Callable[[list[Any]], Any],
        tuple[int, ...] | None,
        tuple[int, ...] | None,
        tuple[int, ...],
    ]:
        # Re-canonicalize defensively: the emitted backward records canonical
        # masks, but finalize can be handed masks observed on a superseded
        # sibling backend (a stateful automatic-dynamic recompile) whose
        # canonical form is computed against ITS metadata.
        mask = self.canonical_mask(mask)
        if mask in self._observed_variants:
            compiled = self._specializations[mask]
            return (
                compiled.call,
                compiled.kept_arg_indices,
                compiled.pruned_output_indices,
                compiled.skip_materialize_grad_output_indices,
            )

        import torch
        from torch._guards import compile_context, tracing
        from torch._inductor import (
            compile_to_python as _inductor_compile_to_python,
            load_from_python as _inductor_load_from_python,
        )

        from . import runtime_wrappers as _rw
        from .graph_compile import _retrace_backward_for_undefined_grad_outputs

        compiled = self._specializations.get(mask)
        if compiled is None:
            if not mask:
                call = self._capture_backward_call
                if call is None:
                    call = _inductor_load_from_python(
                        self.backward_python, self.backward_cache
                    )
                compiled = _CompiledBackwardSpecialization(
                    python_code=self.backward_python,
                    call=call,
                    cache=self.backward_cache,
                )
            else:
                specialization_indices = _rw._bitmask_to_indices(mask)
                result = None
                attempted_exact_retrace = False
                retrace_decline_reasons: list[str] = []
                lazy_info = self.spec.lazy_backward_info
                if (
                    isinstance(lazy_info, _rw.AutogradLazyBackwardCompileInfo)
                    and lazy_info.autograd_trace_info is not None
                ):
                    attempted_exact_retrace = True
                    from .graph_compile import retrace_backward_handling_errors

                    def _retrace() -> Any:
                        with (
                            tracing(lazy_info.saved_context),
                            compile_context(lazy_info.saved_compile_context),
                        ):
                            return _retrace_backward_for_undefined_grad_outputs(
                                lazy_info.autograd_trace_info,
                                self.spec.aot_config,
                                specialization_indices,
                                self.backward_graph,
                                self.backward_inputs,
                                decline_reason=retrace_decline_reasons,
                            )

                    result = retrace_backward_handling_errors(
                        _retrace, retrace_decline_reasons, specialization_indices
                    )
                if result is None:
                    # The retrace can decline for reasons unrelated to the ABI
                    # (e.g. the ambient autocast state differs between forward
                    # and backward, the standard AMP pattern); structural
                    # specialization preserves the saved-activation ABI and
                    # stays available, exactly as on the live runtime path.
                    result = _rw._specialize_bw_module_for_undefined_grad_outputs(
                        self.backward_graph,
                        self.backward_inputs,
                        self.spec.fw_metadata,
                        specialization_indices,
                        list(self.backward_inputs),
                    )
                if (
                    result is None
                    and attempted_exact_retrace
                    and not any(
                        isinstance(meta, _rw.SubclassCreationMeta)
                        for meta in self.spec.fw_metadata.subclass_tangent_meta
                    )
                ):
                    reasons = (
                        "; ".join(retrace_decline_reasons)
                        or "the backward retrace declined"
                    )
                    raise torch.compiler.PrecompileError(
                        "precompile could not compile an observed undefined-output "
                        f"tangent pattern: {reasons}; structural specialization "
                        "did not apply either."
                    )
                if result is None:
                    pruned = (
                        _rw._pruned_backward_output_indices_for_undefined_grad_outputs(
                            self.backward_graph,
                            self.spec.fw_metadata,
                            specialization_indices,
                        )
                    )
                    call = self._capture_backward_call
                    if call is None:
                        call = _inductor_load_from_python(
                            self.backward_python, self.backward_cache
                        )
                    compiled = _CompiledBackwardSpecialization(
                        python_code=self.backward_python,
                        call=call,
                        cache=self.backward_cache,
                        pruned_output_indices=pruned,
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
                    compiled = _CompiledBackwardSpecialization(
                        python_code=python_code,
                        call=_inductor_load_from_python(python_code, cache),
                        cache=cache,
                        kept_arg_indices=kept_arg_indices,
                        skip_materialize_grad_output_indices=specialization_indices,
                    )
            self._specializations[mask] = compiled

        self._observed_variants[mask] = _StandaloneBackwardVariant(
            undefined_grad_out_mask=mask,
            python_code=compiled.python_code,
            kept_arg_indices=compiled.kept_arg_indices,
            pruned_output_indices=compiled.pruned_output_indices,
            skip_materialize_grad_output_indices=(
                compiled.skip_materialize_grad_output_indices
            ),
        )
        return (
            compiled.call,
            compiled.kept_arg_indices,
            compiled.pruned_output_indices,
            compiled.skip_materialize_grad_output_indices,
        )

    def finalize(self, tangent_masks: Sequence[int]) -> tuple[str, bytes | None]:
        with _COMPILE_LOCK:
            return self._finalize(tangent_masks)

    def _finalize(self, tangent_masks: Sequence[int]) -> tuple[str, bytes | None]:
        from torch.compiler._cache import CacheArtifactManager

        canonical = {self.canonical_mask(mask) for mask in tangent_masks}
        masks = sorted(canonical) if canonical else [0]
        for mask in masks:
            if mask not in self._observed_variants:
                self._compile_mask(mask)

        variants = [self._observed_variants[mask] for mask in masks]
        caches = [self._specializations[mask].cache for mask in masks]

        return (
            _compose_training_module(
                self.forward_python,
                variants,
                self.captured,
                self.spec,
                self.backward_graph,
                self.forward_inner_call,
            ),
            CacheArtifactManager.merge((self.forward_cache, *caches)),
        )


def _compose_training_module(
    fw_python: str,
    backward_variants: Sequence[_StandaloneBackwardVariant],
    captured: list[GeneratedSource],
    spec: Any,
    bw_gm: GraphModule,
    forward_inner_call: Callable[..., Any],
) -> str:
    """Compose a FORWARD and BACKWARD lowering into one standalone module.

    The inference composer nests wrappers around a single ``call``. Training is a
    different shape: AOTAutograd's own bridge is a ``torch.autograd.Function``
    whose ``forward``/``backward`` bodies ARE codegen'd source
    (``compiled_function_forward`` / ``compiled_function_backward``, with
    ``backward_prologue`` / ``backward_epilogue`` beside them), but the class
    holding them is ordinary Python, so the composer emits it.

    Both inductor modules are spliced at module level and each one's entry is
    snapshotted immediately, because the second block rebinds ``call`` /
    ``Runner`` / the kernels that the first block's code resolves as late-bound
    globals.
    """

    from . import runtime_wrappers as _rw

    if spec.backward_state_indices:
        raise NotImplementedError(
            "aot_autograd.compile_to_python cannot compose a training graph that "
            "carries a BackwardState into standalone source yet."
        )

    orchestrations = [
        gen for gen in captured if gen.artifact_name == "runtime_wrapper_orchestration"
    ]
    if orchestrations:
        target_origin = orchestrations[-1].origin_id
        captured = [gen for gen in captured if gen.origin_id == target_origin]

    by_name: dict[str, list[GeneratedSource]] = {}
    for gen in captured:
        by_name.setdefault(gen.artifact_name, []).append(gen)
    missing = [
        name
        for name in (
            "backward_prologue",
            "backward_epilogue",
            "compiled_function_forward",
            "compiled_function_backward",
            "runtime_wrapper_orchestration",
        )
        if name not in by_name
    ]
    if missing:
        raise NotImplementedError(
            f"aot_autograd.compile_to_python: the training compose expects "
            f"AOTAutograd to codegen {missing}, which this graph did not produce."
        )

    imports: set[str] = set()
    helper_table = _known_helper_table()
    core_artifacts = {
        "backward_prologue",
        "backward_epilogue",
        "compiled_fn_wrapper",
        "compiled_function_forward",
        "compiled_function_backward",
        "runtime_wrapper_orchestration",
    }
    auxiliary = [gen for gen in captured if gen.artifact_name not in core_artifacts]
    fn_id_to_name = {id(gen.fn): gen.fn_name for gen in auxiliary}
    orchestration = by_name["runtime_wrapper_orchestration"]
    if len(orchestration) != 1:
        raise NotImplementedError(
            "aot_autograd.compile_to_python expected exactly one training "
            f"orchestration wrapper, captured {len(orchestration)}."
        )
    orch = orchestration[0]

    # Dedupe and synthetic-base wrappers sit outside AOTAutograd's orchestration.
    # Their innermost closure is not itself a captured function, so identify it and
    # rebuild the chain around a generated adapter for the autograd Function.
    inner_names = ("compiled_fn", "_compiled_fn_")

    def inner_ref(gen: GeneratedSource) -> Any:
        for name in inner_names:
            if name in gen.globals_dict:
                return gen.globals_dict[name]
        return None

    known_inner_ids = {id(forward_inner_call), *fn_id_to_name}
    orch_closure_ids = {
        id(ref)
        for gen in auxiliary
        if (ref := inner_ref(gen)) is not None and id(ref) not in known_inner_ids
    }
    if len(orch_closure_ids) > 1:
        raise NotImplementedError(
            "aot_autograd.compile_to_python captured multiple training wrappers "
            "whose inner callable could not be identified."
        )
    orch_closure_id = next(iter(orch_closure_ids), None)
    outer_wrappers: list[GeneratedSource] = []
    if orch_closure_id is not None:
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

    forward_call_name = "_inner_call_fw"
    if id(spec.compiled_fw_func) != id(forward_inner_call):
        forward_wrapper = fn_id_to_name.get(id(spec.compiled_fw_func))
        if forward_wrapper is None or id(spec.compiled_fw_func) in {
            id(gen.fn) for gen in outer_wrappers
        }:
            raise NotImplementedError(
                "aot_autograd.compile_to_python could not identify the captured "
                "training forward wrapper chain."
            )
        forward_call_name = forward_wrapper

    entry_name = "_autograd_orchestration_entry"
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
            if previous is not None and previous != expr:
                raise NotImplementedError(
                    "aot_autograd.compile_to_python cannot inline training wrappers "
                    f"that bind {name!r} to different values."
                )
            if previous is None:
                global_bindings[name] = expr
                hoists.append(f"{name} = {expr}")
        return "\n".join(hoists + [gen.source, ""])

    blocks = [splice(gen) for gen in inner_auxiliary]
    blocks += [
        splice(gen)
        for name in (
            "backward_prologue",
            "backward_epilogue",
            "compiled_fn_wrapper",
            "compiled_function_forward",
            "compiled_function_backward",
        )
        for gen in by_name.get(name, [])
    ]
    orchestration_block = splice(orch)
    outer_blocks = [splice(gen) for gen in outer_wrappers]

    fw_metadata_src = emit_value(spec.fw_metadata, imports)
    # Emit a constructor call, not the pickled object: the tracker's runtime
    # state belongs fresh in the artifact anyway, and its curr_fwd_iter
    # (itertools.count) default is unpicklable on Python >= 3.14.
    rng_src = (
        "_AutogradRngStateTracker("
        f"num_rng={emit_value(spec.fw_metadata.num_graphsafe_rng_states, imports)}, "
        f"graphsafe_idx={emit_value(spec.fw_metadata.graphsafe_rng_state_index, imports)}, "
        f"device={emit_value(spec.fw_metadata.graphsafe_rng_device, imports)})"
    )
    backward_output_dependencies = _rw._backward_output_tangent_dependencies(
        bw_gm, spec.fw_metadata
    )
    backward_output_dependencies_src = emit_value(backward_output_dependencies, imports)
    # The grad-output indices a backward variant may specialize on (surviving
    # user outputs). The emitted backward canonicalizes its scanned
    # undefined-tangent mask over these, exactly like the live runtime path
    # (_specializable_user_grad_output_mask): a non-differentiable output's
    # tangent is ALWAYS undefined, so keying the variant table on the raw mask
    # would make the auto-covered mask 0 unreachable and fail every backward.
    specializable_indices = _rw._specializable_user_grad_output_indices(
        spec.fw_metadata, range(spec.fw_metadata.num_forward_returns)
    )
    specializable_src = f"frozenset({tuple(sorted(specializable_indices))!r})"
    # Provably-zero bit per backward output, computed once with EVERY
    # specializable tangent pruned: a dependency-eligible output's cone
    # contains exactly the visible tangents its dependency entry lists, so one
    # pass answers "is output i zero once the tangents it depends on are
    # undefined" for all of them. The default-variant fallback in
    # _backward_impl masks only outputs that are BOTH dependency-implied AND in
    # this set -- dependency alone is wrong for a backward that is not linear
    # in its tangents (a custom Function returning g + 1 yields a nonzero grad
    # from an undefined tangent, which eager materializes zeros for). Mirrors
    # the runtime _pruned_backward_output_indices_for_undefined_grad_outputs.
    tangent_placeholders = [
        node
        for node in bw_gm.graph.find_nodes(op="placeholder")
        if _rw._is_grad_tangent(node)
    ]
    expected_tangents = sum(
        _rw._tangent_meta_arg_count(meta)
        for meta in spec.fw_metadata.subclass_tangent_meta
    )
    provably_zero_outputs: frozenset[int] = frozenset()
    if len(tangent_placeholders) == expected_tangents:
        flat_indices = _rw._undefined_tangent_flat_indices(
            spec.fw_metadata, specializable_indices
        )
        provably_zero_outputs = _rw._provably_zero_backward_output_indices(
            bw_gm, {tangent_placeholders[i] for i in flat_indices}
        )
    provably_zero_src = f"frozenset({tuple(sorted(provably_zero_outputs))!r})"
    unique_backward_sources: list[str] = []
    source_indices: dict[str, int] = {}
    variant_source_indices: list[int] = []
    for variant in backward_variants:
        source_index = source_indices.get(variant.python_code)
        if source_index is None:
            source_index = len(unique_backward_sources)
            source_indices[variant.python_code] = source_index
            unique_backward_sources.append(variant.python_code)
        variant_source_indices.append(source_index)
    namespaced_sources = namespace_module_names([fw_python, *unique_backward_sources])
    fw_ns = namespaced_sources[0]
    backward_blocks = namespaced_sources[1:]
    exact_variants: list[str] = []
    default_variant = "None"
    for variant, source_index in zip(
        backward_variants, variant_source_indices, strict=True
    ):
        call_name = f"_inner_call_bw_{source_index}"
        value = (
            f"({call_name}, {variant.kept_arg_indices!r}, "
            f"{variant.pruned_output_indices!r}, "
            f"{variant.skip_materialize_grad_output_indices!r})"
        )
        if variant.undefined_grad_out_mask is None:
            default_variant = value
        else:
            exact_variants.append(f"    {variant.undefined_grad_out_mask!r}: {value},")
    variants_src = "\n".join(["_AOT_BACKWARD_VARIANTS = {", *exact_variants, "}"])
    imports |= {
        "import contextlib",
        "import torch",
        "import weakref",
        "from torch._functorch._aot_autograd.runtime_wrappers import "
        "_AutogradRngStateTracker, _AutogradSavedState, "
        "_mask_pruned_backward_outputs, "
        "_pruned_backward_output_indices_from_dependencies, "
        "_snapshot_external_objects, index_to_external_object_weakref",
        "from torch._functorch._aot_autograd.standalone_runtime import "
        "normalize_as_list",
    }

    # The artifact bakes aot_autograd_prune_unused_outputs=True at emission
    # time instead of calling _set_grad_output_prototypes, which would re-read
    # the ambient config wherever the artifact happens to be loaded. Like the
    # runtime helper, prototypes are built even for a node with a single
    # differentiable output: a downstream custom Function whose backward
    # returns None hands that sole tangent in as undefined, and materializing
    # it needs a prototype.
    prototypes_expr = "_grad_output_prototypes(raw_returns, _fw_metadata)"
    imports.add(
        "from torch._functorch._aot_autograd.runtime_wrappers import "
        "_grad_output_prototypes"
    )

    glue = f"""
_fw_metadata = {fw_metadata_src}
# ViewAndMutationMeta.__post_init__ reads the ambient functionalize_rng_ops config.
# Restore the values captured when this graph was traced.
_fw_metadata.is_rng_op_functionalized = {spec.fw_metadata.is_rng_op_functionalized!r}
_fw_metadata.num_outputs_rng_offset = {spec.fw_metadata.num_outputs_rng_offset!r}
_fw_metadata.num_forward = {spec.fw_metadata.num_forward!r}
_saved_state = _AutogradSavedState(metadata=_fw_metadata)
_rng_state = {rng_src}
_BACKWARD_OUTPUT_DEPENDENCIES = {backward_output_dependencies_src}
_BACKWARD_OUTPUT_PROVABLY_ZERO = {provably_zero_src}
_AOT_SPECIALIZABLE_GRAD_OUT_INDICES = {specializable_src}
_AOT_OBSERVED_UNDEFINED_TANGENT_MASKS = set()
_AOT_BACKWARD_VARIANT_COMPILER = None
_NUM_FORWARD_RETURNS = {spec.fw_metadata.num_forward_returns}
_DISABLE_AMP = {spec.disable_amp!r}


def _finalize(ctx, fw_outs):
    raw_returns = list(fw_outs[:_NUM_FORWARD_RETURNS])
    # Prune-unused-outputs behavior is baked in at emission time; the ambient
    # aot_autograd_prune_unused_outputs config must not change the artifact.
    ctx._aot_prune_unused_outputs_enabled = True
    ctx.set_materialize_grads(False)
    protos = {prototypes_expr}
    ctx._aot_grad_output_prototypes, ctx._aot_grad_output_prototype_objects = protos
    ctx.mark_non_differentiable(*_transform_raw_returns(raw_returns))
    ctx._materialize_non_diff_grads = False
    _snapshot_external_objects(ctx)
    return tuple(raw_returns)


def _backward_impl(ctx, all_args):
    ctx.maybe_clear_saved_tensors()
    for idx, obj in getattr(ctx, "_external_objects", {{}}).items():
        index_to_external_object_weakref[idx] = weakref.ref(obj)
    inner_call_bw, kept_arg_indices, pruned, _ = ctx._aot_backward_variant
    if kept_arg_indices is not None:
        kept_args = [all_args[index] for index in kept_arg_indices]
        all_args.clear()
        all_args.extend(kept_args)
    amp = torch._C._DisableAutocast if _DISABLE_AMP else contextlib.nullcontext
    with amp():
        out = inner_call_bw(all_args)
    out = normalize_as_list(out)
    if pruned is None:
        # Dependency alone is not enough to null a grad: only outputs that are
        # ALSO provably zero with their tangents undefined may be masked (an
        # affine custom backward yields a nonzero grad from a zero tangent).
        pruned = tuple(
            index
            for index in _pruned_backward_output_indices_from_dependencies(
                _BACKWARD_OUTPUT_DEPENDENCIES,
                ctx._undefined_grad_out_indices,
            )
            if index in _BACKWARD_OUTPUT_PROVABLY_ZERO
        )
    return _mask_pruned_backward_outputs(out, pruned)


def _double_backward(ctx, impl_fn, all_args):
    class _DoubleBackward(torch.autograd.Function):
        @staticmethod
        def forward(double_ctx, *unused_args):
            return impl_fn(double_ctx)

        @staticmethod
        def backward(ctx, *args):
            raise RuntimeError(
                "torch.compile with aot_autograd does not currently support "
                "double backward"
            )

    if not any(t.requires_grad for t in all_args if isinstance(t, torch.Tensor)):
        all_args = [torch.empty(0, requires_grad=True)] + all_args
    return _DoubleBackward.apply(*all_args)


class _CompiledFunction(torch.autograd.Function):
    boxed_grads_call = True

    @staticmethod
    def forward(ctx, *deduped_flat_tensor_args):
        return _compiled_forward(
            ctx,
            deduped_flat_tensor_args,
            _rng_state.add_forward_args,
            _saved_state.save_from_forward,
            _finalize,
            {forward_call_name},
        )

    @staticmethod
    def backward(ctx, *flat_args):
        if len(flat_args) == 1 and isinstance(flat_args[0], list):
            ctx._undefined_grad_out_indices = tuple(
                index for index, grad in enumerate(flat_args[0]) if grad is None
            )
        else:
            ctx._undefined_grad_out_indices = ()
        # Canonical mask: only specializable (surviving user-output) indices
        # key the variant table -- a non-differentiable output's tangent is
        # always undefined and must not fork variants (mirrors the live
        # runtime's _specializable_user_grad_output_mask).
        mask = sum(
            1 << index
            for index in ctx._undefined_grad_out_indices
            if index in _AOT_SPECIALIZABLE_GRAD_OUT_INDICES
        )
        if _AOT_BACKWARD_VARIANT_COMPILER is not None:
            _AOT_OBSERVED_UNDEFINED_TANGENT_MASKS.add(mask)
        variant = _AOT_BACKWARD_VARIANTS.get(mask)
        if variant is None and _AOT_BACKWARD_VARIANT_COMPILER is not None:
            variant = _AOT_BACKWARD_VARIANT_COMPILER(mask)
            _AOT_BACKWARD_VARIANTS[mask] = variant
        if variant is None:
            variant = _AOT_DEFAULT_BACKWARD_VARIANT
        if variant is None:
            raise torch.compiler.PrecompileError(
                "precompile artifact encountered an undefined-output-tangent "
                f"bitmask that was not covered by example_inputs: {{mask:#b}}"
            )
        ctx._aot_backward_variant = variant
        ctx._aot_skip_materialize_grad_output_indices = variant[3]
        return _compiled_backward(
            flat_args,
            ctx,
            _backward_prologue,
            _rng_state.add_backward_args,
            _backward_impl,
            _backward_epilogue,
            _double_backward,
        )


def _boxed_autograd_apply(args):
    return _CompiledFunction.apply(*args)
"""

    if outer_wrappers:
        entry = f"""
def {entry_name}(args):
    return {orch.fn_name}(
        _boxed_autograd_apply, contextlib.nullcontext, lambda: None, args
    )
"""
        outermost_name = fn_id_to_name[id(outer_wrappers[-1].fn)]
        public_call = f"""
def call(flat_inputs):  # noqa: F811
    return {outermost_name}(list(flat_inputs))
"""
    else:
        entry = ""
        public_call = f"""
def call(flat_inputs):  # noqa: F811
    return {orch.fn_name}(
        _boxed_autograd_apply, contextlib.nullcontext, lambda: None, list(flat_inputs)
    )
"""

    parts = [
        "# Generated by aot_autograd.compile_to_python (training) -- do not edit.",
        "",
        "import torch._dynamo",
        *sorted(imports),
        "",
        "# === Inner Inductor output code: FORWARD ===",
        fw_ns,
        "_inner_call_fw = call_s0",
        "",
        *[
            block
            for index, source in enumerate(backward_blocks, 1)
            for block in (
                f"# === Inner Inductor output code: BACKWARD variant {index - 1} ===",
                source,
                f"_inner_call_bw_{index - 1} = call_s{index}",
                "",
            )
        ],
        variants_src,
        f"_AOT_DEFAULT_BACKWARD_VARIANT = {default_variant}",
        "",
        "# === AOTAutograd runtime wrappers ===",
        *blocks,
        orchestration_block,
        glue,
        entry,
        *outer_blocks,
        public_call,
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
    ``_inductor_compile_to_python`` now reports.

    The rewrite lands on ``node.meta["val"]``, NOT on an example-inputs list:
    ``inductor.compile_to_python`` rebuilds its fakes from the placeholders'
    metadata and ignores the inputs it is handed, so restriding a list is a
    silent no-op through that entry point.
    """
    import torch
    from torch._inductor.utils import shape_env_from_inputs
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


def _graph_differentiates(gm: GraphModule) -> bool:
    """Whether a dense graph contains a backward computation inline.

    Scoped to call nodes and to the callable target's operator name, matched
    against aten's ``*_backward`` naming convention (e.g. ``convolution_backward``,
    ``threshold_backward``). Matching the suffix rather than any occurrence of
    "backward" avoids a spurious hit on an op whose name merely contains the
    substring, and avoids stringifying placeholders/get_attrs entirely.
    """
    import torch

    for node in gm.graph.nodes:
        if node.op not in ("call_function", "call_method"):
            continue
        target = node.target
        if isinstance(target, torch._ops.OpOverload):
            name = target.overloadpacket.__name__
        else:
            name = getattr(target, "__name__", None) or str(target)
        if name.endswith("_backward"):
            return True
    return False


def _compile_to_python_with_state(
    gm: GraphModule,
    example_inputs: Sequence[Any],
    *,
    options: dict[str, Any] | None = None,
    grad_enabled: bool = False,
) -> tuple[str, bytes | None, _CompileToPythonState | None]:
    """Compile ``gm`` to ``(python_code, cache)``; see the module docstring.

    ``grad_enabled`` runs the AOTAutograd capture pass under ``enable_grad`` instead of the
    default ``no_grad``, and covers two different graphs. One performs autograd INTERNALLY
    -- a Dynamo graph captured with ``trace_autograd_ops``, whose traced
    ``torch.autograd.grad`` call needs a live autograd graph to differentiate and otherwise
    fails the capture pass with "element 0 of tensors does not require grad"; that is still
    an INFERENCE graph at the AOT boundary, since its backward lives inside the traced call.
    The other has INPUTS that require grad, and AOTAutograd then emits a joint
    forward+backward: two dense graphs, composed into a module whose ``call`` returns
    outputs carrying ``grad_fn``, so the caller's ``.backward()`` runs the compiled
    backward. Leaving it off pins the inference path, which is what you want for a forward
    you will never differentiate.

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
        # the captured dense graph, via the ``_inductor_compile_to_python`` call (which
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
        if "gm" not in dense:
            raise RuntimeError(
                "aot_autograd.compile_to_python: AOTAutograd never reached the inner "
                "forward compiler, so no dense graph was captured."
            )

        # Lower the FORWARD first and take the strides inductor actually chose,
        # then restride the backward's placeholders to match before lowering it.
        # This is the fw->bw coupling torch.compile gets from
        # TracingContext.report_output_strides (graph_compile.py:2841) and feeds
        # to _aot_stage2b_bw_compile; a capture pass that never lowers cannot
        # observe it, and two independently-lowered graphs then disagree about
        # layout -- loudly on a conv net, silently if size asserts are off.
        has_joint = "bw" in dense
        differentiates = has_joint or _graph_differentiates(dense["gm"])
        fwd_output_strides: list[tuple[str, ...] | None] = []
        inner_python, forward_cache = _inductor_compile_to_python(
            dense["gm"],
            example_inputs,
            options=options,
            is_inference=not differentiates,
            output_strides=fwd_output_strides,
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
        from torch.compiler._cache import CacheArtifactManager

        cache = CacheArtifactManager.merge((forward_cache, backward_cache))
        source = _compose_training_module(
            inner_python,
            [
                _StandaloneBackwardVariant(
                    undefined_grad_out_mask=None,
                    python_code=bw_python,
                    pruned_output_indices=None,
                )
            ],
            captured,
            spec,
            dense["bw"],
            dense["placeholder"],
        )
        state = _CompileToPythonState(
            forward_python=inner_python,
            backward_python=bw_python,
            captured=captured,
            spec=spec,
            backward_graph=dense["bw"],
            backward_inputs=dense["bw_inputs"],
            options=options,
            forward_cache=forward_cache,
            backward_cache=backward_cache,
            forward_inner_call=dense["placeholder"],
        )
    return source, cache, state


def compile_to_python(
    gm: GraphModule,
    example_inputs: Sequence[Any],
    *,
    options: dict[str, Any] | None = None,
    grad_enabled: bool = False,
) -> tuple[str, bytes | None]:
    """Compile ``gm`` to standalone Python source and an optional cache bundle.

    See the module documentation for the generated module's calling convention.
    ``grad_enabled=True`` includes AOTAutograd's differentiable forward and backward.
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
