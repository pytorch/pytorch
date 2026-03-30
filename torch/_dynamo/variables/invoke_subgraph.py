"""
This module contains the InvokeSubgraphHigherOrderVariable class and its
supporting helpers for subgraph reuse (auto-cache) in Dynamo's invoke_subgraph
higher-order operator.
"""

import enum
import logging
import traceback
from dataclasses import dataclass
from typing import Any, cast, NamedTuple, TYPE_CHECKING

import torch
import torch._higher_order_ops
from torch._dynamo import graph_break_hints
from torch._dynamo.exc import unimplemented
from torch._dynamo.guards import (
    extract_tensor_metadata,
    GUARD_VALUE_DISPATCH,
    GuardCheckSpec,
    SKIP_GUARD,
    UnsupportedGuardCheckSpec,
)
from torch._dynamo.source import SyntheticLocalSource
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.constant import ConstantVariable
from torch._dynamo.variables.functions import UserFunctionVariable
from torch._dynamo.variables.higher_order_ops import WrapHigherOrderVariable
from torch._dynamo.variables.lists import ListVariable, TupleVariable
from torch._dynamo.variables.nn_module import UnspecializedNNModuleVariable
from torch._dynamo.variables.tensor import SymNodeVariable, TensorVariable
from torch._guards import (
    Guard,
    InvokeSubgraphReuseCondition,
    InvokeSubgraphReuseEntry,
    Source,
)
from torch._higher_order_ops.invoke_subgraph import NestedCompileRegionOptions
from torch.fx.graph_module import GraphModule
from torch.fx.proxy import Proxy
from torch.utils import _pytree as pytree
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch._dynamo.symbolic_convert import InstructionTranslator
    from torch._dynamo.variables.higher_order_ops import SubgraphTracingInfo

log = logging.getLogger(__name__)
hc_log = torch._logging.getArtifactLogger(__name__, "hierarchical_compile")

# Note: [invoke_subgraph subgraph reuse]
#
# When mark_compile_region wraps a function called N times (e.g. 80 identical
# transformer layers), Dynamo traces the subgraph once and stamps out cached
# copies for subsequent calls. It does safety checks to ensure that a subgraph
# is reusable, if not (e.g. side-effect), it will fallback to tracing the
# next invocation.
#
# HIGH-LEVEL FLOW
# ===============
#   User code: model.layers[0](x), model.layers[1](x), ..., model.layers[79](x)
#                     |                     |                        |
#                     v                     v                        v
#              +--------------+     +--------------+        +--------------+
#              |  First Call  |     |  Second Call  |  ...   |  80th Call   |
#              +------+-------+     +------+-------+        +------+-------+
#                     |                    |                        |
#                     v                    v                        v
#              +--------------+     +--------------+        +--------------+
#              | Full subgraph|     | Cache lookup  |        | Cache lookup  |
#              |   trace      |     | (is_reusable) |        | (is_reusable) |
#              +------+-------+     +------+-------+        +------+-------+
#                     |                    |                        |
#                     v                    v                        v
#              +--------------+     +--------------+        +--------------+
#              | save_reuse_  |     | stamp_out_   |        | stamp_out_   |
#              | entry()      |     | subgraph()   |        | subgraph()   |
#              +--------------+     +--------------+        +--------------+
#
# WHAT GETS CACHED
# ================
# After the first trace, save_reuse_entry stores an InvokeSubgraphReuseEntry
# (in _guards.py) containing:
#   - body_name/body_gmod: the traced subgraph
#   - arg_sources: sources of the original call's arguments
#   - subgraph_input_mapping: how each lifted arg maps back to user inputs or captures
#   - output_metadata: shape/stride/dtype/device of outputs
#
# Paired with an InvokeSubgraphReuseCondition containing:
#   - input_checks: (tag, tensor_metadata) per input
#   - guards: (source, handler, expected, guard) tuples
#   - treespec: pytree structure of the args
#   - traced_sources: sources accessed during the trace
#
# CACHE LOOKUP (is_reusable)
# ==========================
# On subsequent calls:
#   1. Input structure match -- same treespec, tags, tensor metadata.
#   2. Source replacement -- clone each guard's source with a replacement map
#      (old: L['self'].layers[0].weight -> new: L['self'].layers[1].weight),
#      then evaluate against the new source's runtime value.
#   3. Mutation check -- reject if the subgraph mutated any captured var.
#
# A shared resolve_cache memoizes intermediate source resolution (e.g.
# L['self'].layers is evaluated once and reused across all guards).
#
# STAMP OUT (stamp_out_subgraph)
# ==============================
# On cache hit, reconstruct the argument list using the freevar mapping
# (list[LiftedArgOrigin]):
#
#   LiftedUserArg(index)
#       User arg (activation / explicit input).
#       Looked up from new call's flat proxies.
#
#   LiftedCapturedSource(source)
#       Sourceful captured var (weight, param, etc).
#       Source is cloned with replacement map, resolved via
#       VariableBuilder. Deduplicates via input_source_to_var.
#
#   LiftedSyntheticObject(ctor_fn, ctor_args, ctor_arg_sources)
#       Synthetic object (opaque type with SyntheticLocalSource).
#       Reconstructed via synthetic_graph_input with cached constructor info.
#
# SAFETY
# ======
# In normal Dynamo compilation, safety is enforced at runtime: guards are
# installed during tracing and re-evaluated on every subsequent call against
# real Python objects.  Subgraph reuse operates differently — we are in the
# middle of tracing, there are no real Python objects, only VariableTrackers.
# We must answer: what could cause the second invocation of a nested compile
# region to produce a different trace than the first?
#
# VariableTrackers fall into two categories:
#
# 1. Intermediates — values produced during tracing with no originating source
#    (e.g. the result of a prior FX op). These can reach a nested compile region
#    only via (a) the region's explicit function arguments, or (b) closure
#    capture. We do not support nested-function regions that close over tensors,
#    so only (a) applies. For explicit arguments, the set of types that can
#    appear is small and well-defined: TensorVariable, SymNodeVariable,
#    ConstantVariable, and NNModuleVariable. Each has a cheap structural
#    comparison (tensor metadata, symnode identity, constant value equality).
#    We also snapshot the pytree treespec of the argument list and verify it
#    matches on lookup, ensuring the flattened structure is identical.
#
# 2. Sourceful variables — values with a known originating source (e.g. a
#    module attribute or a local variable visible in the outer frame). For these
#    we collect the guard delta from the first trace, parameterize the guard
#    sources by replacing the original arg sources with the new arg sources, and
#    re-evaluate the guards by resolving each source against the live f_locals /
#    f_globals. The one extra hazard here is mutation: if the outer trace
#    mutates a sourceful object between the first and second invocations, the
#    cached guards would evaluate against stale values. We therefore also check
#    that none of the sources read by the cached subgraph have been mutated in
#    the outer SideEffects tracker before accepting a reuse.
#
# - max_reuse_entries (default 8, configurable via nested_compile_region arg)
#   caps cache entries per function. Exceeding it raises RuntimeError.
# - Guard failures logged with guard type + user stack trace.
#   Enable: TORCH_LOGS='+hierarchical_compile'
# ---------------------------------------------------------------------------
# Auto-cache helpers for invoke_subgraph
# ---------------------------------------------------------------------------


class InputTag(enum.Enum):
    TENSOR = "tensor"
    SYMNODE = "symnode"
    CONSTANT = "constant"
    MODULE = "module"


class InputFingerprint(NamedTuple):
    # (InputTag, VariableTracker) pairs for each leaf input.
    flat_vts: list[tuple[InputTag, VariableTracker]]
    # 1-1 mapping to flat_vts: source for each leaf, or None if the VT has no source.
    arg_sources: list[Source | None]
    # True if any leaf VT had an unsupported type for reuse.
    has_unknown: bool = False
    # TreeSpec from pytree.tree_flatten of the (args, kwargs) structure.
    treespec: pytree.TreeSpec | None = None


def classify_vt(vt: Any) -> InputTag | None:
    """Return the tag for a leaf VT, or None if unsupported."""
    if isinstance(vt, TensorVariable):
        return InputTag.TENSOR
    elif isinstance(vt, SymNodeVariable):
        return InputTag.SYMNODE
    elif isinstance(vt, ConstantVariable):
        return InputTag.CONSTANT
    elif isinstance(vt, UnspecializedNNModuleVariable):
        return InputTag.MODULE
    return None


def build_input_fingerprint(
    tx: "InstructionTranslator",
    fn_args_vt: Any,
    kwargs: dict[str, Any],
) -> InputFingerprint:
    """Build an InputFingerprint by flattening (args, kwargs) via pytree.

    Uses _make_inlined(tx, pytree.tree_flatten) to recursively flatten
    the argument structure into leaf VTs, classifying each leaf as
    tensor/symnode/constant/module. Also records the TreeSpec so that
    cache lookups can verify structural equivalence.

    Fast path: when kwargs is empty and all args are already leaf VTs
    (tensor/symnode/constant/module), skip the expensive pytree flatten.
    """
    # Fast path: flat args, no kwargs — skip pytree machinery.
    if not kwargs:
        all_leaf = True
        for vt in fn_args_vt:
            if classify_vt(vt) is None:
                all_leaf = False
                break
        if all_leaf:
            return build_fingerprint_fast(fn_args_vt)

    return build_fingerprint_with_pytree(tx, fn_args_vt, kwargs)


def build_fingerprint_fast(fn_args_vt: Any) -> InputFingerprint:
    """Build fingerprint for the common case of flat leaf args, no kwargs."""
    flat_vts: list[tuple[InputTag, VariableTracker]] = []
    arg_sources: list[Source | None] = []
    for vt in fn_args_vt:
        tag = classify_vt(vt)
        assert tag is not None
        flat_vts.append((tag, vt))
        # Always append (even None) to keep positional alignment with flat_vts
        # so that source_replacement zip pairing is correct across calls.
        arg_sources.append(getattr(vt, "source", None))
    return InputFingerprint(flat_vts, arg_sources)


def build_fingerprint_with_pytree(
    tx: "InstructionTranslator",
    fn_args_vt: Any,
    kwargs: dict[str, Any],
) -> InputFingerprint:
    """Build fingerprint via pytree flatten for nested/kwargs cases."""
    from torch._dynamo.variables.builder import SourcelessBuilder
    from torch._dynamo.variables.higher_order_ops import _make_inlined

    container_vt = SourcelessBuilder.create(tx, (list(fn_args_vt), kwargs))
    flat_list_vt, treespec_vt = _make_inlined(tx, pytree.tree_flatten)(
        container_vt
    ).unpack_var_sequence(tx)
    treespec = treespec_vt.as_python_constant()

    flat_vts: list[tuple[InputTag, VariableTracker]] = []
    arg_sources: list[Source | None] = []
    has_unknown = False

    for vt in flat_list_vt.unpack_var_sequence(tx):
        tag = classify_vt(vt)
        if tag is not None:
            flat_vts.append((tag, vt))
        else:
            has_unknown = True
            continue

        # Always append (even None) to keep positional alignment with flat_vts.
        arg_sources.append(getattr(vt, "source", None))

    return InputFingerprint(flat_vts, arg_sources, has_unknown, treespec)


def get_flat_proxies(fingerprint: InputFingerprint) -> list[Proxy]:
    """Collect deduplicated proxies from tensor/symnode leaves."""
    seen: set[torch.fx.Node] = set()
    flat_proxies: list[Proxy] = []
    for tag, vt in fingerprint.flat_vts:
        if tag in (InputTag.TENSOR, InputTag.SYMNODE):
            proxy = vt.as_proxy()
            if proxy.node not in seen:
                seen.add(proxy.node)
                flat_proxies.append(proxy)
    return flat_proxies


@dataclass
class LiftedUserArg:
    """Lifted arg that came from a user argument (intermediate activation or explicit input)."""

    index: int


@dataclass
class LiftedCapturedSource:
    """Lifted arg that is a captured variable (e.g. a weight or parameter) with a Source."""

    source: Any  # Source


@dataclass
class LiftedSyntheticObject:
    """Lifted arg that is a TorchScriptObject with a SyntheticLocalSource."""

    ctor_fn: Any  # Callable
    ctor_args: tuple[Any, ...]
    ctor_arg_sources: tuple[Any, ...] | None


@dataclass
class LiftedBoundSymbol:
    """Lifted arg that is a SymInt already bound as a graph input.

    SymInt graph inputs are created during tensor wrapping (not through
    VariableBuilder.wrap_symint), so they aren't registered in
    unspec_variable_map or variable_tracker_cache. Using LiftedCapturedSource
    for these would resolve the source to a concrete Python int via
    source.get_value() instead of reusing the existing symbolic proxy.
    """

    expr: Any  # sympy.Expr


LiftedArgOrigin = (
    LiftedUserArg | LiftedCapturedSource | LiftedSyntheticObject | LiftedBoundSymbol
)


def get_fn_id(fn_var: Any) -> int | None:
    if isinstance(fn_var, UserFunctionVariable):
        return id(fn_var.get_function())
    elif isinstance(fn_var, UnspecializedNNModuleVariable):
        return id(fn_var.value.forward.__func__)  # pyrefly: ignore[missing-attribute]
    return None


def has_mutated_vars(
    tx: "InstructionTranslator",
    traced_sources: OrderedSet[Source],
) -> bool:
    """Check if any source accessed by the subgraph has been mutated.

    SideEffects.mutated_sources records the exact AttrSource for every
    store_attr call. A simple set intersection with traced_sources tells
    us whether any source the subgraph read was later written to.
    """
    overlap = tx.output.side_effects.mutated_sources & traced_sources
    if overlap:
        hc_log.debug(
            "subgraph_reuse: mutated sources detected -- %s",
            overlap,
        )
        return True
    return False


def is_reuse_eligible(
    tx: "InstructionTranslator",
    body_r: Any,
    fingerprint: InputFingerprint,
    tracing_info: "SubgraphTracingInfo",
    traced_sources: OrderedSet[Source] | None = None,
) -> bool:
    """Best-effort check for whether a traced subgraph result can be reused.

    It is possible that a subgraph is morally reusable but does not fall
    into the limited support that Dynamo has today. Current limitations:
      - The subgraph must not have side effects.
      - No sourceful variable accessed by the subgraph may have been
        mutated, because guards are snapshotted on source values at trace
        time — if the underlying object changed since then, the cached
        guards would silently evaluate against stale values.
      - Output must be a single tensor, or a tuple/list of plain tensors.
      - All flattened inputs must be one of: tensor, symnode, constant,
        unspecialized NN module — for sourceless or other input types we
        rely on the treespec and tags for structural matching, so only
        types with well-defined comparison semantics are supported.
    """
    if tracing_info.side_effect_stack is not None:
        stack_msg = "\n" + "".join(
            traceback.format_list(tracing_info.side_effect_stack)
        )
        hc_log.debug(
            "subgraph_reuse: not eligible -- subgraph has side effects%s",
            stack_msg,
        )
        return False

    if traced_sources and has_mutated_vars(tx, traced_sources):
        return False

    if isinstance(body_r, TensorVariable):
        pass
    elif isinstance(body_r, (TupleVariable, ListVariable)):
        non_tensor = [
            type(item).__name__
            for item in body_r.items
            if not isinstance(item, TensorVariable)
        ]
        if non_tensor:
            hc_log.debug(
                "subgraph_reuse: not eligible -- output contains non-tensor types: %s",
                non_tensor,
            )
            return False
    else:
        hc_log.debug(
            "subgraph_reuse: not eligible -- output type %s is not tensor or tuple/list",
            type(body_r).__name__,
        )
        return False

    if fingerprint.has_unknown:
        hc_log.debug(
            "subgraph_reuse: not eligible -- unsupported input VT types",
        )
        return False

    return True


def build_reuse_condition(
    tx: "InstructionTranslator",
    fingerprint: InputFingerprint,
    traced_sources: OrderedSet[Source],
) -> InvokeSubgraphReuseCondition | None:
    """Build an InvokeSubgraphReuseCondition from a traced subgraph.

    A reuse condition is a mix of two kinds of checks:

    1. **Input tag checks** (from flat_vts): For each flattened leaf VT,
       we record its tag (_VtTag.TENSOR/SYMNODE/CONSTANT/MODULE) and
       metadata (e.g. tensor shape/stride/dtype/device/requires_grad).
       At lookup time, the treespec ensures structural equivalence, and
       then we compare tags and metadata leaf-by-leaf.

    2. **Guard checks** (from traced_sources): During the subgraph trace,
       every source accessed via VariableBuilder is recorded. We look up
       all guards installed on those sources (and on the arg_sources) to
       build the set of guards that must be re-evaluated on cache hit.
       This is more robust than guard diffing because it catches guards
       that were already installed before the subgraph trace began.

    Raise if any guard type is unsupported, as a feedback for compiler
    developers to support that guard type.
    """
    from torch._guards import InvokeSubgraphReuseCondition

    input_checks: list[tuple[InputTag, object]] = []
    for tag, vt in fingerprint.flat_vts:
        if tag == InputTag.TENSOR:
            assert isinstance(vt, TensorVariable)
            example = vt.proxy.node.meta.get("example_value", None)
            if example is None:
                hc_log.debug(
                    "subgraph_reuse: cannot build condition -- tensor input has no example_value"
                )
                return None
            input_checks.append((InputTag.TENSOR, extract_tensor_metadata(example)))
        elif tag == InputTag.SYMNODE:
            assert isinstance(vt, SymNodeVariable)
            # Store the SymInt/SymFloat/SymBool object itself. Two accesses to
            # the same symbolic dimension (e.g. x.shape[0] twice) produce the
            # same Python object, so identity comparison in is_reusable is
            # correct and avoids false matches between distinct symbols.
            input_checks.append((InputTag.SYMNODE, vt.sym_num))
        elif tag == InputTag.CONSTANT:
            assert isinstance(vt, ConstantVariable)
            input_checks.append((InputTag.CONSTANT, vt.value))
        elif tag == InputTag.MODULE:
            input_checks.append((InputTag.MODULE, None))
        else:
            raise AssertionError(
                f"Unexpected input tag '{tag}' for {type(vt).__name__} -- "
                f"is_reuse_eligible should have rejected this"
            )

    # Collect all guards for sources accessed during the subgraph trace
    # and for the flattened arg sources.
    all_sources = set(traced_sources)
    all_sources.update(s for s in fingerprint.arg_sources if s is not None)
    all_relevant_guards: set[Guard] = set()
    for source in all_sources:
        all_relevant_guards.update(tx.output.guards.get_guards_for_source(source))

    guard_tuples: list[tuple[Source, GuardCheckSpec, object, Guard]] = []
    for guard in all_relevant_guards:
        source = guard.originating_source
        type_str = guard.create_fn_name()
        handler = GUARD_VALUE_DISPATCH.get(type_str)

        if handler is SKIP_GUARD:
            continue

        if handler is None or isinstance(handler, UnsupportedGuardCheckSpec):
            raise RuntimeError(
                f"subgraph_reuse: unsupported guard type '{type_str}' on source '{source.name}'"
            )

        try:
            value = tx.output.resolve_source_value(source)
        except Exception:
            raise RuntimeError(
                f"subgraph_reuse: failed to resolve source '{source.name}' for {type_str} guard"
            ) from None

        # TODO(anijain2305): vLLM workaround -- skip CONSTANT_MATCH on
        # strings. Re-evaluate once vLLM migrates off this pattern.
        # if type_str == "CONSTANT_MATCH" and isinstance(value, str):
        #     continue

        handler = cast(GuardCheckSpec, handler)
        expected = handler.get_metadata_fn(guard, value)
        guard_tuples.append((source, handler, expected, guard))

    hc_log.debug("Number of guards %s", len(guard_tuples))

    return InvokeSubgraphReuseCondition(
        input_checks=input_checks,
        guards=guard_tuples,
        treespec=fingerprint.treespec,
        traced_sources=traced_sources,
    )


def build_source_replacement(
    old_arg_sources: list[Source | None],
    new_arg_sources: list[Source | None],
) -> dict[Source, Source]:
    """Map old arg sources to new arg sources for remapping captured variable sources."""
    return {
        old: new
        for old, new in zip(old_arg_sources, new_arg_sources)
        if old is not None and new is not None and old != new
    }


def is_reusable(
    tx: "InstructionTranslator",
    condition: "InvokeSubgraphReuseCondition",
    fingerprint: InputFingerprint,
    cached_entry: InvokeSubgraphReuseEntry,
) -> bool:
    """Check if a cached subgraph can be reused for the current call.

    Three-phase check:
    (1) Verify that intermediates (tensor metadata, symnode types, constant
        values) match the cached input_checks — these are lightweight
        structural comparisons that don't require source resolution.
    (2) Check for mutations on the remapped traced_sources — if any source
        the subgraph read has been mutated since the original trace, the
        cached guards would evaluate against stale values.
    (3) Build a source replacement mapping (old sources → new sources) and
        re-evaluate the snapshotted guards under the new sources.
    """
    # Structural check: treespec must match first.
    if condition.treespec is not None and fingerprint.treespec != condition.treespec:
        hc_log.debug(
            "subgraph_reuse: reuse failed -- treespec mismatch",
        )
        return False

    # Input count, tags, and metadata must match.
    # Tensor metadata (shape, stride, dtype, device, requires_grad) is checked
    # here because TENSOR_MATCH guards for subgraph inputs typically already
    # exist in the outer graph before tracing and thus won't appear in the
    # guard delta.
    if len(condition.input_checks) != len(fingerprint.flat_vts):
        hc_log.debug(
            "subgraph_reuse: reuse failed -- input count mismatch: cached %d vs current %d",
            len(condition.input_checks),
            len(fingerprint.flat_vts),
        )
        return False

    for i, ((cached_tag, cached_val), (cur_tag, cur_vt)) in enumerate(
        zip(condition.input_checks, fingerprint.flat_vts)
    ):
        if cached_tag != cur_tag:
            hc_log.debug(
                "subgraph_reuse: reuse failed -- input %d tag mismatch: cached '%s' vs current '%s'",
                i,
                cached_tag,
                cur_tag,
            )
            return False
        if cached_tag == InputTag.TENSOR:
            assert isinstance(cur_vt, TensorVariable)
            example = cur_vt.proxy.node.meta.get("example_value", None)
            if example is None:
                hc_log.debug(
                    "subgraph_reuse: reuse failed -- input %d tensor has no example_value",
                    i,
                )
                return False
            cur_meta = extract_tensor_metadata(example)
            if cur_meta != cached_val:
                hc_log.debug(
                    "subgraph_reuse: reuse failed -- input %d tensor metadata mismatch",
                    i,
                )
                return False
        elif cached_tag == InputTag.SYMNODE:
            assert isinstance(cur_vt, SymNodeVariable)
            if cur_vt.sym_num is not cached_val:
                return False
        elif cached_tag == InputTag.CONSTANT:
            assert isinstance(cur_vt, ConstantVariable)
            if cur_vt.value != cached_val:
                # If both the cached and current arg have sources, source
                # replacement in stamp_out will resolve the correct value.
                cached_src = (
                    cached_entry.arg_sources[i]
                    if i < len(cached_entry.arg_sources)
                    else None
                )
                new_src = (
                    fingerprint.arg_sources[i]
                    if i < len(fingerprint.arg_sources)
                    else None
                )
                if cached_src is None or new_src is None:
                    return False

    source_replacement = build_source_replacement(
        cached_entry.arg_sources, fingerprint.arg_sources
    )

    # Parameterized source - this function gives you new sources parameterized
    # on the arg_sources. For example, if the input to the nested compile region
    # is a nn Module layer with source `layers[0]`, then old source
    # `layers[0].weight` gets remapped to `layers[1].weight`. This
    # parameterization is central in getting the new sources and then running
    # guards on them.
    def replacement_fn(s: Source) -> Source:
        return source_replacement.get(s, s)

    # Check for mutations on remapped traced_sources.
    if source_replacement:
        remapped = OrderedSet(s.clone(replacement_fn) for s in condition.traced_sources)
    else:
        remapped = condition.traced_sources
    if has_mutated_vars(tx, remapped):
        return False

    # If no sources changed, all guards were already checked during the
    # original trace and will trivially pass again.
    if not source_replacement:
        return True

    # Shared resolution context so source.get_value memoizes intermediate
    # results (e.g. common base sources) across all guards in this check.
    resolve_globals: dict[str, Any] = {
        "G": tx.output.root_tx.f_globals,
        "L": tx.output.root_tx.f_locals,
    }
    resolve_locals: dict[str, Any] = {}
    resolve_cache: dict[Source, Any] = {}

    for source, handler, expected, guard in condition.guards:
        new_source = source.clone(replacement_fn)
        # Source unchanged after replacement — guard already passed during
        # the original trace, skip re-evaluation.
        if new_source == source:
            continue

        try:
            value = new_source.get_value(resolve_globals, resolve_locals, resolve_cache)
        except Exception:
            hc_log.debug(
                "subgraph_reuse: reuse failed -- cannot resolve source '%s' "
                "(guard type: %s, user stack:\n%s)",
                new_source.name,
                guard.create_fn_name(),
                "".join(guard.user_stack.format())
                if guard.user_stack
                else "<no stack>",
            )
            return False

        if not handler.eval_fn(value, expected):
            hc_log.debug(
                "subgraph_reuse: reuse failed -- guard on '%s': expected %s, got mismatch "
                "(guard type: %s, user stack:\n%s)",
                new_source.name,
                expected,
                guard.create_fn_name(),
                "".join(guard.user_stack.format())
                if guard.user_stack
                else "<no stack>",
            )
            return False

    return True


def has_reuse_entries(
    tx: "InstructionTranslator",
    fn_var: Any,
) -> bool:
    """Cheap check: does the cache have any entries for this function?"""
    from torch._guards import InvokeSubgraphCache

    invoke_subgraph_cache = tx.output.tracing_context.hop_dispatch_set_cache.get_cache(
        torch._higher_order_ops.invoke_subgraph
    )
    if not isinstance(invoke_subgraph_cache, InvokeSubgraphCache):
        return False
    fn_id = get_fn_id(fn_var)
    return fn_id is not None and fn_id in invoke_subgraph_cache.subgraph_reuse_cache


def find_reuse_match(
    tx: "InstructionTranslator",
    fn_var: Any,
    fingerprint: InputFingerprint,
) -> InvokeSubgraphReuseEntry | None:
    from torch._guards import InvokeSubgraphCache

    invoke_subgraph_cache = tx.output.tracing_context.hop_dispatch_set_cache.get_cache(
        torch._higher_order_ops.invoke_subgraph
    )
    if not isinstance(invoke_subgraph_cache, InvokeSubgraphCache):
        return None
    fn_id = get_fn_id(fn_var)
    if fn_id is None:
        return None

    # this evaluator function is called one by one for all the invoke subgraph
    # reuse entries - the one that evaluates to True is stamped out in the
    # graph.
    def evaluator(
        cond: "InvokeSubgraphReuseCondition", entry: InvokeSubgraphReuseEntry
    ) -> bool:
        return is_reusable(tx, cond, fingerprint, entry)

    return invoke_subgraph_cache.find_reuse_entry(fn_id, evaluator)


def save_reuse_entry(
    tx: "InstructionTranslator",
    fn_var: Any,
    fingerprint: InputFingerprint,
    body_name: str,
    body_gmod: torch.fx.GraphModule,
    config: NestedCompileRegionOptions | None,
    p_args: tuple[Any, ...],
    body_r: VariableTracker,
    example_value: Any,
    condition: "InvokeSubgraphReuseCondition",
    max_reuse_entries: int = 8,
) -> None:
    """Save a traced subgraph into the reuse cache for future cache hits.

    Builds an InvokeSubgraphReuseEntry with the freevar mapping (how each
    lifted arg maps back to user inputs or captured variables), output
    metadata, and arg sources. On a future cache hit, stamp_out_subgraph
    uses this entry to emit a new invoke_subgraph call without re-tracing.
    """
    from torch._guards import InvokeSubgraphCache

    invoke_subgraph_cache = tx.output.tracing_context.hop_dispatch_set_cache.get_cache(
        torch._higher_order_ops.invoke_subgraph
    )
    if not isinstance(invoke_subgraph_cache, InvokeSubgraphCache):
        return

    fn_id = get_fn_id(fn_var)
    if fn_id is None:
        return

    subgraph_input_mapping = build_subgraph_input_mapping(
        tx, p_args, fingerprint.flat_vts
    )
    single_tensor_output = isinstance(body_r, TensorVariable)

    # Count user-visible outputs from body_r. The graph may have additional
    # outputs from side-effect intermediates that stamp_out_subgraph must
    # not include when reconstructing the user-visible return value.
    user_output_vts: list[VariableTracker] = []
    VariableTracker.visit(
        lambda vt: user_output_vts.append(vt)
        if vt.is_tensor() or isinstance(vt, SymNodeVariable)
        else None,
        body_r,
    )
    num_user_outputs = len(user_output_vts)

    # Cache output tensor metadata so we can construct fresh FakeTensors on
    # cache hit without re-running the subgraph. This is safe because
    # invoke_subgraph does not support aliasing between inputs and outputs
    # (speculate_subgraph will fail if that happens).
    # example_value may contain SymInts (e.g. shape values for backward);
    # only record metadata for actual tensors.
    output_metadata = [
        (t.shape, t.stride(), t.dtype, t.device, t.requires_grad)
        for t in example_value
        if isinstance(t, torch.Tensor)
    ]

    entry = InvokeSubgraphReuseEntry(
        body_name=body_name,
        body_gmod=body_gmod,
        config=config,
        subgraph_input_mapping=subgraph_input_mapping,
        single_tensor_output=single_tensor_output,
        output_metadata=output_metadata,
        # Record arg sources so that on cache hit we can build a
        # source replacement mapping (old sources → new sources) to
        # rewrite captured variable sources for the current invocation.
        arg_sources=fingerprint.arg_sources,
        num_user_outputs=num_user_outputs,
    )
    invoke_subgraph_cache.add_reuse_entry(fn_id, condition, entry, max_reuse_entries)


def stamp_out_subgraph(
    tx: "InstructionTranslator",
    fingerprint: InputFingerprint,
    cached: InvokeSubgraphReuseEntry,
) -> VariableTracker:
    """Emit a new invoke_subgraph call by stamping out a cached subgraph.

    Sources in the cached entry are parameterized: they refer to the original
    call's sources and must be rewritten to the current call's sources via
    source replacement before we can look up or create the corresponding
    graph placeholders.
    """
    from torch._dynamo.variables.builder import VariableBuilder
    from torch._dynamo.variables.higher_order_ops import add_call_function, make_attr

    flat_proxies = get_flat_proxies(fingerprint)
    new_arg_sources = fingerprint.arg_sources

    source_replacement = build_source_replacement(cached.arg_sources, new_arg_sources)

    new_lifted_args = []
    # Shared resolution context so get_value memoizes intermediate results
    # (e.g. L['self'].layers) across all freevars in this stamp-out.
    resolve_globals: dict[str, Any] = {
        "G": tx.output.root_tx.f_globals,
        "L": tx.output.root_tx.f_locals,
    }
    resolve_locals: dict[str, Any] = {}
    resolve_cache: dict[Source, Any] = {}

    # Find the args for the about-to-be-inserted invoke_subgraph call.
    for subgraph_input in cached.subgraph_input_mapping:
        if isinstance(subgraph_input, LiftedUserArg):
            new_lifted_args.append(flat_proxies[subgraph_input.index])
        elif isinstance(subgraph_input, LiftedBoundSymbol):
            from torch._dynamo.output_graph import LazyProxy

            proxy = tx.output.current_tracer.bound_symbols[subgraph_input.expr]
            if isinstance(proxy, LazyProxy):
                proxy = proxy()
                tx.output.current_tracer.bound_symbols[subgraph_input.expr] = proxy
            new_lifted_args.append(proxy)
        elif isinstance(subgraph_input, LiftedSyntheticObject):
            ctor_args = subgraph_input.ctor_args
            ctor_arg_sources = subgraph_input.ctor_arg_sources
            if ctor_arg_sources and source_replacement:
                new_ctor_args = []
                new_ctor_arg_sources = []
                for val, arg_src in zip(ctor_args, ctor_arg_sources):
                    if arg_src is not None:
                        new_src = arg_src.clone(lambda s: source_replacement.get(s, s))
                        val = new_src.get_value(
                            resolve_globals, resolve_locals, resolve_cache
                        )
                        arg_src = new_src
                    new_ctor_args.append(val)
                    new_ctor_arg_sources.append(arg_src)
                ctor_args = tuple(new_ctor_args)
                ctor_arg_sources = tuple(new_ctor_arg_sources)
            vt = tx.output.synthetic_graph_input(
                subgraph_input.ctor_fn, ctor_args, ctor_arg_sources
            )
            new_lifted_args.append(vt.as_proxy())
        elif isinstance(subgraph_input, LiftedCapturedSource):
            new_source = subgraph_input.source
            if source_replacement:
                new_source = new_source.clone(lambda s: source_replacement.get(s, s))
            # VariableBuilder deduplicates via input_source_to_var,
            # so this reuses existing graph placeholders automatically.
            value = new_source.get_value(resolve_globals, resolve_locals, resolve_cache)
            vt = VariableBuilder(tx, new_source)(value)
            new_lifted_args.append(vt.as_proxy())

    # Generate fake tensor outputs
    assert tx.fake_mode is not None
    with tx.fake_mode:
        example_value = tuple(
            torch.empty_strided(
                shape,
                stride,
                dtype=dtype,
                device=device,
                requires_grad=req_grad,
            )
            for shape, stride, dtype, device, req_grad in cached.output_metadata
        )

    # Install the invoke_subgraph call
    body_node = make_attr(tx, cached.body_name)
    p_args = (body_node, cached.body_name, *new_lifted_args)
    flat_variable = add_call_function(
        tx,
        torch._higher_order_ops.invoke_subgraph,
        tuple(p_args),
        {},
        example_value,
        cached.config,
    )

    # Return only the user-visible outputs. The graph may have extra
    # intermediate outputs from side effects (allow_side_effects=True)
    # that should not be part of the user-facing return value.
    if cached.single_tensor_output:
        items = flat_variable.items  # pyrefly: ignore[missing-attribute]
        assert isinstance(items[0], TensorVariable), (
            f"Expected tensor output but got {type(items[0]).__name__}"
        )
        return items[0]

    items = flat_variable.items  # pyrefly: ignore[missing-attribute]
    n = cached.num_user_outputs
    if n > 0 and n < len(items):
        from .builder import SourcelessBuilder

        return SourcelessBuilder.create(tx, tuple(items[:n]))
    return flat_variable


def build_subgraph_input_mapping(
    tx: "InstructionTranslator",
    p_args: tuple[Any, ...],
    flat_vts: list[tuple[InputTag, VariableTracker]],
) -> list[LiftedArgOrigin]:
    """Build a mapping that records the origin of each lifted arg for a subgraph.

    On a cache hit, we stamp out a new invoke_subgraph call and need to
    reconstruct its argument list in the correct order. Each lifted arg
    (p_args[2:], skipping body_node and body_name) comes from one of:

    - LiftedUserArg: a user argument (intermediate activation or explicit input)
    - LiftedCapturedSource: a captured variable (e.g. a weight or parameter)
    - LiftedSyntheticObject: a TorchScriptObject with a SyntheticLocalSource
    - LiftedBoundSymbol: a SymInt already bound as a graph input
    """
    proxy_node_to_idx: dict[torch.fx.Node, int] = {}
    idx = 0
    for tag, vt in flat_vts:
        if tag in (InputTag.TENSOR, InputTag.SYMNODE):
            node = vt.as_proxy().node
            if node not in proxy_node_to_idx:
                proxy_node_to_idx[node] = idx
                idx += 1

    subgraph_input_mapping: list[LiftedArgOrigin] = []
    for outer_proxy in p_args[2:]:
        matched_idx = proxy_node_to_idx.get(outer_proxy.node, -1)
        if matched_idx >= 0:
            subgraph_input_mapping.append(LiftedUserArg(matched_idx))
        else:
            grapharg = outer_proxy.node.meta.get("grapharg", None)
            source = grapharg.source if grapharg is not None else None
            # SymInt freevars must reuse the existing symbolic proxy rather
            # than resolving via source.get_value() (which returns the
            # concrete int). They appear as either:
            # - placeholder nodes with grapharg.example being a SymInt
            # - call_function nodes (e.g. sym_size_int) with no grapharg
            # In both cases, store the sympy expression and look it up in
            # bound_symbols during stamp-out.
            example = (
                grapharg.example
                if grapharg is not None
                else outer_proxy.node.meta.get("example_value", None)
            )
            if isinstance(example, torch.SymInt):
                subgraph_input_mapping.append(LiftedBoundSymbol(example.node.expr))
                continue
            assert source is not None, (
                f"Freevar has no source: node.op={outer_proxy.node.op} "
                f"node.name={outer_proxy.node.name} -- this likely means a "
                f"function argument was not included in the proxy matching"
            )
            if isinstance(source, SyntheticLocalSource):
                ctor_info = tx.output.synthetic_source_ctor_info.get(source)
                if ctor_info is not None:
                    ctor_fn, ctor_args, ctor_arg_sources = ctor_info
                    subgraph_input_mapping.append(
                        LiftedSyntheticObject(ctor_fn, ctor_args, ctor_arg_sources)
                    )
                    continue
            subgraph_input_mapping.append(LiftedCapturedSource(source))
    return subgraph_input_mapping


class InvokeSubgraphHigherOrderVariable(WrapHigherOrderVariable):
    _HOP_NAME = "torch.ops.higher_order.invoke_subgraph"
    _ALLOW_FALLBACK_TO_EAGER = False
    supports_input_mutation = True
    supports_aliasing = False
    allow_side_effects = True
    # invoke_subgraph is NOT desugared in AOTAutograd, so the HOP input/output
    # shouldn't alias. For checkpoint HOP, we inline it so we don't need
    # alias analysis as functionalization would just work on the flat graph.
    filter_aliased_intermediates = True

    # pyrefly: ignore[bad-override]
    def install_subgraph_in_output_graph(
        self,
        tx: "InstructionTranslator",
        fn_vt: VariableTracker,
        fn_args_vt: "Sequence[VariableTracker]",
        kwargs: dict[str, VariableTracker],
        body_gmod: GraphModule,
        attr_name: str,
    ) -> str:
        # Check if the subgraph from speculate_subgraph (body_gmod) and the fake
        # inputs have already been seen before. If yes, the subgraph is already
        # installed in the output graph and we can just access the subgraph
        # using the saved attr name.

        if not isinstance(fn_vt, (UnspecializedNNModuleVariable, UserFunctionVariable)):
            unimplemented(
                gb_type="Encountered non user function variable during invoke_subgraph HOP tracing",
                context=str(fn_vt),
                explanation="invoke_subgraph does not support non user function variable",
                hints=[*graph_break_hints.SUPPORTABLE],
            )

        invoke_subgraph_cache = (
            tx.output.tracing_context.hop_dispatch_set_cache.get_cache(
                torch._higher_order_ops.invoke_subgraph
            )
        )

        if isinstance(fn_vt, UserFunctionVariable):
            fn_id = id(fn_vt.get_function())
            fn_name = fn_vt.get_function().__name__
        else:
            assert isinstance(fn_vt, UnspecializedNNModuleVariable)
            fn_id = id(fn_vt.value.forward.__func__)  # type: ignore[attr-defined]
            fn_name = fn_vt.value.forward.__name__  # type: ignore[attr-defined]
        # pyrefly: ignore [implicit-any]
        previously_installed_submodules = []
        if invoke_subgraph_cache:
            previously_installed_submodules = (
                invoke_subgraph_cache.get_dynamo_installed_submodules(fn_id)
            )
            current_mod = body_gmod
            # NB - reverse is more likely to cause a hit sooner because first
            # graph can have requires_grad=False for a few inputs
            for submodule_name in reversed(previously_installed_submodules):
                assert submodule_name in tx.output.nn_modules
                previous_mod = tx.output.nn_modules[submodule_name]
                assert tx.fake_mode
                from torch._dynamo.variables.higher_order_ops import (
                    are_same_graph_modules,
                )

                if are_same_graph_modules(
                    fn_name, previous_mod, current_mod, tx.fake_mode
                ):
                    return submodule_name

        body_name = super().install_subgraph_in_output_graph(
            tx, fn_vt, fn_args_vt, kwargs, body_gmod, "subgraph"
        )
        hc_log.debug(
            "%s: Installing subgraph with identifier '%s', bringing total count for '%s' function to %s",
            fn_name,
            body_name,
            fn_name,
            len(previously_installed_submodules) + 1,
        )
        if invoke_subgraph_cache:
            invoke_subgraph_cache.add_dynamo_installed_submodule(fn_id, body_name)

        return body_name

    def _call_function(
        self,
        tx: "InstructionTranslator",
        args: "Sequence[VariableTracker]",
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from torch._dynamo.utils import dynamo_timed
        from torch._dynamo.variables.higher_order_ops import (
            _call_function_with_auto_output_flattening,
        )

        fn_var = args[0]
        fn_args_vt = args[1:]

        config = None
        max_reuse_entries = 8
        if hasattr(fn_var, "get_function"):
            try:
                fn = fn_var.get_function()
                config = getattr(fn, "__marked_compile_region_config__", None)
                max_reuse_entries = getattr(
                    fn, "__marked_compile_region_max_reuse_entries__", 8
                )
            except Exception:
                log.warning(
                    "Failed to extract nested_compile_region() config from InvokeSubgraphHigherOrderVariable. ",
                    exc_info=True,
                )
                raise

        # TODO (anijain2305) - Collect issues why this does not work for export,
        # and enable if request arises.
        reuse = not tx.output.export

        # Reuse lookup: check fn_id first (cheap) to avoid the
        # expensive pytree flatten in build_input_fingerprint on the
        # first call when there's nothing in the cache yet.
        if reuse and has_reuse_entries(tx, fn_var):
            with dynamo_timed("invoke_subgraph_reuse_lookup"):
                fingerprint = build_input_fingerprint(tx, fn_args_vt, kwargs)
                match = find_reuse_match(
                    tx,
                    fn_var,
                    fingerprint,
                )
            if match is not None:
                hc_log.debug(
                    "subgraph_reuse: cache hit for '%s', reusing subgraph '%s'",
                    fn_var,
                    match.body_name,
                )
                with dynamo_timed("invoke_subgraph_reuse_stamp_out"):
                    return stamp_out_subgraph(tx, fingerprint, match)

        assert self._HOP_NAME is not None
        with dynamo_timed("invoke_subgraph_trace"):
            (
                p_args,
                p_kwargs,
                example_value,
                body_r,
                body_gmod,
                body_name,
                body_graph_output_vts,
                tracing_info,
            ) = self.create_wrapped_node(tx, fn_var, fn_args_vt, kwargs, self._HOP_NAME)

        if len(p_kwargs) > 0:
            unimplemented(
                gb_type="invoke_subgraph: kwargs unexpected",
                context=f"args: {args}, kwargs: {kwargs}",
                explanation="kwargs should have been flattened into lifted args.",
                hints=[
                    *graph_break_hints.DYNAMO_BUG,
                ],
            )

        # Store config in the body graph module meta
        if isinstance(config, NestedCompileRegionOptions):
            body_gmod.meta["nested_region_config"] = config

        p_args = (
            p_args[0],
            body_name,
            *p_args[1:],
        )

        # Subgraph reuse: save entry for future cache hits
        if reuse:
            fingerprint = build_input_fingerprint(tx, fn_args_vt, kwargs)
            traced_sources = tracing_info.traced_sources
            if is_reuse_eligible(tx, body_r, fingerprint, tracing_info, traced_sources):
                condition = build_reuse_condition(
                    tx,
                    fingerprint,
                    traced_sources,
                )
                if condition is not None:
                    save_reuse_entry(
                        tx,
                        fn_var,
                        fingerprint,
                        body_name,
                        body_gmod,
                        config,
                        p_args,
                        body_r,
                        example_value,
                        condition,
                        max_reuse_entries,
                    )

        return _call_function_with_auto_output_flattening(  # type: ignore[return-value]
            tx,
            torch._higher_order_ops.invoke_subgraph,
            tuple(p_args),
            p_kwargs,
            example_value,
            body_r,
            body_graph_output_vts,
            config=config,
        )
