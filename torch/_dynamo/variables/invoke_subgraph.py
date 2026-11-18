"""
This module contains the InvokeSubgraphHigherOrderVariable class and its
supporting helpers for subgraph reuse (auto-cache) in Dynamo's invoke_subgraph
higher-order operator.
"""

import collections
import enum
import logging
import traceback
import types
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Any, cast, NamedTuple, TYPE_CHECKING, TypeGuard

import sympy

import torch
import torch._higher_order_ops
from torch._dynamo import config as dynamo_config, graph_break_hints
from torch._dynamo.exc import unimplemented
from torch._dynamo.guards import (
    _receiver_alias_pattern,
    extract_tensor_metadata,
    GUARD_VALUE_DISPATCH,
    GuardBuilder,
    GuardCheckSpec,
    install_guard,
    record_lexical_alias,
    SKIP_GUARD,
    UnsupportedGuardCheckSpec,
)
from torch._dynamo.source import (
    ConstDictKeySource,
    is_from_source,
    SyntheticLocalSource,
    TensorPropertySource,
)
from torch._dynamo.utils import (
    _make_inlined,
    constants_identical,
    tensor_always_has_static_shape,
    unpack_iterable,
)
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.constant import ConstantVariable
from torch._dynamo.variables.functions import UserFunctionVariable
from torch._dynamo.variables.higher_order_ops import WrapHigherOrderVariable
from torch._dynamo.variables.lists import ListVariable, TupleVariable
from torch._dynamo.variables.nn_module import UnspecializedNNModuleVariable
from torch._dynamo.variables.tensor import SymNodeVariable, TensorVariable
from torch._dynamo.variables.user_defined import UserDefinedObjectVariable
from torch._guards import (
    ChainedSource,
    Guard,
    InvokeSubgraphReuseCondition,
    InvokeSubgraphReuseEntry,
    Source,
)
from torch._higher_order_ops.invoke_subgraph import NestedCompileRegionOptions
from torch.fx.experimental import symbolic_shapes
from torch.fx.graph_module import GraphModule
from torch.fx.proxy import Proxy
from torch.utils import _pytree as pytree
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from torch._dynamo.output_graph import SizesStridesInfo
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase
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
#   - guards: metadata checks and source dependencies keyed by the original guard
#   - treespec: pytree structure of the args
#   - traced_sources: sources accessed during the trace
#
# CACHE LOOKUP (is_reusable)
# ==========================
# On subsequent calls:
#   1. Input structure match -- same treespec, tags, tensor metadata.
#   2. Reject mutations of remapped sources read by the cached trace.
#   3. Source replacement -- clone each guard's source with a replacement map
#      (old: L['self'].layers[0].weight -> new: L['self'].layers[1].weight),
#      then evaluate against the new source's runtime value.
#   4. Validate aliases, remapped shapes and capture metadata before replaying guards.
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
#   LiftedBoundSymbol: reuse the explicit binding or rebuild the captured property.
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
#    so only (a) applies. For explicit arguments, the set of types we support is
#    small and well-defined: TensorVariable, SymNodeVariable, and
#    ConstantVariable (enum members included). Each has a cheap structural
#    comparison (tensor metadata, symnode identity, constant value equality).
#    We also snapshot the pytree treespec of the argument list and verify it
#    matches on lookup, ensuring the flattened structure is identical.
#    A sourceless object -- an nn.Module built during tracing, say -- has
#    neither a structural comparison nor guards, so it is not eligible at all.
#
# 2. Sourceful variables — values with a known originating source (e.g. a
#    module attribute or a local variable visible in the outer frame). For these
#    we collect guards for dependencies from successful child traces, parameterize the guard
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
    OBJECT = "object"


class InputFingerprint(NamedTuple):
    # (InputTag, VariableTracker) pairs for each leaf input.
    flat_vts: list[tuple[InputTag, VariableTracker]]
    # 1-1 mapping to flat_vts: source for each leaf, or None if the VT has no source.
    arg_sources: list[Source | None]
    # True if any leaf VT had an unsupported type for reuse.
    has_unknown: bool = False
    # TreeSpec from pytree.tree_flatten of the (args, kwargs) structure.
    treespec: pytree.TreeSpec | None = None


def is_constant_like(
    vt: Any,
) -> TypeGuard[ConstantVariable | UserDefinedObjectVariable]:
    """Whether a leaf VT is compared by value, via ``vt.value``.

    Enum members are UserDefinedObjectVariable (not ConstantVariable) in
    Dynamo, but they are immutable singletons, so value comparison is as
    sound as it is for ConstantVariable.
    """
    if isinstance(vt, ConstantVariable):
        return True
    return isinstance(vt, UserDefinedObjectVariable) and isinstance(vt.value, enum.Enum)


def classify_vt(vt: Any) -> InputTag | None:
    """Return the tag for a leaf VT, or None if unsupported."""
    if isinstance(vt, TensorVariable):
        return InputTag.TENSOR
    elif isinstance(vt, SymNodeVariable):
        return InputTag.SYMNODE
    elif is_constant_like(vt):
        return InputTag.CONSTANT
    elif isinstance(vt, UserDefinedObjectVariable) and vt.source is not None:
        # Covers nn.Modules too -- UnspecializedNNModuleVariable is a
        # UserDefinedObjectVariable subclass. No metadata is recorded; reuse
        # safety comes entirely from re-evaluating the guards installed on this
        # object's source and on the sources derived from it. A sourceless
        # object has no guards to re-evaluate, so it stays unsupported.
        return InputTag.OBJECT
    return None


def build_input_fingerprint(
    tx: "InstructionTranslatorBase",
    fn_args_vt: Any,
    kwargs: dict[str, Any],
) -> InputFingerprint:
    """Build an InputFingerprint by flattening (args, kwargs) via pytree.

    Flattens the argument structure into leaf VTs, classifying each leaf as
    tensor/symnode/constant/module. Also records the TreeSpec so that
    cache lookups can verify structural equivalence.

    Fast path: when kwargs is empty and all args are already leaf VTs
    (tensor/symnode/constant/module), skip the pytree flatten entirely.
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
        if tag is None:
            raise AssertionError(
                f"classify_vt returned None for {type(vt).__name__} in fast path"
            )
        flat_vts.append((tag, vt))
        # Always append (even None) to keep positional alignment with flat_vts
        # so that source_replacement zip pairing is correct across calls.
        arg_sources.append(getattr(vt, "source", None))
    return InputFingerprint(flat_vts, arg_sources)


def build_fingerprint_with_pytree(
    tx: "InstructionTranslatorBase",
    fn_args_vt: Any,
    kwargs: dict[str, Any],
) -> InputFingerprint:
    """Build fingerprint via pytree flatten for nested/kwargs cases.

    Recurses over the pytree structure natively (untraced), inlining/tracing
    only each container node's own ``flatten_fn`` rather than the full
    recursive tree_flatten dispatch around it. This is safe because node-type
    classification only depends on ``type()``, never on tensor values.

    Note: skipping the traced registry dispatch also means we no longer
    install guards on the pytree registry itself, for the plain-function
    flatten_fn case (the non-FunctionType fallback below still traces the
    full tree_flatten and so still installs them). That's fine either way:
    this fingerprint is built fresh from the live registry on every
    reuse-lookup (register_pytree_node isn't traceable, so the registry
    can't change mid-trace), and reuse is separately gated on treespec/tag
    equality. So a registry change between compiles can only make a cached
    subgraph ineligible for reuse (has_unknown / treespec mismatch), never
    silently wrong.
    """
    from torch._dynamo.variables.builder import SourcelessBuilder

    flat_vts: list[tuple[InputTag, VariableTracker]] = []
    arg_sources: list[Source | None] = []
    has_unknown = False

    def add_leaf(vt: VariableTracker) -> None:
        nonlocal has_unknown
        tag = classify_vt(vt)
        if tag is None:
            has_unknown = True
        else:
            flat_vts.append((tag, vt))
            # Always append (even None) to keep positional alignment with flat_vts.
            arg_sources.append(getattr(vt, "source", None))

    def flatten(node_vt: VariableTracker) -> pytree.TreeSpec:
        nonlocal has_unknown
        try:
            node_type = node_vt.python_type()
        except NotImplementedError:
            has_unknown = True
            return pytree.treespec_leaf()
        # Keep in sync with pytree._get_node_type.
        if pytree.is_namedtuple_class(node_type):
            node_type = collections.namedtuple

        if node_type not in pytree.SUPPORTED_NODES:
            add_leaf(node_vt)
            return pytree.treespec_leaf()

        flatten_fn = pytree.SUPPORTED_NODES[node_type].flatten_fn
        if not isinstance(flatten_fn, types.FunctionType):
            # _make_inlined only supports plain Python functions (it always
            # wraps its argument in a UserFunctionVariable). A flatten_fn
            # registered as e.g. a functools.partial, bound method, or
            # callable object can't go through it directly. Fall back to
            # tracing the full recursive tree_flatten for this subtree, which
            # dispatches calls generically and so handles any callable.
            leaves_vt, treespec_vt = unpack_iterable(
                tx, _make_inlined(tx, pytree.tree_flatten)(node_vt)
            )
            for leaf_vt in unpack_iterable(tx, leaves_vt):
                add_leaf(leaf_vt)
            return treespec_vt.as_python_constant()

        children_vt, context_vt = unpack_iterable(
            tx, _make_inlined(tx, flatten_fn)(node_vt)
        )
        context = context_vt.as_python_constant()
        child_specs = [flatten(child) for child in unpack_iterable(tx, children_vt)]
        return pytree.TreeSpec(node_type, context, child_specs)

    container_vt = SourcelessBuilder.create(tx, (list(fn_args_vt), kwargs))
    treespec = flatten(container_vt)

    return InputFingerprint(flat_vts, arg_sources, has_unknown, treespec)


def sym_num_key(sym_num: Any) -> Any:
    """Key for matching a symbolic input against a cached one.

    Compares the symbolic expression rather than the SymInt object. Each tensor
    holds its own SymInt objects, so two arguments carrying the same symbol
    (e.g. the atom count threaded through successive layers) are distinct
    objects. Equal expressions mean the same value, which is what reuse needs;
    distinct symbols still have distinct expressions.

    ``expr`` rather than ``_expr`` on purpose: it has the ShapeEnv's
    replacements applied, so a symbol that was specialized keys on the value it
    was specialized to.
    """
    return sym_num.node.expr


def input_aliases(
    tx: "InstructionTranslatorBase",
    fingerprint: InputFingerprint,
    tensors: Iterable[Any],
) -> tuple[int, ...]:
    from .builder import lookup_tracked_variable

    values = []
    for tag, vt in fingerprint.flat_vts:
        if tag in (InputTag.TENSOR, InputTag.SYMNODE):
            node = vt.as_proxy().node
            values.append(node)  # Stamp-out deduplicates placeholder operands by node.
            if tag is InputTag.TENSOR:
                values.append(node.meta["example_value"])
        elif tag is InputTag.OBJECT:
            values.append(vt.value)
    for tensor in tensors:
        tracked = lookup_tracked_variable(tx, tensor)
        # A metadata-only capture can have a guard without a fake or a placeholder.
        if isinstance(tracked, TensorVariable):
            tensor = tracked.proxy.node.meta["example_value"]
        values.append(tensor)
    return _receiver_alias_pattern(tuple(values))


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
    The source, when read by the region, rebinds the symbol at the next call site.
    """

    expr: Any  # sympy.Expr
    source: Source | None = None


LiftedArgOrigin = (
    LiftedUserArg | LiftedCapturedSource | LiftedSyntheticObject | LiftedBoundSymbol
)


def get_fn_key(
    fn_var: Any, tx: "InstructionTranslatorBase | None" = None
) -> types.FunctionType | types.CodeType | None:
    if isinstance(fn_var, UserFunctionVariable):
        fn = fn_var.get_function()
    elif isinstance(fn_var, UnspecializedNNModuleVariable):
        fn = fn_var.value.forward.__func__  # pyrefly: ignore[missing-attribute]
    else:
        return None
    state = fn.__closure__ or fn.__defaults__ or fn.__kwdefaults__
    if tx is not None and (state or fn.__globals__ is not tx.output.root_tx.f_globals):
        return fn
    return fn.__code__


def has_mutated_vars(
    tx: "InstructionTranslatorBase",
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
    tx: "InstructionTranslatorBase",
    body_r: Any,
    fingerprint: InputFingerprint,
    tracing_info: "SubgraphTracingInfo",
    traced_sources: OrderedSet[Source] | None = None,
    has_reuse_hash_fn: bool = False,
) -> bool:
    """Best-effort check for whether a traced subgraph result can be reused.

    It is possible that a subgraph is morally reusable but does not fall
    into the limited support that Dynamo has today. Current limitations:
      - Without a reuse hash, no side effects or builtin-method identity dependencies.
      - No sourceful variable accessed by the subgraph may have been
        mutated, because guards are snapshotted on source values at trace
        time — if the underlying object changed since then, the cached
        guards would silently evaluate against stale values.
      - Output must be a single tensor, or a tuple/list of plain tensors.
      - All flattened inputs must be one of: tensor, symnode, constant
        (including enum members), or a sourceful user-defined object
        (nn.Modules included) — for sourceless or other input types we
        rely on the treespec and tags for structural matching, so only
        types with well-defined comparison semantics are supported.

    When ``has_reuse_hash_fn`` is True, side-effect and mutation checks are
    skipped because the hash key replaces guards — there are no guards to
    go stale from mutations.
    The caller must include these dependencies in the key.
    """
    if not has_reuse_hash_fn:
        reads = tracing_info.lexical_sources
        for _, vt in fingerprint.flat_vts:
            record_lexical_alias(reads, vt.source, vt.unwrap().source)
        if any(
            is_from_source(vt.source, read)
            for tag, vt in fingerprint.flat_vts
            if tag is not InputTag.CONSTANT and vt.source is not None
            for read, _ in reads
        ):
            return _reject_reuse("argument aliases a lexical read")
        if tracing_info.has_builtin_method_identity:
            # Stamp-out cannot replay remapped receiver-alias guards.
            hc_log.debug("subgraph_reuse: not eligible -- builtin-method identity")
            return False
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
    tx: "InstructionTranslatorBase",
    fingerprint: InputFingerprint,
    traced_sources: OrderedSet[Source],
) -> InvokeSubgraphReuseCondition | None:
    """Build an InvokeSubgraphReuseCondition from a traced subgraph.

    A reuse condition is a mix of two kinds of checks:

    1. **Input tag checks** (from flat_vts): For each flattened leaf VT,
       we record its tag (InputTag.TENSOR/SYMNODE/CONSTANT/OBJECT) and
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
            if not isinstance(vt, TensorVariable):
                raise AssertionError(
                    f"expected TensorVariable for TENSOR tag, got {type(vt).__name__}"
                )
            example = vt.proxy.node.meta.get("example_value", None)
            if example is None:
                hc_log.debug(
                    "subgraph_reuse: cannot build condition -- tensor input has no example_value"
                )
                return None
            input_checks.append((InputTag.TENSOR, extract_tensor_metadata(example)))
        elif tag == InputTag.SYMNODE:
            if not isinstance(vt, SymNodeVariable):
                raise AssertionError(
                    f"expected SymNodeVariable for SYMNODE tag, got {type(vt).__name__}"
                )
            input_checks.append((InputTag.SYMNODE, sym_num_key(vt.sym_num)))
        elif tag == InputTag.CONSTANT:
            if not is_constant_like(vt):
                raise AssertionError(
                    f"expected constant-like VT for CONSTANT tag, got {type(vt).__name__}"
                )
            # Type is part of the key: `Mode.ADD == 1` and `True == 1` compare
            # equal, but an isinstance() check inside the region traces
            # differently for each, so value equality alone is not enough.
            input_checks.append((InputTag.CONSTANT, (type(vt.value), vt.value)))
        elif tag == InputTag.OBJECT:
            input_checks.append((InputTag.OBJECT, None))
        else:
            raise AssertionError(
                f"Unexpected input tag '{tag}' for {type(vt).__name__} -- "
                f"is_reuse_eligible should have rejected this"
            )

    # Collect all guards for sources accessed during the subgraph trace
    # and for the flattened arg sources.
    all_sources = OrderedSet(traced_sources)
    all_sources.update(s for s in fingerprint.arg_sources if s is not None)
    all_relevant_guards: OrderedSet[Guard] = OrderedSet()
    for source in all_sources:
        all_relevant_guards.update(tx.output.guards.get_guards_for_source(source))

    traced_sources = traced_sources.copy()
    guards: dict[Guard, tuple[GuardCheckSpec, object, OrderedSet[Source]]] = {}
    tensors = {}
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

        # Explicit object aliases are checked by the positional alias pattern.
        if type_str == "DUPLICATE_INPUT" and not isinstance(value, torch.Tensor):
            continue

        # TODO(anijain2305): vLLM workaround -- skip CONSTANT_MATCH on
        # strings. Re-evaluate once vLLM migrates off this pattern.
        # if type_str == "CONSTANT_MATCH" and isinstance(value, str):
        #     continue

        handler = cast(GuardCheckSpec, handler)
        extra_sources = ()
        if handler.get_extra_sources_fn is not None:
            extra_sources = handler.get_extra_sources_fn(guard)
            if has_mutated_vars(tx, OrderedSet(extra_sources)):
                return None
            traced_sources.update(extra_sources)
            try:
                value = (value, *map(tx.output.resolve_source_value, extra_sources))
            except Exception:
                raise RuntimeError(
                    f"subgraph_reuse: failed to resolve dependencies of '{source.name}' for {type_str} guard"
                ) from None
        if type_str == "TENSOR_MATCH" and source not in fingerprint.arg_sources:
            tensors[source] = value
        expected = handler.get_metadata_fn(guard, value)
        guards[guard] = (handler, expected, guard.source_dependencies(extra_sources))

    hc_log.debug("Number of guards %s", len(guards))
    needed_sources = traced_sources.union(*(deps for _, _, deps in guards.values()))
    roots = {s: None for s in fingerprint.arg_sources if s is not None}

    return InvokeSubgraphReuseCondition(
        input_checks=input_checks,
        overlapping_args=tuple(s for s in roots if _project_source(s, roots) is None),
        tensor_sources=tuple(tensors),
        aliases=input_aliases(tx, fingerprint, tensors.values()),
        guards=guards,
        treespec=fingerprint.treespec,
        traced_sources=traced_sources,
        guard_on_key_order=tx.output.guard_on_key_order.intersection(needed_sources),
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


def _project_source(
    source: Source, roots: Mapping[Source, Source | None]
) -> Source | None:
    """Substitute original argument sources; replacements cannot trigger another substitution."""
    children: dict[str, Source] = {}
    if isinstance(source, ChainedSource):
        for attr in fields(source):
            child = getattr(source, attr.name)
            if not isinstance(child, Source):
                continue
            value = roots[child] if child in roots else _project_source(child, roots)
            if value is None:
                return None
            if value is not child:
                children[attr.name] = value
    return replace(source, **children) if children else source


@dataclass
class SubgraphReuseMatch:
    """Replay guards on acceptance. Building a capture can register graph inputs and update PGO."""

    entry: InvokeSubgraphReuseEntry
    source_replacement: dict[Source, Source]
    tx: "InstructionTranslatorBase"
    condition: InvokeSubgraphReuseCondition | None = None
    resolve_cache: dict[Source, Any] = field(default_factory=dict)
    cloned_sources: dict[Source, Source] = field(default_factory=dict)
    runtime_guards: list[Guard] = field(default_factory=list)
    captured_variables: dict[Source, VariableTracker] = field(default_factory=dict)
    static_metadata: dict[Source, "SizesStridesInfo"] = field(default_factory=dict)
    pending_shape_guards: list[tuple[Any, bool]] = field(default_factory=list)
    symbol_replacements: dict[Any, Any] = field(default_factory=dict)

    def remap_source(self, source: Source) -> Source:
        if not self.source_replacement or source in self.source_replacement:
            return self.source_replacement.get(source, source)
        if source not in self.cloned_sources:
            cloned = _project_source(source, self.source_replacement)
            self.cloned_sources[source] = cloned or source
        return self.cloned_sources[source]

    def resolve(self, source: Source) -> Any:
        root = self.tx.output.root_tx
        scope = {"G": root.f_globals, "L": root.f_locals}
        return source.get_value(scope, {}, self.resolve_cache)

    def build_capture(self, source: Source) -> VariableTracker:
        from .builder import VariableBuilder

        if source not in self.captured_variables:
            build = VariableBuilder(self.tx, source)
            self.captured_variables[source] = build(self.resolve(source))
        return self.captured_variables[source]

    def remap_bound_symbol(self, lifted: LiftedBoundSymbol) -> Any:
        original = lifted.source
        if original is None or (source := self.remap_source(original)) is original:
            return lifted.expr
        if lifted.expr not in self.symbol_replacements:
            if isinstance(source, TensorPropertySource):
                tensor = self.build_capture(source.base)
                example = tensor.as_proxy().node.meta["example_value"]
                prop = getattr(example, source.prop.method_name())
                value = prop() if source.idx is None else prop(source.idx)
            else:
                vt = self.build_capture(source)
                value = (
                    vt.sym_num
                    if isinstance(vt, SymNodeVariable)
                    else vt.as_python_constant()
                )
            if isinstance(value, torch.SymInt):
                value = value.node._expr
            else:
                value = sympy.Integer(value)
            self.symbol_replacements[lifted.expr] = value
        return self.symbol_replacements[lifted.expr]

    def remap_symint(self, value: Any) -> Any:
        if not self.symbol_replacements or not isinstance(value, torch.SymInt):
            return value
        # Apply substitutions before ShapeEnv replacements can erase the old symbol.
        raw = cast(sympy.Expr, value.node._expr)
        expr = raw.xreplace(self.symbol_replacements)
        if expr == raw:
            return value
        return self.tx.output.shape_env.create_symintnode(expr, hint=None)

    def capture_policy_matches(self, source: Source, new_source: Source) -> bool:
        """Captured sizes, strides and offsets must agree with the cached symbol bindings."""
        from .builder import _automatic_dynamic, lookup_tracked_variable

        tx = self.tx
        cached = tx.output.input_source_to_var.get(source)
        try:
            value = self.resolve(new_source)
            if not isinstance(cached, TensorVariable):
                cached = lookup_tracked_variable(tx, self.resolve(source))
        except Exception:
            return _reject_reuse("cannot resolve %s", source.name, exc_info=True)
        if (
            not isinstance(cached, TensorVariable)
            or type(value) is not cached.python_type()
            or (example := cached.proxy.node.meta.get("example_value")) is None
        ):
            return _reject_reuse("incompatible cached tensor %s", source.name)
        plain = type(value) is torch.Tensor and tx.output.is_root_tracer()
        # An unchanged plain tensor has no symbolic metadata to rebind.
        if plain and source is new_source and not example._has_symbolic_sizes_strides:
            return True
        parameter = (
            type(value) is torch.nn.Parameter
            and dynamo_config.skip_no_tensor_aliasing_guards_on_parameters
        )
        if (
            source is not new_source
            and (plain or parameter)
            and value.layout is torch.strided
            and dynamo_config._dynamic_shapes_spec is None
            and value not in tx.output.tracing_context.tensor_to_context
            and not example._has_symbolic_sizes_strides
            and source not in self.entry.lifted_capture_sources
        ):
            static_shapes, _ = tensor_always_has_static_shape(value, True, new_source)
            context = _automatic_dynamic(value, tx, new_source, static_shapes)
            if isinstance(context, symbolic_shapes.StatelessSymbolicContext) and all(
                dim is symbolic_shapes.DimDynamic.STATIC
                for dim in context.dynamic_sizes
            ):
                self.static_metadata[new_source] = {
                    "size": value.size(),
                    "stride": value.stride(),
                    "needs_alias_check": plain,
                }
                guard = new_source.make_guard(GuardBuilder.TENSOR_MATCH)
                self.runtime_guards.append(guard)
                return True
        current = cached if source is new_source else self.build_capture(new_source)
        if (
            not isinstance(current, TensorVariable)
            or (current_example := current.proxy.node.meta.get("example_value")) is None
            or example.ndim != current_example.ndim
            or example.layout != current_example.layout
        ):
            return _reject_reuse("incompatible capture metadata %s", new_source.name)

        def properties(tensor: torch.Tensor) -> tuple[int | torch.SymInt, ...]:
            if tensor.layout is torch.strided:
                offset = tensor.storage_offset()
                # A concrete storage_offset() read installs its own value guard.
                symbolic_offset = offset if isinstance(offset, torch.SymInt) else 0
                return (*tensor.size(), *tensor.stride(), symbolic_offset)
            return tensor.size()

        for old, new in zip(properties(example), properties(current_example)):
            if isinstance(old, torch.SymInt) and isinstance(new, torch.SymInt):
                raw = cast(sympy.Expr, old.node._expr)
                expr = raw.xreplace(self.symbol_replacements)
                if tx.output.shape_env.replace(expr) == sym_num_key(new):
                    continue
            elif type(old) is int and type(new) is int and old == new:
                continue
            return _reject_reuse("capture policy mismatch %s", new_source.name)
        return True


def _treespecs_identical(a: pytree.TreeSpec, b: pytree.TreeSpec, /) -> bool:
    if a is b:
        return True
    return (
        a.__class__ is b.__class__
        and str(a.type) == str(b.type)
        and constants_identical(a.context, b.context)
        and a.num_children == b.num_children
        and all(
            _treespecs_identical(a.child(i), b.child(i)) for i in range(a.num_children)
        )
    )


def _reject_reuse(reason: str, *args: object, exc_info: bool = False) -> bool:
    hc_log.debug("subgraph_reuse: reuse failed -- " + reason, *args, exc_info=exc_info)
    return False


def _range_implies_bound(env: symbolic_shapes.ShapeEnv, expr: sympy.Basic) -> bool:
    if isinstance(expr, (sympy.Ge, sympy.Le)) and isinstance(expr.rhs, sympy.Integer):
        if (vr := env.var_to_range.get(expr.lhs)) is not None:
            edge = vr.lower if isinstance(expr, sympy.Ge) else vr.upper
            return expr.func(edge, expr.rhs) is sympy.true
    return False


def _shape_guards_match(
    env: symbolic_shapes.ShapeEnv, match: SubgraphReuseMatch
) -> bool:
    for predicate, oblivious in match.entry.shape_guards:
        try:
            expr = predicate.xreplace(match.symbol_replacements)
        except ZeroDivisionError:
            return _reject_reuse("zero divisor in %s", predicate)
        if expr == predicate or (not oblivious and _range_implies_bound(env, expr)):
            continue
        if expr.free_symbols <= env.var_to_range.keys():
            try:
                if env.guarding_hint_or_throw(expr):
                    match.pending_shape_guards.append((expr, oblivious))
                    continue
            except symbolic_shapes.GuardOnDataDependentSymNode:
                bounds = env.bound_sympy(expr, size_oblivious=oblivious)
                if bounds.is_singleton() and bool(bounds.lower):
                    continue
        return _reject_reuse("%s rejected as %s", predicate, expr)
    return True


def is_reusable(
    tx: "InstructionTranslatorBase",
    condition: "InvokeSubgraphReuseCondition",
    fingerprint: InputFingerprint,
    match: SubgraphReuseMatch,
) -> bool:
    """Check if a cached subgraph can be reused for the current call.

    Three-phase check:
    (1) Verify that intermediates (tensor metadata, symnode types, constant
        values) match the cached input_checks — these are lightweight
        structural comparisons that don't require source resolution.
    (2) Check for mutations on the remapped traced_sources — if any source
        the subgraph read has been mutated since the original trace, the
        cached guards would evaluate against stale values.
    (3) Use the match's source replacement mapping (old sources -> new sources) to
        re-evaluate the snapshotted guards under the new sources.
        Check aliases, rebound symbols and capture metadata, then collect guards for stamp-out.
    """
    cached_entry = match.entry
    # Structural check: treespec must match first.
    if condition.treespec is not None and (
        fingerprint.treespec is None
        or not _treespecs_identical(fingerprint.treespec, condition.treespec)
    ):
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
            if not isinstance(cur_vt, TensorVariable):
                raise AssertionError(
                    f"expected TensorVariable for TENSOR tag, got {type(cur_vt).__name__}"
                )
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
            if not isinstance(cur_vt, SymNodeVariable):
                raise AssertionError(
                    f"expected SymNodeVariable for SYMNODE tag, got {type(cur_vt).__name__}"
                )
            if sym_num_key(cur_vt.sym_num) != cached_val:
                hc_log.debug(
                    "subgraph_reuse: reuse failed -- input %d symnode mismatch: cached '%s' vs current '%s'",
                    i,
                    cached_val,
                    cur_vt.sym_num,
                )
                return False
        elif cached_tag == InputTag.CONSTANT:
            if not is_constant_like(cur_vt):
                raise AssertionError(
                    f"expected constant-like VT for CONSTANT tag, got {type(cur_vt).__name__}"
                )
            cached_type, cached_value = cast(tuple[type, Any], cached_val)
            if type(cur_vt.value) is not cached_type:
                # Not deferred to the source check below: a value guard cannot
                # catch a type change that alters the trace (isinstance, etc).
                hc_log.debug(
                    "subgraph_reuse: reuse failed -- input %d constant type "
                    "mismatch: cached '%s' vs current '%s'",
                    i,
                    cached_type,
                    type(cur_vt.value),
                )
                return False
            if not constants_identical(cur_vt.value, cached_value):
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
                if cached_src is not None and any(
                    is_from_source(cached_src, read)
                    for read, _ in condition.lexical_sources
                ):
                    return _reject_reuse("lexical constant changed")
                if cached_src is None or new_src is None:
                    hc_log.debug(
                        "subgraph_reuse: reuse failed -- input %d constant mismatch "
                        "with no source to replace: cached '%s' vs current '%s'",
                        i,
                        cached_val,
                        cur_vt.value,
                    )
                    return False

    bindings: dict[Source, Source | None] = {}
    for old, new in zip(cached_entry.arg_sources, fingerprint.arg_sources):
        if old is not None and bindings.setdefault(old, new) != new:
            return _reject_reuse("conflicting argument binding")
        if isinstance(old, ConstDictKeySource) and type(old) is not type(new):
            return _reject_reuse("incompatible dictionary key source")
    for old in condition.overlapping_args:
        projected = _project_source(old, bindings)
        if projected is None or projected != bindings[old]:
            return _reject_reuse("overlapping argument binding changed")
    # Parameterized source - this function gives you new sources parameterized
    # on the arg_sources. For example, if the input to the nested compile region
    # is a nn Module layer with source `layers[0]`, then old source
    # `layers[0].weight` gets remapped to `layers[1].weight`. This
    # parameterization is central in getting the new sources and then running
    # guards on them.
    remap = match.remap_source

    # Check for mutations on remapped traced_sources.
    if tx.output.side_effects.mutated_sources and has_mutated_vars(
        tx, OrderedSet(remap(s) for s in condition.traced_sources)
    ):
        return False
    # Shared resolution context so source.get_value memoizes intermediate
    # results (e.g. common base sources) across all guards in this check.
    resolve = match.resolve
    capture_checks: list[tuple[Source, Source]] = []
    guards_to_replay: list[Guard] = []
    for guard, (handler, expected, dependencies) in condition.guards.items():
        source = guard.originating_source
        changed = not match.source_replacement.keys().isdisjoint(dependencies)
        new_source = remap(source) if changed else source
        is_tensor = guard.create_fn_name() == "TENSOR_MATCH"
        if is_tensor:
            capture_checks.append((source, new_source))
        if new_source is source and (
            not changed or all(remap(d) is d for d in dependencies)
        ):
            continue

        try:
            value = resolve(new_source)
            if handler.get_extra_sources_fn is not None:
                extras = handler.get_extra_sources_fn(guard)
                value = (value, *(resolve(remap(dep)) for dep in extras))
        except Exception:
            hc_log.debug(
                "subgraph_reuse: reuse failed -- cannot resolve source\n"
                "  guard type: %s\n"
                "  guard source: %s\n"
                "  guard source name: %s\n"
                "  user stack:\n%s",
                guard.create_fn_name(),
                new_source,
                new_source.name,
                "".join(guard.user_stack.format())
                if guard.user_stack
                else "<no stack>",
            )
            return False

        if not handler.eval_fn(value, expected):
            hc_log.debug(
                "subgraph_reuse: reuse failed --\n"
                "  guard type: %s\n"
                "  guard source: %s\n"
                "  guard source name: %s\n"
                "  expected: %s\n"
                "  got: %s\n"
                "  user stack:\n%s",
                guard.create_fn_name(),
                new_source,
                new_source.name,
                expected,
                value,
                "".join(guard.user_stack.format())
                if guard.user_stack
                else "<no stack>",
            )
            return False

        if not is_tensor:
            guards_to_replay.append(guard)

    try:
        for source, _ in condition.lexical_sources:
            remapped = remap(source)
            if remapped is not source and resolve(remapped) is not resolve(source):
                return _reject_reuse("lexical alias changed")
        tensors = [resolve(remap(s)) for s in condition.tensor_sources]
    except Exception:
        # Resolving a candidate source can run arbitrary user getters.
        return _reject_reuse("cannot resolve aliases", exc_info=True)
    if input_aliases(tx, fingerprint, tensors) != condition.aliases:
        return _reject_reuse("argument or capture aliases differ")
    if not match.source_replacement:
        return True

    for lifted in cached_entry.subgraph_input_mapping:
        if isinstance(lifted, LiftedBoundSymbol):
            match.remap_bound_symbol(lifted)

    if match.symbol_replacements and not _shape_guards_match(
        tx.output.shape_env, match
    ):
        return False
    for source, new_source in capture_checks:
        if new_source is not source or match.symbol_replacements:
            if not match.capture_policy_matches(source, new_source):
                return False

    match.runtime_guards.extend(g.remap_sources(remap) for g in guards_to_replay)
    return True


def has_reuse_entries(
    tx: "InstructionTranslatorBase",
    fn_var: Any,
) -> bool:
    """Cheap check: does the cache have any entries for this function?"""
    from torch._guards import InvokeSubgraphCache

    invoke_subgraph_cache = tx.output.tracing_context.hop_dispatch_set_cache.get_cache(
        torch._higher_order_ops.invoke_subgraph
    )
    if not isinstance(invoke_subgraph_cache, InvokeSubgraphCache):
        return False
    fn_code = get_fn_key(fn_var, tx)
    return fn_code is not None and fn_code in invoke_subgraph_cache.subgraph_reuse_cache


def find_reuse_match(
    tx: "InstructionTranslatorBase",
    fn_var: Any,
    fingerprint: InputFingerprint,
) -> SubgraphReuseMatch | None:
    from torch._guards import InvokeSubgraphCache

    invoke_subgraph_cache = tx.output.tracing_context.hop_dispatch_set_cache.get_cache(
        torch._higher_order_ops.invoke_subgraph
    )
    if not isinstance(invoke_subgraph_cache, InvokeSubgraphCache):
        return None
    fn_code = get_fn_key(fn_var, tx)
    if fn_code is None:
        return None

    # this evaluator function is called one by one for all the invoke subgraph
    # reuse entries - the one that evaluates to True is stamped out in the
    # graph.
    def evaluator(
        cond: "InvokeSubgraphReuseCondition", entry: InvokeSubgraphReuseEntry
    ) -> SubgraphReuseMatch | None:
        sources = build_source_replacement(entry.arg_sources, fingerprint.arg_sources)
        match = SubgraphReuseMatch(entry, sources, tx, cond)
        return match if is_reusable(tx, cond, fingerprint, match) else None

    return invoke_subgraph_cache.find_reuse_entry(fn_code, evaluator)


def save_reuse_entry(
    tx: "InstructionTranslatorBase",
    fn_var: Any,
    fingerprint: InputFingerprint,
    body_name: str,
    body_gmod: torch.fx.GraphModule,
    config: NestedCompileRegionOptions | None,
    p_args: tuple[Any, ...],
    body_r: VariableTracker,
    example_value: Any,
    max_reuse_entries: int = 8,
    condition: "InvokeSubgraphReuseCondition | None" = None,
    hash_key: int | None = None,
    traced_sources: OrderedSet[Source] | None = None,
) -> None:
    """Save a traced subgraph into the reuse cache for future cache hits.

    Builds an InvokeSubgraphReuseEntry with the freevar mapping (how each
    lifted arg maps back to user inputs or captured variables), output
    metadata, and arg sources. On a future cache hit, stamp_out_subgraph
    uses this entry to emit a new invoke_subgraph call without re-tracing.

    Exactly one of ``condition`` or ``hash_key`` must be provided.
    ``condition`` stores the entry in the guard-based cache (linear scan);
    ``hash_key`` stores it in the hash-key cache (O(1) lookup).
    """
    from torch._guards import InvokeSubgraphCache

    if not ((condition is None) != (hash_key is None)):
        raise AssertionError("Exactly one of condition or hash_key must be provided")

    invoke_subgraph_cache = tx.output.tracing_context.hop_dispatch_set_cache.get_cache(
        torch._higher_order_ops.invoke_subgraph
    )
    if not isinstance(invoke_subgraph_cache, InvokeSubgraphCache):
        return

    fn_code = get_fn_key(fn_var, tx if condition is not None else None)
    if fn_code is None:
        return

    subgraph_input_mapping = build_subgraph_input_mapping(
        tx, p_args, fingerprint.flat_vts, traced_sources or OrderedSet()
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

    rebound: set[sympy.Symbol] = set()
    captures = set()
    for item in subgraph_input_mapping:
        if isinstance(item, LiftedBoundSymbol) and item.source is not None:
            rebound.update(item.expr.free_symbols)
        elif isinstance(item, LiftedCapturedSource):
            captures.add(item.source)
    shape_guards: list[tuple[sympy.Basic, bool]] = []
    if rebound and condition is not None:
        shape_env = tx.output.shape_env
        candidates = [
            (guard.expr, guard.size_oblivious)
            for guard in shape_env.guards
            if guard.expr.free_symbols & rebound
        ]
        # Range facts can have removed a branch before any ShapeGuard was emitted.
        for symbol in rebound & shape_env.var_to_range.keys():
            bounds = shape_env.var_to_range[symbol]
            for rel, limit in ((sympy.Ge, bounds.lower), (sympy.Le, bounds.upper)):
                if isinstance(limit, sympy.Integer):
                    candidates.append((rel(symbol, limit), False))
        for assertions in shape_env.deferred_runtime_asserts.values():
            candidates.extend(
                (a.expr, False) for a in assertions if a.expr.free_symbols & rebound
            )
        # Implications cannot cross the size-oblivious evaluation policy.
        implied: dict[bool, set[sympy.Basic]] = {False: set(), True: set()}
        for predicate, oblivious in candidates:
            known = implied[oblivious]
            if symbolic_shapes.canonicalize_bool_expr(predicate) in known:
                continue
            shape_guards.append((predicate, oblivious))
            facts = shape_env.get_implications(predicate)
            known.update(expr for expr, truth in facts if truth is sympy.true)

    entry = InvokeSubgraphReuseEntry(
        body_name=body_name,
        body_gmod=body_gmod,
        config=config,
        subgraph_input_mapping=subgraph_input_mapping,
        lifted_capture_sources=frozenset(captures),
        single_tensor_output=single_tensor_output,
        output_metadata=output_metadata,
        # Record arg sources so that on cache hit we can build a
        # source replacement mapping (old sources → new sources) to
        # rewrite captured variable sources for the current invocation.
        arg_sources=fingerprint.arg_sources,
        num_user_outputs=num_user_outputs,
        shape_guards=shape_guards,
    )
    if condition is not None:
        invoke_subgraph_cache.add_reuse_entry(
            fn_code, condition, entry, max_reuse_entries
        )
    else:
        if hash_key is None or not isinstance(fn_code, types.CodeType):
            raise AssertionError("expected a hash key and function code")
        invoke_subgraph_cache.add_reuse_entry_by_key(
            fn_code, hash_key, entry, max_reuse_entries
        )


def trace_reuse_hash_fn(
    tx: "InstructionTranslatorBase",
    reuse_hash_fn: Any,
    fn_args_vt: "list[VariableTracker]",
    kwargs: dict[str, VariableTracker],
) -> int:
    """Trace the user's reuse_hash_fn to get a constant integer hash key.

    Guards installed during the hash function tracing are skipped — the hash
    key itself is the reuse condition, not the guards.
    """
    from torch._dynamo.exc import Unsupported

    with tx.output.tracing_context.guards_context.skip_guard_install():
        try:
            result = _make_inlined(tx, reuse_hash_fn)(*fn_args_vt, **kwargs)
        except Unsupported as e:
            raise RuntimeError(
                f"reuse_hash_fn must be fully traceable without graph breaks. Got: {e}"
            ) from e

    if not isinstance(result, ConstantVariable) or not isinstance(result.value, int):
        raise RuntimeError(
            f"reuse_hash_fn must return a constant integer, got {result}"
        )

    return result.value


def find_reuse_entry_by_key(
    tx: "InstructionTranslatorBase",
    fn_var: Any,
    hash_key: int,
) -> InvokeSubgraphReuseEntry | None:
    from torch._guards import InvokeSubgraphCache

    invoke_subgraph_cache = tx.output.tracing_context.hop_dispatch_set_cache.get_cache(
        torch._higher_order_ops.invoke_subgraph
    )
    if not isinstance(invoke_subgraph_cache, InvokeSubgraphCache):
        return None
    fn_code = get_fn_key(fn_var)
    if not isinstance(fn_code, types.CodeType):
        return None
    return invoke_subgraph_cache.find_reuse_entry_by_key(fn_code, hash_key)


def stamp_out_subgraph(
    tx: "InstructionTranslatorBase",
    fingerprint: InputFingerprint,
    match: SubgraphReuseMatch,
) -> VariableTracker:
    """Emit a new invoke_subgraph call by stamping out a cached subgraph.

    Sources in the cached entry are parameterized: they refer to the original
    call's sources and must be rewritten to the current call's sources via
    source replacement before we can look up or create the corresponding
    graph placeholders. Replay guards, static metadata, order annotations and lexical dependencies.
    """
    from torch._dynamo.variables.higher_order_ops import add_call_function, make_attr

    cached = match.entry
    tracer = tx.output.current_tracer
    tx.output.input_source_to_sizes_strides.update(match.static_metadata)
    for expr, oblivious in match.pending_shape_guards:
        tx.output.shape_env.evaluate_expr(expr, hint=True, size_oblivious=oblivious)
    flat_proxies = get_flat_proxies(fingerprint)
    for guard in match.runtime_guards:
        existing = tx.output.guards.get_guards_for_source(guard.originating_source)
        if not any(guard.is_equivalent_to(other) for other in existing):
            install_guard(guard)
    if (condition := match.condition) is not None:
        remap = match.remap_source
        tx.output.guard_on_key_order.update(map(remap, condition.guard_on_key_order))
        if tracer.tracks_lexical_sources:
            tracer.traced_sources.update(map(remap, condition.traced_sources))
            for source, scope in condition.lexical_sources:
                if scope <= tracer.debug_level:
                    tracer.lexical_sources.add((source, scope))
                    tracer.lexical_sources.add((remap(source), scope))

    new_lifted_args = []
    # Find the args for the about-to-be-inserted invoke_subgraph call.
    for subgraph_input in cached.subgraph_input_mapping:
        if isinstance(subgraph_input, LiftedUserArg):
            new_lifted_args.append(flat_proxies[subgraph_input.index])
        elif isinstance(subgraph_input, LiftedBoundSymbol):
            from torch._dynamo.output_graph import LazyProxy

            binding_key = match.remap_bound_symbol(subgraph_input)
            bindings = tracer.bound_symbols
            if binding_key in bindings or subgraph_input.source is None:
                proxy = bindings[binding_key]
            else:
                source = match.remap_source(subgraph_input.source)
                if isinstance(source, TensorPropertySource):
                    tensor = match.build_capture(source.base)
                    idx = source.idx
                    args = [] if idx is None else [ConstantVariable.create(idx)]
                    var = tensor.call_method(tx, source.prop.method_name(), args, {})
                else:
                    var = match.build_capture(source)
                proxy = var.as_proxy()
            if isinstance(proxy, LazyProxy):
                proxy = proxy()
                bindings[binding_key] = proxy
            new_lifted_args.append(proxy)
        elif isinstance(subgraph_input, LiftedSyntheticObject):
            ctor_args = subgraph_input.ctor_args
            ctor_arg_sources = subgraph_input.ctor_arg_sources
            if ctor_arg_sources and match.source_replacement:
                new_ctor_args = []
                new_ctor_arg_sources = []
                for val, arg_src in zip(ctor_args, ctor_arg_sources):
                    if arg_src is not None:
                        arg_src = match.remap_source(arg_src)
                        val = match.resolve(arg_src)
                    new_ctor_args.append(val)
                    new_ctor_arg_sources.append(arg_src)
                ctor_args = tuple(new_ctor_args)
                ctor_arg_sources = tuple(new_ctor_arg_sources)
            vt = tx.output.synthetic_graph_input(
                subgraph_input.ctor_fn, ctor_args, ctor_arg_sources
            )
            new_lifted_args.append(vt.as_proxy())
        elif isinstance(subgraph_input, LiftedCapturedSource):
            new_source = match.remap_source(subgraph_input.source)
            # VariableBuilder deduplicates via input_source_to_var,
            # so this reuses existing graph placeholders automatically.
            vt = match.build_capture(new_source)
            new_lifted_args.append(vt.as_proxy())

    # Generate fake tensor outputs
    if tx.fake_mode is None:
        raise AssertionError("tx.fake_mode must not be None for stamp_out_subgraph")
    with tx.fake_mode:
        example_value = tuple(
            torch.empty_strided(
                tuple(match.remap_symint(dim) for dim in shape),
                tuple(match.remap_symint(dim) for dim in stride),
                dtype=dtype,
                device=device,
                requires_grad=req_grad,
            )
            for shape, stride, dtype, device, req_grad in cached.output_metadata
        )

    # Install the invoke_subgraph call
    body = tracer.reused_subgraph_proxies.get(cached.body_name)
    if body is None or body.node._erased or body.node.graph is not tracer.graph:
        body = make_attr(tx, cached.body_name)
        tracer.reused_subgraph_proxies[cached.body_name] = body
    p_args = (body, cached.body_name, *new_lifted_args)
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
        if not isinstance(items[0], TensorVariable):
            raise AssertionError(
                f"Expected tensor output but got {type(items[0]).__name__}"
            )
        return items[0]

    items = flat_variable.items  # pyrefly: ignore[missing-attribute]
    n = cached.num_user_outputs
    if n > 0 and n < len(items):
        from .builder import SourcelessBuilder

        return SourcelessBuilder.create(tx, tuple(items[:n]))
    return flat_variable


def _proxy_source(tx: "InstructionTranslatorBase", proxy: Proxy) -> Source | None:
    if (grapharg := proxy.node.meta.get("grapharg")) is not None:
        return grapharg.source
    if proxy.node.op == "get_attr" and tx.output.param_name_to_source:
        return tx.output.param_name_to_source.get(cast(str, proxy.node.target))
    return None


def build_subgraph_input_mapping(
    tx: "InstructionTranslatorBase",
    p_args: tuple[Any, ...],
    flat_vts: list[tuple[InputTag, VariableTracker]],
    traced_sources: OrderedSet[Source],
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
    explicit_symbols = set()
    proxy_node_to_idx: dict[torch.fx.Node, int] = {}
    idx = 0
    for tag, vt in flat_vts:
        if tag in (InputTag.TENSOR, InputTag.SYMNODE):
            node = vt.as_proxy().node
            if node not in proxy_node_to_idx:
                proxy_node_to_idx[node] = idx
                idx += 1
                example = node.meta.get("example_value")
                for sym in symbolic_shapes._iterate_nodes(example):
                    explicit_symbols.update(cast(sympy.Expr, sym._expr).free_symbols)

    subgraph_input_mapping: list[LiftedArgOrigin] = []
    for outer_proxy in p_args[2:]:
        matched_idx = proxy_node_to_idx.get(outer_proxy.node, -1)
        if matched_idx >= 0:
            subgraph_input_mapping.append(LiftedUserArg(matched_idx))
        else:
            grapharg = outer_proxy.node.meta.get("grapharg", None)
            source = _proxy_source(tx, outer_proxy)
            # SymInt freevars must reuse the existing symbolic proxy rather
            # than resolving via source.get_value() (which returns the
            # concrete int). They appear as either:
            # - placeholder nodes with grapharg.example being a SymInt
            # - call_function nodes (e.g. sym_size_int) with no grapharg
            example = (
                grapharg.example
                if grapharg is not None
                else outer_proxy.node.meta.get("example_value", None)
            )
            if isinstance(example, torch.SymInt):
                # _expr rather than expr: expr applies the ShapeEnv's
                # replacements, so a symbol the region lifted before it was
                # specialized comes back as a constant, which bound_symbols has
                # no entry for.
                # The first creator of a shared symbol need not be read by this region.
                expr = example.node._expr
                dep = (
                    source.base if isinstance(source, TensorPropertySource) else source
                )
                if dep not in traced_sources or expr in explicit_symbols:
                    source = None
                subgraph_input_mapping.append(LiftedBoundSymbol(expr, source))
                continue
            if source is None:
                raise AssertionError(
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
    allow_side_effects = True
    # invoke_subgraph is NOT desugared in AOTAutograd, so the HOP input/output
    # shouldn't alias. For checkpoint HOP, we inline it so we don't need
    # alias analysis as functionalization would just work on the flat graph.
    filter_aliased_intermediates = True

    # pyrefly: ignore[bad-override]
    def install_subgraph_in_output_graph(
        self,
        tx: "InstructionTranslatorBase",
        fn_vt: VariableTracker,
        fn_args_vt: "list[VariableTracker]",
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
            fn_code = fn_vt.get_function().__code__
            fn_name = fn_vt.get_function().__name__
        else:
            if not isinstance(fn_vt, UnspecializedNNModuleVariable):
                raise AssertionError(
                    f"expected UnspecializedNNModuleVariable, got {type(fn_vt).__name__}"
                )
            fn_code = fn_vt.value.forward.__func__.__code__  # type: ignore[attr-defined]
            fn_name = fn_vt.value.forward.__name__  # type: ignore[attr-defined]
        # pyrefly: ignore [implicit-any]
        previously_installed_submodules = []
        if invoke_subgraph_cache:
            previously_installed_submodules = (
                invoke_subgraph_cache.get_dynamo_installed_submodules(fn_code)
            )
            current_mod = body_gmod
            # NB - reverse is more likely to cause a hit sooner because first
            # graph can have requires_grad=False for a few inputs
            for submodule_name in reversed(previously_installed_submodules):
                if submodule_name not in tx.output.nn_modules:
                    raise AssertionError(
                        f"submodule '{submodule_name}' not found in nn_modules"
                    )
                previous_mod = tx.output.nn_modules[submodule_name]
                if not tx.fake_mode:
                    raise AssertionError(
                        "tx.fake_mode must be set for subgraph comparison"
                    )
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
            invoke_subgraph_cache.add_dynamo_installed_submodule(fn_code, body_name)

        return body_name

    def _call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: "list[VariableTracker]",
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
        reuse_hash_fn = None
        if hasattr(fn_var, "get_function"):
            try:
                fn = fn_var.get_function()
                config = getattr(fn, "__marked_compile_region_config__", None)
                max_reuse_entries = getattr(
                    fn, "__marked_compile_region_max_reuse_entries__", 8
                )
                reuse_hash_fn = getattr(
                    fn, "__marked_compile_region_reuse_hash_fn__", None
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

        # User-provided reuse_hash_fn path: hash key determines cache lookup.
        if reuse and reuse_hash_fn is not None:
            with dynamo_timed("invoke_subgraph_reuse_hash_fn"):
                hash_key = trace_reuse_hash_fn(tx, reuse_hash_fn, fn_args_vt, kwargs)

            cached = find_reuse_entry_by_key(tx, fn_var, hash_key)
            if cached is not None:
                hc_log.debug(
                    "subgraph_reuse: hash key %d hit for '%s', reusing subgraph '%s'",
                    hash_key,
                    fn_var,
                    cached.body_name,
                )
                fingerprint = build_input_fingerprint(tx, fn_args_vt, kwargs)
                with dynamo_timed("invoke_subgraph_reuse_stamp_out"):
                    sources = build_source_replacement(
                        cached.arg_sources, fingerprint.arg_sources
                    )
                    match = SubgraphReuseMatch(cached, sources, tx)
                    return stamp_out_subgraph(tx, fingerprint, match)

        # Automatic reuse lookup (guard-based): check fn_code first (cheap) to
        # avoid the expensive pytree flatten in build_input_fingerprint on
        # the first call when there's nothing in the cache yet.
        elif reuse and has_reuse_entries(tx, fn_var):
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
                    match.entry.body_name,
                )
                with dynamo_timed("invoke_subgraph_reuse_stamp_out"):
                    return stamp_out_subgraph(tx, fingerprint, match)

        if self._HOP_NAME is None:
            raise AssertionError("_HOP_NAME must not be None")
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
            traced_sources = tracing_info.traced_sources.copy()
            # Reused VariableTrackers bypass VariableBuilder's source recording.
            for proxy in p_args[2:]:
                if (source := _proxy_source(tx, proxy)) is not None:
                    traced_sources.add(source)
            if reuse_hash_fn is not None:
                if not is_reuse_eligible(
                    tx,
                    body_r,
                    fingerprint,
                    tracing_info,
                    traced_sources,
                    has_reuse_hash_fn=True,
                ):
                    raise RuntimeError(
                        "reuse_hash_fn was provided but the subgraph is not "
                        "eligible for reuse. Check the logs with "
                        "TORCH_LOGS='+hierarchical_compile' for details."
                    )
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
                    max_reuse_entries,
                    hash_key=hash_key,  # type: ignore[possibly-undefined]
                    traced_sources=traced_sources,
                )
            else:
                if is_reuse_eligible(
                    tx, body_r, fingerprint, tracing_info, traced_sources
                ):
                    condition = build_reuse_condition(
                        tx,
                        fingerprint,
                        traced_sources,
                    )
                    if condition is not None:
                        condition.lexical_sources = tracing_info.lexical_sources.copy()
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
                            max_reuse_entries,
                            condition=condition,
                            traced_sources=traced_sources,
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
