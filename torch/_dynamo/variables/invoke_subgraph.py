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
from dataclasses import dataclass
from typing import Any, cast, NamedTuple, TYPE_CHECKING, TypeGuard

import torch
import torch._higher_order_ops
from torch._dynamo import graph_break_hints
from torch._dynamo.exc import unimplemented
from torch._dynamo.guards import (
    extract_tensor_metadata,
    GUARD_VALUE_DISPATCH,
    GuardBuilder,
    GuardCheckSpec,
    install_guard,
    SKIP_GUARD,
    UnsupportedGuardCheckSpec,
)
from torch._dynamo.source import (
    _get_source_debug_name,
    GetItemSource,
    SyntheticLocalSource,
)
from torch._dynamo.utils import _make_inlined, unpack_iterable
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
from torch.fx.graph_module import GraphModule
from torch.fx.proxy import Proxy
from torch.utils import _pytree as pytree
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase
    from torch._dynamo.variables.higher_order_ops import SubgraphTracingInfo
    from torch._dynamo.variables.lazy import LazyConstantVariable

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


# Note: [invoke_subgraph index parameterization]
#
# A region that subscripts a captured container with a value it read from a
# guarded location -- `pool.buffers[self.layer_id]` -- bakes the index into the
# lifted capture's Source. The guard on the index then fails for every distinct
# index, so the region is retraced once per layer.
#
# The fix parameterizes the cached entry over the index instead of specializing
# to it: the (element_source, index_source) pairs are recorded, and on every
# reuse lookup the index is resolved again from its own source and every source
# derived from that subscript is rebuilt around the new index. The rebuilt
# sources feed both the guard re-evaluation in is_reusable (so the *element's*
# own guards -- tensor metadata, a folded float constant, a nested subscript --
# are re-checked against the element this call actually selects) and the
# capture reconstruction in stamp_out_subgraph.
#
# Re-deriving is only sound when the index was read *purely* to subscript. If
# the region also branched on it, or baked it into the graph as an operand, the
# cached body is wrong for a different index and there is no guard left to
# catch that once the index guard is dropped.
#
# The guard set cannot answer that question as it stands, because a subscript
# and a branch install the same CONSTANT_MATCH on the same source. What
# separates them is that only the subscript can do without one: it needs the
# value at trace time, but the region can be rebuilt for a different value,
# whereas a branch or an operand bakes it into the graph.
#
# LazyConstantVariable is what makes reading the value without guarding it
# possible: it holds a primitive whose guard has not been installed yet, and
# peek_value() reads it without realizing. So a subscript inside a region takes
# the value that way and leaves the guard *deferred*, registering it with the
# open region. Every other use realizes the constant and installs the guard
# itself. When the region closes, a value guard on the index source therefore
# means -- and only means -- that the region read the index for something
# besides subscripting, and the entry is not parameterized. Either way the
# region then pays back the deferred guard, so no read leaves the frame less
# guarded than it is today; only the *reuse condition* treats the index as a
# parameter. The deferred read is stricter in two spots: an index that would
# have realized symbolically, or that today is never realized at all, comes out
# CONSTANT_MATCH specialized where it otherwise would not be.
#
# The deferral window is one region: the guard is installed by the time the
# region closes, before its reuse condition is built. A second region reading
# the same index therefore sees that guard and falls back to retracing.
#
# Everything else falls back to the existing behaviour of retracing: a
# container reached through sq_item alone rather than mp_subscript (a deque,
# say), dict and ModuleList containers, negative or out of range indices, and
# indices arrived at through arithmetic.


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


GUARDS_PINNING_A_VALUE = frozenset({"CONSTANT_MATCH", "EQUALS_MATCH", "ID_MATCH"})


@dataclass
class IndexedSubscript:
    """A subscript the region did with an index whose guard it deferred.

    ``element_source`` is the GetItemSource the subscript produced and
    ``index_source`` is where the index itself was read from, e.g.
    ``pool.buffers[0]`` selected by ``self.layer_id``. See Note:
    [invoke_subgraph index parameterization].
    """

    element_source: Source
    index_source: Source
    index_vt: Any


class open_index_parameterized_region:
    """Defer index guards for one invoke_subgraph region and settle them on exit.

    ``reindexable`` is the region's verdict, computed on exit while the guards
    are still deferred: element source -> index source for every subscript the
    region can be re-derived for. The guards are then installed, so the reuse
    condition built afterwards guards every index the verdict left out.
    """

    def __init__(self, tx: "InstructionTranslatorBase") -> None:
        self.tx = tx
        self.records: list[IndexedSubscript] = []
        # Elements the region also reached by subscripting with a literal
        # index. Such a read produces the same source, and the same
        # VariableTracker, as a deferred one at the index the region traced
        # with, so re-deriving would move a capture the literal read expects
        # to stay put. See Note: [invoke_subgraph index parameterization].
        self.literal_elements: OrderedSet[Source] = OrderedSet()
        self.reindexable: dict[Source, Source] = {}

    def __enter__(self) -> "open_index_parameterized_region":
        self.tx.output.deferred_index_regions.append(self)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.tx.output.deferred_index_regions.pop()
        if exc_info[0] is not None or not self.records:
            # Either the region unwound (graph break, restart), which produces
            # no entry, or it deferred nothing, so there is nothing to settle.
            # Specializing the frame on indexes it read would be gratuitous.
            return
        self.reindexable = resolve_reindexable(
            self.tx, self.records, self.literal_elements
        )
        # Pay back every deferred guard, including the ones the verdict kept:
        # an entry parameterized on an index is still a graph specialized to
        # the index it traced with, and the frame has to guard that.
        for index_source in OrderedSet(r.index_source for r in self.records):
            install_guard(index_source.make_guard(GuardBuilder.CONSTANT_MATCH))


def subscript_without_realizing_index(
    tx: "InstructionTranslatorBase",
    container_vt: VariableTracker,
    index_vt: VariableTracker,
) -> VariableTracker | None:
    """``container_vt[index_vt]`` inside a region, deferring the index's guard.

    Returns the selected element, or None to fall back to the normal
    (index-realizing) path. Falling back is always correct: it installs the
    index's guard right away, which is exactly what tells the enclosing region
    that the index was read for something other than a subscript.
    """
    from torch._dynamo.variables.lazy import LazyConstantVariable
    from torch._dynamo.variables.lists import ListVariable, TupleVariable

    regions = tx.output.deferred_index_regions
    if not regions:
        return None
    if type(index_vt) is not LazyConstantVariable or index_vt.is_realized():
        return None
    # Exactly list/tuple: subclasses (SizeVariable, namedtuples, ...) source
    # their elements differently.
    if type(container_vt) not in (ListVariable, TupleVariable):
        return None
    container_vt = cast("ListVariable | TupleVariable", container_vt)
    container_source = container_vt.source
    index_source = index_vt.source
    if container_source is None or index_source is None:
        return None

    index = index_vt.peek_value()
    # `is int` and not isinstance: bool is an int subclass, and `buffers[True]`
    # would rebuild as a bool-indexed source.
    if type(index) is not int:
        return None
    # Forward, in-range indices only. Anything else does not produce a source
    # whose literal is the index we would re-derive, so it falls back and is
    # guarded normally -- which also blocks re-deriving any *other* subscript
    # that shares the index, since the fallback installs its guard.
    if not 0 <= index < len(container_vt.items):
        return None

    item = container_vt.items[index]
    # The element must be reachable by re-subscripting the same container at
    # the literal index; otherwise there is nothing to re-derive.
    element_source = GetItemSource(container_source, index)
    if item.source != element_source:
        return None

    # Normally VariableBuilder records this when it realizes the constant.
    # Without it the deferred guard would never make it into a reuse condition,
    # and an index this region turns out not to be able to re-derive would go
    # unchecked.
    tx.output.current_tracer.traced_sources.add(index_source)
    # The element too. Re-derivation only stays sound because the element's own
    # guards are rebased onto whatever this call selects, and the condition
    # collects guards per traced source. A caller that already materialized
    # this element before the region ran leaves it out of traced_sources, so
    # without this the element would move with nothing checking its metadata.
    tx.output.current_tracer.traced_sources.add(element_source)
    # Only the innermost region records. An enclosing region sees the guard
    # this one installs on exit and falls back, which is what we want: its own
    # captures came through a body it cannot re-derive by itself.
    regions[-1].records.append(IndexedSubscript(element_source, index_source, index_vt))
    return item


def realized_to_non_constant(index_vt: "LazyConstantVariable") -> bool:
    """Whether ``index_vt`` realized to something other than a ConstantVariable.

    An int can realize to a SymNodeVariable rather than a ConstantVariable, and
    then a branch on it installs a shape guard instead of the CONSTANT_MATCH
    the verdict looks for. Treat that as read-for-something-else.
    """
    if not index_vt.is_realized():
        return False
    return not isinstance(index_vt.realize(), ConstantVariable)


def resolve_reindexable(
    tx: "InstructionTranslatorBase",
    records: list[IndexedSubscript],
    literal_elements: "OrderedSet[Source]",
) -> dict[Source, Source]:
    """The subscripts of ``records`` that can be re-derived, as element -> index.

    Drops any index that already carries a value guard -- the region read it
    for something besides subscripting, and the cached body is specialized to
    that read -- and any element that two different indexes both selected, since
    one lifted capture cannot follow two indexes at once.

    The result keeps ``records`` order, which add_reindexing relies on to
    resolve a subscript after whatever it was derived from.
    """
    rejected: set[Source] = {
        record.index_source
        for record in records
        if any(
            guard.create_fn_name() in GUARDS_PINNING_A_VALUE
            for guard in tx.output.guards.get_guards_for_source(record.index_source)
        )
        or realized_to_non_constant(record.index_vt)
    }
    by_element: dict[Source, Source] = {}
    for record in records:
        previous = by_element.setdefault(record.element_source, record.index_source)
        if previous != record.index_source:
            rejected.add(previous)
            rejected.add(record.index_source)
    return {
        element: index
        for element, index in by_element.items()
        if index not in rejected and element not in literal_elements
    }


def add_reindexing(
    source_replacement: dict[Source, Source],
    reindex_nodes: list[tuple[Source, Source]],
    resolve_globals: dict[str, Any],
    resolve_locals: dict[str, Any],
    resolve_cache: dict[Source, Any],
) -> dict[Source, Source] | None:
    """Extend an arg-source replacement with this call's subscript indexes.

    Each recorded subscript node is resolved again -- its index read from the
    current call, its base carried through the replacement built so far -- and
    registered as a further replacement.

    ``reindex_nodes`` is in the order the region performed the subscripts, and
    that order has to be preserved: a subscript can only depend on one the
    region did earlier, either through its base (``groups[gid].buffers[bid]``)
    or through its index (``buffers[ids[gid]]``), because the region had to
    evaluate that part first. Ordering by anything else -- source depth, say --
    would rebuild a subscript before the thing it is derived from moves, and it
    would silently follow the index the *trace* saw rather than this call's.

    Returns None if any index cannot be resolved into a usable subscript, which
    means the call cannot reuse this entry.
    """
    if not reindex_nodes:
        return source_replacement
    augmented = dict(source_replacement)

    def replacement_fn(s: Source) -> Source:
        return augmented.get(s, s)

    for element_source, index_source in reindex_nodes:
        if not isinstance(element_source, GetItemSource) or (
            element_source.index_is_slice
        ):
            return None
        try:
            index = index_source.clone(replacement_fn).get_value(
                resolve_globals, resolve_locals, resolve_cache
            )
        except Exception:
            return None
        if type(index) is not int or index < 0:
            return None
        # Only the base is carried through the replacement. Cloning the whole
        # node would also apply the entry registered for it here, or for
        # another subscript that happens to land on the same element.
        rebased = GetItemSource(
            element_source.base.clone(replacement_fn), element_source.index
        )
        if rebased.index == index:
            continue
        reindexed = GetItemSource(rebased.base, index)
        if augmented.setdefault(rebased, reindexed) != reindexed:
            # Two subscripts want the same node to become different elements.
            return None
    return augmented


def get_fn_code(fn_var: Any) -> types.CodeType | None:
    if isinstance(fn_var, UserFunctionVariable):
        return fn_var.get_function().__code__
    elif isinstance(fn_var, UnspecializedNNModuleVariable):
        return (
            fn_var.value.forward.__func__.__code__  # pyrefly: ignore[missing-attribute]
        )
    return None


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
      - The subgraph must not have side effects.
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
    """
    if not has_reuse_hash_fn:
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
    reindexable: dict[Source, Source] | None = None,
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
    all_sources = set(traced_sources)
    all_sources.update(s for s in fingerprint.arg_sources if s is not None)
    all_relevant_guards: set[Guard] = set()
    for source in all_sources:
        all_relevant_guards.update(tx.output.guards.get_guards_for_source(source))

    # A re-derived index is a parameter of the entry, not part of its condition:
    # its value guard is exactly the one that would reject every other index.
    # Guards that pin something else about it (its type, say) still apply.
    index_sources = set(reindexable.values()) if reindexable else set()

    guard_tuples: list[tuple[Source, GuardCheckSpec, object, Guard]] = []
    for guard in all_relevant_guards:
        source = guard.originating_source
        type_str = guard.create_fn_name()
        if source in index_sources and type_str in GUARDS_PINNING_A_VALUE:
            continue
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


def find_indexed_container(
    cached_entry: InvokeSubgraphReuseEntry, expected: object
) -> str | None:
    """Name of a container a capture was subscripted out of at index ``expected``.

    A subscript like ``pool.buffers[self.layer_id]`` is normally re-derived per
    call, so reaching here means the region read that index for something
    besides subscripting -- see Note: [invoke_subgraph index parameterization]
    -- and so kept the guard that rejects every other index. Naming the
    container lets the reuse log point at what to rewrite.

    Captures reached through an argument's own source are skipped: the region
    argument is already parameterized, so an index inside it is not what is
    blocking reuse.
    """
    # type() rather than isinstance: True == 1 and False == 0, so a bool guard
    # value would match any capture indexed at 0 or 1.
    if type(expected) is not int:
        return None
    arg_sources = {s for s in cached_entry.arg_sources if s is not None}
    for lifted in cached_entry.subgraph_input_mapping:
        if not isinstance(lifted, LiftedCapturedSource):
            continue
        source = lifted.source
        while isinstance(source, ChainedSource):
            if source in arg_sources:
                break
            if (
                isinstance(source, GetItemSource)
                and not source.index_is_slice
                and source.index == expected
            ):
                return _get_source_debug_name(source.base)
            source = source.base
    return None


def is_reusable(
    tx: "InstructionTranslatorBase",
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
            if cur_vt.value != cached_value:
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
                    hc_log.debug(
                        "subgraph_reuse: reuse failed -- input %d constant mismatch "
                        "with no source to replace: cached '%s' vs current '%s'",
                        i,
                        cached_val,
                        cur_vt.value,
                    )
                    return False

    source_replacement = build_source_replacement(
        cached_entry.arg_sources, fingerprint.arg_sources
    )

    # Shared resolution context so source.get_value memoizes intermediate
    # results (e.g. common base sources) across all guards in this check.
    resolve_globals: dict[str, Any] = {
        "G": tx.output.root_tx.f_globals,
        "L": tx.output.root_tx.f_locals,
    }
    resolve_locals: dict[str, Any] = {}
    resolve_cache: dict[Source, Any] = {}

    # Re-derive the subscripts the entry was parameterized over, so the guards
    # below run against the elements *this* call selects rather than the ones
    # the trace saw.
    augmented = add_reindexing(
        source_replacement,
        cached_entry.reindex_nodes,
        resolve_globals,
        resolve_locals,
        resolve_cache,
    )
    if augmented is None:
        hc_log.debug(
            "subgraph_reuse: reuse failed -- cannot re-derive a subscripted capture",
        )
        return False
    source_replacement = augmented

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
            log_details = hc_log.isEnabledFor(logging.DEBUG)
            # Only value-match guards carry the read value itself; a length or
            # type guard's metadata is not an index the region subscripted with.
            container = (
                find_indexed_container(cached_entry, expected)
                if log_details
                and guard.create_fn_name() in ("CONSTANT_MATCH", "EQUALS_MATCH")
                else None
            )
            # Phrased as an observation, not a diagnosis: the match is on the
            # index value, so an unrelated guard that happens to equal an index
            # reaches here too.
            hint = (
                f"\n  hint: a captured value is selected from '{container}' at "
                "this index. If this value is that index, the region also read "
                "it for something other than the subscript -- a branch, an "
                "operand, an argument -- which is what keeps this guard. Read "
                "it once, only to subscript, and the region is re-derived per "
                "index instead of retraced."
                if container
                else ""
            )
            hc_log.debug(
                "subgraph_reuse: reuse failed --\n"
                "  guard type: %s\n"
                "  guard source: %s\n"
                "  guard source name: %s\n"
                "  expected: %s\n"
                "  got: %s\n"
                "  user stack:\n%s%s",
                guard.create_fn_name(),
                new_source,
                new_source.name,
                expected,
                value,
                "".join(guard.user_stack.format())
                if guard.user_stack
                else "<no stack>",
                hint,
            )
            return False

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
    fn_code = get_fn_code(fn_var)
    return fn_code is not None and fn_code in invoke_subgraph_cache.subgraph_reuse_cache


def find_reuse_match(
    tx: "InstructionTranslatorBase",
    fn_var: Any,
    fingerprint: InputFingerprint,
) -> InvokeSubgraphReuseEntry | None:
    from torch._guards import InvokeSubgraphCache

    invoke_subgraph_cache = tx.output.tracing_context.hop_dispatch_set_cache.get_cache(
        torch._higher_order_ops.invoke_subgraph
    )
    if not isinstance(invoke_subgraph_cache, InvokeSubgraphCache):
        return None
    fn_code = get_fn_code(fn_var)
    if fn_code is None:
        return None

    # this evaluator function is called one by one for all the invoke subgraph
    # reuse entries - the one that evaluates to True is stamped out in the
    # graph.
    def evaluator(
        cond: "InvokeSubgraphReuseCondition", entry: InvokeSubgraphReuseEntry
    ) -> bool:
        return is_reusable(tx, cond, fingerprint, entry)

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
    reindexable: dict[Source, Source] | None = None,
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

    fn_code = get_fn_code(fn_var)
    if fn_code is None:
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
        reindex_nodes=list(reindexable.items()) if reindexable else [],
    )
    if condition is not None:
        invoke_subgraph_cache.add_reuse_entry(
            fn_code, condition, entry, max_reuse_entries
        )
    else:
        if hash_key is None:
            raise AssertionError("hash_key must not be None when condition is None")
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
    fn_code = get_fn_code(fn_var)
    if fn_code is None:
        return None
    return invoke_subgraph_cache.find_reuse_entry_by_key(fn_code, hash_key)


def stamp_out_subgraph(
    tx: "InstructionTranslatorBase",
    fingerprint: InputFingerprint,
    cached: InvokeSubgraphReuseEntry,
) -> VariableTracker | None:
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

    # Re-derive the entry's parameterized subscripts for this call. is_reusable
    # already resolved them to accept this call, so a failure here is a bug.
    augmented = add_reindexing(
        source_replacement,
        cached.reindex_nodes,
        resolve_globals,
        resolve_locals,
        resolve_cache,
    )
    if augmented is None:
        # is_reusable resolves these before accepting a call, so the
        # guard-based path does not get here. The reuse_hash_fn path has no
        # such check, and whether an index resolves depends on the calling
        # site rather than on the entry, so there is nothing to reject when
        # the entry is saved. Report a miss and let the caller trace.
        hc_log.debug(
            "subgraph_reuse: stamp out failed -- an index this region "
            "subscripted with does not resolve for this call"
        )
        return None
    source_replacement = augmented

    def replacement_fn(s: Source) -> Source:
        return source_replacement.get(s, s)

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

    # The stamped-out call reads the element each index selects, but never the
    # index itself, so pin it the way the traced region would have. After the
    # args resolve, so a stamp out that gives up partway does not leave the
    # frame specialized on a call it did not serve. OrderedSet since several
    # elements can share one index source (e.g. a K/V cache pair).
    for index_source in OrderedSet(
        index_source for _, index_source in cached.reindex_nodes
    ):
        install_guard(
            index_source.clone(replacement_fn).make_guard(GuardBuilder.CONSTANT_MATCH)
        )

    # Generate fake tensor outputs
    if tx.fake_mode is None:
        raise AssertionError("tx.fake_mode must not be None for stamp_out_subgraph")
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


def build_subgraph_input_mapping(
    tx: "InstructionTranslatorBase",
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
                # _expr rather than expr: expr applies the ShapeEnv's
                # replacements, so a symbol the region lifted before it was
                # specialized comes back as a constant, which bound_symbols has
                # no entry for.
                subgraph_input_mapping.append(LiftedBoundSymbol(example.node._expr))
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
    supports_aliasing = False
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
                    stamped = stamp_out_subgraph(tx, fingerprint, cached)
                if stamped is not None:
                    return stamped

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
                    match.body_name,
                )
                with dynamo_timed("invoke_subgraph_reuse_stamp_out"):
                    stamped = stamp_out_subgraph(tx, fingerprint, match)
                if stamped is not None:
                    return stamped

        if self._HOP_NAME is None:
            raise AssertionError("_HOP_NAME must not be None")
        with (
            dynamo_timed("invoke_subgraph_trace"),
            open_index_parameterized_region(tx) as index_region,
        ):
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
            if reuse_hash_fn is not None:
                traced_sources = tracing_info.traced_sources
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
                    # A hash key says the body is interchangeable across calls,
                    # not that the captures are. Without this a key that
                    # deliberately ignores the layer id -- the reason to reach
                    # for reuse_hash_fn on a KV cache -- would pin every layer
                    # to the first one's slot.
                    reindexable=index_region.reindexable,
                )
            else:
                traced_sources = tracing_info.traced_sources
                if is_reuse_eligible(
                    tx, body_r, fingerprint, tracing_info, traced_sources
                ):
                    reindexable = index_region.reindexable
                    condition = build_reuse_condition(
                        tx,
                        fingerprint,
                        traced_sources,
                        reindexable,
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
                            max_reuse_entries,
                            condition=condition,
                            reindexable=reindexable,
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
