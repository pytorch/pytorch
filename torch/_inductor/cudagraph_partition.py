# mypy: allow-untyped-defs
"""AOTI regional cuda-graph partitioning helpers.

Pure helpers extracted from scheduler.py and codegen/memory_planning.py: node
classification for cuda-graph eligibility, agglomerative convex clustering of the
scheduler node order, and cross-partition handoff-slab planning. Kept free of
scheduler/memory_planning instance state so both modules can import this module
at top level; the few scheduler- and memory_planning-local classes needed at
runtime are imported lazily inside the functions to avoid an import cycle.
"""

from __future__ import annotations

import heapq
from typing import Any, TYPE_CHECKING

import sympy
import torch
from torch.utils._ordered_set import OrderedSet

from . import config
from .virtualized import V


if TYPE_CHECKING:
    from .codegen.memory_planning import Allocation, AllocationPool
    from .codegen.wrapper import BufferLike
    from .scheduler import BaseSchedulerNode, SchedulerBuffer


# Minimum number of kernels in a cudagraph-eligible partition. Capturing a tiny
# partition as its own cuda graph costs per-replay staging + launch overhead that
# outweighs the kernel-launch saving when there are only a few kernels to amortize
# it over, so partitions below this size are demoted to eager.
MIN_CG_SIZE = 4


def _cudagraph_shared_storage_names(
    nodes: list[BaseSchedulerNode],
) -> OrderedSet[str]:
    """Names of all buffers whose storage is shared via aliasing or mutation,
    in BOTH directions: every view/ReinterpretView buffer AND every base buffer
    it points into. Capturing only part of such a storage group inside a cuda
    graph splits a concat/chunk across the capture boundary (see
    _node_touches_shared_storage). Computed over `nodes`; the caller caches the
    result."""
    names: OrderedSet[str] = OrderedSet()
    for n in nodes:
        for buf in n.get_outputs():
            aliases = buf.get_aliases()
            mutations = buf.get_mutations()
            if aliases or mutations:
                names.add(buf.get_name())
                names.update(aliases)
                names.update(mutations)
    return names


def _node_touches_shared_storage(
    node: BaseSchedulerNode, shared_names: OrderedSet[str]
) -> bool:
    """True if any output of node is part of a shared-storage group (see
    _cudagraph_shared_storage_names). Catches both a view buffer and the base
    buffer that views point into."""
    from .scheduler import FusedSchedulerNode

    if isinstance(node, FusedSchedulerNode):
        return any(
            _node_touches_shared_storage(snode, shared_names) for snode in node.snodes
        )
    return any(buf.get_name() in shared_names for buf in node.get_outputs())


def _node_uses_broadcast_gather(node: BaseSchedulerNode) -> bool:
    """True if the node performs an index_select / advanced-index gather.
    In ads (ROCS) models these implement user->ad broadcasting: a gather of
    user-level rows by a per-request index tensor (batch_indices/rev_indices).
    They are not safe inside an AOTI regional cuda graph -- the gather is the
    batch-dim broadcast boundary and is request-structure dependent, so it
    must stay eager (matches _is_batch_dim_transition, but also catches gathers
    fused into otherwise single-symbol s13 kernels where dim0 does not change).
    Detect through FX origins: match the aten op of the origin target or its
    meta['original_aten'] (the gather may be fused/lowered, but the source op
    is preserved in origins)."""
    from .scheduler import FusedSchedulerNode

    if isinstance(node, FusedSchedulerNode):
        return any(_node_uses_broadcast_gather(snode) for snode in node.snodes)
    ir_node = node.node
    if ir_node is None:
        return False
    origins = getattr(ir_node, "origins", None)
    if not origins:
        return False
    aten = torch.ops.aten
    gather_packets: OrderedSet[object] = OrderedSet()
    for name in ("index_select", "index", "_unsafe_index"):
        op = getattr(aten, name, None)
        if op is not None:
            gather_packets.add(op)

    def _packet(op: object) -> object:
        return op.overloadpacket if isinstance(op, torch._ops.OpOverload) else op

    for fx_node in origins:
        target = getattr(fx_node, "target", None)
        if target is not None and _packet(target) in gather_packets:
            return True
        meta = getattr(fx_node, "meta", None)
        if meta is not None:
            orig = meta.get("original_aten")
            if orig is not None and _packet(orig) in gather_packets:
                return True
    return False


def _node_uses_rng(node: BaseSchedulerNode) -> bool:
    """True if the node involves RNG (random draw or seed generation). Such
    nodes are unsafe inside an AOTI regional cuda graph: capture freezes the
    philox seed+offset, so every replay reproduces identical draws instead of
    advancing, diverging from the eager reference. (cuda graph trees handle
    this via capturable generator state registered on the graph; AOTI has no
    equivalent, so we keep RNG nodes eager.) Detect through FX origins: aten
    RNG ops carry Tag.nondeterministic_seeded, and Inductor lowers them to
    inductor_prims.{random,randint,seed,seeds,lookup_seed} (see
    fx_passes/replace_random.py)."""
    from .scheduler import FusedSchedulerNode

    if isinstance(node, FusedSchedulerNode):
        return any(_node_uses_rng(snode) for snode in node.snodes)
    ir_node = node.node
    if ir_node is None:
        return False
    origins = getattr(ir_node, "origins", None)
    if not origins:
        return False
    # When fallback_random is False (default), replace_random rewrites aten
    # RNG into these inductor_prims ops, which become the FX targets (and the
    # IR origins). seed/seeds also carry nondeterministic_seeded; random/
    # randint/lookup_seed do not, so match the op objects directly. The tag
    # check additionally covers aten RNG kept as fallbacks.
    from torch._inductor import inductor_prims

    rng_ops: OrderedSet[object] = OrderedSet()
    for name in ("seed", "seeds", "lookup_seed", "random", "randint"):
        op = getattr(inductor_prims, name, None)
        if op is not None:
            rng_ops.add(op)
    for fx_node in origins:
        target = getattr(fx_node, "target", None)
        if target is None:
            continue
        if target in rng_ops:
            return True
        if isinstance(target, torch._ops.OpOverload):
            if torch.Tag.nondeterministic_seeded in target.tags:
                return True
            if target.overloadpacket in rng_ops:
                return True
    return False


def _get_node_dynamic_symbols(
    node: BaseSchedulerNode,
) -> OrderedSet[sympy.Symbol]:
    """Return the set of free dynamic symbols in this node's output shapes."""
    from .scheduler import FusedSchedulerNode

    result: OrderedSet[sympy.Symbol] = OrderedSet()
    if isinstance(node, FusedSchedulerNode):
        for snode in node.snodes:
            result.update(_get_node_dynamic_symbols(snode))
        return result
    ir_node = node.node
    if ir_node is None:
        return result
    for out in ir_node.get_outputs():
        try:
            layout = getattr(out, "layout", None)
            if layout is None and hasattr(out, "get_layout"):
                layout = out.get_layout()
            if layout and hasattr(layout, "size"):
                for d in layout.size:
                    if isinstance(d, sympy.Expr) and not d.is_number:
                        result.update(d.free_symbols)
        except NotImplementedError:
            pass
    return result


def _get_output_dim0_symbol(
    node: BaseSchedulerNode,
) -> sympy.Expr | int | None:
    """
    Return the dim0 expression for this node's outputs.
    - A sympy.Symbol if dim0 is dynamic (e.g. s0 for user batch, s1 for ads batch)
    - An int if dim0 is static and > 1
    - None if unknown or scalar (dim0 == 1)
    """
    from .scheduler import FusedSchedulerNode

    if isinstance(node, FusedSchedulerNode):
        for snode in node.snodes:
            r = _get_output_dim0_symbol(snode)
            if r is not None:
                return r
        return None
    ir_node = node.node
    if ir_node is None:
        return None
    for out in ir_node.get_outputs():
        try:
            layout = getattr(out, "layout", None)
            if layout is None and hasattr(out, "get_layout"):
                layout = out.get_layout()
            if layout and hasattr(layout, "size") and len(layout.size) > 0:
                d = layout.size[0]
                if isinstance(d, sympy.Expr) and not d.is_number:
                    syms = d.free_symbols
                    if len(syms) == 1:
                        return next(iter(syms))
                    return d
                if isinstance(d, (int, sympy.Integer)) and int(d) > 1:
                    return int(d)
        except NotImplementedError:
            pass
    return None


def _is_batch_dim_transition(node: BaseSchedulerNode) -> str | None:
    """
    Check if this node is a broadcast/gather boundary: it reads from
    inputs with one batch dimension and writes outputs with a *different*
    batch dimension. Covers both dynamic->static (user batch -> fixed) and
    dynamic->different-dynamic (user batch s0 -> ads batch s1) transitions.
    """
    from .scheduler import FusedSchedulerNode

    if isinstance(node, FusedSchedulerNode):
        for snode in node.snodes:
            if reason := _is_batch_dim_transition(snode):
                return reason
        return None

    ir_node = node.node
    if ir_node is None:
        return None

    def _get_dim0(layout: object) -> sympy.Expr | int | None:
        if layout and hasattr(layout, "size") and len(layout.size) > 0:
            return layout.size[0]
        return None

    out_symbols: set[sympy.Symbol] = set()
    out_statics: set[int] = set()
    for out in ir_node.get_outputs():
        try:
            layout = getattr(out, "layout", None)
            if layout is None and hasattr(out, "get_layout"):
                layout = out.get_layout()
            d = _get_dim0(layout)
            if d is None:
                continue
            if isinstance(d, sympy.Expr) and not d.is_number:
                out_symbols.update(d.free_symbols)
            elif isinstance(d, (int, sympy.Integer)) and int(d) > 1:
                out_statics.add(int(d))
        except NotImplementedError:
            pass

    if not out_symbols and not out_statics:
        return None

    input_symbols: set[sympy.Symbol] = set()
    dep_names = {dep.name for dep in node.read_writes.reads}
    for dep_name in dep_names:
        try:
            sz = None
            if dep_name in V.graph.graph_inputs:
                inp = V.graph.graph_inputs[dep_name]
                if hasattr(inp, "get_size"):
                    sz = inp.get_size()
            else:
                buf = V.graph.get_buffer(dep_name)
                if buf is not None and hasattr(buf, "get_size"):
                    sz = buf.get_size()
            if sz and isinstance(sz[0], sympy.Expr) and not sz[0].is_number:
                input_symbols.update(sz[0].free_symbols)
        except (NotImplementedError, AttributeError):
            pass

    if not input_symbols:
        return None

    if out_statics and not out_symbols:
        return (
            f"batch dim transition (reads dynamic {input_symbols}, writes static dim0)"
        )

    if out_symbols and out_statics:
        return f"batch dim transition (mixed dynamic {out_symbols} and static outputs)"

    if out_symbols and input_symbols and out_symbols != input_symbols:
        return f"batch dim transition (reads {input_symbols}, writes {out_symbols})"

    return None


def _get_node_input_dynamic_symbols(
    node: BaseSchedulerNode, name_to_buf: dict[str, SchedulerBuffer]
) -> OrderedSet[sympy.Symbol]:
    """Free dynamic symbols in this node's READ buffers' layout sizes.
    Mirrors _get_node_dynamic_symbols but over inputs, so eligibility can
    union both: a node reading [s13,s14] and writing [s13] otherwise slips
    through (output={s13}) and a second dynamic symbol leaks into the
    captured partition via a non-dim0 input."""
    from .scheduler import FusedSchedulerNode

    result: OrderedSet[sympy.Symbol] = OrderedSet()
    if isinstance(node, FusedSchedulerNode):
        for snode in node.snodes:
            result.update(_get_node_input_dynamic_symbols(snode, name_to_buf))
        return result
    for dep in node.read_writes.reads:
        buf = name_to_buf.get(dep.name)
        if buf is None or buf.node is None:
            continue
        try:
            layout = getattr(buf.node, "layout", None)
            if layout is None and hasattr(buf.node, "get_layout"):
                layout = buf.node.get_layout()
            if layout and hasattr(layout, "size"):
                for d in layout.size:
                    if isinstance(d, sympy.Expr) and not d.is_number:
                        result.update(d.free_symbols)
        except NotImplementedError:
            pass
    return result


def _partition_io_dynamic_symbols(
    node: BaseSchedulerNode, name_to_buf: dict[str, SchedulerBuffer]
) -> OrderedSet[sympy.Symbol]:
    """Union of free dynamic symbols across this node's INPUTS and OUTPUTS,
    simplified. Used for cuda-graph eligibility so a node reading
    [s13, s14] and writing [s13] is correctly counted as 2-symbol -- the
    output-only check misses it and a second dynamic symbol can leak into the
    captured partition via a non-dim0 input, causing wrong replays."""
    syms: OrderedSet[sympy.Symbol] = OrderedSet()
    combined = _get_node_dynamic_symbols(node) | _get_node_input_dynamic_symbols(
        node, name_to_buf
    )
    for s in combined:
        simplified = V.graph.sizevars.simplify(s)
        if simplified.free_symbols:
            syms.update(simplified.free_symbols)
    return syms


def _agglomerative_convex_clusters(
    n: int,
    succ: list[list[int]],
    ancestors: list[set[int]],
    eligible: list[bool],
    sym: list[str | None],
) -> list[int]:
    """Cuda graph partitioning step 2: iteratively merge eligible neighbor
    subgraphs when (a) combined dynamic symbols <= 1 and (b) the merge keeps
    the partition graph acyclic (convex). Ineligible nodes are never merged.
    Returns a cluster id (a representative node index) per node. Nodes are
    indexed 0..n-1 in a topological order; `ancestors[i]` are i's transitive
    ancestor indices."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Per-cluster (keyed by rep): dynamic-symbol set and ancestor-cluster set.
    csyms: list[set[str]] = [
        set() if sym[i] is None else {sym[i]}
        for i in range(n)  # type: ignore[arg-type]
    ]
    canc: list[set[int]] = [set(ancestors[i]) - {i} for i in range(n)]

    def try_merge(ru: int, rv: int) -> bool:
        # Edge u->v: rv depends on ru, so ru is an ancestor of rv.
        if ru == rv or not (eligible[ru] and eligible[rv]):
            return False
        if len(csyms[ru] | csyms[rv]) > 1:
            return False
        # Convex iff no third cluster z strictly between ru and rv.
        for z in canc[rv]:
            if z != ru and z != rv and ru in canc[z]:
                return False
        parent[rv] = ru
        csyms[ru] |= csyms[rv]
        canc[ru] |= canc[rv]
        canc[ru].discard(ru)
        canc[ru].discard(rv)
        for z in range(n):
            if parent[z] == z and rv in canc[z]:
                canc[z].discard(rv)
                canc[z].add(ru)
        return True

    changed = True
    while changed:
        changed = False
        for u in range(n):
            ru = find(u)
            for v in succ[u]:
                if try_merge(ru, find(v)):
                    changed = True
                    ru = find(u)

    return [find(i) for i in range(n)]


def _linearize_clusters(n: int, cluster: list[int], succ: list[list[int]]) -> list[int]:
    """Cluster-level topological sort; nodes within a cluster keep their
    original relative order. Returns a node-index order where each cluster is
    contiguous and all dependencies are respected (valid because clusters are
    convex)."""
    members: dict[int, list[int]] = {}
    for i in range(n):
        members.setdefault(cluster[i], []).append(i)
    cedges: dict[int, OrderedSet[int]] = {r: OrderedSet() for r in members}
    cindeg: dict[int, int] = {r: 0 for r in members}
    for u in range(n):
        cu = cluster[u]
        for v in succ[u]:
            cv = cluster[v]
            if cu != cv and cv not in cedges[cu]:
                cedges[cu].add(cv)
                cindeg[cv] += 1
    minidx = {r: members[r][0] for r in members}
    ready = [(minidx[r], r) for r in members if cindeg[r] == 0]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        _, r = heapq.heappop(ready)
        order.extend(members[r])
        for cv in cedges[r]:
            cindeg[cv] -= 1
            if cindeg[cv] == 0:
                heapq.heappush(ready, (minidx[cv], cv))
    assert len(order) == n, "cluster graph not a DAG (linearize failed)"
    return order


def _resolve_handoff_buffer(name: str) -> BufferLike | None:
    """Return the underlying ir.Buffer for a handoff name, or None.

    V.graph.try_get_buffer may return a TensorBox/StorageBox (MutableBox)
    wrapping the Buffer; peel the box so the synthetic Allocation reads the
    real layout. WorkspaceArg / TorchBindObject have no tensor layout and
    are not valid handoff buffers, so they resolve to None.
    """
    from . import ir

    node = V.graph.try_get_buffer(name)
    while isinstance(node, ir.MutableBox):
        node = node.data
    if isinstance(node, ir.Buffer):
        return node
    return None


def _pack_handoff_pool(
    pool_name: str,
    named_allocations: list[tuple[str, Allocation]],
    pool_id: int,
) -> AllocationPool:
    """Pack one per-symbol group of handoff allocations into a single pool.

    Seeds the pool with the first block, then allocates the rest with
    is_last=True so a block that fits no existing slot grows the pool at the
    end; allocate_at_end is the fallback when allocate() refuses a block
    whose earliest_available (unbacked-symint shapes) is past the pool
    begin. finalize() assigns each Allocation its byte offset. pool_id is the
    reserved cudagraph-slab-cache id for this pool.
    """
    from .codegen.memory_planning import AllocationPool, TemporalSplit

    first_allocation = named_allocations[0][1]
    first_allocation.mark_allocated()
    pool = AllocationPool(
        first_allocation.device,
        TemporalSplit([first_allocation]),
        can_expand=config.memory_pool != "none",
        pool_id=pool_id,
    )
    for _name, allocation in named_allocations[1:]:
        if not pool.allocate(allocation, is_last=True):
            pool.allocate_at_end(allocation)
    pool.finalize(pool_name)
    return pool


def _cudagraph_handoff_slab_planning_enabled() -> bool:
    """Wrapper-free gate for the handoff slab planning, equivalent to
    _cudagraph_slab_cache_enabled but evaluated without a wrapper instance so it
    can run in the scheduler pre-pass (Phase 1).

    AOTI cpp-wrapper + GPU target, with cuda-graph + memory_planning on. Because
    this runs once on the outer graph in the scheduler pre-pass (NOT per subgraph
    wrapper), the original "only the outer wrapper" guard (subgraph_name is None)
    is satisfied implicitly. Off (any condition false) -> early return, leaving
    non-cuda-graph behavior unchanged.
    """
    from .utils import is_gpu

    if not config.aot_inductor.enable_cuda_graph:
        return False
    if not config.memory_planning:
        return False
    if not (V.graph.aot_mode and V.graph.cpp_wrapper):
        return False
    return is_gpu(V.graph.device_type)


def plan_cudagraph_handoff_slab() -> None:
    """Hoist captured-partition handoff buffers into OUTER-owned, PER-SYMBOL slab
    regions.

    Memory planning is per-partition: each captured partition is codegen'd
    through its own subgraph wrapper with its own MemoryPlanner -> its own
    AllocationPools, so pool names restart per partition (pool0 in every
    partition). The outer wrapper sees a partition only as one
    codegen_partition_call line, so it never builds an Allocation for any
    cross-partition handoff buffer (a partition output consumed by a LATER
    partition). Those handoff buffers therefore live in the per-partition
    runtime pool, not the slab.

    PHASE ORDERING (why this is a scheduler pre-pass, not a wrapper plan pass):
    the two readers of the dicts published here both run in Phase 1
    (Scheduler.codegen): (a) the per-partition body's AllocFromPoolLine.codegen
    reads _cudagraph_boundary_offsets, and (b) the outer codegen_partition_call
    reads _cudagraph_handoff_pools. The outer wrapper's MemoryPlanner.plan runs
    in Phase 2 (wrapper_code.generate, after Scheduler.codegen). So this MUST be
    called from the scheduler pre-pass, before the partition codegen loop, or the
    dicts would still be empty when both readers run.

    This pass runs ONCE on the outer graph and builds dedicated AllocationPools
    for exactly the buffers in V.graph._cudagraph_handoff_liveness (populated by
    the scheduler pre-pass). These buffers have NO outer AllocateLine, so we
    construct synthetic BufferGroup/Allocation objects directly from each
    buffer's ir.Buffer layout (size/stride/dtype live on V.graph, independent of
    which partition emits the buffer).

    PER-SYMBOL POOLS (the key ABI decision): a handoff's byte offset is a sum of
    aligned sizes of the OTHER handoffs packed before it in the same pool. The
    producer partition body emits alloc_from_pool(pool, offset, ...) inline, and
    a partition is restricted to <=1 dynamic symbol in its C++ scope. So we group
    handoffs by the single dynamic symbol of their symbolic nbytes and give each
    group its own pool: every offset in a pool then references only that pool's
    one symbol, which is exactly the symbol live in any producing partition's
    body. Static-size handoffs (no dynamic symbol) go to a separate static pool
    (int offsets). A handoff whose nbytes carries >=2 dynamic symbols cannot be
    made single-symbol-safe and is EXCLUDED (left on the runtime copy_in path ==
    current behavior).

    Within each pool, offsets are assigned by pid-interval overlap, NOT by
    timestep: each handoff's live_range is its [first_producer_pid ..
    last_consumer_pid] captured-partition execution interval, so two handoffs
    share a slot iff their pid intervals are disjoint. There is no global
    timestep axis (per-partition planners restart timestep at 0), so we bypass
    compute_live_ranges and feed the pid-interval live_range straight into the
    existing AllocationPool / TemporalSplit / SpatialSplit packer; pool.finalize()
    then assigns each Allocation a byte offset via Allocation.finalize(pool,
    offset).

    The result is exposed as
    V.graph._cudagraph_boundary_offsets[name] = (per_symbol_pool_name, offset),
    where per_symbol_pool_name is the single stable pool name for that symbol
    (resolving the per-partition pool0 ambiguity) and offset is the byte offset
    into that pool (int for static handoffs, a sympy.Expr in that pool's single
    symbol otherwise).

    Gated (via _cudagraph_handoff_slab_planning_enabled) on AOTI cpp-wrapper +
    GPU + cuda-graph + memory_planning; when off this function early-returns and
    leaves non-cuda-graph behavior unchanged.
    """
    if not _cudagraph_handoff_slab_planning_enabled():
        return

    from .codegen.memory_planning import BufferGroup, LiveRange

    handoff_liveness: dict[str, tuple[int, int]] = (
        getattr(V.graph, "_cudagraph_handoff_liveness", None) or {}
    )
    if not handoff_liveness:
        V.graph._cudagraph_boundary_offsets = {}
        return

    # Group handoffs by the single dynamic symbol of their symbolic nbytes.
    # sym_key is "" for static (no dynamic symbol) -> the static pool.
    groups_by_sym: dict[str, list[tuple[str, Allocation]]] = {}
    for name, (producer_pid, last_consumer_pid) in handoff_liveness.items():
        node = _resolve_handoff_buffer(name)
        if node is None:
            # No materialized ir.Buffer (e.g. unexpected alias/view). Skip
            # slab placement; the codegen side keeps such a buffer on the
            # existing copy_in path.
            continue
        group = BufferGroup(node)
        syms = sorted(
            (s for s in group.sym_nbytes().free_symbols if isinstance(s, sympy.Symbol)),
            key=str,
        )
        if len(syms) >= 2:
            # Offsets in a shared pool would reference >1 symbol, but a
            # producer partition body has only its single symbol in scope.
            # Cannot place safely -> fall back to copy_in (current behavior).
            continue
        sym_key = str(syms[0]) if syms else ""
        # pid interval [producer .. last_consumer], both inclusive. LiveRange
        # end is exclusive, so +1 makes a handoff consumed at pid K and one
        # produced at pid K correctly overlap (both live during pid K's run)
        # and thus get disjoint slots, while truly disjoint intervals share.
        group.live_range = LiveRange(producer_pid, last_consumer_pid + 1)
        group.make_allocation()
        if group.allocation is None:
            raise AssertionError("handoff group has no allocation")
        groups_by_sym.setdefault(sym_key, []).append((name, group.allocation))

    boundary_offsets: dict[str, tuple[str, Any]] = {}
    # Retain the finalized pools so the OUTER cpp wrapper can emit each
    # handoff slab once (via AllocationPool.codegen_create, reusing the
    # per-instance cached-slab path). Keyed by stable pool name.
    handoff_pools: dict[str, AllocationPool] = {}
    # Reserved cache-key id base for handoff pools. The slab-cache key packs the
    # pool id into the high bits as ((int64_t)pool_id << 40) (see
    # _codegen_create_cudagraph_cached and cudagraph_tree.h encode()). With a
    # signed int64 key, the base must satisfy (base << 40) < 2**63: 1<<22 gives
    # (1<<22) << 40 == 2**62, in range and positive. (The earlier 1<<30 made
    # (1<<30) << 40 == 2**70, which overflows int64 and aliases to 0 -- handoff
    # pool index i would then collide with the ordinary pool{i} slab.) The base
    # also stays above the legacy hash-bucket range 1<<20 used for non-poolN
    # names, so handoff slabs never alias a per-partition pool0 slab in
    # this->cudagraph_slabs_.
    handoff_pool_id_base = 1 << 22
    for pool_index, (sym_key, named_allocations) in enumerate(
        sorted(groups_by_sym.items())
    ):
        # Keep base+index < 2**23 so (pool_id << 40) < 2**63 stays in signed
        # int64 range and distinct from the small ordinary pool ids 0..N and the
        # legacy 1<<20 hash bucket. 1<<17 distinct handoff pools is far beyond the
        # handful of dynamic symbols a model has.
        if pool_index >= (1 << 17):
            raise AssertionError(
                f"too many cudagraph handoff pools ({pool_index + 1}); "
                "pool id would exceed the reserved int64 key range"
            )
        pool_name = (
            f"cudagraph_handoff_pool_{sym_key}"
            if sym_key
            else "cudagraph_handoff_pool_static"
        )
        pool = _pack_handoff_pool(
            pool_name, named_allocations, handoff_pool_id_base + pool_index
        )
        # Compare by symbol NAME, not Symbol object: Inductor size symbols
        # carry assumptions (positive/integer) so a bare sympy.Symbol(name)
        # would not equal them and would spuriously fail the subset check.
        allowed = {sym_key} if sym_key else set()
        placed_any = False
        for name, allocation in named_allocations:
            if allocation.offset is None:
                raise AssertionError("handoff allocation not finalized")
            offset_syms = set()
            if isinstance(allocation.offset, sympy.Expr):
                offset_syms = {
                    str(s)
                    for s in allocation.offset.free_symbols
                    if isinstance(s, sympy.Symbol)
                }
            if not offset_syms.issubset(allowed):
                # Defensive: packing produced a cross-symbol offset despite
                # single-symbol grouping. Exclude rather than emit an
                # out-of-scope symbol in the producer body.
                continue
            boundary_offsets[name] = (pool_name, allocation.offset)
            placed_any = True
        if placed_any:
            handoff_pools[pool_name] = pool

    V.graph._cudagraph_boundary_offsets = boundary_offsets
    V.graph._cudagraph_handoff_pools = handoff_pools
