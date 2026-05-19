"""
Decompose collective communication patterns into mathematically equivalent
local computation + lighter collectives.

Provides reusable utilities for:
- Detecting Gram matrices and other decomposable patterns in FX graphs
- Tracing data provenance backward through collectives
- Finding shard split/getitem patterns downstream

Current passes:
- decomp_gram_matrix_all_gather:
    all_gather + mm(X.T, X) -> local mm(Xi.T, Xi) + all_reduce(sum)
    Exploits the Gram matrix identity X.T @ X = sum_i(Xi.T @ Xi).
    Applicable to Muon (Newton-Schulz), Shampoo/K-FAC (Kronecker factors),
    and other optimizers with self-product patterns.
- batch_all_reduces:
    N independent all_reduce(sum) -> flatten + cat + 1 all_reduce + split + reshape
    Reduces NCCL launch overhead for per-weight reductions.
    Works for any tensor shapes (scalars, vectors, matrices).

Gated by config.aten_distributed_optimizations.allow_comms_decompositions.
"""

import logging
import operator
from collections import defaultdict
from typing import NamedTuple

import torch
import torch.fx as fx
from torch._inductor.fx_passes.bucketing import get_collective_type, is_wait_tensor
from torch._inductor.pattern_matcher import CallFunction, Ignored, KeywordArg
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.utils._ordered_set import OrderedSet


aten = torch.ops.aten
c10d = torch.ops._c10d_functional
logger = logging.getLogger(__name__)


def _get_fake_mode(graph: fx.Graph) -> FakeTensorMode | None:
    """Extract FakeTensorMode from any FakeTensor in the graph's metadata."""
    for n in graph.nodes:
        val = n.meta.get("val")
        if isinstance(val, FakeTensor):
            return val.fake_mode
        if isinstance(val, (tuple, list)):
            for v in val:
                if isinstance(v, FakeTensor):
                    return v.fake_mode
    return None


def _retrace_node_meta(node: fx.Node) -> None:
    """Compute node.meta["val"] by executing the op on input FakeTensors.

    Extracts FakeTensor values from input nodes' metadata, runs the
    target op under the graph's FakeTensorMode, and stores the result.
    """
    if node.op != "call_function":
        return

    def _resolve(x):  # type: ignore[no-untyped-def]
        if isinstance(x, fx.Node):
            return x.meta.get("val")
        if isinstance(x, list):
            return [_resolve(v) for v in x]
        if isinstance(x, tuple):
            return tuple(_resolve(v) for v in x)
        return x

    args = _resolve(node.args)
    kwargs = {k: _resolve(v) for k, v in node.kwargs.items()}

    fake_mode = _get_fake_mode(node.graph)
    try:
        target = node.target
        assert callable(target)
        if fake_mode is not None:
            with fake_mode:
                node.meta["val"] = target(*args, **kwargs)
        else:
            node.meta["val"] = target(*args, **kwargs)
    except Exception as e:
        logger.debug("_retrace_node_meta: failed for %s: %s", node.name, e)


def _is_collective_or_wait(node: fx.Node) -> bool:
    """True if node is any distributed collective or wait_tensor."""
    return get_collective_type(node) != "" or is_wait_tensor(node)


def _is_2d_transpose(node: fx.Node) -> bool:
    """True if node is a 2D matrix transpose.

    Matches:
    - aten.t(X)
    - permute(X, [1, 0])
    - transpose(X, 0, 1) / transpose(X, -2, -1)
    """
    if node.op != "call_function":
        return False
    if node.target is aten.t.default:
        return True
    if node.target is aten.permute.default:
        dims = node.args[1] if len(node.args) > 1 else None
        return isinstance(dims, (list, tuple)) and list(dims) == [1, 0]
    if node.target is aten.transpose.int:
        if len(node.args) >= 3:
            d0, d1 = node.args[1], node.args[2]
            return (d0, d1) in ((0, 1), (1, 0), (-2, -1), (-1, -2))
    return False


def is_gram_mm(node: fx.Node) -> bool:
    """True if node computes mm(X, X.T) or mm(X.T, X).

    Matches transpose via permute([1,0]) or transpose(-2,-1).
    A Gram matrix is a self-product of a matrix with its transpose.
    This pattern appears in Newton-Schulz iterations (Muon optimizer),
    Kronecker-factored preconditioners (Shampoo, K-FAC), and other
    second-order methods.
    """
    if node.op != "call_function" or node.target is not aten.mm.default:
        return False
    if len(node.args) != 2:
        return False
    a = node.args[0]
    b = node.args[1]
    if not isinstance(a, fx.Node) or not isinstance(b, fx.Node):
        return False
    if _is_2d_transpose(a) and a.args[0] is b:
        return True
    if _is_2d_transpose(b) and b.args[0] is a:
        return True
    return False


def find_gram_mms(graph: fx.Graph) -> list[fx.Node]:
    """Find all Gram matrix mm nodes in the graph."""
    return [n for n in graph.nodes if is_gram_mm(n)]


def gram_source(gram_node: fx.Node) -> fx.Node:
    """Return the base tensor X from a Gram mm(X, X.T) or mm(X.T, X).

    For mm(permute(X), X) returns X (args[1]).
    For mm(X, permute(X)) returns X (args[0]).
    """
    a = gram_node.args[0]
    b = gram_node.args[1]
    assert isinstance(a, fx.Node) and isinstance(b, fx.Node)
    if _is_2d_transpose(a):
        return b
    return a


_all_gather_wait_slice_pattern = CallFunction(
    aten.slice.Tensor,
    CallFunction(
        c10d.wait_tensor.default,
        CallFunction(
            c10d.all_gather_into_tensor.default,
            KeywordArg("shard"),
            KeywordArg("world_size"),
            KeywordArg("group_name"),
        ),
    ),
    Ignored(),
    Ignored(),
    Ignored(),
)


class AllGatherInfo(NamedTuple):
    shard: fx.Node
    group_name: str
    ag_node: fx.Node
    wait_node: fx.Node
    slice_node: fx.Node


def find_all_gather_ancestor(
    node: fx.Node, max_depth: int = 30
) -> AllGatherInfo | None:
    """Walk backward from node to find all_gather -> wait -> slice chain.

    BFS through input nodes looking for a slice that matches the
    all_gather -> wait_tensor -> slice pattern. Stops at placeholders
    and respects max_depth to bound the search.
    """
    visited: OrderedSet[fx.Node] = OrderedSet()
    queue = [(node, 0)]
    while queue:
        n, depth = queue.pop(0)
        if n in visited or depth > max_depth or n.op == "placeholder":
            continue
        visited.add(n)
        if n.op == "call_function" and n.target is aten.slice.Tensor:
            match = _all_gather_wait_slice_pattern.match(n)
            if match:
                wait_node = n.args[0]
                assert isinstance(wait_node, fx.Node)
                ag_node = wait_node.args[0]
                assert isinstance(ag_node, fx.Node)
                return AllGatherInfo(
                    shard=match.kwargs["shard"],
                    group_name=match.kwargs["group_name"],
                    ag_node=ag_node,
                    wait_node=wait_node,
                    slice_node=n,
                )
        for inp in n.all_input_nodes:
            queue.append((inp, depth + 1))
    return None


def find_split_getitem(
    start: fx.Node, max_search: int = 3000
) -> tuple[fx.Node, fx.Node] | None:
    """Walk forward from start to find split(dim=0) -> getitem in the dependent chain.

    Tracks all nodes transitively dependent on start. Returns the first
    (split_node, getitem_node) pair found, or None.
    Rejects chains that contain collectives (would break distributed semantics).
    Stops after max_search nodes to bound cost on large graphs.
    """
    chain: OrderedSet[fx.Node] = OrderedSet([start])
    searched = 0
    node = start.next
    while node is not None and searched < max_search:
        searched += 1
        if node.op != "call_function":
            node = node.next
            continue

        if not any(inp in chain for inp in node.all_input_nodes):
            node = node.next
            continue

        if node.target is aten.split.Tensor:
            split_dim = node.args[2] if len(node.args) > 2 else 0
            if split_dim != 0:
                return None
            getitems = [
                u
                for u in node.users
                if u.op == "call_function" and u.target is operator.getitem
            ]
            if len(getitems) != 1:
                return None
            return node, getitems[0]

        if _is_collective_or_wait(node):
            return None

        chain.add(node)
        node = node.next

    return None


def decomp_gram_matrix_all_gather(gm: fx.GraphModule) -> fx.GraphModule:
    """Eliminate all_gather when the gathered tensor feeds a Gram matrix computation.

    Many distributed optimizers (Muon, Shampoo, K-FAC) follow this pattern:

        X = all_gather(X_shard)          # (S*W, K) — expensive collective
        G = X.T @ X                      # (K, K)   — Gram matrix
        ... polynomial f(G) ...          # (K, K)   — e.g. Newton-Schulz
        Y = f(G) @ X                     # (S*W, K) — update
        Y_shard = split(Y)[rank]         # (S, K)   — back to shard

    The all_gather is unnecessary because the Gram matrix decomposes:

        X.T @ X = [X0; X1; ...; Xw].T @ [X0; X1; ...; Xw]
                = X0.T@X0 + X1.T@X1 + ... + Xw.T@Xw

    Each rank computes its local partial Gram Xi.T @ Xi on its shard, then
    a single all_reduce(sum) produces the exact global Gram matrix. The
    downstream polynomial f(G) operates on the (K, K) Gram which is now
    globally correct and identical on every rank. The final update step
    f(G) @ X distributes over row slicing: f(G) @ X_shard equals rank's
    slice of f(G) @ X_gathered, so no further communication is needed.

    Net effect: replaces one all_gather (O(S*W*K) bytes) with one or more
    all_reduce (O(K*K) bytes each). For typical optimizer shapes where
    S*W >> K, this is a major reduction in both communication volume and
    compute (shard-sized matmul instead of gathered-sized).

    Benchmark: 2.1x speedup on 8xH100 Muon optimizer (48ms -> 22ms).

    Graph before (Muon, 1 NS step)::

        X_shard ─► all_gather ─► wait ─► slice ─► norm ─► permute ─┐
                                                     │              │
                                                     │    mm (Gram: X.T @ X)
                                                     │              │
                                                     │         polynomial f(G)
                                                     │              │
                                                     └──── mm (f(G) @ X) ──► split ──► getitem[rank]

    Graph after::

        X_shard ─► norm ─► permute ─────────────────────────────────┐
                     │                                              │
                     │                                    mm (Gram: X.T @ X)
                     │                                              │
                     │                                   all_reduce(sum) ─► wait
                     │                                              │
                     │                                     polynomial f(G)
                     │                                              │
                     └──────────────────────── mm (f(G) @ X_shard) ─┘

    The all_gather, split, and getitem are eliminated. The all_reduce is
    O(K*K) instead of the all_gather's O(S*W*K), and the mm operates on
    the (S, K) shard instead of the (S*W, K) gathered tensor.

    Algorithm:
    1. Find all Gram mms in the graph (anchor points)
    2. Trace each backward to find the feeding all_gather collective
    3. Group Gram mms by their source all_gather
    4. For each group, find the downstream split+getitem
    5. Transform: replace all_gather with shard, insert all_reduce after each Gram mm
    """
    graph = gm.graph
    transformed = 0

    # 1. Find Gram mms and trace each to its all_gather source
    ag_to_grams: dict[fx.Node, list[fx.Node]] = defaultdict(list)
    ag_infos: dict[fx.Node, AllGatherInfo] = {}

    for gram_mm in find_gram_mms(graph):
        info = find_all_gather_ancestor(gram_source(gram_mm))
        if info is None:
            continue
        ag_to_grams[info.ag_node].append(gram_mm)
        ag_infos[info.ag_node] = info

    # 2. Process each all_gather group
    for ag_node, gram_mms in ag_to_grams.items():
        info = ag_infos[ag_node]

        if len(ag_node.users) != 1 or len(info.wait_node.users) != 1:
            continue

        split_result = find_split_getitem(info.slice_node)
        if split_result is None:
            continue

        split_node, getitem_node = split_result

        split_size = split_node.args[1] if len(split_node.args) > 1 else None
        if not isinstance(split_size, int):
            continue

        # === TRANSFORM ===

        # Route computation to use shard directly
        info.slice_node.replace_all_uses_with(info.shard)

        # Insert all_reduce(sum) after each Gram mm
        for mm_node in gram_mms:
            with graph.inserting_before(mm_node.next):
                ar = graph.call_function(
                    c10d.all_reduce.default,
                    args=(mm_node, "sum", info.group_name),
                )
                _retrace_node_meta(ar)
                wait_ar = graph.call_function(
                    c10d.wait_tensor.default,
                    args=(ar,),
                )
                _retrace_node_meta(wait_ar)
            for user in list(mm_node.users):
                if user is not ar:
                    user.replace_input_with(mm_node, wait_ar)

        # Remove split+getitem (result is already shard-sized)
        pre_split = split_node.args[0]
        assert isinstance(pre_split, fx.Node)
        getitem_node.replace_all_uses_with(pre_split)

        # Erase dead collective nodes (side-effectful, so
        # eliminate_dead_code won't remove them)
        for dead in [
            getitem_node,
            split_node,
            info.slice_node,
            info.wait_node,
            ag_node,
        ]:
            if len(dead.users) == 0:
                graph.erase_node(dead)

        transformed += 1

    if transformed > 0:
        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()
        logger.info(
            "decomp_gram_matrix_all_gather: transformed %d pattern(s)",
            transformed,
        )

    return gm


def _depends_on_any(
    node: fx.Node, targets: OrderedSet[fx.Node], max_depth: int = 100
) -> bool:
    """BFS backward: True if node transitively depends on any node in targets."""
    visited: OrderedSet[fx.Node] = OrderedSet()
    queue: list[tuple[fx.Node, int]] = [(node, 0)]
    while queue:
        n, depth = queue.pop(0)
        if n in visited or depth > max_depth or n.op == "placeholder":
            continue
        visited.add(n)
        if n in targets:
            return True
        for inp in n.all_input_nodes:
            queue.append((inp, depth + 1))
    return False


def batch_all_reduces(gm: fx.GraphModule) -> fx.GraphModule:
    """Batch independent scalar all_reduce calls into a single all_reduce.

    Muon-h and similar optimizers compute per-weight norms that each require
    an independent all_reduce of a scalar value. With N weights this means
    N separate all_reduces, each paying full NCCL launch + latency overhead.

    This pass detects independent scalar all_reduce → wait_tensor pairs with
    the same (reduce_op, group_name), and batches them:

    Before (N all_reduces of any shape)::

        ar_0 = all_reduce(tensor_4x4, "sum", group) → wait_0
        ar_1 = all_reduce(scalar, "sum", group)      → wait_1
        ar_2 = all_reduce(tensor_128x64, "sum", group) → wait_2

    After (1 all_reduce)::

        flat_0 = flatten(tensor_4x4)  # [16]
        flat_1 = flatten(scalar)  # [1]
        flat_2 = flatten(tensor_128x64)  # [8192]
        catted = cat([flat_0, flat_1, flat_2])  # [8209]
        ar = all_reduce(catted, "sum", group)
        wait = wait_tensor(ar)
        chunk_0, chunk_1, chunk_2 = split(wait, [16, 1, 8192])
        result_0 = view(chunk_0, [4, 4])
        result_1 = view(chunk_1, [])  # back to scalar
        result_2 = view(chunk_2, [128, 64])

    Reduces N NCCL calls to 1, saving (N-1) * latency_per_call.
    Works for any combination of tensor shapes via flatten+cat+split+view.
    """
    graph = gm.graph

    # 1. Find all_reduce → wait_tensor pairs where all_reduce has single user
    ar_wait_pairs: list[tuple[fx.Node, fx.Node]] = []
    for node in graph.nodes:
        if (
            node.op == "call_function"
            and node.target is c10d.all_reduce.default
            and len(node.users) == 1
        ):
            wait = next(iter(node.users))
            if wait.op == "call_function" and wait.target is c10d.wait_tensor.default:
                ar_wait_pairs.append((node, wait))

    if len(ar_wait_pairs) < 2:
        return gm

    # 2. Group by (reduce_op, group_name)
    groups: dict[tuple, list[tuple[fx.Node, fx.Node]]] = defaultdict(list)
    for ar_node, wait_node in ar_wait_pairs:
        reduce_op = ar_node.args[1] if len(ar_node.args) > 1 else None
        group_name = ar_node.args[2] if len(ar_node.args) > 2 else None
        groups[(reduce_op, group_name)].append((ar_node, wait_node))

    transformed = 0

    for (reduce_op, group_name), pairs in groups.items():
        if len(pairs) < 2:
            continue

        # 3. Validate inputs: must be fx.Nodes with known shapes
        valid_pairs: list[tuple[fx.Node, fx.Node, torch.Size]] = []
        for ar_node, wait_node in pairs:
            inp = ar_node.args[0]
            if not isinstance(inp, fx.Node):
                continue
            val = inp.meta.get("val")
            if val is None:
                continue
            valid_pairs.append((ar_node, wait_node, val.shape))

        if len(valid_pairs) < 2:
            continue

        # 4. Check independence: no input depends on any wait in the group
        wait_set: OrderedSet[fx.Node] = OrderedSet(wait for _, wait, _ in valid_pairs)
        independent = True
        for ar_node, _, _ in valid_pairs:
            assert isinstance(ar_node.args[0], fx.Node)
            if _depends_on_any(ar_node.args[0], wait_set):
                independent = False
                break
        if not independent:
            continue

        # 5. Find insertion point: after the last input in graph order
        inputs = [ar_node.args[0] for ar_node, _, _ in valid_pairs]
        assert all(isinstance(inp, fx.Node) for inp in inputs)
        input_set: OrderedSet[fx.Node] = OrderedSet(inputs)  # type: ignore[arg-type]
        last_input: fx.Node | None = None
        for n in graph.nodes:
            if n in input_set:
                last_input = n
        assert last_input is not None

        # 6. Compute flat sizes and original shapes for split+reshape
        flat_sizes: list[int] = []
        orig_shapes: list[list[int]] = []
        for _, _, shape in valid_pairs:
            numel = 1
            for s in shape:
                numel *= s
            flat_sizes.append(max(numel, 1))
            orig_shapes.append(list(shape))

        # 7. Create batched all_reduce: flatten → cat → all_reduce → split → view
        with graph.inserting_before(last_input.next):
            flat_nodes = []
            for inp in inputs:
                flat = graph.call_function(aten.reshape.default, args=(inp, [-1]))
                _retrace_node_meta(flat)
                flat_nodes.append(flat)

            catted = graph.call_function(aten.cat.default, args=(flat_nodes,))
            _retrace_node_meta(catted)

            batched_ar = graph.call_function(
                c10d.all_reduce.default,
                args=(catted, reduce_op, group_name),
            )
            _retrace_node_meta(batched_ar)
            batched_wait = graph.call_function(
                c10d.wait_tensor.default, args=(batched_ar,)
            )
            _retrace_node_meta(batched_wait)

            chunks = graph.call_function(
                aten.split_with_sizes.default, args=(batched_wait, flat_sizes)
            )
            _retrace_node_meta(chunks)

            results = []
            for i, shape in enumerate(orig_shapes):
                chunk = graph.call_function(operator.getitem, args=(chunks, i))
                _retrace_node_meta(chunk)
                reshaped = graph.call_function(
                    aten.reshape.default, args=(chunk, shape)
                )
                _retrace_node_meta(reshaped)
                results.append(reshaped)

        # 8. Replace original waits with reshaped chunks
        for i, (ar_node, wait_node, _) in enumerate(valid_pairs):
            wait_node.replace_all_uses_with(results[i])

        # 9. Erase dead nodes
        for ar_node, wait_node, _ in valid_pairs:
            if len(wait_node.users) == 0:
                graph.erase_node(wait_node)
            if len(ar_node.users) == 0:
                graph.erase_node(ar_node)

        transformed += 1
        logger.info(
            "batch_all_reduces: batched %d all_reduces (group=%s, op=%s) into 1",
            len(valid_pairs),
            group_name,
            reduce_op,
        )

    if transformed > 0:
        from torch._inductor.pattern_matcher import stable_topological_sort

        stable_topological_sort(graph)
        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()

    return gm


def decomp_comms(gm: fx.GraphModule) -> fx.GraphModule:
    """Run all collective decomposition passes sequentially."""
    from torch._logging import trace_structured

    logger.info(
        "decomp_comms: running on graph with %d nodes",
        len(list(gm.graph.nodes)),
    )
    decomp_gram_matrix_all_gather(gm)
    batch_all_reduces(gm)

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "post_decomp_comms",
            "encoding": "string",
        },
        payload_fn=lambda: gm.print_readable(
            print_output=False, include_stride=True, include_device=True
        ),
    )
    return gm
