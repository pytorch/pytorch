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
            if not getitems:
                return None
            # Return the first getitem (rank 0's chunk); other ranks'
            # getitems become dead after the transform.
            return node, getitems[0]

        if _is_collective_or_wait(node):
            return None

        chain.add(node)
        node = node.next

    return None


def _is_reduction(node: fx.Node) -> bool:
    """True if node is a reduction op (sum, mean, norm, etc.)."""
    if node.op != "call_function":
        return False
    target = node.target
    if not isinstance(target, torch._ops.OpOverload):
        return False
    return torch.Tag.reduction in target.tags


def _is_norm_op(node: fx.Node) -> bool:
    """True if node computes a vector/matrix norm."""
    return node.op == "call_function" and node.target in (
        aten.linalg_vector_norm.default,
        aten.frobenius_norm.dim,
        aten.norm.Scalar,
        aten.norm.ScalarOpt_dim,
    )


def _gram_shape_is_decomposable(gram_node: fx.Node, gathered_dim: int) -> bool:
    """Check if a Gram mm's output shape is invariant under row sharding.

    The Gram decomposition X.T @ X = sum(Xi.T @ Xi) only works when
    the contraction dimension is the gathered (sharded) dimension.

    For mm(X.T, X) with X sharded along dim 0: contraction is along
    dim 0 of X → decomposes. Output is (K, K), independent of shard size.

    For mm(X, X.T) with X sharded along dim 0: contraction is along
    dim 1 of X → does NOT decompose unless X was transposed from the
    shard (making dim 1 effectively the gathered dim). Output (N, N)
    depends on gathered dim → shape changes with shard.

    We detect this by checking: if the Gram output has any dimension
    equal to the gathered dim, it depends on gathering and can't decompose.
    """
    mm_val = gram_node.meta.get("val")
    if mm_val is None:
        return False
    return int(mm_val.shape[0]) != gathered_dim and int(mm_val.shape[1]) != gathered_dim


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
    4. Validate each Gram's output shape is invariant under sharding
    5. Require >= 3 decomposable Gram mms per group (filters forward/backward
       patterns, targets iterative optimizer patterns like Newton-Schulz)
    6. For each group, find the downstream split+getitem
    7. Transform: replace all_gather with shard, insert all_reduce after each Gram mm
    8. Retrace meta["val"] for all affected nodes (shapes changed)
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

        # 3. Validate shape: gathered dim must differ from shard dim,
        # and each Gram output must not depend on the gathered dim.
        slice_val = info.slice_node.meta.get("val")
        shard_val = (
            info.shard.meta.get("val") if isinstance(info.shard, fx.Node) else None
        )
        if slice_val is None or shard_val is None:
            continue
        gathered_dim = int(slice_val.shape[0])
        shard_dim = int(shard_val.shape[0])
        if gathered_dim == shard_dim:
            continue

        valid_grams = [
            mm for mm in gram_mms if _gram_shape_is_decomposable(mm, gathered_dim)
        ]

        # Require multiple decomposable Gram mms per all_gather.
        # Newton-Schulz iterations produce N Gram mms (one per step, typically 5).
        # Forward/backward patterns typically produce 0-1. This threshold
        # restricts the pass to iterative optimizer patterns.
        _MIN_GRAM_MMS = 2
        if len(valid_grams) < _MIN_GRAM_MMS:
            logger.debug(
                "decomp_gram_matrix_all_gather: skip all_gather %s — "
                "only %d decomposable Gram mms (need >= %d for NS pattern)",
                ag_node.name,
                len(valid_grams),
                _MIN_GRAM_MMS,
            )
            continue
        gram_mms = valid_grams

        # === TRANSFORM ===

        # Route computation to use shard directly. If FSDP pads the shard
        # (shard rows > split_size), the slice was trimming padding off the
        # gathered tensor. We must trim the shard to match.
        replacement: fx.Node = info.shard
        if split_size < shard_dim:
            with graph.inserting_before(info.slice_node):
                replacement = graph.call_function(
                    aten.slice.Tensor, args=(info.shard, 0, 0, split_size)
                )
                _retrace_node_meta(replacement)
        info.slice_node.replace_all_uses_with(replacement)

        # Find all nodes in the dependent chain (needed for reduction detection)
        chain: OrderedSet[fx.Node] = OrderedSet([replacement])
        gram_set: OrderedSet[fx.Node] = OrderedSet(gram_mms)
        n = replacement.next
        while n is not None and n is not split_node:
            if n.op == "call_function" and any(
                inp in chain for inp in n.all_input_nodes
            ):
                chain.add(n)
            n = n.next

        # Insert all_reduce(sum) after each Gram mm
        for mm_node in gram_mms:
            with graph.inserting_before(mm_node.next):
                ar = graph.call_function(
                    c10d.all_reduce.default,
                    args=(mm_node, "sum", info.group_name),
                )
                wait_ar = graph.call_function(
                    c10d.wait_tensor.default,
                    args=(ar,),
                )
            for user in list(mm_node.users):
                if user is not ar:
                    user.replace_input_with(mm_node, wait_ar)

        # Insert all_reduce after reduction ops that reduce the sharded dim.
        # Reductions like norm/sum compute partial results on the shard;
        # they need global aggregation for correctness.
        #
        # For norm: local_norm^2 → all_reduce(sum) → sqrt gives global norm.
        # For sum/mean: all_reduce(sum) directly.
        for node in list(chain):
            if node in gram_set or not _is_reduction(node):
                continue

            if _is_norm_op(node):
                # norm decomposes as: global = sqrt(all_reduce(local^2))
                with graph.inserting_before(node.next):
                    sq = graph.call_function(aten.pow.Tensor_Scalar, args=(node, 2.0))
                    ar = graph.call_function(
                        c10d.all_reduce.default,
                        args=(sq, "sum", info.group_name),
                    )
                    wait_ar = graph.call_function(c10d.wait_tensor.default, args=(ar,))
                    corrected = graph.call_function(
                        aten.pow.Tensor_Scalar, args=(wait_ar, 0.5)
                    )
                for user in list(node.users):
                    if user is not sq:
                        user.replace_input_with(node, corrected)
            else:
                # sum, mean, etc: all_reduce(sum) directly
                with graph.inserting_before(node.next):
                    ar = graph.call_function(
                        c10d.all_reduce.default,
                        args=(node, "sum", info.group_name),
                    )
                    wait_ar = graph.call_function(c10d.wait_tensor.default, args=(ar,))
                for user in list(node.users):
                    if user is not ar:
                        user.replace_input_with(node, wait_ar)

        # Remove split+getitem (result is already shard-sized)
        pre_split = split_node.args[0]
        assert isinstance(pre_split, fx.Node)
        getitem_node.replace_all_uses_with(pre_split)

        # Retrace meta["val"] only for nodes in the dependent chain.
        affected: OrderedSet[fx.Node] = OrderedSet([replacement])
        n = replacement.next
        while n is not None and n is not split_node:
            if n.op == "call_function" and any(
                inp in affected for inp in n.all_input_nodes
            ):
                _retrace_node_meta(n)
                affected.add(n)
            n = n.next

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


def decomp_comms(gm: fx.GraphModule) -> fx.GraphModule:
    """Run all collective decomposition passes sequentially."""
    from torch._logging import trace_structured

    logger.info(
        "decomp_comms: running on graph with %d nodes",
        len(list(gm.graph.nodes)),
    )
    decomp_gram_matrix_all_gather(gm)

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
