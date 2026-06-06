"""
Decompose collective communication patterns into mathematically equivalent
local computation + lighter collectives.

Current passes:
- decomp_gram_matrix_all_gather:
    all_gather + mm(X.T, X) -> local mm(Xi.T, Xi) + all_reduce(sum)
    Exploits X.T @ X = sum_i(Xi.T @ Xi).

Gated by config.aten_distributed_optimizations.allow_comms_decompositions.
"""

import logging
import operator
from collections import defaultdict
from typing import NamedTuple

import torch
import torch.fx as fx
from torch._inductor.fx_passes.bucketing import get_collective_type, is_wait_tensor
from torch._inductor.fx_utils import get_fake_args_kwargs
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet


aten = torch.ops.aten
c10d = torch.ops._c10d_functional
logger = logging.getLogger(__name__)

# Require >= 2 Gram mms: targets iterative optimizers (Newton-Schulz),
# skips single fwd/bwd Grams (decomposable but not yet validated).
# TODO: validate single-Gram fwd/bwd decomposition on a real model.
_MIN_GRAM_MMS = 2


def _get_fake_mode(node: fx.Node) -> torch._subclasses.fake_tensor.FakeTensorMode:
    """Get FakeTensorMode: V.fake_mode during compilation, from node meta in tests."""
    mode = V.fake_mode
    if isinstance(mode, torch._subclasses.fake_tensor.FakeTensorMode):
        return mode
    # Standalone call (tests) -- extract from node metadata
    for inp in node.all_input_nodes:
        val = inp.meta.get("val")
        if isinstance(val, torch._subclasses.fake_tensor.FakeTensor):
            return val.fake_mode
    raise RuntimeError(f"No FakeTensorMode available for {node.name}")


def _retrace_node_meta(node: fx.Node) -> None:
    """Recompute node.meta["val"] by running the op on input FakeTensors."""
    if node.op != "call_function":
        return
    valid, args, kwargs = get_fake_args_kwargs(node)
    if not valid:
        return
    target = node.target
    assert callable(target)
    with _get_fake_mode(node):
        node.meta["val"] = target(*args, **kwargs)


def _is_2d_transpose(node: fx.Node) -> bool:
    """True if node is aten.t, permute([1,0]), or transpose(0,1) on a 2D input."""
    if node.op != "call_function":
        return False
    inp_val = (
        node.args[0].meta.get("val")
        if node.args and isinstance(node.args[0], fx.Node)
        else None
    )
    if inp_val is not None and inp_val.dim() != 2:
        return False
    if node.target is aten.t.default:
        return True
    if node.target is aten.permute.default:
        dims = node.args[1] if len(node.args) > 1 else None
        return isinstance(dims, (list, tuple)) and list(dims) == [1, 0]
    if node.target is aten.transpose.int and len(node.args) >= 3:
        d0, d1 = node.args[1], node.args[2]
        return (d0, d1) in ((0, 1), (1, 0), (-2, -1), (-1, -2))
    return False


def is_gram_mm(node: fx.Node) -> bool:
    """True if node computes mm(X, X.T) or mm(X.T, X)."""
    if node.op != "call_function" or node.target is not aten.mm.default:
        return False
    a, b = node.args[0], node.args[1]
    if not isinstance(a, fx.Node) or not isinstance(b, fx.Node):
        return False
    return (_is_2d_transpose(a) and a.args[0] is b) or (
        _is_2d_transpose(b) and b.args[0] is a
    )


def gram_source(gram_node: fx.Node) -> fx.Node:
    """Return the base tensor X (non-transposed arg) from a Gram mm."""
    a = gram_node.args[0]
    b = gram_node.args[1]
    assert isinstance(a, fx.Node) and isinstance(b, fx.Node)
    return b if _is_2d_transpose(a) else a


class AllGatherInfo(NamedTuple):
    shard: fx.Node
    group_name: str
    ag_node: fx.Node
    wait_node: fx.Node
    # Full gathered tensor consumed downstream: an identity/trim slice
    # wrapping the wait when present, else the wait itself.
    entry_node: fx.Node


def find_all_gather_ancestor(
    node: fx.Node, max_depth: int = 30
) -> AllGatherInfo | None:
    """Walk backward from node to find an all_gather -> wait [-> slice] chain.

    The trailing slice is optional. Inductor folds the FSDP identity slice
    slice(wait, 0, 0, gathered_rows) when sharding is even (it is a no-op), so
    we anchor on the wait and use a slice that wraps it as the entry node only
    when one is present (FSDP padding trims, or overlap scheduling keeps it).
    """
    visited: OrderedSet[fx.Node] = OrderedSet()
    queue = [(node, 0)]
    while queue:
        n, depth = queue.pop(0)
        if n in visited or depth > max_depth or n.op == "placeholder":
            continue
        visited.add(n)
        for inp in n.all_input_nodes:
            queue.append((inp, depth + 1))
        if n.op != "call_function" or n.target is not c10d.wait_tensor.default:
            continue
        ag_node = n.args[0]
        if (
            not isinstance(ag_node, fx.Node)
            or ag_node.target is not c10d.all_gather_into_tensor.default
        ):
            continue
        # all_gather_into_tensor(shard, world_size, group_name)
        shard, group_name = ag_node.args[0], ag_node.args[2]
        if not isinstance(shard, fx.Node) or not isinstance(group_name, str):
            continue
        entry_node = n
        if len(n.users) == 1:
            (only_user,) = n.users
            if (
                only_user.op == "call_function"
                and only_user.target is aten.slice.Tensor
            ):
                entry_node = only_user
        return AllGatherInfo(
            shard=shard,
            group_name=group_name,
            ag_node=ag_node,
            wait_node=n,
            entry_node=entry_node,
        )
    return None


def _select_getitem(split_node: fx.Node, rank: int | None) -> fx.Node | None:
    """Pick getitem from a split: by rank if known, else lowest-index with users."""
    getitems = [
        u
        for u in split_node.users
        if u.op == "call_function" and u.target is operator.getitem
    ]
    if not getitems:
        return None
    if rank is not None:
        matches = [g for g in getitems if len(g.args) > 1 and g.args[1] == rank]
        return matches[0] if matches else None
    used = sorted(
        (g for g in getitems if g.users),
        key=lambda g: g.args[1] if len(g.args) > 1 else 0,
    )
    return used[0] if used else None


def find_split_getitem(
    start: fx.Node, rank: int | None = None, max_search: int = 3000
) -> tuple[fx.Node, fx.Node] | None:
    """Walk forward from start to find split(dim=0) -> getitem[rank]."""
    chain: OrderedSet[fx.Node] = OrderedSet([start])
    searched = 0
    node = start.next
    while node is not None and searched < max_search:
        searched += 1
        if node.op != "call_function" or not any(
            inp in chain for inp in node.all_input_nodes
        ):
            node = node.next
            continue
        if get_collective_type(node) != "" or is_wait_tensor(node):
            return None
        if node.target is aten.split.Tensor:
            if (node.args[2] if len(node.args) > 2 else 0) != 0:
                return None
            gi = _select_getitem(node, rank)
            return (node, gi) if gi else None
        chain.add(node)
        node = node.next
    return None


def _is_reduction(node: fx.Node) -> bool:
    """True if node is a reduction op (sum, mean, norm, etc.)."""
    return (
        node.op == "call_function"
        and isinstance(node.target, torch._ops.OpOverload)
        and torch.Tag.reduction in node.target.tags
    )


def _is_l2_norm_op(node: fx.Node) -> bool:
    """True if node computes an L2 (Frobenius) norm."""
    if node.op != "call_function":
        return False
    if node.target in (aten.frobenius_norm.dim, aten.norm.Scalar):
        return True
    if node.target in (aten.linalg_vector_norm.default, aten.norm.ScalarOpt_dim):
        ord_arg = node.args[1] if len(node.args) > 1 else 2
        return ord_arg in (2, 2.0)
    return False


# Reductions whose global result decomposes from per-shard results
# via all_reduce: L2 norms (pow2 + sum + sqrt), sums (sum).
# Unknown reductions cause the transform to bail out entirely.
_DECOMPOSABLE_REDUCTIONS = (
    aten.sum.default,
    aten.sum.dim_IntList,
    aten.linalg_vector_norm.default,
    aten.norm.ScalarOpt_dim,
    aten.norm.Scalar,
    aten.frobenius_norm.dim,
)


def _valid_entry_slice(node: fx.Node) -> bool:
    val = node.meta.get("val")
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
    start = node.args[2] if len(node.args) > 2 else node.kwargs.get("start", 0)
    end = node.args[3] if len(node.args) > 3 else node.kwargs.get("end", None)
    step = node.args[4] if len(node.args) > 4 else node.kwargs.get("step", 1)
    if (
        not isinstance(val, torch.Tensor)
        or not isinstance(dim, int)
        or not isinstance(start, int)
        or not isinstance(step, int)
        or (end is not None and not isinstance(end, int))
    ):
        return False
    return (
        -val.dim() <= dim < val.dim()
        and dim % val.dim() == 0
        and start == 0
        and step == 1
        and (end is None or end >= int(val.shape[0]))
    )


def _reduction_includes_sharded_dim(node: fx.Node) -> bool:
    """True if a reduction reduces over dim 0 (the sharded dimension).

    Returns False for reductions over only non-sharded dims. Unknown reduction
    ops are assumed to include the sharded dim so the caller bails out.
    """
    if not _is_reduction(node):
        return False

    def _dims_include_zero(dims: object) -> bool:
        if isinstance(dims, int):
            dims = [dims]
        elif not isinstance(dims, (list, tuple)):
            return True
        inp = node.args[0] if node.args else None
        inp_val = inp.meta.get("val") if isinstance(inp, fx.Node) else None
        ndim: int = inp_val.dim() if inp_val is not None else 2
        return any(int(d) % ndim == 0 for d in dims)

    # Norms with dim arg
    if node.target in (
        aten.linalg_vector_norm.default,
        aten.norm.ScalarOpt_dim,
        aten.frobenius_norm.dim,
    ):
        dims = node.args[2] if len(node.args) > 2 else None
        return dims is None or _dims_include_zero(dims)

    # sum/mean with dim arg
    if node.target in (aten.sum.dim_IntList, aten.mean.dim):
        dims = node.args[1] if len(node.args) > 1 else None
        return dims is None or _dims_include_zero(dims)

    if len(node.args) > 1 and isinstance(node.args[1], (int, list, tuple)):
        return _dims_include_zero(node.args[1])
    return True


def _gram_shape_is_decomposable(gram_node: fx.Node, gathered_rows: int) -> bool:
    """True if the Gram output shape doesn't depend on the gathered row count.

    X.T @ X = sum(Xi.T @ Xi) only when the contraction dim is the sharded
    dim. If any output dim equals gathered_rows, the decomposition is invalid.
    """
    mm_val = gram_node.meta.get("val")
    if mm_val is None:
        return False
    return (
        int(mm_val.shape[0]) != gathered_rows and int(mm_val.shape[1]) != gathered_rows
    )


def _insert_all_reduce_wait(graph: fx.Graph, node: fx.Node, group_name: str) -> fx.Node:
    """Insert all_reduce(sum) + wait_tensor after node, rewire users."""
    with graph.inserting_before(node.next):
        ar = graph.call_function(
            c10d.all_reduce.default, args=(node, "sum", group_name)
        )
        wait_ar = graph.call_function(c10d.wait_tensor.default, args=(ar,))
    for user in list(node.users):
        if user is not ar:
            user.replace_input_with(node, wait_ar)
    return wait_ar


def _collect_dependent_chain(start: fx.Node, stop: fx.Node) -> OrderedSet[fx.Node]:
    """Collect call_function nodes transitively dependent on start, up to stop."""
    chain: OrderedSet[fx.Node] = OrderedSet([start])
    n = start.next
    while n is not None and n is not stop:
        if n.op == "call_function" and any(inp in chain for inp in n.all_input_nodes):
            chain.add(n)
        n = n.next
    return chain


def _collect_split_path_chain(start: fx.Node, stop: fx.Node) -> OrderedSet[fx.Node]:
    chain = _collect_dependent_chain(start, stop)
    path: OrderedSet[fx.Node] = OrderedSet()
    stack = [stop]
    while stack:
        node = stack.pop()
        if node in path:
            continue
        path.add(node)
        stack.extend(inp for inp in node.all_input_nodes if inp in chain)
    return OrderedSet(n for n in chain if n in path)


def _has_unsupported_sharded_reduction(
    chain: OrderedSet[fx.Node], gram_set: OrderedSet[fx.Node]
) -> bool:
    """True if chain contains a reduction over dim 0 we can't correct."""
    for node in chain:
        if node in gram_set:
            continue
        if not _reduction_includes_sharded_dim(node):
            continue
        if _is_l2_norm_op(node) or node.target in (
            aten.sum.default,
            aten.sum.dim_IntList,
        ):
            continue
        logger.debug(
            "decomp_gram: unsupported reduction %s (%s)", node.name, node.target
        )
        return True
    return False


def decomp_gram_matrix_all_gather(gm: fx.GraphModule) -> fx.GraphModule:
    """Eliminate all_gather when the gathered tensor feeds Gram matrix computation.

    Replaces all_gather + mm(X.T, X) with local mm(Xi.T, Xi) + all_reduce(sum).
    The Gram identity X.T @ X = sum(Xi.T @ Xi) makes each rank's partial
    Gram summable via all_reduce, eliminating the O(S*W*K) all_gather in
    favor of O(K^2) all_reduces.

    Also corrects reductions over the sharded dim: L2 norms become
    pow(2) + all_reduce(sum) + sqrt, sums become all_reduce(sum).
    """
    # Lazy: a module-level distributed import breaks test_circular_dependencies.
    from torch.distributed.distributed_c10d import _resolve_process_group, GroupName

    graph = gm.graph
    transformed = 0

    # Find Gram mms and trace each to its all_gather source
    ag_to_grams: dict[fx.Node, list[fx.Node]] = defaultdict(list)
    ag_infos: dict[fx.Node, AllGatherInfo] = {}

    for mm_node in graph.find_nodes(op="call_function", target=aten.mm.default):
        if not is_gram_mm(mm_node):
            continue
        info = find_all_gather_ancestor(gram_source(mm_node))
        if info is None:
            continue
        ag_to_grams[info.ag_node].append(mm_node)
        ag_infos[info.ag_node] = info

    for ag_node, gram_mms in ag_to_grams.items():
        info = ag_infos[ag_node]

        # ag must feed only its wait. The wait may have multiple users when no
        # slice wraps it (the gathered tensor feeds the chain directly); those
        # users are validated to stay inside the dependent chain below.
        if len(ag_node.users) != 1:
            continue

        rank = _resolve_process_group(GroupName(info.group_name)).rank()
        split_result = find_split_getitem(info.entry_node, rank=rank)
        if split_result is None:
            continue
        split_node, getitem_node = split_result

        split_size = split_node.args[1] if len(split_node.args) > 1 else None
        if not isinstance(split_size, int):
            continue

        # Validate: gathered rows must differ from shard rows
        entry_val = info.entry_node.meta.get("val")
        shard_val = (
            info.shard.meta.get("val") if isinstance(info.shard, fx.Node) else None
        )
        if entry_val is None or shard_val is None:
            continue
        gathered_rows = int(entry_val.shape[0])
        shard_rows = int(shard_val.shape[0])
        if gathered_rows == shard_rows:
            continue
        if info.entry_node is not info.wait_node and not _valid_entry_slice(
            info.entry_node
        ):
            continue

        valid_grams = [
            mm for mm in gram_mms if _gram_shape_is_decomposable(mm, gathered_rows)
        ]

        if len(valid_grams) < _MIN_GRAM_MMS:
            logger.debug(
                "decomp_gram: skip %s -- %d < %d decomposable Gram mms",
                ag_node.name,
                len(valid_grams),
                _MIN_GRAM_MMS,
            )
            continue
        gram_mms = valid_grams

        # Even split: all getitems must have the same shape (SPMD constraint)
        all_getitems = [
            u
            for u in split_node.users
            if u.op == "call_function" and u.target is operator.getitem
        ]

        # Collapsing the split into the rank-local result invalidates sibling
        # getitems; in real SPMD only the rank's own getitem is consumed, so bail.
        if any(g is not getitem_node and len(g.users) > 0 for g in all_getitems):
            logger.debug(
                "decomp_gram: skip %s -- split has other consumed getitems",
                ag_node.name,
            )
            continue

        shapes = [
            tuple(int(d) for d in u.meta["val"].shape)
            for u in all_getitems
            if u.meta.get("val") is not None
        ]
        if len(OrderedSet(shapes)) > 1:
            logger.debug("decomp_gram: skip -- uneven split shapes %s", shapes[:3])
            continue

        getitem_val = getitem_node.meta.get("val")
        if getitem_val is None:
            continue
        expected_rows = int(getitem_val.shape[0])

        # Shard trimming for FSDP padding
        replacement: fx.Node = info.shard
        if expected_rows < shard_rows:
            with graph.inserting_before(info.entry_node):
                replacement = graph.call_function(
                    aten.slice.Tensor,
                    args=(info.shard, 0, 0, expected_rows),
                )
                _retrace_node_meta(replacement)
        elif expected_rows > shard_rows:
            continue

        # Verify the rewritten region is closed over the path to split.
        chain = _collect_split_path_chain(info.entry_node, split_node)
        if any(u is not split_node and u not in chain for n in chain for u in n.users):
            continue

        gram_set: OrderedSet[fx.Node] = OrderedSet(gram_mms)
        if _has_unsupported_sharded_reduction(chain, gram_set):
            continue

        # === Transform ===

        info.entry_node.replace_all_uses_with(replacement)

        for mm_node in gram_mms:
            _insert_all_reduce_wait(graph, mm_node, info.group_name)

        # Correct reductions over the sharded dim
        for node in list(chain):
            if node in gram_set or not _reduction_includes_sharded_dim(node):
                continue
            if _is_l2_norm_op(node):
                # L2: global = sqrt(all_reduce_sum(local^2))
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
            elif _is_reduction(node):
                _insert_all_reduce_wait(graph, node, info.group_name)

        # Result is already shard-sized, bypass split+getitem
        pre_split = split_node.args[0]
        assert isinstance(pre_split, fx.Node)
        getitem_node.replace_all_uses_with(pre_split)

        # Retrace meta["val"] for nodes whose shapes changed
        affected: OrderedSet[fx.Node] = OrderedSet([replacement])
        n = replacement.next
        while n is not None and n is not split_node:
            if n.op == "call_function" and any(
                inp in affected for inp in n.all_input_nodes
            ):
                _retrace_node_meta(n)
                affected.add(n)
            n = n.next

        # entry_node may be the wait itself; dedup so we don't erase twice.
        for dead in OrderedSet(
            [getitem_node, split_node, info.entry_node, info.wait_node, ag_node]
        ):
            if len(dead.users) == 0:
                graph.erase_node(dead)

        transformed += 1

    if transformed > 0:
        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()
        logger.info("decomp_gram: transformed %d pattern(s)", transformed)

    return gm


def decomp_comms(gm: fx.GraphModule) -> fx.GraphModule:
    """Run all collective decomposition passes."""
    from torch._logging import trace_structured

    logger.info(
        "decomp_comms: running on graph with %d nodes",
        len(list(gm.graph.nodes)),
    )

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "pre_decomp_comms",
            "encoding": "string",
        },
        payload_fn=lambda: gm.print_readable(
            print_output=False, include_stride=True, include_device=True
        ),
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
