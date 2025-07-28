import logging
from collections import defaultdict
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
import torch.utils._pytree as pytree
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import detect_fake_mode
from torch._logging import trace_structured
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._ordered_set import OrderedSet


logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def bucket_cap_mb_by_bucket_idx_default(bucket_id: int) -> float:
    """
    Determine the size of a bucket based on its ID.

    Args:
    bucket_id (int): The ID of the bucket.

    Returns:
    float: The size of the bucket.
    """
    return 2000.0


def bucket_all_gather(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Optional[Callable[[int], float]] = None,
) -> None:
    if bucket_cap_mb_by_bucket_idx is None:
        from torch._inductor.fx_passes.bucketing import (
            bucket_cap_mb_by_bucket_idx_default,
        )

        bucket_cap_mb_by_bucket_idx = bucket_cap_mb_by_bucket_idx_default
    ag_buckets = bucket_all_gather_by_mb(gm, bucket_cap_mb_by_bucket_idx)
    if len(ag_buckets) == 0:
        return
    merge_all_gather(gm, ag_buckets)


def bucket_reduce_scatter(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Optional[Callable[[int], float]] = None,
) -> None:
    if bucket_cap_mb_by_bucket_idx is None:
        from torch._inductor.fx_passes.bucketing import (
            bucket_cap_mb_by_bucket_idx_default,
        )

        bucket_cap_mb_by_bucket_idx = bucket_cap_mb_by_bucket_idx_default
    rs_buckets = bucket_reduce_scatter_by_mb(gm, bucket_cap_mb_by_bucket_idx)
    if len(rs_buckets) == 0:
        return
    merge_reduce_scatter(gm, rs_buckets)


def is_all_gather_into_tensor(node: torch.fx.Node) -> bool:  # type: ignore[arg-type]
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.all_gather_into_tensor.default
    )


def is_reduce_scatter_tensor(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.reduce_scatter_tensor.default
    )


def is_wait_tensor(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.wait_tensor.default
    )


def is_wait_tensor_from_all_gather_into_tensor(node: torch.fx.Node) -> bool:
    return is_wait_tensor(node) and is_all_gather_into_tensor(node.args[0])  # type: ignore[arg-type]


def greedy_bucket_collective_by_mb(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float],
    filter_node: Callable[[torch.fx.Node], bool],
    node_group_key: Callable[[torch.fx.Node], Any],
    filter_wait_node: Optional[Callable[[torch.fx.Node], bool]] = None,
) -> list[list[torch.fx.Node]]:
    g = gm.graph
    found_candidates = False
    for node in g.nodes:
        if filter_node(node):
            found_candidates = True
            break
    if not found_candidates:
        return []

    nodes_groups: dict[Any, list[torch.fx.Node]] = defaultdict(list)
    nodes_successors: dict[torch.fx.Node, OrderedSet[torch.fx.Node]] = defaultdict(
        OrderedSet
    )

    for node in g.nodes:
        for n, successors in nodes_successors.items():
            if any(arg in successors for arg in node.args):
                successors.add(n)
        if is_wait_tensor(node) and filter_node(node.args[0]):
            if (filter_wait_node is None) or filter_wait_node(node):
                coll_node = node.args[0]
                group_key = node_group_key(coll_node)
                nodes_groups[group_key].append(coll_node)

    buckets: list[list[torch.fx.Node]] = []
    for nodes in nodes_groups.values():
        cur_bucket: list[torch.fx.Node] = []
        cur_bucket_successors: OrderedSet[torch.fx.Node] = OrderedSet()
        cur_bucket_size_bytes: int = 0
        cur_bucket_id: int = 0
        bucket_size_bytes = int(
            bucket_cap_mb_by_bucket_idx(cur_bucket_id) * 1024 * 1024
        )
        for node in nodes:
            if node in cur_bucket_successors:
                # We can not bucket successors with the node
                continue
            assert "val" in node.meta
            n_val = node.meta["val"]
            out_size_bytes = n_val.numel() * n_val.element_size()
            if (
                cur_bucket_size_bytes + out_size_bytes > bucket_size_bytes
                and cur_bucket
            ):
                # Current bucket is full, create new bucket
                if len(cur_bucket) > 1:
                    buckets.append(cur_bucket)
                cur_bucket = []
                cur_bucket_size_bytes = 0
                cur_bucket_id += 1
                cur_bucket_successors = OrderedSet()
            cur_bucket_size_bytes += out_size_bytes
            cur_bucket.append(node)
            cur_bucket_successors |= nodes_successors[node]
        if len(cur_bucket) > 1:
            buckets.append(cur_bucket)
    return buckets


def bucket_all_gather_by_mb(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float],
    filter_wait_node: Optional[Callable[[torch.fx.Node], bool]] = None,
) -> list[list[torch.fx.Node]]:
    """
    Identifies all all_gather nodes and groups them into buckets,
    based on size limit `bucket_cap_mb_by_bucket_idx`.

    Args:
        gm (torch.fx.GraphModule): GraphModule where to bucket all_gathers.
        bucket_cap_mb_bucket_idx (Callable[[int], float]): Callable to specify cap of the bucket
            in megabytes by bucket idx.  The idea of `bucket_cap_mb_by_bucket_idx` is to allow
            to specify different sizes of the buckets at the start,
            as first all_gather is usually exposed.  Interface of bucket_cap_mb_by_bucket_idx
            is `bucket_cap_mb_by_bucket_idx_default` function that is default value for `bucket_cap_mb_by_bucket_idx`.
        filter_wait_node (Optional[Callable[[torch.fx.Node], bool]]): If specified,
            only all_gather nodes with wait_node that satisfy `filter_wait_node` will be bucketed.

    Returns:
        list[list[torch.fx.Node]]: List of buckets, where each bucket is a list of all_gather nodes.
    """

    def _ag_group_key(node: torch.fx.Node) -> tuple[str, torch.dtype]:
        _, group_size, group_name = node.args
        dtype = node.meta["val"].dtype
        assert isinstance(group_name, str)
        return (group_name, dtype)

    return greedy_bucket_collective_by_mb(
        gm,
        bucket_cap_mb_by_bucket_idx,
        is_all_gather_into_tensor,
        _ag_group_key,
        filter_wait_node,
    )


def bucket_reduce_scatter_by_mb(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float],
    filter_wait_node: Optional[Callable[[torch.fx.Node], bool]] = None,
) -> list[list[torch.fx.Node]]:
    """
    Identifies all reduce_scatter nodes and groups them into buckets,
        based on size limit `bucket_cap_mb_by_bucket_idx`.

    Args:
        gm (torch.fx.GraphModule): GraphModule where to bucket reduce_scatters.
        bucket_cap_mb_bucket_idx (Callable[[int], float]): Callable to specify cap of the bucket
            in megabytes by bucket idx.  The idea of `bucket_cap_mb_by_bucket_idx` is to allow
            to specify different sizes of the buckets.
        filter_wait_node (Optional[Callable[[torch.fx.Node], bool]]): If specified,
            only reduce_scatter nodes with wait_node that satisfy `filter_wait_node` will be bucketed.

    Returns:
        list[list[torch.fx.Node]]: List of buckets, where each bucket is a list of all_gather nodes.
    """

    def _rs_group_key(node: torch.fx.Node) -> tuple[str, str, torch.dtype]:
        _, reduce_op, group_size, group_name = node.args
        dtype = node.meta["val"].dtype
        assert isinstance(group_name, str)
        assert isinstance(reduce_op, str)
        return (group_name, reduce_op, dtype)

    return greedy_bucket_collective_by_mb(
        gm,
        bucket_cap_mb_by_bucket_idx,
        is_reduce_scatter_tensor,
        _rs_group_key,
        filter_wait_node,
    )


def reduce_scatter_merge_fn_to_trace(
    rs_ins: list[torch.Tensor],
    group_size: int,
    group_name: str,
    reduce_op: str,
    reduce_dtype: torch.dtype,  # type: ignore[name-defined]
    device: torch.device,  # type: ignore[name-defined]
) -> list[torch.Tensor]:  # type: ignore[no-untyped-def]
    rs_ins_flattened = [rs_in.view(-1) for rs_in in rs_ins]

    rs_ins_srcs = [
        rs_in_f.split([rs_in_f.numel() // group_size] * group_size)
        for rs_in_f in rs_ins_flattened
    ]

    foreach_copy_srcs = []
    for rank_idx in range(group_size):
        for rs_in_idx in range(len(rs_ins)):
            foreach_copy_srcs.append(rs_ins_srcs[rs_in_idx][rank_idx])

    new_rs_in = torch.cat(foreach_copy_srcs, dim=0)

    wait_tensor = torch.ops.c10d_functional.wait_tensor(
        torch.ops._c10d_functional.reduce_scatter_tensor.default(
            new_rs_in, reduce_op, group_size, group_name
        )
    )
    new_rs_out = wait_tensor

    new_outs = []
    new_rs_out_offset = 0
    for rs_in in rs_ins:
        new_out_size = torch.Size((rs_in.shape[0] // group_size,) + rs_in.shape[1:])  # type: ignore[attr-defined]
        new_out = new_rs_out.narrow(0, new_rs_out_offset, new_out_size.numel()).reshape(
            new_out_size
        )
        new_outs.append(new_out)
        new_rs_out_offset += new_out_size.numel()
    return new_outs


def all_gather_merge_fn_to_trace(
    ag_ins: list[torch.Tensor],
    group_size: int,
    group_name: str,
    dtype: torch.dtype,  # type: ignore[name-defined]
    rank: int,
) -> list[torch.Tensor]:
    ins_sizes = [ag_in.shape for ag_in in ag_ins]
    ins_split_sizes = [ag_in.numel() for ag_in in ag_ins]
    ag_input_numel = sum(ins_split_sizes)
    device = ag_ins[0].device
    new_ag_out = torch.empty(ag_input_numel * group_size, dtype=dtype, device=device)
    new_ag_in = new_ag_out.narrow(0, ag_input_numel * rank, ag_input_numel)
    foreach_copy_dsts = torch.split(new_ag_in, ins_split_sizes)
    ag_ins_flattened = [ag_in.reshape(-1) for ag_in in ag_ins]
    torch._foreach_copy_(foreach_copy_dsts, ag_ins_flattened)
    wait_tensor = torch.ops.c10d_functional.wait_tensor(
        torch.ops._c10d_functional.all_gather_into_tensor_out.default(
            new_ag_in, group_size, group_name, out=new_ag_out
        )
    )
    new_ag_out_reshaped = wait_tensor.reshape(group_size, -1)
    outs = torch.split_with_sizes(
        new_ag_out_reshaped,
        ins_split_sizes,
        dim=1,
    )
    outs_reshaped = [
        o.reshape((shape[0] * group_size,) + shape[1:])
        for o, shape in zip(outs, ins_sizes)
    ]
    return outs_reshaped


def all_gather_merge_fn_to_trace_functional(
    ag_ins: list[torch.Tensor],
    group_size: int,
    group_name: str,
    dtype: torch.dtype,  # type: ignore[name-defined]
    rank: int,
    use_fsdp_ag_copy_in: bool = False,
) -> list[torch.Tensor]:
    # Implementation that is functional in graph,
    # but uses custom op torch.ops.fsdp.all_gather_copy_in.
    ins_sizes = [ag_in.shape for ag_in in ag_ins]
    ins_split_sizes = [ag_in.numel() for ag_in in ag_ins]
    ag_input_numel = sum(ins_split_sizes)
    device = ag_ins[0].device
    new_ag_out = torch.empty(ag_input_numel * group_size, dtype=dtype, device=device)
    ag_ins_flattened = [ag_in.reshape(-1) for ag_in in ag_ins]
    if use_fsdp_ag_copy_in:
        new_ag_in, new_ag_out = torch.ops.fsdp.all_gather_copy_in(
            ag_ins_flattened, new_ag_out, ins_split_sizes, ag_input_numel, rank
        )
    else:
        new_ag_in = torch.cat(ag_ins_flattened, dim=0)
    wait_tensor = torch.ops.c10d_functional.wait_tensor(
        torch.ops._c10d_functional.all_gather_into_tensor_out.default(
            new_ag_in, group_size, group_name, out=new_ag_out
        )
    )
    new_ag_out_reshaped = wait_tensor.reshape(group_size, -1)
    outs = torch.split_with_sizes(
        new_ag_out_reshaped,
        ins_split_sizes,
        dim=1,
    )
    outs_reshaped = [
        o.reshape((shape[0] * group_size,) + shape[1:])
        for o, shape in zip(outs, ins_sizes)
    ]
    return outs_reshaped


def _trace(fn, inps) -> torch.fx.GraphModule:  # type: ignore[no-untyped-def]
    fake_mode = detect_fake_mode(inps)
    assert fake_mode is not None
    with fake_mode, enable_python_dispatcher():
        return make_fx(fn)(*inps)


def _insert_fn_trace_before_node(  # type: ignore[no-untyped-def]
    g: torch.fx.Graph,
    fn_to_trace,
    inps,
    insert_before_node: torch.fx.Node,
    g_fn_inps: list[torch.fx.Node],
    g_fn_outs: list[torch.fx.Node],
) -> dict[torch.fx.Node, torch.fx.Node]:  # type: ignore[no-untyped-def]
    """
    Helper function that traces :attr:`fn_to_trace` with inputs
    :attr:`inps`.
    The result function graph will be inserted before :attr:`insert_before_node`,
    using :attr:`g_fn_inps` nodes of original graphas inputs of function graph,
    function graph outputs will replace :attr:`g_fn_outs` in original graph.
    """
    fn_gm = _trace(
        fn_to_trace,
        inps,
    )
    fn_g = fn_gm.graph
    fn_g_ins = fn_g.find_nodes(op="placeholder")
    env = {fn_g_ins[idx]: g_fn_inps[idx] for idx in range(len(g_fn_inps))}
    g_fn_new_outs: list[torch.fx.Node] = []
    with g.inserting_before(insert_before_node):
        for _n in fn_g.nodes:
            if _n.op == "placeholder":
                continue
            _new_n = g.node_copy(_n, lambda x: env[x])
            env[_n] = _new_n
            if _n.op == "output":
                g_fn_new_outs = _new_n.args[0]  # type: ignore[assignment]
                g.erase_node(_new_n)
    replacements = {  # noqa: C416
        orig_out: new_out for orig_out, new_out in zip(g_fn_outs, g_fn_new_outs)
    }
    for orig_out, new_out in zip(g_fn_outs, g_fn_new_outs):
        orig_out.replace_all_uses_with(new_out)
    return replacements


def merge_reduce_scatter(
    gm: torch.fx.GraphModule, rs_buckets: list[list[torch.fx.Node]]
) -> None:
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "fx_bucketing_passes_reduce_scatter_buckets",
            "encoding": "string",
        },
        payload_fn=lambda: str(rs_buckets),
    )
    n_buckets = len(rs_buckets)
    g = gm.graph
    rs_ins: list[list[torch.fx.Node]] = [[] for _ in range(n_buckets)]
    rs_waits: list[list[torch.fx.Node]] = [[] for _ in range(n_buckets)]

    for bucket_idx, rs_nodes in enumerate(rs_buckets):
        rs0 = rs_nodes[0]
        rs0_val = rs0.meta["val"]
        _, reduce_op, group_size, group_name = rs0.args
        reduce_dtype = rs0_val.dtype
        device = rs0_val.device
        for n in rs_nodes:
            rs_val = n.meta["val"]
            assert (
                n.args[1] == reduce_op
                and n.args[2] == group_size
                and n.args[3] == group_name
                and rs_val.device == device
                and rs_val.dtype == reduce_dtype
            )
            assert len(n.users) == 1
            wait_n = next(iter(n.users))
            rs_ins[bucket_idx].append(n.args[0])  # type: ignore[arg-type]
            rs_waits[bucket_idx].append(wait_n)

    for bucket_idx in range(n_buckets):
        _rs_ins = rs_ins[bucket_idx]
        _rs_waits = rs_waits[bucket_idx]
        _rs_ns = rs_buckets[bucket_idx]

        rs0 = _rs_ns[0]
        rs0_val = rs0.meta["val"]
        _, reduce_op, group_size, group_name = rs0.args
        reduce_dtype = rs0_val.dtype
        device = rs0_val.device

        replacements = _insert_fn_trace_before_node(
            g,
            reduce_scatter_merge_fn_to_trace,
            (
                pytree.tree_map(lambda node: node.meta["val"], _rs_ins),
                group_size,
                group_name,
                reduce_op,
                reduce_dtype,
                device,
            ),
            _rs_ns[-1].next,
            _rs_ins,
            _rs_waits,
        )
        # [Note: Replacement in bucketing passes]
        # After bucketing _rs_waits will be replaced with output nodes of
        # fn_to_trace graph that will be inserted in the graph g.
        # By this time we already prepared rs_ins, rs_waits.
        # rs_ins for following buckets can be replaced _rs_waits with new nodes.
        # We apply replacements to rs_ins.

        def _replace(x: torch.fx.Node) -> torch.fx.Node:
            return replacements.get(x, x)

        for j in range(bucket_idx + 1, n_buckets):
            rs_ins[j] = pytree.tree_map(_replace, rs_ins[j])

        for rs_n, wait_n in zip(_rs_ns, _rs_waits):
            g.erase_node(wait_n)
            g.erase_node(rs_n)


def merge_all_gather(
    gm: torch.fx.GraphModule, ag_buckets: list[list[torch.fx.Node]]
) -> None:  # type: ignore[union-attr]
    """
    Merges specified buckets of all_gather to joint all_gather.
    """
    from torch.distributed.distributed_c10d import _resolve_process_group

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "fx_bucketing_passes_all_gather_buckets",
            "encoding": "string",
        },
        payload_fn=lambda: str(ag_buckets),
    )
    n_buckets = len(ag_buckets)

    ag_node_to_pre_nodes = defaultdict(list)

    ag_ins: list[list[torch.fx.Node]] = [[] for _ in range(n_buckets)]
    ag_waits: list[list[torch.fx.Node]] = [[] for _ in range(n_buckets)]
    for bucket_idx, ag_bucket in enumerate(ag_buckets):
        _, group_size, group_name = ag_bucket[0].args
        assert isinstance(group_name, str)
        dtype = ag_bucket[0].meta["val"].dtype

        for ag_node in ag_bucket:
            assert len(ag_node.users) == 1, (
                f"Expect only one user for {ag_node}, but got {ag_node.users}"
            )
            wait_node = next(iter(ag_node.users))
            assert (
                ag_node.args[1] == group_size
                and ag_node.args[2] == group_name
                and ag_node.meta["val"].dtype == dtype
            )
            ag_node_in = ag_node.args[0]
            if (
                ag_node_in.op == "call_function"  # type: ignore[union-attr]
                and ag_node_in.target == torch.ops.prims.convert_element_type.default  # type: ignore[union-attr]
                and len(ag_node_in.users) == 1  # type: ignore[union-attr]
            ):
                ag_node_to_pre_nodes[ag_node].append(ag_node_in)
                ag_node_in = ag_node_in.args[0]  # type: ignore[union-attr]

            ag_ins[bucket_idx].append(ag_node_in)  # type: ignore[union-attr, arg-type]
            ag_waits[bucket_idx].append(wait_node)

    g = gm.graph

    for bucket_idx in range(n_buckets):
        _ag_ins = ag_ins[bucket_idx]
        _ag_waits = ag_waits[bucket_idx]
        _ag_ns = ag_buckets[bucket_idx]

        ag0 = _ag_ns[0]
        ag0_val = ag0.meta["val"]
        _, group_size, group_name = ag0.args
        dtype = ag0_val.dtype
        assert isinstance(group_name, str)

        rank: int = dist.get_rank(_resolve_process_group(group_name))

        replacements = _insert_fn_trace_before_node(
            g,
            all_gather_merge_fn_to_trace,
            (
                pytree.tree_map(lambda node: node.meta["val"], _ag_ins),
                group_size,
                group_name,
                dtype,
                rank,
            ),
            ag0.next,
            _ag_ins,
            _ag_waits,
        )

        # See Note: [Replacement in bucketing passes]
        def _replace(x: torch.fx.Node) -> torch.fx.Node:
            return replacements.get(x, x)

        for j in range(bucket_idx + 1, n_buckets):
            ag_ins[j] = pytree.tree_map(_replace, ag_ins[j])

        # Erasing old nodes in reverse order
        for ag_n, wait_n in zip(ag_buckets[bucket_idx], _ag_waits):
            g.erase_node(wait_n)
            g.erase_node(ag_n)
            for n in reversed(ag_node_to_pre_nodes[ag_n]):
                g.erase_node(n)  # type: ignore[arg-type]
