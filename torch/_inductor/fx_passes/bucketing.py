import collections
import logging
from collections import defaultdict
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
import torch.utils._pytree as pytree
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import detect_fake_mode
from torch._inductor.runtime.runtime_utils import dynamo_timed
from torch._logging import trace_structured
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._ordered_set import OrderedSet


logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Helper functions moved to top for better organization
def _ag_group_key(node: torch.fx.Node) -> tuple[str, torch.dtype]:
    _, group_size, group_name = node.args
    dtype = node.meta["val"].dtype
    assert isinstance(group_name, str)
    return (group_name, dtype)


def _rs_group_key(node: torch.fx.Node) -> tuple[str, str, torch.dtype]:
    _, reduce_op, group_size, group_name = node.args
    dtype = node.meta["val"].dtype
    assert isinstance(group_name, str)
    assert isinstance(reduce_op, str)
    return (group_name, reduce_op, dtype)


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
    mode: Optional[str] = None,
) -> None:
    if bucket_cap_mb_by_bucket_idx is None:
        from torch._inductor.fx_passes.bucketing import (
            bucket_cap_mb_by_bucket_idx_default,
        )

        bucket_cap_mb_by_bucket_idx = bucket_cap_mb_by_bucket_idx_default
    ag_buckets = bucket_all_gather_by_mb(gm, bucket_cap_mb_by_bucket_idx)
    if len(ag_buckets) == 0:
        return
    merge_all_gather(gm, ag_buckets, mode)


def bucket_reduce_scatter(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Optional[Callable[[int], float]] = None,
    mode: Optional[str] = None,
) -> None:
    if bucket_cap_mb_by_bucket_idx is None:
        from torch._inductor.fx_passes.bucketing import (
            bucket_cap_mb_by_bucket_idx_default,
        )

        bucket_cap_mb_by_bucket_idx = bucket_cap_mb_by_bucket_idx_default
    rs_buckets = bucket_reduce_scatter_by_mb(gm, bucket_cap_mb_by_bucket_idx)
    if len(rs_buckets) == 0:
        return
    merge_reduce_scatter(gm, rs_buckets, mode)


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


def collect_node_descendants(
    graph: torch.fx.Graph,
) -> dict[torch.fx.Node, OrderedSet[torch.fx.Node]]:
    """
    Collects the descendants of each node in the graph.
    Args:
        graph (torch.fx.Graph): The graph to collect descendants from.
    Returns:
        dict[torch.fx.Node, OrderedSet[torch.fx.Node]]: A dictionary mapping each node to its descendants.
    """
    node_descendants: dict[torch.fx.Node, OrderedSet[torch.fx.Node]] = (
        collections.defaultdict(OrderedSet)
    )
    outdegree = collections.defaultdict(int)
    queue = []

    for node in graph.nodes:
        n_outdegree = len(node.users)
        if n_outdegree == 0:
            queue.append(node)
        else:
            outdegree[node] = len(node.users)

    while queue:
        node = queue.pop()
        for input_node in node.all_input_nodes:
            node_descendants[input_node] |= node_descendants[node]
            node_descendants[input_node].add(node)
            outdegree[input_node] -= 1

            if outdegree[input_node] == 0:
                queue.append(input_node)

    return node_descendants


def greedy_bucket_collective_by_mb(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float],
    filter_node: Callable[[torch.fx.Node], bool],
    node_group_key: Callable[[torch.fx.Node], Any],
    filter_wait_node: Optional[Callable[[torch.fx.Node], bool]] = None,
) -> list[list[torch.fx.Node]]:
    """
    Bucketing adjacent collectives with equal node_group_key.
    We can not bucket non adjacent collectives,
    as this will effectively change the order of collectives.
    Reordering can lead to different order on different ranks.
    """
    g = gm.graph
    found_candidates = False
    for node in g.nodes:
        if filter_node(node):
            found_candidates = True
            break
    if not found_candidates:
        return []

    # TODO: pearce kelly algorithm for detecting cycles
    node_descendents = collect_node_descendants(gm.graph)

    nodes_groups: list[list[torch.fx.Node]] = []
    cur_group: list[torch.fx.Node] = []
    cur_group_key = None

    for node in g.nodes:
        if is_wait_tensor(node) and filter_node(node.args[0]):
            if (filter_wait_node is None) or filter_wait_node(node):
                coll_node = node.args[0]
                group_key = node_group_key(coll_node)
                if group_key == cur_group_key:
                    cur_group.append(coll_node)
                else:
                    if len(cur_group) > 1:
                        nodes_groups.append(cur_group)
                    cur_group = [coll_node]
                    cur_group_key = group_key

    if len(cur_group) > 1:
        nodes_groups.append(cur_group)

    buckets: list[list[torch.fx.Node]] = []
    for nodes in nodes_groups:
        cur_bucket: list[torch.fx.Node] = []
        cur_bucket_descendents: OrderedSet[torch.fx.Node] = OrderedSet()
        cur_bucket_size_bytes: int = 0
        cur_bucket_id: int = 0
        bucket_size_bytes = int(
            bucket_cap_mb_by_bucket_idx(cur_bucket_id) * 1024 * 1024
        )
        for node in nodes:
            if node in cur_bucket_descendents:
                # if there is a path from node to the current bucket, we cannot horizontally fuse (bucket)
                continue
            assert "val" in node.meta
            n_val = node.meta["val"]
            out_size_bytes = n_val.numel() * n_val.element_size()
            n_input_val = node.all_input_nodes[0].meta["val"]
            in_size_bytes = n_input_val.numel() * n_input_val.element_size()
            size_bytes = max(out_size_bytes, in_size_bytes)
            if cur_bucket_size_bytes + size_bytes > bucket_size_bytes and cur_bucket:
                # Current bucket is full, create new bucket
                if len(cur_bucket) > 1:
                    buckets.append(cur_bucket)
                cur_bucket = []
                cur_bucket_size_bytes = 0
                cur_bucket_id += 1
                cur_bucket_descendents = OrderedSet()
            cur_bucket_size_bytes += size_bytes
            cur_bucket.append(node)
            cur_bucket_descendents |= node_descendents[node]
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
        bucket_cap_mb_by_bucket_idx (Callable[[int], float]): Callable to specify cap of the bucket
            in megabytes by bucket idx.  The idea of `bucket_cap_mb_by_bucket_idx` is to allow
            to specify different sizes of the buckets at the start,
            as first all_gather is usually exposed.  Interface of bucket_cap_mb_by_bucket_idx
            is `bucket_cap_mb_by_bucket_idx_default` function that is default value for `bucket_cap_mb_by_bucket_idx`.
        filter_wait_node (Optional[Callable[[torch.fx.Node], bool]]): If specified,
            only all_gather nodes with wait_node that satisfy `filter_wait_node` will be bucketed.

    Returns:
        list[list[torch.fx.Node]]: List of buckets, where each bucket is a list of all_gather nodes.
    """

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
        bucket_cap_mb_by_bucket_idx (Callable[[int], float]): Callable to specify cap of the bucket
            in megabytes by bucket idx.  The idea of `bucket_cap_mb_by_bucket_idx` is to allow
            to specify different sizes of the buckets.
        filter_wait_node (Optional[Callable[[torch.fx.Node], bool]]): If specified,
            only reduce_scatter nodes with wait_node that satisfy `filter_wait_node` will be bucketed.

    Returns:
        list[list[torch.fx.Node]]: List of buckets, where each bucket is a list of reduce_scatter nodes.
    """

    return greedy_bucket_collective_by_mb(
        gm,
        bucket_cap_mb_by_bucket_idx,
        is_reduce_scatter_tensor,
        _rs_group_key,
        filter_wait_node,
    )


@torch.library.custom_op("bucketing::_pre_bucket_reduce_scatter", mutates_args={})
def _pre_bucket_reduce_scatter(
    rs_ins: list[torch.Tensor],
    group_size: int,
) -> torch.Tensor:
    rs_ins_flattened = [x.view(group_size, -1) for x in rs_ins]
    new_rs_in = torch.cat(rs_ins_flattened, dim=1).flatten()
    return new_rs_in


def _pre_bucket_reduce_scatter_fake(
    rs_ins: list[torch.Tensor],
    group_size: int,
) -> torch.Tensor:
    out_numel = sum(rs_in.numel() for rs_in in rs_ins)
    return torch.empty((out_numel,), device=rs_ins[0].device, dtype=rs_ins[0].dtype)


_pre_bucket_reduce_scatter.register_fake(_pre_bucket_reduce_scatter_fake)


def reduce_scatter_merge_fn_to_trace_custom_ops(
    rs_ins: list[torch.Tensor],
    group_size: int,
    group_name: str,
    reduce_op: str,
    reduce_dtype: torch.dtype,  # type: ignore[name-defined]
    device: torch.device,  # type: ignore[name-defined]
) -> list[torch.Tensor]:  # type: ignore[no-untyped-def]
    new_out_sizes = [(x.shape[0] // group_size,) + x.shape[1:] for x in rs_ins]
    new_out_numels = [x.numel() // group_size for x in rs_ins]

    new_rs_in = torch.ops.bucketing._pre_bucket_reduce_scatter(rs_ins, group_size)

    # TODO - either use torch.cat or make sure inductor foreach codegen
    # fires more reliably
    new_rs_out = torch.ops.c10d_functional.wait_tensor(
        torch.ops._c10d_functional.reduce_scatter_tensor.default(
            new_rs_in, reduce_op, group_size, group_name
        )
    )
    new_out_flat = new_rs_out.split(new_out_numels, 0)
    new_outs = [x.view(s) for x, s in zip(new_out_flat, new_out_sizes)]
    return new_outs


def reduce_scatter_merge_fn_to_trace(
    rs_ins: list[torch.Tensor],
    group_size: int,
    group_name: str,
    reduce_op: str,
    reduce_dtype: torch.dtype,  # type: ignore[name-defined]
    device: torch.device,  # type: ignore[name-defined]
) -> list[torch.Tensor]:  # type: ignore[no-untyped-def]
    rs_ins_flattened = [x.view(group_size, -1) for x in rs_ins]

    new_out_sizes = [(x.shape[0] // group_size,) + x.shape[1:] for x in rs_ins]
    new_out_numels = [x.numel() // group_size for x in rs_ins]

    new_rs_in = torch.cat(rs_ins_flattened, dim=1).flatten()

    new_rs_out = torch.ops.c10d_functional.wait_tensor(
        torch.ops._c10d_functional.reduce_scatter_tensor.default(
            new_rs_in, reduce_op, group_size, group_name
        )
    )
    new_out_flat = new_rs_out.split(new_out_numels, 0)
    new_outs = [x.view(s) for x, s in zip(new_out_flat, new_out_sizes)]
    return new_outs


@torch.library.custom_op("bucketing::_pre_bucket_all_gather", mutates_args={})
def _pre_bucket_all_gather(
    ag_ins: list[torch.Tensor],
    group_size: int,
    group_name: str,
    dtype: torch.dtype,  # type: ignore[name-defined]
    rank: int,
) -> torch.Tensor:
    ins_split_sizes = [ag_in.numel() for ag_in in ag_ins]
    ag_input_numel = sum(ins_split_sizes)
    device = ag_ins[0].device
    new_ag_out = torch.empty(ag_input_numel * group_size, dtype=dtype, device=device)
    new_ag_in = new_ag_out.narrow(0, ag_input_numel * rank, ag_input_numel)
    foreach_copy_dsts = torch.split(new_ag_in, ins_split_sizes)
    ag_ins_flattened = [ag_in.reshape(-1) for ag_in in ag_ins]
    torch._foreach_copy_(foreach_copy_dsts, ag_ins_flattened)
    return new_ag_out


def _pre_bucket_all_gather_fake(
    ag_ins: list[torch.Tensor],
    group_size: int,
    group_name: str,
    dtype: torch.dtype,  # type: ignore[name-defined]
    rank: int,
) -> torch.Tensor:
    ins_split_sizes = [ag_in.numel() for ag_in in ag_ins]
    ag_input_numel = sum(ins_split_sizes)
    device = ag_ins[0].device
    new_ag_out = torch.empty(ag_input_numel * group_size, dtype=dtype, device=device)
    return new_ag_out


_pre_bucket_all_gather.register_fake(_pre_bucket_all_gather_fake)


def all_gather_merge_fn_to_trace_custom_ops(
    ag_ins: list[torch.Tensor],
    group_size: int,
    group_name: str,
    dtype: torch.dtype,  # type: ignore[name-defined]
    rank: int,
) -> list[torch.Tensor]:
    ins_sizes = [ag_in.shape for ag_in in ag_ins]
    ins_split_sizes = [ag_in.numel() for ag_in in ag_ins]
    ag_input_numel = sum(ins_split_sizes)
    new_ag_out = torch.ops.bucketing._pre_bucket_all_gather(
        ag_ins, group_size, group_name, dtype, rank
    )
    new_ag_in = new_ag_out.narrow(0, ag_input_numel * rank, ag_input_numel)
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
    with dynamo_timed("fx.bucketing._trace", log_pt2_compile_event=True):
        fake_mode = detect_fake_mode(inps)
        assert fake_mode is not None
        with fake_mode, enable_python_dispatcher():
            out = make_fx(fn)(*inps)
            for node in out.graph.find_nodes(
                op="call_function", target=torch.ops.aten.detach.default
            ):
                node.replace_all_uses_with(node.args[0])
                out.graph.erase_node(node)
            return out


def _insert_fn_trace_before_node(  # type: ignore[no-untyped-def]
    g: torch.fx.Graph,
    fn_to_trace,
    inps,
    insert_before_node: torch.fx.Node,
    g_fn_inps: list[torch.fx.Node],
    g_fn_outs: list[torch.fx.Node],
) -> tuple[dict[torch.fx.Node, torch.fx.Node], list[torch.fx.Node]]:  # type: ignore[no-untyped-def]
    """
    Helper function that traces :attr:`fn_to_trace` with inputs
    :attr:`inps`.
    The result function graph will be inserted before :attr:`insert_before_node`,
    using :attr:`g_fn_inps` nodes of original graph as inputs of function graph,
    function graph outputs will replace :attr:`g_fn_outs` in original graph.

    Returns:
        (replacements, new_nodes): Dictionary mapping old to new nodes, and list of all newly inserted nodes
    """
    with dynamo_timed(
        "fx.bucketing._insert_fn_trace_before_node", log_pt2_compile_event=True
    ):
        fn_gm = _trace(
            fn_to_trace,
            inps,
        )
        fn_g = fn_gm.graph
        fn_g_ins = fn_g.find_nodes(op="placeholder")
        env = {fn_g_ins[idx]: g_fn_inps[idx] for idx in range(len(g_fn_inps))}
        g_fn_new_outs: list[torch.fx.Node] = []
        new_nodes: list[torch.fx.Node] = []  # Track all newly inserted nodes

        with g.inserting_before(insert_before_node):
            for _n in fn_g.nodes:
                if _n.op == "placeholder":
                    continue
                _new_n = g.node_copy(_n, lambda x: env[x])
                env[_n] = _new_n
                if _n.op == "output":
                    g_fn_new_outs = _new_n.args[0]  # type: ignore[assignment]
                    g.erase_node(_new_n)
                else:
                    new_nodes.append(_new_n)  # Track non-output nodes

        replacements = {  # noqa: C416
            orig_out: new_out for orig_out, new_out in zip(g_fn_outs, g_fn_new_outs)
        }
        for orig_out, new_out in zip(g_fn_outs, g_fn_new_outs):
            orig_out.replace_all_uses_with(new_out)

        return replacements, new_nodes


def process_collective_bucket(
    g: torch.fx.Graph,
    bucket_nodes: list[torch.fx.Node],
    fn_to_trace: Callable[..., list[torch.Tensor]],
    trace_args_fn: Callable[[list[torch.fx.Node]], tuple[Any, ...]],
    insert_before: Optional[torch.fx.Node] = None,
    wait_insertion_point: Optional[torch.fx.Node] = None,
) -> tuple[list[torch.fx.Node], dict[torch.fx.Node, torch.fx.Node]]:
    """
    Process a single bucket of collective operation nodes with flexible insertion control.

    Args:
        g: The graph to modify
        bucket_nodes: Nodes in the current bucket to process
        fn_to_trace: Function to trace and insert
        trace_args_fn: Function to create trace arguments from inputs
        insert_before: Where to insert the traced function (default: after last bucket node)
        wait_insertion_point: If provided, move all nodes from wait() onwards to before this node

    Returns:
        new_nodes: List of all newly inserted nodes
        replacements: Dictionary mapping old wait nodes to new output nodes
    """
    # Collect inputs and waits from current bucket
    bucket_ins: list[torch.fx.Node] = []
    bucket_waits: list[torch.fx.Node] = []
    ag_node_to_pre_nodes: dict[torch.fx.Node, list[torch.fx.Node]] = defaultdict(list)

    for n in bucket_nodes:
        assert len(n.users) == 1, f"Expected single user for {n}, got {n.users}"
        wait_n = next(iter(n.users))

        # Handle convert_element_type operations (for all_gather)
        node_in = n.args[0]
        if (
            is_all_gather_into_tensor(n)
            and isinstance(node_in, torch.fx.Node)  # Add type check
            and node_in.op == "call_function"
            and node_in.target == torch.ops.prims.convert_element_type.default
            and len(node_in.users) == 1
        ):
            ag_node_to_pre_nodes[n].append(node_in)
            node_in = node_in.args[0]

        assert isinstance(node_in, torch.fx.Node)  # Ensure node_in is a Node
        bucket_ins.append(node_in)
        bucket_waits.append(wait_n)

    # Create trace arguments
    trace_args = trace_args_fn(bucket_ins)

    # Determine insertion point
    if insert_before is None:
        insert_before = bucket_nodes[-1].next

    # Insert traced function and get replacements + new nodes
    replacements, new_nodes = _insert_fn_trace_before_node(
        g,
        fn_to_trace,
        trace_args,
        insert_before,
        bucket_ins,
        bucket_waits,
    )

    # If requested, move wait nodes and everything after to specified location
    if wait_insertion_point is not None:
        # Find the first wait node in new_nodes
        wait_start_idx = None
        for i, node in enumerate(new_nodes):
            if is_wait_tensor(node):
                wait_start_idx = i
                break

        # Move all nodes from wait onwards (including the wait)
        if wait_start_idx is not None:
            nodes_to_move = new_nodes[wait_start_idx:]
            for node in nodes_to_move:
                wait_insertion_point.prepend(node)

    # Erase old nodes
    for node, wait_n in zip(bucket_nodes, bucket_waits):
        g.erase_node(wait_n)
        g.erase_node(node)
        # Erase any convert_element_type nodes we tracked
        for pre_node in reversed(ag_node_to_pre_nodes[node]):
            g.erase_node(pre_node)

    return new_nodes, replacements


def merge_reduce_scatter_bucket(
    g: torch.fx.Graph,
    rs_nodes: list[torch.fx.Node],
    mode: Optional[str] = None,
    insert_before: Optional[torch.fx.Node] = None,
    wait_insertion_point: Optional[torch.fx.Node] = None,
) -> tuple[list[torch.fx.Node], dict[torch.fx.Node, torch.fx.Node]]:
    # Validate bucket consistency
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

    # Choose merge function based on mode
    rs_merge_fn = reduce_scatter_merge_fn_to_trace
    if mode and "custom_ops" in mode:
        rs_merge_fn = reduce_scatter_merge_fn_to_trace_custom_ops

    # Process bucket with lazy input collection
    def create_trace_args(bucket_ins: list[torch.fx.Node]) -> tuple[Any, ...]:
        return (
            pytree.tree_map(lambda node: node.meta["val"], bucket_ins),
            group_size,
            group_name,
            reduce_op,
            reduce_dtype,
            device,
        )

    return process_collective_bucket(
        g,
        rs_nodes,
        rs_merge_fn,
        create_trace_args,
        insert_before=insert_before,
        wait_insertion_point=wait_insertion_point,
    )


def merge_all_gather_bucket(
    g: torch.fx.Graph,
    ag_nodes: list[torch.fx.Node],
    mode: Optional[str] = None,
    insert_before: Optional[torch.fx.Node] = None,
    wait_insertion_point: Optional[torch.fx.Node] = None,
) -> tuple[list[torch.fx.Node], dict[torch.fx.Node, torch.fx.Node]]:
    from torch.distributed.distributed_c10d import _resolve_process_group

    ag0 = ag_nodes[0]
    ag0_val = ag0.meta["val"]
    _, group_size, group_name = ag0.args
    dtype = ag0_val.dtype
    assert isinstance(group_name, str)

    for n in ag_nodes:
        assert (
            n.args[1] == group_size
            and n.args[2] == group_name
            and n.meta["val"].dtype == dtype
        )

    # Choose merge function based on mode
    ag_merge_fn = all_gather_merge_fn_to_trace
    if mode and "custom_ops" in mode:
        ag_merge_fn = all_gather_merge_fn_to_trace_custom_ops

    # Process bucket with lazy input collection
    rank: int = dist.get_rank(_resolve_process_group(group_name))

    def create_trace_args(bucket_ins: list[torch.fx.Node]) -> tuple[Any, ...]:
        return (
            pytree.tree_map(lambda node: node.meta["val"], bucket_ins),
            group_size,
            group_name,
            dtype,
            rank,
        )

    return process_collective_bucket(
        g,
        ag_nodes,
        ag_merge_fn,
        create_trace_args,
        wait_insertion_point=wait_insertion_point,
    )


def merge_reduce_scatter(
    gm: torch.fx.GraphModule,
    rs_buckets: list[list[torch.fx.Node]],
    mode: Optional[str] = None,
) -> None:
    """
    Merges specified buckets of reduce_scatter to joint reduce_scatter.
    """
    with dynamo_timed("fx.bucketing.merge_reduce_scatter", log_pt2_compile_event=True):
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "fx_bucketing_passes_reduce_scatter_buckets",
                "encoding": "string",
            },
            payload_fn=lambda: str(rs_buckets),
        )

        g = gm.graph

        for rs_nodes in rs_buckets:
            merge_reduce_scatter_bucket(g, rs_nodes, mode)


def merge_all_gather(
    gm: torch.fx.GraphModule,
    ag_buckets: list[list[torch.fx.Node]],
    mode: Optional[str] = None,
) -> None:
    """
    Merges specified buckets of all_gather to joint all_gather.
    """
    with dynamo_timed("fx.bucketing.merge_all_gather", log_pt2_compile_event=True):
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "fx_bucketing_passes_all_gather_buckets",
                "encoding": "string",
            },
            payload_fn=lambda: str(ag_buckets),
        )

        g = gm.graph

        for ag_nodes in ag_buckets:
            merge_all_gather_bucket(g, ag_nodes, mode)
