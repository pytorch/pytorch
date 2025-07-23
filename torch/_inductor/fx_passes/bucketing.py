import logging
from collections import defaultdict
from typing import Callable, Optional

import torch
import torch.utils._pytree as pytree
from torch._dispatch.python import enable_python_dispatcher
from torch._logging import trace_structured
from torch._subclasses.fake_tensor import maybe_get_fake_mode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._ordered_set import OrderedSet


logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def bucket_size_determinator(bucket_id: int) -> float:
    """
    Determine the size of a bucket based on its ID.

    Args:
    bucket_id (int): The ID of the bucket.

    Returns:
    float: The size of the bucket.
    """
    return 2000.0


def bucket_all_gather(
    gm: torch.fx.GraphModule, all_gather_bucket_cap_mb_callback: Callable[[int], float]
) -> None:
    ag_buckets = bucket_all_gather_by_mb(gm, all_gather_bucket_cap_mb_callback)
    if len(ag_buckets) == 0:
        return
    merge_all_gather(gm, ag_buckets)


def bucket_reduce_scatter(
    gm: torch.fx.GraphModule,
    reduce_scatter_bucket_cap_mb_callback: Callable[[int], float],
) -> None:
    rs_buckets = bucket_reduce_scatter_by_mb(gm, reduce_scatter_bucket_cap_mb_callback)
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


def bucket_all_gather_by_mb(
    gm: torch.fx.GraphModule,
    all_gather_bucket_cap_mb_callback: Callable[[int], float],
    filter_wait_node: Optional[Callable[[torch.fx.Node], bool]] = None,
) -> list[list[torch.fx.Node]]:
    """
    Identifies all all_gather nodes and groups them into buckets based on size limit `all_gather_bucket_cap_mb_callback`.
    Returns a list of buckets, where each bucket is a list of all_gather nodes.
    """
    node_list = gm.graph.nodes
    # Prerequisite: Check if there is any all_gather node
    found_all_gather = False
    for node in node_list:
        if is_all_gather_into_tensor(node):
            found_all_gather = True
            break
    if not found_all_gather:
        return []
    group_name_ag_nodes: dict[tuple[str, torch.dtype], list[torch.fx.Node]] = (  # type: ignore[name-defined]
        defaultdict(list)
    )
    # Step 1: Find all all_gather nodes
    for node in node_list:
        if is_wait_tensor(node) and is_all_gather_into_tensor(node.args[0]):
            if (filter_wait_node is None) or filter_wait_node(node):
                ag_node = node.args[0]
                _, group_size, group_name = ag_node.args
                dtype = ag_node.meta["val"].dtype
                assert isinstance(group_name, str)
                group_name_ag_nodes[(group_name, dtype)].append(ag_node)
    # Step 2: Put all_gather nodes into buckets
    ag_buckets: list[list[torch.fx.Node]] = []
    for (group_name, dtype), ag_nodes in group_name_ag_nodes.items():
        cur_bucket: list[torch.fx.Node] = []
        cur_bucket_recursive_users: OrderedSet[torch.fx.Node] = OrderedSet()
        cur_bucket_size_bytes: int = 0
        cur_bucket_id: int = 0
        all_gather_bucket_size_bytes = int(
            all_gather_bucket_cap_mb_callback(cur_bucket_id) * 1024 * 1024
        )
        for ag_node in ag_nodes:
            assert is_all_gather_into_tensor(ag_node)
            if ag_node in cur_bucket_recursive_users:
                # We can not bucket successors with the node
                continue
            assert "val" in ag_node.meta
            ag_n_val = ag_node.meta["val"]
            ag_output_size_bytes = ag_n_val.numel() * ag_n_val.element_size()
            if (
                cur_bucket_size_bytes + ag_output_size_bytes
                > all_gather_bucket_size_bytes
                and cur_bucket
            ):
                # Current bucket is full, create new bucket
                if len(cur_bucket) > 1:
                    ag_buckets.append(cur_bucket)
                cur_bucket = []
                cur_bucket_size_bytes = 0
                cur_bucket_id += 1
            cur_bucket_size_bytes += ag_output_size_bytes
            cur_bucket.append(ag_node)
            find_recursive_users_of_fx_node(ag_node, cur_bucket_recursive_users)
        if len(cur_bucket) > 1:
            ag_buckets.append(cur_bucket)
    return ag_buckets


def bucket_reduce_scatter_by_mb(
    gm: torch.fx.GraphModule,
    reduce_scatter_bucket_cap_mb_callback: Callable[[int], float],
    filter_wait_node: Optional[Callable[[torch.fx.Node], bool]] = None,
) -> list[list[torch.fx.Node]]:
    """
    Identifies all reduce_scatter nodes and groups them into buckets based on size limit `reduce_scatter_bucket_cap_mb_callback`.
    Returns a list of buckets, where each bucket is a list of reduce_scatter nodes.
    """
    node_list = list(gm.graph.nodes)
    # Prerequisite: Check if there is any reduce_scatter node
    found_reduce_scatter = False
    for node in node_list:
        if is_reduce_scatter_tensor(node):
            found_reduce_scatter = True
            break
    if not found_reduce_scatter:
        return []
    group_name_rs_nodes: dict[tuple[str, str, torch.dtype], list[torch.fx.Node]] = (  # type: ignore[name-defined]
        defaultdict(list)
    )
    # Step 1: Find all reduce_scatter nodes
    for node in node_list:
        if is_wait_tensor(node) and is_reduce_scatter_tensor(node.args[0]):
            if (filter_wait_node is None) or filter_wait_node(node):
                rs_node = node.args[0]
                _, reduce_op, group_size, group_name = rs_node.args
                dtype = rs_node.meta["val"].dtype
                assert isinstance(group_name, str)
                assert isinstance(reduce_op, str)
                group_name_rs_nodes[(group_name, reduce_op, dtype)].append(rs_node)
    # Step 2: Put reduce_scatter nodes into buckets
    rs_buckets: list[list[torch.fx.Node]] = []
    for (group_name, reduce_op, dtype), rs_nodes in group_name_rs_nodes.items():
        cur_bucket: list[torch.fx.Node] = []
        cur_bucket_recursive_users: OrderedSet[torch.fx.Node] = OrderedSet()
        cur_bucket_size_bytes: int = 0
        cur_bucket_id: int = 0
        # Convert MiB to bytes
        reduce_scatter_bucket_size_bytes = int(
            reduce_scatter_bucket_cap_mb_callback(cur_bucket_id) * 1024 * 1024
        )
        for rs_node in rs_nodes:
            assert is_reduce_scatter_tensor(rs_node)
            if rs_node in cur_bucket_recursive_users:
                # We can not bucket successors with the node
                continue
            rs_input = rs_node.args[0]
            assert "val" in rs_input.meta  # type: ignore[union-attr]
            rs_in_val = rs_input.meta["val"]  # type: ignore[union-attr]
            rs_input_size_bytes = rs_in_val.numel() * rs_in_val.element_size()
            if (
                cur_bucket_size_bytes + rs_input_size_bytes
                > reduce_scatter_bucket_size_bytes
                and cur_bucket
            ):
                # Current bucket is full, create new bucket
                if len(cur_bucket) > 1:
                    rs_buckets.append(cur_bucket)
                cur_bucket = []
                cur_bucket_size_bytes = 0
                cur_bucket_id += 1
                reduce_scatter_bucket_size_bytes = int(
                    reduce_scatter_bucket_cap_mb_callback(cur_bucket_id) * 1024 * 1024
                )
            cur_bucket_size_bytes += rs_input_size_bytes
            cur_bucket.append(rs_node)
            find_recursive_users_of_fx_node(rs_node, cur_bucket_recursive_users)
        if cur_bucket:
            if len(cur_bucket) > 1:
                rs_buckets.append(cur_bucket)
    return rs_buckets


def _rank_idx_dict(group_name: str) -> dict[int, int]:
    from torch.distributed.distributed_c10d import (
        _resolve_process_group,
        get_process_group_ranks,
    )

    pg = _resolve_process_group(group_name)
    ranks = get_process_group_ranks(pg)
    rank_idx_dict: dict[int, int] = {rank: idx for idx, rank in enumerate(ranks)}
    return rank_idx_dict


def find_recursive_users_of_fx_node(node, collected_node_set) -> None:  # type: ignore[no-untyped-def]
    for user_node in node.users:
        if user_node in collected_node_set:
            continue
        collected_node_set.add(user_node)
        find_recursive_users_of_fx_node(
            user_node,
            collected_node_set,
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
    local_rank: int,
) -> list[torch.Tensor]:
    ins_sizes = [ag_in.shape for ag_in in ag_ins]
    ins_split_sizes = [ag_in.numel() for ag_in in ag_ins]
    ag_input_numel = sum(ins_split_sizes)
    device = ag_ins[0].device
    new_ag_out = torch.empty(ag_input_numel * group_size, dtype=dtype, device=device)
    new_ag_in = new_ag_out.narrow(0, ag_input_numel * local_rank, ag_input_numel)
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


def all_gather_merge_fn_to_trace_fsdp_ag_copy_in(
    ag_ins: list[torch.Tensor],
    group_size: int,
    group_name: str,
    dtype: torch.dtype,  # type: ignore[name-defined]
    local_rank: int,
) -> list[torch.Tensor]:
    # Implementation that is functional in graph,
    # but uses custom op torch.ops.fsdp.all_gather_copy_in.
    ins_sizes = [ag_in.shape for ag_in in ag_ins]
    ins_split_sizes = [ag_in.numel() for ag_in in ag_ins]
    ag_input_numel = sum(ins_split_sizes)
    device = ag_ins[0].device
    new_ag_out = torch.empty(ag_input_numel * group_size, dtype=dtype, device=device)
    ag_ins_flattened = [ag_in.reshape(-1) for ag_in in ag_ins]
    new_ag_in, new_ag_out = torch.ops.fsdp.all_gather_copy_in(
        ag_ins_flattened, new_ag_out, ins_split_sizes, ag_input_numel, local_rank
    )
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
    local_rank: int,
) -> list[torch.Tensor]:
    # Suboptimal functional implementation without mutations.
    ins_sizes = [ag_in.shape for ag_in in ag_ins]
    ins_split_sizes = [ag_in.numel() for ag_in in ag_ins]
    ag_input_numel = sum(ins_split_sizes)
    device = ag_ins[0].device
    new_ag_out = torch.empty(ag_input_numel * group_size, dtype=dtype, device=device)
    ag_ins_flattened = [ag_in.reshape(-1) for ag_in in ag_ins]
    new_ag_in = torch.cat(ag_ins_flattened, dim=0)
    new_ag_in = new_ag_out.narrow(0, ag_input_numel * local_rank, ag_input_numel)
    foreach_copy_dsts = torch.split(new_ag_in, ins_split_sizes)
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


def _trace(fn, inps) -> torch.fx.GraphModule:  # type: ignore[no-untyped-def]
    fake_mode = maybe_get_fake_mode(inps[0][0])
    assert fake_mode is not None
    with fake_mode, enable_python_dispatcher():
        return make_fx(fn)(*inps)


def _insert_fn_trace_before_node(  # type: ignore[no-untyped-def]
    g: torch.fx.Graph,
    fn_to_trace,
    inps,
    insert_before_node,
    g_fn_inps,
    g_fn_outs,
) -> dict[torch.fx.Node, torch.fx.Node]:  # type: ignore[no-untyped-def]
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
    rs_to_bucket_idx: dict[torch.fx.Node, int] = {}
    for bucket_idx, rs_nodes in enumerate(rs_buckets):
        for rs_node in rs_nodes:
            rs_to_bucket_idx[rs_node] = bucket_idx

    for bucket_idx, rs_bucket in enumerate(rs_buckets):
        for n in rs_bucket:
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

        def _replace(x: torch.fx.Node) -> torch.fx.Node:
            return replacements.get(x, x)

        for j in range(bucket_idx + 1, n_buckets):
            rs_ins[j] = pytree.tree_map(_replace, rs_ins[j])
            rs_waits[j] = pytree.tree_map(_replace, rs_waits[j])
            rs_buckets[j] = pytree.tree_map(_replace, rs_buckets[j])

        for rs_n, wait_n in zip(_rs_ns, _rs_waits):
            g.erase_node(wait_n)
            g.erase_node(rs_n)


def merge_all_gather(
    gm: torch.fx.GraphModule, ag_buckets: list[list[torch.fx.Node]]
) -> None:  # type: ignore[union-attr]
    """
    Merges specified buckets of all_gather to joint all_gather.
    """
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

    group_name_to_rank_idx_dict: dict[str, dict[int, int]] = {}
    ag_ins: list[list[torch.fx.Node]] = [[] for _ in range(n_buckets)]
    ag_waits: list[list[torch.fx.Node]] = [[] for _ in range(n_buckets)]
    for bucket_idx, ag_bucket in enumerate(ag_buckets):
        _, group_size, group_name = ag_bucket[0].args
        assert isinstance(group_name, str)
        dtype = ag_bucket[0].meta["val"].dtype
        if group_name not in group_name_to_rank_idx_dict:
            group_name_to_rank_idx_dict[group_name] = _rank_idx_dict(group_name)  # type: ignore[index]

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
        device = ag0_val.device
        assert isinstance(group_name, str)

        rank_idx_dict = group_name_to_rank_idx_dict[group_name]
        rank = device.index
        local_rank = rank_idx_dict[rank]

        replacements = _insert_fn_trace_before_node(
            g,
            all_gather_merge_fn_to_trace,
            (
                pytree.tree_map(lambda node: node.meta["val"], _ag_ins),
                group_size,
                group_name,
                dtype,
                local_rank,
            ),
            ag0.next,
            _ag_ins,
            _ag_waits,
        )

        def _replace(x: torch.fx.Node) -> torch.fx.Node:
            return replacements.get(x, x)

        for j in range(bucket_idx + 1, n_buckets):
            ag_ins[j] = pytree.tree_map(_replace, ag_ins[j])
            ag_waits[j] = pytree.tree_map(_replace, ag_waits[j])
            ag_buckets[j] = pytree.tree_map(_replace, ag_buckets[j])
        # Erasing old nodes in reverse order
        for ag_n, wait_n in zip(ag_buckets[bucket_idx], _ag_waits):
            g.erase_node(wait_n)
            g.erase_node(ag_n)
            for n in reversed(ag_node_to_pre_nodes[ag_n]):
                g.erase_node(n)  # type: ignore[arg-type]
