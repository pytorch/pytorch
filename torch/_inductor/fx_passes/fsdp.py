import logging
from collections import defaultdict
from typing import Callable

import torch
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import maybe_get_fake_mode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._ordered_set import OrderedSet


logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def bucket_fsdp_all_gather_concat(
    gm: torch.fx.GraphModule, all_gather_bucket_cap_mb_callback: Callable[[int], float]
) -> None:
    """
    Bucketing pass for SimpleFSDP all_gather ops.

    Attributes:
        gm (torch.fx.GraphModule): Graph module of the graph.
        all_gather_bucket_cap_mb_callback (Callable[[int], float]): callback function that
            takes in bucket id and returns size of a bucket in megabytes.

    Usage:
    ```
    def apply_simplefsdp_bucketing_passes(graph: torch.fx.Graph) -> None:
        gm = graph.owning_module
        bucket_fsdp_all_gather_concat(
            gm, all_gather_bucket_cap_mb_callback=lambda bucket_id: 2000
        )
        bucket_fsdp_reduce_scatter_concat(
            gm, reduce_scatter_bucket_cap_mb_callback=lambda bucket_id: 2000
        )


    # NOTE: this overwrites the existing post_grad_custom_post_pass
    torch._inductor.config.post_grad_custom_post_pass = (
        apply_simplefsdp_bucketing_passes
    )
    ```
    """

    ag_buckets = bucket_all_gather_by_mb(
        gm, all_gather_bucket_cap_mb_callback, only_fsdp=True
    )
    if len(ag_buckets) == 0:
        return
    merge_all_gather(gm, ag_buckets)


def bucket_all_gather_concat(
    gm: torch.fx.GraphModule, all_gather_bucket_cap_mb_callback: Callable[[int], float]
) -> None:
    ag_buckets = bucket_all_gather_by_mb(
        gm, all_gather_bucket_cap_mb_callback, only_fsdp=False
    )
    if len(ag_buckets) == 0:
        return
    merge_all_gather(gm, ag_buckets)


def is_all_gather_into_tensor(node: torch.fx.Node) -> bool:  # type: ignore[arg-type]
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.all_gather_into_tensor.default
    )


def is_wait_tensor(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.wait_tensor.default
    )


def is_graph_input(node: torch.fx.Node) -> bool:
    return node.op == "placeholder"


def is_wait_tensor_from_all_gather_into_tensor(node: torch.fx.Node) -> bool:
    return is_wait_tensor(node) and is_all_gather_into_tensor(node.args[0])  # type: ignore[arg-type]


def is_wait_tensor_from_all_gather_into_tensor_fsdp(node: torch.fx.Node) -> bool:
    if is_wait_tensor(node) and is_all_gather_into_tensor(node.args[0]):  # type: ignore[arg-type]
        ag_node = node.args[0]
        # Assume all_gather_into_tensor input is either graph input
        # or dtype conversion of graph input
        if not (
            is_graph_input(ag_node.args[0])  # type: ignore[arg-type, union-attr]
            or (  # type: ignore[arg-type, union-attr]
                ag_node.args[0].op == "call_function"  # type: ignore[arg-type, union-attr]
                and ag_node.args[0].target  # type: ignore[arg-type, union-attr]
                == torch.ops.prims.convert_element_type.default  # type: ignore[arg-type, union-attr]
                and is_graph_input(ag_node.args[0].args[0])  # type: ignore[arg-type, union-attr]
            )
        ):
            return False
        return True
    return False


def bucket_all_gather_by_mb(
    gm: torch.fx.GraphModule,
    all_gather_bucket_cap_mb_callback: Callable[[int], float],
    only_fsdp: bool = True,
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

    ag_nodes: list[torch.fx.Node] = []

    # Step 1: Find all all_gather nodes
    for node in node_list:
        to_add = False
        if only_fsdp:
            to_add = is_wait_tensor_from_all_gather_into_tensor_fsdp(node)
        else:
            to_add = is_wait_tensor_from_all_gather_into_tensor(node)

        if to_add:
            ag_node = node.args[0]
            ag_nodes.append(ag_node)

    # Step 2: Put all_gather nodes into buckets
    ag_buckets: list[list[torch.fx.Node]] = []
    cur_bucket: list[torch.fx.Node] = []
    cur_bucket_size_bytes: int = 0
    cur_bucket_id: int = 0
    # Convert MiB to bytes
    all_gather_bucket_size_bytes = int(
        all_gather_bucket_cap_mb_callback(cur_bucket_id) * 1024 * 1024
    )
    for ag_node in ag_nodes:
        assert is_all_gather_into_tensor(ag_node)
        assert "val" in ag_node.meta
        ag_output_size_bytes = (
            ag_node.meta["val"].numel()
            * torch.finfo(ag_node.meta["val"].dtype).bits
            // 8
        )
        if (
            cur_bucket_size_bytes + ag_output_size_bytes > all_gather_bucket_size_bytes
            and cur_bucket
        ):
            # Current bucket is full, create new bucket
            ag_buckets.append(cur_bucket)
            cur_bucket = []
            cur_bucket_size_bytes = 0
            cur_bucket_id += 1
        cur_bucket_size_bytes += ag_output_size_bytes
        cur_bucket.append(ag_node)
    if cur_bucket:
        # add remaining nodes in the last bucket
        ag_buckets.append(cur_bucket)

    return ag_buckets


def all_gather_merge_fn_to_trace(  # type: ignore[no-untyped-def]
    ag_inputs, group_size: int, group_name: str, dtype: torch.dtype
) -> list[torch.Tensor]:
    inp_sizes = [ag_in.shape for ag_in in ag_inputs]
    inp_split_sizes = [ag_in.numel() for ag_in in ag_inputs]
    ag_input_numel = sum(inp_split_sizes)
    device = ag_inputs[0].device
    rank = device.index
    new_ag_out = torch.empty(ag_input_numel * group_size, dtype=dtype, device=device)
    foreach_copy_dsts = torch.split(new_ag_out, inp_split_sizes)
    ag_inputs_reshaped = [ag_in.reshape(-1) for ag_in in ag_inputs]
    # Using non-functional foreach_copy_ to match perf with fsdp implementation
    # In future It will be good to have fully functional implementation,
    # To keep invariant functional graph invariant.
    torch._foreach_copy_(foreach_copy_dsts, ag_inputs_reshaped)
    new_ag_in = new_ag_out.narrow(0, ag_input_numel * rank, ag_input_numel)
    wait_tensor = torch.ops.c10d_functional.wait_tensor(
        torch.ops._c10d_functional.all_gather_into_tensor_out.default(
            new_ag_in, group_size, group_name, out=new_ag_out
        )
    )
    new_ag_out_reshaped = wait_tensor.reshape(group_size, -1)
    outs = torch.split_with_sizes(
        new_ag_out_reshaped,
        inp_split_sizes,
        dim=1,
    )
    outs_reshaped = [out.reshape(shape) for out, shape in zip(outs, inp_sizes)]
    return outs_reshaped


def _trace(fn, inps) -> torch.fx.GraphModule:  # type: ignore[no-untyped-def]
    fake_mode = maybe_get_fake_mode(inps[0][0])
    assert fake_mode is not None
    with fake_mode:
        return make_fx(fn)(*inps)


def _insert_fn_trace_before_node(  # type: ignore[no-untyped-def]
    g: torch.fx.Graph,
    fn_to_trace,
    inps,
    insert_before_node,
    g_fn_inps,
    g_fn_outs,
    nodes_to_erase,
) -> None:  # type: ignore[no-untyped-def]
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
    for orig_out, new_out in zip(g_fn_outs, g_fn_new_outs):
        orig_out.replace_all_uses_with(new_out)
        g.erase_node(orig_out)
    for node in nodes_to_erase:
        g.erase_node(node)


def merge_all_gather(
    gm: torch.fx.GraphModule, ag_buckets: list[list[torch.fx.Node]]
) -> None:  # type: ignore[union-attr]
    """
    Merges specified buckets of all_gather to joint all_gather.
    """
    ag_node_to_wait_node: dict[torch.fx.Node, torch.fx.Node] = {}
    ag_node_to_bucket_idx = {}
    ag_nodes_set: OrderedSet[torch.fx.Node] = OrderedSet()

    for bucket_idx, bucket in enumerate(ag_buckets):
        for ag_node in bucket:
            assert len(ag_node.users) == 1, (
                f"Expect only one user for {ag_node}, but got {ag_node.users}"
            )
            ag_nodes_set.add(ag_node)
            wait_node = next(iter(ag_node.users))
            ag_node_to_wait_node[ag_node] = wait_node
            ag_node_to_bucket_idx[ag_node] = bucket_idx
    _, group_size, group_name = next(iter(ag_node_to_wait_node.keys())).args
    ag_node_to_pre_nodes = defaultdict(list)
    bucket_idx_to_bucketed_op_info = {}
    for bucket_idx, ag_bucket in enumerate(ag_buckets):
        ag_input_nodes = []
        wait_nodes = []
        dtype = None
        for ag_node in ag_bucket:
            assert (
                ag_node in ag_node_to_wait_node
                and ag_node.args[1] == group_size
                and ag_node.args[2] == group_name
            )
            ag_node_in = ag_node.args[0]
            ag_node_dtype = ag_node.meta["val"].dtype
            if dtype:
                assert ag_node_dtype == dtype
            else:
                dtype = ag_node_dtype
            if (
                ag_node_in.op == "call_function"  # type: ignore[union-attr]
                and ag_node_in.target == torch.ops.prims.convert_element_type.default  # type: ignore[union-attr]
            ):
                ag_node_to_pre_nodes[ag_node].append(ag_node_in)
                ag_node_in = ag_node_in.args[0]  # type: ignore[union-attr]

            ag_input_nodes.append(ag_node_in)  # type: ignore[union-attr]
            wait_nodes.append(ag_node_to_wait_node[ag_node])
        bucket_idx_to_bucketed_op_info[bucket_idx] = (
            ag_input_nodes,
            group_size,
            group_name,
            dtype,
            wait_nodes,
        )

    ag_nodes_found: list[int] = [0] * len(ag_buckets)
    g = gm.graph
    for n in g.nodes:
        if n in ag_nodes_set:
            bucket_idx = ag_node_to_bucket_idx[n]
            ag_nodes_found[bucket_idx] += 1
            if ag_nodes_found[bucket_idx] < len(ag_buckets[bucket_idx]):
                continue

            ag_nodes_ins, group_size, group_name, dtype, orig_wait_nodes = (
                bucket_idx_to_bucketed_op_info[bucket_idx]
            )
            nodes_to_erase = []
            for ag_n in ag_buckets[bucket_idx]:
                nodes_to_erase.append(ag_n)
                nodes_to_erase.extend(reversed(ag_node_to_pre_nodes[ag_n]))  # type: ignore[arg-type]

            _insert_fn_trace_before_node(
                g,
                all_gather_merge_fn_to_trace,
                (
                    pytree.tree_map(lambda node: node.meta["val"], ag_nodes_ins),
                    group_size,
                    group_name,
                    dtype,
                ),
                n.next,
                ag_nodes_ins,
                orig_wait_nodes,
                nodes_to_erase,
            )


def bucket_size_determinator(bucket_id: int) -> float:
    """
    Determine the size of a bucket based on its ID.

    Args:
    bucket_id (int): The ID of the bucket.

    Returns:
    float: The size of the bucket.
    """
    return 2000.0
