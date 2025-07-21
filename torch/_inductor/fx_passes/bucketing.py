import logging
import math
import operator
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch
import torch.utils._pytree as pytree
from torch._dispatch.python import enable_python_dispatcher
from torch._inductor.virtualized import V
from torch._subclasses.fake_tensor import maybe_get_fake_mode
from torch.distributed.distributed_c10d import (
    _resolve_process_group,
    get_process_group_ranks,
)
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
    merge_all_gather_trace(gm, ag_buckets)


def bucket_reduce_scatter(
    gm: torch.fx.GraphModule,
    reduce_scatter_bucket_cap_mb_callback: Callable[[int], float],
) -> None:
    rs_buckets = bucket_reduce_scatter_by_mb(gm, reduce_scatter_bucket_cap_mb_callback)
    if len(rs_buckets) == 0:
        return
    merge_reduce_scatter_trace(gm, rs_buckets)


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

    group_name_ag_nodes: dict[str, list[torch.fx.Node]] = defaultdict(list)

    # Step 1: Find all all_gather nodes
    for node in node_list:
        if is_wait_tensor(node) and is_all_gather_into_tensor(node.args[0]):
            if (filter_wait_node is None) or filter_wait_node(node):
                ag_node = node.args[0]
                _, group_size, group_name = ag_node.args
                assert isinstance(group_name, str)
                group_name_ag_nodes[group_name].append(ag_node)

    # Step 2: Put all_gather nodes into buckets
    ag_buckets: list[list[torch.fx.Node]] = []
    for group_name, ag_nodes in group_name_ag_nodes.items():
        cur_bucket: list[torch.fx.Node] = []
        cur_bucket_size_bytes: int = 0
        cur_bucket_id: int = 0
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
        if len(cur_bucket) > 1:
            # add remaining nodes in the last bucket
            ag_buckets.append(cur_bucket)

    return ag_buckets


def bucket_reduce_scatter_by_mb(
    gm: torch.fx.GraphModule,
    reduce_scatter_bucket_cap_mb_callback: Callable[[int], float],
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

    group_name_rs_nodes: dict[tuple[str, str], list[torch.fx.Node]] = defaultdict(list)

    # Step 1: Find all reduce_scatter nodes
    for node in node_list:
        if is_wait_tensor(node) and is_reduce_scatter_tensor(node.args[0]):
            rs_node = node.args[0]
            _, reduce_op, group_size, group_name = rs_node.args
            assert isinstance(group_name, str)
            assert isinstance(reduce_op, str)
            group_name_rs_nodes[(group_name, reduce_op)].append(rs_node)

    # Step 2: Put reduce_scatter nodes into buckets
    rs_buckets: list[list[torch.fx.Node]] = []
    for (group_name, reduce_op), rs_nodes in group_name_rs_nodes.items():
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
            rs_input_size_bytes = (
                rs_input.meta["val"].numel()  # type: ignore[union-attr]
                * torch.finfo(rs_input.meta["val"].dtype).bits  # type: ignore[union-attr]
                // 8
            )
            if (
                cur_bucket_size_bytes + rs_input_size_bytes
                > reduce_scatter_bucket_size_bytes
                and cur_bucket
            ):
                # Current bucket is full, create new bucket
                total_size = cur_bucket_size_bytes + rs_input_size_bytes
                logger.info(
                    f"Reduce scatter bucket {cur_bucket_id} full: "  # noqa: G004
                    f"total_size = {total_size} = cur_bucket_size_bytes + rs_input_size_bytes = "
                    f"{cur_bucket_size_bytes} + {rs_input_size_bytes},"
                    f"bucket_cap = {reduce_scatter_bucket_size_bytes}"
                )
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
            # add remaining nodes in the last bucket
            logger.info(
                f"Reduce scatter last bucket {cur_bucket_id}: "  # noqa: G004
                f"total_size = {cur_bucket_size_bytes}, "
                f"bucket_cap = {reduce_scatter_bucket_size_bytes}"
            )
            if len(cur_bucket) > 1:
                rs_buckets.append(cur_bucket)

    return rs_buckets


def node_copy(  # type: ignore[no-untyped-def]
    env,
    new_graph,
    node: torch.fx.Node,
    arg_transform: Callable[[torch.fx.Node], torch.fx.node.Argument],
) -> torch.fx.Node:
    if node not in env:
        new_node = new_graph.node_copy(node, arg_transform=arg_transform)
        env[node] = new_node
    else:
        new_node = env[node]
    return new_node


def new_graph_call_function(  # type: ignore[no-untyped-def]
    new_graph,
    target: Callable[..., Any],
    args: Optional[tuple[torch.fx.node.Argument, ...]] = None,
    kwargs: Optional[dict[str, torch.fx.node.Argument]] = None,
    type_expr: Optional[Any] = None,
) -> torch.fx.Node:
    from torch.utils._pytree import tree_map_only

    new_node = new_graph.call_function(target, args, kwargs)
    args_val = tree_map_only(torch.fx.Node, lambda x: x.meta["val"], args)
    kwargs_val = tree_map_only(torch.fx.Node, lambda x: x.meta["val"], kwargs)
    with V.fake_mode, enable_python_dispatcher():
        new_fake_tensor = target(*args_val, **kwargs_val)
    new_node.meta["val"] = new_fake_tensor
    return new_node


def env_lookup(  # type: ignore[no-untyped-def]
    env, x: torch.fx.Node, node_user: Union[torch.fx.Node, str]
) -> torch.fx.Node:
    assert x in env, (
        f"Dependent node {x} not in env when creating downstream node {node_user}"
    )
    return env[x]


def _rank_idx_dict(group_name: str) -> dict[int, int]:
    pg = _resolve_process_group(group_name)
    ranks = get_process_group_ranks(pg)
    rank_idx_dict: dict[int, int] = {rank: idx for idx, rank in enumerate(ranks)}
    return rank_idx_dict


def merge_all_gather(
    gm: torch.fx.GraphModule, ag_buckets: list[list[torch.fx.Node]]
) -> None:
    """
    Transforms the graph to use bucketed all_gather operations based on `ag_buckets`.
    """
    assert len(ag_buckets) > 0

    ag_nodes: list[torch.fx.Node] = []
    cast_nodes: list[torch.fx.Node] = []
    ag_node_to_wait_node: dict[torch.fx.Node, torch.fx.Node] = {}
    ag_node_to_bucket_id = {}
    cast_node_to_bucket_id = {}

    # Map nodes to buckets and identify wait nodes
    for bucket_id, bucket in enumerate(ag_buckets):
        for ag_node in bucket:
            assert len(ag_node.users) == 1, (
                f"Expect only one user for {ag_node}, but got {ag_node.users}"
            )
            wait_node = next(iter(ag_node.users))
            ag_node_to_wait_node[ag_node] = wait_node
            ag_nodes.append(ag_node)
            ag_node_to_bucket_id[ag_node] = bucket_id
            if (
                ag_node.args[0].op == "call_function"  # type: ignore[union-attr]
                and ag_node.args[0].target  # type: ignore[union-attr]
                == torch.ops.prims.convert_element_type.default
            ):
                cast_nodes.append(ag_node.args[0])  # type: ignore[arg-type]
                cast_node_to_bucket_id[ag_node.args[0]] = bucket_id  # type: ignore[arg-type]

    # Step 3: Create new (bucketed) all_gather nodes
    bucket_id_to_bucketed_op_info = {}
    bucket_id_is_scheduled = {}
    cast_bucket_id_is_scheduled = {}
    _, group_size, group_name = next(iter(ag_node_to_wait_node.keys())).args

    group_name_to_rank_idx_dict: dict[str, dict[int, int]] = {}

    for bucket_id, ag_bucket in enumerate(ag_buckets):
        ag_input_nodes = []
        wait_nodes = []
        for ag_node in ag_bucket:
            ag_input_nodes.append(ag_node.args[0])
            wait_nodes.append(ag_node_to_wait_node[ag_node])
        bucket_id_to_bucketed_op_info[bucket_id] = (
            ag_input_nodes,
            group_size,
            group_name,
            wait_nodes,
        )
        if group_name not in group_name_to_rank_idx_dict:
            group_name_to_rank_idx_dict[group_name] = _rank_idx_dict(group_name)  # type: ignore[arg-type, index]

    ag_wait_nodes = list(ag_node_to_wait_node.values())
    ag_and_wait_nodes = OrderedSet(ag_nodes + ag_wait_nodes)
    cast_nodes = OrderedSet(cast_nodes)
    new_graph: torch.fx.Graph = torch.fx.Graph()
    env: dict[torch.fx.Node, torch.fx.Node] = {}

    node_list = gm.graph.nodes
    for node in node_list:
        if node not in ag_and_wait_nodes and node not in cast_nodes:
            # not cast-before-all_gather, all_gather or its wait_tensor - schedule it normally
            node_copy(env, new_graph, node, lambda x: env_lookup(env, x, node))
        elif node in cast_nodes:
            # batch cast nodes together into one foreach_copy node
            assert node in cast_node_to_bucket_id
            bucket_id = cast_node_to_bucket_id[node]
            if bucket_id not in cast_bucket_id_is_scheduled:
                ag_input_nodes, group_size, group_name, orig_wait_nodes = (
                    bucket_id_to_bucketed_op_info[bucket_id]
                )
                if all(
                    n.op == "call_function"  # type: ignore[union-attr]
                    and n.target == torch.ops.prims.convert_element_type.default  # type: ignore[union-attr]
                    for n in ag_input_nodes
                ):
                    param_all_gather_inputs = [
                        new_graph_call_function(
                            new_graph,
                            torch.ops.aten.empty.memory_format,
                            (n.meta["val"].shape,),  # type: ignore[union-attr]
                            {
                                "dtype": n.args[1],  # type: ignore[union-attr]
                                "device": n.meta["val"].device,  # type: ignore[union-attr]
                                "pin_memory": False,
                            },
                        )
                        for n in ag_input_nodes
                    ]
                    for pp, n in zip(param_all_gather_inputs, ag_input_nodes):
                        pp.meta = n.meta.copy()  # type: ignore[union-attr]

                    cast_input_nodes = [env[n.args[0]] for n in ag_input_nodes]  # type: ignore[union-attr, index]
                    foreach_copy = new_graph_call_function(
                        new_graph,
                        torch.ops.aten._foreach_copy.default,
                        (param_all_gather_inputs, cast_input_nodes),
                        {},
                    )
                    foreach_copy.meta["val"] = [n.meta["val"] for n in ag_input_nodes]  # type: ignore[union-attr]
                    getitems = [
                        new_graph_call_function(
                            new_graph,
                            operator.getitem,
                            (foreach_copy, i),
                            {},
                        )
                        for i in range(len(ag_input_nodes))
                    ]

                    for new_n, old_n in zip(getitems, ag_input_nodes):
                        env[old_n] = new_n  # type: ignore[index] # noqa: PERF403
                else:
                    param_all_gather_inputs_orig = [
                        node_copy(
                            env,
                            new_graph,
                            ag_input_node,  # type: ignore[arg-type]
                            lambda x: env_lookup(env, x, ag_input_node),  # type: ignore[arg-type]
                        )
                        for ag_input_node in ag_input_nodes
                    ]
                cast_bucket_id_is_scheduled[bucket_id] = True
            else:
                continue
        elif node in ag_node_to_wait_node:
            assert node in ag_node_to_bucket_id
            bucket_id = ag_node_to_bucket_id[node]
            if bucket_id not in bucket_id_is_scheduled:
                ag_input_nodes, group_size, group_name, orig_wait_nodes = (
                    bucket_id_to_bucketed_op_info[bucket_id]
                )
                rank_idx_dict = group_name_to_rank_idx_dict[group_name]  # type: ignore[index]
                device = ag_input_nodes[0].meta["val"].device  # type: ignore[union-attr]
                rank = device.index
                dtype = ag_input_nodes[0].meta["val"].dtype  # type: ignore[union-attr]
                # TODO: if we want to support mixed dtype in the same bucket,
                # we need to first view all all_gather inputs as uint8 (common denominator),
                # then do the all_gather, then view the output back to the original dtype.
                # Look at FSDP2 to see how to do this.
                assert all(n.meta["val"].dtype == dtype for n in ag_input_nodes), (  # type: ignore[union-attr]
                    "All all_gather inputs in the same bucket must have the same dtype"
                )
                # must schedule all the all_gather input nodes first, before the bucketed all_gather node
                param_all_gather_inputs_orig = [
                    node_copy(
                        env,
                        new_graph,
                        ag_input_node,  # type: ignore[arg-type]
                        lambda x: env_lookup(env, x, ag_input_node),  # type: ignore[arg-type]
                    )
                    for ag_input_node in ag_input_nodes
                ]
                # schedule the bucketed all_gather node
                param_all_gather_inputs_flattened = [
                    new_graph_call_function(
                        new_graph, torch.ops.aten.reshape.default, (n, [-1]), {}
                    )
                    for n in param_all_gather_inputs_orig
                ]
                inp_split_sizes = [
                    n.meta["val"].numel() for n in param_all_gather_inputs_orig
                ]
                param_all_gather_outputs = [
                    new_graph_call_function(
                        new_graph,
                        torch.ops.aten.empty.memory_format,
                        ([n.meta["val"].numel() * group_size],),
                        {
                            "dtype": n.meta["val"].dtype,
                            "device": n.meta["val"].device,
                            "pin_memory": False,
                        },
                    )
                    for n in param_all_gather_inputs_orig
                ]
                # TODO: This assumes dim-0 sharding.
                # If we need to support sharding on another dim, we should look at how FSDP2 does it
                # (e.g. search for `shard_dim` in FSDP2 codebase)
                param_all_gather_outputs_shape_orig = [
                    (n.meta["val"].shape[0] * group_size,) + n.meta["val"].shape[1:]
                    for n in param_all_gather_inputs_orig
                ]
                all_gather_input_numel = sum(inp_split_sizes)

                all_gather_output = new_graph_call_function(
                    new_graph,
                    torch.ops.aten.empty.memory_format,
                    ([all_gather_input_numel * group_size],),
                    {
                        "dtype": dtype,
                        "device": device,
                        "pin_memory": False,
                    },
                )
                # BEGIN
                local_rank = rank_idx_dict[rank]
                all_gather_input = new_graph_call_function(
                    new_graph,
                    torch.ops.aten.slice.Tensor,
                    (
                        all_gather_output,
                        0,
                        all_gather_input_numel * local_rank,
                        all_gather_input_numel * (local_rank + 1),
                    ),
                    {},
                )
                split_with_sizes = new_graph_call_function(
                    new_graph,
                    torch.ops.aten.split_with_sizes.default,
                    (
                        all_gather_input,
                        inp_split_sizes,
                    ),
                    {},
                )
                splits = [
                    new_graph_call_function(
                        new_graph,
                        operator.getitem,
                        (
                            split_with_sizes,
                            i,
                        ),
                        {},
                    )
                    for i in range(len(inp_split_sizes))
                ]
                new_graph_call_function(
                    new_graph,
                    torch.ops.aten._foreach_copy_.default,
                    (
                        splits,
                        param_all_gather_inputs_flattened,
                    ),
                    {},
                )
                # END
                # all_gather_copy_in = new_graph_call_function(
                #     new_graph,
                #     torch.ops.fsdp.all_gather_copy_in.default,
                #     (
                #         param_all_gather_inputs_flattened,
                #         all_gather_output,
                #         inp_split_sizes,
                #         all_gather_input_numel,
                #         rank_idx_dict[rank],
                #     ),
                #     {},
                # )
                # all_gather_input = new_graph_call_function(
                #     new_graph,
                #     operator.getitem,
                #     (all_gather_copy_in, 0),
                #     {},
                # )
                all_gather_into_tensor_out = new_graph_call_function(
                    new_graph,
                    torch.ops._c10d_functional.all_gather_into_tensor_out.default,
                    (all_gather_input, group_size, group_name),
                    {"out": all_gather_output},
                )
                wait_tensor = new_graph_call_function(
                    new_graph,
                    torch.ops._c10d_functional.wait_tensor.default,
                    (all_gather_into_tensor_out,),
                    {},
                )
                all_gather_output_reshaped = new_graph_call_function(
                    new_graph,
                    torch.ops.aten.reshape.default,
                    (wait_tensor, [group_size, -1]),
                    {},
                )
                outs_flattened = [
                    new_graph_call_function(
                        new_graph,
                        torch.ops.aten.reshape.default,
                        (n, [group_size, -1]),
                        {},
                    )
                    for n in param_all_gather_outputs
                ]
                split_with_sizes_copy = new_graph_call_function(  # noqa: F841
                    new_graph,
                    torch.ops.fsdp.split_with_sizes_copy.default,
                    (all_gather_output_reshaped, inp_split_sizes),
                    {"dim": 1, "out": outs_flattened},
                )
                outs = [
                    new_graph_call_function(
                        new_graph,
                        torch.ops.aten.reshape.default,
                        (n, orig_shape),
                        {},
                    )
                    for n, orig_shape in zip(
                        outs_flattened, param_all_gather_outputs_shape_orig
                    )
                ]
                assert len(orig_wait_nodes) == len(outs)
                assert len(orig_wait_nodes) > 0
                for out, orig_wait_node in zip(outs, orig_wait_nodes):
                    env[orig_wait_node] = out  # noqa: PERF403
                bucket_id_is_scheduled[bucket_id] = True
        else:
            continue
    gm.graph = new_graph


def find_recursive_users_of_fx_node(node, collected_node_set, criteria_cb=None) -> None:  # type: ignore[no-untyped-def]
    if criteria_cb and criteria_cb(node):
        return
    for user_node in node.users:
        if user_node in collected_node_set:
            continue
        collected_node_set.add(user_node)
        find_recursive_users_of_fx_node(
            user_node,
            collected_node_set,
            criteria_cb=criteria_cb,
        )


def reduce_scatter_merge_fn_to_trace(
    rs_ins, group_size, group_name, reduce_op, reduce_dtype, device
):
    rs_ins_sizes = [rs_in.size() for rs_in in rs_ins]
    new_rs_in_numel = sum(s.numel() for s in rs_ins_sizes)
    new_rs_in = torch.empty(new_rs_in_numel, dtype=reduce_dtype, device=device)
    rs_ins_flattened = [rs_in.view(-1) for rs_in in rs_ins]
    split_sizes_per_rank = [rs_in.numel() // group_size for rs_in in rs_ins_flattened]

    rs_ins_srcs = [
        rs_in_f.split([rs_in_f.numel() // group_size] * group_size)
        for rs_in_f in rs_ins_flattened
    ]

    foreach_copy_dst_splits = split_sizes_per_rank * group_size
    foreach_copy_dsts = torch.split(new_rs_in, foreach_copy_dst_splits)

    foreach_copy_srcs = []
    for rank_idx in range(group_size):
        for rs_in_idx in range(len(rs_ins)):
            foreach_copy_srcs.append(rs_ins_srcs[rs_in_idx][rank_idx])

    torch._foreach_copy_(foreach_copy_dsts, foreach_copy_srcs)

    wait_tensor = torch.ops.c10d_functional.wait_tensor(
        torch.ops._c10d_functional.reduce_scatter_tensor.default(
            new_rs_in, reduce_op, group_size, group_name
        )
    )
    new_rs_out = wait_tensor

    new_outs = []
    new_rs_out_offset = 0
    for rs_in in rs_ins:
        new_out_size = torch.Size((rs_in.shape[0] // group_size,) + rs_in.shape[1:])
        new_out = torch.as_strided(
            new_rs_out,
            size=new_out_size,
            stride=torch._prims_common.make_contiguous_strides_for(new_out_size),
            storage_offset=new_rs_out_offset,
        )
        new_outs.append(new_out)
        new_rs_out_offset += new_out_size.numel()
    return new_outs


def all_gather_merge_fn_to_trace(  # type: ignore[no-untyped-def]
    ag_ins, group_size: int, group_name: str, dtype: torch.dtype, local_rank
) -> list[torch.Tensor]:
    inp_sizes = [ag_in.shape for ag_in in ag_ins]
    inp_split_sizes = [ag_in.numel() for ag_in in ag_ins]
    ag_input_numel = sum(inp_split_sizes)
    device = ag_ins[0].device
    new_ag_out = torch.empty(ag_input_numel * group_size, dtype=dtype, device=device)
    foreach_copy_dsts = torch.split(new_ag_out, inp_split_sizes)
    ag_ins_reshaped = [ag_in.reshape(-1) for ag_in in ag_ins]
    torch._foreach_copy_(foreach_copy_dsts, ag_ins_reshaped)
    new_ag_in = new_ag_out.narrow(0, ag_input_numel * local_rank, ag_input_numel)
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
    with fake_mode, enable_python_dispatcher():
        return make_fx(fn)(*inps)


def _insert_fn_trace_before_node(  # type: ignore[no-untyped-def]
    g: torch.fx.Graph,
    fn_to_trace,
    inps,
    insert_before_node,
    g_fn_inps,
    g_fn_outs,
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


def merge_reduce_scatter_trace(
    gm: torch.fx.GraphModule, rs_buckets: list[list[torch.fx.Node]]
) -> None:
    for i, rs_bucket in enumerate(rs_buckets):
        print(f"XXX RS_BUCKET[{i}]:{rs_bucket}")
    n_buckets = len(rs_buckets)
    buckets_lens = [len(rs_bucket) for rs_bucket in rs_buckets]
    g = gm.graph
    rs_ins = [[] for _ in range(n_buckets)]
    rs_waits = [[] for _ in range(n_buckets)]
    rs_ns = [[] for _ in range(n_buckets)]
    rs_to_bucket_idx = {}
    for bucket_idx, rs_nodes in enumerate(rs_buckets):
        for rs_node in rs_nodes:
            rs_to_bucket_idx[rs_node] = bucket_idx

    for n in g.nodes:
        bucket_idx = rs_to_bucket_idx.get(n, -1)
        if bucket_idx == -1:
            continue

        assert len(n.users) == 1
        wait_n = next(iter(n.users))
        rs_ins[bucket_idx].append(n.args[0])
        rs_ns[bucket_idx].append(n)
        rs_waits[bucket_idx].append(wait_n)
        print(f"XXX {n} -> {bucket_idx}")
        print(f"XXX len(rs_ns[{bucket_idx}])=={len(rs_ns[bucket_idx])}")
        print(f"XXX buckets_lens[{bucket_idx}]=={buckets_lens[bucket_idx]}")
        if len(rs_ns[bucket_idx]) < buckets_lens[bucket_idx]:
            continue

        _, reduce_op, group_size, group_name = n.args
        reduce_dtype = n.meta["val"].dtype
        device = n.meta["val"].device

        _rs_ins = rs_ins[bucket_idx]
        _rs_waits = rs_waits[bucket_idx]
        _rs_ns = rs_ns[bucket_idx]
        print(f"XXX PROCESS {bucket_idx}")
        print(f"XXX _rs_ns:{_rs_ns}")
        print(f"XXX _rs_waits:{_rs_waits}")

        _insert_fn_trace_before_node(
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
            n.next,
            _rs_ins,
            _rs_waits,
        )
        for rs_n, wait_n in zip(_rs_ns, _rs_waits):
            g.erase_node(wait_n)
            g.erase_node(rs_n)


def merge_reduce_scatter(
    gm: torch.fx.GraphModule, rs_buckets: list[list[torch.fx.Node]]
) -> None:
    """
    Transforms the graph to use bucketed reduce_scatter operations based on `rs_buckets`.
    """
    assert len(rs_buckets) > 0
    g = gm.graph
    group_name_to_rank_idx_dict: dict[str, dict[int, int]] = {}

    for bucket_id, rs_bucket in enumerate(rs_buckets):
        new_g: torch.fx.Graph = torch.fx.Graph()
        node_list = list(g.nodes)
        rs_nodes: list[torch.fx.Node] = []
        rs_node_to_wait_node: dict[torch.fx.Node, torch.fx.Node] = {}
        rs_input_nodes = []
        wait_nodes = []
        wait_node_recursive_users = OrderedSet()  # type: ignore[var-annotated]

        _, reduce_op, group_size, group_name = rs_bucket[0].args
        for rs_node in rs_bucket:
            users = list(rs_node.users)
            assert len(users) == 1, (
                f"Expected exactly one user for {rs_node}, got {users}"
            )
            wait_node = users[0]
            assert is_wait_tensor(wait_node), (
                f"Expected wait_tensor node, got {wait_node}"
            )

            rs_node_to_wait_node[rs_node] = wait_node
            rs_nodes.append(rs_node)
            rs_input_nodes.append(rs_node.args[0])
            wait_node = rs_node_to_wait_node[rs_node]
            wait_nodes.append(wait_node)

        rs_and_its_recursive_users = OrderedSet(rs_nodes + wait_nodes)
        for wait_node in wait_nodes:
            find_recursive_users_of_fx_node(wait_node, wait_node_recursive_users)
            rs_and_its_recursive_users |= wait_node_recursive_users

        order = {x: i for i, x in enumerate(g.nodes)}

        if group_name not in group_name_to_rank_idx_dict:
            group_name_to_rank_idx_dict[group_name] = _rank_idx_dict(group_name)  # type: ignore[arg-type, index]

        env: dict[torch.fx.Node, torch.fx.Node] = {}

        for node in g.nodes:
            if node not in rs_and_its_recursive_users:
                node_copy(env, new_g, node, lambda x: env_lookup(env, x, node))
            elif node in rs_node_to_wait_node:
                # Looking for the last node in the bucket
                if rs_bucket[-1] != node:
                    continue

                rank_idx_dict = group_name_to_rank_idx_dict[group_name]  # type: ignore[index]
                # parents of rs have been scheduled, so we can directly use the env
                new_inputs = [env[x] for x in rs_input_nodes]  # type: ignore[index]
                reduce_dtype = new_inputs[0].meta["val"].dtype
                # Only float32 and bfloat16 are supported for now.
                # To support fp16, please see FSDP2 `_get_gradient_divide_factors`.
                assert reduce_dtype in (
                    torch.float32,
                    torch.bfloat16,
                ), f"reduce_dtype {reduce_dtype} is not supported"
                assert all(
                    grad.meta["val"].dtype == reduce_dtype for grad in new_inputs
                )
                device = new_inputs[0].meta["val"].device
                rank = device.index
                rank_idx = rank_idx_dict[rank]
                shard_dim = 0

                def _get_dim0_padded_size(
                    tensor_size: torch.Size, dim0_factor: int
                ) -> torch.Size:
                    padded_dim0 = math.ceil(tensor_size[0] / dim0_factor) * dim0_factor
                    return torch.Size([padded_dim0]) + tensor_size[1:]

                padded_unsharded_sizes = tuple(
                    _get_dim0_padded_size(grad.meta["val"].size(), group_size)  # type: ignore[arg-type]
                    for grad in new_inputs
                )
                reduce_scatter_input_numel = sum(
                    s.numel() for s in padded_unsharded_sizes
                )
                reduce_scatter_input = new_graph_call_function(
                    new_g,
                    torch.ops.aten.empty.memory_format,
                    ([reduce_scatter_input_numel],),
                    {
                        "dtype": reduce_dtype,
                        "device": device,
                        "pin_memory": False,
                    },
                )
                new_inputs_flattened = [
                    new_graph_call_function(
                        new_g, torch.ops.aten.reshape.default, (n, [-1]), {}
                    )
                    for n in new_inputs
                ]
                inp_split_sizes = [n.meta["val"].numel() for n in new_inputs_flattened]
                split_with_sizes = new_graph_call_function(
                    new_g,
                    torch.ops.aten.split_with_sizes.default,
                    (
                        reduce_scatter_input,
                        inp_split_sizes,
                    ),
                    {},
                )
                splits = [
                    new_graph_call_function(
                        new_g,
                        operator.getitem,
                        (
                            split_with_sizes,
                            i,
                        ),
                        {},
                    )
                    for i in range(len(inp_split_sizes))
                ]
                new_graph_call_function(
                    new_g,
                    torch.ops.aten._foreach_copy_.default,
                    (
                        splits,
                        new_inputs_flattened,
                    ),
                    {},
                )
                reduce_scatter_tensor = new_graph_call_function(
                    new_g,
                    torch.ops._c10d_functional.reduce_scatter_tensor.default,
                    (reduce_scatter_input, reduce_op, group_size, group_name),
                    {},
                )

                wait_tensor = new_graph_call_function(
                    new_g,
                    torch.ops._c10d_functional.wait_tensor.default,
                    (reduce_scatter_tensor,),
                    {},
                )

                def _chunk_with_empty(
                    tensor: torch.Tensor, num_chunks: int, dim: int
                ) -> list[torch.Tensor]:
                    chunks = list(torch.chunk(tensor, num_chunks, dim=dim))
                    while len(chunks) < num_chunks:
                        chunks.append(chunks[0].new_empty(0))
                    return chunks

                reduce_output = wait_tensor
                # View out and accumulate sharded gradients
                new_sharded_grads = []
                flat_grad_offset = 0  # [0, reduce_scatter_output_numel - 1]
                for padded_unsharded_size, unsharded_grad in zip(
                    padded_unsharded_sizes, new_inputs
                ):
                    # NOTE: we only care about the shape of tensors in `chunks`, so using meta tensor here
                    chunks = _chunk_with_empty(
                        torch.empty_like(unsharded_grad.meta["val"], device="meta"),
                        group_size,  # type: ignore[arg-type]
                        dim=shard_dim,
                    )
                    sharded_param = chunks[rank_idx]
                    sharded_size = sharded_param.size()
                    contiguous_sharded_stride = (
                        torch._prims_common.make_contiguous_strides_for(sharded_size)
                    )
                    # Assume even sharding for Shard(i), i > 0; otherwise would require
                    # copy-out for contiguous strides
                    new_sharded_grad = new_graph_call_function(
                        new_g,
                        torch.ops.aten.as_strided.default,
                        (reduce_output,),
                        {
                            "size": sharded_size,
                            "stride": contiguous_sharded_stride,
                            "storage_offset": flat_grad_offset,
                        },
                    )
                    new_sharded_grads.append(new_sharded_grad)
                    padded_sharded_numel = padded_unsharded_size.numel() // group_size  # type: ignore[operator]
                    flat_grad_offset += padded_sharded_numel  # type: ignore[assignment]
                assert len(wait_nodes) == len(new_sharded_grads)
                assert len(wait_nodes) > 0
                for new_sharded_grad, orig_wait_node in zip(
                    new_sharded_grads, wait_nodes
                ):
                    env[orig_wait_node] = new_sharded_grad  # noqa: PERF403
                for user in sorted(wait_node_recursive_users, key=lambda x: order[x]):
                    if user.op != "output":
                        node_copy(env, new_g, user, lambda x: env_lookup(env, x, user))
            else:
                continue
        output_node = node_list[-1]
        node_copy(env, new_g, output_node, lambda x: env_lookup(env, x, output_node))
        g = new_g
        for j in range(bucket_id + 1, len(rs_buckets)):
            rs_buckets[j] = [env[x] for x in rs_buckets[j]]
    gm.graph = g


def merge_all_gather_trace(
    gm: torch.fx.GraphModule, ag_buckets: list[list[torch.fx.Node]]
) -> None:  # type: ignore[union-attr]
    """
    Merges specified buckets of all_gather to joint all_gather.
    """
    buckets_lens = [len(ag_bucket) for ag_bucket in ag_buckets]
    ag_node_to_wait_node: dict[torch.fx.Node, torch.fx.Node] = {}
    ag_node_to_bucket_idx = {}

    ag_node_to_pre_nodes = defaultdict(list)
    bucket_idx_to_bucketed_op_info = {}

    group_name_to_rank_idx_dict: dict[str, dict[int, int]] = {}
    for bucket_idx, ag_bucket in enumerate(ag_buckets):
        ag_input_nodes = []
        wait_nodes = []
        _, group_size, group_name = ag_bucket[0].args
        dtype = ag_bucket[0].meta["val"].dtype
        if group_name not in group_name_to_rank_idx_dict:
            group_name_to_rank_idx_dict[group_name] = _rank_idx_dict(group_name)

        for ag_node in ag_bucket:
            assert len(ag_node.users) == 1, (
                f"Expect only one user for {ag_node}, but got {ag_node.users}"
            )
            wait_node = next(iter(ag_node.users))

            ag_node_to_wait_node[ag_node] = wait_node
            ag_node_to_bucket_idx[ag_node] = bucket_idx

            assert (
                ag_node.args[1] == group_size
                and ag_node.args[2] == group_name
                and ag_node.meta["val"].dtype == dtype
            )
            ag_node_in = ag_node.args[0]
            if (
                ag_node_in.op == "call_function"  # type: ignore[union-attr]
                and ag_node_in.target == torch.ops.prims.convert_element_type.default  # type: ignore[union-attr]
            ):
                ag_node_to_pre_nodes[ag_node].append(ag_node_in)
                ag_node_in = ag_node_in.args[0]  # type: ignore[union-attr]

            ag_input_nodes.append(ag_node_in)  # type: ignore[union-attr]
            wait_nodes.append(wait_node)
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
        bucket_idx = ag_node_to_bucket_idx.get(n, -1)
        if bucket_idx == -1:
            continue

        ag_nodes_found[bucket_idx] += 1
        if ag_nodes_found[bucket_idx] < buckets_lens[bucket_idx]:
            continue

        ag_nodes_ins, group_size, group_name, dtype, orig_wait_nodes = (
            bucket_idx_to_bucketed_op_info[bucket_idx]
        )
        rank_idx_dict = group_name_to_rank_idx_dict[group_name]
        rank = n.meta["val"].device.index
        local_rank = rank_idx_dict[rank]

        _insert_fn_trace_before_node(
            g,
            all_gather_merge_fn_to_trace,
            (
                pytree.tree_map(lambda node: node.meta["val"], ag_nodes_ins),
                group_size,
                group_name,
                dtype,
                local_rank,
            ),
            n.next,
            ag_nodes_ins,
            orig_wait_nodes,
        )
        # Erasing old nodes in reverse order
        for ag_n, wait_n in zip(ag_buckets[bucket_idx], orig_wait_nodes):
            g.erase_node(wait_n)
            g.erase_node(ag_n)
            for n in reversed(ag_node_to_pre_nodes[ag_n]):
                g.erase_node(n)
