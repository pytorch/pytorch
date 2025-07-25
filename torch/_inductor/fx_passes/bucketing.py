import logging
import math
import operator
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch
from torch._dispatch.python import enable_python_dispatcher
from torch._inductor.virtualized import V
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
    group_name_rs_nodes: dict[tuple[str, str, torch.dtype], list[torch.fx.Node]] = (  # type: ignore[name-defined]
        defaultdict(list)
    )
    # Step 1: Find all reduce_scatter nodes
    for node in node_list:
        if is_wait_tensor(node) and is_reduce_scatter_tensor(node.args[0]):
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
    from torch.distributed.distributed_c10d import (
        _resolve_process_group,
        get_process_group_ranks,
    )

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
                all_gather_copy_in = new_graph_call_function(
                    new_graph,
                    torch.ops.fsdp.all_gather_copy_in.default,
                    (
                        param_all_gather_inputs_flattened,
                        all_gather_output,
                        inp_split_sizes,
                        all_gather_input_numel,
                        rank_idx_dict[rank],
                    ),
                    {},
                )
                all_gather_input = new_graph_call_function(
                    new_graph,
                    operator.getitem,
                    (all_gather_copy_in, 0),
                    {},
                )
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


def merge_reduce_scatter(
    gm: torch.fx.GraphModule, rs_buckets: list[list[torch.fx.Node]]
) -> None:
    """
    Transforms the graph to use bucketed reduce_scatter operations based on `rs_buckets`.
    """
    assert len(rs_buckets) > 0

    rs_nodes: list[torch.fx.Node] = []
    rs_node_to_wait_node: dict[torch.fx.Node, torch.fx.Node] = {}
    rs_node_to_bucket_id = {}

    # Map nodes to buckets and identify wait nodes
    for bucket_id, bucket in enumerate(rs_buckets):
        for rs_node in bucket:
            assert is_reduce_scatter_tensor(rs_node), (
                f"Expected reduce_scatter node, got {rs_node}"
            )
            # Find the wait_tensor node that uses this reduce_scatter node
            wait_nodes = list(rs_node.users)
            assert len(wait_nodes) == 1, (
                f"Expected exactly one user for {rs_node}, got {wait_nodes}"
            )
            wait_node = wait_nodes[0]
            assert is_wait_tensor(wait_node), (
                f"Expected wait_tensor node, got {wait_node}"
            )

            rs_node_to_wait_node[rs_node] = wait_node
            rs_nodes.append(rs_node)
            rs_node_to_bucket_id[rs_node] = bucket_id

    order = {x: i for i, x in enumerate(gm.graph.nodes)}
    rs_wait_nodes = list(rs_node_to_wait_node.values())
    rs_and_its_recursive_users = OrderedSet(rs_nodes + rs_wait_nodes)

    # Prepare bucketed operation info
    bucket_id_to_bucketed_op_info = {}
    bucket_id_is_scheduled = {}
    group_name_to_rank_idx_dict: dict[str, dict[int, int]] = {}
    for bucket_id, rs_bucket in enumerate(rs_buckets):
        _, reduce_op, group_size, group_name = next(
            iter(rs_node_to_wait_node.keys())
        ).args
        rs_input_nodes = []
        wait_nodes = []
        wait_node_recursive_users = OrderedSet()  # type: ignore[var-annotated]
        for rs_node in rs_bucket:
            assert (
                rs_node in rs_node_to_wait_node
                and rs_node.args[1] == reduce_op
                and rs_node.args[2] == group_size
                and rs_node.args[3] == group_name
            )
            rs_input_nodes.append(rs_node.args[0])
            wait_node = rs_node_to_wait_node[rs_node]
            wait_nodes.append(wait_node)
            find_recursive_users_of_fx_node(wait_node, wait_node_recursive_users)
            rs_and_its_recursive_users |= wait_node_recursive_users
        bucket_id_to_bucketed_op_info[bucket_id] = (
            rs_input_nodes,
            reduce_op,
            group_size,
            group_name,
            wait_nodes,
            wait_node_recursive_users,
        )
        if group_name not in group_name_to_rank_idx_dict:
            group_name_to_rank_idx_dict[group_name] = _rank_idx_dict(group_name)  # type: ignore[arg-type, index]

    new_graph: torch.fx.Graph = torch.fx.Graph()
    env: dict[torch.fx.Node, torch.fx.Node] = {}

    node_list = list(gm.graph.nodes)
    for node in node_list:
        if node not in rs_and_its_recursive_users:
            # not reduce_scatter or its (recursive) users - schedule it normally
            node_copy(env, new_graph, node, lambda x: env_lookup(env, x, node))
        elif node in rs_node_to_wait_node:
            assert node in rs_node_to_bucket_id
            bucket_id = rs_node_to_bucket_id[node]
            if not (
                bucket_id not in bucket_id_is_scheduled
                and rs_buckets[bucket_id][-1] == node
            ):
                continue

            # If we are at the last node in the bucket, we can start to schedule the bucketed reduce_scatter node
            (
                rs_input_nodes,
                reduce_op,
                group_size,
                group_name,
                orig_wait_nodes,
                orig_wait_node_recursive_users,
            ) = bucket_id_to_bucketed_op_info[bucket_id]
            rank_idx_dict = group_name_to_rank_idx_dict[group_name]  # type: ignore[index]
            # parents of rs have been scheduled, so we can directly use the env
            unsharded_grads = [env[x] for x in rs_input_nodes]  # type: ignore[index]
            reduce_dtype = unsharded_grads[0].meta["val"].dtype
            # Only float32 and bfloat16 are supported for now.
            # To support fp16, please see FSDP2 `_get_gradient_divide_factors`.
            assert reduce_dtype in (
                torch.float32,  # type: ignore[attr-defined]
                torch.bfloat16,  # type: ignore[attr-defined]
            ), f"reduce_dtype {reduce_dtype} is not supported"
            assert all(
                grad.meta["val"].dtype == reduce_dtype for grad in unsharded_grads
            )
            device = unsharded_grads[0].meta["val"].device
            rank = device.index
            rank_idx = rank_idx_dict[rank]
            shard_dim = 0

            def _get_dim0_padded_size(
                tensor_size: torch.Size,
                dim0_factor: int,  # type: ignore[name-defined]
            ) -> torch.Size:  # type: ignore[name-defined]
                padded_dim0 = math.ceil(tensor_size[0] / dim0_factor) * dim0_factor  # type: ignore[attr-defined]
                return torch.Size([padded_dim0]) + tensor_size[1:]

            padded_unsharded_sizes = tuple(
                _get_dim0_padded_size(grad.meta["val"].size(), group_size)  # type: ignore[arg-type]
                for grad in unsharded_grads
            )
            reduce_scatter_input_numel = sum(s.numel() for s in padded_unsharded_sizes)

            """
            NOTE: the relationship between the next few nodes is tricky:
            - reduce_scatter_input_reshaped is a view of reduce_scatter_input
            (same storage, same # elems, different shape).
            - chunk_cat writes into reduce_scatter_input_reshaped,
            which indirectly writes into reduce_scatter_input
            (since they share the same storage).
            - reduce_scatter_tensor reads from reduce_scatter_input.
            """
            reduce_scatter_input = new_graph_call_function(
                new_graph,
                torch.ops.aten.empty.memory_format,
                ([reduce_scatter_input_numel],),
                {
                    "dtype": reduce_dtype,
                    "device": device,
                    "pin_memory": False,
                },
            )
            reduce_scatter_input_reshaped = new_graph_call_function(
                new_graph,
                torch.ops.aten.reshape.default,
                (reduce_scatter_input, [group_size, -1]),
                {},
            )
            new_graph_call_function(
                new_graph,
                torch.ops.fsdp.chunk_cat.default,
                (unsharded_grads,),
                {
                    "dim": 0,
                    "num_chunks": group_size,
                    "out": reduce_scatter_input_reshaped,
                },
            )
            reduce_scatter_tensor = new_graph_call_function(
                new_graph,
                torch.ops._c10d_functional.reduce_scatter_tensor.default,
                (reduce_scatter_input, reduce_op, group_size, group_name),
                {},
            )

            wait_tensor = new_graph_call_function(
                new_graph,
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
                padded_unsharded_sizes, unsharded_grads
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
                    new_graph,
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
            assert len(orig_wait_nodes) == len(new_sharded_grads)
            assert len(orig_wait_nodes) > 0
            for new_sharded_grad, orig_wait_node in zip(
                new_sharded_grads, orig_wait_nodes
            ):
                env[orig_wait_node] = new_sharded_grad  # noqa: PERF403
            for user in sorted(orig_wait_node_recursive_users, key=lambda x: order[x]):
                # We skip output node here, because output node will be inserted (later)
                # as the last node in the new graph.
                if user.op != "output":
                    node_copy(env, new_graph, user, lambda x: env_lookup(env, x, user))
            bucket_id_is_scheduled[bucket_id] = True
        else:
            continue
    assert node_list[-1].op == "output"
    # Finally, insert the output node
    output_node = node_list[-1]
    node_copy(env, new_graph, output_node, lambda x: env_lookup(env, x, output_node))
    gm.graph = new_graph
