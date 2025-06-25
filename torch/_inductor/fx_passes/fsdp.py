import logging
import operator
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch._dispatch.python import enable_python_dispatcher
from torch._inductor.virtualized import V
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
        all_gather_bucket_cap_mb_callback (Callable[[int], float]): callback function that takes in bucket id and returns size of a bucket in megabytes.

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

    ag_buckets = bucket_fsdp_all_gather_by_mb(gm, all_gather_bucket_cap_mb_callback)
    if len(ag_buckets) == 0:
        return
    merge_fsdp_all_gather_concat(gm, ag_buckets)


def is_all_gather_into_tensor(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.all_gather_into_tensor.default
    )


def is_wait_tensor(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.wait_tensor.default
    )


def env_lookup(
    env, x: torch.fx.Node, node_user: Union[torch.fx.Node, str]
) -> torch.fx.Node:
    assert x in env, (
        f"Dependent node {x} not in env when creating downstream node {node_user}"
    )
    return env[x]


def node_copy(
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


def new_graph_call_function(
    new_graph,
    target: Callable[..., Any],
    args: Optional[Tuple[torch.fx.node.Argument, ...]] = None,
    kwargs: Optional[Dict[str, torch.fx.node.Argument]] = None,
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


def is_graph_input(node: torch.fx.Node) -> bool:
    return node.op == "placeholder"


def is_wait_tensor_from_fsdp(node: torch.fx.Node) -> bool:
    if is_wait_tensor(node) and is_all_gather_into_tensor(node.args[0]):
        ag_node = node.args[0]
        assert is_graph_input(ag_node.args[0]) or (
            ag_node.args[0].op == "call_function"
            and ag_node.args[0].target == torch.ops.prims.convert_element_type.default
            and is_graph_input(ag_node.args[0].args[0])
        ), (
            "Assume all_gather_into_tensor input is either graph input "
            + f"or dtype conversion of graph input, but got {ag_node.args[0]}"
        )
        return True
    return False


def bucket_fsdp_all_gather_by_mb(
    gm: torch.fx.GraphModule, all_gather_bucket_cap_mb_callback: Callable[[int], float]
) -> List[List[torch.fx.Node]]:
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

    ag_nodes: List[torch.fx.Node] = []

    # Step 1: Find all all_gather nodes
    for node in node_list:
        if is_wait_tensor_from_fsdp(node):
            ag_node = node.args[0]
            ag_nodes.append(ag_node)

    # Step 2: Put all_gather nodes into buckets
    ag_buckets: List[List[torch.fx.Node]] = []
    cur_bucket: List[torch.fx.Node] = []
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


def merge_fsdp_all_gather_concat(
    gm: torch.fx.GraphModule, ag_buckets: List[List[torch.fx.Node]]
) -> None:
    """
    Transforms the graph to use bucketed all_gather operations based on `ag_buckets`.
    """
    assert len(ag_buckets) > 0

    ag_nodes: List[torch.fx.Node] = []
    cast_nodes: List[torch.fx.Node] = []
    ag_node_to_wait_node: Dict[torch.fx.Node, torch.fx.Node] = {}
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
                ag_node.args[0].op == "call_function"
                and ag_node.args[0].target
                == torch.ops.prims.convert_element_type.default
            ):
                cast_nodes.append(ag_node.args[0])
                cast_node_to_bucket_id[ag_node.args[0]] = bucket_id

    # Step 3: Create new (bucketed) all_gather nodes
    bucket_id_to_bucketed_op_info = {}
    bucket_id_is_scheduled = {}
    cast_bucket_id_is_scheduled = {}
    _, group_size, group_name = list(ag_node_to_wait_node.keys())[0].args
    for bucket_id, ag_bucket in enumerate(ag_buckets):
        ag_input_nodes = []
        wait_nodes = []
        for ag_node in ag_bucket:
            assert (
                ag_node in ag_node_to_wait_node
                and ag_node.args[1] == group_size
                and ag_node.args[2] == group_name
            )
            ag_input_nodes.append(ag_node.args[0])
            wait_nodes.append(ag_node_to_wait_node[ag_node])
        bucket_id_to_bucketed_op_info[bucket_id] = (
            ag_input_nodes,
            group_size,
            group_name,
            wait_nodes,
        )

    ag_wait_nodes = list(ag_node_to_wait_node.values())
    ag_and_wait_nodes = OrderedSet(ag_nodes + ag_wait_nodes)
    cast_nodes = OrderedSet(cast_nodes)
    new_graph: torch.fx.Graph = torch.fx.Graph()
    env: Dict[torch.fx.Node, torch.fx.Node] = {}

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
                # device = ag_input_nodes[0].meta["val"].device
                # rank = device.index
                # dtype = ag_input_nodes[0].meta["val"].dtype
                if all(
                    n.op == "call_function"
                    and n.target == torch.ops.prims.convert_element_type.default
                    for n in ag_input_nodes
                ):
                    param_all_gather_inputs = [
                        new_graph_call_function(
                            new_graph,
                            torch.ops.aten.empty.memory_format,
                            (n.meta["val"].shape,),
                            {
                                "dtype": n.args[1],  # n.meta["val"].dtype,
                                "device": n.meta["val"].device,
                                "pin_memory": False,
                            },
                        )
                        for n in ag_input_nodes
                    ]
                    for pp, n in zip(param_all_gather_inputs, ag_input_nodes):
                        pp.meta = n.meta.copy()

                    cast_input_nodes = [env[n.args[0]] for n in ag_input_nodes]
                    foreach_copy = new_graph_call_function(
                        new_graph,
                        torch.ops.aten._foreach_copy.default,
                        (param_all_gather_inputs, cast_input_nodes),
                        {},
                    )
                    foreach_copy.meta["val"] = [n.meta["val"] for n in ag_input_nodes]
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
                        env[old_n] = new_n
                else:
                    param_all_gather_inputs_orig = [
                        node_copy(
                            env,
                            new_graph,
                            ag_input_node,
                            lambda x: env_lookup(env, x, ag_input_node),
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
                device = ag_input_nodes[0].meta["val"].device
                rank = device.index
                dtype = ag_input_nodes[0].meta["val"].dtype
                # TODO: if we want to support mixed dtype in the same bucket,
                # we need to first view all all_gather inputs as uint8 (common denominator),
                # then do the all_gather, then view the output back to the original dtype.
                # Look at FSDP2 to see how to do this.
                assert all(n.meta["val"].dtype == dtype for n in ag_input_nodes), (
                    "All all_gather inputs in the same bucket must have the same dtype"
                )
                # must schedule all the all_gather input nodes first, before the bucketed all_gather node
                param_all_gather_inputs_orig = [
                    node_copy(
                        env,
                        new_graph,
                        ag_input_node,
                        lambda x: env_lookup(env, x, ag_input_node),
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
                all_gather_copy_in = new_graph_call_function(
                    new_graph,
                    torch.ops.fsdp.all_gather_copy_in.default,
                    (
                        param_all_gather_inputs_flattened,
                        inp_split_sizes,
                        all_gather_input_numel,
                        group_size,
                        rank,
                        dtype,
                        device,
                        group_name,
                        True,  # allocate_memory_from_process_group
                    ),
                    {},
                )
                all_gather_input = new_graph_call_function(
                    new_graph,
                    operator.getitem,
                    (all_gather_copy_in, 0),
                    {},
                )
                all_gather_output = new_graph_call_function(
                    new_graph,
                    operator.getitem,
                    (all_gather_copy_in, 1),
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
                split_with_sizes_copy = new_graph_call_function(
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
                    env[orig_wait_node] = out
                bucket_id_is_scheduled[bucket_id] = True
        else:
            continue
    gm.graph = new_graph


def bucket_size_determinator(bucket_id: int) -> float:
    """
    Determine the size of a bucket based on its ID.

    Args:
    bucket_id (int): The ID of the bucket.

    Returns:
    float: The size of the bucket.
    """
    return 2000.0
