# mypy: allow-untyped-defs
# pyre-strict
from __future__ import annotations

import heapq
import logging
import operator
import sys
from collections import defaultdict
from typing import Dict, List, Set, TYPE_CHECKING, Union, Callable, cast, Tuple, Any, Optional
import math
from torch._dispatch.python import enable_python_dispatcher
from .virtualized import V
from torch.utils._ordered_set import OrderedSet

import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._ordered_set import OrderedSet

from . import config, ir
from .dependencies import WeakDep
from .utils import (
    contains_collective,
    contains_wait,
    find_recursive_deps_of_node,
    find_recursive_users_of_node,
    is_collective,
    is_fallback_op,
    is_wait,
)


log = logging.getLogger(__name__)
overlap_log = torch._logging.getArtifactLogger(__name__, "overlap")

if TYPE_CHECKING:
    from .scheduler import BaseSchedulerNode


def sink_waits(snodes: List[BaseSchedulerNode]) -> List[BaseSchedulerNode]:
    """
    Greedily schedules waits as late as possible.
    """
    return _schedule_for_comm(
        snodes, raise_comms=False, sink_waits=True, reorder_for_overlap=False
    )


def raise_comms(snodes: List[BaseSchedulerNode]) -> List[BaseSchedulerNode]:
    """
    Greedily schedules comms as early as possible.
    """
    return _schedule_for_comm(
        snodes, raise_comms=True, sink_waits=False, reorder_for_overlap=False
    )


def reorder_compute_for_overlap(
    snodes: List[BaseSchedulerNode],
) -> List[BaseSchedulerNode]:
    """
    This achieves the following overall scheduling procedure:
        Step 1: Given that we've currently scheduled comm N, we now schedule all compute nodes
            that are required for comm N + 1 but do not depend on comm N, to run at the same time with comm N.
        Step 2: If all those compute nodes are sufficient to overlap comm N, we're done.
            Otherwise, we now need to look elsewhere to find compute that overlaps with comm N.
            We prioritize compute nodes that are needed sooner.
        Step 3: We schedule the compute nodes dependent on comm N and required for comm N + 1.
        Step 4: We schedule comm N + 1.
        Repeat this for subsequent comm nodes.
    """
    return _schedule_for_comm(
        snodes, raise_comms=True, sink_waits=True, reorder_for_overlap=True
    )


def _schedule_for_comm(
    snodes: List[BaseSchedulerNode],
    raise_comms: bool,
    sink_waits: bool,
    reorder_for_overlap: bool,
) -> List[BaseSchedulerNode]:
    """
    Schedule `snodes` for various comm optimization objectives.

    Args:
        snodes: the nodes to be scheduled.
        raise_comms: whether to greedily schedule collectives as early as possible
        sink_wait: whether to greedily schedule waits as late as possible
        reorder_compute_for_overlap: whether to reorder compute nodes to
            optimize for compute/communication overlapping.

    Returns:
        The new schedule order.

    Some notes on the synergy between different options:
        - `raise_comms` provides more overlapping oppurtunies for `reorder_compute_for_overlap`.
        - When both `raise_comms` and `sink_waits` is `True`, `raise_comms` is prioritized.
    """
    # We assign each node a tuple of scores (score_0, score_1, score_2),
    # decreasing in importance, with a lower value indicating a higher ranking:
    #
    # - score_0: the lowest comm_idx among the comm nodes that the node blocks.
    # If a node doesn't block any comm nodes, its score_0 is set to
    # sys.maxsize. This score ensures that comm nodes get scheduled as early as
    # possible.
    # - score_1: 1 if the node is a wait node, 0 otherwise. This score ensures
    # that wait nodes are deferred as late as possible.
    # - score_2: the index of the node in the original topological order. This
    # score provides stability in case of ties.
    #
    # When only raise_comms is True, only score_0 and score_2 are considered.
    # When only sink_waits is True, only score_1 and score_2 are considered.
    # When neither is True, the original order is yielded.
    buf_name_to_snode = {}
    name_to_fused_node = {}
    scores_0, scores_1, scores_2 = {}, {}, {}
    for idx, snode in enumerate(snodes):
        for buf_name in snode.get_buffer_names():
            buf_name_to_snode[buf_name] = snode

        for op_name in snode.get_operation_names():
            name_to_fused_node[op_name] = snode
        name_to_fused_node[snode.get_name()] = snode

        node_name = snode.get_name()
        scores_0[node_name] = sys.maxsize
        scores_1[node_name] = 0
        scores_2[node_name] = idx

    comm_idx = 0
    for snode in snodes:
        if raise_comms and contains_collective(snode):
            scores_0[snode.get_name()] = comm_idx
            for anc in snode.ancestors:
                anc_fused_name = name_to_fused_node[anc].get_name()
                scores_0[anc_fused_name] = min(scores_0[anc_fused_name], comm_idx)
            comm_idx += 1
        elif sink_waits and contains_wait(snode):
            scores_1[snode.get_name()] = 1

    class Runnable:
        def __init__(self, snode) -> None:
            self.snode = snode
            name = next(iter(snode.get_operation_names()))
            fused_name = name_to_fused_node[name].get_name()
            self.score = (
                scores_0[fused_name],
                scores_1[fused_name],
                scores_2[fused_name],
            )

        def __lt__(self, other):
            return self.score < other.score

    unmet_deps: Dict[BaseSchedulerNode, OrderedSet[str]] = {
        snode: OrderedSet(dep.name for dep in snode.unmet_dependencies)
        for snode in snodes
    }

    ready: List[Runnable] = []
    buffer_users: Dict[str, OrderedSet[BaseSchedulerNode]] = defaultdict(OrderedSet)
    snode_to_cost = {snode: estimate_op_runtime(snode) for snode in snodes}

    for snode, deps in unmet_deps.items():
        if len(deps) == 0:
            heapq.heappush(ready, Runnable(snode))
        for dep in deps:
            buffer_users[dep].add(snode)

    scheduled = []

    def schedule(snode):
        """
        Schedules `snode` and put all unblocked nodes onto the ready queue.
        """
        scheduled.append(snode)
        for buf_name in snode.get_buffer_names():
            for snode in buffer_users[buf_name]:
                unmet_deps[snode].remove(buf_name)
                if len(unmet_deps[snode]) == 0:
                    heapq.heappush(ready, Runnable(snode))

    def get_overlapping_candidate():
        """
        Return the next node in the ready queue that's neither a collective or
        a wait.
        """
        candidates = [
            x
            for x in ready
            if not contains_collective(x.snode) and not contains_wait(x.snode)
        ]
        if len(candidates) == 0:
            return None
        return min(candidates, key=lambda x: x.score)

    def schedule_collective_for_overlap(snode):
        """
        Schedules collective node `snode`, along with one or more compute nodes
        to overlap with it. The strategy is described in the comment of
        `reorder_compute_for_overlap`.
        """
        assert contains_collective(snode)
        schedule(snode)

        collective_cost = snode_to_cost[snode]
        while (
            collective_cost > 0
            and (candidate := get_overlapping_candidate()) is not None
        ):
            ready.remove(candidate)
            schedule(candidate.snode)
            collective_cost -= snode_to_cost[candidate.snode]
        heapq.heapify(ready)

    while len(ready):
        snode = heapq.heappop(ready).snode
        if reorder_for_overlap and contains_collective(snode):
            schedule_collective_for_overlap(snode)
        else:
            schedule(snode)

    for snode, deps in unmet_deps.items():
        assert len(deps) == 0, (
            "Detected unscheduled nodes. "
            f"Nodes with unmet dependencies: {unmet_deps}"
        )
    return scheduled


def decide_global_ordering_of_comms(
    nodes: List[BaseSchedulerNode], name_to_buf, name_to_fused_node
) -> List[BaseSchedulerNode]:
    """
    Decide global ordering of comms, by just enforcing the ordering that's in the input graph
    (might not be the same ordering as the eager mode program).
    TODO: Come up with a better approach
    """
    if not torch.distributed.is_available():
        return nodes

    comm_nodes = [n for n in nodes if contains_collective(n)]

    for i in range(1, len(comm_nodes)):
        # Enforce ordering by making previous comm a `WeakDep` dependency of the next comm
        mutating_buf = next(iter(comm_nodes[i].get_buffer_names()))
        for buf in comm_nodes[i - 1].get_buffer_names():
            comm_nodes[i].add_fake_dep(WeakDep(buf, mutating_buf=mutating_buf))

    return nodes


def estimate_op_runtime(snode: BaseSchedulerNode) -> float:
    """
    Returns estimated op runtime in nanoseconds (ns)
    """
    if config.estimate_op_runtime == "default":
        runtime = snode.get_estimated_runtime()
    else:
        assert callable(config.estimate_op_runtime)
        runtime = config.estimate_op_runtime(snode)
    return runtime


def node_summary(snode):
    detail = ""
    if isinstance(snode.node, ir.ExternKernelOut):
        detail = f" ({snode.node.python_kernel_name})"
    out_tensor_info = ""
    layout = snode.node.get_output_spec()
    if isinstance(layout, ir.Layout):
        out_tensor_info = f" (size={layout.size}, stride={layout.stride})"
    node_name = snode.node.maybe_get_name() or ""
    return f"{snode.node.__class__.__name__}{detail}{out_tensor_info} ({node_name})"


def visualize_overlap(order):
    total_est_runtime: float = 0.0
    cur_comm_node = None
    for snode in order:
        if cur_comm_node is None:
            if contains_collective(snode):
                total_est_runtime += estimate_op_runtime(snode)
                cur_comm_node = snode.node
            elif is_wait(snode.node):
                raise AssertionError(
                    "Wait is not expected when there is no collective running"
                )
            else:  # exposed compute op
                total_est_runtime += estimate_op_runtime(snode)
            overlap_log.debug(f"{node_summary(snode)}")  # noqa: G004
        else:  # cur_comm_node is not None
            if contains_collective(snode):
                raise AssertionError(
                    "Found two collectives running at the same time. "
                    "`visualize_overlap` needs to be updated to handle this case"
                )
            elif is_wait(snode.node):  # end of this comm op
                overlap_log.debug(f"{node_summary(snode)}")  # noqa: G004
                cur_comm_node = None
            else:  # overlapped compute op
                overlap_log.debug(f"| {node_summary(snode)}")  # noqa: G004
    overlap_log.debug(
        f"Est. runtime (ms): {total_est_runtime / 1000 / 1000}"  # noqa: G004
    )


def reorder_compute_and_comm_for_overlap(
    snodes: List[BaseSchedulerNode],
) -> List[BaseSchedulerNode]:
    order = snodes

    for p in config.reorder_for_compute_comm_overlap_passes:
        if isinstance(p, str) and p in globals():
            p = globals()[p]  # it is a builtin pass
        if torch.distributed.get_rank() == 0:
            overlap_log.debug(
                f"==== Visualize overlap before reordering pass {p} ===="  # noqa: G004
            )
            try:
                visualize_overlap(order)
            except Exception as e:
                overlap_log.debug(str(e))
        order = p(order)  # type: ignore[operator]
        if torch.distributed.get_rank() == 0:
            overlap_log.debug(
                f"==== Visualize overlap after reordering pass {p} ===="  # noqa: G004
            )
            try:
                visualize_overlap(order)
            except Exception as e:
                overlap_log.debug(str(e))
    return order


def bucket_fsdp_all_gather_concat(gm: torch.fx.GraphModule, all_gather_bucket_cap_mb: float) -> None:
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

    def is_graph_input(node: torch.fx.Node) -> bool:
        return node.op == "placeholder"

    node_list = gm.graph.nodes

    # Prerequisite: Check if there is any all_gather node
    found_all_gather = False
    for node in node_list:
        if is_all_gather_into_tensor(node):
            found_all_gather = True
            break
    if not found_all_gather:
        return

    ag_nodes: List[torch.fx.Node] = []
    cast_nodes: List[torch.fx.Node] = []
    ag_node_to_wait_node: Dict[torch.fx.Node, torch.fx.Node] = {}

    # Step 1: Find all all_gather nodes
    for node in node_list:
        if (
            is_wait_tensor(node)
            and is_all_gather_into_tensor(node.args[0])
        ):
            ag_wait_node = node
            ag_node = node.args[0]
            assert is_graph_input(ag_node.args[0]) or (
                ag_node.args[0].op == "call_function"
                and ag_node.args[0].target == torch.ops.prims.convert_element_type.default
                and is_graph_input(ag_node.args[0].args[0])
            ), f"Assume all_gather_into_tensor input is either graph input or dtype conversion of graph input, but got {ag_node.args[0]}"
            ag_nodes.append(ag_node)
            ag_node_to_wait_node[ag_node] = ag_wait_node
            if ag_node.args[0].op == "call_function" and ag_node.args[0].target == torch.ops.prims.convert_element_type.default:
                cast_nodes.append(ag_node.args[0])
    
    # Step 2: Put all_gather nodes into buckets
    ag_buckets: List[List[torch.fx.Node]] = []
    ag_node_to_bucket_id = {}
    cast_node_to_bucket_id = {}
    bucket_id_to_actual_bucket_size = {}
    cur_bucket: List[torch.fx.Node] = []
    cur_bucket_size_bytes: int = 0
    cur_bucket_id: int = 0
    # Convert MiB to bytes
    all_gather_bucket_size_bytes = int(all_gather_bucket_cap_mb * 1024 * 1024)
    for ag_node in ag_nodes:
        assert is_all_gather_into_tensor(ag_node)
        assert "val" in ag_node.meta
        ag_output_size_bytes = ag_node.meta["val"].numel() * torch.finfo(ag_node.meta["val"].dtype).bits // 8 
        if cur_bucket_size_bytes + ag_output_size_bytes > all_gather_bucket_size_bytes and cur_bucket:
            # Current bucket is full, create new bucket
            ag_buckets.append(cur_bucket)
            for n in cur_bucket:
                ag_node_to_bucket_id[n] = cur_bucket_id
                if n.args[0].op == "call_function" and n.args[0].target == torch.ops.prims.convert_element_type.default:
                    cast_node_to_bucket_id[n.args[0]] = cur_bucket_id
            bucket_id_to_actual_bucket_size[cur_bucket_id] = cur_bucket_size_bytes
            cur_bucket = []
            cur_bucket_size_bytes = 0
            cur_bucket_id += 1
        cur_bucket_size_bytes += ag_output_size_bytes
        cur_bucket.append(ag_node)
    if cur_bucket:
        # add remaining nodes in the last bucket
        ag_buckets.append(cur_bucket)
        for n in cur_bucket:
            ag_node_to_bucket_id[n] = cur_bucket_id
            if n.args[0].op == "call_function" and n.args[0].target == torch.ops.prims.convert_element_type.default:
                cast_node_to_bucket_id[n.args[0]] = cur_bucket_id
        bucket_id_to_actual_bucket_size[cur_bucket_id] = cur_bucket_size_bytes

    assert len(ag_buckets) > 0
    # for bucket_id, ag_bucket in enumerate(ag_buckets):
    #     log.warning(f"AG Bucket {bucket_id}: size: {bucket_id_to_actual_bucket_size[bucket_id]}, # AG nodes: {len(ag_bucket)}, AG nodes: {ag_bucket}")

    # Step 3: Create new (bucketed) all_gather nodes
    bucket_id_to_bucketed_op_info = {}
    bucket_id_is_scheduled = {}
    cast_bucket_id_is_scheduled = {}
    for bucket_id, ag_bucket in enumerate(ag_buckets):
        _, group_size, group_name = list(ag_node_to_wait_node.keys())[0].args
        ag_input_nodes = []
        wait_nodes = []
        for ag_node in ag_bucket:
            assert ag_node in ag_node_to_wait_node and ag_node.args[1] == group_size and ag_node.args[2] == group_name
            ag_input_nodes.append(ag_node.args[0])
            wait_nodes.append(ag_node_to_wait_node[ag_node])
        bucket_id_to_bucketed_op_info[bucket_id] = (ag_input_nodes, group_size, group_name, wait_nodes)

    ag_wait_nodes = list(ag_node_to_wait_node.values())
    ag_and_wait_nodes = OrderedSet(ag_nodes + ag_wait_nodes)
    cast_nodes = OrderedSet(cast_nodes)
    new_graph: torch.fx.Graph = torch.fx.Graph()
    env: Dict[torch.fx.Node, torch.fx.Node] = {}

    def env_lookup(x: torch.fx.Node, node_user: Union[torch.fx.Node, str]) -> torch.fx.Node:
        assert x in env, f"Dependent node {x} not in env when creating downstream node {node_user}"
        return env[x]

    def node_copy(node: torch.fx.Node, arg_transform: Callable[[torch.fx.Node], "Argument"]) -> torch.fx.Node:
        if node not in env:
            new_node = new_graph.node_copy(node, arg_transform=arg_transform)
            env[node] = new_node
        else:
            new_node = env[node]
        return new_node

    def new_graph_call_function(
        target: Callable[..., Any],
        args: Optional[Tuple["Argument", ...]] = None,
        kwargs: Optional[Dict[str, "Argument"]] = None,
        type_expr: Optional[Any] = None,
    ) -> torch.fx.Node:
        from torch.utils._pytree import tree_map_only
        new_node = new_graph.call_function(target, args, kwargs)
        args_val = tree_map_only(
            torch.fx.Node, lambda x: x.meta["val"], args
        )
        kwargs_val = tree_map_only(
            torch.fx.Node, lambda x: x.meta["val"], kwargs
        )
        with V.fake_mode, enable_python_dispatcher():
            new_fake_tensor = target(*args_val, **kwargs_val)
        new_node.meta["val"] = new_fake_tensor
        return new_node

    for node in node_list:
        if node not in ag_and_wait_nodes and node not in cast_nodes:
            # not cast-before-all_gather, all_gather or its wait_tensor - schedule it normally
            node_copy(node, lambda x: env_lookup(x, node))
        elif node in cast_nodes:
            # group cast nodes together
            assert node in cast_node_to_bucket_id
            bucket_id = cast_node_to_bucket_id[node]
            if bucket_id not in cast_bucket_id_is_scheduled:
                ag_input_nodes, group_size, group_name, orig_wait_nodes = bucket_id_to_bucketed_op_info[bucket_id]
                # device = ag_input_nodes[0].meta["val"].device
                # rank = device.index
                # dtype = ag_input_nodes[0].meta["val"].dtype
                if all(n.op == "call_function" and n.target == torch.ops.prims.convert_element_type.default for n in ag_input_nodes):
                    param_all_gather_inputs = [
                        new_graph_call_function(
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
                        torch.ops.aten._foreach_copy.default,
                        (param_all_gather_inputs, cast_input_nodes),
                        {}
                    )
                    foreach_copy.meta["val"] = [n.meta["val"] for n in ag_input_nodes]
                    getitems = [
                        new_graph_call_function(
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
                        node_copy(ag_input_node, lambda x: env_lookup(x, ag_input_node))
                        for ag_input_node in ag_input_nodes
                    ]
                cast_bucket_id_is_scheduled[bucket_id] = True
            else:
                continue
        elif node in ag_node_to_wait_node:
            assert node in ag_node_to_bucket_id
            bucket_id = ag_node_to_bucket_id[node]
            if bucket_id not in bucket_id_is_scheduled:
                ag_input_nodes, group_size, group_name, orig_wait_nodes = bucket_id_to_bucketed_op_info[bucket_id]
                device = ag_input_nodes[0].meta["val"].device
                rank = device.index
                dtype = ag_input_nodes[0].meta["val"].dtype
                # TODO(yf225): if we want to support mixed dtype in the same bucket, we need to first view all all_gather inputs as uint8 (common denominator),
                # then do the all_gather, then view the output back to the original dtype. Look at FSDP2 to see how to do this.
                assert all(n.meta["val"].dtype == dtype for n in ag_input_nodes), "All all_gather inputs in the same bucket must have the same dtype"
                # must schedule all the all_gather input nodes first, before the bucketed all_gather node
                param_all_gather_inputs_orig = [
                    node_copy(ag_input_node, lambda x: env_lookup(x, ag_input_node))
                    for ag_input_node in ag_input_nodes
                ]
                # schedule the bucketed all_gather node
                param_all_gather_inputs_flattened = [
                    new_graph_call_function(
                        torch.ops.aten.reshape.default,
                        (n, [-1]),
                        {}
                    )
                    for n in param_all_gather_inputs_orig
                ]
                inp_split_sizes = [n.meta["val"].numel() for n in param_all_gather_inputs_orig]
                param_all_gather_outputs = [
                    new_graph_call_function(
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
                # TODO(yf225): This assumes dim-0 sharding.
                # If we need to support sharding on another dim, we should look at how FSDP2 does it (e.g. search for `shard_dim` in FSDP2 codebase)
                param_all_gather_outputs_shape_orig = [
                    (n.meta["val"].shape[0] * group_size,) + n.meta["val"].shape[1:] for n in param_all_gather_inputs_orig
                ]
                all_gather_input_numel = sum(inp_split_sizes)
                all_gather_copy_in = new_graph_call_function(
                    torch.ops.fsdp.all_gather_copy_in.default,
                    (
                        param_all_gather_inputs_flattened,
                        inp_split_sizes,
                        all_gather_input_numel,
                        group_size,
                        rank,
                        dtype,
                        device,
                    ),
                    {},
                )
                all_gather_input = new_graph_call_function(
                    operator.getitem,
                    (all_gather_copy_in, 0),
                    {},
                )
                all_gather_output = new_graph_call_function(
                    operator.getitem,
                    (all_gather_copy_in, 1),
                    {},
                )
                all_gather_into_tensor_out = new_graph_call_function(
                    torch.ops._c10d_functional.all_gather_into_tensor_out.default,
                    (all_gather_input, group_size, group_name),
                    {"out": all_gather_output},
                )
                wait_tensor = new_graph_call_function(
                    torch.ops._c10d_functional.wait_tensor.default,
                    (all_gather_into_tensor_out,),
                    {},
                )
                all_gather_output_reshaped = new_graph_call_function(
                    torch.ops.aten.reshape.default,
                    (wait_tensor, [group_size, -1]),
                    {},
                )
                outs_flattened = [new_graph_call_function(
                    torch.ops.aten.reshape.default,
                    (n, [group_size, -1]),
                    {},
                ) for n in param_all_gather_outputs]
                split_with_sizes_copy = new_graph_call_function(
                    torch.ops.fsdp.split_with_sizes_copy.default,
                    (all_gather_output_reshaped, inp_split_sizes),
                    {
                        "dim": 1, "out": outs_flattened
                    },
                )
                outs = [new_graph_call_function(
                    torch.ops.aten.reshape.default,
                    (n, orig_shape),
                    {},
                ) for n, orig_shape in zip(outs_flattened, param_all_gather_outputs_shape_orig)]
                assert len(orig_wait_nodes) == len(outs), f"len(orig_wait_nodes)={len(orig_wait_nodes)}, len(outs)={len(outs)}, orig_wait_nodes={orig_wait_nodes}, outs={outs}"
                assert len(orig_wait_nodes) > 0
                for orig_wait_node, out in zip(orig_wait_nodes, outs):
                    env[orig_wait_node] = out
                bucket_id_is_scheduled[bucket_id] = True
        else:
            continue
    gm.graph = new_graph


def bucket_fsdp_reduce_scatter_concat(gm: torch.fx.GraphModule, reduce_scatter_bucket_cap_mb: float) -> None:
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

    def find_recursive_users_of_fx_node(
        node, collected_node_set, criteria_cb=None
    ):
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

    node_list = list(gm.graph.nodes)

    # Prerequisite: Check if there is any reduce_scatter node
    found_reduce_scatter = False
    for node in node_list:
        if is_reduce_scatter_tensor(node):
            found_reduce_scatter = True
            break
    if not found_reduce_scatter:
        return

    rs_nodes: List[torch.fx.Node] = []
    rs_node_to_wait_node: Dict[torch.fx.Node, torch.fx.Node] = {}

    # Step 1: Find all reduce_scatter nodes
    for node in node_list:
        if (
            is_wait_tensor(node)
            and is_reduce_scatter_tensor(node.args[0])
        ):
            rs_wait_node = node
            rs_node = node.args[0]
            rs_nodes.append(rs_node)
            rs_node_to_wait_node[rs_node] = rs_wait_node
    
    # Step 2: Put reduce_scatter nodes into buckets
    rs_buckets: List[List[torch.fx.Node]] = []
    rs_node_to_bucket_id = {}
    bucket_id_to_actual_bucket_size = {}
    cur_bucket: List[torch.fx.Node] = []
    cur_bucket_size_bytes: int = 0
    cur_bucket_id: int = 0
    # Convert MiB to bytes
    reduce_scatter_bucket_size_bytes = int(reduce_scatter_bucket_cap_mb * 1024 * 1024)
    for rs_node in rs_nodes:
        assert is_reduce_scatter_tensor(rs_node)
        rs_input = rs_node.args[0]
        assert "val" in rs_input.meta
        rs_input_size_bytes = rs_input.meta["val"].numel() * torch.finfo(rs_input.meta["val"].dtype).bits // 8 
        if cur_bucket_size_bytes + rs_input_size_bytes > reduce_scatter_bucket_size_bytes and cur_bucket:
            # Current bucket is full, create new bucket
            rs_buckets.append(cur_bucket)
            for n in cur_bucket:
                rs_node_to_bucket_id[n] = cur_bucket_id
            bucket_id_to_actual_bucket_size[cur_bucket_id] = cur_bucket_size_bytes
            cur_bucket = []
            cur_bucket_size_bytes = 0
            cur_bucket_id += 1
        cur_bucket_size_bytes += rs_input_size_bytes
        cur_bucket.append(rs_node)
    if cur_bucket:
        # add remaining nodes in the last bucket
        rs_buckets.append(cur_bucket)
        for n in cur_bucket:
            rs_node_to_bucket_id[n] = cur_bucket_id
        bucket_id_to_actual_bucket_size[cur_bucket_id] = cur_bucket_size_bytes

    assert len(rs_buckets) > 0
    # for bucket_id, rs_bucket in enumerate(rs_buckets):
    #     log.warning(f"RS Bucket {bucket_id}: size: {bucket_id_to_actual_bucket_size[bucket_id]}, # RS nodes: {len(rs_bucket)}, RS nodes: {rs_bucket}")

    # Step 3: Create new (bucketed) reduce_scatter nodes
    order = {x: i for i, x in enumerate(node_list)}
    rs_wait_nodes = list(rs_node_to_wait_node.values())
    for n in rs_wait_nodes:
        assert len(n.users) == 1, f"Expect only one user for {n}, but got {n.users}"
    rs_and_its_recursive_users = OrderedSet(rs_nodes + rs_wait_nodes)
    
    bucket_id_to_bucketed_op_info = {}
    bucket_id_is_scheduled = {}
    for bucket_id, rs_bucket in enumerate(rs_buckets):
        _, reduce_op, group_size, group_name = list(rs_node_to_wait_node.keys())[0].args
        rs_input_nodes = []
        wait_nodes = []
        wait_node_recursive_users = OrderedSet()
        for rs_node in rs_bucket:
            assert rs_node in rs_node_to_wait_node and rs_node.args[1] == reduce_op and rs_node.args[2] == group_size and rs_node.args[3] == group_name
            rs_input_nodes.append(rs_node.args[0])
            wait_node = rs_node_to_wait_node[rs_node]
            wait_nodes.append(wait_node)
            find_recursive_users_of_fx_node(wait_node, wait_node_recursive_users)
            rs_and_its_recursive_users |= wait_node_recursive_users
        bucket_id_to_bucketed_op_info[bucket_id] = (rs_input_nodes, reduce_op, group_size, group_name, wait_nodes, wait_node_recursive_users)

    new_graph: torch.fx.Graph = torch.fx.Graph()
    env: Dict[torch.fx.Node, torch.fx.Node] = {}

    def env_lookup(x: torch.fx.Node, node_user: Union[torch.fx.Node, str]) -> torch.fx.Node:
        assert x in env, f"Dependent node {x} not in env when creating downstream node {node_user}"
        return env[x]

    def node_copy(node: torch.fx.Node, arg_transform: Callable[[torch.fx.Node], "Argument"]) -> torch.fx.Node:
        new_node = new_graph.node_copy(node, arg_transform=arg_transform)
        env[node] = new_node
        return new_node

    def new_graph_call_function(
        target: Callable[..., Any],
        args: Optional[Tuple["Argument", ...]] = None,
        kwargs: Optional[Dict[str, "Argument"]] = None,
        type_expr: Optional[Any] = None,
    ) -> torch.fx.Node:
        from torch.utils._pytree import tree_map_only
        new_node = new_graph.call_function(target, args, kwargs)
        args_val = tree_map_only(
            torch.fx.Node, lambda x: x.meta["val"], args
        )
        kwargs_val = tree_map_only(
            torch.fx.Node, lambda x: x.meta["val"], kwargs
        )
        with V.fake_mode, enable_python_dispatcher():
            new_fake_tensor = target(*args_val, **kwargs_val)
        new_node.meta["val"] = new_fake_tensor
        return new_node

    for node in node_list:
        if node not in rs_and_its_recursive_users:
            # not reduce_scatter or its (recursive) users - schedule it normally
            node_copy(node, lambda x: env_lookup(x, node))
        elif node in rs_node_to_wait_node:
            assert node in rs_node_to_bucket_id
            bucket_id = rs_node_to_bucket_id[node]
            if bucket_id not in bucket_id_is_scheduled and rs_buckets[bucket_id][-1] == node:
                # If we are at the last node in the bucket, we can start to schedule the bucketed reduce_scatter node
                rs_input_nodes, reduce_op, group_size, group_name, orig_wait_nodes, orig_wait_node_recursive_users = bucket_id_to_bucketed_op_info[bucket_id]
                unsharded_grads = [
                    node_copy(rs_input_node, lambda x: env_lookup(x, rs_input_node))
                    for rs_input_node in rs_input_nodes
                ]
                reduce_dtype = unsharded_grads[0].meta["val"].dtype
                # Only float32 and bfloat16 are supported for now.
                # To support fp16, please see FSDP2 `_get_gradient_divide_factors`.
                assert reduce_dtype in (torch.float32, torch.bfloat16), f"reduce_dtype {reduce_dtype} is not supported"
                assert all(grad.meta["val"].dtype == reduce_dtype for grad in unsharded_grads)
                device = unsharded_grads[0].meta["val"].device
                rank = device.index
                # TODO(yf225): need more work if we want to support non-dim-0 sharding (e.g. search for `shard_dim` in FSDP2 codebase)
                shard_dim = 0

                def _get_dim0_padded_size(tensor_size: torch.Size, dim0_factor: int) -> torch.Size:
                    padded_dim0 = math.ceil(tensor_size[0] / dim0_factor) * dim0_factor
                    return cast(torch.Size, torch.Size([padded_dim0]) + tensor_size[1:])

                padded_unsharded_sizes = tuple(
                    _get_dim0_padded_size(grad.meta["val"].size(), group_size) for grad in unsharded_grads
                )
                reduce_scatter_input_numel = sum(s.numel() for s in padded_unsharded_sizes)
                reduce_scatter_input = new_graph_call_function(
                    torch.ops.aten.empty.memory_format,
                    ([reduce_scatter_input_numel],),
                    {
                        "dtype": reduce_dtype,
                        "device": device,
                        "pin_memory": False,
                    }
                )
                reduce_scatter_input_reshaped = new_graph_call_function(
                    torch.ops.aten.reshape.default,
                    (reduce_scatter_input, [group_size, -1]),
                    {},
                )
                chunk_cat = new_graph_call_function(
                    torch.ops.fsdp.chunk_cat.default,
                    (unsharded_grads,),
                    {
                        "dim": 0,
                        "num_chunks": group_size,
                        "out": reduce_scatter_input_reshaped,
                    }
                )
                reduce_scatter_tensor = new_graph_call_function(
                    torch.ops._c10d_functional.reduce_scatter_tensor.default,
                    (reduce_scatter_input, reduce_op, group_size, group_name),
                    {},
                )
                wait_tensor = new_graph_call_function(
                    torch.ops._c10d_functional.wait_tensor.default,
                    (reduce_scatter_tensor,),
                    {},
                )

                def _chunk_with_empty(
                    tensor: torch.Tensor, num_chunks: int, dim: int
                ) -> List[torch.Tensor]:
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
                    chunks = _chunk_with_empty(torch.empty_like(unsharded_grad.meta["val"], device="meta"), group_size, dim=shard_dim)
                    sharded_param = chunks[rank]
                    sharded_size = sharded_param.size()
                    contiguous_sharded_stride = torch._prims_common.make_contiguous_strides_for(sharded_size)
                    # Assume even sharding for Shard(i), i > 0; otherwise would require
                    # copy-out for contiguous strides
                    new_sharded_grad = new_graph_call_function(
                        torch.ops.aten.as_strided.default,
                        (reduce_output,),
                        {
                            "size": sharded_size,
                            "stride": contiguous_sharded_stride,
                            "storage_offset": flat_grad_offset,
                        },
                    )
                    new_sharded_grads.append(new_sharded_grad)
                    padded_sharded_numel = padded_unsharded_size.numel() // group_size
                    flat_grad_offset += padded_sharded_numel
                assert len(orig_wait_nodes) == len(new_sharded_grads), f"len(orig_wait_nodes)={len(orig_wait_nodes)}, len(new_sharded_grads)={len(new_sharded_grads)}, orig_wait_nodes={orig_wait_nodes}, new_sharded_grads={new_sharded_grads}"
                assert len(orig_wait_nodes) > 0
                for orig_wait_node, new_sharded_grad in zip(orig_wait_nodes, new_sharded_grads):
                    env[orig_wait_node] = new_sharded_grad
                for user in sorted(orig_wait_node_recursive_users, key=lambda x: order[x]):
                    if user.op != "output":
                        node_copy(user, lambda x: env_lookup(x, user))
                bucket_id_is_scheduled[bucket_id] = True
        else:
            continue
    assert node_list[-1].op == "output"
    output_node = node_list[-1]
    node_copy(output_node, lambda x: env_lookup(x, output_node))
    gm.graph = new_graph


def remove_fsdp2_unsharded_param_graph_input_usage(graph: torch.fx.Graph):
    """
    This FX graph pass replaces uses of FSDP2 unsharded params with their corresponding
    graph intermediates that were fsdp.copy_ into the unsharded params in the original graph.

    NOTE: Can only apply this pass to any of the FSDP2 unsharded params that have this pattern
    (or repetition of): `resize_(full) -> copy_ -> resize_(0)`. Because of this, for partial-graph case
    where `resize_(full) -> copy_` is in one graph and `resize_(0)` is in another graph, we can't
    remove these resize and copy ops and thus we will have worse performance there.

    In other words, "do we try to remove all the resize_(full) -> copy_ -> resize_(0) nodes for this unsharded param"
    is actually a per-unsharded-param decision, since for each unsharded param, we look at its resize sequence pattern
    (in `check_resize_pattern()`) to determine if its set of resize and copy nodes can be removed.
    """
    node_list = list(graph.nodes)

    # Find all graph inputs and their resize counts
    graph_input_to_resized_to_full_node_idxes = defaultdict(list)
    graph_input_to_resized_to_0_node_idxes = defaultdict(list)
    for idx, node in enumerate(node_list):
        if (
            node.op == "call_function"
            and node.target == torch.ops.inductor.resize_storage_bytes_.default
        ):
            assert (
                node.args[0].op == "placeholder"
            ), f"""\
Resize can only operate on graph inputs, but got {node} which is resizing non-graph-input {node.args[0]}
"""
            graph_input = node.args[0]
            new_size = node.args[1]
            if new_size > 0:
                graph_input_to_resized_to_full_node_idxes[graph_input].append(idx)
            else:
                graph_input_to_resized_to_0_node_idxes[graph_input].append(idx)

    def check_resize_pattern(graph_input):
        # Check the number of resize-to-full and resize-to-0 nodes are equal,
        # and that for each (resize-to-full, resize-to-0) pair, the resize-to-full node
        # always happens before the resize-to-0 node.
        # This is the precondition for being able to remove all the resize and copy nodes
        # for this specific unsharded param.
        resized_to_full_idxes = graph_input_to_resized_to_full_node_idxes.get(
            graph_input, []
        )
        resized_to_0_idxes = graph_input_to_resized_to_0_node_idxes.get(graph_input, [])

        if not len(resized_to_full_idxes) == len(resized_to_0_idxes):
            log.warning(
                f"""
Unequal number of resize-to-full and resize-to-0 nodes for graph input {graph_input}:
{len(resized_to_full_idxes)} vs. {len(resized_to_0_idxes)}.
Skipping `remove_fsdp2_unsharded_param_graph_input_usage` FX graph pass.
"""  # noqa: G004
            )
            return False

        # Check the sequence: (resize_to_full -> resize_to_0)+
        for resize_to_full_idx, resize_to_0_idx in zip(
            resized_to_full_idxes, resized_to_0_idxes
        ):
            if resize_to_full_idx >= resize_to_0_idx:
                log.warning(
                    f"""
For graph input {graph_input}: resize-to-full node {node_list[resize_to_full_idx]} at index {resize_to_full_idx}
happens after resize-to-0 node {node_list[resize_to_0_idx]} at index {resize_to_0_idx}.
Skipping `remove_fsdp2_unsharded_param_graph_input_usage` FX graph pass for that unsharded param.
"""  # noqa: G004
                )
                return False
        return True

    # Find all eligible unsharded params and their corresponding graph intermediates.
    unsharded_param_to_fsdp_copy_node_idxes = defaultdict(list)
    for idx, node in enumerate(node_list):
        if node.op == "call_function" and node.target == torch.ops.fsdp.copy_.default:
            fsdp_copy_node = node
            unsharded_param = node.args[0]
            assert (
                unsharded_param.op == "placeholder"
            ), f"""
Assumed all FSDP2 `unsharded_param`s to be graph input, but it's not true!
Offending node: {unsharded_param}. Graph: {graph}
"""
            if check_resize_pattern(unsharded_param):
                unsharded_param_to_fsdp_copy_node_idxes[unsharded_param].append(idx)

    def is_allowed_mutation(node):
        return (
            node.target == torch.ops.fsdp.copy_.default
            or node.target == torch.ops.inductor.resize_storage_bytes_.default
        )

    def is_node_mutating_unsharded_param_or_its_alias(node, unsharded_params):
        # Check whether the node is mutating any of the unsharded params or their aliases.
        mutated_arg_idxes = (
            [
                i
                for i, x in enumerate(node.target._schema.arguments)
                if x.alias_info is not None and x.alias_info.is_write
            ]
            if isinstance(node.target, torch._ops.OpOverload)
            else []
        )
        mutated_node_arg_storages = OrderedSet(
            [
                StorageWeakRef(node.args[i].meta["val"].untyped_storage())
                for i in mutated_arg_idxes
            ]
        )
        storages_of_unsharded_params = OrderedSet(
            [
                StorageWeakRef(unsharded_param.meta["val"].untyped_storage())
                for unsharded_param in unsharded_params
            ]
        )
        return len(mutated_node_arg_storages & storages_of_unsharded_params) > 0

    # Check no user mutation on any unsharded_param
    for node in node_list:
        if (
            node.op == "call_function"
            and isinstance(node.target, torch._ops.OpOverload)
            and node.target._schema.is_mutable
            and not is_allowed_mutation(node)
        ):
            assert not is_node_mutating_unsharded_param_or_its_alias(
                node, unsharded_param_to_fsdp_copy_node_idxes.keys()
            ), f"""\
User mutation on FSDP2 unsharded param is not allowed when Traceable FSDP2 is used. Violating node: {node}
"""

    # For each `fsdp.copy_(unsharded_param, Y)`, replace downstream usage of `unsharded_param` with `Y`.
    #
    # NOTE: Because of "layer reuse" use case, there could be multiple `fsdp.copy_` to the same `unsharded_param` graph input.
    # e.g.
    # ```
    #     fsdp_copy_1 = fsdp.copy_(unsharded_param_1, Y1)
    #     ... (use of unsharded_param_1)                     -> Subgraph 1
    #     fsdp_copy_2 = fsdp.copy_(unsharded_param_1, Y2)
    #     ... (use of unsharded_param_1)                     -> Subgraph 2
    #     fsdp_copy_3 = fsdp.copy_(unsharded_param_1, Y3)
    #     ... (use of unsharded_param_1)                     -> Subgraph 3
    # ```
    # We must do the replacement only within each subgraph.
    for (
        unsharded_param,
        fsdp_copy_node_idxes,
    ) in unsharded_param_to_fsdp_copy_node_idxes.items():
        for i, fsdp_copy_node_idx in enumerate(fsdp_copy_node_idxes):
            fsdp_copy_node = node_list[fsdp_copy_node_idx]
            assert fsdp_copy_node.args[0] is unsharded_param
            _, replacement = fsdp_copy_node.args
            # subgraph_start_idx is exclusive
            subgraph_start_idx = fsdp_copy_node_idx + 1
            # subgraph_end_idx is exclusive (also intentionally don't replace args in return op)
            subgraph_end_idx = (
                fsdp_copy_node_idxes[i + 1]
                if i < len(fsdp_copy_node_idxes) - 1
                else len(node_list) - 1
            )
            subgraph_nodes = node_list[subgraph_start_idx:subgraph_end_idx]
            assert not any(
                is_node_mutating_unsharded_param_or_its_alias(node, [unsharded_param])
                for node in subgraph_nodes
            ), f"""\
Assumed no ops mutating unsharded param {unsharded_param} in subgraph {subgraph_nodes}, but it's not true!
Graph: {graph}
"""
            for node in subgraph_nodes:
                if (
                    node.op == "call_function"
                    and unsharded_param in node.args
                    and node.target != torch.ops.inductor.resize_storage_bytes_.default
                ):  # TODO(yf225): implement replacement in kwargs
                    new_args = tuple(
                        replacement if arg is unsharded_param else arg
                        for arg in node.args
                    )
                    node.args = new_args

    # Delete `fsdp.copy_(unsharded_param, Y)` nodes
    for (
        unsharded_param,
        fsdp_copy_node_idxes,
    ) in unsharded_param_to_fsdp_copy_node_idxes.items():
        for i, fsdp_copy_node_idx in enumerate(fsdp_copy_node_idxes):
            fsdp_copy_node = node_list[fsdp_copy_node_idx]
            graph.erase_node(fsdp_copy_node)

    # Delete `resize_(unsharded_param, ...)` nodes
    for node in node_list:
        if (
            node.op == "call_function"
            and node.target == torch.ops.inductor.resize_storage_bytes_.default
            and node.args[0] in unsharded_param_to_fsdp_copy_node_idxes
        ):
            graph.erase_node(node)


def reinplace_fsdp_all_gather(graph: torch.fx.Graph) -> None:
    try:
        import torch.distributed.fsdp._fully_shard._fsdp_collectives

        assert torch.distributed.is_available()
        # Assert existence of these ops
        assert (
            torch.ops._c10d_functional.all_gather_into_tensor
            and torch.ops._c10d_functional.all_gather_into_tensor_out
        )
    except (ImportError, AttributeError, AssertionError):
        return

    from .pattern_matcher import (
        CallFunction,
        KeywordArg,
        Match,
        PatternMatcherPass,
        register_graph_pattern,
    )

    """
    all_gather_copy_in = torch.ops.fsdp.all_gather_copy_in.default(...);
    getitem = all_gather_copy_in[0];
    (getitem_1 = all_gather_copy_in[1];)  # optional

    all_gather_into_tensor = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem, ...);

    ->

    all_gather_copy_in = torch.ops.fsdp.all_gather_copy_in.default(...);
    getitem = all_gather_copy_in[0];
    getitem_1 = all_gather_copy_in[1];

    all_gather_into_tensor = torch.ops._c10d_functional.all_gather_into_tensor_out.default(getitem, ..., out=getitem_1);
    """

    def remove_unused_getitem(g):
        # Remove `getitem_X = all_gather_copy_in[1]` which is never used.
        node_list = list(g.nodes)
        for n in node_list:
            if (
                n.target == operator.getitem
                and n.args[0].target is torch.ops.fsdp.all_gather_copy_in.default
                and n.args[1] == 1
            ):
                g.erase_node(n)

    graph_pass = PatternMatcherPass()

    @register_graph_pattern(
        CallFunction(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            CallFunction(
                operator.getitem,
                CallFunction(
                    torch.ops.fsdp.all_gather_copy_in.default,
                    KeywordArg("all_gather_inputs"),
                    KeywordArg("inp_split_sizes"),
                    KeywordArg("all_gather_input_numel"),
                    KeywordArg("world_size"),
                    KeywordArg("rank"),
                    KeywordArg("dtype"),
                    KeywordArg("device"),
                ),
                KeywordArg("item_idx"),
            ),
            KeywordArg("group_size"),
            KeywordArg("group_name"),
        ),
        pass_dict=graph_pass,
        extra_check=lambda match: match.kwargs["item_idx"] == 0,
    )
    def reinplace_all_gather(match: Match, *args, **kwargs):
        def repl(
            *args,
        ):
            copy_in_args = args[:-2]
            group_size = args[-2]
            group_name = args[-1]
            all_gather_copy_in = torch.ops.fsdp.all_gather_copy_in.default(
                *copy_in_args
            )
            getitem = all_gather_copy_in[0]
            getitem_1 = all_gather_copy_in[1]
            all_gather_into_tensor = (
                torch.ops._c10d_functional.all_gather_into_tensor_out.default(
                    getitem, group_size, group_name, out=getitem_1
                )
            )
            return all_gather_into_tensor

        match.replace_by_example(
            repl,
            [
                kwargs["all_gather_inputs"],
                kwargs["inp_split_sizes"],
                kwargs["all_gather_input_numel"],
                kwargs["world_size"],
                kwargs["rank"],
                kwargs["dtype"],
                kwargs["device"],
                kwargs["group_size"],
                kwargs["group_name"],
            ],
        )

    remove_unused_getitem(graph)
    graph_pass.apply(graph)  # type: ignore[arg-type]


def get_op_idx(snode):
    assert not isinstance(
        snode,
        (
            torch._inductor.scheduler.FusedSchedulerNode,
            torch._inductor.scheduler.GroupedSchedulerNode,
        ),
    )
    return int(snode.get_name()[2:])


def enforce_comm_ordering_for_fsdp(
    snodes: List[torch._inductor.scheduler.BaseSchedulerNode],
    name_to_buf: Dict[str, torch._inductor.scheduler.SchedulerBuffer],
    name_to_fused_node: Dict[str, BaseSchedulerNode],
) -> List[torch._inductor.scheduler.BaseSchedulerNode]:
    from . import scheduler

    new_order: list[BaseSchedulerNode] = []
    scheduled = OrderedSet[Any]()
    ag_exists = False
    rs_exists = False
    ag_grouped_node_to_wait_grouped_node = {}
    rs_grouped_node_to_wait_grouped_node = {}
    snode_name_to_final_snode = {}

    def _create_group_node(snodes_to_group):
        group_node = scheduler.GroupedSchedulerNode.create(snodes_to_group)
        for snode in snodes_to_group:
            snode_name_to_final_snode[snode.get_name()] = group_node
        snode_name_to_final_snode[group_node.get_name()] = group_node
        return group_node

    # Create grouped nodes for specific sets of ops
    for snode in snodes:
        # Case 1: Handle AllGather
        if is_collective(
            snode.node, op=torch.ops._c10d_functional.all_gather_into_tensor_out.default
        ) and any(
            is_fallback_op(
                name_to_fused_node[x].node, torch.ops.fsdp.all_gather_copy_in.default
            )
            for x in snode.ancestors
        ):
            ag_exists = True
            ag_snode = snode
            ag_related_snode_set: OrderedSet[scheduler.BaseSchedulerNode] = OrderedSet()

            # Find the "cast + copy_in + getitem + all_gather" code block
            find_recursive_deps_of_node(
                ag_snode,
                ag_related_snode_set,
                name_to_buf,
                name_to_fused_node,
            )

            # Find the "all_gather + all_gather_wait_tensor + copy_out" code block
            allowed_ops = OrderedSet(
                [
                    torch.ops._c10d_functional.all_gather_into_tensor_out.default,
                    torch.ops._c10d_functional.wait_tensor.default,
                    torch.ops.fsdp.split_with_sizes_copy.default,
                ]
            )
            find_recursive_users_of_node(
                ag_snode,
                ag_related_snode_set,
                name_to_buf,
                name_to_fused_node,
                criteria_cb=lambda x: not (
                    isinstance(x, scheduler.NopKernelSchedulerNode)
                    or (
                        isinstance(x, scheduler.ExternKernelSchedulerNode)
                        and x.node.op_overload in allowed_ops  # type: ignore[union-attr]
                    )
                ),
            )

            # sort nodes by original operation order
            ag_related_snodes = sorted(
                ag_related_snode_set, key=lambda x: get_op_idx(x)
            )

            # In the "reuse layer" case, some ops in the 2nd all-gather code block could also
            # depend on ops in the 1st all-gather code block, and we don't want to group them together.
            end_idx_of_current_ag_block = len(ag_related_snodes)
            copy_out_count = 0
            for i in range(len(ag_related_snodes)):
                cur_snode = ag_related_snodes[i]
                if is_fallback_op(
                    cur_snode.node, torch.ops.fsdp.split_with_sizes_copy.default
                ):
                    copy_out_count += 1
                if copy_out_count > 1:
                    end_idx_of_current_ag_block = i
                    break

            ag_related_snodes = ag_related_snodes[:end_idx_of_current_ag_block]

            # Group "cast + copy_in + getitem + all_gather" into one GroupedSchedulerNode
            wait_node_idx = None
            for i in range(len(ag_related_snodes) - 1):
                if isinstance(ag_related_snodes[i + 1].node, ir._WaitKernel):
                    wait_node_idx = i + 1
                    break
            assert wait_node_idx is not None
            ag_group_node = _create_group_node(ag_related_snodes[:wait_node_idx])

            # Group "all_gather_wait_tensor + copy_out" into one GroupedSchedulerNode
            ag_wait_group_node = _create_group_node(ag_related_snodes[wait_node_idx:])

            ag_grouped_node_to_wait_grouped_node[ag_group_node] = ag_wait_group_node

        # Case 2: Handle ReduceScatter
        elif is_fallback_op(snode.node, torch.ops.fsdp.chunk_cat.default):
            rs_exists = True
            rs_snode = snode

            # Find the "reduce_scatter copy-in + reduce_scatter comm + reduce_scatter wait" code block
            rs_related_snode_set: OrderedSet[scheduler.BaseSchedulerNode] = OrderedSet()
            find_recursive_users_of_node(
                rs_snode,
                rs_related_snode_set,
                name_to_buf,
                name_to_fused_node,
            )

            # sort nodes by original operation order
            rs_related_snodes = sorted(
                rs_related_snode_set, key=lambda x: get_op_idx(x)
            )

            # Group "reduce_scatter copy-in + reduce_scatter comm" into one GroupedSchedulerNode
            wait_node_idx = None
            for i in range(len(rs_related_snodes) - 1):
                if isinstance(rs_related_snodes[i + 1].node, ir._WaitKernel):
                    wait_node_idx = i + 1
                    break
            assert wait_node_idx is not None
            rs_group_node = _create_group_node(rs_related_snodes[:wait_node_idx])

            # Group "reduce_scatter wait + related output nodes" into one GroupedSchedulerNode
            rs_wait_group_node = _create_group_node(rs_related_snodes[wait_node_idx:])

            rs_grouped_node_to_wait_grouped_node[rs_group_node] = rs_wait_group_node

    assert len(snode_name_to_final_snode) > 0
    if ag_exists:
        assert len(ag_grouped_node_to_wait_grouped_node) > 0
    if rs_exists:
        assert len(rs_grouped_node_to_wait_grouped_node) > 0

    # Build the new node schedule, taking GroupedSchedulerNode into account
    for snode in snodes:
        if snode.get_name() in snode_name_to_final_snode:
            snode = snode_name_to_final_snode[snode.get_name()]
        if snode in scheduled:
            continue
        new_order.append(snode)
        scheduled.add(snode)

    # Enforce AllGather ordering: previous AllGather's "wait then copy_out" group node must run
    # before next AllGather's "copy_in then AG" group node
    prev_ag_wait = None
    for ag_group_node, wait_group_node in ag_grouped_node_to_wait_grouped_node.items():
        if prev_ag_wait is not None:
            mutating_buf = next(iter(ag_group_node.get_buffer_names()))
            for o in prev_ag_wait.get_outputs():
                ag_group_node.add_fake_dep(
                    WeakDep(o.get_name(), mutating_buf=mutating_buf)
                )
        prev_ag_wait = wait_group_node

    # Enforce ReduceScatter ordering: previous ReduceScatter's "wait" group node must run
    # before next ReduceScatter's "copy_in then RS" group node
    prev_rs_wait = None
    for rs_group_node, wait_group_node in rs_grouped_node_to_wait_grouped_node.items():
        if prev_rs_wait is not None:
            mutating_buf = next(iter(rs_group_node.get_buffer_names()))
            for o in prev_rs_wait.get_outputs():
                rs_group_node.add_fake_dep(
                    WeakDep(o.get_name(), mutating_buf=mutating_buf)
                )
        prev_rs_wait = wait_group_node

    return new_order  # type: ignore[return-value]
