# mypy: allow-untyped-defs
# pyre-strict

import logging
from typing import Dict, List

import torch

from . import config, ir, scheduler
from .dependencies import WeakDep
from .pass_utils import (
    collect_node_to_input_prev_writes,
    get_all_names,
    get_all_reads,
    get_users_from_unfused_nodes,
)
from .utils import contains_collective, contains_wait, tuple_sorted

overlap_log = torch._logging.getArtifactLogger(__name__, "overlap")
torch_log = logging.getLogger("torch")


def sink_waits(
    snodes: List["scheduler.BaseSchedulerNode"],
) -> List["scheduler.BaseSchedulerNode"]:
    """
    Greedily moves waits as late as possible (i.e. until we reach a use). Optimal in terms of
    communication overlap.
    """
    new_order = []
    cur_waits = set()
    for snode in snodes:
        if contains_wait(snode):
            cur_waits.add(snode)
        else:
            for wait in tuple_sorted(cur_waits):
                if snode in wait.node_users:
                    new_order.append(wait)
                    cur_waits.remove(wait)
            new_order.append(snode)
    new_order.extend(tuple_sorted(cur_waits))
    return new_order


def raise_comms(
    snodes: List["scheduler.BaseSchedulerNode"],
) -> List["scheduler.BaseSchedulerNode"]:
    """
    Greedily moves comms as early as possible (i.e. until we reach an input).
    Optimal in terms of communication overlap.
    """
    new_order_reversed: List[scheduler.BaseSchedulerNode] = []
    cur_comms: List[scheduler.BaseSchedulerNode] = []
    for snode in reversed(snodes):
        if contains_collective(snode):
            cur_comms.append(snode)
        else:
            while len(cur_comms) > 0 and any(
                snode in comm.inverse_users for comm in cur_comms
            ):
                comm = cur_comms.pop(0)
                new_order_reversed.append(comm)
            new_order_reversed.append(snode)
    assert len(cur_comms) <= 1
    new_order_reversed.extend(tuple_sorted(cur_comms))
    return new_order_reversed[::-1]


def get_ancestors(node):
    ancestors = set()
    cur_nodes = [node]
    while len(cur_nodes) > 0:
        new_nodes = []
        for node in cur_nodes:
            for inp in node.inverse_users:
                if inp not in ancestors:
                    ancestors.add(inp)
                    new_nodes.append(inp)
        cur_nodes = new_nodes
    return ancestors


def get_descendants(node):
    descendants = set()
    cur_nodes = [node]
    while len(cur_nodes) > 0:
        new_nodes = []
        for node in cur_nodes:
            for inp in node.node_users:
                if inp not in descendants:
                    descendants.add(inp)
                    new_nodes.append(inp)
        cur_nodes = new_nodes
    return descendants


def decide_global_ordering_of_comms(nodes: List["scheduler.BaseSchedulerNode"]):
    """
    Decide global ordering of comms, by just enforcing the ordering that's in the input graph
    (might not be the same ordering as the eager mode program).
    TODO: Come up with a better approach
    """
    comm_nodes = [n for n in nodes if contains_collective(n)]
    for i in range(1, len(comm_nodes)):
        # Enforce ordering by making previous comm a `WeakDep` dependency of the next comm
        comm_nodes[i].add_fake_dep(WeakDep(comm_nodes[i - 1].get_name()))


def assert_no_comm_nodes(snodes: List["scheduler.BaseSchedulerNode"]) -> None:
    assert not any(contains_collective(snode) for snode in snodes)


def estimate_op_runtime(snode: "scheduler.BaseSchedulerNode") -> float:
    """
    Returns estimated op runtime in nanoseconds (ns)
    """
    if config.estimate_op_runtime == "default":
        runtime = snode.get_estimated_runtime()
    else:
        assert callable(config.estimate_op_runtime)
        runtime = config.estimate_op_runtime(snode)
    return runtime


def reorder_compute_for_overlap(
    snodes: List["scheduler.BaseSchedulerNode"],
) -> List["scheduler.BaseSchedulerNode"]:
    """
    Decides a global ordering of all compute and communication nodes,
    assuming that we already have a global ordering of communication nodes.

    Overall scheduling procedure is:
        Step 1: Given that we've currently scheduled comm N, we now schedule all compute nodes
            that are required for comm N + 1 but do not depend on comm N, to run at the same time with comm N.
        Step 2: If all those compute nodes are sufficient to overlap comm N, we're done.
            Otherwise, we now need to look elsewhere to find compute that overlaps with comm N.
            We prioritize compute nodes that are needed sooner.
        Step 3: We schedule the compute nodes dependent on comm N and required for comm N + 1.
        Step 4: We schedule comm N + 1.
        Repeat this for subsequent comm nodes.
    """
    final_order = []

    comm_nodes = []
    for snode in snodes:
        if contains_collective(snode):
            comm_nodes.append(snode)
    if len(comm_nodes) == 0:
        # if there is no comm nodes, return the current order
        return snodes

    comm_ancestors = {node: get_ancestors(node) for node in comm_nodes}
    comm_descendants = {node: get_descendants(node) for node in comm_nodes}

    indeg = dict.fromkeys(snodes, 0)
    for snode in snodes:
        for user in snode.node_users:
            if user in indeg:
                indeg[user] += 1
    ready_to_schedule_nodes = {node for node in snodes if indeg[node] == 0}

    unscheduled_nodes = set()
    unscheduled_nodes = set(snodes)

    def schedule_node(snode):
        """
        Schedule a single node.
        """
        assert snode in unscheduled_nodes
        assert snode in ready_to_schedule_nodes
        ready_to_schedule_nodes.remove(snode)
        unscheduled_nodes.remove(snode)
        final_order.append(snode)
        for user in tuple_sorted(snode.node_users):
            if user in indeg:
                indeg[user] -= 1
                if indeg[user] == 0:
                    ready_to_schedule_nodes.add(user)

    def schedule_nodes(snodes):
        """
        Schedules all nodes in `snodes` in an arbitrary topologically valid order.
        """
        all_nodes = set(snodes)
        assert all(node in unscheduled_nodes for node in all_nodes)
        while len(all_nodes) > 0:
            # NOTE: since model graph is always a DAG and does not have circular dependency inside,
            # there should be at least one node that is a "free node" (i.e. indeg == 0),
            # hence infinite loop is not possible. But we check here just to be safe.
            progress = False
            for node in tuple_sorted(all_nodes):
                if node in ready_to_schedule_nodes:
                    schedule_node(node)
                    all_nodes.remove(node)
                    progress = True
            if not progress:
                raise AssertionError(
                    "Unable to find a free node (indeg == 0). This is an impossible state to reach. "
                    "Please report a bug to PyTorch."
                )

    # First, schedule all compute nodes that are required by first comm node,
    # as well as the first comm node itself.
    assert len(comm_nodes) > 0
    schedule_nodes(
        list(comm_ancestors[comm_nodes[0]]) + [comm_nodes[0]],
    )

    rolled_over_compute_cost = 0
    for idx in range(1, len(comm_ancestors)):
        # Step 1: Given that we've currently scheduled comm `idx-1`, we now schedule
        # all compute nodes that are required for comm `idx` but do not depend on comm `idx-1`,
        # to run at the same time with comm `idx-1`.
        needed_by_next_comm_and_ready_compute_nodes = unscheduled_nodes & (
            comm_ancestors[comm_nodes[idx]] - comm_descendants[comm_nodes[idx - 1]]
        )
        assert_no_comm_nodes(needed_by_next_comm_and_ready_compute_nodes)

        total_compute_runtime_cost = rolled_over_compute_cost + sum(
            estimate_op_runtime(node)
            for node in needed_by_next_comm_and_ready_compute_nodes
        )
        prev_comm_runtime_cost = estimate_op_runtime(comm_nodes[idx - 1])
        schedule_nodes(tuple_sorted(needed_by_next_comm_and_ready_compute_nodes))

        # Step 2: If all those compute nodes are sufficient to overlap comm `idx-1`, we're done.
        # Otherwise, we now need to look elsewhere to find compute that overlaps with comm `idx`.
        # We prioritize compute nodes that are needed sooner.
        step1_runtime_cost = total_compute_runtime_cost
        if step1_runtime_cost >= prev_comm_runtime_cost:
            pass
        else:
            # Find all ready to schedule compute nodes that do not depend on comm `idx-1`.
            ready_to_schedule_compute_nodes = tuple_sorted(
                ready_to_schedule_nodes - comm_descendants[comm_nodes[idx - 1]]
            )
            assert_no_comm_nodes(ready_to_schedule_compute_nodes)

            def earliest_comm_descendant(node):
                for idx in range(len(comm_nodes)):
                    if node in comm_ancestors[comm_nodes[idx]]:
                        return idx
                return len(comm_nodes)

            # Prioritize compute nodes that are needed sooner.
            ready_to_schedule_compute_nodes = sorted(
                ready_to_schedule_compute_nodes, key=earliest_comm_descendant
            )

            for snode in ready_to_schedule_compute_nodes:
                if total_compute_runtime_cost >= prev_comm_runtime_cost:
                    # If accumulated compute runtime cost is greater than comm `idx-1` runtime cost,
                    # it means we have maximized overlap for comm `idx-1`, and hence we stop looking
                    # for more compute to schedule.
                    break
                compute_runtime_cost = estimate_op_runtime(snode)
                # If we're not able to leverage more than half of this
                # node's compute to overlap, we skip it.
                # TODO: Smarter heuristics here
                if (
                    prev_comm_runtime_cost - total_compute_runtime_cost
                ) <= compute_runtime_cost / 2:
                    continue
                schedule_node(snode)
                total_compute_runtime_cost += compute_runtime_cost
        rollable_compute_cost = total_compute_runtime_cost - step1_runtime_cost

        # Step 3: We schedule the compute nodes dependent on comm `idx-1` and required for comm `idx`.
        needed_by_next_comm_nodes = unscheduled_nodes & comm_ancestors[comm_nodes[idx]]
        schedule_nodes(list(needed_by_next_comm_nodes))

        # Step 4: We schedule comm `idx`.
        schedule_nodes([comm_nodes[idx]])

        is_prev_comm_blocking_next_comm = len(needed_by_next_comm_nodes) > 0
        # The idea here is that if there are no compute nodes from Step 3
        # (i.e. if prev comm is not blocking next comm), we can roll over the compute nodes
        # in Step 2 to overlap with the next comm, since they're not required to finish
        # before the next comm starts.
        if is_prev_comm_blocking_next_comm:
            rolled_over_compute_cost = 0
        else:
            rolled_over_compute_cost = rollable_compute_cost  # type: ignore[assignment]

    schedule_nodes(unscheduled_nodes)
    return final_order


def node_summary(snode):
    detail = ""
    if isinstance(snode.node, ir.ExternKernelOut):
        detail = f" ({snode.node.python_kernel_name})"
    out_tensor_info = ""
    if (
        hasattr(snode.node, "layout")
        and hasattr(snode.node.layout, "size")
        and hasattr(snode.node.layout, "stride")
    ):
        out_tensor_info = (
            f" (size={snode.node.layout.size}, stride={snode.node.layout.stride})"
        )
    node_name = ""
    if hasattr(snode.node, "name"):
        node_name = snode.node.name
    return f"{snode.node.__class__.__name__}{detail}{out_tensor_info} ({node_name})"


def visualize_overlap(order):
    total_est_runtime: float = 0.0
    cur_comm_node = None
    for snode in order:
        if cur_comm_node is None:
            if contains_collective(snode):
                total_est_runtime += estimate_op_runtime(snode)
                cur_comm_node = snode.node
            elif contains_wait(snode):
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
            elif contains_wait(snode):  # end of this comm op
                overlap_log.debug(f"{node_summary(snode)}")  # noqa: G004
                cur_comm_node = None
            else:  # overlapped compute op
                overlap_log.debug(f"| {node_summary(snode)}")  # noqa: G004
    overlap_log.debug(
        f"Est. runtime (ms): {total_est_runtime / 1000 / 1000}"  # noqa: G004
    )


def reorder_compute_and_comm_for_overlap(
    snodes: List["scheduler.BaseSchedulerNode"],
) -> List["scheduler.BaseSchedulerNode"]:
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


def enforce_comm_ordering_for_fsdp(
    name_to_fused_node: Dict[str, "scheduler.BaseSchedulerNode"],
    graph_inputs: Dict[str, "Buffer"],
    snodes: List["scheduler.BaseSchedulerNode"],
) -> List["scheduler.BaseSchedulerNode"]:
    def _find_all_recursive_deps_of_node_up_to_the_graph_input(
        snode, collected_node_set
    ):
        collected_node_set.add(snode)
        for dep in get_all_reads(snode):
            if dep.name not in graph_inputs:
                dep_node = name_to_fused_node[dep.name]
                if dep_node in collected_node_set:
                    continue
                _find_all_recursive_deps_of_node_up_to_the_graph_input(
                    dep_node, collected_node_set
                )

    def _find_all_recursive_users_of_node_down_to_the_target_node_type(
        snode, target_node_type_check_fn, collected_node_set
    ):
        collected_node_set.add(snode)
        if target_node_type_check_fn(snode):
            return
        else:
            for user in snode.users:
                if user.node in collected_node_set:
                    continue
                if isinstance(user.node, scheduler.OutputNode):
                    continue
                _find_all_recursive_users_of_node_down_to_the_target_node_type(
                    user.node, target_node_type_check_fn, collected_node_set
                )

    for snode in snodes:
        torch_log.warning(
            f"snode: {snode}, snode.node: {snode.node}, snode.debug_str(): {snode.debug_str()}"
        )

    new_order = []
    scheduled = set()
    prev_all_gather_wait_tensor_then_copy_out_node = None
    rs_copyin_to_comm_and_wait = {}

    def _create_group_node(nodes_to_group):
        group_node = scheduler.GroupedSchedulerNode.create_group(nodes_to_group)
        scheduled.update(nodes_to_group)
        new_order.append(group_node)
        name_to_fused_node[group_node.get_name()] = group_node
        for node in nodes_to_group:
            name_to_fused_node[node.get_name()] = group_node
        return group_node

    for snode in snodes:
        if snode in scheduled:
            continue
        if (
            isinstance(snode.node, ir.FallbackKernel)
            and snode.node.op_overload is torch.ops.fsdp.all_gather_copy_in.default
        ):
            # Find the "cast + copy_in + getitem + all_gather + all_gather_wait_tensor + copy_out" block
            collected_node_set = set()
            # torch_log.warning(f"before: len(collected_node_set): {len(collected_node_set)}, collected_node_set: {collected_node_set}")
            _find_all_recursive_deps_of_node_up_to_the_graph_input(
                snode,
                collected_node_set,
            )
            # torch_log.warning(f"after collect dep: len(collected_node_set): {len(collected_node_set)}, collected_node_set: {collected_node_set}")
            _find_all_recursive_users_of_node_down_to_the_target_node_type(
                snode,
                lambda sn: (
                    isinstance(sn.node, ir.ExternKernel)
                    and sn.node.op_overload
                    is torch.ops.fsdp.split_with_sizes_copy.default
                ),
                collected_node_set,
            )
            # torch_log.warning(f"after collect users: len(collected_node_set): {len(collected_node_set)}, collected_node_set: {collected_node_set}")

            # sort nodes by original buffer order
            collected_nodes = sorted(
                list(collected_node_set), key=lambda x: int(x.get_name()[3:])
            )
            # # torch_log.warning(f"collected_nodes: {collected_nodes}")
            # copy_out_node = None
            # for n in collected_nodes:
            #     # torch_log.warning(f"type(n.node): {type(n.node)}")
            #     if isinstance(n.node, ir.ExternKernel) and n.node.op_overload is torch.ops.fsdp.split_with_sizes_copy.default:
            #         copy_out_node = n
            #         break
            # assert copy_out_node is not None
            # copy_out_multioutput_nodes = [x.node for x in copy_out_node.users if x.node.get_name() in name_to_fused_node]
            # assert all(isinstance(snode.node, ir.MultiOutput) for snode in copy_out_multioutput_nodes)
            # collected_nodes = collected_nodes + copy_out_multioutput_nodes

            # Group "cast + copy_in + getitem + all_gather" into one GroupedSchedulerNode
            nodes_to_group = []
            wait_node_idx = None
            for i in range(len(collected_nodes) - 1):
                node = collected_nodes[i]
                nodes_to_group.append(node)
                if isinstance(collected_nodes[i + 1].node, ir._WaitKernel):
                    wait_node_idx = i + 1
                    break
            comm_group_node = _create_group_node(nodes_to_group)

            # Enforce ordering: previous AllGather's "wait then copy_out" group node must run before next AllGather's "copy_in then AG" group node
            if prev_all_gather_wait_tensor_then_copy_out_node is not None:
                comm_group_node.add_fake_dep(
                    WeakDep(prev_all_gather_wait_tensor_then_copy_out_node.get_name())
                )

            # Group "all_gather_wait_tensor + copy_out" into one GroupedSchedulerNode
            nodes_to_group = collected_nodes[wait_node_idx:]
            wait_group_node = _create_group_node(nodes_to_group)
            prev_all_gather_wait_tensor_then_copy_out_node = wait_group_node
        elif (
            isinstance(snode.node, ir.FallbackKernel)
            and snode.node.op_overload is torch.ops.fsdp.chunk_cat.default
        ):
            # Find the "reduce_scatter copy-in + reduce_scatter comm + reduce_scatter wait" block
            collected_node_set = set()
            _find_all_recursive_users_of_node_down_to_the_target_node_type(
                snode,
                lambda sn: (
                    isinstance(sn.node, ir._WaitKernel)
                    and isinstance(sn.node.inputs[0], ir._CollectiveKernel)
                    and sn.node.inputs[0].op_overload
                    is torch.ops._c10d_functional.reduce_scatter_tensor.default
                ),
                collected_node_set,
            )
            # sort nodes by original buffer order
            collected_nodes = sorted(
                list(collected_node_set), key=lambda x: int(x.get_name()[3:])
            )

            # Group "reduce_scatter copy-in + reduce_scatter comm" into one GroupedSchedulerNode
            nodes_to_group = []
            wait_node = None
            wait_node_idx = None
            for i in range(len(collected_nodes) - 1):
                node = collected_nodes[i]
                nodes_to_group.append(node)
                if isinstance(collected_nodes[i + 1].node, ir._WaitKernel):
                    wait_node = collected_nodes[i + 1]
                    wait_node_idx = i + 1
                    break
            assert wait_node is not None
            comm_group_node = _create_group_node(nodes_to_group)
            # torch_log.warning(f"Created RS comm group node for: {nodes_to_group}")

            # Group "reduce_scatter wait + related output node" into one GroupedSchedulerNode
            assert isinstance(wait_node.users[0].node.node, ir.MutationOutput)
            nodes_to_group = [wait_node, wait_node.users[0].node]
            wait_group_node = _create_group_node(nodes_to_group)
            # torch_log.warning(f"Created RS wait group node for: {nodes_to_group}")

            # Create mapping from RS copy-in node to comm node and RS wait node
            chunk_cat_node = comm_group_node.snodes[0]
            rs_copyin_to_comm_and_wait[chunk_cat_node.get_name()] = (
                comm_group_node,
                wait_group_node,
            )
        else:
            scheduled.add(snode)
            new_order.append(snode)

    if len(rs_copyin_to_comm_and_wait) == 0:
        return new_order

    snode_to_idx = {}
    for i, snode in enumerate(new_order):
        snode_to_idx[snode.get_name()] = i
        if isinstance(snode, scheduler._BaseGroupedSchedulerNode):
            for sub_snode in snode.snodes:
                snode_to_idx[sub_snode.get_name()] = i
    snode_to_idx.update({inp: 0 for inp in graph_inputs.keys()})

    def _schedule_node(node, node_order, scheduled):
        if node.get_name() in scheduled:
            return None
        node_order.append(node)
        scheduled.update(get_all_names(node))

    # Step 2: Prioritize scheduling all ReduceScatter and their wait nodes (including their deps)
    node_to_input_prev_writes = collect_node_to_input_prev_writes(new_order)

    def _get_dep_chain(snode_name, dist):
        # We need `dist` to limit the amount of backtracking we do, to avoid this being too expensive.
        snode = name_to_fused_node[snode_name]
        dep_chain = set()
        dep_chain.add(snode_name)
        if dist == 0:
            return dep_chain
        for dep in get_all_reads(snode):
            dep_name = dep.name
            if (
                dep_name in graph_inputs
            ):  # for now, we don't count distance to graph inputs
                continue
            if dep_name not in graph_inputs:
                upstream_dep_chain = _get_dep_chain(dep_name, dist - 1)
                dep_chain.update(upstream_dep_chain)
        return dep_chain

    recursive_deps_of_rs_copyin = {
        n: sorted(
            list(_get_dep_chain(n, dist=2)),
            key=lambda name: int(name.split("_")[0][3:]),
        )
        for n in rs_copyin_to_comm_and_wait.keys()
    }

    new_order2 = []
    scheduled = set()
    scheduled.update(graph_inputs.keys())
    for snode in new_order:
        if snode.get_name() not in scheduled:
            _schedule_node(snode, new_order2, scheduled)
        for user in get_users_from_unfused_nodes(snode):
            if user.node.get_name() in scheduled:
                continue
            if user.node.get_name() == "OUTPUT":
                continue
            user_names = get_all_names(user.node)
            input_prev_writes_of_user = node_to_input_prev_writes[user.node]
            # If a user node U is a read dep of a prioritized node A, schedule U and also try to see if we can schedule A.
            for rs_copyin_name in recursive_deps_of_rs_copyin:
                dep_chain = recursive_deps_of_rs_copyin[rs_copyin_name]
                for dep_name in dep_chain:
                    if dep_name in scheduled:
                        continue
                    dep_node = name_to_fused_node[dep_name]
                    all_reads_of_dep = set(x.name for x in get_all_reads(dep_node))
                    input_prev_writes_of_dep = node_to_input_prev_writes[dep_node]
                    if any(name in all_reads_of_dep for name in user_names) and all(
                        prev_write_name in scheduled
                        for prev_write_name in input_prev_writes_of_user
                    ):
                        _schedule_node(user.node, new_order2, scheduled)
                    if all(
                        prev_write_name in scheduled
                        for prev_write_name in input_prev_writes_of_dep
                    ):
                        _schedule_node(dep_node, new_order2, scheduled)
                        if rs_copyin_name in scheduled:
                            rs_comm_node, rs_wait_node = rs_copyin_to_comm_and_wait[
                                rs_copyin_name
                            ]
                            _schedule_node(rs_comm_node, new_order2, scheduled)
                            _schedule_node(rs_wait_node, new_order2, scheduled)
    return new_order2
