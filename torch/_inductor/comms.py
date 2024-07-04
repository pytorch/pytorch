# mypy: allow-untyped-defs
# pyre-strict
from __future__ import annotations

import heapq

import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple, TYPE_CHECKING

import torch

from . import config, ir
from .dependencies import WeakDep
from .utils import contains_collective, contains_wait, tuple_sorted

overlap_log = torch._logging.getArtifactLogger(__name__, "overlap")

if TYPE_CHECKING:
    from .scheduler import BaseSchedulerNode


def sink_waits(snodes: List[BaseSchedulerNode], *args) -> List[BaseSchedulerNode]:
    """
    Greedily moves waits as late as possible.
    Optimal in terms of communication overlap.
    """
    return raise_comms_and_sink_waits(snodes, raise_comms=False, sink_waits=True)


def raise_comms(snodes: List[BaseSchedulerNode], *args) -> List[BaseSchedulerNode]:
    """
    Greedily moves comms as early as possible.
    Optimal in terms of communication overlap.
    """
    return raise_comms_and_sink_waits(snodes, raise_comms=True, sink_waits=False)


def raise_comms_and_sink_waits(
    snodes: List[BaseSchedulerNode],
    *args,
    raise_comms: bool = True,
    sink_waits: bool = True,
) -> List[BaseSchedulerNode]:
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
    name_to_snode = {}
    scores_0, scores_1, scores_2 = {}, {}, {}
    for idx, snode in enumerate(snodes):
        for name in snode.get_names():
            name_to_snode[name] = snode
            scores_0[name] = sys.maxsize
            scores_1[name] = 0
            scores_2[name] = idx

    comm_idx = 0
    for snode in snodes:
        if raise_comms and contains_collective(snode.node):
            scores_0[snode.get_name()] = comm_idx
            for anc in snode.ancestors:
                scores_0[anc] = min(scores_0[anc], comm_idx)
            comm_idx += 1
        elif sink_waits and contains_wait(snode.node):
            scores_1[snode.get_name()] = 1

    class Runnable:
        def __init__(self, snode):
            self.snode = snode
            name = next(iter(snode.get_names()))
            self.score = (
                scores_0[name],
                scores_1[name],
                scores_2[name],
            )

        def __lt__(self, other):
            return self.score < other.score

    unmet_deps: Dict[BaseSchedulerNode, Set[str]] = {}
    for snode in snodes:
        # A mutating node's unmet_dependencies doesn't cover the dependencies
        # caused by the mutation. Instead, they are described by associated
        # MutationOutput node. Thus, to safely schedule a mutating node, we
        # have to add the unmet_dependencies of the associated MutationOutput
        # nodes to the mutating node.
        if isinstance(snode.node, ir.MutationOutput):
            src_name = snode.node.node_doing_mutating.get_name()
            src_snode = name_to_snode[src_name]
            assert src_snode in unmet_deps
            unmet_deps[src_snode] |= {
                dep.name for dep in snode.unmet_dependencies if dep.name != src_name
            }
        assert snode not in unmet_deps
        unmet_deps[snode] = {dep.name for dep in snode.unmet_dependencies}

    ready: List[Runnable] = []
    snode_num_deps: Dict[BaseSchedulerNode, int] = {}
    buffer_users: Dict[str, Set[BaseSchedulerNode]] = defaultdict(set)

    for snode, deps in unmet_deps.items():
        snode_num_deps[snode] = len(deps)
        if len(deps) == 0:
            heapq.heappush(ready, Runnable(snode))
        for dep in deps:
            buffer_users[dep].add(snode)

    scheduled = []
    while len(ready):
        curr = heapq.heappop(ready).snode
        scheduled.append(curr)
        for curr_name in curr.get_names():
            for snode in buffer_users[curr_name]:
                snode_num_deps[snode] -= 1
                if snode_num_deps[snode] == 0:
                    heapq.heappush(ready, Runnable(snode))

    for snode, num_deps in snode_num_deps.items():
        assert num_deps == 0, "Unscheduled nodes"
    return scheduled


def get_ancestors(node, inverse_users):
    ancestors = set()
    cur_nodes = [node]
    while len(cur_nodes) > 0:
        new_nodes = []
        for node in cur_nodes:
            for inp in inverse_users[node]:
                if inp not in ancestors:
                    ancestors.add(inp)
                    new_nodes.append(inp)
        cur_nodes = new_nodes
    return ancestors


def get_descendants(node, node_users):
    descendants = set()
    cur_nodes = [node]
    while len(cur_nodes) > 0:
        new_nodes = []
        for node in cur_nodes:
            for inp in node_users[node]:
                if inp not in descendants:
                    descendants.add(inp)
                    new_nodes.append(inp)
        cur_nodes = new_nodes
    return descendants


def decide_global_ordering_of_comms(nodes: List[BaseSchedulerNode]):
    """
    Decide global ordering of comms, by just enforcing the ordering that's in the input graph
    (might not be the same ordering as the eager mode program).
    TODO: Come up with a better approach
    """
    comm_nodes = [n for n in nodes if contains_collective(n.node)]

    def item(x: Set[str]) -> str:
        assert len(x) == 1
        return next(iter(x))

    for i in range(1, len(comm_nodes)):
        # Enforce ordering by making previous comm a `WeakDep` dependency of the next comm
        comm_nodes[i].add_fake_dep(WeakDep(item(comm_nodes[i - 1].get_buffer_names())))


def assert_no_comm_nodes(snodes: List[BaseSchedulerNode]) -> None:
    assert not any(contains_collective(snode.node) for snode in snodes)


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


def compute_node_users(
    snodes: List[BaseSchedulerNode],
) -> Tuple[
    Dict[BaseSchedulerNode, Set[BaseSchedulerNode]],
    Dict[BaseSchedulerNode, Set[BaseSchedulerNode]],
]:
    from .scheduler import FusedSchedulerNode

    # set up buffer name to (fused)snode mapping
    buf_to_snode: Dict[str, BaseSchedulerNode] = {}
    for node in snodes:
        if isinstance(node, FusedSchedulerNode):
            for x in node.snodes:
                for buf in x.get_outputs():
                    buf_to_snode[buf.get_name()] = node

        for buf in node.get_outputs():
            buf_to_snode[buf.get_name()] = node

    # compute inverse_users
    inverse_users = {
        node: {buf_to_snode[dep.name] for dep in node.unmet_dependencies}
        for node in snodes
    }

    # compute node_users
    # TODO: ideally, we should deduplicate .users and .node_users,
    # but currently .users contains extra information that's difficult to
    # extract into a standalone container.
    node_users: Dict[BaseSchedulerNode, Set[BaseSchedulerNode]] = defaultdict(set)
    for node, node_inverse_users in inverse_users.items():
        for inverse_user in node_inverse_users:
            node_users[inverse_user].add(node)

    return inverse_users, node_users


def reorder_compute_for_overlap(
    snodes: List[BaseSchedulerNode],
    node_users: Dict[BaseSchedulerNode, Set[BaseSchedulerNode]],
    inverse_users: Dict[BaseSchedulerNode, Set[BaseSchedulerNode]],
) -> List[BaseSchedulerNode]:
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
        if contains_collective(snode.node):
            comm_nodes.append(snode)
    if len(comm_nodes) == 0:
        # if there is no comm nodes, return the current order
        return snodes

    comm_ancestors = {node: get_ancestors(node, inverse_users) for node in comm_nodes}
    comm_descendants = {node: get_descendants(node, node_users) for node in comm_nodes}

    indeg = dict.fromkeys(snodes, 0)
    for snode in snodes:
        for user in node_users[snode]:
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
        for user in tuple_sorted(node_users[snode]):
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
            if contains_collective(snode.node):
                total_est_runtime += estimate_op_runtime(snode)
                cur_comm_node = snode.node
            elif contains_wait(snode.node):
                raise AssertionError(
                    "Wait is not expected when there is no collective running"
                )
            else:  # exposed compute op
                total_est_runtime += estimate_op_runtime(snode)
            overlap_log.debug(f"{node_summary(snode)}")  # noqa: G004
        else:  # cur_comm_node is not None
            if contains_collective(snode.node):
                raise AssertionError(
                    "Found two collectives running at the same time. "
                    "`visualize_overlap` needs to be updated to handle this case"
                )
            elif contains_wait(snode.node):  # end of this comm op
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
    inverse_users, node_users = compute_node_users(snodes)

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
        order = p(order, node_users, inverse_users)  # type: ignore[operator]
        if torch.distributed.get_rank() == 0:
            overlap_log.debug(
                f"==== Visualize overlap after reordering pass {p} ===="  # noqa: G004
            )
            try:
                visualize_overlap(order)
            except Exception as e:
                overlap_log.debug(str(e))
    return order


def get_all_reads(snode):
    reads = set()
    reads.update(snode.read_writes.reads)
    if isinstance(
        snode, (scheduler.FusedSchedulerNode, scheduler.GroupedSchedulerNode)
    ):
        for sub_snode in snode.snodes:
            reads.update(get_all_reads(sub_snode))
    return reads


def is_collective_op(snode, op):
    return isinstance(snode.node, ir.CollectiveKernel) and snode.node.op_overload is op


def is_fallback_op(snode):
    return isinstance(snode.node, ir._FallbackKernel) and snode.node.op_overload is op


# TODO: can we simplify the input args to just be `snodes`?
def enforce_comm_ordering_for_fsdp(
    name_to_fused_node: Dict[str, scheduler.BaseSchedulerNode],
    graph_inputs: Dict[str, Buffer],
    snodes: List[scheduler.BaseSchedulerNode],
) -> List[scheduler.BaseSchedulerNode]:
    def _find_all_recursive_deps_of_node_up_to_criteria(
        snode, collected_node_set, criteria_cb
    ):
        collected_node_set.add(snode)
        for dep in get_all_reads(snode):
            if not criteria_cb(dep):
                dep_node = name_to_fused_node[dep.name]
                if dep_node in collected_node_set:
                    continue
                _find_all_recursive_deps_of_node_up_to_criteria(
                    dep_node, collected_node_set, criteria_cb
                )

    new_order = []
    scheduled = set()
    ag_nodes = []
    rs_nodes = []

    def _create_group_node(nodes_to_group):
        group_node = scheduler.GroupedSchedulerNode.create(nodes_to_group)
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
            and snode.node.op_overload is torch.ops.fsdp.split_with_sizes_copy.default
        ):
            # Case 1: Handle AllGather

            # Find the "cast + copy_in + getitem + all_gather + all_gather_wait_tensor + copy_out" block
            collected_node_set = set()
            _find_all_recursive_deps_of_node_up_to_criteria(
                snode,
                collected_node_set,
                criteria_cb=lambda dep: dep.name in graph_inputs,
            )

            # sort nodes by original buffer order
            collected_nodes = sorted(
                collected_node_set, key=lambda x: int(x.get_name()[3:])
            )

            # Group "cast + copy_in + getitem + all_gather" into one GroupedSchedulerNode
            nodes_to_group = []
            wait_node_idx = None
            for i in range(len(collected_nodes) - 1):
                node = collected_nodes[i]
                nodes_to_group.append(node)
                if isinstance(collected_nodes[i + 1].node, ir._WaitKernel):
                    wait_node_idx = i + 1
                    break
            ag_group_node = _create_group_node(nodes_to_group)

            # Group "all_gather_wait_tensor + copy_out" into one GroupedSchedulerNode
            nodes_to_group = collected_nodes[wait_node_idx:]
            wait_group_node = _create_group_node(nodes_to_group)
            ag_nodes.append(
                (
                    ag_group_node,
                    wait_group_node,
                )
            )
        elif (
            isinstance(snode.node, ir._WaitKernel)
            and any(
                is_collective_op(
                    x, torch.ops._c10d_functional.reduce_scatter_tensor.default
                )
                for x in snode.ancestors
            )
            and any(
                is_fallback_op(x, torch.ops.fsdp.chunk_cat.default)
                for x in snode.ancestors
            )
        ):
            # Case 2: Handle ReduceScatter

            # Find the "reduce_scatter copy-in + reduce_scatter comm + reduce_scatter wait" block
            collected_node_set = set()
            _find_all_recursive_deps_of_node_up_to_criteria(
                snode,
                collected_node_set,
                criteria_cb=lambda dep: is_fallback_op(
                    name_to_fused_node[dep.name], torch.ops.fsdp.chunk_cat.default
                ),
            )
            # sort nodes by original buffer order
            collected_nodes = sorted(
                list(collected_node_set), key=lambda x: int(x.get_name()[3:])
            )

            # Group "reduce_scatter copy-in + reduce_scatter comm" into one GroupedSchedulerNode
            nodes_to_group = []
            wait_node_idx = None
            for i in range(len(collected_nodes) - 1):
                node = collected_nodes[i]
                nodes_to_group.append(node)
                if isinstance(collected_nodes[i + 1].node, ir._WaitKernel):
                    wait_node_idx = i + 1
                    break
            assert wait_node_idx is not None
            rs_group_node = _create_group_node(nodes_to_group)

            # Group "reduce_scatter wait + related output nodes" into one GroupedSchedulerNode
            wait_group_node = _create_group_node(collected_nodes[wait_node_idx:])

            rs_nodes.append(
                (
                    rs_group_node,
                    wait_group_node,
                )
            )
        else:
            scheduled.add(snode)
            new_order.append(snode)

    # Enforce AllGather ordering: previous AllGather's "wait then copy_out" group node must run before next AllGather's "copy_in then AG" group node
    prev_ag_wait = None
    for ag_group_node, wait_group_node in ag_nodes:
        if prev_ag_wait is not None:
            ag_group_node.add_fake_dep(WeakDep(item(prev_ag_wait.get_buffer_names())))
        prev_ag_wait = wait_group_node

    # Enforce ReduceScatter ordering: previous ReduceScatter's "wait" group node must run before next ReduceScatter's "copy_in then RS" group node
    prev_rs_wait = None
    for rs_group_node, wait_group_node in rs_nodes:
        if prev_rs_wait is not None:
            rs_group_node.add_fake_dep(WeakDep(item(prev_rs_wait.get_buffer_names())))
        prev_rs_wait = wait_group_node

    return new_order
