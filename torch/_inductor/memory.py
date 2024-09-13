from __future__ import annotations

import dataclasses
import heapq
import logging
from typing import Dict, List, Set, Tuple, TYPE_CHECKING, Union

from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from .dependencies import Dep

from .ir import MultiOutputLayout
from .scheduler import (
    BaseSchedulerNode,
    MemoryPlanningInfoForBuffer,
    MemoryPlanningInfoForNode,
    NodeUser,
    OutputNode,
    SchedulerBuffer,
)
from .utils import get_dtype_size
from .virtualized import V


torch_log = logging.getLogger("torch")


@dataclasses.dataclass
class InputBuffer:
    dep: Dep
    mpi: MemoryPlanningInfoForBuffer = dataclasses.field(
        default_factory=MemoryPlanningInfoForBuffer
    )

    def get_name(self) -> str:
        return self.dep.name

    def set_users(self, users: List[NodeUser]) -> None:
        # deduplicate
        result: Dict[int, NodeUser] = {}
        for use in users:
            if id(use.node) in result:
                result[id(use.node)] = use.merge(result[id(use.node)])
            else:
                result[id(use.node)] = use
        self.users = list(result.values())

    def __hash__(self) -> int:
        return hash(self.dep.name)


def dep_size_hint(dep: Dep) -> int:
    res = 0
    try:
        if not dep.has_unbacked_symbols():
            res = dep.numbytes_hint()
    except KeyError:
        # In at least one test (test/inductor/test_torchbind.py) we
        # create a StarDep that doesn't exist in the graph and calling
        # `has_unbacked_symbols()` throws an error.
        pass
    return res


def get_freeable_input_buf(
    nodes: List[BaseSchedulerNode],
    name_to_users: Dict[str, List[NodeUser]],
    graph_inputs: Set[str],
) -> Dict[str, InputBuffer]:
    """
    Create and keep track of all input buffers that can be freed during the program
    """
    name_to_input_buf: Dict[str, InputBuffer] = {}
    for node in nodes:
        for dep in node.read_writes.reads:
            if (
                dep.name in graph_inputs
                and not dep.name.startswith("primals_")
                and dep.name not in name_to_input_buf
            ):
                name_to_input_buf[dep.name] = InputBuffer(dep)
                name_to_input_buf[dep.name].mpi.size_free = dep_size_hint(dep)
    # copy user information onto freeable input buffers
    for input_buf_name, input_buf in name_to_input_buf.items():
        input_buf.set_users(name_to_users[input_buf_name])

    return name_to_input_buf


def compute_size_for_scheduler_buffer(name_to_buf: Dict[str, SchedulerBuffer]) -> None:
    """
    Compute the size of each scheduler buffer, including (1) memory allocated when
    it is created and (2) memory deallocated when it is freed.

    We specially handle the case of MultiOutputLayout.
    Consider the following case:
        buf0 = some_ops_with_multi_outputs(...)
        buf1 = buf0[0] # assume 10 bytes
        buf2 = buf0[1] # assume 20 bytes
    In such cases,
        buf0: at creation, 30 bytes allocated, when deleted, 0 bytes freed
        buf1: at creation, 0 bytes allocated, when deleted, 10 bytes freed
        buf2: at creation, 0 bytes allocated, when deleted, 20 bytes freed
    """
    # compute the size of SchedulerBuffer without MultiOutputLayout layout
    for sched_buf in name_to_buf.values():
        if not isinstance(sched_buf.node.layout, MultiOutputLayout):
            sched_buf.mpi.size_alloc = V.graph.sizevars.size_hint(
                sched_buf.node.get_numel(), fallback=0
            ) * get_dtype_size(sched_buf.node.get_dtype())
            sched_buf.mpi.size_free = sched_buf.mpi.size_alloc

    # compute the size of SchedulerBuffer with MultiOutputLayout layout
    for sched_buf in name_to_buf.values():
        if isinstance(sched_buf.node.layout, MultiOutputLayout):
            for user in sched_buf.users:
                if isinstance(user.node, OutputNode):
                    continue
                assert isinstance(user.node, BaseSchedulerNode)
                for buf in user.node.get_outputs():
                    sched_buf.mpi.size_alloc += buf.mpi.size_alloc
                    buf.mpi.size_alloc = 0


def map_successor_nodes_with_predecessor_buffers(
    nodes: List[BaseSchedulerNode],
    name_to_input_buf: Dict[str, InputBuffer],
    name_to_buf: Dict[str, SchedulerBuffer],
) -> None:
    """
    For scheduling and memory estimation, for each scheduler node, we maintain
    a list of its dependency buffers (SchedulerBuffer and InputBuffer).
    This is similar to node.read_writes.reads, which is a list of Dep.
    Reversely, for each SchedulerBuffer / InputBuffer, assign its successor nodes.
    A buffer's successor nodes determines when a buffer can be freed.
    """
    for node in nodes:
        node.mpi = MemoryPlanningInfoForNode()
        node.mpi.pred_buffers = []
        for dep_name in {dep.name for dep in node.unmet_dependencies}:
            sched_buf = name_to_buf.get(dep_name)
            if sched_buf:
                node.mpi.pred_buffers.append(sched_buf)
                sched_buf.mpi.succ_nodes.add(node)
        for dep_name in {
            dep.name for dep in node.read_writes.reads - node.unmet_dependencies
        }:
            input_buf = name_to_input_buf.get(dep_name)
            if input_buf:
                node.mpi.pred_buffers.append(input_buf)
                input_buf.mpi.succ_nodes.add(node)


def estimate_peak_memory(
    nodes: List[BaseSchedulerNode],
    name_to_input_buf: Dict[str, InputBuffer],
    graph_outputs: Set[str],
) -> Tuple[int, List[int]]:
    """
    Given a list of nodes in their execution order, estimate the peak memory, by
    keeping track of the liveliness of SchedulerBuffers and InputBuffers.

    Returns:
        int: peak memory
        List[int]: memory usage at each node.
    """

    # map each scheduler buffer to its size, start time, and end time
    @dataclasses.dataclass
    class BufferInfo:
        buffer: Union[SchedulerBuffer, InputBuffer]
        size_alloc: int
        size_free: int
        start_time: int
        end_time: int

    name_to_buf_info: Dict[str, BufferInfo] = {}
    node_name_to_time: Dict[str, int] = {}

    # assign start_time
    for buf_name, input_buf in name_to_input_buf.items():
        name_to_buf_info[buf_name] = BufferInfo(
            input_buf,
            input_buf.mpi.size_free,
            input_buf.mpi.size_free,
            0,
            0,
        )
    for t, node in enumerate(nodes):
        node_name_to_time[node.get_name()] = t
        for sched_buf in node.get_outputs():
            name_to_buf_info[sched_buf.get_name()] = BufferInfo(
                sched_buf,
                sched_buf.mpi.size_alloc,
                sched_buf.mpi.size_free,
                t,
                t,
            )

    # assign end_time
    for buf_name, buf_info in name_to_buf_info.items():
        succ_node_time = [
            node_name_to_time[succ_node.get_name()]
            for succ_node in buf_info.buffer.mpi.succ_nodes
            if succ_node.get_name() in node_name_to_time
        ]
        if succ_node_time:
            buf_info.end_time = max(succ_node_time)

    # the end time of output buffers should be at the end of the horizon
    for buf_name in graph_outputs:
        if buf_name in name_to_buf_info:
            name_to_buf_info[buf_name].end_time = len(nodes) - 1

    # incremental memory changes at each time period
    memory = [0 for _ in range(len(nodes) + 1)]

    # for each buffer, update memory when created and when freed
    for buf_name, buf_info in name_to_buf_info.items():
        memory[buf_info.start_time] += buf_info.size_alloc
        memory[buf_info.end_time + 1] -= buf_info.size_free

    # get peak memory by compute the cumulative memories
    max_memory = 0
    cur_memory = 0
    memories_at_nodes = []
    for t in range(len(nodes) + 1):
        cur_memory += memory[t]
        memories_at_nodes.append(cur_memory)
        max_memory = max(max_memory, cur_memory)

    return (max_memory, memories_at_nodes)


def assign_predcessor_and_successor_nodes_to_nodes(
    nodes: List[BaseSchedulerNode], name_to_fused_node: Dict[str, BaseSchedulerNode]
) -> None:
    """
    Assign to each scheduler node its predecessor and successor nodes.
    """
    for node in nodes:
        node.mpi.pred_nodes = list(
            {
                name_to_fused_node[pred_buffer.defining_op.get_name()]
                for pred_buffer in node.mpi.pred_buffers
                if (
                    isinstance(pred_buffer, SchedulerBuffer)
                    and pred_buffer.defining_op.get_name() in name_to_fused_node
                )
            }
        )
        node.mpi.succ_nodes = list(
            {
                succ_node
                for buffer in node.get_outputs()
                for succ_node in buffer.mpi.succ_nodes
            }
        )


def topological_sort_lpmf(
    nodes: List[BaseSchedulerNode],
    name_to_input_buf: Dict[str, InputBuffer],
    name_to_buf: Dict[str, SchedulerBuffer],
    graph_outputs: Set[str],
) -> List[BaseSchedulerNode]:
    """
    A bfs-based greedy topological order.
    The algorithm maintain the max memory so far.
    At every iteration, for each scheduleable node, it computes:
        - how much memory needs to be allocated for the output buffers of this node;
        - how much memory can be freed as a result of executing this node.
    This gives us two values for each node:
        (1) mem1: memory during the execution of the node;
        (2) mem2: memory after executing the node, after some input buffers are freed.
    The greedy approach select as follows:
        (i) if there are nodes whose mem1 values are below the max memory so far,
            then pick the node with the lowest mem2 value;
        (ii) otherwise, pick the one with the lowest mem1 value.
    """

    # compute nodes' number of unmet dependencies (for schedulability)
    # initialize the list of nodes ready to be scheduled
    nodes_to_schedule: Set[BaseSchedulerNode] = set()
    for node in nodes:
        # note that .unmet_dependencies could have deps with the same name
        # and in that case, it should only be counted once
        node.mpi.indegree = len(node.mpi.pred_nodes)
        if node.mpi.indegree == 0:
            nodes_to_schedule.add(node)

    # compute buffers' number of unmet successors (used to decide when to free)
    for buf in list(name_to_buf.values()) + list(name_to_input_buf.values()):
        buf.mpi.outdegree = len(buf.mpi.succ_nodes)
        if buf.get_name() in graph_outputs:
            buf.mpi.outdegree += 1

    # initialize memory estimations
    live_memory = sum(
        input_buf.mpi.size_free for input_buf in name_to_input_buf.values()
    )

    # this is the total output memory, which is a lower bound for peak memory
    output_memory = sum(
        name_to_buf[buf_name].mpi.size_free
        for buf_name in graph_outputs
        if buf_name in name_to_buf
    )
    max_memory = max(live_memory, output_memory)

    # compute the amount of memory that is allocated when a node is scheduled
    # and the amount of memory that can be freed when a node is scheduled
    for i, node in enumerate(nodes):
        node.mpi.index = i  # keep track of the original order
        node.mpi.size = sum(buffer.mpi.size_alloc for buffer in node.get_outputs())
        node.mpi.memory_to_free = 0
        # 1. if a buffer read by this node is last used by this node
        #    then the buffer can be freed
        for buf in node.mpi.pred_buffers:
            if buf.mpi.outdegree == 1:
                node.mpi.memory_to_free += buf.mpi.size_free
        # 2. if a buffer written by this node is used internally and
        #    not needed afterwards, it can be freed
        for buf in node.get_outputs():
            if buf.mpi.outdegree == 0:
                node.mpi.memory_to_free += buf.mpi.size_free

    # schedule nodes one at a time
    schedule: List[BaseSchedulerNode] = []
    while nodes_to_schedule:
        # select a node to schedule:
        selected_node = min(
            nodes_to_schedule,
            key=lambda node: (
                max(live_memory + node.mpi.size, max_memory),
                node.mpi.size - node.mpi.memory_to_free,
                node.mpi.index,
            ),
        )
        nodes_to_schedule.remove(selected_node)
        schedule.append(selected_node)

        # update memory usage
        live_memory += selected_node.mpi.size
        max_memory = max(max_memory, live_memory)
        live_memory -= selected_node.mpi.memory_to_free

        # update successor nodes and nodes_to_schedule
        for succ_node in selected_node.mpi.succ_nodes:
            assert succ_node.mpi.indegree > 0
            succ_node.mpi.indegree -= 1
            if succ_node.mpi.indegree == 0:
                nodes_to_schedule.add(succ_node)

        # update predecessor nodes
        for buf in selected_node.mpi.pred_buffers:
            assert buf.mpi.outdegree > 0
            buf.mpi.outdegree -= 1
            if buf.mpi.outdegree == 1:
                for succ_node in buf.mpi.succ_nodes:
                    succ_node.mpi.memory_to_free += buf.mpi.size_free

    return schedule


def topological_sort_bfs(nodes: List[BaseSchedulerNode]) -> List[BaseSchedulerNode]:
    """
    A BFS topological sort that selects nodes whose depednecies are executed
    the earliest. This follows a FIFO idea.
    """

    @dataclasses.dataclass
    class HeapElement:
        priority: List[int]
        node: BaseSchedulerNode

        def __lt__(self, other: HeapElement) -> bool:
            if self.priority == other.priority:
                return self.node.mpi.index < other.node.mpi.index
            return self.priority > other.priority
            return self.priority < other.priority

    def _node_priority(node: BaseSchedulerNode) -> List[int]:
        assert node.mpi.indegree == 0
        ids = sorted({pred_node.mpi.index for pred_node in node.mpi.pred_nodes})
        ids.append(node.mpi.index)
        return ids

    # compute nodes' number of unmet dependencies (for schedulability)
    # initialize the list of nodes ready to be scheduled
    nodes_to_schedule: List[HeapElement] = []
    for t, node in enumerate(nodes):
        node.mpi.index = t
        # note that .unmet_dependencies could have deps with the same name
        # and in that case, it should only be counted once
        node.mpi.indegree = len(node.mpi.pred_nodes)
        if node.mpi.indegree == 0:
            heapq.heappush(nodes_to_schedule, HeapElement(_node_priority(node), node))

    # schedule nodes one at a time
    schedule: List[BaseSchedulerNode] = []
    while nodes_to_schedule:
        # select a node to schedule
        selected_node = heapq.heappop(nodes_to_schedule).node
        selected_node.mpi.index = len(schedule)
        schedule.append(selected_node)

        # update successor nodes and nodes_to_schedule
        for succ_node in selected_node.mpi.succ_nodes:
            assert succ_node.mpi.indegree > 0
            succ_node.mpi.indegree -= 1
            if succ_node.mpi.indegree == 0:
                heapq.heappush(
                    nodes_to_schedule,
                    HeapElement(_node_priority(succ_node), succ_node),
                )
    return schedule


def topological_sort_dfs(nodes: List[BaseSchedulerNode]) -> List[BaseSchedulerNode]:
    """
    Ensure nodes is in topologically sorted order
    """
    seen: OrderedSet[BaseSchedulerNode] = OrderedSet()
    name_to_node: Dict[str, BaseSchedulerNode] = dict()
    result: List[BaseSchedulerNode] = []

    def visit(n: BaseSchedulerNode) -> None:
        if n not in seen:
            seen.add(n)
            for dep in sorted(n.unmet_dependencies, key=lambda d: d.name):
                # We only care about doing toposort within `nodes`
                if dep.name not in name_to_node:
                    continue
                visit(name_to_node[dep.name])
            result.append(n)

    for node in nodes:
        for name in node.get_buffer_names():
            name_to_node[name] = node

    for t, node in enumerate(nodes):
        node.mpi.index = t
        node.mpi.size = sum(buffer.mpi.size_alloc for buffer in node.get_outputs())
        node.mpi.measure = node.mpi.size + sum(
            pred_buf.mpi.size_free for pred_buf in node.mpi.pred_buffers
        )
    for node in sorted(nodes, key=lambda x: (x.mpi.measure, x.mpi.index)):
        visit(node)
    return result


def reorder_for_peak_memory(
    nodes: List[BaseSchedulerNode],
    name_to_users: Dict[str, List[NodeUser]],
    name_to_buf: Dict[str, SchedulerBuffer],
    name_to_fused_node: Dict[str, BaseSchedulerNode],
    graph_inputs: Set[str],
    graph_outputs: Set[str],
) -> List[BaseSchedulerNode]:
    """
    Try a few heuristics based topological sort algorithms, and pick the one whose
    resulting topological order has the lowest peak memory estimation.
    """

    @dataclasses.dataclass
    class PeakMemoryResult:
        order: List[BaseSchedulerNode]
        peak_memory: int

    # preparation --  as nodes are scheduled one at a time, these help
    # keep track of when a buffer can be freed, and when a node can be scheduled
    name_to_input_buf: Dict[str, InputBuffer] = get_freeable_input_buf(
        nodes, name_to_users, graph_inputs
    )
    compute_size_for_scheduler_buffer(name_to_buf)
    map_successor_nodes_with_predecessor_buffers(nodes, name_to_input_buf, name_to_buf)
    assign_predcessor_and_successor_nodes_to_nodes(nodes, name_to_fused_node)

    # keep track of the peak memory estimates of different methods
    peak_memory_diff_methods: List[PeakMemoryResult] = []

    # the default
    estimated_peak_memory, _ = estimate_peak_memory(
        nodes, name_to_input_buf, graph_outputs
    )
    peak_memory_diff_methods.append(PeakMemoryResult(nodes, estimated_peak_memory))

    # lpmf based method
    order_lpmf = topological_sort_lpmf(
        nodes, name_to_input_buf, name_to_buf, graph_outputs
    )
    assert len(order_lpmf) == len(nodes)
    peak_memory_lpmf, _ = estimate_peak_memory(
        order_lpmf, name_to_input_buf, graph_outputs
    )
    peak_memory_diff_methods.append(PeakMemoryResult(order_lpmf, peak_memory_lpmf))

    # bfs based method
    order_bfs = topological_sort_bfs(nodes)
    assert len(order_bfs) == len(nodes)
    peak_memory_bfs, _ = estimate_peak_memory(
        order_bfs, name_to_input_buf, graph_outputs
    )
    peak_memory_diff_methods.append(PeakMemoryResult(order_bfs, peak_memory_bfs))

    # dfs based method
    order_dfs = topological_sort_dfs(nodes)
    assert len(order_dfs) == len(nodes)
    peak_memory_dfs, _ = estimate_peak_memory(
        order_dfs, name_to_input_buf, graph_outputs
    )
    peak_memory_diff_methods.append(PeakMemoryResult(order_dfs, peak_memory_dfs))

    # get the optimal one
    best_result = min(peak_memory_diff_methods, key=lambda x: x.peak_memory)
    return best_result.order
