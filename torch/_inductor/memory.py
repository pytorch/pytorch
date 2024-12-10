from __future__ import annotations

import collections
import dataclasses
import heapq
import logging
from typing import Callable, Dict, List, Tuple, TYPE_CHECKING, TypedDict, Union

from torch._utils_internal import signpost_event
from torch.utils._ordered_set import OrderedSet

from .ir import MultiOutputLayout, NoneLayout
from .utils import get_dtype_size
from .virtualized import V


if TYPE_CHECKING:
    from .dependencies import Dep
    from .scheduler import BaseSchedulerNode, SchedulerBuffer


torch_log = logging.getLogger(__name__)


@dataclasses.dataclass
class MemoryPlanningInfoForBuffer:
    size_alloc: int = 0
    size_free: int = 0
    succ_nodes: OrderedSet[BaseSchedulerNode] = dataclasses.field(
        default_factory=OrderedSet
    )


@dataclasses.dataclass
class MemoryPlanningInfoForNode:
    index: int = 0
    size: int = 0
    pred_buffers: OrderedSet[
        Union[SchedulerBuffer, FreeableInputBuffer]
    ] = dataclasses.field(default_factory=OrderedSet)
    pred_nodes: OrderedSet[BaseSchedulerNode] = dataclasses.field(
        default_factory=OrderedSet
    )
    succ_nodes: OrderedSet[BaseSchedulerNode] = dataclasses.field(
        default_factory=OrderedSet
    )


@dataclasses.dataclass
class FreeableInputBuffer:
    name: str
    mpi_buffer: MemoryPlanningInfoForBuffer = dataclasses.field(
        default_factory=MemoryPlanningInfoForBuffer
    )

    def get_name(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)


def get_freeable_input_buf(
    nodes: List[BaseSchedulerNode],
    graph_inputs: OrderedSet[str],
) -> Dict[str, FreeableInputBuffer]:
    """
    Create and keep track of all input buffers that can be freed during the program

    Returns:
        A dictionary containing all freeble input buffers, keyed by their names.
    """

    # this function is copied from torch/_inductor/scheduler.py
    # TODO: would be nice to remove the try/except block for both places
    def _dep_size_hint(dep: Dep) -> int:
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

    # get freeable input buffers' successor nodes and their sizes
    # note that different deps can have the same name, so we use name as keys
    dep_name_to_succ_nodes: Dict[
        str, OrderedSet[BaseSchedulerNode]
    ] = collections.defaultdict(OrderedSet)
    dep_name_to_size: Dict[str, int] = dict()
    for node in nodes:
        for dep in node.read_writes.reads:
            if dep.name in graph_inputs and not dep.name.startswith(
                ("primals_", "arg")
            ):
                dep_name_to_succ_nodes[dep.name].add(node)
                dep_name_to_size[dep.name] = _dep_size_hint(dep)

    # create FreeableInputBuffer objects and add them to the returned dictionary
    name_to_freeable_input_buf: Dict[str, FreeableInputBuffer] = dict()
    for dep_name, succ_nodes in dep_name_to_succ_nodes.items():
        name_to_freeable_input_buf[dep_name] = FreeableInputBuffer(
            dep_name,
            MemoryPlanningInfoForBuffer(
                size_free=dep_name_to_size[dep_name], succ_nodes=succ_nodes
            ),
        )
    return name_to_freeable_input_buf


def compute_size_for_scheduler_buffer(
    name_to_buf: Dict[str, SchedulerBuffer]
) -> Dict[str, Tuple[int, int]]:
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

    Returns:
        A dictionary mapping a scheduler buffer to a tuple of (size_alloc, size_free).
    """
    from .ir import MultiOutput
    from .scheduler import OutputNode

    sched_buf_to_size: Dict[str, Tuple[int, int]] = dict()

    def _compute_and_update_buf_size(
        sched_buf: SchedulerBuffer, user_of_MultiOutputLayout: bool = False
    ) -> int:
        if isinstance(sched_buf.node.layout, NoneLayout):
            sched_buf_to_size[sched_buf.get_name()] = (0, 0)
            return 0
        elif isinstance(sched_buf.node.layout, MultiOutputLayout):
            size_alloc = 0
            for user in sched_buf.users:
                if isinstance(user.node, OutputNode):
                    continue
                for buf in user.node.get_outputs():
                    if isinstance(buf.node, MultiOutput):
                        size_alloc += _compute_and_update_buf_size(buf, True)
            sched_buf_to_size[sched_buf.get_name()] = (
                0 if user_of_MultiOutputLayout else size_alloc,
                0,
            )
            return size_alloc
        else:
            buf_size = V.graph.sizevars.size_hint(
                sched_buf.node.get_numel(), fallback=0
            ) * get_dtype_size(sched_buf.node.get_dtype())
            sched_buf_to_size[sched_buf.get_name()] = (
                0 if user_of_MultiOutputLayout else buf_size,
                buf_size,
            )
            return buf_size

    for sched_buf in name_to_buf.values():
        # skip if sched_buf is already processed as an user of another SchedulerBuffer
        # whose layout is of the type MultiOutputLayout
        if sched_buf.get_name() not in sched_buf_to_size:
            _compute_and_update_buf_size(sched_buf)

    return sched_buf_to_size


def assign_memory_planning_info_for_scheduler_buffers(
    nodes: List[BaseSchedulerNode],
    name_to_buf: Dict[str, SchedulerBuffer],
) -> None:
    """
    For each SchedulerBuffer, assign its size info and successor nodes.
    A buffer's successor nodes determines when a buffer can be freed.
    """
    # get buffer sizes
    sched_buf_to_size = compute_size_for_scheduler_buffer(name_to_buf)

    # get buffer's successor nodes
    # note that different deps can have the same name, so we use name as keys
    dep_name_to_succ_nodes: Dict[
        str, OrderedSet[BaseSchedulerNode]
    ] = collections.defaultdict(OrderedSet)
    for node in nodes:
        for dep in node.unmet_dependencies:
            dep_name_to_succ_nodes[dep.name].add(node)

    # populate the MemoryPlanningInfoForBuffer attribute to each scheduler buffer
    # note: there are scheduler buffers not in dep_name_to_succ_nodes (e.g., graph outputs)
    for buf_name in name_to_buf.keys():
        name_to_buf[buf_name].mpi_buffer = MemoryPlanningInfoForBuffer(
            size_alloc=sched_buf_to_size[buf_name][0],
            size_free=sched_buf_to_size[buf_name][1],
            succ_nodes=dep_name_to_succ_nodes[buf_name],
        )


def assign_memory_planning_info_for_scheduler_nodes(
    nodes: List[BaseSchedulerNode],
    name_to_fused_node: Dict[str, BaseSchedulerNode],
    name_to_buf: Dict[str, SchedulerBuffer],
    name_to_freeable_input_buf: Dict[str, FreeableInputBuffer],
) -> None:
    """
    Assign to each scheduler node its predecessor and successor nodes.
    """
    from .scheduler import SchedulerBuffer

    for index, node in enumerate(nodes):
        size_alloc = sum(buffer.mpi_buffer.size_alloc for buffer in node.get_outputs())
        pred_buffers = OrderedSet[Union[SchedulerBuffer, FreeableInputBuffer]]()
        for dep in node.read_writes.reads:
            if dep.name in name_to_buf and dep in node.unmet_dependencies:
                pred_buffers.add(name_to_buf[dep.name])
            elif dep.name in name_to_freeable_input_buf:
                pred_buffers.add(name_to_freeable_input_buf[dep.name])
        pred_nodes = OrderedSet(
            name_to_fused_node[pred_buffer.defining_op.get_name()]
            for pred_buffer in pred_buffers
            if (isinstance(pred_buffer, SchedulerBuffer))
        )
        succ_nodes = OrderedSet(
            succ_node
            for buffer in node.get_outputs()
            for succ_node in buffer.mpi_buffer.succ_nodes
        )
        node.mpi_node = MemoryPlanningInfoForNode(
            index=index,
            size=size_alloc,
            pred_buffers=pred_buffers,
            pred_nodes=pred_nodes,
            succ_nodes=succ_nodes,
        )


def estimate_peak_memory(
    nodes: List[BaseSchedulerNode],
    name_to_freeable_input_buf: Dict[str, FreeableInputBuffer],
    graph_outputs: OrderedSet[str],
) -> Tuple[int, List[int]]:
    """
    Given a list of nodes in their execution order, estimate the peak memory, by
    keeping track of the liveliness of SchedulerBuffers and FreeableInputBuffers.

    Returns:
        int: peak memory
        List[int]: memory usage at each node (or each step).
    """

    # map each scheduler buffer to its size, start step, and end step
    @dataclasses.dataclass
    class BufferInfo:
        buffer: Union[SchedulerBuffer, FreeableInputBuffer]
        size_alloc: int
        size_free: int
        start_step: int
        end_step: int

    # get the execution step of each node, this will be used to determine
    # the end_step of buffers
    node_to_step: Dict[BaseSchedulerNode, int] = dict()
    for step, node in enumerate(nodes):
        node_to_step[node] = step

    # get buffers' size and liveliness information
    buf_info_list: List[BufferInfo] = []
    # 1. for freeable input buffers
    for buf_name, input_buf in name_to_freeable_input_buf.items():
        end_step = (
            len(nodes) - 1
            if buf_name in graph_outputs
            else max(
                node_to_step[succ_node] for succ_node in input_buf.mpi_buffer.succ_nodes
            )
        )
        buf_info_list.append(
            BufferInfo(
                input_buf,
                input_buf.mpi_buffer.size_free,
                input_buf.mpi_buffer.size_free,
                0,
                end_step,
            )
        )

    # 2. for scheduler buffers
    for step, node in enumerate(nodes):
        for sched_buf in node.get_outputs():
            # note: it is possible for a non-graph-output sched_buf to have no succ_nodes and
            # to be only used by its defining op (e.g., due to fusion when all consumers of
            # the buffer are fused with its defining op). In such cases, end_step is step.
            end_step = (
                len(nodes) - 1
                if sched_buf.get_name() in graph_outputs
                else max(
                    [
                        node_to_step[succ_node]
                        for succ_node in sched_buf.mpi_buffer.succ_nodes
                    ],
                    default=step,
                )
            )
            buf_info_list.append(
                BufferInfo(
                    sched_buf,
                    sched_buf.mpi_buffer.size_alloc,
                    sched_buf.mpi_buffer.size_free,
                    step,
                    end_step,
                )
            )

    # incremental memory changes at each step
    memory = [0 for _ in range(len(nodes) + 1)]

    # for each buffer, update memory when created and when freed
    for buf_info in buf_info_list:
        memory[buf_info.start_step] += buf_info.size_alloc
        memory[buf_info.end_step + 1] -= buf_info.size_free

    # get peak memory by compute the cumulative memories
    max_memory = 0
    cur_memory = 0
    memories_at_nodes = []
    for t in range(len(nodes) + 1):
        cur_memory += memory[t]
        memories_at_nodes.append(cur_memory)
        max_memory = max(max_memory, cur_memory)

    return (max_memory, memories_at_nodes)


def topological_sort_lpmf(
    nodes: List[BaseSchedulerNode],
    name_to_freeable_input_buf: Dict[str, FreeableInputBuffer],
    name_to_buf: Dict[str, SchedulerBuffer],
    graph_outputs: OrderedSet[str],
) -> List[BaseSchedulerNode]:
    """
    A bfs-based greedy topological order. LPMF stands for "Least Peak Memory First".

    The idea is from this paper:
    Buffer memory optimization for video codec application modeled in Simulink
    https://www.cs.york.ac.uk/rts/docs/DAC-1964-2006/PAPERS/2006/DAC06/PDFFILES/P0689.PDF

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

    class NodeInfo(TypedDict):
        indegree: int
        memory_to_free: int

    class BufferInfo(TypedDict):
        outdegree: int

    node_info: Dict[BaseSchedulerNode, NodeInfo] = dict()
    buf_info: Dict[Union[SchedulerBuffer, FreeableInputBuffer], BufferInfo] = dict()

    # compute nodes' number of unmet dependencies (for schedulability)
    # initialize the list of nodes ready to be scheduled
    nodes_to_schedule: OrderedSet[BaseSchedulerNode] = OrderedSet()
    for node in nodes:
        node_info[node] = {
            "indegree": len(node.mpi_node.pred_nodes),
            "memory_to_free": 0,
        }
        if node_info[node]["indegree"] == 0:
            nodes_to_schedule.add(node)

    # compute buffers' number of unmet successors (used to decide when to free)
    for buf in list(name_to_buf.values()) + list(name_to_freeable_input_buf.values()):
        buf_info[buf] = {
            "outdegree": len(buf.mpi_buffer.succ_nodes)
            + (1 if buf.get_name() in graph_outputs else 0)
        }

    # initialize memory estimations
    live_memory = sum(
        input_buf.mpi_buffer.size_free
        for input_buf in name_to_freeable_input_buf.values()
    )

    # this is the total output memory, which is a lower bound for peak memory
    # we do not include the memory of non freeable input buffers
    output_memory = 0
    for buf_name in graph_outputs:
        if buf_name in name_to_buf:
            output_memory += name_to_buf[buf_name].mpi_buffer.size_free
        elif buf_name in name_to_freeable_input_buf:
            output_memory += name_to_freeable_input_buf[buf_name].mpi_buffer.size_free
    max_memory = max(live_memory, output_memory)

    # compute the amount of memory that is allocated when a node is scheduled
    # and the amount of memory that can be freed when a node is scheduled
    for i, node in enumerate(nodes):
        # 1. if a buffer read by this node is last used by this node
        for buf in node.mpi_node.pred_buffers:
            if buf_info[buf]["outdegree"] == 1:
                node_info[node]["memory_to_free"] += buf.mpi_buffer.size_free
        # 2. if a buffer written by this node is used internally and not used later
        for buf in node.get_outputs():
            if buf_info[buf]["outdegree"] == 0:
                node_info[node]["memory_to_free"] += buf.mpi_buffer.size_free

    # schedule nodes one at a time
    schedule: List[BaseSchedulerNode] = []
    num_iters: int = 0
    while num_iters < len(nodes) and nodes_to_schedule:
        # select a node to schedule:
        selected_node = min(
            nodes_to_schedule,
            key=lambda node: (
                max(live_memory + node.mpi_node.size, max_memory),
                node.mpi_node.size - node_info[node]["memory_to_free"],
                node.mpi_node.index,
            ),
        )
        nodes_to_schedule.remove(selected_node)
        schedule.append(selected_node)
        num_iters += 1

        # update memory usage
        live_memory += selected_node.mpi_node.size
        max_memory = max(max_memory, live_memory)
        live_memory -= node_info[selected_node]["memory_to_free"]

        # update successor nodes and nodes_to_schedule
        for succ_node in selected_node.mpi_node.succ_nodes:
            assert node_info[succ_node]["indegree"] > 0
            node_info[succ_node]["indegree"] -= 1
            if node_info[succ_node]["indegree"] == 0:
                nodes_to_schedule.add(succ_node)

        # update predecessor nodes
        for buf in selected_node.mpi_node.pred_buffers:
            assert buf_info[buf]["outdegree"] > 0
            buf_info[buf]["outdegree"] -= 1
            if buf_info[buf]["outdegree"] == 1:
                for succ_node in buf.mpi_buffer.succ_nodes:
                    node_info[succ_node]["memory_to_free"] += buf.mpi_buffer.size_free

    if num_iters > len(nodes):
        raise RuntimeError("Failed to schedule, while loop ran too long for lpmf")

    return schedule


def topological_sort_bfs(nodes: List[BaseSchedulerNode]) -> List[BaseSchedulerNode]:
    """
    A BFS topological sort that selects nodes whose dependencies are executed the
    earliest. This follows a FIFO idea. Specifically, at every iteration, for each node
    that is schedulable, we gather the order in which its predecessor nodes are executed,
    and this sorted list of execution orders of predecessor nodes defines the priority.
    We select the node whose predecessors nodes are executed the earliest. The FIFO
    idea aims to reduce the liveness duration of buffers created.
    """

    class NodeInfo(TypedDict):
        indegree: int
        order: int

    node_info: Dict[BaseSchedulerNode, NodeInfo] = dict()

    @dataclasses.dataclass
    class NodeWithPriority:
        priority: List[int]
        node: BaseSchedulerNode

        def __lt__(self, other: NodeWithPriority) -> bool:
            if self.priority == other.priority:
                return self.node.mpi_node.index < other.node.mpi_node.index
            return self.priority < other.priority

    def _node_priority(node: BaseSchedulerNode) -> List[int]:
        # priority is the order in which predecessor nodes are executed
        assert node_info[node]["indegree"] == 0
        exec_orders = sorted(
            OrderedSet(
                node_info[pred_node]["order"] for pred_node in node.mpi_node.pred_nodes
            )
        )
        return exec_orders

    # compute nodes' number of unmet dependencies (for schedulability)
    # initialize the list of nodes ready to be scheduled
    nodes_to_schedule: List[NodeWithPriority] = []
    for node in nodes:
        node_info[node] = {"indegree": len(node.mpi_node.pred_nodes), "order": -1}
        if node_info[node]["indegree"] == 0:
            heapq.heappush(
                nodes_to_schedule, NodeWithPriority(_node_priority(node), node)
            )

    # schedule nodes one at a time
    schedule: List[BaseSchedulerNode] = []
    num_iters: int = 0
    while num_iters < len(nodes) and nodes_to_schedule:
        # select a node to schedule
        selected_node = heapq.heappop(nodes_to_schedule).node
        node_info[selected_node]["order"] = len(schedule)
        schedule.append(selected_node)
        num_iters += 1

        # update successor nodes and nodes_to_schedule
        for succ_node in selected_node.mpi_node.succ_nodes:
            assert node_info[succ_node]["indegree"] > 0
            node_info[succ_node]["indegree"] -= 1
            if node_info[succ_node]["indegree"] == 0:
                heapq.heappush(
                    nodes_to_schedule,
                    NodeWithPriority(_node_priority(succ_node), succ_node),
                )

    if num_iters > len(nodes):
        raise RuntimeError("Failed to schedule, while loop ran too long for bfs")

    return schedule


def topological_sort_dfs(nodes: List[BaseSchedulerNode]) -> List[BaseSchedulerNode]:
    """
    This is a DFS topological sort. The setup is similar to `topological_sort_schedule`
    in scheduler.py. The difference is the order nodes are visited in the outer loop.
    In `topological_sort_schedule`, nodes are visited in their original order.
    In this function, nodes are visited based on their priority -- for each node, we
    compute the total memory of all buffers it reads from or writes to, and we visit
    the nodes in ascending order of this priority.
    """
    seen: OrderedSet[BaseSchedulerNode] = OrderedSet()
    name_to_node: Dict[str, BaseSchedulerNode] = dict()
    result: List[BaseSchedulerNode] = []
    size_with_reads: Dict[BaseSchedulerNode, int] = dict()

    def visit(n: BaseSchedulerNode) -> None:
        if n not in seen:
            seen.add(n)
            dep_nodes = [
                name_to_node[dep.name]
                for dep in n.unmet_dependencies
                if dep.name in name_to_node
            ]
            for node in sorted(
                dep_nodes, key=lambda n: (size_with_reads[n], n.mpi_node.index)
            ):
                visit(node)
            result.append(n)

    for node in nodes:
        for name in node.get_buffer_names():
            name_to_node[name] = node

    for node in nodes:
        size_with_reads[node] = node.mpi_node.size + sum(
            pred_buf.mpi_buffer.size_free for pred_buf in node.mpi_node.pred_buffers
        )
    for node in sorted(nodes, key=lambda n: (size_with_reads[n], n.mpi_node.index)):
        visit(node)

    return result


def reorder_for_peak_memory(
    nodes: List[BaseSchedulerNode],
    name_to_buf: Dict[str, SchedulerBuffer],
    name_to_fused_node: Dict[str, BaseSchedulerNode],
    graph_inputs: OrderedSet[str],
    graph_outputs: OrderedSet[str],
    methods: List[Callable[..., List[BaseSchedulerNode]]] = [  # noqa: B006
        topological_sort_lpmf,
        topological_sort_bfs,
        topological_sort_dfs,
    ],
) -> List[BaseSchedulerNode]:
    """
    Try a few heuristics based topological sort algorithms, and pick the one whose
    resulting topological order has the lowest peak memory estimation.
    """

    torch_log.info("Reordering for peak memory -- %d nodes", len(nodes))

    @dataclasses.dataclass
    class PeakMemoryResult:
        order: List[BaseSchedulerNode]
        peak_memory: int
        method: str

    # preparation --  as nodes are scheduled one at a time, these help
    # keep track of when a buffer can be freed, and when a node can be scheduled
    name_to_freeable_input_buf: Dict[str, FreeableInputBuffer] = get_freeable_input_buf(
        nodes, graph_inputs
    )
    assign_memory_planning_info_for_scheduler_buffers(nodes, name_to_buf)
    assign_memory_planning_info_for_scheduler_nodes(
        nodes, name_to_fused_node, name_to_buf, name_to_freeable_input_buf
    )

    # keep track of the peak memory estimates of different methods
    peak_memory_diff_methods: List[PeakMemoryResult] = []

    # the default
    estimated_peak_memory, _ = estimate_peak_memory(
        nodes, name_to_freeable_input_buf, graph_outputs
    )
    peak_memory_diff_methods.append(
        PeakMemoryResult(nodes, estimated_peak_memory, "baseline")
    )
    torch_log.info("Baseline peak memory: %d", estimated_peak_memory)

    # other methods
    for method in methods:
        try:
            if method == topological_sort_lpmf:
                order = method(
                    nodes, name_to_freeable_input_buf, name_to_buf, graph_outputs
                )
            else:
                order = method(nodes)
            assert len(order) == len(nodes)
            peak_memory, _ = estimate_peak_memory(
                order, name_to_freeable_input_buf, graph_outputs
            )
            peak_memory_diff_methods.append(
                PeakMemoryResult(order, peak_memory, method.__name__)
            )
            torch_log.info("%s peak memory: %d", method.__name__, peak_memory)
        except Exception as e:
            torch_log.error("Failed to reorder for %s: %s", method.__name__, e)

    signpost_event(
        category="inductor",
        name="memory",
        parameters={
            "orm": {elem.method: elem.peak_memory for elem in peak_memory_diff_methods},
        },
    )

    # get the optimal one
    best_result = min(peak_memory_diff_methods, key=lambda x: x.peak_memory)

    return best_result.order
