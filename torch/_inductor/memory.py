from __future__ import annotations

import collections
import dataclasses
import heapq
import logging
from typing import Optional, TYPE_CHECKING, TypedDict, Union

from torch._environment import is_fbcode
from torch._utils_internal import signpost_event
from torch.utils._ordered_set import OrderedSet

from . import config
from .ir import MultiOutputLayout, NoneLayout
from .utils import get_dtype_size, is_nonfreeable_buffers
from .virtualized import V


if TYPE_CHECKING:
    from collections.abc import Callable

    from .dependencies import Dep
    from .scheduler import BaseSchedulerNode, SchedulerBuffer


torch_log = logging.getLogger(__name__)


@dataclasses.dataclass
class PeakMemoryResult:
    order: list[BaseSchedulerNode]
    peak_memory: int
    method: str


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
    pred_buffers: OrderedSet[Union[SchedulerBuffer, FreeableInputBuffer]] = (
        dataclasses.field(default_factory=OrderedSet)
    )
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
    nodes: list[BaseSchedulerNode],
    graph_inputs: OrderedSet[str],
) -> dict[str, FreeableInputBuffer]:
    """
    Create and keep track of all input buffers that can be freed during the program

    Returns:
        A dictionary containing all freeable input buffers, keyed by their names.
    """

    def _dep_size_hint(dep: Dep) -> int:
        return V.graph.get_dep_size_hint(dep)

    # get freeable input buffers' successor nodes and their sizes
    # note that different deps can have the same name, so we use name as keys
    dep_name_to_succ_nodes: dict[str, OrderedSet[BaseSchedulerNode]] = (
        collections.defaultdict(OrderedSet)
    )
    dep_name_to_size: dict[str, int] = dict()

    for node in nodes:
        for dep in node.read_writes.reads:
            if dep.name in graph_inputs:
                if not is_nonfreeable_buffers(dep):
                    dep_name_to_succ_nodes[dep.name].add(node)
                    dep_name_to_size[dep.name] = _dep_size_hint(dep)

    # create FreeableInputBuffer objects and add them to the returned dictionary
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer] = dict()
    for dep_name, succ_nodes in dep_name_to_succ_nodes.items():
        name_to_freeable_input_buf[dep_name] = FreeableInputBuffer(
            dep_name,
            MemoryPlanningInfoForBuffer(
                size_free=dep_name_to_size[dep_name], succ_nodes=succ_nodes
            ),
        )
    return name_to_freeable_input_buf


def compute_size_for_scheduler_buffer(
    name_to_buf: dict[str, SchedulerBuffer],
) -> dict[str, tuple[int, int]]:
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

    When an operation mutates a buffer in-place, the scheduler creates a new buffer name
    to track the "before" and "after" states, even though they share the same memory.

    The mutated buffer represents a rename with zero allocation and deallocation cost.
    During dependency tracking, we transfer dependencies from the mutated name back to
    the original buffer, ensuring the original memory is only freed when all aliases
    are done.

    This handles cases where a buffer has multiple non-overlapping aliases - rather than
    trying to assign free costs to individual aliases, we forward all alias dependencies
    to the original buffer.

    Consider:
        buf0 = op0()
        buf1 = mutation_op_(buf0)
        del buf0
        ...
        op(buf1)
        del buf1

    The only memory events are the creation prior to op0, and the deletion following buf1.

    Returns:
        A dictionary mapping a scheduler buffer to a tuple of (size_alloc, size_free).
    """
    from .ir import MultiOutput
    from .scheduler import OutputNode

    sched_buf_to_size: dict[str, tuple[int, int]] = dict()

    def _compute_and_update_buf_size(
        sched_buf: SchedulerBuffer, user_of_MultiOutputLayout: bool = False
    ) -> int:
        if sched_buf.get_name() in V.graph.scheduler.mutation_real_name:
            sched_buf_to_size[sched_buf.get_name()] = (0, 0)
            return 0
        elif isinstance(sched_buf.node.layout, NoneLayout):
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
    nodes: list[BaseSchedulerNode],
    name_to_buf: dict[str, SchedulerBuffer],
) -> None:
    """
    For each SchedulerBuffer, assign its size info and successor nodes.
    A buffer's successor nodes determines when a buffer can be freed.
    """
    # get buffer sizes
    sched_buf_to_size = compute_size_for_scheduler_buffer(name_to_buf)

    # get buffer's successor nodes
    # note that different deps can have the same name, so we use name as keys
    dep_name_to_succ_nodes: dict[str, OrderedSet[BaseSchedulerNode]] = (
        collections.defaultdict(OrderedSet)
    )
    for node in nodes:
        for dep in node.unmet_dependencies:
            dep_name_to_succ_nodes[dep.name].add(node)

    # iterate in reverse, so dependencies are picked up transitively.
    for mutating_buf_name, real_buf_name in reversed(
        V.graph.scheduler.mutation_real_name.items()
    ):
        dep_name_to_succ_nodes[real_buf_name] |= dep_name_to_succ_nodes[
            mutating_buf_name
        ]

    # populate the MemoryPlanningInfoForBuffer attribute to each scheduler buffer
    # note: there are scheduler buffers not in dep_name_to_succ_nodes (e.g., graph outputs)
    for buf_name in name_to_buf:
        name_to_buf[buf_name].mpi_buffer = MemoryPlanningInfoForBuffer(
            size_alloc=sched_buf_to_size[buf_name][0],
            size_free=sched_buf_to_size[buf_name][1],
            succ_nodes=dep_name_to_succ_nodes[buf_name],
        )


def assign_memory_planning_info_for_scheduler_nodes(
    nodes: list[BaseSchedulerNode],
    name_to_fused_node: dict[str, BaseSchedulerNode],
    name_to_buf: dict[str, SchedulerBuffer],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
) -> None:
    """
    Assign to each scheduler node its predecessor and successor nodes.
    """

    node_to_pred_nodes: dict[BaseSchedulerNode, OrderedSet[BaseSchedulerNode]] = (
        collections.defaultdict(OrderedSet)
    )
    node_to_succ_nodes: dict[BaseSchedulerNode, OrderedSet[BaseSchedulerNode]] = {}
    node_to_pred_buffers: dict[
        BaseSchedulerNode, OrderedSet[SchedulerBuffer | FreeableInputBuffer]
    ] = collections.defaultdict(OrderedSet)

    # collect all predecessors using existing successor mappings
    for node in nodes:
        succ_nodes = OrderedSet(
            succ_node
            for buffer in node.get_outputs()
            for succ_node in buffer.mpi_buffer.succ_nodes
        )
        node_to_succ_nodes[node] = succ_nodes

        # For each successor, add current node as its predecessor
        for succ_node in succ_nodes:
            node_to_pred_nodes[succ_node].add(node)

        # For each output buffer, add it as predecessor to its successor nodes
        # TODO - is pred buffers needed ?
        for buffer in node.get_outputs():
            for succ_node in buffer.mpi_buffer.succ_nodes:
                node_to_pred_buffers[succ_node].add(buffer)

    for freeable_buffer in name_to_freeable_input_buf.values():
        for succ_node in freeable_buffer.mpi_buffer.succ_nodes:
            node_to_pred_buffers[succ_node].add(freeable_buffer)

    # Second pass: assign memory planning info using completed predecessor mappings
    for index, node in enumerate(nodes):
        size_alloc = sum(buffer.mpi_buffer.size_alloc for buffer in node.get_outputs())
        succ_nodes = node_to_succ_nodes[node]
        pred_nodes = node_to_pred_nodes[node]

        # make sure we do not make node a successor or predecessor of itself
        succ_nodes.discard(node)
        pred_nodes.discard(node)

        node.mpi_node = MemoryPlanningInfoForNode(
            index=index,
            size=size_alloc,
            pred_buffers=node_to_pred_buffers[node],
            pred_nodes=node_to_pred_nodes[node],
            succ_nodes=succ_nodes,
        )


# map each scheduler buffer to its size, start step, and end step
@dataclasses.dataclass
class BufferInfo:
    buffer: Union[SchedulerBuffer, FreeableInputBuffer]
    size_alloc: int
    size_free: int
    start_step: int
    end_step: int


def compute_memory_timeline(
    nodes: list[BaseSchedulerNode],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
    graph_outputs: OrderedSet[str],
) -> tuple[
    list[BufferInfo],
    dict[BaseSchedulerNode, int],
    dict[Union[FreeableInputBuffer, SchedulerBuffer], BaseSchedulerNode],
]:
    """
    Compute buffer allocation and deallocation sizes and map their
    lifetime to the node schedule
    """

    # get the execution step of each node, this will be used to determine
    # the end_step of buffers
    node_to_step: dict[BaseSchedulerNode, int] = {
        node: step for step, node in enumerate(nodes)
    }

    # get buffers' size and liveliness information
    buf_info_list: list[BufferInfo] = []
    buf_to_snode_last_use: dict[
        Union[FreeableInputBuffer, SchedulerBuffer], BaseSchedulerNode
    ] = {}

    def _get_end_step_and_snode(
        buf: Union[FreeableInputBuffer, SchedulerBuffer],
    ) -> tuple[int, Optional[BaseSchedulerNode]]:
        max_step: int = -1
        max_step_snode: Optional[BaseSchedulerNode] = None
        succ_nodes = buf.mpi_buffer.succ_nodes
        if succ_nodes:
            for succ_node in succ_nodes:
                step = node_to_step[succ_node]
                if step > max_step:
                    max_step = step
                    max_step_snode = succ_node
            assert max_step_snode is not None
        return max_step, max_step_snode

    # 1. for freeable input buffers
    for buf_name, input_buf in name_to_freeable_input_buf.items():
        end_step = -1
        if buf_name not in graph_outputs:
            end_step, end_step_snode = _get_end_step_and_snode(input_buf)
            assert end_step_snode is not None
            buf_to_snode_last_use[input_buf] = end_step_snode

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
            buf_name = sched_buf.get_name()
            end_step = -1
            if buf_name not in graph_outputs:
                end_step, end_step_snode = _get_end_step_and_snode(sched_buf)
                if end_step == -1:
                    end_step = step
                    buf_to_snode_last_use[sched_buf] = node
                else:
                    assert end_step_snode is not None
                    buf_to_snode_last_use[sched_buf] = end_step_snode

            buf_info_list.append(
                BufferInfo(
                    sched_buf,
                    sched_buf.mpi_buffer.size_alloc,
                    sched_buf.mpi_buffer.size_free,
                    step,
                    end_step,
                )
            )

    return buf_info_list, node_to_step, buf_to_snode_last_use


def estimate_peak_memory(
    nodes: list[BaseSchedulerNode],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
    graph_outputs: OrderedSet[str],
) -> tuple[int, list[int]]:
    """
    Given a list of nodes in their execution order, estimate the peak memory, by
    keeping track of the liveliness of SchedulerBuffers and FreeableInputBuffers.

    Returns:
        int: peak memory
        List[int]: memory usage at each node (or each step).
    """

    buf_info_list, _, _ = compute_memory_timeline(
        nodes, name_to_freeable_input_buf, graph_outputs
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


@dataclasses.dataclass
class SNodeMemory:
    size_alloc: int
    size_free: int


def estimate_peak_memory_allocfree(
    nodes: list[BaseSchedulerNode],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
    graph_outputs: OrderedSet[str],
) -> tuple[
    int,
    list[tuple[int, int]],
    dict[BaseSchedulerNode, SNodeMemory],
    dict[Union[FreeableInputBuffer, SchedulerBuffer], BaseSchedulerNode],
]:
    """
    Alternative version of estimate_peak_memory, that respects the fact,
    that every SchedulerNode has multiple phases:
    1. alloc ( outputs )
    2. run_kernel
    3. dealloc last_use buffers
    estimate_peak_memory collapses memory into one value: size_alloc - size_free
    While peak memory happens after alloc.

    Duplicating the code to not migrate all callsites at once,
    In future usages of estimate_peak_memory will migrate to this version.
    """

    buf_info_list, _, buf_to_snode_last_use = compute_memory_timeline(
        nodes, name_to_freeable_input_buf, graph_outputs
    )

    # incremental memory changes at each step
    step_idx_allocfree = [SNodeMemory(0, 0) for _ in range(len(nodes))]

    # for each buffer, update memory when created and when freed
    for buf_info in buf_info_list:
        step_idx_allocfree[buf_info.start_step].size_alloc += buf_info.size_alloc
        if buf_info.end_step != -1:
            step_idx_allocfree[buf_info.end_step].size_free += buf_info.size_free

    snodes_allocfree = {}
    for i, node in enumerate(nodes):
        snodes_allocfree[node] = step_idx_allocfree[i]

    max_memory = 0
    cur_memory = 0
    snodes_curr_memory = []
    for t in range(len(nodes)):
        alloc = step_idx_allocfree[t].size_alloc
        free = step_idx_allocfree[t].size_free
        cur_memory += alloc
        post_alloc = cur_memory
        max_memory = max(max_memory, cur_memory)
        cur_memory -= free
        post_free = cur_memory
        snodes_curr_memory.append((post_alloc, post_free))

    return (
        max_memory,
        snodes_curr_memory,
        snodes_allocfree,
        buf_to_snode_last_use,
    )


def topological_sort_lpmf(
    nodes: list[BaseSchedulerNode],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
    name_to_buf: dict[str, SchedulerBuffer],
    graph_outputs: OrderedSet[str],
) -> list[BaseSchedulerNode]:
    """
    A bfs-based greedy topological order. LPMF stands for "Least Peak Memory First".

    The idea is from this paper:
    Buffer memory optimization for video codec application modeled in Simulink
    https://www.cs.york.ac.uk/rts/docs/DAC-1964-2006/PAPERS/2006/DAC06/PDFFILES/P0689.PDF

    The algorithm maintains the max memory so far.
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

    node_info: dict[BaseSchedulerNode, NodeInfo] = dict()
    buf_info: dict[Union[SchedulerBuffer, FreeableInputBuffer], BufferInfo] = dict()

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
    memory_gap = max_memory - live_memory

    # compute the amount of memory that is allocated when a node is scheduled
    # and the amount of memory that can be freed when a node is scheduled
    for node in nodes:
        # 1. if a buffer read by this node is last used by this node
        for buf in node.mpi_node.pred_buffers:
            if buf_info[buf]["outdegree"] == 1:
                node_info[node]["memory_to_free"] += buf.mpi_buffer.size_free
        # 2. if a buffer written by this node is used internally and not used later
        for buf in node.get_outputs():
            if buf_info[buf]["outdegree"] == 0:
                node_info[node]["memory_to_free"] += buf.mpi_buffer.size_free

    # schedule nodes one at a time
    schedule: list[BaseSchedulerNode] = []
    size_threshold = config.size_threshold_for_succ_based_strategy
    num_iters: int = 0
    while num_iters < len(nodes) and nodes_to_schedule:
        # select a node to schedule:
        if (
            size_threshold > 0
            and min(node.mpi_node.size for node in nodes_to_schedule) > size_threshold
        ):
            selected_node = min(
                nodes_to_schedule,
                key=lambda node: min(
                    (
                        succ_node.mpi_node.index
                        for succ_node in node.mpi_node.succ_nodes
                    ),
                    default=len(nodes),
                ),
            )
        else:
            selected_node = min(
                nodes_to_schedule,
                key=lambda node: (
                    node.mpi_node.size if node.mpi_node.size > memory_gap else 0,
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
        memory_gap = max_memory - live_memory

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


def topological_sort_bfs(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
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

    node_info: dict[BaseSchedulerNode, NodeInfo] = dict()

    @dataclasses.dataclass
    class NodeWithPriority:
        priority: list[int]
        node: BaseSchedulerNode

        def __lt__(self, other: NodeWithPriority) -> bool:
            if self.priority == other.priority:
                return self.node.mpi_node.index < other.node.mpi_node.index
            return self.priority < other.priority

    def _node_priority(node: BaseSchedulerNode) -> list[int]:
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
    nodes_to_schedule: list[NodeWithPriority] = []
    for node in nodes:
        node_info[node] = {"indegree": len(node.mpi_node.pred_nodes), "order": -1}
        if node_info[node]["indegree"] == 0:
            heapq.heappush(
                nodes_to_schedule, NodeWithPriority(_node_priority(node), node)
            )

    # schedule nodes one at a time
    schedule: list[BaseSchedulerNode] = []
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


def topological_sort_dfs(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """
    This is a DFS topological sort. The setup is similar to `topological_sort_schedule`
    in scheduler.py. The difference is the order nodes are visited in the outer loop.
    In `topological_sort_schedule`, nodes are visited in their original order.
    In this function, nodes are visited based on their priority -- for each node, we
    compute the total memory of all buffers it reads from or writes to, and we visit
    the nodes in ascending order of this priority.
    """
    seen: OrderedSet[BaseSchedulerNode] = OrderedSet()
    name_to_node: dict[str, BaseSchedulerNode] = dict()
    result: list[BaseSchedulerNode] = []
    size_with_reads: dict[BaseSchedulerNode, int] = dict()

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


def validate_graph_acyclic(nodes: list[BaseSchedulerNode]) -> None:
    """
    Validate that the graph is acyclic by checking predecessor relationships.

    Raises:
        RuntimeError: If a cycle is detected in the graph
    """
    # DFS coloring scheme for cycle detection:
    # WHITE (0): Node has not been visited yet
    # GRAY (1): Node is currently being processed (in the recursion stack)
    # BLACK (2): Node has been completely processed (finished exploring all its predecessors)
    # A back edge (cycle) is detected when we encounter a GRAY node during DFS traversal
    WHITE, GRAY, BLACK = 0, 1, 2
    color = dict.fromkeys(nodes, WHITE)
    path: list[BaseSchedulerNode] = []  # Track current DFS path

    def dfs_visit(node: BaseSchedulerNode) -> None:
        if color[node] == BLACK:
            return

        if color[node] == GRAY:
            path.append(node)
            path_info = " -> ".join([node.get_name() for node in path])

            raise RuntimeError(
                f"Cycle detected in memory planning graph"
                f"Path containing cycle (i -> j: j is a dependency of i): {path_info} "
                f"This indicates invalid dependency relationships in the scheduler graph"
            )

        color[node] = GRAY
        path.append(node)

        for pred_node in node.mpi_node.pred_nodes:
            assert pred_node != node
            dfs_visit(pred_node)

        path.pop()
        color[node] = BLACK

    # Start DFS from all unvisited nodes
    for node in nodes:
        if color[node] == WHITE:
            dfs_visit(node)


def validate_unique_buffer_names(
    nodes: list[BaseSchedulerNode],
    name_to_buf: dict[str, SchedulerBuffer],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
) -> None:
    """
    Validate that for each node's output buffer, the name_to_buf mapping is correct.
    For each output buffer buf, we should have name_to_buf[buf.get_name()] == buf.
    Also validate that no buffer names overlap with freeable input buffer names.

    Raises:
        RuntimeError: If buffer name mapping is incorrect or names overlap
    """
    for node in nodes:
        for buf in node.get_outputs():
            buf_name = buf.get_name()

            # Check if buffer name exists in the mapping
            if buf_name not in name_to_buf:
                raise RuntimeError(
                    f"{buf_name} from {node.get_name()} is not found in name_to_buf mapping."
                    f" This indicates a missing buffer mapping."
                )

            # Check if the mapping points to the correct buffer object
            if name_to_buf[buf_name] != buf:
                raise RuntimeError(
                    f"Buffer name mapping is incorrect for '{buf_name}'."
                    f"Expected name_to_buf['{buf_name}'] to be {buf.debug_str()}"
                    f"but got {name_to_buf[buf_name].debug_str()}"
                    f"This indicates some buffers share the same name"
                )

            # Check if buffer name conflicts with freeable input buffer names
            if buf_name in name_to_freeable_input_buf:
                raise RuntimeError(
                    f"Buffer name conflict detected: '{buf_name}' from node {node.get_name()} "
                    f"is also used as a freeable input buffer name. "
                )


def prepare_planning_info(
    nodes: list[BaseSchedulerNode],
    name_to_buf: dict[str, SchedulerBuffer],
    name_to_fused_node: dict[str, BaseSchedulerNode],
    graph_inputs: OrderedSet[str],
    graph_outputs: OrderedSet[str],
) -> tuple[int, dict[str, FreeableInputBuffer]]:
    """
    Prepare planning info. As nodes are scheduled one at a time, these help
    keep track of when a buffer can be freed, and when a node can be scheduled

    Returns:
        int: peak memory estimation
        dict[str, FreeableInputBuffer]: name to freeable input buffer
    """
    name_to_freeable_input_buf = get_freeable_input_buf(nodes, graph_inputs)
    assign_memory_planning_info_for_scheduler_buffers(nodes, name_to_buf)
    assign_memory_planning_info_for_scheduler_nodes(
        nodes, name_to_fused_node, name_to_buf, name_to_freeable_input_buf
    )

    # the default
    estimated_peak_memory, _ = estimate_peak_memory(
        nodes, name_to_freeable_input_buf, graph_outputs
    )

    return estimated_peak_memory, name_to_freeable_input_buf


def reorder_for_peak_memory(
    nodes: list[BaseSchedulerNode],
    name_to_buf: dict[str, SchedulerBuffer],
    name_to_fused_node: dict[str, BaseSchedulerNode],
    graph_inputs: OrderedSet[str],
    graph_outputs: OrderedSet[str],
    methods: list[Callable[..., list[BaseSchedulerNode]]] = [  # noqa: B006
        topological_sort_lpmf,
        topological_sort_bfs,
        topological_sort_dfs,
    ],
) -> list[BaseSchedulerNode]:
    """
    Try a few heuristics based topological sort algorithms, and pick the one whose
    resulting topological order has the lowest peak memory estimation.
    """

    torch_log.info("Reordering for peak memory -- %d nodes", len(nodes))

    estimated_peak_memory, name_to_freeable_input_buf = prepare_planning_info(
        nodes,
        name_to_buf,
        name_to_fused_node,
        graph_inputs,
        graph_outputs,
    )

    # export graph for simulator if needed
    if config.reorder_for_peak_memory_debug:
        export_graph_for_simulator(
            nodes,
            name_to_freeable_input_buf,
            name_to_fused_node,
            graph_inputs,
            graph_outputs,
        )

    # Validate planning info before proceeding with reordering
    try:
        validate_graph_acyclic(nodes)
        validate_unique_buffer_names(nodes, name_to_buf, name_to_freeable_input_buf)
    except RuntimeError:
        torch_log.exception("Memory planning validation failed")
        if not is_fbcode():  # TODO: remove after ensuring OSS side is safe
            raise

    # keep track of the peak memory estimates of different methods
    peak_memory_diff_methods: list[PeakMemoryResult] = []
    peak_memory_diff_methods.append(
        PeakMemoryResult(nodes, estimated_peak_memory, "baseline")
    )
    torch_log.info("Baseline peak memory: %d", estimated_peak_memory)

    # other methods
    for method in methods:
        try:
            if method is topological_sort_lpmf:
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
        except Exception:
            torch_log.exception("Failed to reorder for %s", method.__name__)
            if not is_fbcode():  # TODO: remove after ensuring OSS side is safe
                raise

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


def export_graph_for_simulator(
    nodes: list[BaseSchedulerNode],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
    name_to_fused_node: dict[str, BaseSchedulerNode],
    graph_inputs: OrderedSet[str],
    graph_outputs: OrderedSet[str],
) -> None:
    """
    This is for debugging purposes. It will dump a json file that records graph information.
    The graph can then be used in a simulator: https://fburl.com/code/3l3d3qi4
    """

    class ORMBuffer(TypedDict):
        name: str
        size_alloc: int
        size_free: int
        size: int  # for backward compatibility
        is_input: bool
        is_output: bool
        deps: list[str]
        unmet_deps: list[str]

    class ORMNode(TypedDict):
        name: str
        buffer_names: list[str]

    class ORMGraph(TypedDict):
        nodes: list[ORMNode]
        buffers: list[ORMBuffer]

    orm_buffers: list[ORMBuffer] = []
    orm_nodes: list[ORMNode] = []

    # get orm buffers for freeable input buffers
    for buf_name, input_buf in name_to_freeable_input_buf.items():
        orm_buf_input_buffer: ORMBuffer = {
            "name": buf_name,
            "size_alloc": input_buf.mpi_buffer.size_free,
            "size_free": input_buf.mpi_buffer.size_free,
            "size": input_buf.mpi_buffer.size_free,
            "is_input": True,
            "is_output": buf_name in graph_outputs,
            "deps": [],
            "unmet_deps": [],
        }
        orm_buffers.append(orm_buf_input_buffer)

    # get orm buffers for scheduler buffers
    name_to_buf: dict[str, SchedulerBuffer] = {
        buf.get_name(): buf for node in nodes for buf in node.get_outputs()
    }  # need to reassign due to probably node pruning
    for buf_name, sched_buf in name_to_buf.items():
        if sched_buf.defining_op is None:
            continue
        deps = [
            pred_buf.get_name()
            for pred_buf in name_to_fused_node[
                sched_buf.defining_op.get_name()
            ].mpi_node.pred_buffers
        ]
        orm_buf_scheduler_buffer: ORMBuffer = {
            "name": buf_name,
            "size_alloc": sched_buf.mpi_buffer.size_alloc,
            "size_free": sched_buf.mpi_buffer.size_free,
            "size": sched_buf.mpi_buffer.size_free,
            "is_input": False,
            "is_output": buf_name in graph_outputs,
            "deps": deps,
            "unmet_deps": [
                buf_name for buf_name in deps if buf_name not in graph_inputs
            ],
        }
        orm_buffers.append(orm_buf_scheduler_buffer)

    # get orm nodes
    for node in nodes:
        orm_node: ORMNode = {
            "name": node.get_name(),
            "buffer_names": list(node.get_buffer_names()),
        }
        orm_nodes.append(orm_node)

    # create the graph object
    g: ORMGraph = {
        "nodes": orm_nodes,
        "buffers": orm_buffers,
    }

    # dump the graph
    import json
    import os

    import torch
    from functorch.compile import get_graph_being_compiled

    name = os.path.splitext(get_graph_being_compiled())[0] + "_fused"

    g_str = json.dumps(g, indent=2)

    torch._logging.trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": name,
            "encoding": "string",
        },
        payload_fn=lambda: g_str,
    )
