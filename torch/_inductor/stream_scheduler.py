import logging
from os import environ

from .virtualized import V
from typing import Union
from .scheduler import NopKernelSchedulerNode, FusedSchedulerNode, OutputNode, BaseSchedulerNode
from torch.utils._ordered_set import OrderedSet
log = logging.getLogger(__name__)
import os
import sys

import torch


# It is 1000 by default. But the call depth for dig_node can be larger than this number. @TODO: FIXME
sys.setrecursionlimit(5000)

UNASSIGNED_STREAM_ID = -1


def in_output(snode: Union[BaseSchedulerNode, FusedSchedulerNode]) -> bool:
    if isinstance(snode, FusedSchedulerNode):
        return any(in_output(x) for x in snode.snodes)
    return any(
        isinstance(user.node, OutputNode)
        for buf in snode.get_outputs()
        for user in buf.users
    )

class SSNode:
    """
    Stream Scheduler Node is a wrapper of the original node. It contains the information of the original node and the information for stream scheduling.
    Attributes:
        node_id: the index of the node in the graph
        successors: {buf_name: SSNode}. It records the successors of the node. The buf_name is the name of the original node(scheduler node or fused node).
        predecessors: {buf_name: SSNode}. It records the predecessors of the node. The buf_name is the name of the original node(scheduler node or fused node).
        first_predecessor: SSNode. A node should have same stream id with its first predecessor.
        fake_successors, fake_predecessors: {buf_name: SSNode}.
        name: the name of the original node.
        original_user_names: the names of the original users of the node. If the original node is a fused node, we'll check the users of scheduler nodes included in this fused node.
        stream_id: the stream id of the node. -1 means not assigned.
        snode_names: the names of the scheduler nodes included in this fused node.
        cuda_event: mark if this node needs to generate a CUDA event. CUDA events are used to keep the data dependencies between streams.
        is_fused: mark if this node is a fused node.
        is_nop_node: mark if this node is a nop node.
        node_type: the type of the node. It can be "template", "extern", "foreach", "fused_or_schedule", "nop"
        device: by default is None. If it is a cpu node, we will skip it.
        skip_cpu_nodes: It is meaningless to add some stream switch for cpu nodes.
    """

    def __init__(self, original_node, skip_cpu_nodes=True) -> None:
        assert original_node is not None
        self.successors = OrderedSet()
        self.predecessors = OrderedSet()
        self.first_successor = None
        self.first_predecessor = None
        # self.fake_successors = {}
        # self.fake_predecessors = {}
        self.name = original_node.get_name()
        self.original_user_names = []
        # SchedulerNode is operation node now which is different from SchedulerBuffer.
        self.original_node = original_node
        self.to_output_node = False
        # -1 means not assigned. It is important to set it to -1 instead of 0, because 0 is a valid stream id.
        self.stream_id = UNASSIGNED_STREAM_ID
        if isinstance(original_node, FusedSchedulerNode):
            self.snode_names = [node.get_name() for node in original_node.snodes]
        else:
            self.snode_names = []
        # mark if this node needs to generate a CUDA event
        self.cuda_event = False
        self.is_nop_node = isinstance(original_node, NopKernelSchedulerNode)
        self.node_type = None
        self.device = None
        self.skip_cpu_nodes = skip_cpu_nodes
        # is it enough to check if the node is a cpu node?
        if self.skip_cpu_nodes and original_node:
            if hasattr(original_node, "group"):
                for device in original_node.group:
                    if isinstance(device, torch.device) and device.type != "cpu":
                        self.device = device.type
                        break
                else:
                    self.device = "cpu"
            elif hasattr(original_node, "get_device"):
                self.device = original_node.get_device().type
        self.to_output_node = in_output(original_node)
        self.is_fused = isinstance(original_node, FusedSchedulerNode)

    def get_name(self):
        return self.name

class SSGraph:
    """
    Stream Scheduler Graph records all the information for stream scheduling.
    Attributes:
        ssnodes: [SSNode]. It records all the SSNodes in the graph. The order matters.
        op_to_ssnode: {buf name: SSNode}. It records the mapping from the original node name to the SSNode. The names include scheduler node name and fused node. For example, buf4, buf5, and buf4_buf5 are all pointed to the same SSNode.
        reverse_level: {SSNode: level}. It records the levels back from the OUTPUT node. The level of OUTPUT node is 0. The level of the predecessors of OUTPUT node is 1. The level of the predecessors of the predecessors of OUTPUT node is 2. And so on.
        reverse_level_predecessors: {SSNode:reverse_predecessor_node, }. It records a node's predecessor in reverse order.
        critical_path: [SSNode]. It records the critical path of the graph. All nodes in the critical path will be assigned to the default stream.
        stream_pool_size: how many extra CUDA streams used to allocate. TODO: it's better to use the max number of nodes in the same level in reverse_level
        stream_pool: [stream_index, ]. It records the CUDA streams used to allocate.
        final_order: [SSNode]. It records the final order of the nodes after reorder. The order matters.
        max_stream_id: the max stream id used in the graph.
        arg_to_stream: the argument to stream mapping. {arg_name: stream_id}
    """

    def __init__(self, snodes) -> None:
        self.ssnodes = []
        self.op_to_ssnode = {}
        self.buf_to_ssnode = {}
        # It records the levels back from the OUTPUT node. {ssnode: level, }
        self.reverse_level = {}
        self.reverse_level_predecessors = {}
        self.critical_path = []
        self.stream_pool_size = int(environ.get("STREAM_POOL_SIZE", 31))
        self.stream_pool = [0] * (self.stream_pool_size + 1)
        self.arg_to_stream = {}
        self.final_order = []
        self.max_stream_id = 0
        self.skip_cpu_nodes = True
        self.to_output_nodes = []
        # By default, we use the same mechanism with the original default stream assignment, which take the stream{device_index} as default stream id. 
        self.DEFAULT_STREAM_ID = None
        self.build_graph(snodes)

    def build_graph(self, snodes):
        for snode in snodes:
            if self.DEFAULT_STREAM_ID is None and not isinstance(snode, NopKernelSchedulerNode) and (device := snode.get_device()):
                self.DEFAULT_STREAM_ID = device.index
            new_ssnode = SSNode(snode)
            self.ssnodes.append(new_ssnode)
            self.op_to_ssnode[snode.get_name()] = new_ssnode
            if new_ssnode.to_output_node:
                self.to_output_nodes.append(new_ssnode.name)
            if new_ssnode.is_fused:
                for tmp_name in new_ssnode.snode_names:
                    self.op_to_ssnode[tmp_name] = new_ssnode
            for schedulerbuffer in snode.get_outputs():
                self.buf_to_ssnode[schedulerbuffer.get_name()] = new_ssnode
        # build dependencies
        # {buf1: op2, }
        buf_last_update_op: Dict[str, str] = {}
        for snode in snodes:
            deps = snode.read_writes.reads
            for schedulerbuffer in snode.get_outputs():
                self.buf_to_ssnode[schedulerbuffer.get_name()] = self.op_to_ssnode[
                    schedulerbuffer.defining_op.get_name()
                ]
            for dep in deps:
                # we only need to care about buffers here
                last_update_op = buf_last_update_op.get(dep.name, None)
                if last_update_op:
                    dep_node = self.op_to_ssnode[last_update_op]
                    self.op_to_ssnode[snode.get_name()].predecessors.add(dep_node)
                    dep_node.successors.add(self.op_to_ssnode[snode.get_name()])
            for output in snode.outputs_by_name.keys():
                buf_last_update_op[output] = snode.get_name()

    def pattern_distributed(self):
        tmp_queue = self.to_output_nodes
        from .ir import FallbackKernel
        from .scheduler import ExternKernelSchedulerNode
        
        for ssnode in self.ssnodes:
            if ssnode.stream_id != UNASSIGNED_STREAM_ID:
                continue
            # copy-in
            if isinstance(ssnode.original_node, ExternKernelSchedulerNode) and isinstance(ssnode.original_node.node, FallbackKernel) and "torch.ops.fsdp.all_gather_copy_in.default" in ssnode.original_node.node.python_kernel_name:
                new_stream_id = self.stream_pool_pop()
                ssnode.stream_id = new_stream_id
                tmp_queue = list(ssnode.successors)
                while tmp_queue:
                    tmp_ssnode = tmp_queue.pop()
                    if tmp_ssnode.stream_id != UNASSIGNED_STREAM_ID:
                        continue
                    # copy-out
                    if isinstance(tmp_ssnode.original_node, ExternKernelSchedulerNode) and isinstance(tmp_ssnode.original_node.node, FallbackKernel) and "torch.ops.fsdp.split_with_sizes_copy.default" in tmp_ssnode.original_node.node.python_kernel_name:
                        extern_kernel_node_count = 0
                        for predecessor in tmp_ssnode.predecessors:
                            predecessor.stream_id = new_stream_id
                            if isinstance(predecessor.original_node, ExternKernelSchedulerNode):
                                extern_kernel_node_count += 1
                            elif isinstance(predecessor.original_node, NopKernelSchedulerNode):
                                continue
                            else:
                                raise RuntimeError(f"Unexpected predecessor {predecessor} for copy_out node {tmp_ssnode}")
                        assert extern_kernel_node_count == 1, f"Expected exactly one extern kernel node as predecessor for copy_out node {tmp_ssnode}, but got {extern_kernel_node_count}. Pattern match failed."
                    else:
                        tmp_queue += list(tmp_ssnode.successors)
                    tmp_ssnode.stream_id = new_stream_id
            else:
                ssnode.stream_id = self.DEFAULT_STREAM_ID

    def stream_pool_pop(self, predecessor=None):
        if predecessor is not None:
            self.stream_pool[predecessor.stream_id] += 1
            return predecessor.stream_id
        else:
            min_value = min(self.stream_pool)
            min_stream = self.stream_pool.index(min_value)
            self.stream_pool[min_stream] += 1
        return min_stream


    def event_assign(self):
        # if at least one of the node's successors is not in the same stream, then we need to add an event
        for ssnode in self.ssnodes:
            for successor in ssnode.successors:
                if successor.stream_id != ssnode.stream_id:
                    ssnode.cuda_event = True
                    break
                # TODO: double check how we process nop nodes now.
                if successor.is_nop_node:
                    for successor_successor in successor.successors:
                        if successor_successor.stream_id != ssnode.stream_id:
                            ssnode.cuda_event = True
                            break

    def stream_assign(self):
        # To avoid assigning default stream when we want to pop a new stream from the pool.
        self.stream_pool[self.DEFAULT_STREAM_ID] = len(self.ssnodes) + 2
        self.pattern_distributed()
        def check_all_nodes_assigned():
            for ssnode in self.ssnodes:
                if ssnode.stream_id == UNASSIGNED_STREAM_ID:
                    log.info(
                        f"Hanging node {ssnode.get_name()} found when doing stream assignment."
                    )
                    self.dig_node(ssnode)
                    return False
            return True

        while not check_all_nodes_assigned():
            pass


def stream_schedule(snodes):
    # Need to be same with where calls `write_get_raw_stream`
    ssgraph = SSGraph(snodes)
    ssgraph.stream_assign()
    ssgraph.event_assign()
    V.graph.stream_graph = ssgraph
    return ssgraph
