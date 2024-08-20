import logging
from os import environ

from .virtualized import V


log = logging.getLogger(__name__)
import os
import sys

import torch


# It is 1000 by default. But the call depth for dig_node can be larger than this number. @TODO: FIXME
sys.setrecursionlimit(5000)
# By default, we use the same mechanism with the original default stream assignment, which take the stream{device_index} as default stream id. It is initialized in stream_schedule
DEFAULT_STREAM_ID = None


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
        kernel_volumn: the product of the node's arguments' size. It usually represents the number of kernel threads.
        device: by default is None. If it is a cpu node, we will skip it.
        skip_cpu_nodes: It is meaningless to add some stream switch for cpu nodes.
    """

    def __init__(self, original_node, skip_cpu_nodes=True) -> None:
        from .scheduler import NopKernelSchedulerNode

        self.successors = {}
        self.predecessors = {}
        self.first_successor = None
        self.first_predecessor = None
        self.fake_successors = {}
        self.fake_predecessors = {}
        self.name = original_node.get_name() if original_node else ""
        self.original_user_names = []
        self.original_node = original_node
        self.to_output_node = False
        # -1 means not assigned. It is important to set it to -1 instead of 0, because 0 is a valid stream id.
        self.stream_id = -1
        self.snode_names = []
        # mark if this node needs to generate a CUDA event
        self.cuda_event = False
        self.is_nop_node = isinstance(original_node, NopKernelSchedulerNode)
        self.node_type = None
        self.kernel_volumn = 0
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
        if self.name and original_node.read_writes.var_ranges:
            results = 1
            for key, value in original_node.read_writes.var_ranges.items():
                results *= value
            self.kernel_volumn = results
        if hasattr(original_node, "snodes"):
            self.is_fused = True
            for snode in original_node.snodes:
                self.snode_names.append(snode.get_name())
            for snode in original_node.snodes:
                for user in snode.users:
                    if user.get_name() == "OUTPUT":
                        self.to_output_node = True
                # TODO: change to follow rules same with create_fx_from_snodes
                for user in snode.read_writes.reads:
                    if (
                        user.name not in self.snode_names
                        and user.name not in self.original_user_names
                    ):
                        if user.name != snode.get_name():
                            self.original_user_names.append(user.name)

        else:
            self.is_fused = False
            if original_node is not None:
                for user in original_node.users:
                    if user.get_name() == "OUTPUT":
                        self.to_output_node = True
                for user in original_node.read_writes.reads:
                    if user.name != original_node.get_name():
                        self.original_user_names.append(user.name)

    def get_name(self):
        return self.name


class SSGraph:
    """
    Stream Scheduler Graph records all the information for stream scheduling.
    Attributes:
        ssnodes: [SSNode]. It records all the SSNodes in the graph. The order matters.
        name_mapping: {buf name: SSNode}. It records the mapping from the original node name to the SSNode. The names include scheduler node name and fused node. For example, buf4, buf5, and buf4_buf5 are all pointed to the same SSNode.
        reverse_level: {SSNode: level}. It records the levels back from the OUTPUT node. The level of OUTPUT node is 0. The level of the predecessors of OUTPUT node is 1. The level of the predecessors of the predecessors of OUTPUT node is 2. And so on.
        reverse_level_predecessors: {SSNode:reverse_predecessor_node, }. It records a node's predecessor in reverse order.
        critical_path: [SSNode]. It records the critical path of the graph. All nodes in the critical path will be assigned to the default stream.
        stream_pool_size: how many extra CUDA streams used to allocate. TODO: it's better to use the max number of nodes in the same level in reverse_level
        stream_pool: [stream_index, ]. It records the CUDA streams used to allocate.
        final_order: [SSNode]. It records the final order of the nodes after reorder. The order matters.
        max_stream_id: the max stream id used in the graph.
        arg_to_stream: the argument to stream mapping. {arg_name: stream_id}
    """

    def __init__(self, nodes) -> None:
        self.ssnodes = []
        self.name_mapping = {}
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
        self.build_graph(nodes)

    def build_graph(self, nodes):
        output_node = SSNode(None, skip_cpu_nodes=self.skip_cpu_nodes)
        output_node.name = "OUTPUT"
        self.name_mapping["OUTPUT"] = output_node
        for node in nodes:
            new_ssnode = SSNode(node)
            self.ssnodes.append(new_ssnode)
            self.name_mapping[node.get_name()] = new_ssnode
            if new_ssnode.is_fused:
                for snode in new_ssnode.snode_names:
                    self.name_mapping[snode] = new_ssnode
            # huggingface/YituTechConvBert try to get buf1144 which is part of a fused node but not in new_ssnode.snode_names
            possible_names = node.get_name().split("_")
            for name in possible_names:
                self.name_mapping[name] = new_ssnode

        # clean the freed buffers
        for snode in self.ssnodes:
            for user in snode.original_user_names:
                if user not in self.name_mapping:
                    snode.original_user_names = [
                        user
                        for user in snode.original_user_names
                        if user in self.name_mapping
                    ]
                    break
        self.ssnodes.append(output_node)

        def update_successor_predecessor(ssnode, user_names):
            for user in user_names:
                user_ssnode = self.name_mapping[user]
                user_ssnode.successors[ssnode.get_name()] = ssnode
                ssnode.predecessors[user_ssnode.get_name()] = user_ssnode

        for ssnode in self.ssnodes:
            if ssnode.to_output_node:
                output_node.predecessors[ssnode.get_name()] = ssnode
                ssnode.successors[output_node.get_name()] = output_node
            update_successor_predecessor(ssnode, ssnode.original_user_names)
        tmp_queue = []
        self.reverse_level[output_node] = 0
        for predecessor in output_node.predecessors.values():
            # only append the node that has only one successor OUTPUT
            if len(predecessor.successors) == 1:
                tmp_queue.append(predecessor)
        if len(tmp_queue) == 0:
            log.warning(
                "This graph doesn't have nodes whose single successor is the OUTPUT node."
            )
            for predecessor in output_node.predecessors.values():
                tmp_queue.append(predecessor)
        finished = set()
        finished.add(output_node)
        # TODO(Yueming): the theorectical maximum iteration is (n-1)^n for a fully connected graph. May need a better solution for fully connected graph.
        for i in range(len(self.ssnodes) ** 3):
            if len(tmp_queue) == 0:
                break
            cur_node = tmp_queue.pop(0)
            # if one of the successors is not assigned, then we cannot assign the level to cur_node
            for successor in cur_node.successors.values():
                if successor not in self.reverse_level:
                    if successor not in tmp_queue:
                        if len(cur_node.successors) == 1 and successor not in tmp_queue:
                            tmp_queue.append(successor)
                        # a workaround to fix hanging nodes
                        if (
                            len(successor.successors.values()) == 0
                            and successor not in tmp_queue
                            and successor != output_node
                        ):
                            tmp_queue.append(successor)
                        tmp_queue.append(cur_node)
                    break
            else:
                # some graphs have multiple nodes whose out degree is 0 (hanging nodes) same with OUTPUT. https://github.com/pytorch/pytorch/issues/127981 It is somekind of common for graphs in training.
                # a workaround to fix hanging nodes
                if len(cur_node.successors) == 0:
                    self.reverse_level[cur_node] = 0
                    self.reverse_level_predecessors[cur_node] = None
                    for predecessor in cur_node.predecessors.values():
                        if predecessor not in finished and predecessor not in tmp_queue:
                            tmp_queue.append(predecessor)
                    finished.add(cur_node)
                    continue
                first_successor = list(cur_node.successors.values())[0]
                max_value = self.reverse_level[first_successor]
                self.reverse_level[cur_node] = max_value + 1
                self.reverse_level_predecessors[cur_node] = first_successor
                for successor in list(cur_node.successors.values())[1:]:
                    if self.reverse_level[successor] > max_value:
                        max_value = self.reverse_level[successor]
                        # the 1 here could be changed later for weighted graph
                        self.reverse_level[cur_node] = max_value + 1
                        self.reverse_level_predecessors[cur_node] = successor
                for predecessor in cur_node.predecessors.values():
                    if predecessor not in finished and predecessor not in tmp_queue:
                        tmp_queue.append(predecessor)
                finished.add(cur_node)
        else:
            log.warning(
                "This warning is not supposed to happen. The graph may be too densely connected."
            )
        if len(tmp_queue) != 0:
            raise RuntimeError(
                "Error when processing the queue. The queue is not empty after the loop."
            )
        cur_node = self.ssnodes[0]
        while cur_node.get_name() != "OUTPUT":
            self.critical_path.append(cur_node)
            cur_node = self.reverse_level_predecessors[cur_node]
        self.critical_path.append(cur_node)

    def stream_pool_pop(self, predecessor=None):
        if predecessor is not None:
            self.stream_pool[predecessor.stream_id] += 1
            return predecessor.stream_id
        else:
            min_value = min(self.stream_pool)
            min_stream = self.stream_pool.index(min_value)
            self.stream_pool[min_stream] += 1
        return min_stream

    def dig_node(self, cur_node: SSNode):
        """
        use DFS to assign the stream_id to each node.
        """
        # check the predecessors first
        for predecessor in cur_node.predecessors.values():
            if predecessor.stream_id == -1:
                self.dig_node(predecessor)
        if cur_node.stream_id == -1:
            if cur_node in self.critical_path:
                cur_node.stream_id = CRITICAL_PATH_STREAM_ID
            # if the node has only one predecessor and the predecessor has only one successor, then we can assign the same stream_id to the node
            elif self.skip_cpu_nodes and cur_node.device == "cpu":
                if len(cur_node.predecessors) == 0:
                    cur_node.stream_id = CRITICAL_PATH_STREAM_ID
                else:
                    if len(cur_node.predecessors) > 0:
                        tmp_predecessor = list(cur_node.predecessors.values())[0]
                    else:
                        tmp_predecessor = None
                    cur_node.stream_id = self.stream_pool_pop(tmp_predecessor)
            elif len(cur_node.predecessors) == 1:
                predecessor = list(cur_node.predecessors.values())[0]
                if len(predecessor.successors) == 1:
                    cur_node.stream_id = self.stream_pool_pop(predecessor)
                else:
                    tmp_arg = None
                    if (
                        os.getenv("TORCHINDUCTOR_BYPASS_TINY", "0") == "1"
                        and cur_node.kernel_volumn < HIGH_KERNEL_VOLUMN
                    ):
                        tmp_arg = predecessor
                    cur_node.stream_id = self.stream_pool_pop(tmp_arg)
            else:
                tmp_arg = None
                if (
                    os.getenv("TORCHINDUCTOR_BYPASS_TINY", "0") == "1"
                    and cur_node.kernel_volumn < HIGH_KERNEL_VOLUMN
                    and len(cur_node.predecessors) != 0
                ):
                    tmp_arg = list(cur_node.predecessors.values())[0]
                cur_node.stream_id = self.stream_pool_pop(tmp_arg)
        for successor in cur_node.successors.values():
            if successor.stream_id == -1:
                self.dig_node(successor)

    def event_assign(self):
        # if at least one of the node's successors is not in the same stream, then we need to add an event
        for ssnode in self.ssnodes:
            for successor in ssnode.successors.values():
                if successor.stream_id != ssnode.stream_id:
                    ssnode.cuda_event = True
                    break
                if successor.is_nop_node:
                    for successor_successor in successor.successors.values():
                        if successor_successor.stream_id != ssnode.stream_id:
                            ssnode.cuda_event = True
                            break

    def stream_assign(self):
        # The stream 0 is reserved when do the stream pool pop
        self.stream_pool[0] = len(self.ssnodes) + 2
        self.dig_node(self.ssnodes[0])

        def check_all_nodes_assigned():
            for ssnode in self.ssnodes:
                if ssnode.stream_id == -1:
                    log.info(
                        f"Hanging node {ssnode.get_name()} found when doing stream assignment."
                    )
                    self.dig_node(ssnode)
                    return False
            return True

        while not check_all_nodes_assigned():
            pass

    def dfs_search(self, cur_node: SSNode):
        # sort the predecessors by their stream_id and sort from large to small
        if len(cur_node.predecessors) != 0:
            sorted_predecessors = sorted(
                cur_node.predecessors.values(), key=lambda x: x.stream_id, reverse=True
            )
            for predecesor in sorted_predecessors:
                if predecesor not in self.final_order:
                    self.dfs_search(predecesor)
        if cur_node not in self.final_order:
            self.final_order.append(cur_node)
        if len(cur_node.successors) != 0:
            sorted_successors = sorted(
                cur_node.successors.values(), key=lambda x: x.stream_id, reverse=True
            )
            if self.name_mapping["OUTPUT"] in sorted_successors:
                sorted_successors.remove(self.name_mapping["OUTPUT"])
                sorted_successors.append(self.name_mapping["OUTPUT"])

            for successor in sorted_successors:
                if successor not in self.final_order:
                    self.dfs_search(successor)


def stream_schedule(snodes):
    global DEFAULT_STREAM_ID
    DEFAULT_STREAM_ID = V.graph.scheduler.get_current_device_or_throw().index
    ssgraph = SSGraph(snodes)
    ssgraph.stream_assign()
    ssgraph.event_assign()
    V.graph.stream_graph = ssgraph
    return ssgraph
