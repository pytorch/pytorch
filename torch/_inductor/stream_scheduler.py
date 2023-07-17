from os import environ
from .virtualized import V
import logging
log = logging.getLogger(__name__)


def temporary_log_path(temp_log_path):
    # Store original configuration
    root_logger = log
    handlers = root_logger.handlers[:]

    # Set new log path
    log_format = logging.Formatter('[%(asctime)s]%(name)s:[%(levelname)s] %(message)s')

    log_file_handler = logging.FileHandler(temp_log_path)
    log_file_handler.setFormatter(log_format)
    root_logger.addHandler(log_file_handler)

    def reset_log_path():
        # Restore original configuration
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        root_logger.handlers = handlers

    return reset_log_path


class SSNode:
    """
    Stream Scheduler Node is a wrapper of the original node. It contains the information of the original node and the information for stream scheduling.
    Attributes:
        node_id: the index of the node in the graph
        successors: {buf_name: SSNode}. It records the successors of the node. The buf_name is the name of the original node(scheduler node or fused node).
        predecessors: {buf_name: SSNode}. It records the predecessors of the node. The buf_name is the name of the original node(scheduler node or fused node).
        name: the name of the original node.
        original_user_names: the names of the original users of the node. If the original node is a fused node, we'll check the users of scheduler nodes included in this fused node.
        stream_id: the stream id of the node. -1 means not assigned.
        snode_names: the names of the scheduler nodes included in this fused node.
        cuda_event: mark if this node needs to generate a CUDA event. CUDA events are used to keep the data dependencies between streams.
        is_fused: mark if this node is a fused node.
    """

    def __init__(self, original_node) -> None:
        from .scheduler import NopKernelSchedulerNode
        self.successors = {}
        self.predecessors = {}
        self.name = original_node.get_name() if original_node else ""
        self.original_user_names = []
        self.original_node = original_node
        self.to_output_node = False
        # -1 means not assigned
        self.stream_id = -1
        self.snode_names = []
        # mark if this node needs to generate a CUDA event
        self.cuda_event = False
        self.is_nop_node = isinstance(original_node, NopKernelSchedulerNode)
        if hasattr(original_node, "snodes"):
            self.is_fused = True
            for snode in original_node.snodes:
                self.snode_names.append(snode.get_name())
            for snode in original_node.snodes:
                for user in snode.users:
                    if user.get_name() == 'OUTPUT':
                        self.to_output_node = True
                # TODO: change to follow rules same with create_fx_from_snodes
                for user in snode.read_writes.reads:
                    if user.name not in self.snode_names and user.name not in self.original_user_names:
                        if user.name != snode.get_name():
                            self.original_user_names.append(user.name)

        else:
            self.is_fused = False
            if original_node is not None:
                for user in original_node.users:
                    if user.get_name() == 'OUTPUT':
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
        reverse_level_predecessors: {level: [SSNode]}. It records the predecessors of each level.
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
        self.stream_pool_size = int(environ.get("STREAM_POOL_SIZE", 1))
        self.stream_pool = [0] * (self.stream_pool_size + 1)
        self.arg_to_stream = {}
        self.final_order = []
        self.max_stream_id = 0
        self.build_graph(nodes)
        self.stream_scheduling()

    def build_graph(self, nodes):
        output_node = SSNode(None)
        output_node.name = "OUTPUT"
        self.name_mapping["OUTPUT"] = output_node
        for node in nodes:
            new_ssnode = SSNode(node)
            self.ssnodes.append(new_ssnode)
            self.name_mapping[node.get_name()] = new_ssnode
            if new_ssnode.is_fused:
                for snode in new_ssnode.snode_names:
                    self.name_mapping[snode] = new_ssnode

        # clean the freed buffers
        for snode in self.ssnodes:
            for user in snode.original_user_names:
                if user not in self.name_mapping:
                    snode.original_user_names = [
                        user for user in snode.original_user_names if user in self.name_mapping]
                    break
        self.ssnodes.append(output_node)
        # breakpoint()

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
        finished = set()
        finished.add(output_node)
        # TODO(Yueming): what's the theorectical maximum iteration? (n-1)^n?
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
                        # Yueming TODO: This can be delayed.
                        tmp_queue.append(cur_node)
                    break
            else:
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
        if len(tmp_queue) != 0:
            raise RuntimeError("Error when processing the queue. The queue is not empty after the loop.")

        # buf0_level = self.reverse_level[self.name_mapping['buf0']]
        # log.info(f"buf0's level is {buf0_level}")
        # for ssnode, level in self.reverse_level.items():
        #     if level > buf0_level:
        #         log.info(f"node {ssnode.get_name()} is in level {level}")
        cur_node = self.ssnodes[0]
        while (cur_node.get_name() != "OUTPUT"):
            self.critical_path.append(cur_node)
            cur_node = self.reverse_level_predecessors[cur_node]
        self.critical_path.append(cur_node)

    # Yueming TODO: need a better stream allocation algorithm
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
                cur_node.stream_id = 0
            # if the node has only one predecessor and the predecessor has only one successor, then we can assign the same stream_id to the node
            elif len(cur_node.predecessors) == 1:
                predecessor = list(cur_node.predecessors.values())[0]
                if len(predecessor.successors) == 1:
                    cur_node.stream_id = self.stream_pool_pop(predecessor)
                else:
                    cur_node.stream_id = self.stream_pool_pop()
            else:
                cur_node.stream_id = self.stream_pool_pop()
        for successor in cur_node.successors.values():
            if successor.stream_id == -1:
                self.dig_node(successor)

    def event_assign(self):
        # if at least one of the node's successors is not in the same stream, then we need to add an event
        for ssnode in self.ssnodes:
            for successor in ssnode.successors.values():
                if successor.stream_id != ssnode.stream_id:
                    ssnode.cuda_event = True

    def stream_scheduling(self):
        # Yueming TODO: do we need keep this fake 0?
        self.stream_pool[0] = len(self.ssnodes) + 2
        self.dig_node(self.ssnodes[0])
        self.event_assign()

    def dfs_search(self, cur_node: SSNode):
        # sort the predecessors by their stream_id and sort from large to small
        if len(cur_node.predecessors) != 0:
            sorted_predecessors = sorted(cur_node.predecessors.values(), key=lambda x: x.stream_id, reverse=True)
            for predecesor in sorted_predecessors:
                if predecesor not in self.final_order:
                    self.dfs_search(predecesor)
        if cur_node not in self.final_order:
            self.final_order.append(cur_node)
        if len(cur_node.successors) != 0:
            sorted_successors = sorted(cur_node.successors.values(), key=lambda x: x.stream_id, reverse=True)
            if self.name_mapping["OUTPUT"] in sorted_successors:
                sorted_successors.remove(self.name_mapping["OUTPUT"])
                sorted_successors.append(self.name_mapping["OUTPUT"])
            
            for successor in sorted_successors:
                if successor not in self.final_order:
                    self.dfs_search(successor)
    """
    For current global order, the nodes off the critical path are usually the adjacent nodes before the next common successor. It will block the critical path if we don't reorder them.
    """

    def reorder(self):
        self.dfs_search(self.ssnodes[0])
        self.final_order.remove(self.name_mapping["OUTPUT"])
        log.info("=====findhao debug final order=====")
        for node in self.final_order:
            log.info(node.get_name())
        log.info("=====findhao debug final order=====")

    def print_graph(self):
        import datetime
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
        reset_log_path = temporary_log_path(f'/tmp/yhao/stream_assignment_{current_time}.log')

        log.info("=====findhao debug=====")
        for node in self.ssnodes:
            log.info(node.get_name())
            log.info("\tsuccessors:")
            tmp_str = '\t\t'
            for successor in node.successors.values():
                tmp_str += successor.get_name() + ', '
            log.info(tmp_str)
            if len(node.predecessors) != 0:
                log.info("\tpredecessors:")
                tmp_str = '\t\t'
                for predecessor in node.predecessors.values():
                    tmp_str += predecessor.get_name() + ', '
                log.info(tmp_str)
        log.info("=====findhao debug end=====")
        log.info("=====findhao debug reverse level=====")

        for node in self.reverse_level.keys():
            log.info(f"{node.get_name()} {self.reverse_level[node]}")
            # also print the predecessors
            if node.get_name() != "OUTPUT":
                log.info(f"\tpredecessor:{self.reverse_level_predecessors[node].get_name()}")
        if len(self.reverse_level.keys()) != len(self.ssnodes):
            log.error("findhao debug reverse level error")
            log.error(f"len(self.reverse_level.keys()):{len(self.reverse_level.keys())}")
            log.error(f"len(self.ssnodes):{len(self.ssnodes)}")
            missing = self.reverse_level.keys() - self.ssnodes
            for node in missing:
                log.error(f"missing node:{node.get_name()}")
        log.info("=====findhao debug reverse level end=====")
        log.info("=====findhao debug critical path=====")
        for node in self.critical_path:
            log.info(f"{node.get_name()}")
        log.info("=====findhao debug critical path end=====")
        log.info("=====findhao debug stream allocation=====")
        for node in self.ssnodes:
            assert node.stream_id != -1
            if node.cuda_event:
                event_str = "cuda_event True"
            else:
                event_str = ""
            log.info(f"{node.get_name()} {node.stream_id} {event_str}")
        log.info("=====findhao debug stream allocation end=====")

        reset_log_path()


def stream_schedule(snodes):
    ssgraph = SSGraph(snodes)
    ssgraph.reorder()
    import os
    if os.getenv("TORCHINDUCTOR_STREAM_PRINT_GRAPH", "0") == "1":
        ssgraph.print_graph()
    V.graph.stream_graph = ssgraph
    return ssgraph
