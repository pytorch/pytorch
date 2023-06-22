import logging
log = logging.getLogger(__name__)
from .virtualized import V


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
    def __init__(self, original_node, node_id) -> None:
        self.node_id = node_id
        self.successors = {}
        self.predecessors = {}
        self.name = original_node.get_name() if original_node else ""
        self.original_user_names = []
        # -1 means not assigned
        self.stream_id = -1
        self.snode_names = []
        # mark if this node needs to generate a CUDA event
        self.cuda_event = False
        if hasattr(original_node, "snodes"):
            self.is_fused = True
            for snode in original_node.snodes:
                self.snode_names.append(snode.get_name())
            for snode in original_node.snodes:
                for user in snode.users:
                    if user.get_name() not in self.snode_names and user.get_name() not in self.original_user_names:
                        self.original_user_names.append(user.get_name())
        else:
            self.is_fused = False
            if original_node is not None:
                for user in original_node.users:
                    self.original_user_names.append(user.get_name())
    def get_name(self):
        return self.name
class SSGraph:
    """
    Stream Scheduler Graph records all the information for stream scheduling.
    Attributes:
        ssnodes: [SSNode]. It records all the SSNodes in the graph.
        name_mapping: {buf name: SSNode}. It records the mapping from the original node name to the SSNode. The names include scheduler node name and fused node. For example, buf4, buf5, and buf4_buf5 are all pointed to the same SSNode.
        reverse_level: {SSNode: level}. It records the levels back from the OUTPUT node. The level of OUTPUT node is 0. The level of the predecessors of OUTPUT node is 1. The level of the predecessors of the predecessors of OUTPUT node is 2. And so on.
        reverse_level_predecessors: {level: [SSNode]}. It records the predecessors of each level.
        critical_path: [SSNode]. It records the critical path of the graph. All nodes in the critical path will be assigned to the default stream.
        stream_pool_size: how many extra CUDA streams used to allocate. TODO: it's better to use the max number of nodes in the same level in reverse_level
        stream_pool: [stream_index, ]. It records the CUDA streams used to allocate.
    """
    def __init__(self, nodes) -> None:
        self.ssnodes = []
        self.name_mapping = {}
        # It records the levels back from the OUTPUT node. {ssnode: level, }
        self.reverse_level = {}
        self.reverse_level_predecessors = {}
        self.critical_path = []
        self.stream_pool_size = 3
        self.stream_pool = []
        self.build_graph(nodes)
        self.stream_scheduling()


    def build_graph(self, nodes):
        for node_id, node in enumerate(nodes):
            new_ssnode = SSNode(node, node_id)
            self.ssnodes.append(new_ssnode)
            self.name_mapping[node.get_name()] = new_ssnode
            if new_ssnode.is_fused:
                for snode in new_ssnode.snode_names:
                    self.name_mapping[snode] = new_ssnode
        output_node = SSNode(None, len(nodes))
        output_node.name = "OUTPUT"
        self.name_mapping["OUTPUT"] = output_node
        self.ssnodes.append(output_node)
        def update_successor_predecessor(ssnode, user_names):
            for user in user_names:
                user_ssnode = self.name_mapping[user]
                ssnode.successors[user_ssnode.get_name()] = user_ssnode
                user_ssnode.predecessors[ssnode.get_name()] = ssnode
        for ssnode in self.ssnodes:
            update_successor_predecessor(ssnode, ssnode.original_user_names)
        tmp_queue = []
        self.reverse_level[output_node] = 0

        for predecessor in output_node.predecessors.values():
            # only append the node that has only one successor OUTPUT
            if len(predecessor.successors) == 1:
                tmp_queue.append(predecessor)
        while len(tmp_queue) != 0:
            cur_node = tmp_queue.pop(0)
            # if one of the successors is not assigned, then we cannot assign the level to cur_node
            for successor in cur_node.successors.values():
                if successor not in self.reverse_level:
                    if successor not in tmp_queue:
                        # Yueming TODO: This can be delayed.
                        tmp_queue.append(cur_node)
                    break;
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
                    tmp_queue.append(predecessor)
        max_key = max(self.reverse_level, key=self.reverse_level.get)
        max_value = self.reverse_level[max_key]
        log.info(f"max level is {max_value}, and the node is {max_key.get_name()}")
        if 'buf0' not in max_key.get_name():
            log.warning(f"buf0's level is {self.reverse_level[self.name_mapping['buf0']]}")
        cur_node = self.name_mapping["buf0"]
        while(cur_node.get_name() != "OUTPUT"):
            self.critical_path.append(cur_node)
            cur_node = self.reverse_level_predecessors[cur_node]
        self.critical_path.append(cur_node)

    # Yueming TODO: need a better stream allocation algorithm
    def stream_pool_pop(self, predecessor=None):
        if predecessor is not None:
            self.stream_pool[predecessor.stream_id] += 1
            return predecessor.stream_id
        else:
            # get the min value and its corresponding key in self.stream_pool
            min_value = min(self.stream_pool)
            min_stream = self.stream_pool.index(min_value)
            self.stream_pool[min_stream] += 1
        return min_stream

    def dig_node(self, cur_node:SSNode):
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
        assert "buf0" in self.ssnodes[0].get_name()
        for i in range(self.stream_pool_size + 1):
            self.stream_pool.append(0)
        # Yueming TODO: do we need keep this fake 0?
        self.stream_pool[0] = len(self.ssnodes) + 2
        self.dig_node(self.ssnodes[0])
        self.event_assign()
    
    def print_graph(self):
        import datetime
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
        reset_log_path = temporary_log_path(f'/home/users/yhao24/p9_inductor/pytorch/torch_compile_debug/tmp/stream_alexnet_{current_time}.log')

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
            log.info(f"{node.get_name()} {node.stream_id}")
        log.info("=====findhao debug stream allocation end=====")


        reset_log_path()


def stream_schedule(snodes):
    pass
    ssgraph = SSGraph(snodes)
    ssgraph.print_graph()
    V.graph.stream_graph = ssgraph
    return ssgraph