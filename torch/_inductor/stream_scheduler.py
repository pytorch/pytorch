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


# StreamSchedulerNode
class SSNode:
    def __init__(self, original_node, node_id) -> None:
        self.node_id = node_id
        self.successors = {}
        self.predecessors = {}
        self.name = original_node.get_name() if original_node else ""
        self.original_user_names = []
        # -1 means not assigned
        self.stream_id = -1
        self.snode_names = []
        # to avoid cyclic import, we use hasattr instead of isinstance
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
            try:
                if original_node is not None:
                    for user in original_node.users:
                        self.original_user_names.append(user.get_name())
            except Exception as e:
                pass
    def get_name(self):
        return self.name
# StreamSchedulerGraph
class SSGraph:
    def __init__(self, nodes) -> None:
        self.ssnodes = []
        self.name_mapping = {}
        # It records the levels back from the OUTPUT node. {ssnode: level, }
        self.reverse_level = {}
        self.reverse_level_predecessors = {}
        self.critical_path = []
        self.build_graph(nodes)


    def build_graph(self, nodes):
        # breakpoint()
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

        # return

        for predecessor in output_node.predecessors.values():
            # only append the node that has only one successor OUTPUT
            if len(predecessor.successors) == 1:
                tmp_queue.append(predecessor)
        # breakpoint()
        while len(tmp_queue) != 0:
            cur_node = tmp_queue.pop(0)
            # if one of the successors is not assigned, then we cannot assign the level to cur_node
            for successor in cur_node.successors.values():
                if successor not in self.reverse_level:
                    if successor not in tmp_queue:
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

    def stream_pool_pop(self):
        return 0

    def dig_node(self, cur_node:SSNode):
        for predecessor in cur_node.predecessors.values():
            if predecessor.stream_id == -1:
                self.dig_node(predecessor)
        if cur_node.stream_id == -1:
            cur_node.stream_id = self.stream_pool_pop()
            log.info(f"assign stream id {cur_node.stream_id} to node {cur_node.get_name()}")
        # find one successor to assign it to critical path
        # path_to_output = {}
        # for successor in cur_node.successors.values():
        #     path_to_output[successor] = [successor,]
            # if successor.get_name() == "OUTPUT":
            #     continue
            # tmp_successor = successor.successors.values()[0]
            # while tmp_successor.get_name() != "OUTPUT":
            #     path_to_output[successor].append(tmp_successor)
            #     tmp_successor = tmp_successor.successors.values()[0]
        # find the latest common successor
        

            # get the number of trasitive successors between cur_node and the last common successor

    def stream_scheduling(self):
        self.dig_node(self.ssnodes[0])
    
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
        reset_log_path()


def stream_schedule(snodes):
    pass
    graph = SSGraph(snodes)
    graph.print_graph()