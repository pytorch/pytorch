from os import environ
from .virtualized import V
import logging
from collections import deque
log = logging.getLogger(__name__)
import torch
import os
import json
import sys
sys.setrecursionlimit(5000)  # by default it is 1000

HIGH_KERNEL_VOLUMN = 36864
import sys
sys.setrecursionlimit(5000)  # set the recursion limit to 5000

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
        fake_successors, fake_predecessors: {buf_name: SSNode}. It is used to set a node's fake successors when reordering.
        name: the name of the original node.
        original_user_names: the names of the original users of the node. If the original node is a fused node, we'll check the users of scheduler nodes included in this fused node.
        stream_id: the stream id of the node. -1 means not assigned.
        snode_names: the names of the scheduler nodes included in this fused node.
        cuda_event: mark if this node needs to generate a CUDA event. CUDA events are used to keep the data dependencies between streams.
        is_fused: mark if this node is a fused node.
        is_nop_node: mark if this node is a nop node.
        node_type: the type of the node. It can be "template", "extern", "foreach", "fused_or_schedule", "nop"
        kernel_volumn: the product of the node's arguments' size. It usually represents the number of kernel threads.
    """

    def __init__(self, original_node) -> None:
        from .scheduler import NopKernelSchedulerNode
        self.successors = {}
        self.predecessors = {}
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

class CheckPoints:
    def __init__(self, debug_path, ssgraph):
        if not os.path.exists(debug_path):
            os.makedirs(debug_path)
        self.debug_path = debug_path
        self.checkpoints_file = os.path.join(debug_path, "checkpoints.json")
        self.checkpoints = self.load()
        self.ss_graph = ssgraph
        self.graph_name = ssgraph.graph_name
        current_graph_checkpoints = self.checkpoints.get(self.graph_name, {})
        self.done_nodes = set(current_graph_checkpoints.get("done_nodes", []))
        self.cur_node_pre_suc_done = []
        # The node is being assigned this time. It is the direct predecessor and successor of a critical node.
        self.this_time_node = None
        self.stream_pool_pop = ssgraph.stream_pool_pop
        
    def check_graph_status(self):
        # check if no graph_names in self.checkpoints
        if len(self.checkpoints) == 0:
            self.checkpoints["cur_graph"] = self.graph_name
            self.checkpoints[self.graph_name] = {}
            if "all_graphs" not in self.checkpoints:
                self.checkpoints["all_graphs"] = []
            self.checkpoints["all_graphs"].append(self.graph_name)
            print("No graph in checkpoints, start from {}".format(self.graph_name))
        else:
            next_graph = None
            if "all_graphs" not in self.checkpoints:
                self.checkpoints["all_graphs"] = []
            if self.graph_name not in self.checkpoints["all_graphs"]:
                self.checkpoints[self.graph_name] = {}
                self.checkpoints["all_graphs"].append(self.graph_name)
            # check if all graphs are done
            for tmp_graph_name in self.checkpoints['all_graphs']:
                if 'done' not in self.checkpoints[tmp_graph_name]:
                    next_graph = tmp_graph_name
                    break
            if next_graph is None:
                self.checkpoints["finished"] = True
            else:
                if 'done' in self.checkpoints['cur_graph']:
                    self.checkpoints['cur_graph'] = next_graph
                
    def load(self):
        if not os.path.exists(self.checkpoints_file):
            open(self.checkpoints_file, "w").close()
            return {}
        if os.stat(self.checkpoints_file).st_size == 0:
            return {}
        else:
            with open(self.checkpoints_file, "r") as f:
                checkpoints = json.load(f)
            
            return checkpoints
    
    def save(self):
        if self.graph_name not in self.checkpoints:
            self.checkpoints[self.graph_name] = {}
        self.checkpoints[self.graph_name]["done_nodes"] = list(self.done_nodes)
        self.checkpoints[self.graph_name]["done_nodes"].sort()
        self.checkpoints[self.graph_name]["this_time_node"] = self.this_time_node.name if self.this_time_node else None
        with open(self.checkpoints_file, "w") as f:
            json.dump(self.checkpoints, f)

    def stream_assign(self):
        self.ss_graph.stream_pool[0] = len(self.ss_graph.ssnodes) + 2
        self.check_graph_status()
        if 'done' in self.checkpoints.get(self.graph_name, {}):
            for node in self.ss_graph.critical_path:
                self.dig_node(node, 0, True, False)
            return
        elif self.checkpoints.get('cur_graph', '') != self.graph_name:
            for node in self.ss_graph.critical_path:
                self.dig_node(node, 0, False, False)
            return

        this_time = None
        for node in self.ss_graph.critical_path:
            if node.name not in self.done_nodes:
                break
        else:
            self.checkpoints[self.graph_name]["done"] = True
            for node in self.ss_graph.ssnodes:
                node.stream_id = 0
            return
        # the first run generates all stream assignments.
        if self.checkpoints[self.graph_name].get("stream_assignment", None) is None:
            self.ss_graph.stream_assign()
            stream_assignment = {}
            for node in self.ss_graph.ssnodes:
                assert node.stream_id != -1, f"stream_id of {node.name} is -1."
                stream_assignment[node.name] = node.stream_id
                # reset it to avoid the accuracy testing failure caused by the first  run 
                node.stream_id = 0
            self.checkpoints[self.graph_name]["stream_assignment"] = stream_assignment
            return
        
        for node in self.ss_graph.critical_path:
            node.stream_id = 0
            if node.name in self.done_nodes:
                self.dig_node(node)
            elif this_time is None:
                self.dig_node(node, 0, False, True)
                this_time = node.name
                all_pre_suc = []
                all_pre_suc.extend(node.predecessors.values())
                all_pre_suc.extend(node.successors.values())
                for tmp_node in all_pre_suc:
                    if tmp_node not in self.ss_graph.critical_path and tmp_node.name not in self.done_nodes:
                        break
                else:
                    # all predecessors and successors are done for current node
                    self.done_nodes.add(node.name)
                    if self.this_time_node is None:
                        this_time = None
            else:
                self.dig_node(node, 0, False)
        

    
    def dig_node(self, cur_node: SSNode, level=0, assign_all=True, target_node = False):
        """
        For each critical node, we use this function to assign a stream id to all its predecessors and successors.
        target_node: if True, we need special process for all its predecessors and successors.
        """
        # find which node is being assigned this time. The first node after done nodes is the one.
        if level == 0 and target_node:
            self.this_time_node = None
            for tmp_node in cur_node.predecessors.values():
                if tmp_node.name in self.done_nodes:
                    self.cur_node_pre_suc_done.append(tmp_node)
                else:
                    self.this_time_node = tmp_node
            for tmp_node in cur_node.successors.values():
                if tmp_node in self.ss_graph.critical_path:
                    continue
                if tmp_node.name in self.done_nodes:
                    self.cur_node_pre_suc_done.append(tmp_node)
                elif not self.this_time_node:
                    self.this_time_node = tmp_node
            if self.this_time_node is None:
                print(f"Info: no valid prede- or suc- cessors for node {cur_node.name}")
                return
            self.done_nodes.add(self.this_time_node.name)

        for predecessor in cur_node.predecessors.values():
            if predecessor.stream_id == -1:
                # iterate all prede- and suc- cessors of the critical node
                if level == 0 and target_node:
                    if predecessor in self.cur_node_pre_suc_done or predecessor.name == self.this_time_node.name:
                        self.dig_node(predecessor, level + 1, True, target_node)
                    else:
                        self.dig_node(predecessor, level + 1, False, target_node)
                else:
                    self.dig_node(predecessor, level + 1, assign_all, target_node)
        if cur_node.stream_id == -1:
            if cur_node in self.ss_graph.critical_path:
                cur_node.stream_id = 0
            # if the node has only one predecessor and the predecessor has only one successor, then we can assign the same stream_id to the node
            elif len(cur_node.predecessors) == 1:
                predecessor = list(cur_node.predecessors.values())[0]
                if len(predecessor.successors) == 1:
                    if assign_all:
                        if "stream_assignment" in self.checkpoints[self.graph_name]:
                            assert self.checkpoints[self.graph_name]["stream_assignment"][cur_node.name] == self.checkpoints[self.graph_name]["stream_assignment"][predecessor.name], f"stream_id of {cur_node.name} is not equal to its predecessor {predecessor.name}."
                            cur_node.stream_id = self.checkpoints[self.graph_name]["stream_assignment"][cur_node.name]
                            self.ss_graph.stream_pool[cur_node.stream_id]+=1
                        else:
                            cur_node.stream_id = 0
                            assert self.checkpoints["cur_graph"] != self.graph_name, f"stream_assignment of {self.graph_name} is not found."
                    else:
                        cur_node.stream_id = 0
                else:
                    if assign_all:
                        if "stream_assignment" in self.checkpoints[self.graph_name]:
                            cur_node.stream_id = self.checkpoints[self.graph_name]["stream_assignment"][cur_node.name]
                            self.ss_graph.stream_pool[cur_node.stream_id]+=1
                        else:
                            cur_node.stream_id = 0
                            assert self.checkpoints["cur_graph"] != self.graph_name, f"stream_assignment of {self.graph_name} is not found."
                    else:
                        cur_node.stream_id = 0
            else:
                if assign_all:
                    if "stream_assignment" in self.checkpoints[self.graph_name]:
                        cur_node.stream_id = self.checkpoints[self.graph_name]["stream_assignment"][cur_node.name]
                        self.ss_graph.stream_pool[cur_node.stream_id]+=1
                    else:
                        cur_node.stream_id = 0
                        assert self.checkpoints["cur_graph"] != self.graph_name, f"stream_assignment of {self.graph_name} is not found."
                else:
                    cur_node.stream_id = 0
        for successor in cur_node.successors.values():
            if successor in self.ss_graph.critical_path:
                continue
            if successor.stream_id == -1:
                if level == 0 and target_node:
                    if successor in self.cur_node_pre_suc_done or successor.name == self.this_time_node.name:
                        self.dig_node(successor, level + 1, True, True)
                    else:
                        self.dig_node(successor, level + 1, False, True)
                else:
                    self.dig_node(successor, level + 1, assign_all, target_node)


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
        self.stream_pool_size = int(environ.get("STREAM_POOL_SIZE", 31))
        self.stream_pool = [0] * (self.stream_pool_size + 1)
        self.arg_to_stream = {}
        self.final_order = []
        self.max_stream_id = 0
        self.build_graph(nodes)

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
                    # fix nop node bug in super_slomo. there are corrosing references between two nop nodes.
                    nop_node_count = 0
                    # for successors in cur_node.successors.values():
                    #     if successors.is_nop_node:
                    #         nop_node_count += 1
                    if nop_node_count >= 2:
                        cur_node.stream_id = 0
                    else:
                        tmp_arg = None
                        if os.getenv("TORCHINDUCTOR_BYPASS_TINY", "0") == "1" and cur_node.kernel_volumn < HIGH_KERNEL_VOLUMN:
                            tmp_arg = predecessor
                        cur_node.stream_id = self.stream_pool_pop(tmp_arg)
            else:
                # fix nop node bug in super_slomo. there are corrosing references between two nop nodes.
                nop_node_count = 0
                # for successors in cur_node.successors.values():
                #     if successors.is_nop_node:
                #         nop_node_count += 1
                if nop_node_count >= 2:
                    cur_node.stream_id = 0
                else:
                    tmp_arg = None
                    if os.getenv("TORCHINDUCTOR_BYPASS_TINY", "0") == "1" and cur_node.kernel_volumn < HIGH_KERNEL_VOLUMN and len(cur_node.predecessors) != 0:
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

    
    def find_fake_predecessor(self, ssnode):
        # the first successor in the critical path
        common_successor = None
        tmp_node = ssnode
        tmp_queue = []
        tmp_queue.append(tmp_node)
        while len(tmp_queue) != 0:
            cur_node = tmp_queue.pop(0)
            if cur_node in self.critical_path:
                common_successor = cur_node
                break
            for successor in cur_node.successors.values():
                tmp_queue.append(successor)
        assert common_successor is not None
        target_type = ''
        if ssnode.node_type == 'extern':
            target_type = 'fused_or_schedule'
        elif ssnode.node_type == 'fused_or_schedule':
            target_type = 'extern'
        
        first_reverse_predecessor = None
        common_successor_index = self.critical_path.index(common_successor)
        if target_type == '' and common_successor_index != 1:
            first_reverse_predecessor = self.critical_path[common_successor_index-1]

        for index in range(common_successor_index-1, -1, -1):
            if self.critical_path[index].node_type == target_type:
                if index != 0:
                    first_reverse_predecessor = self.critical_path[index-1]
                break
        # could be None
        return first_reverse_predecessor

    """
    For current global order, the nodes off the critical path are usually the adjacent nodes before the next common successor. It will block the critical path if we don't reorder them.
    """
    def reorder(self, original_nodes):
        if environ.get("STREAMSCHEDULER_REORDER", "0") == "0":
            return original_nodes
        from .scheduler import SchedulerNode, FusedSchedulerNode
        in_degree = {ssnode: 0 for ssnode in self.ssnodes}
        for ssnode in self.ssnodes:
            node = ssnode.original_node
            for successor in ssnode.successors.values():
                in_degree[successor] += 1
            if node is None:
                continue
            if node.is_template():
                ssnode.node_type = "template"
            elif node.is_extern():
                ssnode.node_type = "extern"
            elif node.is_foreach():
                ssnode.node_type = "foreach"
            elif isinstance(node, (FusedSchedulerNode, SchedulerNode)):
                ssnode.node_type = "fused_or_schedule"
            else:
                ssnode.node_type = "nop"

        init_nodes = [ssnode for ssnode in self.ssnodes if in_degree[ssnode] == 0]
        init_nodes = sorted(init_nodes, key=lambda x: x.stream_id, reverse=True)
        for node in init_nodes:
            if node.stream_id != 0:
                fake_predecessor = self.find_fake_predecessor(node)
                if fake_predecessor:
                    node.fake_predecessors[fake_predecessor.get_name()] = fake_predecessor
                    fake_predecessor.fake_successors[node.get_name()] = node
                    in_degree[node] += 1
        init_nodes = [ssnode for ssnode in init_nodes if in_degree[ssnode] == 0]
        init_nodes = sorted(init_nodes, key=lambda x: x.stream_id, reverse=True)
        queue = deque(init_nodes)
        while queue:
            u = queue.popleft()
            self.final_order.append(u)
            sorted_fake_successors = sorted(u.fake_successors.values(), key=lambda x: x.stream_id, reverse=True)
            for fake_successor in sorted_fake_successors:
                in_degree[fake_successor] -= 1
                if in_degree[fake_successor] == 0:
                    queue.append(fake_successor)
            sorted_successors = sorted(u.successors.values(), key=lambda x: x.stream_id, reverse=True)
            for successor in sorted_successors:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)
        if len(self.final_order) != len(self.ssnodes):
            raise RuntimeError(f"Error when processing the queue. The queue is not empty after the loop.\nlen(self.final_order):{len(self.final_order)}\nlen(self.ssnodes):{len(self.ssnodes)}")
        self.final_order.remove(self.name_mapping["OUTPUT"])
        log.info("=====TorchInductor Stream Scheduler final order=====")
        for node in self.final_order:
            log.info(node.get_name())
        log.info("=====TorchInductor Stream Scheduler final order=====")
        new_node_list = []
        for ssnode in self.final_order:
            new_node_list.append(ssnode.original_node)
        return new_node_list

    def check_fingerprint(self, ssgraphs_from_file):
        """
        This function is used to check if there is matched ssgraph in the file. If found, directly load the stream assignment from the file rather than assigning the stream again.
        Args: 
            ssgraphs_from_file: {graph_name: {'buf0':{successors:[], predecessors:[], stream_id:0,},,,}}. It records all the ssgraphs from the file.
        """
        def two_list_equal(list1, list2):
            if len(list1) != len(list2):
                return False
            for item in list1:
                if item not in list2:
                    return False
            return True
        for graph_name, graph in ssgraphs_from_file.items():
            if graph_name.startswith("graph_"):
                if len(graph) != len(self.ssnodes):
                    continue
                for ssnode in self.ssnodes:
                    if ssnode.get_name() not in graph:
                        break
                else:
                    for ssnode in self.ssnodes:
                        # make sure the predecessors and successors are the same
                        if not (two_list_equal(ssnode.successors.keys(), graph[ssnode.get_name()]["successors"]) and two_list_equal(ssnode.predecessors.keys(), graph[ssnode.get_name()]["predecessors"])):
                            break
                    else:
                        for ssnode in self.ssnodes:
                            ssnode.stream_id = graph[ssnode.get_name()]["stream_id"]
                            self.stream_pool[ssnode.stream_id] += 1
                        self.event_assign()
                        return True
        else:
            return False


    def print_graph(self):
        import datetime
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
        debug_path=V.debug._path
        if not debug_path:
            debug_path = "/tmp/yhao/debug2023/"
            os.makedirs(debug_path, exist_ok=True)
        content = {}
        stream_assign_file = f"{debug_path}/resnet18_stream_assignment.json"
        if os.path.exists(stream_assign_file):
            with open(stream_assign_file, 'r') as fin:
                content = json.load(fin)
        else:
            content = {}
        keys = list(content.keys())
        # let's use graph_X as the key for different graphs
        graphs = [_ for _ in keys if _.startswith("graph_")]
        if len(graphs) == 0:
            graph_name = "graph_0"
        else:
            # get the max number of the graph
            max_graph = max([int(_.split("_")[1]) for _ in graphs])
            graph_name = f"graph_{max_graph+1}"
        content[graph_name] = {}
        content[graph_name]['order'] = []
        for ssnode in self.ssnodes:
            content[graph_name]['order'].append(ssnode.get_name())
            content[graph_name][ssnode.get_name()] = {}
            content[graph_name][ssnode.get_name()]["stream_id"] = ssnode.stream_id
            content[graph_name][ssnode.get_name()]["predecessors"] = list(ssnode.predecessors.keys())
            content[graph_name][ssnode.get_name()]["successors"] = list(ssnode.successors.keys())
        with open(stream_assign_file, 'w') as fout:
            json.dump(content, fout)
        reset_log_path = temporary_log_path(f'{debug_path}/stream_assignment_{current_time}.log')
        log.info("=====TorchInductor Stream Scheduler Tree=====")
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
        log.info("=====TorchInductor Stream Scheduler Tree end=====")
        log.info("=====TorchInductor Stream Scheduler Tree reverse level=====")

        for node in self.reverse_level.keys():
            log.info(f"{node.get_name()} {self.reverse_level[node]}")
            # also print the predecessors
            if node.get_name() != "OUTPUT":
                log.info(f"\tpredecessor:{self.reverse_level_predecessors[node].get_name()}")
        if len(self.reverse_level.keys()) != len(self.ssnodes):
            log.error("TorchInductor Stream Scheduler Tree reverse level error")
            log.error(f"len(self.reverse_level.keys()):{len(self.reverse_level.keys())}")
            log.error(f"len(self.ssnodes):{len(self.ssnodes)}")
            missing = self.reverse_level.keys() - self.ssnodes
            for node in missing:
                log.error(f"missing node:{node.get_name()}")
        log.info("=====TorchInductor Stream Scheduler Tree reverse level end=====")
        log.info("=====TorchInductor Stream Scheduler Tree critical path=====")
        for node in self.critical_path:
            log.info(f"{node.get_name()}")
        log.info("=====TorchInductor Stream Scheduler Tree critical path end=====")
        log.info("=====TorchInductor Stream Scheduler Tree stream allocation=====")
        for node in self.ssnodes:
            assert node.stream_id != -1, f"node {node.get_name()} stream_id is -1"
            if node.cuda_event:
                event_str = "cuda_event True"
            else:
                event_str = ""
            log.info(f"{node.get_name()} {node.stream_id} {node.kernel_volumn} {event_str}")
        log.info("=====TorchInductor Stream Scheduler Tree stream allocation end=====")

        reset_log_path()


def stream_schedule(snodes):
    ssgraph = SSGraph(snodes)
    accuracy_fix = os.getenv("TORCHINDUCTOR_STREAM_ACCURACY_FIX", "0") == "1"
    if accuracy_fix and not V.debug:
        raise RuntimeError("TORCHINDUCTOR_STREAM_ACCURACY_FIX is set to 1 but debug is not enabled")
    if accuracy_fix:
        ssgraph.graph_name = V.debug._path.split(os.sep)[-1]
        osenv_debug_folder = os.getenv("DEBUG_FOLDER")
        debug_path = f"{torch._dynamo.config.debug_dir_root}/{osenv_debug_folder}"
        checkpoints = CheckPoints(debug_path, ssgraph)
        checkpoints.load()
        checkpoints.stream_assign()
        ssgraph.event_assign()
        checkpoints.save()
    else:
        # load updated stream assignment. It saves the json file path to import.
        load_existing_stream_assignment = os.getenv("TORCHINDUCTOR_LOAD_EXISTING_STREAM_ASSIGNMENT", None)
        success = False
        if load_existing_stream_assignment is not None:
            content = {}
            with open(load_existing_stream_assignment, 'r') as fin:
                content = json.load(fin)
            success = ssgraph.check_fingerprint(content)
            if not success:
                # TODO: the performance and accuracy mode could be different. when does the accuracy test, the saved stream assignment may be not valid.
                log.warning("Failed to load existing stream assignment. Reassign the stream.")
        if not success:
            ssgraph.stream_assign()
            ssgraph.event_assign()
    if os.getenv("TORCHINDUCTOR_STREAM_PRINT_GRAPH", "0") == "1" and V.debug:
        ssgraph.print_graph()
    V.graph.stream_graph = ssgraph
    return ssgraph

