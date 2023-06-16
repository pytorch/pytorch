import logging
log = logging.getLogger(__name__)
# StreamSchedulerNode
class SSNode:
    def __init__(self, original_node, node_id) -> None:
        self.original_node = original_node
        self.node_id = node_id
        self.successors = {}
        self.predecessors = {}
        self.name = original_node.get_name() if original_node else ""
    
    def get_name(self):
        return self.name
# StreamSchedulerGraph
class SSGraph:
    def __init__(self, nodes) -> None:
        self.ssnodes = []
        self.original_nodes = nodes
        self.name_mapping = {}
        self.buf_to_fx_node = {}
        self.build_graph()


    def build_graph(self):
        for node_id, node in enumerate(self.original_nodes):
            new_ssnode = SSNode(node, node_id)
            self.ssnodes.append(new_ssnode)
            self.name_mapping[node.get_name()] = new_ssnode
            # check if the node has a snodes attribute
            if hasattr(node, "snodes"):
            # if isinstance(node, FusedSchedulerNode):
                for snode in node.snodes:
                    self.name_mapping[snode.get_name()] = new_ssnode
        output_node = SSNode(None, len(self.original_nodes))
        output_node.name = "OUTPUT"
        self.name_mapping["OUTPUT"] = output_node
        def update_successor_predecessor(ssnode, users):
            for user in users:
                user_ssnode = self.name_mapping[user.get_name()]
                ssnode.successors[user_ssnode.get_name()] = user_ssnode
                user_ssnode.predecessors[ssnode.get_name()] = ssnode
        for ssnode in self.ssnodes:
            if not hasattr(ssnode.original_node, "snodes"):
                update_successor_predecessor(ssnode, ssnode.original_node.users)
            else:
                for snode in ssnode.original_node.snodes:
                    update_successor_predecessor(ssnode, snode.users)

    
    def print_graph(self):
        log.info("=====findhao debug=====")
        for node in self.ssnodes:
            log.info(node.get_name())
            log.info("\tsuccessors:")
            tmp_str = '\t\t'
            try:
                for successor in node.successors.values():
                    tmp_str += successor.get_name() + ', '
            except:
                pass
            log.info(tmp_str)
            if len(node.predecessors) != 0:
                log.info("\tpredecessors:")
                tmp_str = '\t\t'
                for predecessor in node.predecessors.values():
                    tmp_str += predecessor.get_name() + ', '
                log.info(tmp_str)
        log.info("=====findhao debug=====")
            


def stream_schedule(nodes):
    graph = SSGraph(nodes)
    graph.print_graph()