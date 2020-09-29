from typing import List
from torch.fx.graph_module import GraphModule
from typing import Any
from torch.fx.node import Node

"""find_use is used to find out if the node is another node's arg or kwargs."""
def find_use(arg: Any, node: Node) -> bool:
    if isinstance(arg, (tuple, list)):
        return any(find_use(elem, node) for elem in arg)
    elif isinstance(arg, dict):
        return any(find_use(v, node) for k, v in arg.items())
    elif isinstance(arg, slice):
        return any([find_use(arg.start, node), find_use(arg.stop, node), find_use(arg.step, node)])
    elif isinstance(arg, Node):
        return arg is node
    else:
        return False

def get_all_users_of(fx_module: GraphModule, index: int) -> List[int]:
    """Given the graph(fx_module) and an index, return a list of all node indexes that use this node"""
    graph = fx_module.graph
    current_node = graph.nodes[index]
    user_indexes: List[int] = []
    """if the node A is in node B's args, then B is the user of A
       go through all the nodes, if the input node in any node's args,
       then that node is the input node's user
    """
    for i, n in enumerate(graph.nodes):
        if find_use(n.args, current_node) or find_use(n.kwargs, current_node):
            user_indexes.append(i)
    return user_indexes
