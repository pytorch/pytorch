from torch.fx import GraphModule, Node

__all__ = ["generate_numeric_debug_handle"]


def generate_numeric_debug_handle(graph_module: GraphModule) -> None:
    unique_id = 0
    for node in graph_module.graph.nodes:
        if node.op == "call_function":
            node.meta["numeric_debug_handle"] = {}
            for arg in node.args:
                if isinstance(arg, Node):
                    node.meta["numeric_debug_handle"][arg] = unique_id
                    unique_id += 1

            node.meta["numeric_debug_handle"]["output"] = unique_id
            unique_id += 1
