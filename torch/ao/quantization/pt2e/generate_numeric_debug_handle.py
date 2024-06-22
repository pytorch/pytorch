from torch.fx import GraphModule, Node

__all__ = ["generate_numeric_debug_handle", "NUMERIC_DEBUG_HANDLE_KEY"]

NUMERIC_DEBUG_HANDLE_KEY = "_numeric_debug_handle"

def generate_numeric_debug_handle(graph_module: GraphModule) -> None:
    unique_id = 0
    for node in graph_module.graph.nodes:
        if node.op == "call_function":
            node.meta[NUMERIC_DEBUG_HANDLE_KEY] = {}
            for arg in node.args:
                if isinstance(arg, Node):
                    node.meta[NUMERIC_DEBUG_HANDLE_KEY][arg] = unique_id
                    unique_id += 1

            node.meta[NUMERIC_DEBUG_HANDLE_KEY]["output"] = unique_id
            unique_id += 1
