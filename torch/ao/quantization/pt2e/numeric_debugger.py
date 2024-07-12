from torch.fx import GraphModule

__all__ = ["generate_numeric_debug_handle", "NUMERIC_DEBUG_HANDLE_KEY"]

NUMERIC_DEBUG_HANDLE_KEY = "_numeric_debug_handle"

def generate_numeric_debug_handle(graph_module: GraphModule) -> None:
    unique_id = 0
    for node in graph_module.graph.nodes:
        if node.op != "placeholder" and NUMERIC_DEBUG_HANDLE_KEY not in node.meta:
            node.meta[NUMERIC_DEBUG_HANDLE_KEY] = unique_id
            unique_id += 1
