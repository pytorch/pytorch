from torch.fx._compatibility import compatibility
from torch.fx.graph_module import GraphModule
from torch._ops import OpOverload, OpOverloadPacket

@compatibility(is_backward_compatible=False)
def is_canonical(gm: GraphModule) -> bool:
    """
    Checks if a graph_module is "aten canonical"
    """

    for node in gm.graph.nodes:
        if node.op in {"placeholder", "output"}:
            continue

        if node.op == "call_method":
            # canonical aten graph couldn't have call_method nodes
            return False

        if node.op == "get_attr":
            return False

        if node.op == "call_module":
            continue

        if node.op == "call_function":
            if isinstance(node.target, OpOverload):
                if not node.target.is_functional:
                    return False

                # canonical aten graph cannot have private ops
                if node.target._namespace == "aten" and node.target._op_name[0] == "_":
                    return False

            elif isinstance(node.target, OpOverloadPacket):
                # canonical aten graph should have ops resolved to overloads
                return False
            elif node.target.__qualname__ == "getitem":
                continue
            else:
                return False

    return True
