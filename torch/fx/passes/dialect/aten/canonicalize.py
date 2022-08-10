from torch.fx.graph_module import GraphModule
from torch._ops import OpOverload, OpOverloadPacket

def is_canonical(gm: GraphModule) -> bool:

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
            func = None
            if isinstance(node.target, OpOverload):
                if node.target._overloadname == "out":
                    # canonical aten graph cannot have out variant ops
                    return False
                func = node.target.overloadpacket
            elif isinstance(node.target, OpOverloadPacket):
                func = node.target
            elif node.target.__qualname__ == "getitem":
                continue
            else:
                return False

            # TODO: canonical aten graph cannot have private ops
            # if func.__name__[0] == "_":
            #     return False

            # canonical aten graph cannot have inplace ops
            # TODO: this is a hacky way to check if an op is inplace op
            if func.__name__[-1] == "_":
                return False

    return True
