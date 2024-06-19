import torch

aten = torch.ops.aten
prims = torch.ops.prims

def fuse_scatter_upon_allzero(graph):
    return # TODO
    for cur_node in graph.nodes:
        if cur_node.op != "call_function":
            continue
        if cur_node.target.overloadpacket is not aten.full:
            continue

        full = cur_node
        if len(full.users) != 1:
            continue

        nd = next(iter(full.users))

        if nd.target.overloadpacket is not aten.scatter:
            continue

        scatter = nd
        if len(scatter.users) != 1:
            continue

        nd = next(iter(scatter.users))

        # Is it necessary to be more general here
        if nd.target.overloadpacket is not aten.mul:
            continue

        mul = nd
        mul_other_input = mul.args[1] if scatter is mul.args[0] else mul.args[0]

        # TODO: prims.convert_element_type is optional
        if len(mul.users) != 1:
            continue

        nd = next(iter(mul.users))

        if nd.target.overloadpacket is not prims.convert_element_type:
            continue

        cvt = nd

        if len(cvt.users) != 2:
            continue

        nds = list(cvt.users)
        if nds[0].target.overloadpacket is not aten.sum or nds[1].target.overloadpacket is not aten.sub:
            continue

        sum_nd, sub = nds

        # replace sum_nd

        # replace sub
        # TODO
