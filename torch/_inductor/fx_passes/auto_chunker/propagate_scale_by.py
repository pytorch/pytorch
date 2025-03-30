import torch

from .core import get_chunking_meta, get_chunking_metas, CantChunk, update_chunking_meta
from .utils import get_args_of_node_type, format_node_with_chunking_meta, get_scale_by_from_node, get_scale_by_from_metas
import functools

aten = torch.ops.aten
prims = torch.ops.prims

propagate_rules = {
}

def _register_propagate_rule(aten_op, handler):
    if not isinstance(aten_op, (list, tuple)):
        aten_op = [aten_op]

    for op in aten_op:
        propagate_rules[op] = handler
    return handler

def register_propagate_rule(aten_op):
    return functools.partial(_register_propagate_rule, aten_op)

def propagate_scale_by(nodes_with_chunking_meta):
    """
    The input is a list of nodes that have chunking medadata.
    The nodes are already in topological order.
    """
    for node in nodes_with_chunking_meta:
        meta = get_chunking_meta(node)
    
        arg_nodes = get_args_of_node_type(node)
        arg_metas = get_chunking_metas(arg_nodes)

        if all(arg_meta is None for arg_meta in arg_metas):
            # should be graph input of the chunking subgraph
            continue

        print("Propagate scale_by:")
        format_node_with_chunking_meta(node, True) # TODO remove me
        assert all(arg_meta is not None for arg_meta in arg_metas)

        # None of the input has scale_by set
        if all(arg_meta.scale_by is None for arg_meta in arg_metas):
            continue

        target = node.target
        if target not in propagate_rules:
            raise CantChunk(f"Missing scale_by propagation rule for target {target}: {node.format_node()}")

        if not propagate_rules[target](node):
            raise CantChunk(f"scale_by propagate rule for {target} fail: {node.format_node()}")

@register_propagate_rule([
    aten.div.Tensor,
])
def propagate_div(div_node):
    lhs_node, rhs_node = div_node.args[:2]
    lhs_scale_by = get_scale_by_from_node(lhs_node)
    rhs_scale_by = get_scale_by_from_node(rhs_node)
    if lhs_scale_by and rhs_scale_by is None:
        update_chunking_meta(div_node, scale_by=lhs_scale_by)
        return True
    return False

@register_propagate_rule([
    aten.where.self,
])
def propagate_where(where_node):
    cond_node, true_node, false_node = where_node.args
    cond_meta, true_meta, false_meta = get_chunking_metas([cond_node, true_node, false_node])
    out_meta = get_chunking_meta(where_node)

    if true_meta.scale_by and not false_meta.scale_by:
        # the false_node must be all zero
        if false_node.target != aten.full.default:
            return False
        if false_node.args[1] != 0.0:
            return False
        out_meta.scale_by = true_meta.scale_by
        return True
    return False

@register_propagate_rule([
    aten.mul.Tensor,
    prims.convert_element_type.default,
    aten.sum.dim_IntList,
    aten.mm.default,
    aten.permute.default,
])
def propagate_general_copy(out_node):
    """
    A rule that holds for multiple ops: the scale_by of the output is
    set to the only scale_by of input nodes or None if no input has scale_by
    set.
    """
    args_node = get_args_of_node_type(out_node)
    args_meta = get_chunking_metas(args_node)
    out_meta = get_chunking_meta(out_node)

    scale_by = get_scale_by_from_metas(*args_meta)
    out_meta.scale_by = scale_by
    return True

@register_propagate_rule([
    aten.add.Tensor,
    aten.sub.Tensor,
])
def propagate_add_sub(out_node):
    """
    The scale_by node of the two arguments must be the same.
    """
    lhs_node, rhs_node = get_args_of_node_type(out_node)
    lhs_meta, rhs_meta = get_chunking_metas([lhs_node, rhs_node])
    if lhs_meta.scale_by is rhs_meta.scale_by:
        update_chunking_meta(out_node, scale_by=lhs_meta.scale_by)
        return True
    return False
