from typing import Dict, List, NamedTuple
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, Target, map_arg
from torch.fx.graph import Graph
import torch
from torch.fx.experimental.shape_prop import ShapeProp

def replace_target_nodes_with(
    fx_module: GraphModule,
    old_op: str,
    old_target: Target,
    new_op: str,
    new_target: Target,
):
    """Modifies all nodes in fx_module.graph.nodes which match the specified op code and target,
    and updates them to match the new op code and target"""
    new_graph = Graph()
    val_map : Dict[Node, Node] = {}
    for node in fx_module.graph.nodes:
        if node.op == old_op and node.target == old_target:
            args = map_arg(node.args, lambda n: val_map[n])
            kwargs = map_arg(node.kwargs, lambda n: val_map[n])
            assert isinstance(args, tuple)
            assert isinstance(kwargs, dict)
            val_map[node] = new_graph.create_node(new_op, new_target, args, kwargs, node.name)
        else:
            val_map[node] = new_graph.node_copy(node, lambda n : val_map[n])
    fx_module.graph = new_graph

class size_bytes(NamedTuple):
    output_size: int
    total_size: int

def get_size_of_all_nodes(fx_module: GraphModule, args: List[torch.Tensor]) -> None:
    """Given a fx graph module, update each node with its total size (weights + bias + output)
    and its output_size(output). For a non-module node, the total size is the output size.
    return total size"""
    # Mark shape and dtype for each node (node.shape and node.dtype)
    ShapeProp(fx_module).propagate(*args)
    # Calculate the total size of the whole fx graph
    total_size_of_graph = 0.0
    for node in fx_module.graph.nodes:
        if node.op == 'output':
            break
        node.size_bytes = get_size_of_node(fx_module, node)
    return

def get_size_of_node(fx_module: GraphModule, node: Node) -> size_bytes:
    """Given a node with node.dtype and node.shape, return its total size and its output size.
       total_size = weights + bias + output_size
    """
    # Total num of elements
    total_num_of_elems = 0
    # For a module, conside all parameters
    if node.op == 'call_module':
        submodule_dict = dict(fx_module.named_modules())
        submodule = submodule_dict[node.target]
        parameters = submodule.named_parameters()
        # Parameters are named tuples
        for name, p in parameters:
            total_num_of_elems += p.numel()
    # Don't forget the output size
    # node.shape is the shape of this node's output
    shape = getattr(node, 'shape', None)
    if shape:
        output_elem = shape.numel()
    else:
        raise RuntimeError('Node has no shape attr')
    total_num_of_elems += output_elem
    size_per_elem_bytes = 0
    dtype = getattr(node, 'dtype', None)
    if dtype:
        size_per_elem_bytes = torch.tensor([], dtype=dtype).element_size()
    else:
        raise RuntimeError('Node has no dtype attr')
    total_size = size_per_elem_bytes * total_num_of_elems
    output_size = size_per_elem_bytes * output_elem
    return size_bytes(output_size, total_size)
