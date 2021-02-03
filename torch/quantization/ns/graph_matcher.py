import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.quantized as nnq
toq = torch.ops.quantized

from torch.fx import GraphModule
from torch.fx.graph import Graph, Node

from typing import Dict, Tuple, List, Optional, Set, Callable

# TODO(before land): delete this
def _print_node(node: Optional[Node]) -> None:
    if node is None:
        print(None)
    else:
        print(
            node, ', target:', node.target, ', op:', node.op,
            ', args:', node.args, ', kwargs:', node.kwargs)

def _get_output_nodes(g: Graph) -> List[Node]:
    return [n for n in g.nodes if n.op == 'output']

def get_type_a_related_to_b() -> Set[Tuple[Callable, Callable]]:
    # TODO(future PR): allow customizations
    # TODO(future PR): reuse existing quantization mappings
    type_a_related_to_b: Set[Tuple[Callable, Callable]] = set([

        # conv related ops
        (nn.Conv2d, nnq.Conv2d,),
        # TODO(future PR): add all the other flavors of conv
        # (1d, 3d, qat modules, fused modules, fq, etc)

        # linear related ops
        (F.linear, toq.linear,),

        # TODO(future PR): other ops
    ])

    # make the mapping bidirectional
    reverse_mapping = set()
    for k in type_a_related_to_b:
        type_a, type_b = k
        reverse_mapping.add((type_b, type_a))
    type_a_related_to_b.update(reverse_mapping)

    return type_a_related_to_b

# Note: the other thing we will need for prepare_model_stubs
# is a conversion function from inputs of node A to inputs of node B
# i.e. {
#        (F.linear, toq.linear):
#          lambda x, scale, zp: torch.quantize_per_tensor(x, scale, zp),
#        (toq.linear, F.linear):
#          lambda x: x.dequantize(),
#        ...
#      }
#
# This can probably be implemented mostly with heuristics

class _NSGraphMatchableNodesIterator:
    """
    Iterates through the graph of gm, starting with the output nodes
    and continuing backwards.
    1. Returns meaningful nodes, in order
    2. Skips over non-meaningful nodes
    """
    def __init__(self, gm: GraphModule):
        self.gm: GraphModule = gm
        self.seen_nodes: Set[Node] = set()
        self.stack: List[Node] = []
        for start_node in _get_output_nodes(self.gm.graph):
            self.stack.append(start_node)

    def __iter__(self):
        return self

    def __next__(self) -> Node:
        """
        Returns the next meaningful node.
        """
        while len(self.stack) > 0:
            cur_node = self.stack.pop()
            if cur_node in self.seen_nodes:
                continue
            self.seen_nodes.add(cur_node)
            # add args of previous nodes to stack
            # TODO(future PR): handle kwargs as needed
            for arg in cur_node.args:
                if isinstance(arg, Node):
                    self.stack.append(arg)
                # TODO(future PR): handle other arg types such as Tuple, etc

            # skip observers, etc
            if not self._is_meaningful(cur_node):
                continue

            return cur_node

        raise StopIteration

    def _is_meaningful(self, node: Node) -> bool:
        is_meaningful = False
        if node.op == 'call_function':
            is_meaningful = True
            # TODO(future PR): make sure quant/dequant calls are not useful, more generic
            if node.target in (torch.quantize_per_tensor,):
                is_meaningful = False
        elif node.op == 'call_module':
            is_meaningful = True
            assert isinstance(node.target, str)
            target_module = getattr(self.gm, node.target)
            # TODO(future PR): more generic, skip other nodes, etc
            if isinstance(target_module, torch.quantization.ObserverBase):
                is_meaningful = False
        return is_meaningful

class GraphMatchingException(Exception):
    """
    Exception raised when two graphs cannot be matched.
    """
    pass

def _node_a_related_to_b(
    node_a: Node,
    node_b: Node,
    gm_a: GraphModule,
    gm_b: GraphModule,
    type_a_related_to_b: Set[Tuple[Callable, Callable]],
) -> bool:
    if node_a.op != node_a.op:
        # for now, comparing call_module to call_function is not supported
        # this can be added later if needed
        return False

    if node_a.op not in ('call_module', 'call_function'):
        # only call_module and call_function make sense for this use case
        return False

    if node_a.op == 'call_function':
        key = (node_a.target, node_b.target)
        return key in type_a_related_to_b
    elif node_a.op == 'call_module':
        # for call_module, we need to look up the modules to do the type check
        assert isinstance(node_a.target, str)
        mod_a = getattr(gm_a, node_a.target)
        assert isinstance(node_b.target, str)
        mod_b = getattr(gm_b, node_b.target)
        key = (type(mod_a), type(mod_b))
        return key in type_a_related_to_b
    return False

def _get_name_for_node_pair(
    node_a: Node,
    node_b: Node,
) -> str:
    if node_b.op == 'call_module':
        assert isinstance(node_b.target, str)
        return node_b.target
    # for now, use node name.
    # TODO(future PR): find a better solution
    return node_b.name

def get_matching_node_pairs(
    gm_a: GraphModule,
    gm_b: GraphModule,
) -> Dict[str, Tuple[Node, Node]]:
    """
    Matches meaningful nodes of graph_a to graph_b.

    For a node, "meaningful" is defined as a node which is not an observer,
    fake_quants, quant or dequant.

    A pair of nodes is "related" if both nodes represent the same mathematical
    operation across different quantization flavors. For example,
    `F.linear` and `torch.ops.quantized.linear` are related, and
    `F.linear` and `torch.nn.Conv` are not related.

    TODO(before land): align on naming for "meaningful" and "related" and "flavors".

    For each meaningful pair of nodes node_a and node_b, they will match
    if node_a and node_b are related.

    For graphs A and B, they will match iff:
    1. the number of meaningful nodes in A and B is equivalent
    2. when iterating through the meaningful nodes of A and B in the same order, each
       corresponding pair of nodes is related.

    Practically, this enables us to find the corresponding nodes between
    graphs of related models.  For example, if we had two graphs such as:

    graph_a: x0 -> conv_0 (type: nn.Conv2d) -> obs_0 -> x1
             w  -/
             b  -/

    graph_b: x0 -> quant_0 -> conv_0 (type: nnq.Conv2d) -> dequant_0 -> x1
           packed_params_0 -/

    This function will return the following result:
    {
        'conv_0': (  # the name of the node in graph_b
          conv_0,  # the Node object from graph A
          conv_0,  # the Node object from graph B
        ),
    }

    """
    graph_a_iterator = _NSGraphMatchableNodesIterator(gm_a)
    graph_b_iterator = _NSGraphMatchableNodesIterator(gm_b)
    results = {}
    type_a_related_to_b = get_type_a_related_to_b()

    while True:
        cur_node_a, cur_node_b = None, None

        # fetch the next node from a
        try:
            cur_node_a = next(graph_a_iterator)
        except StopIteration:
            pass

        # fetch the next node from b
        try:
            cur_node_b = next(graph_b_iterator)
        except StopIteration:
            pass

        # now that we have two candidate nodes, check if they match
        if cur_node_a is None and cur_node_b is None:
            # we reached the end of both graphs
            break

        elif cur_node_a is not None and cur_node_b is not None:
            # there is a candidate match
            if not _node_a_related_to_b(cur_node_a, cur_node_b, gm_a, gm_b, type_a_related_to_b):
                # matching error
                # TODO(future PR): more descriptive error message
                raise GraphMatchingException()
            key_name = _get_name_for_node_pair(cur_node_a, cur_node_b)
            results[key_name] = (cur_node_a, cur_node_b)
            continue

        else:
            # matching error
            # TODO(future PR): more descriptive error message
            raise GraphMatchingException()

    return results
