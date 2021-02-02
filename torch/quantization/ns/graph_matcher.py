import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.quantized as nnq
toq = torch.ops.quantized

from torch.fx import GraphModule
from torch.fx.graph import Graph, Node

from typing import Dict, Tuple, List, Optional

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

class NSGraphMatchableNodesIterator:
    """
    Iterates through the graph of gm, starting with the output nodes
    and continuing backwards.
    1. Yields nodes which could potentially be matched to other graphs
       (such as linear, conv, etc).
    2. Skips over nodes which are not useful for matching across graphs
       (such as observers, fake_quants, etc).
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
        Returns the next meaningful node, or StopIteration if the
        graph's nodes have been exhausted.
        """
        while len(self.stack) > 0:
            cur_node = self.stack.pop()
            if cur_node in self.seen_nodes:
                continue
            self.seen_nodes.add(cur_node)
            # add args of previous nodes to stack
            # TODO(future?): do we need kwargs?
            for arg in cur_node.args:
                self.stack.append(arg)

            # skip observers, etc
            if not self._is_matchable(cur_node):
                continue

            return cur_node

        raise StopIteration

    def _is_matchable(self, node: Node) -> bool:
        """
        Returns whether this node should be considered for matching
        """
        is_matchable = False
        if node.op == 'call_function':
            is_matchable = True
            # TODO: make sure quant/dequant calls are not useful, more generic
            if node.target in (torch.quantize_per_tensor,):
                is_matchable = False
        elif node.op == 'call_module':
            is_matchable = True
            target_module = getattr(self.gm, node.target)
            # TODO: more generic, skip other nodes, etc
            if isinstance(target_module, torch.quantization.ObserverBase):
                is_matchable = False
        return is_matchable

class GraphMatchingException(Exception):
    """
    Exception raised when two graphs cannot be matched.
    """
    pass

# TODO(future PR): reuse existing quantization mappings
# TODO(before land): fix naming
type_a_related_to_b = set([

    # conv related ops
    (nn.Conv2d, nnq.Conv2d,),
    # TODO(future PR): add all the other flavors of conv
    # (1d, 3d, qat modules, fused modules, fq, etc)

    # linear related ops
    (F.linear, toq.linear,),
])
# make the mapping bidirectional
reverse_mapping = set()
for k in type_a_related_to_b:
    type_a, type_b = k
    reverse_mapping.add((type_b, type_a))
type_a_related_to_b.update(reverse_mapping)

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

def is_match(
    node_a: Node,
    node_b: Node,
    gm_a: GraphModule,
    gm_b: GraphModule,
) -> bool:
    """
    Returns True if node_a and node_b represent the same kind of op
    TODO: define a word for this
    i.e. True for matching F.linear with quantized.linear

    Note: gm_a and gm_b are only required to be able to look up module instances
    for call_module nodes.
    """
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
        mod_a = getattr(gm_a, node_a.target)
        mod_b = getattr(gm_b, node_b.target)
        key = (type(mod_a), type(mod_b))
        return key in type_a_related_to_b
    return False


def get_matching_node_pairs(
    gm_a: GraphModule,
    gm_b: GraphModule,
) -> Dict[str, Tuple[Node, Node]]:
    """
    Matches nodes of graph_a to graph_b. Rules:
    * nodes must be in the same corresponding areas of the graphs to match
    * nodes must be either of the same type (F.linear and F.linear) or
      of a related type (F.linear and quantized.ops.linear, etc) to match
    * observers, fake_quants, quants, dequants are ignored by the matching
      logic

    TODO: finish this docblock
    """

    graph_a, graph_b = gm_a.graph, gm_b.graph

    graph_a_iterator = NSGraphMatchableNodesIterator(gm_a)
    graph_b_iterator = NSGraphMatchableNodesIterator(gm_b)

    results = {}
    # for now, use a dummy name
    # TODO(before land): real name
    cur_idx = -1

    while True:
        cur_idx += 1
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

        print('node_a')
        _print_node(cur_node_a)
        print('node_b')
        _print_node(cur_node_b)

        # check for terminal conditions
        if cur_node_a is None and cur_node_b is None:
            # we reached the end of both graphs
            break

        elif cur_node_a is not None and cur_node_b is not None:
            # there is a candidate match
            matches = is_match(cur_node_a, cur_node_b, gm_a, gm_b)
            if not matches:
                # matching error
                # TODO(future PR): more descriptive error message
                raise GraphMatchingException()
            print('matches', matches)
        else:
            # matching error
            # TODO(future PR): more descriptive error message
            raise GraphMatchingException()

        results[cur_idx] = (cur_node_a, cur_node_b)

    return results
