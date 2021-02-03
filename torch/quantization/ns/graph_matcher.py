import operator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.quantized as nnq
import torch.nn.qat as nnqat
import torch.nn.intrinsic.qat as nniqat
toq = torch.ops.quantized

from torch.fx import GraphModule
from torch.fx.graph import Graph, Node

from typing import Dict, Tuple, List, Optional, Set, Callable, Any

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
    # TODO(future PR): add the rest of modules and ops here
    sets_of_related_ops: List[Set[Callable]] = [
        # conv modules
        set([
            nn.Conv2d,
            nnq.Conv2d,
            nnqat.Conv2d,
            # Note: matching weights may not work with nniqat.ConvBn2d directly
            # leaving that as a problem for a future PR to solve.
            nniqat.ConvBn2d,
        ]),
        # linear modules
        set([
            nn.Linear,
            nnq.Linear,
            nnqat.Linear,
        ]),
        # linear functionals
        set([
            F.linear,
            toq.linear,
        ]),
        # add
        set([
            torch.add,
            toq.add,
            operator.add,  # x + y
        ]),
    ]

    type_a_related_to_b: Set[Tuple[Callable, Callable]] = set()

    for s in sets_of_related_ops:
        s_list = list(s)
        # add every bidirectional pair
        for idx_0 in range(0, len(s_list) - 1):
            for idx_1 in range(idx_0 + 1, len(s_list)):
                type_a_related_to_b.add((s_list[idx_0], s_list[idx_1]))
                type_a_related_to_b.add((s_list[idx_1], s_list[idx_0]))

    return type_a_related_to_b

def get_non_matchable_functions() -> Set[Callable]:
    """
    `call_function` nodes pointing to these functions are non-matchable.
    """
    # TODO(future PR): allow customizations
    return set([
        torch.quantize_per_tensor,
    ])

def get_non_matchable_modules() -> Set[Callable]:
    """
    `call_module` nodes pointing to instances of these types are non-matchable.
    """
    # TODO(future PR): allow customizations
    return set([
        torch.quantization.ObserverBase,
        torch.quantization.FakeQuantizeBase,
    ])

def _getattr_from_fqn(gm: GraphModule, fqn: str) -> Any:
    """
    Given a gm and a fqn such as "foo.bar.baz", returns gm.foo.bar.baz.
    """
    fqn_parts = fqn.split(".")
    cur_val = gm
    for part in fqn_parts:
        cur_val = getattr(cur_val, part)
    return cur_val

class _NSGraphMatchableNodesIterator:
    """
    Iterates through the graph of gm, starting with the output nodes
    and continuing backwards.
    1. Returns matchable nodes, in order
    2. Skips over non-matchable nodes
    """
    def __init__(
        self,
        gm: GraphModule,
        non_matchable_functions: Set[Callable],
        non_matchable_modules: Set[Callable],
    ):
        self.gm: GraphModule = gm
        self.non_matchable_functions: Set[Callable] = non_matchable_functions
        self.non_matchable_modules: Set[Callable] = non_matchable_modules
        self.seen_nodes: Set[Node] = set()
        self.stack: List[Node] = []
        for start_node in _get_output_nodes(self.gm.graph):
            self.stack.append(start_node)

    def __iter__(self):
        return self

    def __next__(self) -> Node:
        """
        Returns the next matchable node.
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
            if not self._is_matchable(cur_node):
                continue

            return cur_node

        raise StopIteration

    def _is_matchable(self, node: Node) -> bool:
        if node.op == 'call_function':
            return not (node.target in self.non_matchable_functions)
        elif node.op == 'call_module':
            assert isinstance(node.target, str)
            # target_mod = getattr(self.gm, node.target)
            target_mod = _getattr_from_fqn(self.gm, node.target)
            return not \
                any(isinstance(target_mod, t)  # type: ignore
                    for t in self.non_matchable_modules)
        else:
            return False

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

    if node_a.op == 'call_function':
        if node_a.target == node_b.target:
            # nodes with equivalent targets always match (i.e. F.linear and F.linear)
            return True
        key = (node_a.target, node_b.target)
        return key in type_a_related_to_b
    elif node_a.op == 'call_module':
        # for call_module, we need to look up the modules to do the type check
        assert isinstance(node_a.target, str)
        mod_a = _getattr_from_fqn(gm_a, node_a.target)
        assert isinstance(node_b.target, str)
        mod_b = _getattr_from_fqn(gm_b, node_b.target)
        # modules with equivalent types always match (i.e. nn.Conv2d and nn.Conv2d)
        if type(mod_a) == type(mod_b):
            return True
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

def _get_node_target_type(node: Node, gm: GraphModule) -> Optional[Callable]:
    if node.op == 'call_function':
        return node.target  # type: ignore
    elif node.op == 'call_module':
        assert isinstance(node.target, str)
        mod = _getattr_from_fqn(gm, node.target)
        return type(mod)
    return None

def get_matching_node_pairs(
    gm_a: GraphModule,
    gm_b: GraphModule,
) -> Dict[str, Tuple[Node, Node]]:
    """
    Matches matchable nodes of graph_a to graph_b.

    For a node, "matchable" is defined as a node which is not an observer,
    fake_quants, quant or dequant.

    A pair of nodes is "related" if both nodes represent the same mathematical
    operation across different quantization flavors. For example,
    `F.linear` and `torch.ops.quantized.linear` are related, and
    `F.linear` and `torch.nn.Conv` are not related.

    For each matchable pair of nodes node_a and node_b, they will match
    if node_a and node_b are related.

    For graphs A and B, they will match iff:
    1. the number of matchable nodes in A and B is equivalent
    2. when iterating through the matchable nodes of A and B in the same order, each
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
    non_matchable_functions = get_non_matchable_functions()
    non_matchable_modules = get_non_matchable_modules()
    graph_a_iterator = _NSGraphMatchableNodesIterator(
        gm_a, non_matchable_functions, non_matchable_modules)
    graph_b_iterator = _NSGraphMatchableNodesIterator(
        gm_b, non_matchable_functions, non_matchable_modules)
    results = {}
    type_a_related_to_b = get_type_a_related_to_b()

    while True:
        # fetch the next nodes from a and b
        cur_node_a, cur_node_b = None, None
        try:
            cur_node_a = next(graph_a_iterator)
        except StopIteration:
            pass
        try:
            cur_node_b = next(graph_b_iterator)
        except StopIteration:
            pass

        # TODO(before land): remove
        if False:
            print('a')
            _print_node(cur_node_a)
            print('b')
            _print_node(cur_node_b)

        # look up types of a and b for useful error messages
        type_a, type_b = None, None
        if cur_node_a is not None:
            type_a = _get_node_target_type(cur_node_a, gm_a)
        if cur_node_b is not None:
            type_b = _get_node_target_type(cur_node_b, gm_b)

        # check for results and determine what to do next
        if cur_node_a is not None and cur_node_b is not None:
            # both nodes were fetched, check for relatedness
            if not _node_a_related_to_b(cur_node_a, cur_node_b,
                                        gm_a, gm_b, type_a_related_to_b):
                msg = f"({cur_node_a}, {type_a}) and ({cur_node_b}, {type_b}) are not related"
                raise GraphMatchingException(msg)
            key_name = _get_name_for_node_pair(cur_node_a, cur_node_b)
            results[key_name] = (cur_node_a, cur_node_b)
            continue
        elif cur_node_a is None and cur_node_b is None:
            # we reached the end of both graphs
            break
        else:
            # only one node was fetched, no match possible, throw error
            msg = f"Matchable nodes count mismatch: ({cur_node_a}, {type_a}) and ({cur_node_b}, {type_b})"
            raise GraphMatchingException(msg)

    return results
