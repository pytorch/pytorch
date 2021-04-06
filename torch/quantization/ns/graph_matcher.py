import enum
import operator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
import torch.nn.qat as nnqat
import torch.nn.intrinsic.quantized as nniq
import torch.nn.intrinsic.qat as nniqat
import torch.nn.intrinsic as nni
toq = torch.ops.quantized

from torch.fx import GraphModule
from torch.fx.graph import Graph, Node

from .utils import getattr_from_fqn
from .ns_types import NSSubgraph

from typing import Dict, Tuple, List, Optional, Set, Callable, Any, Union

def _get_output_nodes(g: Graph) -> List[Node]:
    return [n for n in g.nodes if n.op == 'output']

def get_base_name_to_sets_of_related_ops() -> Dict[str, Set[Callable]]:
    base_name_to_sets_of_related_ops: Dict[str, Set[Callable]] = {
        # conv modules
        'torch.nn.Conv1d': set([
            nn.Conv1d,
            nnq.Conv1d,
            nniqat.ConvBn1d,
            nniq.ConvReLU1d,
            nni.ConvReLU1d,
        ]),
        'torch.nn.Conv2d': set([
            nn.Conv2d,
            nnq.Conv2d,
            nnqat.Conv2d,
            nniqat.ConvBn2d,
            nniq.ConvReLU2d,
            nni.ConvReLU2d,
        ]),
        'torch.nn.Conv3d': set([
            nn.Conv3d,
            nnq.Conv3d,
            nnqat.Conv3d,
            nniqat.ConvBn3d,
            nniq.ConvReLU3d,
            nni.ConvReLU3d,
        ]),
        # linear modules
        'torch.nn.Linear': set([
            nn.Linear,
            nnq.Linear,
            nnqat.Linear,
            nnqd.Linear,
        ]),
        # linear functionals
        'torch.nn.functional.linear': set([
            F.linear,
            toq.linear,
            toq.linear_relu,
        ]),
        # LSTM
        'torch.nn.LSTM': set([
            nn.LSTM,
            nnqd.LSTM,
        ]),
        # add
        'torch.add': set([
            torch.add,
            toq.add,
            operator.add,  # x + y
        ]),
        # cat
        'torch.cat': set([
            torch.cat,
            toq.cat,
        ]),
        # mul
        'torch.mul': set([
            torch.mul,
            toq.mul,
        ]),
    }
    return base_name_to_sets_of_related_ops

def get_type_a_related_to_b(
    base_name_to_sets_of_related_ops: Dict[str, Set[Callable]],
) -> Set[Tuple[Callable, Callable]]:
    # TODO(future PR): allow customizations
    # TODO(future PR): reuse existing quantization mappings
    # TODO(future PR): add the rest of modules and ops here
    type_a_related_to_b: Set[Tuple[Callable, Callable]] = set()

    for base_name, s in base_name_to_sets_of_related_ops.items():
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

NSFusionElType = Union[
    Callable,  # call_function or call_module type, example: F.linear or nn.Conv2d
    str,  # call_method name, example: "dequantize"
    Tuple[str, Any],  # call_method name and first argument, example: ("to", torch.float16)
]
NSFusionType = Union[
    Tuple[NSFusionElType, NSFusionElType],
    Tuple[NSFusionElType, NSFusionElType, NSFusionElType, NSFusionElType],
]

def get_reversed_fusions() -> Set[Tuple[NSFusionType, int]]:
    """
    Set of potential fusions, in reverse order.  The order is reversed
    to match how fusion patterns are defined in quantization code.

    Fusion format:
    ((fusion_op_0, fusion_op_1), base_op_idx)

    Where base_op_idx is the idx of the op we should use to match other related
    ops. Note: base_op_idx is specified in non-reverse order, i.e. a base_op_idx
    of 0 represents the first op in regular (non-reverse) order, 1 represents the
    second op, etc.
    """
    # TODO(future PR): remove the custom syntax for defining fusion patterns
    # and reuse either quantization's syntax or something else.
    return set([
        ((F.relu, F.linear), 0),
        ((nn.ReLU, nn.Conv1d), 0),
        ((nn.ReLU, nn.Conv2d), 0),
        ((nn.ReLU, nn.Conv3d), 0),
        # linear-relu fp16 emulation:
        # fp16_to_fp32 -> linear -> relu -> fp32_to_fp16
        ((("to", torch.float16), F.relu, F.linear, "dequantize"), 1),
    ])

# TODO(future PR): we should see if we can reuse quantization's fusion
# patterns here.
def end_node_matches_reversed_fusion(
    end_node: Node,
    reversed_fusion: NSFusionType,
    gm: GraphModule,
) -> bool:
    """
    Returns true if a pattern ending with `end_node` matches
    the fusion pattern.
    """
    cur_node = end_node
    for fusion_idx in range(len(reversed_fusion)):
        cur_fusion_el = reversed_fusion[fusion_idx]

        if cur_node.op == 'call_function':
            fusion_el_is_fun = (not isinstance(cur_fusion_el, str)) and \
                (not isinstance(cur_fusion_el, type))
            if fusion_el_is_fun:
                if cur_node.target != cur_fusion_el:
                    return False
                if len(cur_node.args) > 0 and isinstance(cur_node.args[0], Node):
                    cur_node = cur_node.args[0]
                else:
                    return False
            else:
                return False

        elif cur_node.op == 'call_module':
            fusion_el_is_mod = isinstance(cur_fusion_el, type)
            if fusion_el_is_mod:
                assert isinstance(cur_node.target, str)
                target_mod = getattr_from_fqn(gm, cur_node.target)
                if not isinstance(cur_fusion_el, type):
                    return False
                if not isinstance(target_mod, cur_fusion_el):
                    return False
                if len(cur_node.args) > 0 and isinstance(cur_node.args[0], Node):
                    cur_node = cur_node.args[0]
                else:
                    return False
            else:
                return False

        elif cur_node.op == 'call_method':
            fusion_el_is_meth_with_second_arg = \
                isinstance(cur_fusion_el, tuple) and len(cur_fusion_el) == 2
            fusion_el_is_meth_without_args = isinstance(cur_fusion_el, str)
            if fusion_el_is_meth_without_args or fusion_el_is_meth_with_second_arg:
                if fusion_el_is_meth_without_args:
                    if cur_node.target != cur_fusion_el:
                        return False
                else:
                    assert isinstance(cur_fusion_el, tuple)
                    if cur_node.target != cur_fusion_el[0]:
                        return False
                    elif len(cur_node.args) < 2:
                        return False
                    elif cur_node.args[1] != cur_fusion_el[1]:
                        return False

                if len(cur_node.args) > 0 and isinstance(cur_node.args[0], Node):
                    cur_node = cur_node.args[0]
                else:
                    return False
            else:
                return False
        else:
            return False

    return True


class _NSGraphMatchableSubgraphsIterator:
    """
    Iterates through the graph of gm, starting with the output nodes
    and continuing backwards.
    1. Returns matchable subgraphs, in order. A subgraph is defined by
       (start_node, end_node).
    2. Skips over non-matchable subgraphs
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

    def __next__(self) -> NSSubgraph:
        """
        Returns the next matchable subgraph.
        """
        while len(self.stack) > 0:
            cur_end_node = self.stack.pop()
            if cur_end_node in self.seen_nodes:
                continue

            # for subgraphs which are single nodes, start_node == end_node
            # for subgraphs with more than one node, start node != end_node
            cur_start_node = cur_end_node
            # Subgraphs like linear-relu have the base node as the start node.
            # Subgraphs like dequantize-linear-relu-to(torch.float16) have the
            #   base node as the second node.
            # The cur_base_op_node var will move to the actual node during
            #   the fusion matching later in this code block.
            cur_base_op_node = cur_end_node

            # Check for potential fusions. For now, we are greedy
            # and always skip all non-base nodes of a fusion.  For example,
            # if we match linear-relu backwards, we will always skip the
            # relu node and attempt to match the linear node.  This can
            # be made configurable later if needed.
            for _reverse_fusion_ops, base_op_idx in get_reversed_fusions():
                is_match = end_node_matches_reversed_fusion(
                    cur_end_node, _reverse_fusion_ops, self.gm)
                if is_match:
                    # navigate to the base node
                    for rev_fusion_idx in range(len(_reverse_fusion_ops) - 1):
                        self.seen_nodes.add(cur_start_node)
                        # for now, assume that there are no other nodes
                        # which need to be added to the stack
                        cur_start_node = cur_start_node.args[0]  # type: ignore
                        # if the base op index matches the current node, set it
                        rev_base_op_idx = \
                            len(_reverse_fusion_ops) - 2 - base_op_idx
                        if rev_fusion_idx == rev_base_op_idx:
                            cur_base_op_node = cur_start_node
                    break

            self.seen_nodes.add(cur_start_node)
            # add args of previous nodes to stack
            for arg in cur_start_node.all_input_nodes:
                self._recursively_add_node_arg_to_stack(arg)

            # skip observers, etc
            # note: this check is done on the start_node, i.e.
            # if we are matching linear-relu in reverse, this would do the matchable
            # check on the linear
            if not self._is_matchable(cur_base_op_node):
                continue

            return NSSubgraph(
                start_node=cur_start_node, end_node=cur_end_node,
                base_op_node=cur_base_op_node)

        raise StopIteration

    def _recursively_add_node_arg_to_stack(self, arg: Any) -> None:
        """
        Adds all of the nodes in this arg to the stack, properly navigating
        through list, dicts and tuples.
        """
        if isinstance(arg, Node):
            self.stack.append(arg)
        elif isinstance(arg, torch.fx.immutable_collections.immutable_list) or type(arg) is tuple:
            for inner_arg in arg:
                self._recursively_add_node_arg_to_stack(inner_arg)
        elif isinstance(arg, torch.fx.immutable_collections.immutable_dict):
            for key, value in arg.items():
                self._recursively_add_node_arg_to_stack(value)

    def _is_matchable(self, node: Node) -> bool:
        if node.op == 'call_function':
            return not (node.target in self.non_matchable_functions)
        elif node.op == 'call_module':
            assert isinstance(node.target, str)
            # target_mod = getattr(self.gm, node.target)
            target_mod = getattr_from_fqn(self.gm, node.target)
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

class SugraphTypeRelationship(enum.Enum):
    # same type
    # example: F.linear and toq.linear, or nn.Conv2d and nn.Conv2d
    EQUAL = enum.auto()
    # same subgraph_relationship set, but not the same type
    # example: F.linear and toq.linear
    RELATED_BUT_NOT_EQUAL = enum.auto()
    # not related
    NOT_RELATED = enum.auto()

def _get_subgraph_relationship_type(
    subgraph_a: NSSubgraph,
    subgraph_b: NSSubgraph,
    gm_a: GraphModule,
    gm_b: GraphModule,
    type_a_related_to_b: Set[Tuple[Callable, Callable]],
) -> SugraphTypeRelationship:
    node_a = subgraph_a.base_op_node
    node_b = subgraph_b.base_op_node

    # TODO(next): make this code handle matching by what is before the base op
    if node_a.op != node_b.op:
        # for now, comparing call_module to call_function is not supported
        # this can be added later if needed
        return SugraphTypeRelationship.NOT_RELATED

    if node_a.op == 'call_function':
        if node_a.target == node_b.target:
            node_a_has_prev = subgraph_a.base_op_node == subgraph_a.start_node
            node_b_has_prev = subgraph_b.base_op_node == subgraph_b.start_node
            if node_a_has_prev and (not node_b_has_prev):
                return SugraphTypeRelationship.RELATED_BUT_NOT_EQUAL
            elif (not node_a_has_prev) and node_b_has_prev:
                return SugraphTypeRelationship.RELATED_BUT_NOT_EQUAL
            elif (not node_a_has_prev) and (not node_b_has_prev):
                return SugraphTypeRelationship.EQUAL
            else:
                # TODO(future PR): check for matches start_op_node and base_op_node
                return SugraphTypeRelationship.EQUAL

        key = (node_a.target, node_b.target)
        if key in type_a_related_to_b:
            return SugraphTypeRelationship.RELATED_BUT_NOT_EQUAL
        else:
            return SugraphTypeRelationship.NOT_RELATED
    elif node_a.op == 'call_module':
        assert (subgraph_a.base_op_node == subgraph_a.start_node and
                subgraph_b.base_op_node == subgraph_b.start_node), \
            "Matching call_module patterns where base_op_node != start_node is not supported yet"
        # for call_module, we need to look up the modules to do the type check
        assert isinstance(node_a.target, str)
        mod_a = getattr_from_fqn(gm_a, node_a.target)
        assert isinstance(node_b.target, str)
        mod_b = getattr_from_fqn(gm_b, node_b.target)
        # modules with equivalent types always match (i.e. nn.Conv2d and nn.Conv2d)
        if type(mod_a) == type(mod_b):
            return SugraphTypeRelationship.EQUAL
        key = (type(mod_a), type(mod_b))
        if key in type_a_related_to_b:
            return SugraphTypeRelationship.RELATED_BUT_NOT_EQUAL
        else:
            return SugraphTypeRelationship.NOT_RELATED
    return SugraphTypeRelationship.NOT_RELATED

def _get_name_for_subgraph(
    subgraph_a: NSSubgraph,
    gm_a: GraphModule,
    base_name_to_sets_of_related_ops: Dict[str, Set[Callable]],
    existing_names: Set[str],
) -> str:
    """
    Returns a unique name for a subgraph. This name is based on two things:
    1. the name of the set containing the underlying type of the base op in the
       subgraph (i.e. 'torch.nn.functional.linear' if this is related to a linear op)
    2. the number of previous subgraphs with related underlying type of the base op

    For example, in the graph

    linear0 -> relu0 -> linear1 -> relu1

    The subgraphs are (linear0, relu0) and (linear1, relu1).  If we iterate
    from the output node backwards, the name given to (linear1, relu1) will be
    `base_op_torch.nn.functional.linear_0`, and the name given to (linear0, relu0)
    will be `base_op_torch.nn.functional.linear_1`.

    Why are we not just using the node name? Answer: because of two requirements:
    A. fusions must be supported
    B. some Numeric Suite APIs can be called without having all of the models in memory

    For example, let's say we need to match nodes of

    (1) ... -> linear0 -> relu0 -> ...

    And

    (2) ... -> linear_relu0 -> ...

    Without being able to inspect them together. With the current naming scheme, if
    we iterate through both of these graphs in the same order, and assuming the rest
    of the graphs match, both of these subgraphs will get the same name without
    (1) and (2) knowing anything about each other.
    """
    target_type = _get_node_target_type(subgraph_a.base_op_node, gm_a)
    target_base_type = None
    for base_name, sets_of_related_ops in base_name_to_sets_of_related_ops.items():
        if target_type in sets_of_related_ops:
            target_base_type = base_name
    target_base_name = 'base_op_' + str(target_base_type)
    counter = 0
    proposed_name = target_base_name + '_' + str(counter)
    while proposed_name in existing_names:
        counter += 1
        proposed_name = target_base_name + '_' + str(counter)
    existing_names.add(proposed_name)
    return proposed_name

def _get_node_target_type(node: Node, gm: GraphModule) -> Optional[Callable]:
    if node.op == 'call_function':
        return node.target  # type: ignore
    elif node.op == 'call_module':
        assert isinstance(node.target, str)
        mod = getattr_from_fqn(gm, node.target)
        return type(mod)
    return None

def get_matching_subgraph_pairs(
    gm_a: GraphModule,
    gm_b: GraphModule,
) -> Dict[str, Tuple[NSSubgraph, NSSubgraph]]:
    """
    Matches matchable subgraphs of graph_a to graph_b.

    For a node, "matchable" is defined as a node which is not an observer,
    fake_quants, quant or dequant.

    A subgraph can contain one or more nodes.  A subgraph is matchable if
    at least one node inside of it is matchable.  Currently, all nodes in
    a subgraph must be matchable (because we assume no observers will be
    inserted in the middle of a fusion).

    A subgraph is defined by (start_node, end_node).  We assume that only
    start_node and end_node are linked with the surrounding graph, all other
    nodes in a subgraph are self-contained.

    A pair of nodes is "related" if both nodes represent the same mathematical
    operation across different quantization flavors. For example,
    `F.linear` and `torch.ops.quantized.linear` are related, and
    `F.linear` and `torch.nn.Conv` are not related.

    For each matchable pair of nodes node_a and node_b, they will match
    if node_a and node_b are related.

    For graphs A and B, they will match iff:
    1. the number of matchable subgraphs in A and B is equivalent
    2. when iterating through the matchable subgraphs of A and B in the same order, each
       corresponding pair of base nodes is related.

    This enables us to find the corresponding subgraphs between
    graphs of related models.  For example, if we had two graphs such as:

    graph_a: x0 -> conv_0 (type: nn.Conv2d) -> obs_0 -> x1
             w  -/
             b  -/

    graph_b: x0 -> quant_0 -> qconv_0 (type: nnq.Conv2d) -> dequant_0 -> x1
           packed_params_0 -/

    This function will return the following result:
    {
        'conv_0': (  # the name of the node in graph_b
          (conv_0, conv_0),  # (start_node_a, end_node_a)
          (qconv_0, qconv_0),  # (start_node_b, end_node_b)
        ),
    }

    Or, if we have a fusion pattern,

    graph_a: x0 -> linear_0 -> relu_0 -> obs_0 -> x1
             w  -/
             b  -/

    graph_b: x0 -> quant_0 -> linear_relu_0 -> dequant_0 -> x1
           packed_params_0 -/

    This function will return the following result:
    {
        'linear_relu_0': (  # the name of the node in graph_b
          (linear_0, relu_0),  # (start_node_a, end_node_a)
          (linear_relu_0, linear_relu_0),  # (start_node_b, end_node_b)
        ),
    }
    """
    non_matchable_functions = get_non_matchable_functions()
    non_matchable_modules = get_non_matchable_modules()
    graph_a_iterator = _NSGraphMatchableSubgraphsIterator(
        gm_a, non_matchable_functions, non_matchable_modules)
    graph_b_iterator = _NSGraphMatchableSubgraphsIterator(
        gm_b, non_matchable_functions, non_matchable_modules)
    results = {}
    base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
    type_a_related_to_b = \
        get_type_a_related_to_b(base_name_to_sets_of_related_ops)

    existing_names_a: Set[str] = set()
    existing_names_b: Set[str] = set()

    while True:
        # fetch the next subgraphs from a and b
        cur_subgraph_a, cur_subgraph_b = None, None
        try:
            cur_subgraph_a = next(graph_a_iterator)
        except StopIteration:
            pass
        try:
            cur_subgraph_b = next(graph_b_iterator)
        except StopIteration:
            pass

        # look up types of a and b for useful error messages
        type_start_a, type_start_b = None, None
        if cur_subgraph_a is not None:
            type_start_a = _get_node_target_type(cur_subgraph_a.start_node, gm_a)  # type: ignore
        if cur_subgraph_b is not None:
            type_start_b = _get_node_target_type(cur_subgraph_b.start_node, gm_b)  # type: ignore

        # check for results and determine what to do next
        if cur_subgraph_a is not None and cur_subgraph_b is not None:
            # both nodes were fetched, check for subgraph_relationship
            # note: subgraph_relationship is checked on the start node, i.e.
            # if a linear-relu pattern is checked, we would check for subgraph_relationship
            # of the linear
            subgraph_relationship = _get_subgraph_relationship_type(
                cur_subgraph_a, cur_subgraph_b,
                gm_a, gm_b, type_a_related_to_b)
            if subgraph_relationship == SugraphTypeRelationship.NOT_RELATED:
                msg = f"""
({cur_subgraph_a}, {type_start_a}) and
({cur_subgraph_b}, {type_start_b}) are not related"""
                raise GraphMatchingException(msg)
            elif subgraph_relationship == SugraphTypeRelationship.EQUAL:
                # For now, skip nodes with equal types. In the future, this can
                # be made configurable.
                continue
            key_name_a = _get_name_for_subgraph(
                cur_subgraph_a, gm_a, base_name_to_sets_of_related_ops,
                existing_names_a)
            key_name_b = _get_name_for_subgraph(
                cur_subgraph_b, gm_b, base_name_to_sets_of_related_ops,
                existing_names_b)
            assert key_name_a == key_name_b, \
                f"Subgraph names {key_name_a} and {key_name_b} do not match"
            results[key_name_a] = (cur_subgraph_a, cur_subgraph_b)
            continue
        elif cur_subgraph_a is None and cur_subgraph_b is None:
            # we reached the end of both graphs
            break
        else:
            # only one node was fetched, no match possible, throw error
            msg = f"""
Matchable nodes count mismatch: ({cur_subgraph_a}, {type_start_a}) and
({cur_subgraph_b}, {type_start_b})"""
            raise GraphMatchingException(msg)

    return results
