import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
toq = torch.ops.quantized
from torch.fx import GraphModule, map_arg
from torch.fx.graph import Node, Graph
from torch.quantization.ns.graph_matcher import (
    get_matching_node_pairs,
    get_type_a_related_to_b,
    _getattr_from_fqn,  # TODO: update name, make reusable
    _print_node,  # TODO: remove this
)
from torch.quantization.fx.quantize import is_activation_post_process
from torch.quantization.fx.utils import get_new_attr_name_with_prefix

from typing import Dict, Tuple, Callable, List


def _get_conv_mod_weight(mod: nn.Module) -> torch.Tensor:
    # TODO(future PR): make more generic, handle everything
    if isinstance(mod, nn.Conv2d):
        return mod.weight.detach()
    else:
        return mod._weight_bias()[0]  # type: ignore

def _get_linear_fun_weight(node: Node, gm: GraphModule) -> torch.Tensor:
    # TODO(before land): better docblock, with example FX IR
    if node.target in (F.linear,):
        # traverse backwards from the weight arg, accounting for
        # any observers
        weight_arg_node = node.args[1]
        # _print_node(weight_arg_node)
        assert isinstance(weight_arg_node, Node)
        weight_node = weight_arg_node.args[0]
        # _print_node(weight_node)
        # TODO(before land): currently assumes 1 observer, handle arbitrary
        # levels of observation, from 0 to N
        assert isinstance(weight_node, Node)
        assert weight_node.op == 'get_attr'
        weight = _getattr_from_fqn(gm, weight_node.target)  # type: ignore
        return weight.detach()

    else:
        assert node.target in (toq.linear,)
        # packed weight is arg 1
        packed_weight_node = node.args[1]
        assert isinstance(packed_weight_node, Node)
        assert packed_weight_node.op == 'get_attr'
        packed_weight = _getattr_from_fqn(gm, packed_weight_node.target)  # type: ignore
        # TODO(future PR): why does packed_weight.unpack() not work?
        # TODO(future PR): discuss if we even need to unpack, or if the
        #   caller can handle the unpacking
        (weight, _bias), _name = packed_weight.__getstate__()
        return weight


# Note: this is not a user facing API
# TODO(future PR): wrap this in a user facing API which does not
#   expose FX types.
def compare_weights(
    name_a: str,
    gm_a: GraphModule,
    name_b: str,
    gm_b: GraphModule,
) -> Dict[str, Dict[str, torch.Tensor]]:
    type_a_related_to_b = get_type_a_related_to_b()
    matched_node_pairs = get_matching_node_pairs(gm_a, gm_b)

    results = {}

    for match_name, match in matched_node_pairs.items():

        node_a, node_b = match
        assert node_a.op == node_b.op and \
            node_a.op in ('call_function', 'call_module')

        if node_a.op == 'call_function':

            # linear
            # TODO(before land): fix the in checks everywhere
            # TODO(before land): other function types
            a_related_to_linear = node_a.target in (F.linear,) or \
                (node_a.target, F.linear) in type_a_related_to_b

            if a_related_to_linear:
                weight_a = _get_linear_fun_weight(node_a, gm_a)
                weight_b = _get_linear_fun_weight(node_b, gm_b)

                results[match_name] = {
                    name_a: weight_a,
                    name_b: weight_b,
                }

        else:  # call_module
            # for call_module, we need to look up the modules to do the type check
            assert isinstance(node_a.target, str)
            mod_a = _getattr_from_fqn(gm_a, node_a.target)
            assert isinstance(node_b.target, str)
            mod_b = _getattr_from_fqn(gm_b, node_b.target)

            # check that A is one the modules we need
            # assume B is related (this is done by graph matcher)
            a_related_to_conv2d_mod = isinstance(mod_a, nn.Conv2d) or \
                (type(mod_a), nn.Conv2d) in type_a_related_to_b

            # TODO(before land): other module types
            if a_related_to_conv2d_mod:
                weight_a = _get_conv_mod_weight(mod_a)
                weight_b = _get_conv_mod_weight(mod_b)
                results[match_name] = {
                    name_a: weight_a,
                    name_b: weight_b,
                }

    return results

def _remove_observers_add_loggers(
    gm: GraphModule,
    nodes_to_instrument: List[Node],
    logger_cls: Callable,
) -> GraphModule:

    new_graph = Graph()
    env: Dict[str, Any] = {}
    modules = dict(gm.named_modules())

    def load_arg(a):
        return map_arg(a, lambda node: env[node.name])

    for node in gm.graph.nodes:
        if node.op == 'output':
            new_graph.output(map_arg(node.args[0], load_arg))
            continue

        if node.op == 'call_module' and is_activation_post_process(modules[node.target]):
            # remove activation post process node
            env[node.name] = env[node.args[0].name]

        elif node in nodes_to_instrument:
            # ensure env is populated with base node
            env[node.name] = new_graph.node_copy(node, load_arg)
            # add the logger after the base node
            env[node.name] = _insert_logger_after_node(
                env[node.name], gm, logger_cls, '_ns_logger_')

        else:
            env[node.name] = new_graph.node_copy(node, load_arg)

    new_gm = GraphModule(gm, new_graph)
    return new_gm

def _insert_domain_translation_after_node(
    node_a: Node,
    node_c: Node,
    prev_node_c: Node,
    gm_a: GraphModule,
    gm_b: GraphModule,
    node_name_prefix: str,
) -> Node:
    """
    Given a starting graph C (derived from graph B) of

    ... -> prev_node_c -> node_c -> ...

    And a corresponding related node_a, inserts the correct domain
    translation node after prev_node_c to translate into the domain expected
    by node_a, resulting in:

                          domain_translation
                        /
    ... -> prev_node_c -> node_c -> ...

    For example, if node_c is an int8 op and node_a is an fp32 op, this function
    will insert a dequant.
    """

    # look up what node A and node B represent
    # TODO(implement)

    # translate the inputs from domain of B to domain of A
    # i.e. add a dequant, etc
    # TODO(before land): make this generic with a mapping
    domain_translation_op = torch.dequantize

    new_domain_translation_name = \
        get_new_attr_name_with_prefix(node_name_prefix)(gm_b)
    return prev_node_c.graph.create_node(
        'call_function', domain_translation_op, (prev_node_c,), {},
        new_domain_translation_name)


def _insert_logger_after_node(
    node: Node,
    gm: GraphModule,
    logger_cls: Callable,
    logger_node_name_suffix: str,
) -> Node:
    """
    Given a starting graph of

    prev_node -> node -> next_node

    This function creates a new logger_cls obj and adds it
    after node, resulting in

    prev_node -> node -> logger_obj -> next_node
    """
    # create new name
    logger_node_name = \
        get_new_attr_name_with_prefix(node.name + logger_node_name_suffix)(gm)
    # create the logger object
    logger_obj = logger_cls(node.name)
    # attach the logger object to the parent module
    setattr(gm, logger_node_name, logger_obj)
    logger_node = node.graph.create_node(
        'call_module', logger_node_name, (node,), {})
    return logger_node

def _insert_copy_of_node_a_after_domain_translation_node_in_graph_c(
    domain_translation_node: Node,
    node_a: Node,
    gm_a: GraphModule,
    gm_b: GraphModule,
    graph_c: Graph,
    node_name_prefix: str,
) -> Node:
    """
    Input subgraphs from A and C (derived from B):

    ... -> node_a -> ...

          domain_translation
        /
    ... -> node_c -> ...

    Output subgraph of C:

          domain_translation -> node_a_copy
        /
    ... -> node_c -> ...

    Copies the module pointed to by node_a to gm_b. Creates a new node in gm_b
    corresponding to applying node_a to the output of domain_translation.
    Returns the newly created node.
    """
    if node_a.op == 'call_module':

        assert node_a.op == 'call_module'
        new_mod_copy_name = \
            get_new_attr_name_with_prefix(node_name_prefix)(gm_b)
        # fetch the corresponding module from gm_a
        assert isinstance(node_a.target, str)
        mod_a = _getattr_from_fqn(gm_a, node_a.target)
        # TODO(future PR): stop using copy.deepcopy, keep just a single
        # instance of each tensor around
        setattr(gm_b, new_mod_copy_name, copy.deepcopy(mod_a))
        new_node = graph_c.create_node(
            'call_module', new_mod_copy_name, (domain_translation_node,), {})
        return new_node

    else:

        assert node_a.op == 'call_function'

        # copy bias
        # _print_node(node_a.kwargs['bias'])
        bias_a_copy_name = \
            get_new_attr_name_with_prefix(node_a.kwargs['bias'].name + '_shadow_copy_')(gm_b)
        bias_a_obj = _getattr_from_fqn(gm_a, node_a.kwargs['bias'].target)
        setattr(gm_b, bias_a_copy_name, bias_a_obj.detach())
        bias_copy_node = graph_c.create_node(
            'get_attr', bias_a_copy_name, (), {}, bias_a_copy_name)

        # copy weight
        # for now, assume F.linear and observed
        # TODO(before land): handle generically for all types of F.linear
        # and other ops
        weight_a_obs = node_a.args[1]
        weight_a = weight_a_obs.args[0]
        weight_a_copy_name = \
            get_new_attr_name_with_prefix(weight_a.name + '_shadow_copy_')(gm_b)
        weight_a_obj = _getattr_from_fqn(gm_a, weight_a.target)
        setattr(gm_b, weight_a_copy_name, weight_a_obj)
        weight_copy_node = graph_c.create_node(
            'get_attr', weight_a_copy_name, (), {}, weight_a_copy_name)

        # create the copy of node_a in B
        node_a_shadows_c_name = \
            get_new_attr_name_with_prefix(node_name_prefix)(gm_b)
        node_a_shadows_c = graph_c.create_node(
            node_a.op, node_a.target,
            (domain_translation_node, weight_copy_node,),
            {'bias': bias_copy_node},
            node_a_shadows_c_name)

        return node_a_shadows_c

def _create_a_shadows_b(
    gm_a: GraphModule,
    gm_b: GraphModule,
    matched_node_pairs: Dict[str, Tuple[Node, Node]],
    logger_cls: Callable,
) -> GraphModule:
    """
    Creates a new GraphModule consisting of the graph of C, with the meaningful
    nodes of A shadowing the corresponding nodes of B.  For example,

    Graph A:
    a0 -> op0_fp32 -> a1 -> op1_fp32 -> a2

    Graph B:
    b0 -> op0_int8 -> b1 -> op1_int8 -> b2

    matched_node_pairs: {'op0': (op0_fp32, op0_int8), 'op1': (op1_fp32, op1_int8)}

    Graph C (A shadows B):

        / dequant0 -> op0_fp32 -> logger_a_0  / dequant_1 -> op1_fp32 -> logger_a_1
       /                                     /
    b0 -------------> op0_int8 -> logger_b_0 --------------> op1_int8 -> logger_b_1

    In a nutshell, this function does the following for each node pair:
    * copies the necessary attributes and modules from gm_a to gm_b,
      keeping names unique
    * adds a domain translation op (dequant, quant, etc)
    * adds a copy of node_a in gm_b's graph
    * adds loggers to the outputs of node_a and node_b
    """

    # graph_c is the graph created from copying the nodes of graph_b and inserting
    # the shadows with the nodes copied from graph_a
    graph_c = Graph()
    env_c: Dict[str, Any] = {}
    modules = dict(gm_b.named_modules())

    def load_arg(a):
        return map_arg(a, lambda node: env_c[node.name])

    nodes_to_instrument_b_to_a = {}
    for match_name, (node_a, node_b) in matched_node_pairs.items():
        nodes_to_instrument_b_to_a[node_b] = node_a

    for node_b in gm_b.graph.nodes:
        if node_b.op == 'output':
            graph_c.output(map_arg(node_b.args[0], load_arg))
            continue

        if node_b.op == 'call_module' and is_activation_post_process(modules[node_b.target]):
            # remove activation post process node
            env_c[node_b.name] = env_c[node_b.args[0].name]

        elif node_b in nodes_to_instrument_b_to_a:
            node_a = nodes_to_instrument_b_to_a[node_b]
            if False:
                print('b')
                _print_node(node_b)
                print('a')
                _print_node(node_a)

            # ensure env_c is populated with base node
            env_c[node_b.name] = graph_c.node_copy(node_b, load_arg)
            node_c = env_c[node_b.name]
            # after this point,
            # node_a is the original node from graph_a, with parent module gm_a
            # node_b is the original node from graph_b, with parent module gm_b
            # node_c is the copy of node_b in graph_c

            # translate the inputs from domain of B to domain of A (dequant, etc)
            domain_translation_node = _insert_domain_translation_after_node(
                node_a, node_c, node_c.args[0], gm_a, gm_b, node_b.name + '_domain_translation_')
            env_c[domain_translation_node.name] = domain_translation_node

            # hook up the new mod_a copy to be in the graph, receiving the
            # same inputs as mod_b does, with domain translated to match a
            # TODO(before land): handle args and kwargs generically
            node_a_shadows_c = \
                _insert_copy_of_node_a_after_domain_translation_node_in_graph_c(
                    env_c[domain_translation_node.name],
                    node_a, gm_a, gm_b, graph_c, node_c.name + '_shadow_copy_')
            env_c[node_a_shadows_c.name] = node_a_shadows_c

            # hook up a logger to the mod_a copy
            env_c[node_a_shadows_c.name] = _insert_logger_after_node(
                env_c[node_a_shadows_c.name], gm_b, logger_cls, '_ns_logger_a_')

            # hook up a logger to the mod_b copy
            env_c[node_b.name] = _insert_logger_after_node(
                env_c[node_b.name], gm_b, logger_cls, '_ns_logger_b_')

        else:
            env_c[node_b.name] = graph_c.node_copy(node_b, load_arg)

    gm_c = GraphModule(gm_b, graph_c)
    return gm_c

# TODO(future PR): add name, etc to NS Loggers and reuse them
class OutputLogger(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.stats = []
        self.name = name

    def forward(self, x):
        self.stats.append(x.detach())
        return x

# TODO(future PR): consider calling this once for each input model (Haixin's suggestion)
def prepare_model_outputs(
    name_a: str,
    gm_a: GraphModule,
    name_b: str,
    gm_b: GraphModule,
    logger_cls: Callable,
) -> Tuple[GraphModule, GraphModule]:

    matched_node_pairs = get_matching_node_pairs(gm_a, gm_b)

    nodes_to_instrument_a = []
    nodes_to_instrument_b = []
    for match_name, (node_a, node_b,) in matched_node_pairs.items():
        print(match_name, node_a, node_b)

        # TODO(before land): do not observe pairs of nodes we do not care
        #   about (both fp32, denylist, etc)

        nodes_to_instrument_a.append(node_a)
        nodes_to_instrument_b.append(node_b)

    gm_a = _remove_observers_add_loggers(gm_a, nodes_to_instrument_a, logger_cls)
    gm_b = _remove_observers_add_loggers(gm_b, nodes_to_instrument_b, logger_cls)

    return (gm_a, gm_b)

def prepare_model_with_stubs(
    name_a: str,
    gm_a: GraphModule,
    name_b: str,
    gm_b: GraphModule,
    logger_cls: Callable,
) -> GraphModule:
    matched_node_pairs = get_matching_node_pairs(gm_a, gm_b)
    gm_a_shadows_b = _create_a_shadows_b(gm_a, gm_b, matched_node_pairs, logger_cls)
    return gm_a_shadows_b
