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

            new_logger_name = \
                get_new_attr_name_with_prefix(node.name + '_ns_logger_')(gm)
            logger_obj = logger_cls(node.name)
            setattr(gm, new_logger_name, logger_obj)
            env[node.name] = new_graph.create_node(
                'call_module', new_logger_name, (node,), {})

        else:
            env[node.name] = new_graph.node_copy(node, load_arg)

    new_gm = GraphModule(gm, new_graph)
    return new_gm

# TODO: hook this up
def _add_logger_after_node(
    node: Node,
    gm: GraphModule,
    logger_cls: Callable,
    logger_obj_name: str,
    logger_node_name_suffix: str,
) -> Node:
    # create new name
    logger_node_name = \
        get_new_attr_name_with_prefix(node.name + logger_node_name_suffix)(gm)
    # create the logger object
    logger_obj = logger_cls(logger_obj_name)
    # attach the logger object to the parent module
    setattr(gm, logger_node_name, logger_obj)
    logger_node = node.graph.create_node(
        'call_module', logger_node_name, (node,), {})
    return logger_node


def _create_a_shadows_b(
    gm_a: GraphModule,
    gm_b: GraphModule,
    matched_node_pairs: Dict[str, Tuple[Node, Node]],
    logger_cls: Callable,
) -> GraphModule:

    new_graph = Graph()
    env: Dict[str, Any] = {}
    modules = dict(gm_b.named_modules())

    def load_arg(a):
        return map_arg(a, lambda node: env[node.name])

    nodes_to_instrument_b_to_a = {}
    for match_name, (node_a, node_b) in matched_node_pairs.items():
        nodes_to_instrument_b_to_a[node_b] = node_a

    for node_b in gm_b.graph.nodes:
        if node_b.op == 'output':
            new_graph.output(map_arg(node_b.args[0], load_arg))
            continue

        if node_b.op == 'call_module' and is_activation_post_process(modules[node_b.target]):
            # remove activation post process node
            env[node_b.name] = env[node_b.args[0].name]

        elif node_b in nodes_to_instrument_b_to_a:
            node_a = nodes_to_instrument_b_to_a[node_b]
            if False:
                print('b')
                _print_node(node_b)
                print('a')
                _print_node(node_a)

            # ensure env is populated with base node
            env[node_b.name] = new_graph.node_copy(node_b, load_arg)

            # Ensure that all of the inputs required to run A,
            # except for the input being shadowed, are copied to env B.
            # For now, just make it work for conv2d module.
            # In the future, extend this to work for all other things
            # we care about.

            # conv_2d module node example:
            # _1 , target: 1 , op: call_module , args: (quantize_per_tensor_1,) , kwargs: {}
            if node_b.op == 'call_module':

                # TODO(before land): make the code nice

                # translate the inputs from domain of B to domain of A
                # i.e. add a dequant, etc
                # TODO(before land): make this generic with a mapping
                domain_translation_op = torch.dequantize

                new_domain_translation_name = \
                    get_new_attr_name_with_prefix(node_b.name + '_domain_translation_')(gm_b)
                env[new_domain_translation_name] = new_graph.create_node(
                    'call_function', domain_translation_op, (load_arg(node_b.args[0]),), {},
                    new_domain_translation_name)

                # hook up the new mod_a copy to be in the graph, receiving the
                # same inputs as mod_b does, with domain translated to match a
                new_mod_copy_name = \
                    get_new_attr_name_with_prefix(node_b.name + '_shadow_copy_')(gm_b)

                # fetch the corresponding module from gm_a
                assert isinstance(node_a.target, str)
                mod_a = _getattr_from_fqn(gm_a, node_a.target)

                # TODO(future PR): stop using copy.deepcopy, keep just a single
                # instance of each tensor around
                setattr(gm_b, new_mod_copy_name, copy.deepcopy(mod_a))

                # TODO(before land): handle args and kwargs generically
                env[new_mod_copy_name] = new_graph.create_node(
                    'call_module', new_mod_copy_name, (env[new_domain_translation_name],), {})

                # hook up a logger to the mod_a copy
                # TODO(before land): reusable instead of copy-pasta
                new_logger_name = \
                    get_new_attr_name_with_prefix(node_b.name + '_ns_logger_a_')(gm_b)
                logger_obj = logger_cls(new_mod_copy_name)
                setattr(gm_b, new_logger_name, logger_obj)
                env[new_mod_copy_name] = new_graph.create_node(
                    'call_module', new_logger_name, (load_arg(env[new_mod_copy_name]),), {})

                # hook up a logger to the mod_b copy
                # TODO(before land): reusable instead of copy-pasta
                new_logger_name = \
                    get_new_attr_name_with_prefix(node_b.name + '_ns_logger_b_')(gm_b)
                logger_obj = logger_cls(node_b.name)
                setattr(gm_b, new_logger_name, logger_obj)
                env[node_b.name] = new_graph.create_node(
                    'call_module', new_logger_name, (load_arg(node_b),), {})

            else:  # call_function

                # TODO(before land): make the code nice

                # translate the inputs from domain of B to domain of A
                # i.e. add a dequant, etc
                # TODO(before land): make this generic with a mapping
                domain_translation_op = torch.dequantize

                new_domain_translation_name = \
                    get_new_attr_name_with_prefix(node_b.name + '_domain_translation_')(gm_b)
                env[new_domain_translation_name] = new_graph.create_node(
                    'call_function', domain_translation_op, (load_arg(node_b.args[0]),), {},
                    new_domain_translation_name)

                # copy bias

                # _print_node(node_a.kwargs['bias'])

                bias_a_copy_name = \
                    get_new_attr_name_with_prefix(node_a.kwargs['bias'].name + '_shadow_copy_')(gm_b)
                bias_a_obj = _getattr_from_fqn(gm_a, node_a.kwargs['bias'].target)
                setattr(gm_b, bias_a_copy_name, bias_a_obj.detach())
                env[bias_a_copy_name] = new_graph.create_node(
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
                env[weight_a_copy_name] = new_graph.create_node(
                    'get_attr', weight_a_copy_name, (), {}, weight_a_copy_name)

                # create the copy of node_a in B
                node_a_shadows_b_name = \
                    get_new_attr_name_with_prefix(node_b.name + '_shadow_copy_')(gm_b)
                env[node_a_shadows_b_name] = new_graph.create_node(
                    node_a.op, node_a.target,
                    (load_arg(env[new_domain_translation_name]), load_arg(env[weight_a_copy_name]),),
                    {'bias': load_arg(env[bias_a_copy_name])},
                    node_a_shadows_b_name)

                # add a logger to node_a_shadows_b
                logger_a_shadows_b_name = \
                    get_new_attr_name_with_prefix(node_a_shadows_b_name + '_ns_logger_a_')(gm_b)
                logger_obj = logger_cls(logger_a_shadows_b_name)
                setattr(gm_b, logger_a_shadows_b_name, logger_obj)
                env[logger_a_shadows_b_name] = new_graph.create_node(
                    'call_module', logger_a_shadows_b_name, (load_arg(env[node_a_shadows_b_name]),),
                    {}, logger_a_shadows_b_name)

                # add a logger to node_b
                logger_b_name = \
                    get_new_attr_name_with_prefix(node_b.name + '_ns_logger_b_')(gm_b)
                logger_obj = logger_cls(logger_b_name)
                setattr(gm_b, logger_b_name, logger_obj)
                env[logger_b_name] = new_graph.create_node(
                    'call_module', logger_b_name, (load_arg(node_b),), {},
                    logger_b_name)

        else:
            env[node_b.name] = new_graph.node_copy(node_b, load_arg)

    new_gm = GraphModule(gm_b, new_graph)
    return new_gm

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

# TODO: other APIs


















