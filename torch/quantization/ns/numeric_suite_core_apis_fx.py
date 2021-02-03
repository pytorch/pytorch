import torch
import torch.nn as nn
import torch.nn.functional as F
toq = torch.ops.quantized
from torch.fx import GraphModule
from torch.fx.graph import Node
from torch.quantization.ns.graph_matcher import (
    get_matching_node_pairs,
    get_type_a_related_to_b,
    _getattr_from_fqn,  # TODO: update name, make reusable
    _print_node,  # TODO: remove this
)

from typing import Dict


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

# TODO: other APIs
