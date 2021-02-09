import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
toq = torch.ops.quantized
from torch.fx import GraphModule
from torch.quantization.ns.graph_matcher import (
    get_matching_node_pairs,
    get_type_a_related_to_b,
)

from .utils import (
    getattr_from_fqn,
)

from .weight_utils import (
    get_conv_mod_weight,
    get_linear_fun_weight,
)

from .graph_passes import (
    remove_observers_add_loggers,
    create_a_shadows_b,
)

from typing import Dict, Tuple, Callable, List, Optional


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
            # TODO(future PR): other function types
            a_related_to_linear = node_a.target in (F.linear,) or \
                (node_a.target, F.linear) in type_a_related_to_b

            if a_related_to_linear:
                weight_a = get_linear_fun_weight(node_a, gm_a)
                weight_b = get_linear_fun_weight(node_b, gm_b)

                results[match_name] = {
                    name_a: weight_a,
                    name_b: weight_b,
                }

        else:  # call_module
            # for call_module, we need to look up the modules to do the type check
            assert isinstance(node_a.target, str)
            mod_a = getattr_from_fqn(gm_a, node_a.target)
            assert isinstance(node_b.target, str)
            mod_b = getattr_from_fqn(gm_b, node_b.target)

            # check that A is one the modules we need
            # assume B is related (this is done by graph matcher)
            a_related_to_conv2d_mod = isinstance(mod_a, nn.Conv2d) or \
                (type(mod_a), nn.Conv2d) in type_a_related_to_b

            # TODO(future PR): other module types
            if a_related_to_conv2d_mod:
                weight_a = get_conv_mod_weight(mod_a)
                weight_b = get_conv_mod_weight(mod_b)
                results[match_name] = {
                    name_a: weight_a,
                    name_b: weight_b,
                }

    return results


class OutputLogger(nn.Module):
    stats: List[torch.Tensor]

    def __init__(
        self,
        node_name: str,
        model_name: str,
        other_node_name: Optional[str] = None,
    ):
        super().__init__()
        self.stats: List[torch.Tensor] = []
        # name of the node whose output this Logger is capturing
        self.node_name = node_name
        # name of the model from which the node originated from
        self.model_name = model_name
        # name of the other node with a matching Logger
        # used to link node_a_copy -> logger_a to node_c -> logger_c
        # in a_shadows_b
        self.other_node_name = other_node_name

    def forward(self, x: torch.Tensor):
        self.stats.append(x.detach())
        return x

    def __repr__(self):
        return f"OutputLogger(node_name={self.node_name}, model_name={self.model_name}, other_node_name={self.other_node_name})"

# Note: this is not a user facing API
# TODO(future PR): wrap this in a user facing API which does not
#   expose FX types.
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
        # TODO(future PR): do not observe pairs of nodes we do not care
        #   about (both fp32, denylist, etc)
        nodes_to_instrument_a.append(node_a)
        nodes_to_instrument_b.append(node_b)

    gm_a = remove_observers_add_loggers(gm_a, nodes_to_instrument_a, logger_cls, name_a)
    gm_b = remove_observers_add_loggers(gm_b, nodes_to_instrument_b, logger_cls, name_b)
    return (gm_a, gm_b)

# Note: this is not a user facing API
# TODO(future PR): wrap this in a user facing API which does not
#   expose FX types.
# TODO(future PR): align on naming
# this is equivalent of just the comparison extraction part of `ns.compare_model_outputs`
def get_matching_activations(
    gm_a: GraphModule,
    gm_b: GraphModule,
    logger_cls: Callable,
) -> Dict[str, Dict[str, List[torch.Tensor]]]:
    """
    Same thing as ns.get_matching_activations, but for FX models prepared with
    this module.

    TODO(future PR): real docblock

    Output format:

    {
        'layer1.stats': {
            'name_a': [torch.Tensor(...), ...],
            'name_b': [torch.Tensor(...), ...],
        },
        ...
    }

    Note, there are three differences from the output format of Eager NS:
    1. `name_a` and `name_b` are used instead of hardcoding names
       to `float` and `quantized`.
    2. Lists of Tensors are returned instead of individual Tensors, to unify
       the return type for calibrating with 1 input vs N inputs.
    3. `logger_cls` is included in the API for easy result extraction
    """
    results: Dict[str, Dict[str, List[torch.Tensor]]] = \
        collections.defaultdict(dict)
    for gm in (gm_a, gm_b):
        for gm_name, mod in gm.named_modules():
            # TODO(future PR): better check when scripted
            is_logger = (
                isinstance(mod, logger_cls)  # type: ignore
                or (
                    isinstance(mod, torch.jit.RecursiveScriptModule)
                    and mod.original_name == 'OutputLogger'
                )
            )
            if is_logger:
                results[mod.node_name + '.stats'][mod.model_name] = mod.stats
    return dict(results)

# Note: this is not a user facing API
# TODO(future PR): wrap this in a user facing API which does not
#   expose FX types.
def prepare_model_with_stubs(
    name_a: str,
    gm_a: GraphModule,
    name_b: str,
    gm_b: GraphModule,
    logger_cls: Callable,
) -> GraphModule:
    """
    Same thing as prepare_model_outputs, but for an `a_shadows_b` model.
    TODO(future PR): real docblock
    """
    matched_node_pairs = get_matching_node_pairs(gm_a, gm_b)
    gm_a_shadows_b = create_a_shadows_b(
        name_a, gm_a, name_b, gm_b, matched_node_pairs, logger_cls)
    return gm_a_shadows_b

# Note: this is not a user facing API
# TODO(future PR): wrap this in a user facing API which does not
#   expose FX types.
# TODO(future PR): align on naming
# this is equivalent of just the comparison extraction part of `ns.compare_model_stub`
def get_matching_activations_a_shadows_b(
    gm_a_shadows_b: GraphModule,
    logger_cls: Callable,
) -> Dict[str, Dict[str, List[torch.Tensor]]]:
    """
    Same thing as get_matching_activations, but for an `a_shadows_b` model.
    TODO(future PR): real docblock
    """
    results: Dict[str, Dict[str, List[torch.Tensor]]] = \
        collections.defaultdict(dict)
    for name, mod in gm_a_shadows_b.named_modules():
        # TODO(future PR): better check when scripted
        is_logger = (
            isinstance(mod, logger_cls)  # type: ignore
            or (
                isinstance(mod, torch.jit.RecursiveScriptModule)
                and mod.original_name == 'OutputLogger'
            )
        )
        if is_logger:
            # If logger_obj.other_node_name is populated, then this logger
            # is from model A, and other_node_name is the name from model B.
            if mod.other_node_name is None:
                results[mod.node_name + '.stats'][mod.model_name] = mod.stats
            else:
                results[mod.other_node_name + '.stats'][mod.model_name] = mod.stats
    return dict(results)
