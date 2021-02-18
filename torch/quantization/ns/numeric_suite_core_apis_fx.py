import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
toq = torch.ops.quantized
from torch.fx import GraphModule
from torch.fx.graph import Node
from torch.quantization.ns.graph_matcher import (
    get_matching_subgraph_pairs,
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

def add_weight_info_to_dict(
    model_name: str,
    model: GraphModule,
    nodes_and_names_to_instrument: List[Tuple[Node, str]],
    results: Dict[str, Dict[str, torch.Tensor]],
) -> None:
    type_a_related_to_b = get_type_a_related_to_b()

    for node, ref_node_name in nodes_and_names_to_instrument:

        if ref_node_name not in results:
            results[ref_node_name] = {}

        if node.op == 'call_function':

            # linear
            # TODO(future PR): other function types
            related_to_linear = node.target in (F.linear,) or \
                (node.target, F.linear) in type_a_related_to_b

            if related_to_linear:
                weight = get_linear_fun_weight(node, model)
                results[ref_node_name][model_name] = weight

        else:  # call_module
            # for call_module, we need to look up the modules to do the type check
            assert isinstance(node.target, str)
            mod = getattr_from_fqn(model, node.target)

            # check that A is one the modules we need
            # assume B is related (this is done by graph matcher)
            related_to_conv2d_mod = isinstance(mod, nn.Conv2d) or \
                (type(mod), nn.Conv2d) in type_a_related_to_b

            # TODO(future PR): other module types
            if related_to_conv2d_mod:
                weight = get_conv_mod_weight(mod)
                results[ref_node_name][model_name] = weight

# Note: this is not a user facing API
# TODO(future PR): wrap this in a user facing API which does not
#   expose FX types.
def compare_weights(
    model_name_a: str,
    gm_a: GraphModule,
    model_name_b: str,
    gm_b: GraphModule,
) -> Dict[str, Dict[str, torch.Tensor]]:
    type_a_related_to_b = get_type_a_related_to_b()
    matched_subgraph_pairs = get_matching_subgraph_pairs(gm_a, gm_b)

    # split the subgraph pairs into one data structure for each model
    nodes_and_names_to_instrument_a: List[Tuple[Node, str]] = []
    nodes_and_names_to_instrument_b: List[Tuple[Node, str]] = []
    for match_name, match in matched_subgraph_pairs.items():
        (node_start_a, node_end_a), (node_start_b, node_end_b) = match
        nodes_and_names_to_instrument_a.append((node_start_a, match_name))
        nodes_and_names_to_instrument_b.append((node_start_b, match_name))

    # populate the results, one model at a time
    results: Dict[str, Dict[str, torch.Tensor]] = {}
    add_weight_info_to_dict(
        model_name_a, gm_a, nodes_and_names_to_instrument_a, results)
    add_weight_info_to_dict(
        model_name_b, gm_b, nodes_and_names_to_instrument_b, results)

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

    matched_subgraph_pairs = get_matching_subgraph_pairs(gm_a, gm_b)

    node_to_instrument_to_other_node_name_a: Dict[Node, Optional[str]] = {}
    node_to_instrument_to_other_node_name_b: Dict[Node, Optional[str]] = {}
    for match_name, match in matched_subgraph_pairs.items():
        (node_start_a, node_end_a), (node_start_b, node_end_b) = match
        # TODO(future PR): do not observe pairs of nodes we do not care
        #   about (both fp32, denylist, etc)
        # Note: for matching activations we always use the end nodes,
        # such as observing the output of relu in linear-relu
        # Note: other_node_name is set to None in model B's loggers,
        # and set to the corresponding model B's node in model A's loggers.
        node_to_instrument_to_other_node_name_a[node_end_a] = node_end_b.name
        node_to_instrument_to_other_node_name_b[node_end_b] = None

    gm_a = remove_observers_add_loggers(
        gm_a, node_to_instrument_to_other_node_name_a, logger_cls, name_a)
    gm_b = remove_observers_add_loggers(
        gm_b, node_to_instrument_to_other_node_name_b, logger_cls, name_b)
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
                # If logger_obj.other_node_name is populated, then this logger
                # is from model A, and other_node_name is the name from model B.
                if mod.other_node_name is None:
                    results[mod.node_name + '.stats'][mod.model_name] = mod.stats
                else:
                    results[mod.other_node_name + '.stats'][mod.model_name] = mod.stats
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
    matched_subgraph_pairs = get_matching_subgraph_pairs(gm_a, gm_b)
    gm_a_shadows_b = create_a_shadows_b(
        name_a, gm_a, name_b, gm_b, matched_subgraph_pairs, logger_cls)
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
