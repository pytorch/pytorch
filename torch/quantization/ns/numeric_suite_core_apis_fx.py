import collections
import enum

import torch
import torch.nn as nn
import torch.nn.functional as F
toq = torch.ops.quantized
from torch.fx import GraphModule
from torch.fx.graph import Node
from torch.quantization.ns.graph_matcher import (
    get_matching_subgraph_pairs,
    get_base_name_to_sets_of_related_ops,
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

from typing import Dict, Tuple, Callable, List, Any

class NSSingleResultValuesType(str, enum.Enum):
    WEIGHT = 'weight'
    NODE_OUTPUT = 'node_output'

# TODO(future PR): see if we can use typing_extensions's TypedDict instead
# to properly type the various keys
# {
#   'logger_name_1': {
#     'model_name_a': {
#       # one of NSSingleResultValuesType
#       'type': 'weight',
#       # the values of type specified above
#       'values': [torch.Tensor(...), ...],
#       # name of the node
#       'node_name': 'linear1',
#       # type of the underlying function or module
#       'node_target_type': torch.nn.functional.linear  # or torch.nn.Linear, etc
#     },
#   },
# }
NSSingleResultType = Dict[str, Any]

# {
#   'logger_name_1': {
#     'model_name_a': NSSingleResultType,
#     'model_name_b': NSSingleResultType,
#   },
# }
#
NSResultsType = Dict[str, Dict[str, NSSingleResultType]]

def add_weight_info_to_dict(
    model_name: str,
    model: GraphModule,
    nodes_and_names_to_instrument: List[Tuple[Node, str]],
    results: NSResultsType,
) -> None:
    base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
    type_a_related_to_b = \
        get_type_a_related_to_b(base_name_to_sets_of_related_ops)

    for node, ref_name in nodes_and_names_to_instrument:

        if ref_name not in results:
            results[ref_name] = {}

        if node.op == 'call_function':

            # linear
            # TODO(future PR): other function types
            related_to_linear = node.target in (F.linear,) or \
                (node.target, F.linear) in type_a_related_to_b

            if related_to_linear:
                weight = get_linear_fun_weight(node, model)
                results[ref_name][model_name] = {
                    'type': NSSingleResultValuesType.WEIGHT.value,
                    'values': [weight],
                    'node_name': node.name,
                    'node_target_type': str(node.target),
                }

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
                results[ref_name][model_name] = {
                    'type': NSSingleResultValuesType.WEIGHT.value,
                    'values': [weight],
                    'node_name': node.name,
                    'node_target_type': str(type(mod)),
                }

# Note: this is not a user facing API
# TODO(future PR): wrap this in a user facing API which does not
#   expose FX types.
def compare_weights(
    model_name_a: str,
    gm_a: GraphModule,
    model_name_b: str,
    gm_b: GraphModule,
) -> NSResultsType:
    base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
    type_a_related_to_b = \
        get_type_a_related_to_b(base_name_to_sets_of_related_ops)
    matched_subgraph_pairs = get_matching_subgraph_pairs(gm_a, gm_b)

    # split the subgraph pairs into one data structure for each model
    nodes_and_names_to_instrument_a: List[Tuple[Node, str]] = []
    nodes_and_names_to_instrument_b: List[Tuple[Node, str]] = []
    for match_name, match in matched_subgraph_pairs.items():
        (node_start_a, node_end_a), (node_start_b, node_end_b) = match
        nodes_and_names_to_instrument_a.append((node_start_a, match_name))
        nodes_and_names_to_instrument_b.append((node_start_b, match_name))

    # populate the results, one model at a time
    results: NSResultsType = {}
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
        ref_name: str,
        node_target_type: str,
    ):
        super().__init__()
        self.stats: List[torch.Tensor] = []
        # name of the node whose output this Logger is capturing
        self.node_name = node_name
        # name of the model from which the node originated from
        self.model_name = model_name
        # reference name, used to match loggers from separate models
        # to each other
        self.ref_name = ref_name
        # type of the target of the node whose output this logger is logging
        self.node_target_type = node_target_type

    def forward(self, x: torch.Tensor):
        self.stats.append(x.detach())
        return x

    def __repr__(self):
        return f"OutputLogger(ref_name={self.ref_name}, model_name={self.model_name}, node_name={self.node_name})"

def prepare_single_model_output(
    model_name: str,
    model: GraphModule,
    nodes_and_names_to_instrument: List[Tuple[Node, str]],
    logger_cls: Callable,
) -> GraphModule:

    # TODO(future PR): do not observe nodes we do not care
    #   about (both fp32, denylist, etc)
    node_to_instrument_to_ref_name: Dict[Node, str] = {}
    for node, ref_name in nodes_and_names_to_instrument:
        node_to_instrument_to_ref_name[node] = ref_name

    model = remove_observers_add_loggers(
        model, node_to_instrument_to_ref_name, logger_cls, model_name)
    return model

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
    nodes_and_names_to_instrument_a = []
    nodes_and_names_to_instrument_b = []
    for match_name, (subgraph_a, subgraph_b) in matched_subgraph_pairs.items():
        node_start_a, node_end_a = subgraph_a
        node_start_b, node_end_b = subgraph_b
        # Note: for matching activations we always use the end nodes,
        # such as observing the output of relu in linear-relu
        nodes_and_names_to_instrument_a.append((node_end_a, match_name))
        nodes_and_names_to_instrument_b.append((node_end_b, match_name))

    gm_a = prepare_single_model_output(
        name_a, gm_a, nodes_and_names_to_instrument_a, logger_cls)
    gm_b = prepare_single_model_output(
        name_b, gm_b, nodes_and_names_to_instrument_b, logger_cls)
    return (gm_a, gm_b)

def add_activation_info_to_dict(
    model: GraphModule,
    results: NSResultsType,
    logger_cls: Callable,
) -> None:
    for gm_name, mod in model.named_modules():
        # TODO(future PR): better check when scripted
        is_logger = (
            isinstance(mod, logger_cls)  # type: ignore
            or (
                isinstance(mod, torch.jit.RecursiveScriptModule)
                and mod.original_name == 'OutputLogger'
            )
        )
        if is_logger:
            key = mod.ref_name
            if key not in results:
                results[key] = {}
            results[key][mod.model_name] = {
                'type': NSSingleResultValuesType.NODE_OUTPUT.value,
                'values': mod.stats,
                'node_name': mod.node_name,
                'node_target_type': mod.node_target_type,
            }

# Note: this is not a user facing API
# TODO(future PR): wrap this in a user facing API which does not
#   expose FX types.
# TODO(future PR): align on naming
# this is equivalent of just the comparison extraction part of `ns.compare_model_outputs`
def get_matching_activations(
    gm_a: GraphModule,
    gm_b: GraphModule,
    logger_cls: Callable,
) -> NSResultsType:
    """
    Same thing as ns.get_matching_activations, but for FX models prepared with
    this module.

    TODO(future PR): real docblock

    Output format: NSResultsType
    """
    results: NSResultsType = {}
    for gm in (gm_a, gm_b):
        add_activation_info_to_dict(gm_a, results, logger_cls)
        add_activation_info_to_dict(gm_b, results, logger_cls)
    return results

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
) -> NSResultsType:
    """
    Same thing as get_matching_activations, but for an `a_shadows_b` model.
    TODO(future PR): real docblock
    """
    results: NSResultsType = collections.defaultdict(dict)
    add_activation_info_to_dict(gm_a_shadows_b, results, logger_cls)
    return dict(results)
