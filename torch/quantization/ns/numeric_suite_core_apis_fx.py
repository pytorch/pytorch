import collections
import enum

import torch
import torch.nn as nn
import torch.nn.functional as F
toq = torch.ops.quantized
from torch.fx import GraphModule
from torch.fx.graph import Node
from torch.fx.symbolic_trace import Tracer
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

from .ns_types import (
    NSSingleResultValuesType,
)

from typing import Dict, Tuple, Callable, List, Any

# TODO(future PR): see if we can use typing_extensions's TypedDict instead
# to properly type the various keys
# {
#   'logger_name_1': {
#     'model_name_a': {
#       # one of NSSingleResultValuesType
#       'type': 'weight',
#       # the values of type specified above
#       'values': [torch.Tensor(...), ...],
#       # name of the node directly before the logger
#       'prev_node_name': 'linear1',
#       # type of the underlying function or module
#       'prev_node_target_type': torch.nn.functional.linear  # or torch.nn.Linear, etc
#       # name of the node responsible for adding this logger
#       # Note: this may differ from prev_node_name if we are logging inputs
#       'ref_node_name': 'linear1',
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


class NSTracer(Tracer):
    """
    Just like a regular tracer, but treats observers and fake_quantize
    modules as leaf modules.
    """
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str) -> bool:
        if isinstance(m, torch.quantization.ObserverBase):
            return True
        elif isinstance(m, torch.quantization.FakeQuantizeBase):
            return True
        return super().is_leaf_module(m, module_qualified_name)


def _add_weight_info_to_dict(
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
                    'prev_node_target_type': str(node.target),
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
                    'prev_node_target_type': str(type(mod)),
                }


def _compare_weights_impl(
    model_name_a: str,
    gm_a: GraphModule,
    model_name_b: str,
    gm_b: GraphModule,
) -> NSResultsType:
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
    _add_weight_info_to_dict(
        model_name_a, gm_a, nodes_and_names_to_instrument_a, results)
    _add_weight_info_to_dict(
        model_name_b, gm_b, nodes_and_names_to_instrument_b, results)

    return results


def compare_weights(
    model_name_a: str,
    model_a: nn.Module,
    model_name_b: str,
    model_b: nn.Module,
) -> NSResultsType:
    base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
    type_a_related_to_b = \
        get_type_a_related_to_b(base_name_to_sets_of_related_ops)

    tracer_a, tracer_b = NSTracer(), NSTracer()
    gm_a = GraphModule(model_a, tracer_a.trace(model_a))
    gm_b = GraphModule(model_b, tracer_b.trace(model_b))
    return _compare_weights_impl(model_name_a, gm_a, model_name_b, gm_b)


class OutputLogger(nn.Module):
    stats: List[torch.Tensor]

    def __init__(
        self,
        ref_node_name: str,
        prev_node_name: str,
        model_name: str,
        ref_name: str,
        prev_node_target_type: str,
        results_type: str,
    ):
        super().__init__()
        self.stats: List[torch.Tensor] = []

        # name of the node which was responsible for adding this logger
        # Note:
        # - if we are logging node outputs, this is the same as prev_node_name
        # - if we are logging node inputs, this is the name of the node
        #   whose input this logger is logging.
        #
        # example, where logger1 is logging input of op1 and logger2 is logging
        #    the output of op1:
        #
        #  x1 -> logger1 -> op1 -> logger2 -> x2
        #
        # in this example,
        #   - logger1's prev_node_name is x1 and ref_node_name is op1
        #   - logger2's prev_node_name is op1 and ref_node_name is op1
        self.ref_node_name = ref_node_name
        # name of the node whose output this Logger is capturing
        self.prev_node_name = prev_node_name

        # name of the model from which the node originated from
        self.model_name = model_name
        # reference name, used to match loggers from separate models
        # to each other
        self.ref_name = ref_name
        # type of the target of the node whose output this logger is logging
        self.prev_node_target_type = prev_node_target_type
        # what kind of values are inside of stats
        self.results_type = results_type

    def forward(self, x: torch.Tensor):
        self.stats.append(x.detach())
        return x

    def __repr__(self):
        return f"OutputLogger(ref_name={self.ref_name}, model_name={self.model_name}, prev_node_name={self.prev_node_name}, ref_node_name={self.ref_node_name})"


def _prepare_single_model_output(
    model_name: str,
    model: GraphModule,
    nodes_and_names_to_instrument: List[Tuple[Node, str]],
    logger_cls: Callable,
    should_log_inputs: bool,
) -> nn.Module:

    # TODO(future PR): do not observe nodes we do not care
    #   about (both fp32, denylist, etc)
    node_to_instrument_to_ref_name: Dict[Node, str] = {}
    for node, ref_name in nodes_and_names_to_instrument:
        node_to_instrument_to_ref_name[node] = ref_name

    model = remove_observers_add_loggers(
        model, node_to_instrument_to_ref_name, logger_cls, model_name,
        should_log_inputs=should_log_inputs)
    return model


def _prepare_model_outputs_impl(
    name_a: str,
    gm_a: GraphModule,
    name_b: str,
    gm_b: GraphModule,
    logger_cls: Callable,
    should_log_inputs: bool,
) -> Tuple[nn.Module, nn.Module]:
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

    new_model_a = _prepare_single_model_output(
        name_a, gm_a, nodes_and_names_to_instrument_a, logger_cls,
        should_log_inputs=should_log_inputs)
    new_model_b = _prepare_single_model_output(
        name_b, gm_b, nodes_and_names_to_instrument_b, logger_cls,
        should_log_inputs=should_log_inputs)
    return (new_model_a, new_model_b)


def prepare_model_outputs(
    name_a: str,
    model_a: nn.Module,
    name_b: str,
    model_b: nn.Module,
    logger_cls: Callable,
    should_log_inputs : bool = False,
) -> Tuple[nn.Module, nn.Module]:
    tracer_a, tracer_b = NSTracer(), NSTracer()
    gm_a = GraphModule(model_a, tracer_a.trace(model_a))
    gm_b = GraphModule(model_b, tracer_b.trace(model_b))
    return _prepare_model_outputs_impl(
        name_a, gm_a, name_b, gm_b, logger_cls,
        should_log_inputs=should_log_inputs)


def add_activation_info_to_dict(
    model: nn.Module,
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
                'type': mod.results_type,
                'values': mod.stats,
                'ref_node_name': mod.ref_node_name,
                'prev_node_name': mod.prev_node_name,
                'prev_node_target_type': mod.prev_node_target_type,
            }


# TODO(future PR): align on naming
# this is equivalent of just the comparison extraction part of `ns.compare_model_outputs`
def get_matching_activations(
    model_a: nn.Module,
    model_b: nn.Module,
    logger_cls: Callable,
) -> NSResultsType:
    """
    Same thing as ns.get_matching_activations, but for models prepared with
    this module.

    TODO(future PR): real docblock

    Output format: NSResultsType
    """
    results: NSResultsType = {}
    for model in (model_a, model_b):
        add_activation_info_to_dict(model, results, logger_cls)
    return results


def _prepare_model_with_stubs_impl(
    name_a: str,
    gm_a: GraphModule,
    name_b: str,
    gm_b: GraphModule,
    logger_cls: Callable,
    should_log_inputs: bool,
) -> nn.Module:
    matched_subgraph_pairs = get_matching_subgraph_pairs(gm_a, gm_b)
    gm_a_shadows_b = create_a_shadows_b(
        name_a, gm_a, name_b, gm_b, matched_subgraph_pairs, logger_cls,
        should_log_inputs=should_log_inputs)
    return gm_a_shadows_b


def prepare_model_with_stubs(
    name_a: str,
    model_a: nn.Module,
    name_b: str,
    model_b: nn.Module,
    logger_cls: Callable,
    should_log_inputs: bool = False,
) -> nn.Module:
    """
    Same thing as prepare_model_outputs, but for an `a_shadows_b` model.
    TODO(future PR): real docblock
    """
    tracer_a, tracer_b = NSTracer(), NSTracer()
    gm_a = GraphModule(model_a, tracer_a.trace(model_a))
    gm_b = GraphModule(model_b, tracer_b.trace(model_b))
    return _prepare_model_with_stubs_impl(
        name_a, gm_a, name_b, gm_b, logger_cls,
        should_log_inputs=should_log_inputs)


# TODO(future PR): align on naming
# this is equivalent of just the comparison extraction part of `ns.compare_model_stub`
def get_matching_activations_a_shadows_b(
    model_a_shadows_b: nn.Module,
    logger_cls: Callable,
) -> NSResultsType:
    """
    Same thing as get_matching_activations, but for an `a_shadows_b` model.
    TODO(future PR): real docblock
    """
    results: NSResultsType = collections.defaultdict(dict)
    add_activation_info_to_dict(model_a_shadows_b, results, logger_cls)
    return dict(results)
