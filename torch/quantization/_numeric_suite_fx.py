import collections

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

from .ns.utils import (
    getattr_from_fqn,
)

from .ns.weight_utils import (
    get_conv_mod_weight,
    get_linear_mod_weight,
    get_lstm_mod_weights,
    get_linear_fun_weight,
)

from .ns.graph_passes import (
    remove_observers_add_loggers,
    create_a_shadows_b,
)

from .ns.ns_types import (
    NSSingleResultValuesType,
)

from typing import Dict, Tuple, Callable, List, Any

RNNReturnType = Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]

class OutputLogger(nn.Module):
    stats: List[torch.Tensor]
    stats_rnn: List[RNNReturnType]

    def __init__(
        self,
        ref_node_name: str,
        prev_node_name: str,
        model_name: str,
        ref_name: str,
        prev_node_target_type: str,
        results_type: str,
        index_within_arg: int,
    ):
        super().__init__()
        self.stats: List[torch.Tensor] = []
        self.stats_rnn: List[RNNReturnType] = []

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
        # index of this node within the arg of the input/output node
        # for example, in cat([x1, x2, x3], dim=0), x2 would have index_within_arg == 1
        self.index_within_arg = index_within_arg

    # Note: cannot annotate the type of x because TorchScript does not support
    #   the Union type.
    def forward(self, x):
        if isinstance(x, torch.Tensor):
            self.stats.append(x.detach())
        elif isinstance(x, tuple) and len(x) == 2 and len(x[1]) == 2:
            new_res = (x[0].detach(), (x[1][0].detach(), x[1][1].detach()))
            self.stats_rnn.append(new_res)
        return x

    def __repr__(self):
        return f"""OutputLogger(ref_name={self.ref_name}, model_name={self.model_name},
prev_node_name={self.prev_node_name}, ref_node_name={self.ref_node_name},
results_type={self.results_type}, index_within_arg={self.index_within_arg})"""


# TODO(future PR): see if we can use typing_extensions's TypedDict instead
# to properly type the various keys
# {
#   'logger_name_1': {
#     'model_name_a': {
#       # one of NSSingleResultValuesType
#       'type': 'weight',
#       # the values of type specified above
#       'values': [torch.tensor(...), ...],
#       # name of the node directly before the logger
#       'prev_node_name': 'linear1',
#       # type of the underlying function or module
#       'prev_node_target_type': torch.nn.functional.linear  # or torch.nn.Linear, etc
#       # name of the node responsible for adding this logger
#       # Note: this may differ from prev_node_name if we are logging inputs
#       'ref_node_name': 'linear1',
#       # index of this node within the arg of the input/output node
#       # for example, in cat([x1, x2, x3], dim=0), x2 would have index_within_arg == 1
#       'index_within_arg': 0,
#     },
#   },
# }
NSSingleResultType = Dict[str, Any]

# {
#   'layer_name_1': {  # subgraph name
#     'node_output': {  # results type (node_output, node_input, weight)
#       'model_name_a':  # model name
#          [NSSingleResultType, ...],  # results, ordered by index_within_arg
#       'model_name_b':
#          [NSSingleResultType, ...],
#     },
#   },
# }
#
NSResultsType = Dict[str, Dict[str, Dict[str, List[NSSingleResultType]]]]


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


def _extract_weights_one_model(
    model_name: str,
    model: GraphModule,
    nodes_and_names_to_instrument: List[Tuple[Node, str]],
    results: NSResultsType,
) -> None:
    base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
    type_a_related_to_b = \
        get_type_a_related_to_b(base_name_to_sets_of_related_ops)

    for node, ref_name in nodes_and_names_to_instrument:

        res_type = NSSingleResultValuesType.WEIGHT.value
        if ref_name not in results:
            results[ref_name] = {res_type: {}}

        if node.op == 'call_function':

            # linear
            # TODO(future PR): other function types
            related_to_linear = node.target in (F.linear,) or \
                (node.target, F.linear) in type_a_related_to_b

            if related_to_linear:
                weight = get_linear_fun_weight(node, model)
                results[ref_name][res_type][model_name] = [{
                    'type': res_type,
                    'values': [weight],
                    'prev_node_name': node.name,
                    'prev_node_target_type': str(node.target),
                    'ref_node_name': node.name,
                    'index_within_arg': 0,
                }]

        else:  # call_module
            # for call_module, we need to look up the modules to do the type check
            assert isinstance(node.target, str)
            mod = getattr_from_fqn(model, node.target)

            # check that A is one the modules we need
            # assume B is related (this is done by graph matcher)
            # TODO(future PR): 1d and 3d convs
            related_to_conv1d_mod = isinstance(mod, nn.Conv1d) or \
                (type(mod), nn.Conv1d) in type_a_related_to_b
            related_to_conv2d_mod = isinstance(mod, nn.Conv2d) or \
                (type(mod), nn.Conv2d) in type_a_related_to_b
            related_to_conv3d_mod = isinstance(mod, nn.Conv3d) or \
                (type(mod), nn.Conv3d) in type_a_related_to_b
            related_to_linear_mod = isinstance(mod, nn.Linear) or \
                (type(mod), nn.Linear) in type_a_related_to_b
            related_to_lstm_mod = isinstance(mod, nn.LSTM) or \
                (type(mod), nn.LSTM) in type_a_related_to_b

            # TODO(future PR): other module types
            if related_to_conv1d_mod or related_to_conv2d_mod or related_to_conv3d_mod:
                weights = [get_conv_mod_weight(mod)]
            elif related_to_lstm_mod:
                weights = get_lstm_mod_weights(mod)
            else:
                assert related_to_linear_mod, f"module type {type(mod)} not handled yet"
                weights = [get_linear_mod_weight(mod)]
            results[ref_name][res_type][model_name] = [{
                'type': res_type,
                'values': weights,
                'prev_node_name': node.name,
                'prev_node_target_type': str(type(mod)),
                'ref_node_name': node.name,
                'index_within_arg': 0,
            }]


def _extract_weights_impl(
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
        subgraph_a, subgraph_b = match
        nodes_and_names_to_instrument_a.append((subgraph_a.base_op_node, match_name))
        nodes_and_names_to_instrument_b.append((subgraph_b.base_op_node, match_name))

    # populate the results, one model at a time
    results: NSResultsType = {}
    _extract_weights_one_model(
        model_name_a, gm_a, nodes_and_names_to_instrument_a, results)
    _extract_weights_one_model(
        model_name_b, gm_b, nodes_and_names_to_instrument_b, results)

    return results


def extract_weights(
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
    return _extract_weights_impl(model_name_a, gm_a, model_name_b, gm_b)


def _add_loggers_one_model(
    model_name: str,
    model: GraphModule,
    nodes_and_names_to_instrument_inputs: List[Tuple[Node, str]],
    nodes_and_names_to_instrument_outputs: List[Tuple[Node, str]],
    logger_cls: Callable,
) -> nn.Module:

    # TODO(future PR): do not observe nodes we do not care
    #   about (both fp32, denylist, etc)
    node_to_instrument_inputs_to_ref_name: Dict[Node, str] = {}
    node_to_instrument_outputs_to_ref_name: Dict[Node, str] = {}
    for node, ref_name in nodes_and_names_to_instrument_inputs:
        node_to_instrument_inputs_to_ref_name[node] = ref_name
    for node, ref_name in nodes_and_names_to_instrument_outputs:
        node_to_instrument_outputs_to_ref_name[node] = ref_name

    model = remove_observers_add_loggers(
        model, node_to_instrument_inputs_to_ref_name,
        node_to_instrument_outputs_to_ref_name, logger_cls, model_name)
    return model


def _add_loggers_impl(
    name_a: str,
    gm_a: GraphModule,
    name_b: str,
    gm_b: GraphModule,
    logger_cls: Callable,
    should_log_inputs: bool,
) -> Tuple[nn.Module, nn.Module]:
    matched_subgraph_pairs = get_matching_subgraph_pairs(gm_a, gm_b)
    nodes_and_names_to_instrument_inputs_a = []
    nodes_and_names_to_instrument_inputs_b = []
    nodes_and_names_to_instrument_outputs_a = []
    nodes_and_names_to_instrument_outputs_b = []
    for match_name, (subgraph_a, subgraph_b) in matched_subgraph_pairs.items():
        # Note: for matching inputs we use start_node, such as observing
        # the input of linear in linear-relu
        if should_log_inputs:
            nodes_and_names_to_instrument_inputs_a.append((subgraph_a.start_node, match_name))
            nodes_and_names_to_instrument_inputs_b.append((subgraph_b.start_node, match_name))
        # Note: for matching activations we always use end_node,
        # such as observing the output of relu in linear-relu
        nodes_and_names_to_instrument_outputs_a.append((subgraph_a.end_node, match_name))
        nodes_and_names_to_instrument_outputs_b.append((subgraph_b.end_node, match_name))

    new_model_a = _add_loggers_one_model(
        name_a, gm_a, nodes_and_names_to_instrument_inputs_a,
        nodes_and_names_to_instrument_outputs_a, logger_cls)
    new_model_b = _add_loggers_one_model(
        name_b, gm_b, nodes_and_names_to_instrument_inputs_b,
        nodes_and_names_to_instrument_outputs_b, logger_cls)
    return (new_model_a, new_model_b)


def add_loggers(
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
    return _add_loggers_impl(
        name_a, gm_a, name_b, gm_b, logger_cls,
        should_log_inputs=should_log_inputs)


def _extract_logger_info_one_model(
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
            assert mod.model_name not in results[key], \
                f"{mod.model_name} is already present in results"
            if mod.results_type not in results[key]:
                results[key][mod.results_type] = {}
            if mod.model_name not in results[key][mod.results_type]:
                results[key][mod.results_type][mod.model_name] = []
            stats_to_use = mod.stats
            if len(mod.stats_rnn) > 0:
                stats_to_use = mod.stats_rnn
            results[key][mod.results_type][mod.model_name].append({
                'type': mod.results_type,
                'values': stats_to_use,
                'ref_node_name': mod.ref_node_name,
                'prev_node_name': mod.prev_node_name,
                'prev_node_target_type': mod.prev_node_target_type,
                'index_within_arg': mod.index_within_arg,
            })
            # ensure the list stays sorted
            results[key][mod.results_type][mod.model_name].sort(
                key=lambda res: res['index_within_arg']
            )


# TODO(future PR): align on naming
# this is equivalent of just the comparison extraction part of `ns.compare_model_outputs`
def extract_logger_info(
    model_a: nn.Module,
    model_b: nn.Module,
    logger_cls: Callable,
) -> NSResultsType:
    """
    Same thing as ns.extract_logger_info, but for models prepared with
    this module.

    TODO(future PR): real docblock

    Output format: NSResultsType
    """
    results: NSResultsType = {}
    for model in (model_a, model_b):
        _extract_logger_info_one_model(model, results, logger_cls)
    return results


def _add_shadow_loggers_impl(
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


def add_shadow_loggers(
    name_a: str,
    model_a: nn.Module,
    name_b: str,
    model_b: nn.Module,
    logger_cls: Callable,
    should_log_inputs: bool = False,
) -> nn.Module:
    """
    Same thing as add_loggers, but for an `a_shadows_b` model.
    TODO(future PR): real docblock
    """
    tracer_a, tracer_b = NSTracer(), NSTracer()
    gm_a = GraphModule(model_a, tracer_a.trace(model_a))
    gm_b = GraphModule(model_b, tracer_b.trace(model_b))
    return _add_shadow_loggers_impl(
        name_a, gm_a, name_b, gm_b, logger_cls,
        should_log_inputs=should_log_inputs)


def extract_shadow_logger_info(
    model_a_shadows_b: nn.Module,
    logger_cls: Callable,
) -> NSResultsType:
    """
    Same thing as extract_logger_info, but for an `a_shadows_b` model.
    TODO(future PR): real docblock
    """
    results: NSResultsType = collections.defaultdict(dict)
    _extract_logger_info_one_model(model_a_shadows_b, results, logger_cls)
    return dict(results)
