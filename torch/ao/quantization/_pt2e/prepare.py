from ._quantizer import Quantizer
from torch._subclasses import FakeTensor
from torch.ao.quantization.prepare import (
    _maybe_insert_input_observers_for_node,
    _maybe_insert_output_observer_for_node,
    _is_observer_in_same_graph,
    _maybe_make_input_output_share_observers,
    _remove_output_observer,
    _maybe_insert_observers_before_graph_output,
    _save_state
)
from torch.fx import GraphModule

from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.custom_config import PrepareCustomConfig
from typing import Dict, Tuple, Any

def prepare(
        model: GraphModule,
        quantizer: Quantizer,
        is_qat: bool,
        node_name_to_scope: Dict[str, Tuple[str, type]],
) -> GraphModule:
    model = quantizer.annotate(model)
    quantizer.validate(model)

    # Since we are mutating the graph as we go, we iterate over the original
    # nodes before observer insertion, instead of model.graph.nodes.
    nodes_before_observation = list(model.graph.nodes)

    for node in nodes_before_observation:

        if node.op == "placeholder":
            pass
        elif node.op in ("call_module", "call_method", "call_function"):
            this_node_dtype_info = node.meta["target_dtype_info"]
            if "val" in node.meta:
                output_is_a_tensor = (
                    this_node_dtype_info is not None and
                    isinstance(node.meta["val"], FakeTensor)
                )
            else:
                output_is_a_tensor = this_node_dtype_info is not None

            skip_inserting_observers = (
                not output_is_a_tensor
            )

            if skip_inserting_observers:
                continue

            _maybe_insert_input_observers_for_node(
                node,
                None, # qconfig
                model,
                named_modules,
                model.graph,
                None, # qhandler
                PrepareCustomConfig(),
                None, # backend_config
            )

            named_modules = dict(model.named_modules(remove_duplicate=False))
            # this returns the new observer node if it was needed
            maybe_output_obs_node = _maybe_insert_output_observer_for_node(node, model, named_modules, model.graph)

            if maybe_output_obs_node is None:
                continue
            # Update users of original node to use the output observer
            # instead. For example, change
            #
            #           next_node
            #          /
            #   cur_node -> obs
            #
            # to
            #
            #                 next_node
            #                 /
            #   cur_node -> obs
            #
            # We need to save orig users before updating uses because
            # the list of users will change as we update uses
            orig_users = list(node.users.keys())
            for user_node in orig_users:
                if user_node is maybe_output_obs_node:
                    continue
                user_node.replace_input_with(node, maybe_output_obs_node)
            _is_observer_in_same_graph_ = _is_observer_in_same_graph(node, named_modules)

            # for general tensor value ops, we modify the graph
            # to make all inputs and outputs use the first input's
            # observer
            if (is_general_tensor_value_op and _is_observer_in_same_graph_) or \
                    _is_reuse_input_qconfig_:
                if not _maybe_make_input_output_share_observers(node, model, named_modules):
                    _remove_output_observer(node, model, named_modules)

        elif node.op == "output":
            _maybe_insert_observers_before_graph_output(
                node, model, named_modules, model.graph)

    model = GraphModule(model, model.graph)

    _save_state(
        model,
        {},  # node_name_to_qconfig
        node_name_to_scope,
        PrepareCustomConfig(),
        {},  # equalization_node_name_to_qconfig
        QConfigMapping(),
        is_qat,
        set()  # observed_node_names
    )
    return model
