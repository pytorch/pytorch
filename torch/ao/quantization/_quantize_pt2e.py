import torch
from torch.fx import GraphModule
import torch._dynamo as torchdynamo

from .qconfig_mapping import QConfigMapping
from .backend_config import BackendConfig
from .fx import prepare
from .quantize_fx import _convert_to_reference_decomposed_fx
from .pt2e.utils import (
    _get_renamed_nn_module_stack,
    _fuse_conv_bn_,
    _rearrange_weight_observer_for_addmm,
)

import copy
from typing import Tuple, Any, Dict

# TODO: longer term, don't need to retrace or parse the string
# we should have node.meta["nn_module_stack"] that store the dict
def _infer_nn_stack_trace_and_append_on_meta(m, gm, args_as_list):
    trace_func, guards = torchdynamo.export(
        m,
        *copy.deepcopy(args_as_list),
        aten_graph=True,
        tracing_mode="real"
    )
    reset_metadata = {}
    for node in trace_func.graph.nodes:
        nn_module_stack = {}
        if (stack_trace := node.meta.get("stack_trace")) is not None:
            for line in stack_trace.split("\n"):
                if line.startswith("Module stack:"):
                    mod_trace = eval(line.replace("Module stack:", ""))  # pyre-ignore
                    nn_module_stack = {"nn_module_stack": mod_trace}
        reset_metadata[node.name] = nn_module_stack

    for n in gm.graph.nodes:
        if (meta := reset_metadata.get(n.name)):
            n.meta.update(meta)

def prepare_pt2e(
    model: GraphModule,
    qconfig_mapping: QConfigMapping,
    example_inputs: Tuple[Any, ...],
    backend_config: BackendConfig,
):
    _infer_nn_stack_trace_and_append_on_meta(model, model, example_inputs)
    # TODO: move this information to fx node itself
    node_name_to_scope: Dict[str, Tuple[str, type]] = {}
    for n in model.graph.nodes:
        renamed_stack = _get_renamed_nn_module_stack(n.meta.get("nn_module_stack", None))
        current_scope = list(renamed_stack.items())[-1]
        node_name_to_scope[n.name] = current_scope

    # TODO: check qconfig_mapping to make sure conv and bn are both configured
    # to be quantized before fusion
    # TODO: (maybe) rewrite this with subgraph_rewriter
    _fuse_conv_bn_(model)
    model = prepare(
        model,
        qconfig_mapping,
        False,  # is_qat
        node_name_to_scope,
        example_inputs,
        backend_config=backend_config
    )

    # TODO: remove hack when we have better support for pattern matching
    # move around the observer for addmm
    _rearrange_weight_observer_for_addmm(model)
    return model

def convert_pt2e(
    model: GraphModule
):
    return _convert_to_reference_decomposed_fx(model)
