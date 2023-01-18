from torch.fx import GraphModule

from .qconfig_mapping import QConfigMapping
from .backend_config import BackendConfig
from .fx import prepare
from .quantize_fx import _convert_to_reference_decomposed_fx
from ._pt2e.utils import (
    _get_renamed_nn_module_stack,
    _fuse_conv_bn_,
    _rearrange_weight_observer_for_addmm,
)

from typing import Tuple, Any, Dict

def prepare_pt2e(
    model: GraphModule,
    qconfig_mapping: QConfigMapping,
    example_inputs: Tuple[Any, ...],
    backend_config: BackendConfig,
):
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
