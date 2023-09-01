from torch.fx import GraphModule

from .pt2e.prepare import prepare
from .pt2e.qat_utils import (
    _fuse_conv_bn_qat,
    _fold_conv_bn_qat,
)
from .pt2e.utils import (
    _get_node_name_to_scope,
    _fuse_conv_bn_,
)
from .pt2e.representation import reference_representation_rewrite
from .fx.prepare import prepare as fx_prepare
from .quantize_fx import _convert_to_reference_decomposed_fx
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.quantizer import (  # noqa: F401
    Quantizer,
    QuantizationSpecBase,
    QuantizationSpec,
    FixedQParamsQuantizationSpec,
    SharedQuantizationSpec,
    DerivedQuantizationSpec,
    QuantizationAnnotation,
)
from torch.ao.quantization.backend_config import BackendConfig

from typing import Any, Tuple

__all__ = [
    "prepare_pt2e",
    "prepare_qat_pt2e",
    "convert_pt2e",
]

def _prepare_pt2e_deprecated(
    model: GraphModule,
    qconfig_mapping: QConfigMapping,
    example_inputs: Tuple[Any, ...],
    backend_config: BackendConfig,
) -> GraphModule:
    node_name_to_scope = _get_node_name_to_scope(model)

    # TODO: check qconfig_mapping to make sure conv and bn are both configured
    # to be quantized before fusion
    # TODO: (maybe) rewrite this with subgraph_rewriter
    _fuse_conv_bn_(model)
    model = fx_prepare(
        model,
        qconfig_mapping,
        False,  # is_qat
        node_name_to_scope,
        example_inputs,
        backend_config=backend_config
    )
    return model

def prepare_pt2e(
    model: GraphModule,
    quantizer: Quantizer,
) -> GraphModule:
    original_graph_meta = model.meta
    node_name_to_scope = _get_node_name_to_scope(model)
    # TODO: check qconfig_mapping to make sure conv and bn are both configured
    # to be quantized before fusion
    # TODO: (maybe) rewrite this with subgraph_rewriter
    _fuse_conv_bn_(model)
    quantizer.annotate(model)
    quantizer.validate(model)
    model = prepare(model, node_name_to_scope, is_qat=False)
    model.meta.update(original_graph_meta)
    return model

def prepare_qat_pt2e(
    model: GraphModule,
    quantizer: Quantizer,
) -> GraphModule:
    original_graph_meta = model.meta
    node_name_to_scope = _get_node_name_to_scope(model)
    quantizer.annotate(model)
    quantizer.validate(model)
    # Perform fusion after annotate to avoid quantizing ops in the new
    # subgraph that don't need to be quantized
    # TODO: only fuse if conv and bn are both configured to be quantized
    _fuse_conv_bn_qat(model)
    model = prepare(model, node_name_to_scope, is_qat=True)
    model.meta.update(original_graph_meta)
    return model

def convert_pt2e(
    model: GraphModule,
    use_reference_representation: bool = False,
) -> GraphModule:
    original_graph_meta = model.meta
    model = _convert_to_reference_decomposed_fx(model)
    model = _fold_conv_bn_qat(model)
    if use_reference_representation:
        model = reference_representation_rewrite(model)

    model.meta.update(original_graph_meta)
    return model
