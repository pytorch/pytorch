import torch
from torch.fx import GraphModule
from torch.fx import Node

from .pt2e.prepare import prepare
from .pt2e.qat_utils import (
    _fuse_conv_bn_qat,
    _fold_conv_bn_qat,
)
from .pt2e.utils import (
    _get_node_name_to_scope,
    _fuse_conv_bn_,
    _disallow_eval_train,
)
from .pt2e.representation import reference_representation_rewrite
from .quantize_fx import _convert_to_reference_decomposed_fx
from torch.ao.quantization.quantizer import (  # noqa: F401
    Quantizer,
    QuantizationSpecBase,
    QuantizationSpec,
    FixedQParamsQuantizationSpec,
    SharedQuantizationSpec,
    DerivedQuantizationSpec,
    QuantizationAnnotation,
)
from torch.fx.passes.infra.pass_manager import PassManager
from torch.ao.quantization.pt2e.duplicate_dq_pass import DuplicateDQPass
from torch.ao.quantization.pt2e.port_metadata_pass import PortNodeMetaForQDQ
from torch._inductor.constant_folding import constant_fold

__all__ = [
    "prepare_pt2e",
    "prepare_qat_pt2e",
    "convert_pt2e",
]


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
    model = _disallow_eval_train(model)
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
    model = _disallow_eval_train(model)
    return model

_QUANT_OPS = [
    torch.ops.quantized_decomposed.quantize_per_tensor.default,
    torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
    torch.ops.quantized_decomposed.quantize_per_channel.default,
]
def _quant_node_filter(n: Node) -> bool:
    """If there is any pure ops between get_attr and quantize op they will be const propagated
    e.g. get_attr(weight) -> transpose -> quantize -> dequantize*
    (Note: dequantize op is not going to be constant propagated)

    Note: this filter does not handle the case when there are any additional ops after quantize:
    e.g. get_attr(weight) -> transpose1 -> quantize -> transpose2 -> dequantize*
    this filter will filter out the above example since the last op in the sequence and we'll not
    do constant propagation for  the above case.
    """
    return n.op == "call_function" and n.target in _QUANT_OPS

def convert_pt2e(
    model: GraphModule,
    use_reference_representation: bool = False,
) -> GraphModule:
    original_graph_meta = model.meta
    model = _convert_to_reference_decomposed_fx(model)
    model = _fold_conv_bn_qat(model)
    pm = PassManager([DuplicateDQPass()])
    model = pm(model).graph_module

    pm = PassManager([PortNodeMetaForQDQ()])
    model = pm(model).graph_module

    constant_fold(model, _quant_node_filter)
    if use_reference_representation:
        model = reference_representation_rewrite(model)

    model.meta.update(original_graph_meta)
    model = _disallow_eval_train(model)
    return model
