import logging
import random
import weakref

import functorch

import torch
from torch import _prims
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode
from torch.overrides import TorchFunctionMode
from .utils import is_cpu_device
from . import config
from .utils import decode_device, is_cpu_device


def is_quantized_graph_module(gm: torch.fx.GraphModule):
    found_quantize = False
    quantize_ops = (
        torch.ops.quantized_decomposed.quantize_per_tensor,
        torch.ops.quantized_decomposed.quantize_per_channel,
        torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
    )
    for node in gm.graph.nodes:
        if node.target in quantize_ops:
            found_quantize = True
            break
    return found_quantize


def _quantize_and_replace_weight(
    gm: torch.fx.GraphModule, dq_per_channel_node: torch.fx.Node
):
    # pattern: w - q - dq - weighted op
    q_per_channel_node = dq_per_channel_node.args[0]
    weight_node = q_per_channel_node.args[0]
    w_attr_name = weight_node.target
    weight = getattr(gm, w_attr_name)

    assert isinstance(weight, torch.Tensor), "Cannot find weight for quantization"
    if weight.is_quantized:
        return
    quantize_args = (
        getattr(gm, n.target) if isinstance(n, torch.fx.Node) else n
        for n in q_per_channel_node.args
    )
    q_arg_list = list(quantize_args)
    q_arg_tuple = tuple(q_arg_list)
    weight_int8 = torch.nn.parameter.Parameter(
        torch.ops.quantized_decomposed.quantize_per_channel(*q_arg_tuple),
        requires_grad=False,
    )

    qw_attr_name = w_attr_name + "_quant"
    setattr(gm, qw_attr_name, weight_int8)
    weight_node.target = qw_attr_name
    gm.graph.owning_module._buffers[qw_attr_name] = weight_int8
    delattr(gm, w_attr_name)
    q_per_channel_node.replace_all_uses_with(weight_node)
    gm.graph.erase_node(q_per_channel_node)


def pre_quantize_weights(gm: torch.fx.GraphModule):
    # pattern: w - q - dq - weighted op
    aten = torch.ops.aten
    decomposed = torch.ops.quantized_decomposed
    for node in gm.graph.nodes:
        dq_per_channel_node = None
        if node.target == aten.convolution.default:
            # conv args = (x, w, ...)
            dq_per_channel_node = node.args[1]
        if dq_per_channel_node is not None:
            assert (
                dq_per_channel_node.target == decomposed.dequantize_per_channel
            ), "Cannot find the dequantize op for weight"
            _quantize_and_replace_weight(gm, dq_per_channel_node)
    gm.graph.lint()
    gm.recompile()

def fuse_quantization(gm: torch.fx.GraphModule, example_inputs):
    # skip if gm is not a quantized graph module
    if not (is_quantized_graph_module(gm) and is_cpu_device(example_inputs)):
        return gm
    gm.graph.eliminate_dead_code()
    gm.recompile()
    # Fuse `quant_per_channel - weight` and replace the original fp32 weight with quantized one
    pre_quantize_weights(gm)

    return gm
