import copy
import operator
from typing import Any, Callable, Tuple

import torch
import torch._dynamo
from torch.fx import GraphModule, Node
from torch.fx.subgraph_rewriter import _replace_pattern
import torch.nn.functional as F


def _conv_bn_pattern(
    x,
    conv_weight,
    conv_bias,
    bn_weight,
    bn_bias,
    bn_running_mean,
    bn_running_var,
):
    x = F.conv2d(x, conv_weight, conv_bias)
    x = F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=True)
    return x


def _fused_qat_conv_bn_pattern(
    x,
    conv_weight,
    conv_bias,
    bn_weight,
    bn_bias,
    bn_running_mean,
    bn_running_var,
):
    """
    Approximated method to fuse conv and bn. It requires only one forward pass.
    conv_orig = conv / scale_factor where scale_factor = bn.weight / running_std.
    This is based on `nniqat.ConvBn2d._forward_approximate`.
    """
    # TODO: don't hardcode eps
    bn_eps = 1e-5
    running_std = torch.sqrt(bn_running_var + bn_eps)
    scale_factor = bn_weight / running_std
    weight_shape = [1] * len(conv_weight.shape)
    weight_shape[0] = -1
    bias_shape = [1] * len(conv_weight.shape)
    bias_shape[1] = -1
    # TODO: wrap this in weight_fake_quant
    scaled_weight = conv_weight * scale_factor.reshape(weight_shape)
    zero_bias = torch.zeros_like(conv_bias, dtype=x.dtype)
    x = F.conv2d(x, scaled_weight, zero_bias)
    x = x / scale_factor.reshape(bias_shape)
    x = x + conv_bias.reshape(bias_shape)
    x = F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=True, eps=bn_eps)
    return x

def _get_aten_graph_module(
        pattern: Callable,
        example_inputs: Tuple[Any, ...],
):
    """
    Convert the pattern to an FX graph with decomposed aten ops.
    """
    aten_pattern, _ = torch._dynamo.export(
        pattern,
        *copy.deepcopy(example_inputs),
        aten_graph=True,
        tracing_mode="real",
    )
    aten_pattern.graph.eliminate_dead_code()
    aten_pattern.recompile()
    return aten_pattern

def _fuse_conv_bn_qat(m: GraphModule):
    """
    Given a graph of decomposed aten ops, replace the (conv + bn) pattern with
    the fused QAT subgraph equivalent. The input graph should already be annotated.
    """
    m.graph.eliminate_dead_code()
    m.recompile()
    example_inputs = (
        torch.randn(1, 1, 3, 3),  # x
        torch.randn(1, 1, 1, 1),  # conv_weight
        torch.randn(1),           # conv_bias
        torch.randn(1),           # bn_weight
        torch.randn(1),           # bn_bias
        torch.randn(1),           # bn_running_mean
        torch.randn(1),           # bn_running_var
    )
    match_pattern = _get_aten_graph_module(_conv_bn_pattern, example_inputs)
    replacement_pattern = _get_aten_graph_module(_fused_qat_conv_bn_pattern, example_inputs)
    match_and_replacement = _replace_pattern(m, match_pattern, replacement_pattern)
    m.recompile()

    # Copy over metadata from original subgraph
    # This ensures the stack traces and annotations are preserved in the new subgraph
    for mr in match_and_replacement:
        # Find replacement conv and bn nodes by climbing upwards from anchor node
        assert len(mr.replacements) == 1, "expected only one replacement node"
        replacement_conv_node = None
        replacement_bn_node = None
        replacement_anchor = mr.replacements[0]
        assert replacement_anchor.target == operator.getitem
        n = replacement_anchor
        while replacement_conv_node is None or replacement_bn_node is None:
            if n.target == torch.ops.aten.convolution.default:
                replacement_conv_node = n
            if n.target == torch.ops.aten._native_batch_norm_legit.default:
                replacement_bn_node = n
            assert isinstance(n.args[0], Node)
            n = n.args[0]

        # Copy over metadata for conv node, bn node, conv args, bn args, and bn users
        for match_pattern_node, original_node in mr.nodes_map.items():
            if match_pattern_node.target == torch.ops.aten.convolution.default:
                # Matched: conv(_, weight, bias, ...)
                (_, weight, bias, *_) = match_pattern_node.args
                # Replaced: conv(_, mul(weight, ...), zeros_like(bias), ...)
                (_, mul_node, zeros_like_node, *_) = replacement_conv_node.args
                assert mul_node.target == torch.ops.aten.mul.Tensor
                assert zeros_like_node.target == torch.ops.aten.zeros_like.default
                mul_node.args[0].meta = mr.nodes_map[weight].meta
                zeros_like_node.args[0].meta = mr.nodes_map[bias].meta
                replacement_conv_node.meta = original_node.meta
            if match_pattern_node.target == torch.ops.aten._native_batch_norm_legit.default:
                (_, weight, bias, running_mean, running_var, *_) = match_pattern_node.args
                replacement_bn_node.args[1].meta = mr.nodes_map[weight].meta
                replacement_bn_node.args[2].meta = mr.nodes_map[bias].meta
                replacement_bn_node.args[3].meta = mr.nodes_map[running_mean].meta
                replacement_bn_node.args[4].meta = mr.nodes_map[running_var].meta
                replacement_bn_node.meta = original_node.meta
            if match_pattern_node.target == operator.getitem:
                replacement_anchor.meta = original_node.meta
    return m
