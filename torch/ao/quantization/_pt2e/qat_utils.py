import copy
import operator
from typing import Any, Tuple

import torch
import torch._dynamo
from torch.fx import GraphModule, Node
from torch.fx.subgraph_rewriter import _replace_pattern
import torch.nn as nn


class _ConvBnPattern(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)
        self.bn = nn.BatchNorm2d(1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class _FusedQATConvBnPattern(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)
        self.bn = nn.BatchNorm2d(1)

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    def forward(self, input):
        """
        Approximated method to fuse conv and bn. It requires only one forward pass.
        conv_orig = conv / scale_factor where scale_factor = bn.weight / running_std
        """
        # TODO: deduplicate from nniqat.ConvBn2d
        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1
        # TODO: wrap this in weight_fake_quant
        scaled_weight = self.weight * scale_factor.reshape(weight_shape)
        zero_bias = torch.zeros_like(self.bias, dtype=input.dtype)
        conv = self.conv._conv_forward(input, scaled_weight, zero_bias)
        conv_orig = conv / scale_factor.reshape(bias_shape)
        conv_orig = conv_orig + self.bias.reshape(bias_shape)
        conv = self.bn(conv_orig)
        return conv

def _get_aten_graph_module(
        pattern: nn.Module,
        example_inputs: Tuple[Any, ...],
    ):
    """
    Convert the nn.Module pattern to an FX graph with decomposed aten ops.
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
    example_inputs = (torch.randn(1, 1, 3, 3),)
    match_pattern = _get_aten_graph_module(_ConvBnPattern(), example_inputs)
    replacement_pattern = _get_aten_graph_module(_FusedQATConvBnPattern(), example_inputs)
    match_and_replacement = _replace_pattern(m, match_pattern, replacement_pattern)
    m.recompile()

    # Copy over metadata from original subgraph
    for mr in match_and_replacement:
        # Find replaced conv and bn nodes by climbing upwards from anchor node
        assert len(mr.replacements) == 1, "expected only one replaced node"
        replaced_conv_node = None
        replaced_bn_node = None
        replaced_anchor = mr.replacements[0]
        assert replaced_anchor.target == operator.getitem
        n = replaced_anchor
        while replaced_conv_node is None or replaced_bn_node is None:
            if n.target == torch.ops.aten.convolution.default:
                replaced_conv_node = n
            if n.target == torch.ops.aten._native_batch_norm_legit.default:
                replaced_bn_node = n
            n = n.args[0]
            assert isinstance(n, Node)

        # Copy over metadata for conv node, bn node, conv args, bn args, and bn users
        for match_pattern_node, original_node in mr.nodes_map.items():
            if match_pattern_node.target == torch.ops.aten.convolution.default:
                # Matched: conv(_, weight, bias, ...)
                (_, weight, bias, *_) = match_pattern_node.args
                # Replaced: conv(_, mul(weight, ...), zeros_like(bias), ...)
                (_, mul_node, zeros_like_node, *_) = replaced_conv_node.args
                assert mul_node.target == torch.ops.aten.mul.Tensor
                assert zeros_like_node.target == torch.ops.aten.zeros_like.default
                mul_node.args[0].meta = mr.nodes_map[weight].meta
                zeros_like_node.args[0].meta = mr.nodes_map[bias].meta
                replaced_conv_node.meta = original_node.meta
            if match_pattern_node.target == torch.ops.aten._native_batch_norm_legit.default:
                (_, weight, bias, running_mean, running_var, *_) = match_pattern_node.args
                replaced_bn_node.args[1].meta = mr.nodes_map[weight].meta
                replaced_bn_node.args[2].meta = mr.nodes_map[bias].meta
                replaced_bn_node.args[3].meta = mr.nodes_map[running_mean].meta
                replaced_bn_node.args[4].meta = mr.nodes_map[running_var].meta
                replaced_bn_node.meta = original_node.meta
            if match_pattern_node.target == operator.getitem:
                replaced_anchor.meta = original_node.meta
    return m
