import copy
from typing import Any, Tuple

import torch
import torch._dynamo
from torch.fx import (
    GraphModule,
    replace_pattern,
)
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
    the fused QAT subgraph equivalent.
    """
    m.graph.eliminate_dead_code()
    m.recompile()
    example_inputs = (torch.randn(1, 1, 3, 3),)
    match_pattern = _get_aten_graph_module(_ConvBnPattern(), example_inputs)
    replacement_pattern = _get_aten_graph_module(_FusedQATConvBnPattern(), example_inputs)
    replace_pattern(m, match_pattern, replacement_pattern)
    m.recompile()
    return m
