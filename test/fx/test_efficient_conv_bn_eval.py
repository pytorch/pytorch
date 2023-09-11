# Owner(s): ["module: fx"]

import torch
import torch.fx
from torch import nn
from torch.fx.experimental.efficient_conv_bn_eval import (
    efficient_conv_bn_eval_graph_transform,
)
from torch.testing._internal.common_utils import TestCase


class BackboneModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(16, 16, 6)
        self.bn1 = nn.BatchNorm2d(16)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        return x


class TestEfficientConvBNEval(TestCase):
    def test_efficient_conv_bn_eval(self):
        net = BackboneModel().eval()
        gm = torch.fx.symbolic_trace(net)
        efficient_conv_bn_eval_graph_transform(gm)
        for node in gm.graph.nodes:
            assert node.op != "call_module"
        input = torch.rand(64, 16, 32, 32)
        output1 = net(input)
        output2 = gm(input)
        self.assertTrue(torch.allclose(output1, output2, atol=1e-5))
