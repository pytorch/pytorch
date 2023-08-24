# Owner(s): ["module: fx"]

from unittest import TestCase

import torch
from torch import nn

from torch.fx.experimental.efficient_conv_bn_eval import turn_on_efficient_conv_bn_eval


class BackboneModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(6, 6, 6)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 6, 6)
        self.bn2 = nn.BatchNorm2d(6)
        self.conv3 = nn.Conv2d(6, 6, 6)
        self.bn3 = nn.BatchNorm2d(6)

    def forward(self, x):
        # this conv-bn pair can use efficient_conv_bn_eval feature
        x = self.bn1(self.conv1(x))
        # this conv-bn pair can use efficient_conv_bn_eval feature
        # only for the second `self.conv2` call.
        x = self.bn2(self.conv2(self.conv2(x)))
        # this conv-bn pair can use efficient_conv_bn_eval feature
        # just for the first forward of the `self.bn3`
        x = self.bn3(self.bn3(self.conv3(x)))
        return x


class TestEfficientConvBNEval(TestCase):
    """Test the turn_on_efficient_conv_bn_eval function."""

    def test_efficient_conv_bn_eval(self):
        model = BackboneModel()
        model.eval()
        input = torch.randn(64, 6, 32, 32)
        output = model(input)
        output.sum().backward()
        grads = [x.grad.clone() for x in model.parameters() if x.grad is not None]
        model.zero_grad()
        turn_on_efficient_conv_bn_eval(model)
        output2 = model(input)
        output2.sum().backward()
        grads2 = [x.grad.clone() for x in model.parameters() if x.grad is not None]
        assert torch.allclose(output, output2, atol=1e-6)
        assert len(grads) == len(grads2)
        for a, b in zip(grads, grads2):
            print((a - b).abs().max().item())
            assert torch.allclose(a, b, atol=1e-3)
