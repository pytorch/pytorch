# Owner(s): ["module: fx"]

import copy
import itertools
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
        model1 = BackboneModel()
        model1.eval()
        model2 = copy.deepcopy(model1)
        turn_on_efficient_conv_bn_eval(model2)
        model3 = copy.deepcopy(model1)
        model3 = torch.compile(model3)

        models = [model1, model2, model3]

        input = torch.randn(64, 6, 32, 32)

        outputs = []
        grads = []
        for model in models:
            output = model(input)
            outputs.append(output.clone())
            output.sum().backward()
            grads.append(
                [x.grad.clone() for x in model.parameters() if x.grad is not None]
            )
            model.zero_grad()

        for output_a, output_b in itertools.product(outputs, outputs):
            assert torch.allclose(output_a, output_b, atol=1e-6)

        for grad_a, grad_b in itertools.product(grads, grads):
            assert len(grad_a) == len(grad_b)
            for a, b in zip(grad_a, grad_b):
                assert torch.allclose(a, b, atol=1e-3)
