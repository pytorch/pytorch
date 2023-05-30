# Owner(s): ["oncall: quantization"]
import copy

import torch
import torch._dynamo as torchdynamo

from torch.ao.quantization._pt2e.graph_utils import find_sequential_partitions
from torch.testing._internal.common_utils import TestCase


class TestGraphUtils(TestCase):
    def test_conv_bn_conv_relu(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3)
                self.bn1 = torch.nn.BatchNorm2d(3)
                self.conv2 = torch.nn.Conv2d(3, 3, 3)
                self.relu2 = torch.nn.ReLU()

            def forward(self, x):
                bn_out = self.bn1(self.conv1(x))
                relu_out = torch.nn.functional.relu(bn_out)
                return self.relu2(self.conv2(relu_out))

        m = M().eval()
        example_inputs = (torch.randn(1, 3, 5, 5),)

        # program capture
        m, guards = torchdynamo.export(
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )
        fused_partitions = find_sequential_partitions(
            m, [torch.nn.Conv2d, torch.nn.BatchNorm2d]
        )
        self.assertEqual(len(fused_partitions), 1)
        fused_partitions = find_sequential_partitions(
            m, [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU]
        )
        self.assertEqual(len(fused_partitions), 1)

        def x():
            find_sequential_partitions(
                m,
                [
                    torch.nn.Conv2d,
                    torch.nn.BatchNorm2d,
                    torch.nn.ReLU,
                    torch.nn.functional.conv2d,
                ],
            )

        self.assertRaises(ValueError, x)

    def test_conv_bn_relu(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn1 = torch.nn.BatchNorm2d(3)
                self.conv2 = torch.nn.Conv2d(3, 3, 3)
                self.relu2 = torch.nn.ReLU()

            def forward(self, x):
                bn_out = self.bn1(x)
                return self.relu2(self.conv2(bn_out))

        m = M().eval()
        example_inputs = (torch.randn(1, 3, 5, 5),)

        # program capture
        m, guards = torchdynamo.export(
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )
        fused_partitions = find_sequential_partitions(
            m, [torch.nn.Conv2d, torch.nn.BatchNorm2d]
        )
        self.assertEqual(len(fused_partitions), 0)
        fused_partitions = find_sequential_partitions(
            m, [torch.nn.BatchNorm2d, torch.nn.Conv2d]
        )
        self.assertEqual(len(fused_partitions), 1)
        fused_partitions = find_sequential_partitions(
            m, [torch.nn.BatchNorm2d, torch.nn.ReLU]
        )
        self.assertEqual(len(fused_partitions), 0)
