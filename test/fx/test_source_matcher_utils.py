# Owner(s): ["module: fx"]

import os
import sys
import unittest

import torch

pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch._dynamo.eval_frame import is_dynamo_supported
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions, check_subgraphs_connected
from torch.testing._internal.jit_utils import JitTestCase

class TestSourceMatcher(JitTestCase):
    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    def test_module_partitioner_linear_relu_linear(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(3, 3)
                self.relu = torch.nn.ReLU()
                self.linear2 = torch.nn.Linear(3, 5)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                return x

        inputs = (torch.randn(3, 3),)
        gm, _ = torch._dynamo.export(M(), *inputs, aten_graph=True)
        gm.graph.eliminate_dead_code()

        module_partitions = get_source_partitions(gm.graph, [torch.nn.Linear, torch.nn.ReLU])

        self.assertEqual(len(module_partitions), 2)
        self.assertEqual(len(module_partitions[torch.nn.Linear]), 3)
        self.assertEqual(len(module_partitions[torch.nn.ReLU]), 1)

        self.assertFalse(check_subgraphs_connected(module_partitions[torch.nn.Linear][0], module_partitions[torch.nn.ReLU][0]))
        self.assertTrue(check_subgraphs_connected(module_partitions[torch.nn.Linear][1], module_partitions[torch.nn.ReLU][0]))
        self.assertFalse(check_subgraphs_connected(module_partitions[torch.nn.Linear][2], module_partitions[torch.nn.ReLU][0]))

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    def test_module_partitioner_conv_relu_maxpool(self):
        class M(torch.nn.Module):
            def __init__(self, constant_tensor: torch.Tensor) -> None:
                super().__init__()
                self.constant_tensor = constant_tensor
                self.conv1 = torch.nn.Conv2d(
                    in_channels=3, out_channels=16, kernel_size=3, padding=1
                )
                self.conv2 = torch.nn.Conv2d(
                    in_channels=16, out_channels=16, kernel_size=3, padding=1
                )
                self.conv3 = torch.nn.Conv2d(
                    in_channels=16, out_channels=16, kernel_size=3, padding=1
                )
                self.relu = torch.nn.ReLU()
                self.maxpool = torch.nn.MaxPool2d(kernel_size=3)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                a = self.conv1(x)
                b = self.conv2(a)
                c = a + self.constant_tensor
                z = self.conv3(b + c)
                return self.maxpool(self.relu(z))

        inputs = (torch.randn(1, 3, 256, 256),)
        gm, _ = torch._dynamo.export(M(torch.ones(1, 16, 256, 256)), *inputs, aten_graph=True)
        gm.graph.eliminate_dead_code()

        module_partitions = get_source_partitions(gm.graph, [torch.nn.Conv2d, torch.nn.ReLU, torch.nn.MaxPool2d])

        self.assertEqual(len(module_partitions), 3)
        self.assertEqual(len(module_partitions[torch.nn.Conv2d]), 3)
        self.assertEqual(len(module_partitions[torch.nn.ReLU]), 1)
        self.assertEqual(len(module_partitions[torch.nn.MaxPool2d]), 1)

        self.assertFalse(check_subgraphs_connected(module_partitions[torch.nn.Conv2d][0], module_partitions[torch.nn.ReLU][0]))
        self.assertFalse(check_subgraphs_connected(module_partitions[torch.nn.Conv2d][1], module_partitions[torch.nn.ReLU][0]))
        self.assertTrue(check_subgraphs_connected(module_partitions[torch.nn.Conv2d][2], module_partitions[torch.nn.ReLU][0]))
        self.assertFalse(check_subgraphs_connected(module_partitions[torch.nn.MaxPool2d][0], module_partitions[torch.nn.ReLU][0]))
        self.assertTrue(check_subgraphs_connected(module_partitions[torch.nn.ReLU][0], module_partitions[torch.nn.MaxPool2d][0]))

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    def test_module_partitioner_functional_conv_relu_conv(self):
        class FunctionalConv2d(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.stride = (1, 1)
                self.padding = (0, 0)
                self.dilation = (1, 1)
                self.groups = 1

            def forward(self, x, weight, bias):
                return torch.nn.functional.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = FunctionalConv2d()
                self.conv2 = FunctionalConv2d()

            def forward(self, x, weight, bias):
                x = self.conv1(x, weight, bias)
                x = torch.nn.functional.relu(x)
                x = self.conv2(x, weight, bias)
                return x

        inputs = (torch.randn(1, 3, 5, 5), torch.rand(3, 3, 3, 3), torch.rand(3))
        gm, _ = torch._dynamo.export(M(), *inputs, aten_graph=True)
        gm.graph.eliminate_dead_code()

        module_partitions = get_source_partitions(gm.graph, [torch.nn.functional.conv2d])

        self.assertEqual(len(module_partitions), 1)
        self.assertEqual(len(module_partitions[torch.nn.functional.conv2d]), 2)

    @unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
    def test_module_partitioner_functional_linear_relu_linear(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, weight, bias):
                x = torch.nn.functional.linear(x, weight, bias)
                x = torch.nn.functional.linear(x, weight, bias)
                x = torch.nn.functional.relu(x)
                x = torch.nn.functional.linear(x, weight, bias)
                x = torch.nn.functional.linear(x, weight, bias)
                x = torch.nn.functional.relu(x)
                return x

        inputs = (torch.randn(1, 5), torch.rand((5, 5)), torch.zeros(5))
        gm, _ = torch._dynamo.export(M(), *inputs, aten_graph=True)
        gm.graph.eliminate_dead_code()

        module_partitions = get_source_partitions(gm.graph, [torch.nn.functional.linear, torch.nn.functional.relu])

        self.assertEqual(len(module_partitions), 2)
        self.assertEqual(len(module_partitions[torch.nn.functional.linear]), 4)
        self.assertEqual(len(module_partitions[torch.nn.functional.relu]), 2)
