# Owner(s): ["oncall: fx"]

import math
import numbers
import operator
import pickle
import sys
import tempfile
import unittest
from typing import Callable, Dict, Union, List, Optional
from types import BuiltinFunctionType

import torch
import torch.fx.experimental.optimization as optimization
from torch.fx._symbolic_trace import symbolic_trace
from torch.fx.passes import graph_manipulation


import torch.fx.experimental.meta_tracer
from torch.fx.experimental.proxy_tensor import make_fx

from torch.fx.partitioner.partitioner import CapabilityBasedPartitioner
from torch.fx.partitioner.nvfuser_operator_support import NvFuserOperatorSupport
import torch._prims as prims
from torch._prims.executor import make_traced
from torch.fx.passes.graph_drawer import FxGraphDrawer
import copy

from torch.fx.passes.shape_prop import _extract_tensor_metadata, ShapeProp
from torch.fx.passes.fuser_utils import fuse_by_partitions

from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.jit_utils import JitTestCase

try:
    import torchvision.models
    from torchvision.models import resnet18

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")
skipIfNoMkldnn = unittest.skipIf(
    not (torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available()),
    "no MKLDNN",
)

class TestFXGraphPasses(JitTestCase):

    # class TestModule0(torch.nn.Module):
    #     def forward(self, a, b):
    #         add_1 = a + b
    #         add_2 = add_1 + torch.rand(4)
    #         add_3 = add_2 + torch.rand(4)
    #         return add_3

    # class TestModule1(torch.nn.Module):
    #     def __init__(self):
    #         super(TestModule1, self).__init__()
    #         self.linear = torch.nn.Linear(4, 4)

    #     def forward(self, a):
    #         add_1 = a + torch.rand(4)
    #         add_2 = add_1 + torch.rand(4)
    #         linear_1 = self.linear(add_1)
    #         add_3 = add_2 + linear_1
    #         add_4 = add_2 + add_3
    #         return add_4

    # torchvision.models.resnet18,
    # torchvision.models.resnet50,
    # torchvision.models.densenet121,
    # torchvision.models.shufflenet_v2_x1_0,
    # torchvision.models.vgg16,
    # torchvision.models.mobilenet_v2,
    # torchvision.models.mnasnet1_0,
    # torchvision.models.resnext50_32x4d,

    def test_nvfuser_partition(self):
        class TestModule2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.linear2 = torch.nn.Linear(4, 4)
                self.param = torch.nn.Parameter(torch.rand(4, 4))

            def forward(self, a, b, c):
                add = a + b

                linear_1 = self.linear(add)

                add_1 = add + c
                add_2 = add_1 + self.param
                add_3 = add_1 + linear_1
                add_4 = add_2 + add_3

                linear_2 = self.linear2(add_4)

                add_5 = linear_2 + add_4
                add_6 = add_5 + a

                return add_4, add_6

        m = TestModule2()
        traced = symbolic_trace(m)

        supported_ops = NvFuserOperatorSupport()
        partitioner = CapabilityBasedPartitioner(traced, supported_ops)

        candidates = partitioner.get_candidates()

        partitions = partitioner.partition(candidates)


        print(partitions)


        drawer = FxGraphDrawer(traced, "test")
        dot_graph = drawer.get_dot_graph()
        dot_graph.write_png("before.png")

        # module_with_submodules = split_module(traced, m, lambda node: assignment[node] if node in assignment else -1)

        fused_graph = fuse_by_partitions(traced, partitions)

        drawer = FxGraphDrawer(fused_graph, "test")
        dot_graph = drawer.get_dot_graph()
        dot_graph.write_png("after.png")

        a, b, c = torch.rand(4), torch.rand(4), torch.rand(4)

        expected = m(a, b, c)
        result = fused_graph(a, b, c)

        torch.testing.assert_close(expected, result)


    def test_fuser_util(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.linear2 = torch.nn.Linear(4, 4)
                self.param = torch.nn.Parameter(torch.rand(4, 4))

            def forward(self, a, b, c):
                add = a + b

                linear_1 = self.linear(add)

                add_1 = add + c
                add_2 = add_1 + self.param
                add_3 = add_1 + linear_1
                add_4 = add_2 + add_3

                linear_2 = self.linear2(add_4)

                add_5 = linear_2 + add_4
                add_6 = add_5 + a
                relu = add_6.relu()

                return add_4, add_6, relu

        m = TestModule()
        traced = symbolic_trace(m)

        print(traced.graph)

        # TODO: support for arbitrate node order
        test_cases = [
            [ ['add', 'add_1'], ['add_5', 'add_6'] ],
            [ ['add', 'add_1', 'add_2'] ],  # vertical fusion
            [ ['add_2', 'add_3'] ],         # horizontal fusion
            [ ['add_3', 'add_4'] ],
            [ ['add_5', 'add_6'], ['add_1', 'add_2', 'add_3', 'add_4'] ],  # arbitray partition order
            [ ['add_6', 'add_5'] ],     # arbitray node order
            [ ['add_4', 'add_1', 'add_3', 'add_2'] ],           # arbitray node order
            [ ['add_5', 'linear2' ] ],   # includes call_function + call_module node
            [ ['add_6', 'relu' ] ],   # includes call_function + call_module node
            [ ['param', 'add_2' ] ],   # includes get_attr + call_module nodes
            [ ['param', 'add_1', 'linear' ] ],   # includes get_attr + call_function + call_module nodes
            [ ["add", "linear", "add_1", "param", "add_2", "add_3", "add_4", "linear2", "add_5", "add_6", "relu"] ], # full graph
        ]

        # expected failing cases
        x_test_cases = [
            [ ['add', 'add_1'], ['add_1', 'add_5', 'add_6'] ],  # add_1 exists in multiple partitions
            [ ['add', 'add_1', 'add_3'] ],    # invalid partition: circular dependency
            [ ['add_4', 'add_5'] ],    # invalid partition: circular dependency
            [ ['relu', 'add_5'] ],    # invalid partition: circular dependency
        ]

        drawer = FxGraphDrawer(traced, "test")
        dot_graph = drawer.get_dot_graph()
        dot_graph.write_png("before.png")

        for id, test_case in enumerate(test_cases):
            gm = copy.deepcopy(traced)
            nodes = gm.graph.nodes
            nodes_by_name = {node.name : node for node in nodes}

            partitions = []
            for names in test_case:
                partitions.append([nodes_by_name[name] for name in names])

            fused_graph = fuse_by_partitions(gm, partitions)

            drawer = FxGraphDrawer(fused_graph, "test")
            dot_graph = drawer.get_dot_graph()
            dot_graph.write_png(f"after_{id}.png")

            a, b, c = torch.rand(4), torch.rand(4), torch.rand(4)

            expected = m(a, b, c)
            result = fused_graph(a, b, c)

            torch.testing.assert_close(expected, result)

    def test_nvfuser_operator_support(self):
        def _wrapper(a, b, broadcast_dimensions):
            a_bc = prims.broadcast_in_dim(a, b.shape, broadcast_dimensions)
            return prims.add(a_bc, b)

        traced = symbolic_trace(_wrapper)

        supported_ops = NvFuserOperatorSupport()
        for node in traced.graph.nodes:
            assert supported_ops.is_node_supported({}, node)


    def test_multiple_outputs(self):
        class TestModule2(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b, c):
                add = a + b
                add_1 = b + c

                return add, add_1

        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = TestModule2()

            def forward(self, a, b, c):
                o0, o1 = self.m(a,b,c)
                x = o0 + o1
                return x

        m = TestModule()

        class TestTracer(torch.fx.Tracer):
            def is_leaf_module(self, module, name):
                return True

        t = TestTracer()

        traced = t.trace(m)

        print(traced)

if __name__ == "__main__":
    run_tests()
