# Owner(s): ["oncall: fx"]

import collections
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

from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests

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


    def forward1(a, b, c):
        add = a + b
        add_1 = add +  b
        add_2 = add_1 + c
        relu_1 = add_2.relu()
        add_3 = add_1 + add_2
        add_4 = add_1 + relu_1 + add_3
        relu_2 = add_4.relu()
        add_5 = relu_2 + add_4
        add_6 = add_5 + add_4
        return add_4, add_6

    def forward2(a, b, _):
        add = a + b
        add_1 = add +  b
        relu_1 = add_1.relu() # blocked by this
        add_3 = add_1 + relu_1
        add_4 = add_1 + add_3
        return add_4, add_1

    def forward3(a, b, c):
        add = a + b
        add_1 = a + c
        add_2 = b + c
        return add, add_1, add_2

    def forward4(a, b, c):
        add = a + b
        add_1 = a + c
        add_2 = b + c
        return torch.where(add > 0, add_1, add_2)

    def forward5(a, b, c):
        # add should be fused right branch, as left branch is not supported
        add = a + 1

        # left branch
        relu = add.relu()
        # right branch
        add_1 = add + 2

        return relu, add_1

    def forward6(a, b, c):
        # add should have its own partition, as neither branchs are supported
        add = a + 1

        # left branch
        relu = add.relu()
        # right branch
        relu_1 = add.relu()

        return relu, relu_1

    def forward7(a, b, c):
        # both branches are supported, but add should be merged with right branch, as right branch is larger
        add = a + 1

        # left branch
        add_1 = add + 2

        # right branch is larger
        add_2 = add + 1
        add_3 = add_2 + 1

        return add_3, add_1

    def forward8(a, b, c):
         # both branches are in the same partition, add should join the same partition
        add = a + 1

        # left branch
        add_1 = add + 2

        # right branch
        add_2 = add + 1

        # left and right branch merges
        add_3 = add_2 + add_1

        return add_3

    def forward9(a, b, c):
        add = a + 1

        # branch 1
        add_1 = add + 1

        # branch 2
        add_2 = add + 1

        # branch_3
        add_3 = add + 1

        out = torch.stack([add_1, add_2, add_3])

        return out

    def forward10(a, b, c):
        add = a + 1

        # branch 1
        add_1 = add + 1

        # branch 2
        add_2 = add + 1

        # branch 3: depends on branch 2
        add_3 = add + add_2

        out = torch.stack([add_1, add_2, add_3])

        return out

    def forward11(a, b, c):
        add = a + 1

        # branch 1
        add_1 = add.relu()

        # branch 2 depends on branch 1
        add_2 = add + add_1

        # branch 3
        add_3 = add.relu()

        out = torch.stack([add_1, add_2, add_3])

        return out


    @parametrize("fn, expected_partition", [
        (forward1, [["add_7", "add_6"], ["add_5", "add_4", "add_3"], ["add_2", "add_1", "add"]]),
        (forward2, [["add_3", "add_2"], ["add_1", "add"]]),

        # 2 branches cases
        (forward5, [["add_1", "add"]]),
        (forward6, [["add"]]),
        (forward7, [["add_3", "add_2", "add"], ["add_1"]]),
        (forward8, [["add_3", "add_2", "add", "add_1"]]),

        # 3 branch cases
        (forward9, [['add_3'], ['add_2'], ['add_1', 'add']]),
        (forward10, [['add_3', 'add_2', 'add'], ['add_1']]),
        (forward11, [['add_1'], ['add']]),
    ])
    # failing cases
    # @parametrize("fn, expected_partition", [
    #     (forward3, [["add_2", "add_1", "add"]]),  # horizontal fusion without a common downstream node, not working yet
    #     (forward4, [["add_2", "add_1", "add"]]),  # horizontal fusion with a common downstream node, not working yet
    # ]
    def test_nvfuser_partition(self, fn, expected_partition):

        traced = symbolic_trace(fn)

        drawer = FxGraphDrawer(traced, "test")
        dot_graph = drawer.get_dot_graph()
        dot_graph.write_png("before.png")


        supported_ops = NvFuserOperatorSupport()
        partitioner = CapabilityBasedPartitioner(traced, supported_ops)

        candidates = partitioner.get_candidates()

        partitions = partitioner.partition(candidates)

        partitions_name = [[node.name for node in partition.nodes] for partition in partitions]

        # print("partitions_name", partitions_name)
        # print("expected_partition", expected_partition)

        assert len(partitions_name) == len(expected_partition)
        for i in range(len(partitions_name)):
            assert set(partitions_name[i]) == set(expected_partition[i])

        fused_graph = partitioner.fuse_partitions(partitions)

        drawer = FxGraphDrawer(fused_graph, "test")
        dot_graph = drawer.get_dot_graph()
        dot_graph.write_png("after.png")

        a, b, c = torch.rand(4), torch.rand(4), torch.rand(4)

        expected = fn(a, b, c)
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

        # drawer = FxGraphDrawer(traced, "test")
        # dot_graph = drawer.get_dot_graph()
        # dot_graph.write_png("before.png")

        for id, test_case in enumerate(test_cases):
            gm = copy.deepcopy(traced)
            nodes = gm.graph.nodes
            nodes_by_name = {node.name : node for node in nodes}

            partitions = []
            for names in test_case:
                partitions.append([nodes_by_name[name] for name in names])

            fused_graph = fuse_by_partitions(gm, partitions)

            # drawer = FxGraphDrawer(fused_graph, "test")
            # dot_graph = drawer.get_dot_graph()
            # dot_graph.write_png(f"after_{id}.png")

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


instantiate_parametrized_tests(TestFXGraphPasses)

if __name__ == "__main__":
    run_tests()
