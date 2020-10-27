import torch
from torch.fx.symbolic_trace import symbolic_trace
from torch.fx.experimental import GraphManipulation
from torch.fx.experimental.Partitioner import Partitioner, Device
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.jit_utils import JitTestCase

class TestFXExperimental(JitTestCase):
    def test_find_single_partition(self):
        class TestModule(torch.nn.Module):
            def forward(self, a, b):
                return a + b
        m = TestModule()
        traced = symbolic_trace(m)
        a = torch.rand(1)
        b = torch.rand(1)
        GraphManipulation.get_size_of_all_nodes(
            traced,
            [a, b]
        )
        partitioner = Partitioner()
        devices = [
            Device('dev_0', 125),
            Device('dev_1', 125),
            Device('dev_2', 125)
        ]
        ret = partitioner.partition_graph(traced, m, devices)
        module_with_submodules = ret.module_with_submodules
        self.assertEqual(traced(a, b), module_with_submodules(a, b))

    def test_size_based_partition(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, a, b):
                add_1 = a + b
                linear = self.linear(add_1)
                e = torch.rand(4)
                add_2 = linear + e
                return add_2

        m = TestModule()
        traced = symbolic_trace(m)
        a = torch.rand(4)
        b = torch.rand(4)
        GraphManipulation.get_size_of_all_nodes(
            traced,
            [a, b]
        )
        partitioner = Partitioner()
        devices = [
            Device('dev_0', 125),
            Device('dev_1', 125),
            Device('dev_2', 125)
        ]
        ret = partitioner.partition_graph(traced, m, devices)
        module_with_submodules = ret.module_with_submodules
        self.assertEqual(traced(a, b), module_with_submodules(a, b))
        assert len(module_with_submodules.graph.nodes) == 7

    def test_partition_combining(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_0 = torch.nn.Linear(4, 4)

            def forward(self, a, b):
                add_1 = a + b
                c = self.linear_0(a)
                add_2 = c + add_1
                return add_2

        m = TestModule()
        traced = symbolic_trace(m)
        a = torch.rand(4)
        b = torch.rand(4)
        GraphManipulation.get_size_of_all_nodes(
            traced,
            [a, b]
        )
        partitioner = Partitioner()
        devices = [
            Device('dev_0', 125),
            Device('dev_1', 125),
            Device('dev_2', 125)
        ]
        ret = partitioner.partition_graph(traced, m, devices)
        module_with_submodules = ret.module_with_submodules
        self.assertEqual(traced(a, b), module_with_submodules(a, b))
        assert len(module_with_submodules.graph.nodes) == 5

if __name__ == '__main__':
    run_tests()
