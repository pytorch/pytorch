import torch
from torch.fx.symbolic_trace import symbolic_trace
from torch.fx.experimental import GraphManipulation
from torch.fx.experimental.Partitioner import Partitioner, Device
from torch.fx.node import get_target_name
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

    def test_assert_no_msg(self):

        class M(torch.nn.Module):
            def forward(self, a, b):
                assert a == b
                return a + b
        m = M()
        traced = symbolic_trace(m)

        # Make sure the graph is well-formed
        traced.graph.lint(traced)

        # Check the IR to make sure there's a call_function node with target == "Assert"
        self.assertTrue(any(node.op == 'call_function' and get_target_name(node.target) == "Assert" for node in traced.graph.nodes))

        # Ensure that the assert throws when it's supposed to and doesn't throw when it's not supposed to
        traced(3, 3)
        with self.assertRaisesRegex(AssertionError, ""):
            traced(3, 5)

    def test_call_to_assert_with_msg(self):

        class M(torch.nn.Module):
            def forward(self, a, b):
                assert a == b, "test message"
                return a + b
        m = M()
        traced = symbolic_trace(m)

        # Make sure the graph is well-formed
        traced.graph.lint(traced)

        # Check the IR to make sure there's a call_function node with target == "Assert"
        self.assertTrue(any(node.op == 'call_function' and get_target_name(node.target) == "Assert" for node in traced.graph.nodes))

        # Ensure that the assert throws when it's supposed to and doesn't throw when it's not supposed to
        traced(3, 3)
        with self.assertRaisesRegex(AssertionError, "test message"):
            traced(3, 5)

    def test_call_to_assert_with_empty_msg(self):

        class M(torch.nn.Module):
            def forward(self, a, b):
                assert a == b, ""
                return a + b
        m = M()
        traced = symbolic_trace(m)

        # Make sure the graph is well-formed
        traced.graph.lint(traced)

        # Check the IR to make sure there's a call_function node with target == "Assert"
        self.assertTrue(any(node.op == 'call_function' and get_target_name(node.target) == "Assert" for node in traced.graph.nodes))

        # Ensure that the assert throws when it's supposed to and doesn't throw when it's not supposed to
        traced(3, 3)
        with self.assertRaisesRegex(AssertionError, ""):
            traced(3, 5)

    def test_call_to_assert_with_multiline_message(self):

        class M(torch.nn.Module):
            def forward(self, a, b):
                error_msg = """
An error message with
terrible spacing
                """
                assert a == b, error_msg
                return a + b
        m = M()
        traced = symbolic_trace(m)

        # Make sure the graph is well-formed
        traced.graph.lint(traced)

        # Check the IR to make sure there's a call_function node with target == "Assert"
        self.assertTrue(any(node.op == 'call_function' and get_target_name(node.target) == "Assert" for node in traced.graph.nodes))

        # Ensure that the assert throws when it's supposed to and doesn't throw when it's not supposed to
        error_msg = """
An error message with
terrible spacing
    """
        traced(3, 3)
        with self.assertRaisesRegex(AssertionError, error_msg):
            traced(3, 5)

    def test_traceable_function_with_nonstandard_name(self):
        def foo(x):
            return torch.relu(x)
        traced = symbolic_trace(foo)

if __name__ == '__main__':
    run_tests()
