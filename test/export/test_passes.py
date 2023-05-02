# Owner(s): ["module: dynamo"]
import unittest

import torch
import torch._dynamo as torchdynamo
from torch.testing._internal.common_utils import run_tests, TestCase
from torch._dynamo.eval_frame import is_dynamo_supported
from torch._export import _export
from torch._export import dynamic_dim
from torch._export.passes import (
    AddRuntimeAssertionsForConstraintsPass,
    ConstPropPass,
    ReplaceBrokenOpsWithFunctionalOpsPass,
)


def count_call_function(graph: torch.fx.Graph, target: torch.ops.OpOverload) -> int:
    count = 0
    for node in graph.nodes:
        if node.op == "call_function" and node.target == target:
            count += 1
    return count


@unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
class TestPasses(TestCase):
    def test_replace_broken_ops(self) -> None:
        x = torch.randn([2, 3, 4, 5])
        model: torch.nn.Linear = torch.nn.Linear(5, 5)

        def f(inp: torch.Tensor) -> torch.Tensor:
            return model(inp)

        gm, _ = torchdynamo.export(f, x, aten_graph=True)

        new_gm = ReplaceBrokenOpsWithFunctionalOpsPass()(gm)
        self.assertIsNotNone(new_gm)
        new_gm = new_gm.graph_module

        count_after = 0
        for node in new_gm.graph.nodes:
            if node.target == torch.ops.aten.view.default:
                count_after += 1
        self.assertEqual(count_after, 0)
        self.assertTrue(torch.allclose(gm(x), f(x)))

    def test_const_prop_pass(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Parameter(torch.ones(1, 2, 3))

            def forward(self, x):
                b = self.a + self.a
                c = torch.cat([self.a, b])
                return (c + c) + x

        def count_additions(gm) -> int:
            return sum(
                (node.target == torch.ops.aten.add.Tensor) for node in gm.graph.nodes
            )

        gm, _ = torchdynamo.export(M(), torch.zeros(2, 2, 3), aten_graph=True)
        self.assertEqual(count_additions(gm), 3)

        new_gm = ConstPropPass()(gm)
        self.assertIsNotNone(new_gm)
        new_gm = new_gm.graph_module
        self.assertEqual(count_additions(new_gm), 1)

    def test_runtime_assert_one_dim(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.cos()

        x = torch.zeros(2, 2, 3)

        gm = _export(M(), (x,), constraints=[dynamic_dim(x, 1) >= 2, dynamic_dim(x, 1) <= 6])

        pass_result = AddRuntimeAssertionsForConstraintsPass()(gm)
        self.assertTrue(pass_result.modified)

        num_assert = count_call_function(pass_result.graph_module.graph, torch.ops.aten._assert_async.msg)
        num_scalar_tensor = count_call_function(pass_result.graph_module.graph, torch.ops.aten.scalar_tensor.default)

        self.assertEqual(num_assert, 3)
        self.assertEqual(num_scalar_tensor, 3)

        with self.assertRaisesRegex(RuntimeError, "Input #0"):
            pass_result.graph_module(torch.zeros(2, 7, 3))

        self.assertEqual(pass_result.graph_module(torch.ones(2, 4, 3)), M().forward(torch.ones(2, 4, 3)))

    def test_runtime_assert_multiple_dims(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x.cos().sum() + y.sin().sum()

        x = torch.zeros(4, 2, 3)
        y = torch.zeros(5, 5, 5)

        constraints = [
            dynamic_dim(x, 1) >= 2,
            dynamic_dim(x, 1) <= 6,
            dynamic_dim(y, 0) >= 3,
            dynamic_dim(x, 0) >= 3
        ]

        gm = _export(M(), (x, y), constraints=constraints)

        pass_result = AddRuntimeAssertionsForConstraintsPass()(gm)
        self.assertTrue(pass_result.modified)

        num_assert = count_call_function(pass_result.graph_module.graph, torch.ops.aten._assert_async.msg)
        num_scalar_tensor = count_call_function(pass_result.graph_module.graph, torch.ops.aten.scalar_tensor.default)

        self.assertEqual(num_assert, 6)
        self.assertEqual(num_scalar_tensor, 6)

        with self.assertRaisesRegex(RuntimeError, "Input #0"):
            pass_result.graph_module(torch.zeros(4, 7, 3), torch.ones(5, 5, 5))

        with self.assertRaisesRegex(RuntimeError, "Input #1"):
            pass_result.graph_module(torch.zeros(4, 2, 3), torch.ones(2, 5, 5))

    def test_runtime_assert_some_dims_not_specified(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x.cos().sum() + y.sin().sum()

        x = torch.zeros(4, 2, 3)
        y = torch.zeros(5, 5, 5)

        constraints = [
            dynamic_dim(x, 1) >= 2,
            dynamic_dim(x, 1) <= 6,
            dynamic_dim(x, 0) >= 3
        ]

        gm = _export(M(), (x, y), constraints=constraints)

        pass_result = AddRuntimeAssertionsForConstraintsPass()(gm)
        self.assertTrue(pass_result.modified)

        num_assert = count_call_function(pass_result.graph_module.graph, torch.ops.aten._assert_async.msg)
        num_scalar_tensor = count_call_function(pass_result.graph_module.graph, torch.ops.aten.scalar_tensor.default)

        # there are 3 asserts from y and 2 from dynamic x dims and 1 from static x dim
        self.assertEqual(num_assert, 6)
        self.assertEqual(num_scalar_tensor, 6)

        with self.assertRaisesRegex(RuntimeError, "Input #0"):
            pass_result.graph_module(torch.zeros(4, 7, 3), torch.ones(5, 5, 5))

        # y is specialized to 5
        with self.assertRaisesRegex(RuntimeError, "Input #1's dimension #0 size is specialized at 5"):
            pass_result.graph_module(torch.zeros(4, 2, 3), torch.ones(2, 5, 5))

        # Since we didn't insert the constraint for x[1] >= 2, it should work for case where x[1] == 1
        gm_result_for_1_size = pass_result.graph_module(torch.ones(3, 1, 3), torch.ones(5, 5, 5))
        eager_result_for_1_size = M().forward(torch.ones(3, 1, 3), torch.ones(5, 5, 5))

        self.assertEqual(gm_result_for_1_size, eager_result_for_1_size)

    def test_runtime_assert_some_inps_not_used(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return y.cos().sum()

        x = torch.zeros(4, 2, 3)
        y = torch.zeros(5, 5, 5)

        constraints = [
            dynamic_dim(y, 1) >= 3,
            dynamic_dim(y, 1) <= 6,
        ]

        gm = _export(M(), (x, y), constraints=constraints)

        pass_result = AddRuntimeAssertionsForConstraintsPass()(gm)
        self.assertTrue(pass_result.modified)

        num_assert = count_call_function(pass_result.graph_module.graph, torch.ops.aten._assert_async.msg)
        num_scalar_tensor = count_call_function(pass_result.graph_module.graph, torch.ops.aten.scalar_tensor.default)

        # there are 4 asserts from y and 3 from x
        self.assertEqual(num_assert, 7)
        self.assertEqual(num_scalar_tensor, 7)

        with self.assertRaisesRegex(RuntimeError, "Input #0"):
            pass_result.graph_module(torch.zeros(4, 7, 3), torch.ones(5, 5, 5))

        # y is specialized to 5
        with self.assertRaisesRegex(RuntimeError, "Input #1's dimension #0 size is specialized at 5"):
            pass_result.graph_module(torch.zeros(4, 2, 3), torch.ones(2, 5, 5))

        # Since we didn't insert the constraint for x[1] >= 2, it should work for case where x[1] == 1
        gm_result_for_1_size = pass_result.graph_module(torch.zeros(4, 2, 3), torch.ones(5, 5, 5))
        eager_result_for_1_size = M().forward(torch.zeros(4, 2, 3), torch.ones(5, 5, 5))

        self.assertEqual(gm_result_for_1_size, eager_result_for_1_size)


if __name__ == '__main__':
    run_tests()
