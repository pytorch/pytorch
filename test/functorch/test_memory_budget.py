# Owner(s): ["module: activation checkpointing"]

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase
from torch._functorch._activation_checkpointing.memory_budget import memory_budget


class TestMemoryBudgetValidation(TestCase):
    def test_rejects_invalid_type(self):
        with self.assertRaises(TypeError):
            with memory_budget("0.5"):
                pass

    def test_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            with memory_budget(1.5):
                pass
        with self.assertRaises(ValueError):
            with memory_budget(-0.1):
                pass

    def test_boundary_values(self):
        with memory_budget(0.0):
            pass
        with memory_budget(1.0):
            pass
        with memory_budget(1):
            pass


def _get_budgets_by_target(gm):
    """Extract {op_name: [budgets]} from a GraphModule."""
    result = {}
    for node in gm.graph.nodes:
        if node.op not in ("call_function", "call_method"):
            continue
        b = node.meta.get("custom", {}).get("memory_budget", None)
        if b is not None:
            name = getattr(node.target, "__name__", str(node.target))
            result.setdefault(name, []).append(b)
    return result


class TestMemoryBudgetCompile(TestCase):
    def test_single_budget(self):
        budgets = {}

        def capture(gm, example_inputs):
            budgets.update(_get_budgets_by_target(gm))
            return gm.forward

        @torch.compile(backend=capture)
        def fn(x):
            with memory_budget(0.3):
                return x.sin().cos()

        fn(torch.randn(4))
        self.assertEqual(budgets["sin"], [0.3])
        self.assertEqual(budgets["cos"], [0.3])

    def test_nesting(self):
        budgets = {}

        def capture(gm, example_inputs):
            budgets.update(_get_budgets_by_target(gm))
            return gm.forward

        @torch.compile(backend=capture)
        def fn(x):
            with memory_budget(0.3):
                x = x.sin()
                with memory_budget(0.8):
                    x = x.cos()
                x = x.relu()
            return x

        fn(torch.randn(4))
        self.assertEqual(budgets["sin"], [0.3])
        self.assertEqual(budgets["cos"], [0.8])
        self.assertEqual(budgets["relu"], [0.3])

    def test_graph_break(self):
        graphs = []

        def capture(gm, example_inputs):
            graphs.append(_get_budgets_by_target(gm))
            return gm.forward

        @torch.compile(backend=capture)
        def fn(x):
            with memory_budget(0.3):
                x = x.sin()
            torch._dynamo.graph_break()
            with memory_budget(0.8):
                x = x.cos()
            return x

        fn(torch.randn(4))
        self.assertEqual(len(graphs), 2)
        self.assertEqual(graphs[0]["sin"], [0.3])
        self.assertEqual(graphs[1]["cos"], [0.8])

    def test_propagates_to_joint_graph(self):
        from functorch.compile import nop
        from torch._dynamo.backends.common import aot_autograd

        joint_budgets = {}

        def my_partition(joint_module, _joint_inputs, *, num_fwd_outputs, **kwargs):
            for node in joint_module.graph.nodes:
                b = node.meta.get("custom", {}).get("memory_budget", None)
                if b is not None:
                    joint_budgets.setdefault(getattr(node.target, "__name__", str(node.target)), []).append(b)
            from torch._functorch.partitioners import (
                min_cut_rematerialization_partition,
            )
            return min_cut_rematerialization_partition(
                joint_module, _joint_inputs, num_fwd_outputs=num_fwd_outputs, **kwargs
            )

        backend = aot_autograd(
            fw_compiler=nop, bw_compiler=nop, partition_fn=my_partition,
        )

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(16, 16)
                self.linear2 = nn.Linear(16, 16)

            def forward(self, x):
                with memory_budget(0.3):
                    x = self.linear1(x).sin()
                with memory_budget(0.8):
                    x = self.linear2(x).cos()
                return x

        model = Model()
        compiled = torch.compile(model, backend=backend)
        x = torch.randn(4, 16, requires_grad=True)
        compiled(x).sum().backward()

        budget_values = set()
        for vals in joint_budgets.values():
            budget_values.update(vals)
        self.assertIn(0.3, budget_values)
        self.assertIn(0.8, budget_values)


if __name__ == "__main__":
    run_tests()
