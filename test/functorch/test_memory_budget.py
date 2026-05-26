# Owner(s): ["module: functorch"]
import unittest

import torch
import torch.fx.traceback as fx_traceback
from torch._functorch._activation_checkpointing.memory_budget import (
    MemoryBudgetMode,
    set_memory_budget,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestSetMemoryBudget(TestCase):
    def setUp(self):
        fx_traceback.current_meta.pop("memory_budget", None)

    def tearDown(self):
        fx_traceback.current_meta.pop("memory_budget", None)

    def test_sets_budget_in_current_meta(self):
        set_memory_budget(0.5)
        self.assertEqual(fx_traceback.current_meta["memory_budget"], 0.5)

    def test_overwrites_previous_budget(self):
        set_memory_budget(0.3)
        set_memory_budget(0.8)
        self.assertEqual(fx_traceback.current_meta["memory_budget"], 0.8)

    def test_accepts_int_budget(self):
        set_memory_budget(0)
        self.assertEqual(fx_traceback.current_meta["memory_budget"], 0.0)
        set_memory_budget(1)
        self.assertEqual(fx_traceback.current_meta["memory_budget"], 1.0)

    def test_rejects_invalid_type(self):
        with self.assertRaises(TypeError):
            set_memory_budget("0.5")

    def test_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            set_memory_budget(-0.1)
        with self.assertRaises(ValueError):
            set_memory_budget(1.1)

    def test_boundary_values(self):
        set_memory_budget(0.0)
        self.assertEqual(fx_traceback.current_meta["memory_budget"], 0.0)
        set_memory_budget(1.0)
        self.assertEqual(fx_traceback.current_meta["memory_budget"], 1.0)


class TestMemoryBudgetMode(TestCase):
    def setUp(self):
        fx_traceback.current_meta.pop("memory_budget", None)

    def tearDown(self):
        fx_traceback.current_meta.pop("memory_budget", None)

    def test_sets_budget_on_enter(self):
        with MemoryBudgetMode(0.3):
            self.assertEqual(fx_traceback.current_meta["memory_budget"], 0.3)

    def test_clears_budget_on_exit(self):
        with MemoryBudgetMode(0.3):
            pass
        self.assertNotIn("memory_budget", fx_traceback.current_meta)

    def test_restores_previous_budget(self):
        set_memory_budget(0.5)
        with MemoryBudgetMode(0.3):
            self.assertEqual(fx_traceback.current_meta["memory_budget"], 0.3)
        self.assertEqual(fx_traceback.current_meta["memory_budget"], 0.5)

    def test_nested_modes(self):
        with MemoryBudgetMode(0.5):
            self.assertEqual(fx_traceback.current_meta["memory_budget"], 0.5)
            with MemoryBudgetMode(0.2):
                self.assertEqual(fx_traceback.current_meta["memory_budget"], 0.2)
            self.assertEqual(fx_traceback.current_meta["memory_budget"], 0.5)
        self.assertNotIn("memory_budget", fx_traceback.current_meta)

    def test_rejects_invalid_budget(self):
        with self.assertRaises(TypeError):
            MemoryBudgetMode("0.5")
        with self.assertRaises(ValueError):
            MemoryBudgetMode(-0.1)
        with self.assertRaises(ValueError):
            MemoryBudgetMode(1.1)

    def test_repr(self):
        mode = MemoryBudgetMode(0.3)
        self.assertEqual(repr(mode), "MemoryBudgetMode(budget=0.3)")

    def test_restores_on_exception(self):
        set_memory_budget(0.5)
        with self.assertRaises(RuntimeError):
            with MemoryBudgetMode(0.3):
                raise RuntimeError("test")
        self.assertEqual(fx_traceback.current_meta["memory_budget"], 0.5)


class TestMemoryBudgetCopyMetaFields(TestCase):
    def test_memory_budget_in_copy_meta_fields(self):
        import torch.fx.proxy as fx_proxy

        self.assertIn("memory_budget", fx_proxy._COPY_META_FIELDS)


@unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
class TestMemoryBudgetCompile(TestCase):
    def test_memory_budget_propagates_to_nodes(self):
        """Verify memory_budget metadata appears on FX graph nodes after compile."""

        class Model(torch.nn.Module):
            def forward(self, x):
                set_memory_budget(0.3)
                x = x.sin()
                set_memory_budget(0.8)
                x = x.cos()
                return x

        graphs = []

        def capture_backend(gm, example_inputs):
            graphs.append(gm)
            return gm

        model = Model()
        compiled = torch.compile(model, backend=capture_backend)
        compiled(torch.randn(4, device="cuda"))

        self.assertEqual(len(graphs), 1)
        gm = graphs[0]
        budgets = {
            node.name: node.meta.get("memory_budget")
            for node in gm.graph.nodes
            if node.meta.get("memory_budget") is not None
        }
        self.assertTrue(len(budgets) > 0)
        self.assertTrue(any(v == 0.3 for v in budgets.values()))
        self.assertTrue(any(v == 0.8 for v in budgets.values()))


if __name__ == "__main__":
    run_tests()
