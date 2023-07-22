# Owner(s): ["module: fx"]

from typing import Set, Type, Optional
import torch
import torch.fx

from torch.testing._internal.common_utils import TestCase


class TestDCE(TestCase):
    def _has_nodes_without_users(self, m: torch.fx.GraphModule):
        for node in m.graph.nodes:
            if node.is_impure():
                continue
            if len(node.users) == 0:
                return True
        return False

    def _get_num_placeholders(self, m: torch.fx.GraphModule) -> int:
        count = 0
        for node in m.graph.nodes:
            if node.op == "placeholder":
                count += 1
        return count

    def _run_dce_and_test(
        self,
        m: torch.nn.Module,
        expect_dce_changes: bool,
        modules_to_be_leafs: Optional[Set[Type]] = None,
    ):
        class TestTracer(torch.fx.Tracer):
            def is_leaf_module(self, m, qualname):
                if modules_to_be_leafs and type(m) in modules_to_be_leafs:
                    return True
                return super().trace(m, qualname)

        traced: torch.fx.GraphModule = torch.fx.GraphModule(m, TestTracer().trace(m))
        print(str(traced.graph))

        # Verify there are nodes without users (if expected).
        has_nodes_without_users = self._has_nodes_without_users(traced)
        if expect_dce_changes:
            self.assertTrue(has_nodes_without_users)
        else:
            self.assertFalse(has_nodes_without_users)

        # Get the original number of placeholders to verify it doesn't change
        # during DCE.
        orig_num_phs = self._get_num_placeholders(traced)
        changed = traced.graph.eliminate_dead_code()

        self.assertTrue(changed if expect_dce_changes else not changed)

        # Verify there are no nodes without users after DCE is run.
        self.assertFalse(self._has_nodes_without_users(traced))
        new_num_phs = self._get_num_placeholders(traced)
        self.assertEqual(orig_num_phs, new_num_phs)

        traced.recompile()
        # Make sure we run and get the same results before/after DCE.
        inputs = [torch.tensor([1.5])] * new_num_phs
        self.assertTrue(torch.equal(m(*inputs), traced(*inputs)))

    def test_simple(self):
        """
        Tests that a single node in the graph is DCE'd correctly.
        """

        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attr_1 = torch.nn.Parameter(torch.tensor([-0.9]))

            def forward(self, x):
                a = x + 1
                return x + self.attr_1

        self._run_dce_and_test(TestModule(), expect_dce_changes=True)

    def test_dead_chain(self):
        """
        Tests that a chain of two nodes in the graph are DCE'd correctly.
        """

        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attr_1 = torch.nn.Parameter(torch.tensor([-0.9]))

            def forward(self, x):
                a = x + 1
                b = a * 7
                return x + self.attr_1

        self._run_dce_and_test(TestModule(), expect_dce_changes=True)

    def test_dead_getattr(self):
        """
        Tests that a getatrr in the graph is DCE'd correctly.
        """

        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attr_1 = torch.nn.Parameter(torch.tensor([-0.9]))

            def forward(self, x):
                a = x + 1
                b = a * self.attr_1
                return x + 11

        self._run_dce_and_test(TestModule(), expect_dce_changes=True)

    def test_dead_placeholder(self):
        """
        Tests that a placeholder in the graph is not DCE'd, as that would change
        the function signature.
        """

        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                return x + 7

        self._run_dce_and_test(TestModule(), expect_dce_changes=False)

    def test_dead_placeholder_with_user(self):
        """
        Tests that a placeholder in the graph is not DCE'd, as that would change
        the function signature. Also verifies that a dead node that uses the
        placeholder is DCE'd.

        """

        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                a = y + 2
                return x + 7

        self._run_dce_and_test(TestModule(), expect_dce_changes=True)

    def test_keep_module_with_side_effects(self):
        """
        Test that DCE doesn't remove a module if it's specified as having side effects.
        """

        class ReLUImpure(torch.nn.ReLU):
            _is_impure = True

        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = ReLUImpure()

            def forward(self, a: torch.Tensor) -> torch.Tensor:
                r = self.relu(a)
                return a * 2

        self._run_dce_and_test(
            TestModule(), expect_dce_changes=False, modules_to_be_leafs={ReLUImpure}
        )

    def test_keep_torch_assert(self):
        """
        Test that DCE doesn't remove torch._assert since it has side effects.
        """

        class TestModule(torch.nn.Module):
            def forward(self, a: torch.Tensor) -> torch.Tensor:
                torch._assert(torch.equal(a, a), "a must equal a")
                return a * 2

        # Note: Don't need to specify torch._assert as having side effects
        # because it's known to.
        self._run_dce_and_test(TestModule(), expect_dce_changes=False)
