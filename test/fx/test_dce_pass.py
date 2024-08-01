# Owner(s): ["module: fx"]

import copy
import unittest
from typing import Set, Type

import torch
import torch.fx
from torch.testing._internal.common_utils import IS_MACOS, TestCase


class TestDCE(TestCase):
    def _custom_is_impure_node(self, node: torch.fx.Node) -> bool:
        if node.is_impure():
            return True

        if node.op == "call_function":
            schema = getattr(node.target, "_schema", None)
            schema_mutable = schema is not None and schema.is_mutable
            return schema_mutable
        return False

    def _has_nodes_without_users(self, m: torch.fx.GraphModule, custom: bool = False):
        for node in m.graph.nodes:
            if (not custom and node.is_impure()) or (
                custom and self._custom_is_impure_node(node)
            ):
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
        modules_to_be_leafs: Set[Type] = None,
        custom: bool = False,
    ):
        class TestTracer(torch.fx.Tracer):
            def is_leaf_module(self, m, qualname):
                if modules_to_be_leafs and type(m) in modules_to_be_leafs:
                    return True
                return super().trace(m, qualname)

        traced: torch.fx.GraphModule = torch.fx.GraphModule(m, TestTracer().trace(m))
        print(str(traced.graph))

        # Verify there are nodes without users (if expected).
        has_nodes_without_users = self._has_nodes_without_users(traced, custom=custom)
        if expect_dce_changes:
            self.assertTrue(has_nodes_without_users)
        else:
            self.assertFalse(has_nodes_without_users)

        # Get the original number of placeholders to verify it doesn't change
        # during DCE.
        orig_num_phs = self._get_num_placeholders(traced)
        if custom:
            changed = traced.graph.eliminate_dead_code(
                is_impure_node=self._custom_is_impure_node
            )
        else:
            changed = traced.graph.eliminate_dead_code()

        self.assertTrue(changed if expect_dce_changes else not changed)

        # Verify there are no nodes without users after DCE is run.
        self.assertFalse(self._has_nodes_without_users(traced, custom=custom))
        new_num_phs = self._get_num_placeholders(traced)
        self.assertEqual(orig_num_phs, new_num_phs)

        traced.recompile()
        # Make sure we run and get the same results before/after DCE.
        inputs = [torch.tensor([1.5])] * new_num_phs
        inputs_copy = copy.deepcopy(inputs)
        self.assertTrue(torch.equal(m(*inputs), traced(*inputs_copy)))

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

    def test_impure_nodes_args(self):
        """
        Test that DCE doesn't remove call_function nodes with side effects.
        """

        class TestModule(torch.nn.Module):
            def forward(self, a: torch.Tensor) -> torch.Tensor:
                torch._ops.ops.aten.add_.Tensor(a, 1)
                return a * 2

        # %add_ node should not be removed because it has side effects.
        self._run_dce_and_test(TestModule(), expect_dce_changes=False, custom=True)

    def test_impure_kwargs(self):
        """
        Test that DCE doesn't remove call_function nodes with side effects on kwargs.
        """

        class TestModule(torch.nn.Module):
            def forward(self, a: torch.Tensor) -> torch.Tensor:
                b = a + 1
                torch._ops.ops.aten.add.out(b, b, out=a, alpha=2)
                return a

        # %add_out node should not be removed because it has side effects.
        self._run_dce_and_test(TestModule(), expect_dce_changes=False, custom=True)

    @unittest.skipIf(IS_MACOS, "Not working on macos")
    def test_keep_collectives(self):
        """
        Test that DCE doesn't remote collective ops even the results are not used.
        """

        from torch.testing._internal.distributed.fake_pg import FakeStore

        class TestModule(torch.nn.Module):
            def forward(
                self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
            ) -> torch.Tensor:
                d = torch.ops.aten.mul.Tensor(a, b)
                e = torch.ops.aten.mul.Tensor(a, c)
                future = torch.ops._c10d_functional.all_reduce.default(e, "sum", "0")
                synced_e = torch.ops._c10d_functional.wait_tensor.default(
                    future
                )  # synced_e is not used
                return d

        torch.distributed.init_process_group(
            backend="fake",
            world_size=2,
            rank=0,
            store=FakeStore(),
        )
        # collective nodes should not be removed because they have side effects.
        self._run_dce_and_test(TestModule(), expect_dce_changes=False, custom=False)
        torch.distributed.destroy_process_group()

    @unittest.skipIf(IS_MACOS, "Not working on macos")
    def test_keep_collectives_no_overload(self):
        """
        Test that DCE doesn't remote collective ops (no overload version) even the results are not used.
        """

        from torch.testing._internal.distributed.fake_pg import FakeStore

        class TestModule(torch.nn.Module):
            def forward(
                self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
            ) -> torch.Tensor:
                d = torch.ops.aten.mul(a, b)
                e = torch.ops.aten.mul(a, c)
                future = torch.ops._c10d_functional.all_reduce(e, "sum", "0")
                synced_e = torch.ops._c10d_functional.wait_tensor(
                    future
                )  # synced_e is not used
                return d

        torch.distributed.init_process_group(
            backend="fake",
            world_size=2,
            rank=0,
            store=FakeStore(),
        )
        # collective nodes should not be removed because they have side effects.
        self._run_dce_and_test(TestModule(), expect_dce_changes=False, custom=False)
        torch.distributed.destroy_process_group()
