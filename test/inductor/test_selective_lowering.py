# Owner(s): ["module: inductor"]
"""
Test selective lowering control via node metadata annotations.
"""

from collections.abc import Callable

import torch
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


@instantiate_parametrized_tests
class SelectiveLoweringTest(InductorTestCase):
    """
    Tests for user-controllable selective lowering using node.meta annotations.
    """

    device = GPU_TYPE

    def _mark_nodes_for_fallback(
        self, gm: torch.fx.GraphModule, predicate: Callable[[torch.fx.Node], bool]
    ) -> torch.fx.GraphModule:
        """
        Helper method to mark nodes with should_fallback metadata based on a predicate.
        """
        for node in gm.graph.nodes:
            if node.op == "call_function" and predicate(node):
                node.meta["should_fallback"] = True
        return gm

    def test_basic_selective_lowering(self):
        """
        Test that nodes marked for fallback use fallback handlers instead of lowerings.
        """

        def foo(x, y):
            a = x + y  # This will be marked for fallback
            b = a * 2  # This will use normal lowering
            return b

        x = torch.randn(10, device=self.device)
        y = torch.randn(10, device=self.device)

        def custom_backend(gm: torch.fx.GraphModule, example_inputs):
            # Mark all add operations for fallback
            def should_fallback_add(node: torch.fx.Node) -> bool:
                return node.target == torch.ops.aten.add.Tensor

            self._mark_nodes_for_fallback(gm, should_fallback_add)

            from torch._inductor.compile_fx import compile_fx

            return compile_fx(gm, example_inputs)

        compiled_fn = torch.compile(foo, backend=custom_backend)
        result = compiled_fn(x, y)
        expected = foo(x, y)

        self.assertTrue(torch.allclose(result, expected))

    def test_no_fallback_when_unmarked(self):
        """
        Test that operations without fallback annotation use normal lowering.
        """

        def foo(x, y):
            return x + y

        x = torch.randn(10, device=self.device)
        y = torch.randn(10, device=self.device)

        def custom_backend(gm: torch.fx.GraphModule, example_inputs):
            # Don't mark anything - all operations should use normal lowering
            from torch._inductor.compile_fx import compile_fx

            return compile_fx(gm, example_inputs)

        compiled_fn = torch.compile(foo, backend=custom_backend)
        result = compiled_fn(x, y)
        expected = foo(x, y)

        self.assertTrue(torch.allclose(result, expected))


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests(needs="filelock")
