import torch
import torch.nn as nn
from torch.func import functional_call
from torch._dynamo.test_case import TestCase


class TestFunctionalCallHigherOrderOps(TestCase):
    def test_cond_with_functional_call(self):
        """Test that functional_call works with torch.cond"""
        mod = nn.Linear(10, 10)
        params = dict(mod.named_parameters())
        buffers = dict(mod.named_buffers())
        x = torch.randn(10)

        def step_fn(x):
            return functional_call(mod, (params, buffers), (x,))

        def run_logic(pred, x):
            return torch.cond(
                pred,
                step_fn,       # True branch: calls functional_call
                lambda x: x,   # False branch: identity
                [x]            # Operands
            )

        # Should not raise UncapturedHigherOrderOpError
        fn = torch.compile(run_logic, fullgraph=True)
        result = fn(torch.tensor(True), x)
        self.assertEqual(result.shape, (10,))

        # Test false branch
        result_false = fn(torch.tensor(False), x)
        self.assertEqual(result_false.shape, (10,))
        self.assertTrue(torch.allclose(result_false, x))

    def test_scan_with_functional_call(self):
        """Test that functional_call works with torch.scan"""
        from torch._higher_order_ops.scan import scan

        mod = nn.Linear(10, 10)
        params = dict(mod.named_parameters())
        buffers = dict(mod.named_buffers())

        def combine_fn(carry, x):
            # Use functional_call in combine function
            y = functional_call(mod, (params, buffers), (x,))
            return carry + y, y

        init = torch.zeros(10)
        xs = torch.randn(5, 10)  # 5 steps

        def run_scan(xs):
            return scan(combine_fn, init, xs, additional_inputs=())

        # Should not raise UncapturedHigherOrderOpError
        fn = torch.compile(run_scan, fullgraph=True)
        carry, ys = fn(xs)
        self.assertEqual(carry.shape, (10,))
        self.assertEqual(ys.shape, (5, 10))

    def test_functional_call_preserves_behavior(self):
        """Ensure functional_call still works correctly outside higher-order ops"""
        mod = nn.Linear(10, 10)
        params = dict(mod.named_parameters())
        buffers = dict(mod.named_buffers())
        x = torch.randn(10)

        def fn(x):
            return functional_call(mod, (params, buffers), (x,))

        compiled_fn = torch.compile(fn, fullgraph=True)
        result = compiled_fn(x)

        # Compare with eager
        expected = functional_call(mod, (params, buffers), (x,))
        self.assertTrue(torch.allclose(result, expected))
