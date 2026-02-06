import torch
from torch._higher_order_ops.compile_print import compile_print, make_compile_print
from torch.testing._internal.common_utils import run_tests, TestCase


class TestCompilePrint(TestCase):
    def test_compile_print_eager(self):
        fwd_values = []
        bwd_values = []

        def fwd_f(t):
            fwd_values.append(t.clone())

        def bwd_f(t):
            bwd_values.append(t.clone())

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)

        out = compile_print(fwd_f, bwd_f, x, y)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0], x)
        self.assertEqual(out[1], y)

        self.assertEqual(len(fwd_values), 2)
        self.assertEqual(fwd_values[0], x)
        self.assertEqual(fwd_values[1], y)

        loss = out[0].sum() + out[1].sum()
        loss.backward()

        self.assertEqual(len(bwd_values), 2)
        self.assertEqual(bwd_values[0], torch.ones_like(x))
        self.assertEqual(bwd_values[1], torch.ones_like(y))

    def test_compile_print_compiled_aot_eager(self):
        fwd_values = []
        bwd_values = []

        def fwd_f(t):
            fwd_values.append(t.clone())

        def bwd_f(t):
            bwd_values.append(t.clone())

        # Create the compile_print function ahead of time (before torch.compile)
        cp = make_compile_print(fwd_f, bwd_f)

        def fn(x, y):
            out = cp(x, y)
            return out[0].sum() + out[1].sum()

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)

        compiled_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        loss = compiled_fn(x, y)
        loss.backward()

        # Forward should have been called with both tensors
        self.assertEqual(len(fwd_values), 2)

        # Backward should have been called with gradients
        self.assertEqual(len(bwd_values), 2)

    def test_compile_print_no_grad_tensor(self):
        fwd_values = []
        bwd_values = []

        def fwd_f(t):
            fwd_values.append(t.clone())

        def bwd_f(t):
            bwd_values.append(t.clone())

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=False)

        out = compile_print(fwd_f, bwd_f, x, y)
        self.assertEqual(len(out), 2)

        # Both tensors should be captured in forward
        self.assertEqual(len(fwd_values), 2)

        loss = out[0].sum()
        loss.backward()

        # bwd_f is called on all grads (including None converted or not passed)
        self.assertGreaterEqual(len(bwd_values), 1)

    def test_make_compile_print_eager(self):
        fwd_values = []
        bwd_values = []

        def fwd_f(t):
            fwd_values.append(t.clone())

        def bwd_f(t):
            bwd_values.append(t.clone())

        cp = make_compile_print(fwd_f, bwd_f)

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)

        out = cp(x, y)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0], x)
        self.assertEqual(out[1], y)

        self.assertEqual(len(fwd_values), 2)

        loss = out[0].sum() + out[1].sum()
        loss.backward()

        self.assertEqual(len(bwd_values), 2)
        self.assertEqual(bwd_values[0], torch.ones_like(x))
        self.assertEqual(bwd_values[1], torch.ones_like(y))


    def test_compile_print_with_actual_printing(self):
        from unittest.mock import patch

        def fwd_f(t):
            print(f"Forward: shape={t.shape}, mean={t.mean().item():.4f}")

        def bwd_f(t):
            print(f"Backward: shape={t.shape}, mean={t.mean().item():.4f}")

        cp = make_compile_print(fwd_f, bwd_f)

        def fn(x, y):
            out = cp(x, y)
            return out[0].sum() + out[1].sum()

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)

        with patch("builtins.print") as mock_print:
            compiled_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
            loss = compiled_fn(x, y)
            loss.backward()

            # Check that print was called for forward (2 tensors) and backward (2 grads)
            call_args = [str(call) for call in mock_print.call_args_list]
            forward_calls = [c for c in call_args if "Forward" in c]
            backward_calls = [c for c in call_args if "Backward" in c]

            self.assertEqual(len(forward_calls), 2)
            self.assertEqual(len(backward_calls), 2)

    def test_compile_print_graph_contains_invoke_leaf_function(self):
        from torch._dynamo.testing import EagerAndRecordGraphs

        def fwd_f(t):
            pass

        def bwd_f(t):
            pass

        cp = make_compile_print(fwd_f, bwd_f)

        def fn(x):
            out = cp(x)
            return out[0].sum()

        x = torch.randn(3, 3, requires_grad=True)

        backend = EagerAndRecordGraphs()
        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        loss = compiled_fn(x)
        loss.backward()

        # Check that invoke_leaf_function appears in the graph
        graph_code = backend.graphs[0].code
        self.assertIn("invoke_leaf_function", graph_code)


if __name__ == "__main__":
    run_tests()
