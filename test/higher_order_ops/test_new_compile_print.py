import torch
from torch._higher_order_ops.new_compile_print import (
    compile_print_fwd,
    compile_print_bwd,
    make_compile_print,
    new_compile_print,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestNewCompilePrint(TestCase):
    def test_new_compile_print_eager(self):
        fwd_values = []
        bwd_values = []

        def fwd_f(t):
            fwd_values.append(t.clone())

        def bwd_f(t):
            bwd_values.append(t.clone())

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)

        out = new_compile_print(fwd_f, bwd_f, x, y)
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

    def test_new_compile_print_compiled_aot_eager(self):
        fwd_values = []
        bwd_values = []

        def fwd_f(t):
            fwd_values.append(t.clone())

        def bwd_f(t):
            bwd_values.append(t.clone())

        cp = make_compile_print(fwd_f, bwd_f)

        def fn(x, y):
            out = cp(x, y)
            return out[0].sum() + out[1].sum()

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)

        compiled_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        loss = compiled_fn(x, y)
        loss.backward()

        self.assertEqual(len(fwd_values), 2)
        self.assertEqual(len(bwd_values), 2)

    def test_new_compile_print_no_grad_tensor(self):
        fwd_values = []
        bwd_values = []

        def fwd_f(t):
            fwd_values.append(t.clone())

        def bwd_f(t):
            bwd_values.append(t.clone())

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=False)

        out = new_compile_print(fwd_f, bwd_f, x, y)
        self.assertEqual(len(out), 2)
        self.assertEqual(len(fwd_values), 2)

        loss = out[0].sum()
        loss.backward()

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

    def test_new_compile_print_with_actual_printing(self):
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
        compiled_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)


        with patch("builtins.print") as mock_print:
            loss = compiled_fn(x, y)
            loss.backward()

            call_args = [str(call) for call in mock_print.call_args_list]
            forward_calls = [c for c in call_args if "Forward" in c]
            backward_calls = [c for c in call_args if "Backward" in c]

            self.assertEqual(len(forward_calls), 2)
            self.assertEqual(len(backward_calls), 2)

    def test_new_compile_print_graph_contains_hop(self):
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

        graph_code = backend.graphs[0].code
        self.assertIn("compile_print_fwd", graph_code)

    def test_new_compile_print_make_fx(self):
        from torch.fx.experimental.proxy_tensor import make_fx

        fwd_values = []
        bwd_values = []

        def fwd_f(t):
            fwd_values.append(t.clone())

        def bwd_f(t):
            bwd_values.append(t.clone())

        cp = make_compile_print(fwd_f, bwd_f)

        def fn(x, y):
            out = cp(x, y)
            return out[0].sum() + out[1].sum()

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)

        # Trace with make_fx
        gm = make_fx(fn, tracing_mode="symbolic")(x, y)

        # Check that compile_print_fwd appears in the graph
        graph_code = gm.code
        self.assertIn("compile_print_fwd", graph_code)

        # Run the traced graph and verify callbacks are called
        fwd_values.clear()
        bwd_values.clear()

        x2 = torch.randn(3, 3, requires_grad=True)
        y2 = torch.randn(3, 3, requires_grad=True)
        result = gm(x2, y2)

        # Forward callbacks should be called
        self.assertEqual(len(fwd_values), 2)

        # Verify result is correct
        expected = x2.sum() + y2.sum()
        self.assertEqual(result, expected)

    def test_new_compile_print_make_fx_with_actual_printing(self):
        from unittest.mock import patch

        from torch.fx.experimental.proxy_tensor import make_fx

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

        # Trace with make_fx
        gm = make_fx(fn, tracing_mode="symbolic")(x, y)

        # Run the traced graph with mock print
        x2 = torch.randn(3, 3, requires_grad=True)
        y2 = torch.randn(3, 3, requires_grad=True)

        with patch("builtins.print") as mock_print:
            result = gm(x2, y2)

            call_args = [str(call) for call in mock_print.call_args_list]
            forward_calls = [c for c in call_args if "Forward" in c]

            self.assertEqual(len(forward_calls), 2)

    def test_new_compile_print_aot_function_with_actual_printing(self):
        from unittest.mock import patch

        from functorch.compile import make_boxed_func
        from torch._functorch.aot_autograd import aot_function

        def fwd_f(t):
            print(f"Forward: shape={t.shape}, mean={t.mean().item():.4f}")

        def bwd_f(t):
            print(f"Backward: shape={t.shape}, mean={t.mean().item():.4f}")

        cp = make_compile_print(fwd_f, bwd_f)

        def fn(x, y):
            out = cp(x, y)
            return out[0].sum() + out[1].sum()

        # No-op compilers that return boxed functions
        def nop_compiler(gm, _):
            return make_boxed_func(gm)

        aot_fn = aot_function(fn, fw_compiler=nop_compiler, bw_compiler=nop_compiler)

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)

        with patch("builtins.print") as mock_print:
            loss = aot_fn(x, y)
            loss.backward()

            call_args = [str(call) for call in mock_print.call_args_list]
            forward_calls = [c for c in call_args if "Forward" in c]
            backward_calls = [c for c in call_args if "Backward" in c]

            self.assertEqual(len(forward_calls), 2)
            self.assertEqual(len(backward_calls), 2)

    def test_new_compile_print_side_effect_only(self):
        """
        Test that compile_print is not DCE'd even when output is not used.

        The compile_print HOP is registered as effectful, so it won't be
        eliminated by dead code elimination even when its output is not used
        in the computation.
        """
        from unittest.mock import patch

        def fwd_f(t):
            print(f"Forward: shape={t.shape}, mean={t.mean().item():.4f}")

        def bwd_f(t):
            print(f"Backward: shape={t.shape}, mean={t.mean().item():.4f}")

        cp = make_compile_print(fwd_f, bwd_f)

        def fn(x, y):
            # Call compile_print but don't use its output
            cp(x, y)
            # Use the original tensors for the computation
            return x.sum() + y.sum()

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)

        with patch("builtins.print") as mock_print:
            compiled_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
            loss = compiled_fn(x, y)
            loss.backward()

            call_args = [str(call) for call in mock_print.call_args_list]
            forward_calls = [c for c in call_args if "Forward" in c]
            backward_calls = [c for c in call_args if "Backward" in c]

            # Forward is called because compile_print is registered as effectful
            # and won't be DCE'd
            self.assertEqual(len(forward_calls), 2)
            # Backward is called because compile_print registers hooks on input
            # tensors that emit compile_print_bwd when they receive gradients
            self.assertEqual(len(backward_calls), 2)

    def test_new_compile_print_side_effect_only_aot_function(self):
        """
        Test that compile_print is not DCE'd even when output is not used (aot_function).

        Same as test_new_compile_print_side_effect_only but using aot_function.
        """
        from unittest.mock import patch

        from functorch.compile import make_boxed_func
        from torch._functorch.aot_autograd import aot_function

        def fwd_f(t):
            print(f"Forward: shape={t.shape}, mean={t.mean().item():.4f}")

        def bwd_f(t):
            print(f"Backward: shape={t.shape}, mean={t.mean().item():.4f}")

        cp = make_compile_print(fwd_f, bwd_f)

        def fn(x, y):
            # Call compile_print but don't use its output
            cp(x, y)
            # Use the original tensors for the computation
            return x.sum() + y.sum()

        def nop_compiler(gm, _):
            return make_boxed_func(gm)

        aot_fn = aot_function(fn, fw_compiler=nop_compiler, bw_compiler=nop_compiler)

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)

        with patch("builtins.print") as mock_print:
            loss = aot_fn(x, y)
            loss.backward()

            call_args = [str(call) for call in mock_print.call_args_list]
            forward_calls = [c for c in call_args if "Forward" in c]
            backward_calls = [c for c in call_args if "Backward" in c]

            # Forward is called because compile_print is registered as effectful
            # and won't be DCE'd
            self.assertEqual(len(forward_calls), 2)
            # Backward is called because compile_print registers hooks on input
            # tensors that emit compile_print_bwd when they receive gradients
            self.assertEqual(len(backward_calls), 2)


if __name__ == "__main__":
    run_tests()
