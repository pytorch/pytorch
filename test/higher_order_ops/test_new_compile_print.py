# Owner(s): ["module: higher order operators"]

import re

import torch
from torch._higher_order_ops.new_compile_print import (
    make_compile_print,
    new_compile_print,
)
from torch.testing._internal.common_utils import run_tests, TestCase


def normalize_opaque_refs(graph_str: str) -> str:
    """Normalize opaque object references in graph output to make tests deterministic."""
    # Normalize opaque object variable names (e.g., _opaque_obj0, _opaque_obj1_1)
    graph_str = re.sub(r"_opaque_obj\d+(_\d+)?", "_opaque_obj", graph_str)
    # Normalize closure variable names
    graph_str = re.sub(
        r"[gGlL]_cp_closure_\d+_cell_contents",
        "callback_var",
        graph_str,
    )
    # Normalize self._opaque_obj references
    graph_str = re.sub(r"self\._opaque_obj\d+", "self._opaque_obj", graph_str)
    return graph_str


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

        result = new_compile_print(fwd_f, bwd_f, x, y)
        self.assertIsNone(result)

        self.assertEqual(len(fwd_values), 2)
        self.assertEqual(fwd_values[0], x)
        self.assertEqual(fwd_values[1], y)

        loss = x.sum() + y.sum()
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
            cp(x, y)
            return x.sum() + y.sum()

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

        result = new_compile_print(fwd_f, bwd_f, x, y)
        self.assertIsNone(result)
        self.assertEqual(len(fwd_values), 2)

        loss = x.sum()
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

        result = cp(x, y)
        self.assertIsNone(result)

        self.assertEqual(len(fwd_values), 2)

        loss = x.sum() + y.sum()
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
            cp(x, y)
            return x.sum() + y.sum()

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
            cp(x)
            return x.sum()

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
            cp(x, y)
            return x.sum() + y.sum()

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
            cp(x, y)
            return x.sum() + y.sum()

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)

        # Trace with make_fx
        gm = make_fx(fn, tracing_mode="symbolic")(x, y)

        # Run the traced graph with mock print
        x2 = torch.randn(3, 3, requires_grad=True)
        y2 = torch.randn(3, 3, requires_grad=True)

        with patch("builtins.print") as mock_print:
            gm(x2, y2)

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
            cp(x, y)
            return x.sum() + y.sum()

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
        Test that compile_print is not DCE'd even when it's a pure side-effect.

        The compile_print HOP is registered as effectful, so it won't be
        eliminated by dead code elimination.
        """
        from unittest.mock import patch

        def fwd_f(t):
            print(f"Forward: shape={t.shape}, mean={t.mean().item():.4f}")

        def bwd_f(t):
            print(f"Backward: shape={t.shape}, mean={t.mean().item():.4f}")

        cp = make_compile_print(fwd_f, bwd_f)

        def fn(x, y):
            cp(x, y)
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
        Test that compile_print is not DCE'd even when it's a pure side-effect (aot_function).

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
            cp(x, y)
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

    def test_callback_computation_not_in_gradients_compiled(self):
        """
        Test that computations inside callbacks don't affect gradient computation.

        Users can do any transformation/computation on tensors inside callbacks
        (e.g., t * 2, t.sum(), torch.matmul), and these should NOT be recorded
        in the autograd graph or affect the final gradients.
        """
        fwd_results = []
        bwd_results = []

        def fwd_f(t):
            # Arbitrary computation that should NOT affect gradients
            result = t * 2 + t.sum()
            squared = t @ t.T
            fwd_results.append((result.clone(), squared.clone()))

        def bwd_f(t):
            # Arbitrary computation on gradients
            result = t * 3 - t.mean()
            norm = torch.linalg.norm(t)
            bwd_results.append((result.clone(), norm.item()))

        cp = make_compile_print(fwd_f, bwd_f)

        def fn(x, y):
            cp(x, y)
            # Simple computation: gradients should be all ones
            return x.sum() + y.sum()

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)

        compiled_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        loss = compiled_fn(x, y)
        loss.backward()

        # Callbacks were called
        self.assertEqual(len(fwd_results), 2)
        self.assertEqual(len(bwd_results), 2)

        # Gradients should be all ones (d/dx sum(x) = 1, d/dy sum(y) = 1)
        # NOT affected by the computations inside callbacks
        self.assertEqual(x.grad, torch.ones_like(x))
        self.assertEqual(y.grad, torch.ones_like(y))

    def test_callback_computation_not_in_gradients_aot_function(self):
        """
        Test that computations inside callbacks don't affect gradient computation (aot_function).

        Same as test_callback_computation_not_in_gradients_compiled but using aot_function.
        """
        from functorch.compile import make_boxed_func
        from torch._functorch.aot_autograd import aot_function

        fwd_results = []
        bwd_results = []

        def fwd_f(t):
            # Arbitrary computation that should NOT affect gradients
            result = t * 2 + t.sum()
            squared = t @ t.T
            fwd_results.append((result.clone(), squared.clone()))

        def bwd_f(t):
            # Arbitrary computation on gradients
            result = t * 3 - t.mean()
            norm = torch.linalg.norm(t)
            bwd_results.append((result.clone(), norm.item()))

        cp = make_compile_print(fwd_f, bwd_f)

        def fn(x, y):
            cp(x, y)
            return x.sum() + y.sum()

        def nop_compiler(gm, _):
            return make_boxed_func(gm)

        aot_fn = aot_function(fn, fw_compiler=nop_compiler, bw_compiler=nop_compiler)

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)

        loss = aot_fn(x, y)
        loss.backward()

        # Callbacks were called
        self.assertEqual(len(fwd_results), 2)
        self.assertEqual(len(bwd_results), 2)

        # Gradients should be all ones - NOT affected by callback computations
        self.assertEqual(x.grad, torch.ones_like(x))
        self.assertEqual(y.grad, torch.ones_like(y))

    def test_compile_print_graph_structure_dynamo(self):
        """Test that the dynamo graph contains compile_print_fwd."""
        from torch._dynamo.testing import EagerAndRecordGraphs, normalize_gm

        def fwd_f(t):
            pass

        def bwd_f(t):
            pass

        cp = make_compile_print(fwd_f, bwd_f)

        def fn(x):
            cp(x)
            return x.sum()

        x = torch.randn(3, 3, requires_grad=True)

        backend = EagerAndRecordGraphs()
        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        loss = compiled_fn(x)
        loss.backward()

        self.assertExpectedInline(
            normalize_opaque_refs(
                normalize_gm(backend.graphs[0].print_readable(print_output=False))
            ),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", callback_var : torch._higher_order_ops.new_compile_print.CallbackWrapper, callback_var : torch._higher_order_ops.new_compile_print.CallbackWrapper):
        l_x_ = L_x_
        callback_var = callback_var
        callback_var = callback_var

        compile_print_fwd = torch.ops.higher_order.compile_print_fwd(callback_var, callback_var, l_x_);  callback_var = callback_var = compile_print_fwd = None
        sum_1: "f32[]" = l_x_.sum();  l_x_ = None
        return (sum_1,)""",  # noqa: B950
            ignore_empty_lines=True,
        )

    def test_compile_print_graph_structure_aot_function(self):
        """Test that aot_function graphs contain compile_print_fwd and compile_print_bwd."""
        from functorch.compile import make_boxed_func
        from torch._dynamo.testing import normalize_gm
        from torch._functorch.aot_autograd import aot_function

        def fwd_f(t):
            pass

        def bwd_f(t):
            pass

        cp = make_compile_print(fwd_f, bwd_f)

        def fn(x):
            cp(x)
            return x.sum()

        fw_graph = None
        bw_graph = None

        def fw_compiler(gm, example_inputs):
            nonlocal fw_graph
            fw_graph = gm
            return make_boxed_func(gm)

        def bw_compiler(gm, example_inputs):
            nonlocal bw_graph
            bw_graph = gm
            return make_boxed_func(gm)

        x = torch.randn(3, 3, requires_grad=True)

        aot_fn = aot_function(fn, fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        loss = aot_fn(x)
        loss.backward()

        self.assertExpectedInline(
            normalize_opaque_refs(
                normalize_gm(fw_graph.print_readable(print_output=False))
            ),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[3, 3]"):
        _opaque_obj = self._opaque_obj
        _opaque_obj = self._opaque_obj
        compile_print_fwd = torch.ops.higher_order.compile_print_fwd(_opaque_obj, _opaque_obj, primals_1);  _opaque_obj = _opaque_obj = compile_print_fwd = None
        sum_1: "f32[]" = torch.ops.aten.sum.default(primals_1);  primals_1 = None
        return (sum_1,)""",  # noqa: B950
            ignore_empty_lines=True,
        )

        self.assertExpectedInline(
            normalize_opaque_refs(
                normalize_gm(bw_graph.print_readable(print_output=False))
            ),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[]"):
        expand: "f32[3, 3]" = torch.ops.aten.expand.default(tangents_1, [3, 3]);  tangents_1 = None
        _opaque_obj = self._opaque_obj
        compile_print_bwd = torch.ops.higher_order.compile_print_bwd(_opaque_obj, expand);  _opaque_obj = compile_print_bwd = None
        return (expand,)""",  # noqa: B950
            ignore_empty_lines=True,
        )


if __name__ == "__main__":
    run_tests()
