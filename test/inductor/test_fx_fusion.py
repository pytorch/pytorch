# Owner(s): ["module: inductor"]
from collections.abc import Callable
from typing import Any

import torch
from torch._inductor.fx_passes.pre_grad import (
    linear_permute_fusion,
    linear_transpose,
    permute_linear_fusion,
    permute_matmul_fusion,
    sink_cat_after_pointwise,
    transpose_linear,
    transpose_matmul,
)
from torch._inductor.test_case import run_tests, TestCase
from torch.fx.passes.shape_prop import ShapeProp


PassFunc = Callable[[torch.fx.GraphModule, Any], torch.fx.GraphModule]


def chain_passes(*passes: PassFunc) -> PassFunc:
    def parent_pass(module: torch.fx.GraphModule, input: Any) -> torch.fx.GraphModule:
        for pass_ in passes:
            if isinstance(module, torch.fx.GraphModule):
                ShapeProp(module).propagate(*input)
            module = pass_(module)
        return module

    return parent_pass


def count_call(module: torch.fx.GraphModule, op: str, target_op: Any) -> int:
    return sum(
        1 if (n.op == op and n.target == target_op) else 0 for n in module.graph.nodes
    )


def count_call_function(module: torch.fx.GraphModule, target_op: Any) -> int:
    return count_call(module, "call_function", target_op)


def count_call_method(module: torch.fx.GraphModule, target_op: Any) -> int:
    return count_call(module, "call_method", target_op)


class TestFxFusion(TestCase):
    def test_sink_cat_after_pointwise(self):
        def test_kwarg(x, y):
            return torch.cat([x, y], dim=-1).view(-1).view(128).tanh()

        def test_arg(x, y):
            return torch.cat([x, y], -1).view(-1).view(128).tanh()

        def test_arg2(x, y):
            return torch.cat([x, y]).view(-1).view(128).tanh()

        def test_kwarg2(x, y):
            return torch.cat(tensors=[x, y], dim=0).tanh()

        def test_kwarg3(x, y):
            return torch.cat(tensors=[x, y], dim=0).view(128).tanh()

        trace_func = chain_passes(torch.fx.symbolic_trace, sink_cat_after_pointwise)
        inputs = [
            torch.randn(8, 8),
            torch.randn(8, 8),
        ]
        for f in [test_kwarg, test_arg, test_arg2, test_kwarg2, test_kwarg3]:
            traced = trace_func(f, inputs)
            torch.testing.assert_close(f(*inputs), traced(*inputs))
            self.assertEqual(count_call_method(traced, "tanh"), 2)

    def test_linear_permute_fusion(self):
        class TestModule(torch.nn.Module):
            def __init__(self, k: int, n: int, has_bias: bool):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(n, k))
                self.has_bias = has_bias
                if has_bias:
                    self.bias = torch.nn.Parameter(torch.randn(n))

            def forward(self, input: torch.Tensor):
                if self.has_bias:
                    a0 = torch.nn.functional.linear(input, self.weight, self.bias)
                else:
                    a0 = torch.nn.functional.linear(input, self.weight)
                b0 = a0.permute(0, 2, 1)
                return b0

        m, k, n = 16, 8, 4
        trace_func = chain_passes(torch.fx.symbolic_trace, linear_permute_fusion)
        for has_bias in [True, False]:
            module = TestModule(k, n, has_bias).eval()
            input = torch.randn(6, m, k)
            traced = trace_func(module, [input])
            num_linear = count_call_function(traced, torch.nn.functional.linear)
            num_linear_transpose = count_call_function(traced, linear_transpose)
            self.assertEqual(num_linear, 0)
            self.assertEqual(num_linear_transpose, 1)

            torch.testing.assert_close(module(input), traced(input))

    def test_permute_linear_fusion(self):
        class TestModule(torch.nn.Module):
            def __init__(self, k: int, n: int, has_bias: bool):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(n, k))
                self.has_bias = has_bias
                if has_bias:
                    self.bias = torch.nn.Parameter(torch.randn(n))

            def forward(self, input: torch.Tensor):
                input1 = input.permute(0, 2, 1)
                if self.has_bias:
                    return torch.nn.functional.linear(input1, self.weight, self.bias)
                return torch.nn.functional.linear(input1, self.weight)

        m, k, n = 16, 8, 4

        trace_func = chain_passes(torch.fx.symbolic_trace, permute_linear_fusion)
        for has_bias in [True, False]:
            module = TestModule(k, n, has_bias).eval()
            input = torch.randn(6, k, m)
            traced = trace_func(module, [input])
            num_linear = count_call_function(traced, torch.nn.functional.linear)
            num_transpose_linear = count_call_function(traced, transpose_linear)
            self.assertEqual(num_linear, 0)
            self.assertEqual(num_transpose_linear, 1)

            torch.testing.assert_close(module(input), traced(input))

    def test_permute_bmm_fusion(self):
        class TestModule(torch.nn.Module):
            def __init__(self, batch: int, k: int, n: int):
                super().__init__()
                self.other = torch.randn(batch, k, n)

            def forward(self, input: torch.Tensor):
                input1 = input.permute(0, 2, 1)
                output = torch.bmm(input1, self.other)
                return output

        batch, m, k, n = 6, 16, 8, 4

        trace_func = chain_passes(torch.fx.symbolic_trace, permute_matmul_fusion)
        module = TestModule(batch, k, n).eval()
        input = torch.randn(batch, k, m)
        traced = trace_func(module, [input])
        num_bmm = count_call_function(traced, torch.bmm)
        num_transpose_matmul = count_call_function(traced, transpose_matmul)
        self.assertEqual(num_bmm, 0)
        self.assertEqual(num_transpose_matmul, 1)

        torch.testing.assert_close(module(input), traced(input))

    def test_view_to_reshape_single_pass(self):
        """
        Test that view_to_reshape correctly replaces view ops with reshape ops
        using a single pass over the graph nodes.
        """
        from torch._inductor.fx_passes.post_grad import view_to_reshape

        class TestModule(torch.nn.Module):
            def forward(self, x):
                # Multiple view ops that should be converted to reshape
                y = x.view(-1)
                z = y.view(2, 6)
                return z.sin()

        module = TestModule()
        # Use make_fx to get the graph in ATen form (with view.default)
        from torch.fx.experimental.proxy_tensor import make_fx

        example_input = torch.randn(3, 4)
        gm = make_fx(module)(example_input)

        # Count view ops before transformation
        view_count_before = sum(
            1
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.aten.view.default
        )
        self.assertGreater(view_count_before, 0, "Should have view ops before transformation")

        # Apply the transformation
        view_to_reshape(gm)

        # Count view ops after transformation (should be 0)
        view_count_after = sum(
            1
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.aten.view.default
        )
        self.assertEqual(view_count_after, 0, "All view ops should be replaced with reshape")

        # Count reshape ops after transformation
        reshape_count = sum(
            1
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is torch.ops.aten.reshape.default
        )
        self.assertEqual(reshape_count, view_count_before, "View ops should become reshape ops")

        # Verify the result is correct
        torch.testing.assert_close(module(example_input), gm(example_input))


if __name__ == "__main__":
    run_tests()
