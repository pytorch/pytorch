# Owner(s): ["module: unknown"]

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestCustomFunction(TestCase):
    def test_autograd_function_with_matmul_folding_at_output(self):
        """
        When tensor folding occurs during matmul operation returned tensor is a view.
        This can cause issues when matmul is used inside a custom function
        and such view is then returned as output. Then it cannot be modified inplace
        and causes errors.
        It can be especially problematic when after such function inplace allreduce
        is performed. This test recreates this behaviour.
        Issue is resolved when unsafe_view is returned from matmul instead.
        """

        class CustomFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp1, inp2):
                ctx.save_for_backward(inp2)
                ctx.output_shape = inp1.size()
                return torch.matmul(inp1, inp2)

            @staticmethod
            def backward(ctx, grad_output):
                output_shape = ctx.output_shape
                (inp2,) = ctx.saved_tensors
                return (
                    torch.mm(grad_output.squeeze(), inp2.t()).view(output_shape),
                    None,
                )

        def outer_function(inp1, inp2):
            res = CustomFunction.apply(inp1, inp2)
            res.add_(1.0)
            return res.sum()

        def usual_function(inp1, inp2) -> torch.Tensor:
            res = torch.matmul(inp1, inp2)
            res.add_(1.0)
            return res.sum()

        inp1_custom = torch.randn(4, 1, 2, requires_grad=True)
        inp1_usual = inp1_custom.detach().clone().requires_grad_(True)

        inp2 = torch.randn(2, 4)
        c_custom_func = torch.compile(outer_function)
        c_usual_func = torch.compile(usual_function)

        result_custom = c_custom_func(inp1_custom, inp2)
        result_custom.backward()
        result_usual = c_usual_func(inp1_usual, inp2)
        result_usual.backward()

        torch.allclose(inp1_custom.grad, inp1_usual.grad)


if __name__ == "__main__":
    run_tests()
