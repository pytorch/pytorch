import torch
from torch.autograd import Function
from torch.testing._internal.common_utils import run_tests, TestCase


class TestAutogradBackwardInputMutation(TestCase):
    def test_custom_function_backward_input_mutation_raises(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, grad_output):
                grad_output.add_(1)
                return grad_output

        x = torch.rand(2, 4, requires_grad=True)
        grad_output = torch.ones_like(x)

        with self.assertRaisesRegex(
            RuntimeError, "modified a gradient input in-place during backward"
        ):
            torch.autograd.grad(MyFunction.apply(x), x, grad_outputs=grad_output)


if __name__ == "__main__":
    run_tests()
