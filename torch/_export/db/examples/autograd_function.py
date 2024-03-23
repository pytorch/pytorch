import torch

from torch._export.db.case import export_case


class MyAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output + 1


@export_case(
    example_inputs=(torch.randn(3, 2),),
)
class AutogradFunction(torch.nn.Module):
    """
    TorchDynamo does not keep track of backward() on autograd functions. We recommend to
    use `allow_in_graph` to mitigate this problem.
    """

    def forward(self, x):
        return MyAutogradFunction.apply(x)
