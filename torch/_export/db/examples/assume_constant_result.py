# mypy: allow-untyped-defs
import torch
import torch._dynamo as torchdynamo


class AssumeConstantResult(torch.nn.Module):
    """
    Applying `assume_constant_result` decorator to burn make non-tracable code as constant.
    """

    @torchdynamo.assume_constant_result
    def get_item(self, y):
        return y.int().item()

    def forward(self, x, y):
        return x[: self.get_item(y)]

example_inputs = (torch.randn(3, 2), torch.tensor(4))
tags = {"torch.escape-hatch"}
model = AssumeConstantResult()
