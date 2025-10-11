# mypy: allow-untyped-defs
import torch


class ConstrainAsValueExample(torch.nn.Module):
    """
    If the value is not known at tracing time, you can provide hint so that we
    can trace further. Please look at torch._check and torch._check_is_size APIs.
    torch._check is used for values that don't need to be used for constructing
    tensor.
    """

    def forward(self, x, y):
        a = x.item()
        torch._check(a >= 0)
        torch._check(a <= 5)

        if a < 6:
            return y.sin()
        return y.cos()


example_args = (torch.tensor(4), torch.randn(5, 5))
tags = {
    "torch.dynamic-value",
    "torch.escape-hatch",
}
model = ConstrainAsValueExample()
