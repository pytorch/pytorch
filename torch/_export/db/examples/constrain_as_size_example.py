# mypy: allow-untyped-defs
import torch


class ConstrainAsSizeExample(torch.nn.Module):
    """
    If the value is not known at tracing time, you can provide hint so that we
    can trace further. Please look at torch._check and torch._check_is_size APIs.
    torch._check_is_size is used for values that NEED to be used for constructing
    tensor.
    """

    def forward(self, x):
        a = x.item()
        torch._check_is_size(a)
        torch._check(a <= 5)
        return torch.zeros((a, 5))


example_args = (torch.tensor(4),)
tags = {
    "torch.dynamic-value",
    "torch.escape-hatch",
}
model = ConstrainAsSizeExample()
