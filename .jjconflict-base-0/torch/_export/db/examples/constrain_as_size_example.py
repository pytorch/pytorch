# mypy: allow-untyped-defs
import torch


class ConstrainAsSizeExample(torch.nn.Module):
    """
    If the value is not known at tracing time, you can provide hint so that we
    can trace further. Please look at torch._check APIs.
    """

    def forward(self, x):
        a = x.item()
        torch._check(a >= 0)
        torch._check(a <= 5)
        return torch.zeros((a, 5))


example_args = (torch.tensor(4),)
tags = {
    "torch.dynamic-value",
    "torch.escape-hatch",
}
model = ConstrainAsSizeExample()
