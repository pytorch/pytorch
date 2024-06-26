# mypy: allow-untyped-defs
import torch

from torch._export.db.case import export_case


@export_case(
    example_inputs=(torch.tensor(4),),
    tags={
        "torch.dynamic-value",
        "torch.escape-hatch",
    },
)
class ConstrainAsSizeExample(torch.nn.Module):
    """
    If the value is not known at tracing time, you can provide hint so that we
    can trace further. Please look at torch._check and torch._check_is_size APIs.
    torch._check_is_size is used for values that NEED to be used for constructing
    tensor.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        a = x.item()
        torch._check_is_size(a)
        torch._check(a <= 5)
        return torch.zeros((a, 5))
