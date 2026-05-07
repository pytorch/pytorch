# mypy: allow-untyped-defs
import torch
from torch._export.db.case import SupportLevel


class TorchSymMin(torch.nn.Module):
    """
    torch.sym_min operator is supported in export.
    """

    def forward(self, x):
        return x.sum() + torch.sym_min(x.size(0), 100)


example_args = (torch.randn(3, 2),)
tags = {"torch.operator"}
support_level = SupportLevel.SUPPORTED
model = TorchSymMin()
