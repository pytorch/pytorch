# mypy: allow-untyped-defs
import torch
from torch._export.db.case import SupportLevel


class OptionalInput(torch.nn.Module):
    """
    Tracing through optional input is not supported yet
    """

    def forward(self, x, y=None):
        # Tensor default args are always created on CPU; build on x.device instead.
        if y is None:
            y = torch.ones_like(x)
        return x + y


example_args = (torch.randn(2, 3),)
tags = {"python.object-model"}
support_level = SupportLevel.SUPPORTED
model = OptionalInput()
