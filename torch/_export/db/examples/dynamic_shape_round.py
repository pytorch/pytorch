# mypy: allow-untyped-defs
import torch

from torch._export.db.case import SupportLevel
from torch.export import Dim

class DynamicShapeRound(torch.nn.Module):
    """
    Calling round on dynamic shapes is not supported.
    """

    def forward(self, x):
        return x[: round(x.shape[0] / 2)]

x = torch.randn(3, 2)
dim0_x = Dim("dim0_x")
example_inputs = (x,)
tags = {"torch.dynamic-shape", "python.builtin"}
support_level = SupportLevel.NOT_SUPPORTED_YET
dynamic_shapes = {"x": {0: dim0_x}}
model = DynamicShapeRound()
