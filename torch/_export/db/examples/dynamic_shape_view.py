# mypy: allow-untyped-defs
import torch

from torch._export.db.case import export_case


@export_case(
    example_inputs=(torch.randn(10, 10),),
    tags={"torch.dynamic-shape"},
)
class DynamicShapeView(torch.nn.Module):
    """
    Dynamic shapes should be propagated to view arguments instead of being
    baked into the exported graph.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        new_x_shape = x.size()[:-1] + (2, 5)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1)
