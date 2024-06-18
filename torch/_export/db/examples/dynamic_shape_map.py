import torch

from torch._export.db.case import export_case
from functorch.experimental.control_flow import map


@export_case(
    example_inputs=(torch.randn(3, 2), torch.randn(2)),
    tags={"torch.dynamic-shape", "torch.map"},
)
class DynamicShapeMap(torch.nn.Module):
    """
    functorch map() maps a function over the first tensor dimension.
    """

    def __init__(self):
        super().__init__()

    def forward(self, xs, y):
        def body(x, y):
            return x + y

        return map(body, xs, y)
