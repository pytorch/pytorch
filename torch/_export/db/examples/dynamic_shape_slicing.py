import torch

from torch._export.db.case import export_case


@export_case(
    example_inputs=(torch.randn(3, 2),),
    tags={"torch.dynamic-shape"},
)
class DynamicShapeSlicing(torch.nn.Module):
    """
    Slices with dynamic shape arguments should be captured into the graph
    rather than being baked in.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[: x.shape[0] - 2, x.shape[1] - 1 :: 2]
