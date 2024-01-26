import torch

from torch._export.db.case import export_case


@export_case(
    example_inputs=(torch.ones(3, 2),),
    tags={"torch.dynamic-shape"},
)
class DynamicShapeConstructor(torch.nn.Module):
    """
    Tensor constructors should be captured with dynamic shape inputs rather
    than being baked in with static shape.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.ones(x.shape[0] * 2)
