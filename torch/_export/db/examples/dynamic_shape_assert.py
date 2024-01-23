import torch

from torch._export.db.case import export_case


@export_case(
    example_inputs=(torch.ones(3, 2),),
    tags={"python.assert"},
)
class DynamicShapeAssert(torch.nn.Module):
    """
    A basic usage of python assertion.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # assertion with error message
        assert x.shape[0] > 2, f"{x.shape[0]} is greater than 2"
        # assertion without error message
        assert x.shape[0] > 1
        return x
