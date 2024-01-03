import torch

from torch._export.db.case import export_case, SupportLevel
from torch.export import Dim

x = torch.ones(3, 2)
dim0_x = Dim("dim0_x")

@export_case(
    example_inputs=(x,),
    tags={"torch.dynamic-shape", "python.builtin"},
    support_level=SupportLevel.NOT_SUPPORTED_YET,
    dynamic_shapes={"x": {0: dim0_x}},
)
def dynamic_shape_round(x):
    """
    Calling round on dynamic shapes is not supported.
    """
    return x[: round(x.shape[0] / 2)]
