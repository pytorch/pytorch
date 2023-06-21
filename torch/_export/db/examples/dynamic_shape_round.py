import torch

from torch._export.db.case import export_case, SupportLevel
from torch._export import dynamic_dim

x = torch.ones(3, 2)
dynamic_constraint = dynamic_dim(x, 0)

@export_case(
    example_inputs=(x,),
    tags={"torch.dynamic-shape", "python.builtin"},
    support_level=SupportLevel.NOT_SUPPORTED_YET,
    constraints=[dynamic_constraint]
)
def dynamic_shape_round(x):
    """
    Calling round on dynamic shapes is not supported.
    """
    return x[: round(x.shape[0] / 2)]
