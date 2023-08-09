import torch

from torch.compiler import dynamic_dim
from torch._export.db.case import export_case, SupportLevel

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
