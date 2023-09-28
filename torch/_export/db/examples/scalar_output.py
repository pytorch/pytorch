import torch

from torch._export.db.case import export_case
from torch._export import dynamic_dim

x = torch.ones(3, 2)
dynamic_constraint = dynamic_dim(x, 1)

@export_case(
    example_inputs=(x,),
    tags={"torch.dynamic-shape"},
    constraints=[dynamic_constraint]
)
def scalar_output(x):
    """
    Returning scalar values from the graph is supported, in addition to Tensor
    outputs. Symbolic shapes are captured and rank is specialized.
    """
    return x.shape[1] + 1
