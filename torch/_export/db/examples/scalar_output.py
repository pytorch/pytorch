import torch

from torch._export.db.case import export_case


@export_case(
    example_inputs=(torch.ones(3, 2),),
    tags={"torch.dynamic-shape"},
)
def scalar_output(x):
    """
    Returning scalar values from the graph is supported, in addition to Tensor
    outputs. Symbolic shapes are captured and rank is specialized.
    """
    return x.shape[1] + 1
