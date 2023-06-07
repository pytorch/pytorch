import torch

from torch._export.db.case import export_case, SupportLevel


@export_case(
    example_inputs=(torch.ones(3, 2),),
    tags={"torch.dynamic-shape", "python.builtin"},
    support_level=SupportLevel.NOT_SUPPORTED_YET,
)
def dynamic_shape_round(x):
    """
    Calling round on dynamic shapes is not supported.
    """
    return x[: round(x.shape[0] / 2)]
