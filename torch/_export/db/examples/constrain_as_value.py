import torch

from torch._export.db.case import export_case
from torch._export.constraints import constrain_as_value


@export_case(
    example_inputs=(torch.tensor(4), torch.randn(5, 5)),
    tags={
        "torch.dynamic-value",
        "torch.escape-hatch",
    },
)
def constrain_as_value(x, y):
    """
    If the value is not known at tracing time, you can provide hint so that we
    can trace further. Please look at constrain_as_value and constrain_as_size APIs.
    constrain_as_value is used for values that don't need to be used for constructing
    tensor.
    """
    a = x.item()
    constrain_as_value(a, min=0, max=5)

    if a < 6:
        return y.sin()
    return y.cos()
