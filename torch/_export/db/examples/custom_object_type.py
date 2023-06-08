import torch

from torch._export.db.case import export_case, SupportLevel


class A:
    @classmethod
    def func(cls, x):
        return 1 + x


@export_case(
    example_inputs=(torch.ones(3, 4),),
    tags={"python.standard-library"},
    support_level=SupportLevel.NOT_SUPPORTED_YET,
)
def custom_object_type(x):
    """
    type() calls on custom objects are not allowed due to its overly dynamic nature.
    """
    a = A()
    return type(a).func(x)
