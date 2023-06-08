import torch

from torch._export.db.case import export_case, SupportLevel


class A:
    @classmethod
    def func(cls, x):
        return 1 + x


@export_case(
    example_inputs=(torch.ones(3, 4),),
    tags={"python.builtin"},
    support_level=SupportLevel.NOT_SUPPORTED_YET,
)
def type_reflection_method(x):
    """
    type() calls on custom objects followed by method calls are not allowed
    due to its overly dynamic nature.
    """
    a = A()
    return type(a).func(x)
