import torch

from torch._export.db.case import export_case, SupportLevel, export_rewrite_case


class A:
    @classmethod
    def func(cls, x):
        return 1 + x


@export_case(
    example_inputs=(torch.ones(3, 4),),
    tags={"python.builtin"},
    support_level=SupportLevel.SUPPORTED,
)
def type_reflection_method(x):
    """
    type() calls on custom objects followed by method calls are not allowed
    due to its overly dynamic nature.
    """
    a = A()
    return type(a).func(x)


@export_rewrite_case(parent=type_reflection_method)
def type_reflection_method_rewrite(x):
    """
    Custom object class methods will be inlined.
    """
    return A.func(x)
