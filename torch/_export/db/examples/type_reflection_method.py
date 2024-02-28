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
class TypeReflectionMethod(torch.nn.Module):
    """
    type() calls on custom objects followed by attribute accesses are not allowed
    due to its overly dynamic nature.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        a = A()
        return type(a).func(x)


@export_rewrite_case(parent=TypeReflectionMethod)
class TypeReflectionMethodRewrite(torch.nn.Module):
    """
    Custom object class methods will be inlined.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return A.func(x)
