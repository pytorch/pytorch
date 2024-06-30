# mypy: allow-untyped-defs
import torch

class A:
    @classmethod
    def func(cls, x):
        return 1 + x

class TypeReflectionMethod(torch.nn.Module):
    """
    type() calls on custom objects followed by attribute accesses are not allowed
    due to its overly dynamic nature.
    """

    def forward(self, x):
        a = A()
        return type(a).func(x)


example_inputs = (torch.randn(3, 4),)
tags = {"python.builtin"}
model = TypeReflectionMethod()
