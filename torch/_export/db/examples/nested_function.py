# mypy: allow-untyped-defs
import torch

class NestedFunction(torch.nn.Module):
    """
    Nested functions are traced through. Side effects on global captures
    are not supported though.
    """

    def forward(self, a, b):
        x = a + b
        z = a - b

        def closure(y):
            nonlocal x
            x += 1
            return x * y + z

        return closure(x)

example_args = (torch.randn(3, 2), torch.randn(2))
tags = {"python.closure"}
model = NestedFunction()
