# mypy: allow-untyped-defs
import torch

class DynamicShapeAssert(torch.nn.Module):
    """
    A basic usage of python assertion.
    """

    def forward(self, x):
        # assertion with error message
        assert x.shape[0] > 2, f"{x.shape[0]} is greater than 2"
        # assertion without error message
        assert x.shape[0] > 1
        return x

example_args = (torch.randn(3, 2),)
tags = {"python.assert"}
model = DynamicShapeAssert()
