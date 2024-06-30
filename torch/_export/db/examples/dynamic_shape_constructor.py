# mypy: allow-untyped-defs
import torch

class DynamicShapeConstructor(torch.nn.Module):
    """
    Tensor constructors should be captured with dynamic shape inputs rather
    than being baked in with static shape.
    """

    def forward(self, x):
        return torch.zeros(x.shape[0] * 2)

example_inputs = (torch.randn(3, 2),)
tags = {"torch.dynamic-shape"}
model = DynamicShapeConstructor()
