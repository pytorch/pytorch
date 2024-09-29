# mypy: allow-untyped-defs
import torch

class DynamicShapeSlicing(torch.nn.Module):
    """
    Slices with dynamic shape arguments should be captured into the graph
    rather than being baked in.
    """

    def forward(self, x):
        return x[: x.shape[0] - 2, x.shape[1] - 1 :: 2]

example_args = (torch.randn(3, 2),)
tags = {"torch.dynamic-shape"}
model = DynamicShapeSlicing()
