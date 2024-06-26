# mypy: allow-untyped-defs
import torch

from functorch.experimental.control_flow import map

class DynamicShapeMap(torch.nn.Module):
    """
    functorch map() maps a function over the first tensor dimension.
    """

    def forward(self, xs, y):
        def body(x, y):
            return x + y

        return map(body, xs, y)

example_inputs = (torch.randn(3, 2), torch.randn(2))
tags = {"torch.dynamic-shape", "torch.map"}
model = DynamicShapeMap()
