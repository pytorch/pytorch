# mypy: allow-untyped-defs
import torch

class DynamicShapeView(torch.nn.Module):
    """
    Dynamic shapes should be propagated to view arguments instead of being
    baked into the exported graph.
    """

    def forward(self, x):
        new_x_shape = x.size()[:-1] + (2, 5)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1)

example_args = (torch.randn(10, 10),)
tags = {"torch.dynamic-shape"}
model = DynamicShapeView()
