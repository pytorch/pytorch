# mypy: allow-untyped-defs
import torch


class TensorSetattr(torch.nn.Module):
    """
    setattr() call onto tensors is not supported.
    """
    def forward(self, x, attr):
        setattr(x, attr, torch.randn(3, 2))
        return x + 4

example_inputs = (torch.randn(3, 2), "attr")
tags = {"python.builtin"}
model = TensorSetattr()
