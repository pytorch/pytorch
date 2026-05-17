# mypy: allow-untyped-defs
import torch

class Dictionary(torch.nn.Module):
    """
    Dictionary structures are inlined and flattened along tracing.
    """

    def forward(self, x, y):
        elements = {}
        elements["x2"] = x * x
        y = y * elements["x2"]
        return {"y": y}

example_args = (torch.randn(3, 2), torch.tensor(4))
tags = {"python.data-structure"}
model = Dictionary()
