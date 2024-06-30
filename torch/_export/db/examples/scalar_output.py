# mypy: allow-untyped-defs
import torch

from torch.export import Dim

x = torch.randn(3, 2)
dim1_x = Dim("dim1_x")

class ScalarOutput(torch.nn.Module):
    """
    Returning scalar values from the graph is supported, in addition to Tensor
    outputs. Symbolic shapes are captured and rank is specialized.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.shape[1] + 1

example_inputs = (x,)
tags = {"torch.dynamic-shape"}
dynamic_shapes = {"x": {1: dim1_x}}
model = ScalarOutput()
