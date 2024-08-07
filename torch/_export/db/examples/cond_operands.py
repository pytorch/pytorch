# mypy: allow-untyped-defs
import torch

from torch.export import Dim
from functorch.experimental.control_flow import cond

x = torch.randn(3, 2)
y = torch.randn(2)
dim0_x = Dim("dim0_x")

class CondOperands(torch.nn.Module):
    """
    The operands passed to cond() must be:
    - a list of tensors
    - match arguments of `true_fn` and `false_fn`

    NOTE: If the `pred` is test on a dim with batch size < 2, it will be specialized.
    """

    def forward(self, x, y):
        def true_fn(x, y):
            return x + y

        def false_fn(x, y):
            return x - y

        return cond(x.shape[0] > 2, true_fn, false_fn, [x, y])

example_args = (x, y)
tags = {
    "torch.cond",
    "torch.dynamic-shape",
}
extra_inputs = (torch.randn(2, 2), torch.randn(2))
dynamic_shapes = {"x": {0: dim0_x}, "y": None}
model = CondOperands()
