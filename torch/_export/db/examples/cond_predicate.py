# mypy: allow-untyped-defs
import torch

from functorch.experimental.control_flow import cond

class CondPredicate(torch.nn.Module):
    """
    The conditional statement (aka predicate) passed to cond() must be one of the following:
      - torch.Tensor with a single element
      - boolean expression

    NOTE: If the `pred` is test on a dim with batch size < 2, it will be specialized.
    """

    def forward(self, x):
        pred = x.dim() > 2 and x.shape[2] > 10

        return cond(pred, lambda x: x.cos(), lambda y: y.sin(), [x])

example_inputs = (torch.randn(6, 4, 3),)
tags = {
    "torch.cond",
    "torch.dynamic-shape",
}
model = CondPredicate()
