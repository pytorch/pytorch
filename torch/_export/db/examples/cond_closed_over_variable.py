# mypy: allow-untyped-defs
import torch

from functorch.experimental.control_flow import cond

class CondClosedOverVariable(torch.nn.Module):
    """
    torch.cond() supports branches closed over arbitrary variables.
    """

    def forward(self, pred, x):
        def true_fn(val):
            return x * 2

        def false_fn(val):
            return x - 2

        return cond(pred, true_fn, false_fn, [x + 1])

example_args = (torch.tensor(True), torch.randn(3, 2))
tags = {"torch.cond", "python.closure"}
model = CondClosedOverVariable()
