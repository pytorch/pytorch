# mypy: allow-untyped-defs
import torch

class DynamicShapeIfGuard(torch.nn.Module):
    """
    `if` statement with backed dynamic shape predicate will be specialized into
    one particular branch and generate a guard. However, export will fail if the
    the dimension is marked as dynamic shape from higher level API.
    """

    def forward(self, x):
        if x.shape[0] == 3:
            return x.cos()

        return x.sin()

example_args = (torch.randn(3, 2, 2),)
tags = {"torch.dynamic-shape", "python.control-flow"}
model = DynamicShapeIfGuard()
