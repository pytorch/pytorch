# mypy: allow-untyped-defs
import torch

from torch.utils import _pytree as pytree

class PytreeFlatten(torch.nn.Module):
    """
    Pytree from PyTorch can be captured by TorchDynamo.
    """

    def forward(self, x):
        y, _spec = pytree.tree_flatten(x)
        return y[0] + 1

example_args = ({1: torch.randn(3, 2), 2: torch.randn(3, 2)},),
model = PytreeFlatten()
