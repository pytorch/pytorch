# mypy: allow-untyped-defs
import torch

from torch._export.db.case import export_case


@export_case(
    example_inputs=(torch.randn(3, 2),),
    tags={"python.control-flow"},
)
class StaticForLoop(torch.nn.Module):
    """
    A for loop with constant number of iterations should be unrolled in the exported graph.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        ret = []
        for i in range(10):  # constant
            ret.append(i + x)
        return ret
