from typing import List

import torch

from torch._export.db.case import export_case


@export_case(
    example_inputs=([torch.randn(3, 2), torch.tensor(4), torch.tensor(5)],),
    tags={"python.control-flow", "python.data-structure"},
)
class ListUnpack(torch.nn.Module):
    """
    Lists are treated as static construct, therefore unpacking should be
    erased after tracing.
    """

    def __init__(self):
        super().__init__()

    def forward(self, args: List[torch.Tensor]):
        """
        Lists are treated as static construct, therefore unpacking should be
        erased after tracing.
        """
        x, *y = args
        return x + y[0]
