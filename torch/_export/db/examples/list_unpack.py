from typing import List

import torch

from torch._export.db.case import export_case


@export_case(
    example_inputs=([torch.ones(3, 2), torch.tensor(4), torch.tensor(5)],),
    tags={"python.control-flow", "python.data-structure"},
)
def list_unpack(args: List[torch.Tensor]):
    """
    Lists are treated as static construct, therefore unpacking should be
    erased after tracing.
    """
    x, *y = args
    return x + y[0]
