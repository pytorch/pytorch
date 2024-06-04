# mypy: allow-untyped-defs
import contextlib

import torch

from torch._export.db.case import export_case


@export_case(
    example_inputs=(torch.randn(3, 2),),
    tags={"python.context-manager"},
)
class NullContextManager(torch.nn.Module):
    """
    Null context manager in Python will be traced out.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Null context manager in Python will be traced out.
        """
        ctx = contextlib.nullcontext()
        with ctx:
            return x.sin() + x.cos()
