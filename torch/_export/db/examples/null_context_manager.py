import contextlib

import torch

from torch._export.db.case import export_case


@export_case(
    example_inputs=(torch.ones(3, 2),),
    tags={"python.context-manager"},
)
def null_context_manager(x):
    """
    Null context manager in Python will be traced out.
    """
    ctx = contextlib.nullcontext()
    with ctx:
        return x.sin() + x.cos()
