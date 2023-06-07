import torch

from torch._export.db.case import export_case, ExportArgs


@export_case(
    example_inputs=ExportArgs(
        torch.randn(4),
        (torch.randn(4), torch.randn(4)),
        *[torch.randn(4), torch.randn(4)],
        mykw0=torch.randn(4),
        **{"input0": torch.randn(4), "input1": torch.randn(4)}
    ),
    tags={"python.data-structure"},
)
def fn_with_kwargs(pos0, tuple0, *myargs, mykw0=None, **mykwargs):
    """
    Keyword arguments as function inputs are flattened.
    """
    out = pos0
    for arg in tuple0:
        out *= arg
    for arg in myargs:
        out *= arg
    out *= mykw0
    out *= mykwargs["input0"] * mykwargs["input1"]
    return out
