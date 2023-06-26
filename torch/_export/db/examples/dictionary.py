import torch

from torch._export.db.case import export_case


@export_case(
    example_inputs=(torch.ones(3, 2), torch.tensor(4)),
    tags={"python.data-structure"},
)
def dictionary(x, y):
    """
    Dictionary structures are inlined and flattened along tracing.
    """
    elements = {}
    elements["x2"] = x * x
    y = y * elements["x2"]
    return {"y": y}
