import torch

from torch._export.db.case import export_case


@export_case(
    example_inputs=(torch.ones(3, 2), torch.tensor(4)),
    tags={"python.data-structure"},
)
class Dictionary(torch.nn.Module):
    """
    Dictionary structures are inlined and flattened along tracing.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        elements = {}
        elements["x2"] = x * x
        y = y * elements["x2"]
        return {"y": y}
