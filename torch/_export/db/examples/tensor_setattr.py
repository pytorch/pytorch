# mypy: allow-untyped-defs
import torch

from torch._export.db.case import export_case, SupportLevel


@export_case(
    example_inputs=(torch.randn(3, 2), "attr"),
    tags={"python.builtin"},
    support_level=SupportLevel.SUPPORTED,
)
class TensorSetattr(torch.nn.Module):
    """
    setattr() call onto tensors is not supported.
    """
    def forward(self, x, attr):
        setattr(x, attr, torch.randn(3, 2))
        return x + 4
