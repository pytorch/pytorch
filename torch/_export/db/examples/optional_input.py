# mypy: allow-untyped-defs
import torch

from torch._export.db.case import export_case, SupportLevel


@export_case(
    example_inputs=(torch.randn(2, 3),),
    tags={"python.object-model"},
    support_level=SupportLevel.NOT_SUPPORTED_YET,
)
class OptionalInput(torch.nn.Module):
    """
    Tracing through optional input is not supported yet
    """

    def forward(self, x, y=torch.randn(2, 3)):
        if y is not None:
            return x + y
        return x
