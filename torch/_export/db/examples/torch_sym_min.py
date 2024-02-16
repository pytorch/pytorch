import torch

from torch._export.db.case import export_case, SupportLevel


@export_case(
    example_inputs=(torch.ones(3, 2),),
    tags={"torch.operator"},
    support_level=SupportLevel.NOT_SUPPORTED_YET,
)
class TorchSymMin(torch.nn.Module):
    """
    torch.sym_min operator is not supported in export.
    """

    def forward(self, x):
        return x.sum() + torch.sym_min(x.size(0), 100)
