import torch

from torch._export.db.case import export_case, SupportLevel


@export_case(
    example_inputs=(torch.randn(3, 2),),
    tags={"torch.mutation"},
    support_level=SupportLevel.SUPPORTED,
)
class UserInputMutation(torch.nn.Module):
    """
    Directly mutate user input in forward
    """

    def forward(self, x):
        x.mul_(2)
        return x.cos()
