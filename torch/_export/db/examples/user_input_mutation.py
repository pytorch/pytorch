import torch

from torch._export.db.case import export_case, SupportLevel


@export_case(
    example_inputs=(torch.ones(3, 2),),
    tags={"torch.mutation"},
    support_level=SupportLevel.NOT_SUPPORTED_YET,
)
class UserInputMutation(torch.nn.Module):
    """
    Can't directly mutate user input in forward
    """

    def forward(self, x):
        x.mul_(2)
        return x.cos()
