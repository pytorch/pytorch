import torch

from torch._export.db.case import export_case


@export_case(
    example_inputs=(torch.ones(3, 2, 2),),
    tags={"python.control-flow"},
)
class StaticIf(torch.nn.Module):
    """
    `if` statement with static predicate value should be traced through with the
    taken branch.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        if len(x.shape) == 3:
            return x + torch.ones(1, 1, 1)

        return x
