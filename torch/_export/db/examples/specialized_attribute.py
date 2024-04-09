from enum import Enum

import torch

from torch._export.db.case import export_case


class Animal(Enum):
    COW = "moo"


@export_case(
    example_inputs=(torch.randn(3, 2),),
)
class SpecializedAttribute(torch.nn.Module):
    """
    Model attributes are specialized.
    """

    def __init__(self):
        super().__init__()
        self.a = "moo"
        self.b = 4

    def forward(self, x):
        if self.a == Animal.COW.value:
            return x * x + self.b
        else:
            raise ValueError("bad")
