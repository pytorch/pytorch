# mypy: allow-untyped-defs
from enum import Enum

import torch

class Animal(Enum):
    COW = "moo"

class SpecializedAttribute(torch.nn.Module):
    """
    Model attributes are specialized.
    """

    def __init__(self) -> None:
        super().__init__()
        self.a = "moo"
        self.b = 4

    def forward(self, x):
        if self.a == Animal.COW.value:
            return x * x + self.b
        else:
            raise ValueError("bad")

example_args = (torch.randn(3, 2),)
model = SpecializedAttribute()
