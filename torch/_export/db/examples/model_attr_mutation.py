# mypy: allow-untyped-defs
import torch
from torch._export.db.case import SupportLevel


class ModelAttrMutation(torch.nn.Module):
    """
    Attribute mutation is not supported.
    """

    def __init__(self) -> None:
        super().__init__()
        self.attr_list = [torch.randn(3, 2), torch.randn(3, 2)]

    def recreate_list(self):
        return [torch.zeros(3, 2), torch.zeros(3, 2)]

    def forward(self, x):
        self.attr_list = self.recreate_list()
        return x.sum() + self.attr_list[0].sum()


example_args = (torch.randn(3, 2),)
tags = {"python.object-model"}
support_level = SupportLevel.NOT_SUPPORTED_YET
model = ModelAttrMutation()
