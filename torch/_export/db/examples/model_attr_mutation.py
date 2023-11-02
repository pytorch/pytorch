import torch

from torch._export.db.case import export_case, SupportLevel


@export_case(
    example_inputs=(torch.ones(3, 2),),
    tags={"python.object-model"},
    support_level=SupportLevel.NOT_SUPPORTED_YET,
)
class ModelAttrMutation(torch.nn.Module):
    """
    Attribute mutation is not supported.
    """

    def __init__(self):
        super().__init__()
        self.attr_list = [torch.ones(3, 2), torch.ones(3, 2)]

    def recreate_list(self):
        return [torch.zeros(3, 2), torch.zeros(3, 2)]

    def forward(self, x):
        self.attr_list = self.recreate_list()
        return x.sum() + self.attr_list[0].sum()
