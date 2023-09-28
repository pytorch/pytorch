import torch
import torch._dynamo as torchdynamo

from torch._export.db.case import export_case


@export_case(
    example_inputs=(torch.ones(3, 2), torch.tensor(4)),
    tags={"torch.escape-hatch"},
)
class AssumeConstantResult(torch.nn.Module):
    """
    Applying `assume_constant_result` decorator to burn make non-tracable code as constant.
    """

    def __init__(self):
        super().__init__()

    @torchdynamo.assume_constant_result
    def get_item(self, y):
        return y.int().item()

    def forward(self, x, y):
        return x[: self.get_item(y)]
