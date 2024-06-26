# mypy: allow-untyped-defs
import torch

from torch._export.db.case import export_case
from functorch.experimental.control_flow import cond


@export_case(
    example_inputs=(torch.randn(6, 4, 3),),
    tags={
        "torch.cond",
        "torch.dynamic-shape",
    },
)
class CondPredicate(torch.nn.Module):
    """
    The conditional statement (aka predicate) passed to cond() must be one of the following:
      - torch.Tensor with a single element
      - boolean expression

    NOTE: If the `pred` is test on a dim with batch size < 2, it will be specialized.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pred = x.dim() > 2 and x.shape[2] > 10

        return cond(pred, lambda x: x.cos(), lambda y: y.sin(), [x])
