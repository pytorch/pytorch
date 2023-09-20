import torch

from torch._export.db.case import export_case
from torch._export import dynamic_dim
from functorch.experimental.control_flow import cond

x = torch.randn(3, 2)
y = torch.ones(2)
dynamic_constraint = dynamic_dim(x, 0)

@export_case(
    example_inputs=(x, y),
    tags={
        "torch.cond",
        "torch.dynamic-shape",
    },
    extra_inputs=(torch.randn(2, 2), torch.ones(2)),
    constraints=[dynamic_constraint]
)
def cond_operands(x, y):
    """
    The operands passed to cond() must be:
      - a list of tensors
      - match arguments of `true_fn` and `false_fn`

    NOTE: If the `pred` is test on a dim with batch size < 2, it will be specialized.
    """

    def true_fn(x, y):
        return x + y

    def false_fn(x, y):
        return x - y

    return cond(x.shape[0] > 2, true_fn, false_fn, [x, y])
