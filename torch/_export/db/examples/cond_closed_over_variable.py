import torch

from torch._export.db.case import export_case, export_rewrite_case, SupportLevel
from functorch.experimental.control_flow import cond


@export_case(
    example_inputs=(torch.tensor(True), torch.ones(3, 2)),
    tags={"torch.cond", "python.closure"},
    support_level=SupportLevel.NOT_SUPPORTED_YET,
)
class CondClosedOverVariable(torch.nn.Module):
    """
    torch.cond() doesn't support branches closed over arbitrary variables.
    """

    def forward(self, pred, x):
        def true_fn(val):
            return x * 2

        def false_fn(val):
            return x - 2

        return cond(pred, true_fn, false_fn, [x + 1])


@export_rewrite_case(parent=CondClosedOverVariable)
class CondClosedOverVariableRewrite(torch.nn.Module):
    """
    Users may need to make captured variables explicitly to arguments.
    """

    def forward(self, pred, x):
        def true_fn(val, x):
            return x * 2

        def false_fn(val, x):
            return x - 2

        return cond(pred, true_fn, false_fn, [x + 1, x])
