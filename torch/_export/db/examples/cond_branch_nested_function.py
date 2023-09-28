import torch

from torch._export.db.case import export_case
from functorch.experimental.control_flow import cond


@export_case(
    example_inputs=(torch.ones(3),),
    tags={
        "torch.cond",
        "torch.dynamic-shape",
    },
)
def cond_branch_nested_function(x):
    """
    The branch functions (`true_fn` and `false_fn`) passed to cond() must follow these rules:
      - both branches must take the same args, which must also match the branch args passed to cond.
      - both branches must return a single tensor
      - returned tensor must have the same tensor metadata, e.g. shape and dtype
      - branch function can be free function, nested function, lambda, class methods
      - branch function can not have closure variables
      - no inplace mutations on inputs or global variables

    This example demonstrates using nested function in cond().

    NOTE: If the `pred` is test on a dim with batch size < 2, it will be specialized.
    """

    def true_fn(x):
        def inner_true_fn(y):
            return x + y

        return inner_true_fn(x)

    def false_fn(x):
        def inner_false_fn(y):
            return x - y

        return inner_false_fn(x)

    return cond(x.shape[0] < 10, true_fn, false_fn, [x])
