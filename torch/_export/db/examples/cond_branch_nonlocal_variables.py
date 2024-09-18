# mypy: allow-untyped-defs
import torch

from functorch.experimental.control_flow import cond

class CondBranchNonlocalVariables(torch.nn.Module):
    """
    The branch functions (`true_fn` and `false_fn`) passed to cond() must follow these rules:
    - both branches must take the same args, which must also match the branch args passed to cond.
    - both branches must return a single tensor
    - returned tensor must have the same tensor metadata, e.g. shape and dtype
    - branch function can be free function, nested function, lambda, class methods
    - branch function can not have closure variables
    - no inplace mutations on inputs or global variables

    This example demonstrates how to rewrite code to avoid capturing closure variables in branch functions.

    The code below will not work because capturing closure variables is not supported.
    ```
    my_tensor_var = x + 100
    my_primitive_var = 3.14

    def true_fn(y):
        nonlocal my_tensor_var, my_primitive_var
        return y + my_tensor_var + my_primitive_var

    def false_fn(y):
        nonlocal my_tensor_var, my_primitive_var
        return y - my_tensor_var - my_primitive_var

    return cond(x.shape[0] > 5, true_fn, false_fn, [x])
    ```

    NOTE: If the `pred` is test on a dim with batch size < 2, it will be specialized.
    """

    def forward(self, x):
        my_tensor_var = x + 100
        my_primitive_var = 3.14

        def true_fn(x, y, z):
            return x + y + z

        def false_fn(x, y, z):
            return x - y - z

        return cond(
            x.shape[0] > 5,
            true_fn,
            false_fn,
            [x, my_tensor_var, torch.tensor(my_primitive_var)],
        )

example_args = (torch.randn(6),)
tags = {
    "torch.cond",
    "torch.dynamic-shape",
}
model = CondBranchNonlocalVariables()
