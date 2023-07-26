from typing import Any

import torch
import torch.utils._pytree as pytree


def autograd_not_implemented(
    result: torch.Tensor, args: Any, op_name: str, delayed=False
) -> Any:
    """If autograd is enabled and any of the operands require grad this will either
    raise an error or return a DelayedError depending on the value of delayed.

    Args:
        result: The result of the HigherOrderOperator, must be a Tensor that supports grad if delayed is set to True
        args: The flattened operands to the HigherOrderOperator
        message: The name of the HigherOrderOperator
        delayed: If True, return a DelayedError instead of raising an error

    Raises:
        RuntimeError: If autograd is enabled and any of the operands require grad
    """
    flat_operands, _ = pytree.tree_flatten(args)
    if torch.is_grad_enabled() and any(
        f.requires_grad for f in flat_operands if isinstance(f, torch.Tensor)
    ):
        if delayed:
            err_fn = torch._C._functions.DelayedError(
                f"Autograd is not supported for {op_name}",
                1,
            )

            def fake_requires_grad(var):
                var = var.detach()
                var.requires_grad = True
                return var

            return err_fn(fake_requires_grad(result))
        else:
            raise RuntimeError(f"Autograd is not supported for {op_name}")
    return result
