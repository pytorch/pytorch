from typing import Any, Callable

import torch
import torch.utils._pytree as pytree
from torch._ops import HigherOrderOperator


def autograd_not_implemented_inner(
    operator: HigherOrderOperator, delayed_error: bool, *args: Any, **kwargs: Any
) -> Any:
    """If autograd is enabled and any of the arguments require grad this will either
    raise an error or return a DelayedError depending on the value of delayed.

    Args:
        operator: The HigherOrderOperator to call with the *args and **kwargs with
        op_name: The name of the HigherOrderOperator
        delayed_error: If True, return a DelayedError instead of raising an error
        args: The flattened operands to the HigherOrderOperator
        kwargs: The keyword arguments to the HigherOrderOperator

    Raises:
        RuntimeError: If autograd is enabled and any of the arguments to the HigherOrderOperator
    """
    with torch._C._AutoDispatchBelowAutograd():
        result = operator(*args, **kwargs)
        flat_operands, _ = pytree.tree_flatten(args)
        if torch.is_grad_enabled() and any(
            f.requires_grad for f in flat_operands if isinstance(f, torch.Tensor)
        ):
            if delayed_error:
                err_fn = torch._C._functions.DelayedError(
                    f"Autograd not implemented for {str(operator)}",
                    1,
                )

                def fake_requires_grad(tensor):
                    if torch.is_floating_point(tensor) or torch.is_complex(tensor):
                        tensor = tensor.detach()
                        tensor.requires_grad = True
                    return tensor

                return pytree.tree_map_only(
                    torch.Tensor, lambda x: err_fn(fake_requires_grad(x)), result
                )
            else:
                raise RuntimeError(f"Autograd not implemented for {str(operator)}")
        return result


def autograd_not_implemented(op: HigherOrderOperator, deferred_error: bool) -> Callable:
    def inner(*args, **kwargs):
        return autograd_not_implemented_inner(op, deferred_error, *args, **kwargs)

    return inner
