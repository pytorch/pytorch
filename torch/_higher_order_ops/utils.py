from typing import Any, Callable

import torch
import torch.utils._pytree as pytree


def autograd_not_implemented(
    operator: Callable, op_name: str, delayed: bool, *args: Any, **kwargs: Any
) -> Any:
    """If autograd is enabled and any of the arguments require grad this will either
    raise an error or return a DelayedError depending on the value of delayed.

    Args:
        operator: The HigherOrderOperator to call with the *args and **kwargs with
        op_name: The name of the HigherOrderOperator
        delayed: If True, return a DelayedError instead of raising an error
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
            if delayed:
                err_fn = torch._C._functions.DelayedError(
                    f"Autograd not implemented for {op_name}",
                    1,
                )

                def fake_requires_grad(tensor):
                    if torch.is_floating_point(tensor) or torch.is_complex(tensor):
                        tensor = tensor.detach()
                        tensor.requires_grad = True
                    return tensor

                pytree.tree_map_only(torch.Tensor, fake_requires_grad, (args, kwargs))

                return err_fn(fake_requires_grad(result))
            else:
                raise RuntimeError(f"Autograd not implemented for {op_name}")
        return result
