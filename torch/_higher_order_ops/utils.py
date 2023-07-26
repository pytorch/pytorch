from typing import Any

import torch
import torch.utils._pytree as pytree


def autograd_not_implemented_check(args: Any, op_name: str) -> None:
    """If autograd is enabled and any of the operands require grad this will raise an error with message.

    Args:
        operands (List[Any]): The flattened operands to the HigherOrderOperator
        message (str): The name of the HigherOrderOperator

    Raises:
        RuntimeError: If autograd is enabled and any of the operands require grad
    """
    flat_operands, _ = pytree.tree_flatten(args)
    if torch.is_grad_enabled() and any(
        f.requires_grad for f in flat_operands if isinstance(f, torch.Tensor)
    ):
        raise RuntimeError(f"Autograd is not supported for {op_name}")
