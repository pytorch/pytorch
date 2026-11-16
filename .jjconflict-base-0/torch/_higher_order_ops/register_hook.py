from collections.abc import Callable
from typing import Any

import torch
from torch._ops import HigherOrderOperator


class RegisterHookOp(HigherOrderOperator):
    """HOP that registers a backward hook on a tensor.

    The hook body is a traced subgraph that takes (grad, *freevars)
    and returns the modified gradient. During make_fx tracing, the
    hook registers on the FakeTensor and fires on the accumulated
    gradient during backward tracing, capturing the hook's effect
    in the joint graph.
    """

    def __init__(self) -> None:
        super().__init__("register_hook", cacheable=True)

    def __call__(
        self,
        tensor: torch.Tensor,
        hook_body: Callable[..., Any],
        *freevars: Any,
    ) -> torch.Tensor:
        def hook(grad: torch.Tensor) -> torch.Tensor | None:
            return hook_body(grad, *freevars)

        tensor.register_hook(hook)
        return tensor


register_hook_op = RegisterHookOp()
