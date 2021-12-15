r"""Functional interface"""
import torch
from torch import Tensor
from typing import List

def multi_tensor_adadelta(params: List[Tensor],
                          grads: List[Tensor],
                          square_avgs: List[Tensor],
                          acc_deltas: List[Tensor],
                          *,
                          lr: float,
                          weight_decay: float,
                          rho: float,
                          eps: float):
    r"""Functional API that performs multi tensor Adadelta algorithm computation.

    See :class:`~torch.optim.Adadelta` for details.
    """

    if weight_decay != 0:
        torch._foreach_add_(grads, params, alpha=weight_decay)

    torch._foreach_mul_(square_avgs, rho)
    torch._foreach_addcmul_(square_avgs, grads, grads, value=1 - rho)

    std = torch._foreach_add(square_avgs, eps)
    torch._foreach_sqrt_(std)

    deltas = torch._foreach_add(acc_deltas, eps)
    torch._foreach_sqrt_(deltas)
    torch._foreach_div_(deltas, std)
    torch._foreach_mul_(deltas, grads)

    torch._foreach_add_(params, deltas, alpha=-lr)

    torch._foreach_mul_(acc_deltas, rho)
    torch._foreach_addcmul_(acc_deltas, deltas, deltas, value=1 - rho)
