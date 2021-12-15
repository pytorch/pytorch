r"""Functional interface"""
import torch
from torch import Tensor
from typing import List
from ._single_tensor_functional import single_tensor_adagrad

def multi_tensor_adagrad(params: List[Tensor],
                         grads: List[Tensor],
                         state_sums: List[Tensor],
                         state_steps: List[int],
                         has_sparse_grad: bool,
                         *,
                         lr: float,
                         weight_decay: float,
                         lr_decay: float,
                         eps: float):
    r"""Functional API that performs multi tensor Adagrad algorithm computation.

    See :class:`~torch.optim.Adagrad` for details.
    """

    if has_sparse_grad:
        return single_tensor_adagrad(params,
                                     grads,
                                     state_sums,
                                     state_steps,
                                     lr=lr,
                                     weight_decay=weight_decay,
                                     lr_decay=lr_decay,
                                     eps=eps)

    if weight_decay != 0:
        if has_sparse_grad:
            raise RuntimeError(
                "weight_decay option is not compatible with sparse gradients"
            )
        torch._foreach_add_(grads, params, alpha=weight_decay)

    minus_clr = [-lr / (1 + (step - 1) * lr_decay) for step in state_steps]

    grads = [torch.view_as_real(x) if torch.is_complex(x) else x for x in grads]
    state_sums = [torch.view_as_real(x) if torch.is_complex(x) else x for x in state_sums]
    torch._foreach_addcmul_(state_sums, grads, grads, value=1)
    std = torch._foreach_add(torch._foreach_sqrt(state_sums), eps)
    toAdd = torch._foreach_div(torch._foreach_mul(grads, minus_clr), std)
    toAdd = [torch.view_as_complex(x) if torch.is_complex(params[i]) else x for i, x in enumerate(toAdd)]
    torch._foreach_add_(params, toAdd)
    state_sums = [torch.view_as_complex(x) if torch.is_complex(params[i]) else x for i, x in enumerate(state_sums)]


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
