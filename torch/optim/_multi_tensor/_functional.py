r"""Functional interface"""
import math
import torch
from torch import Tensor
from typing import List


def asgd(params: List[Tensor],
         grads: List[Tensor],
         axs: List[Tensor],
         mus: List[Tensor],
         etas: List[Tensor],
         state_steps: List[Tensor],
         *,
         lambd: float,
         lr: float,
         t0: float,
         alpha: float,
         weight_decay: float):
    r"""Functional API that performs ASGD algorithm computation.
    See :class:`~torch.optim.ASGD` for details.
    """

    # update step
    torch._foreach_add_(state_steps, 1)

    if weight_decay != 0:
        torch._foreach_add_(grads, params, alpha=weight_decay)

    # decay term
    eta = etas[0].item()
    torch._foreach_mul_(params, 1 - lambd * eta)

    # update parameter
    torch._foreach_add_(params, grads, alpha=-eta)

    # averaging
    for i in range(len(axs)):
        if mus[i].item() != 1:
            axs[i].add_(params[i].sub(axs[i]).mul(mus[i]))
        else:
            axs[i].copy_(params[i])

    # update eta and mu
    for i in range(len(mus)):
        new_eta = lr / math.pow((1 + lambd * lr * state_steps[i].item()), alpha)
        etas[i].fill_(new_eta)
        new_mu = 1 / max(1, state_steps[i].item() - t0)
        mus[i].fill_(new_mu)
