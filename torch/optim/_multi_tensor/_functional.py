r"""Functional interface"""
import math
import torch
from torch import Tensor
from typing import List, Dict


def asgd(params: List[Tensor],
         grads: List[Tensor],
         states: List[Dict],
         lambd: float,
         lr: float,
         t0: float,
         alpha: float,
         weight_decay: float):
    r"""Functional API that performs ASGD algorithm computation.
    See :class:`~torch.optim.ASGD` for details.
    """

    if weight_decay != 0:
        torch._foreach_add_(grads, params, alpha=weight_decay)

    # decay term
    eta = states[0]['eta']
    torch._foreach_mul_(params, 1 - lambd * eta)

    # update parameter
    torch._foreach_add_(params, grads, alpha=-eta)

    # averaging
    for i in range(len(states)):
        if states[i]['mu'] != 1:
            states[i]['ax'].add_(params[i].sub(states[i]['ax']).mul(states[i]['mu']))
        else:
            states[i]['ax'].copy_(params[i])

    # update eta and mu
    for state in states:
        state['eta'] = (lr /
                        math.pow((1 + lambd * lr * state['step']), alpha))
        state['mu'] = 1 / max(1, state['step'] - t0)
