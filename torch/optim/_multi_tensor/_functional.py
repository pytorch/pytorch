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


def radam(params: List[Tensor],
          grads: List[Tensor],
          exp_avg: List[Tensor],
          exp_avg_sq: List[Tensor],
          states: List[Dict],
          *,
          beta1: float,
          beta2: float,
          lr: float,
          weight_decay: float,
          eps: float):
    r"""Functional API that performs RAdam algorithm computation.

    See :class:`~torch.optim.RAdam` for details.
    """

    # maximum length of the approximated SMA
    rho_inf = 2 / (1 - beta2) - 1
    # compute the length of the approximated SMA
    rho_t_list = [rho_inf - 2 * state['step'] * (beta2 ** state['step']) / (1 - beta2 ** state['step']) for state in states]

    bias_correction1 = [1 - beta1 ** state['step'] for state in states]
    bias_correction2 = [1 - beta2 ** state['step'] for state in states]
    if weight_decay != 0:
        torch._foreach_add_(grads, params, alpha=weight_decay)

    # Decay the first and second moment running average coefficient
    torch._foreach_mul_(exp_avg, beta1)
    torch._foreach_add_(exp_avg, grads, alpha=1 - beta1)

    torch._foreach_mul_(exp_avg_sq, beta2)
    torch._foreach_addcmul_(exp_avg_sq, grads, grads, 1 - beta2)

    rect = [math.sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
            if rho_t > 5 else 0 for rho_t in rho_t_list]
    unrectified = [0 if rect > 0 else 1. for rect in rect]

    exp_avg_sq_sqrt = torch._foreach_sqrt(exp_avg_sq)
    bias_correction_sqrt = [math.sqrt(bc) for bc in bias_correction2]
    denom = torch._foreach_div(exp_avg_sq_sqrt, bias_correction_sqrt)
    step_size = [(lr * rect / bc) * -1 for rect, bc in zip(rect, bias_correction1)]
    torch._foreach_addcdiv_(params, exp_avg, denom, step_size)

    denom = [torch.ones_like(exp_av, memory_format=torch.preserve_format) for exp_av in exp_avg]
    step_size = [(lr * rect / bc) * -1 for rect, bc in zip(unrectified, bias_correction1)]
    torch._foreach_addcdiv_(params, exp_avg, denom, step_size)
