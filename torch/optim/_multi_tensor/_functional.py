r"""Functional interface"""
import math
import torch
from torch import Tensor
from typing import List, Dict

def adamax(params: List[Tensor],
           grads: List[Tensor],
           exp_avgs: List[Tensor],
           exp_infs: List[Tensor],
           states: List[Dict],
           *,
           beta1: float,
           beta2: float,
           lr: float,
           weight_decay: float,
           eps: float):
    r"""Functional API that performs Adamax algorithm computation.

    See :class:`~torch.optim.Adamax` for details.
    """

    if weight_decay != 0:
        torch._foreach_add_(grads, params, alpha=weight_decay)

    # Update biased first moment estimate.
    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)

    # Update the exponentially weighted infinity norm.
    torch._foreach_mul_(exp_infs, beta2)

    for exp_inf, grad in zip(exp_infs, grads):
        norm_buf = torch.cat([
            exp_inf.unsqueeze(0),
            grad.abs().add_(eps).unsqueeze_(0)
        ], 0)
        torch.max(norm_buf, 0, keepdim=False, out=(exp_inf, exp_inf.new().long()))

    bias_corrections = [1 - beta1 ** state['step'] for state in states]
    clr = [-1 * (lr / bias_correction) for bias_correction in bias_corrections]
    torch._foreach_addcdiv_(params, exp_avgs, exp_infs, clr)


def adadelta(params: List[Tensor],
             grads: List[Tensor],
             square_avgs: List[Tensor],
             acc_deltas: List[Tensor],
             *,
             lr: float,
             weight_decay: float,
             rho: float,
             eps: float):
    r"""Functional API that performs Adadelta algorithm computation.

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


def nadam(params: List[Tensor],
          grads: List[Tensor],
          exp_avg: List[Tensor],
          exp_avg_sq: List[Tensor],
          mu_products: List[Tensor],
          states: List[Dict],
          *,
          beta1: float,
          beta2: float,
          lr: float,
          weight_decay: float,
          momentum_decay: float,
          eps: float):
    r"""Functional API that performs NAdam algorithm computation.

    See :class:`~torch.optim.NAdam` for details.
    """

    bias_correction1 = [1 - beta1 ** state['step'] for state in states]
    bias_correction2 = [1 - beta2 ** state['step'] for state in states]
    mus = [beta1 * (1. - 0.5 * (0.96 ** (state['step'] * momentum_decay))) for state in states]
    mu_nexts = [beta1 * (1. - 0.5 * (0.96 ** ((state['step'] + 1) * momentum_decay)))
                for state in states]
    if weight_decay != 0:
        torch._foreach_add_(grads, params, alpha=weight_decay)

    # Decay the first and second moment running average coefficient
    torch._foreach_mul_(exp_avg, beta1)
    torch._foreach_add_(exp_avg, grads, alpha=1 - beta1)

    torch._foreach_mul_(exp_avg_sq, beta2)
    torch._foreach_addcmul_(exp_avg_sq, grads, grads, 1 - beta2)

    exp_avg_sq_sqrt = torch._foreach_sqrt(exp_avg_sq)
    bias_correction_sqrt = [math.sqrt(bc) for bc in bias_correction2]
    torch._foreach_div_(exp_avg_sq_sqrt, bias_correction_sqrt)
    denom = torch._foreach_add(exp_avg_sq_sqrt, eps)

    step_size_grads = [(lr * (1. - mu) / (1. - mu_product)) * -1
                       for mu_product, mu in zip(mu_products, mus)]
    step_size_expavg = [(lr * mu_next / (1. - mu_product * mu_next)) * -1
                        for mu_product, mu_next in zip(mu_products, mu_nexts)]
    torch._foreach_addcdiv_(params, grads, denom, step_size_grads)
    torch._foreach_addcdiv_(params, exp_avg, denom, step_size_expavg)


def adam(params: List[Tensor],
         grads: List[Tensor],
         exp_avg: List[Tensor],
         exp_avg_sq: List[Tensor],
         max_exp_avg_sq: List[Tensor],
         states: List[Dict],
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float):
    r"""Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """

    bias_correction1 = [1 - beta1 ** state['step'] for state in states]
    bias_correction2 = [1 - beta2 ** state['step'] for state in states]
    if weight_decay != 0:
        torch._foreach_add_(grads, params, alpha=weight_decay)

    torch._foreach_mul_(exp_avg, beta1)
    torch._foreach_add_(exp_avg, grads, alpha=1 - beta1)

    torch._foreach_mul_(exp_avg_sq, beta2)
    torch._foreach_addcmul_(exp_avg_sq, grads, grads, 1 - beta2)

    if amsgrad:
        # Maintains the maximum of all 2nd moment running avg. till now
        max_exp_avg_sq = torch._foreach_maximum(max_exp_avg_sq, exp_avg_sq)  # type: ignore[assignment]

        # Use the max. for normalizing running avg. of gradient
        max_exp_avg_sq_sqrt = torch._foreach_sqrt(max_exp_avg_sq)
        bias_correction_sqrt = [math.sqrt(bc) for bc in bias_correction2]
        torch._foreach_div_(max_exp_avg_sq_sqrt, bias_correction_sqrt)
        denom = torch._foreach_add(max_exp_avg_sq_sqrt, eps)
    else:
        exp_avg_sq_sqrt = torch._foreach_sqrt(exp_avg_sq)
        bias_correction_sqrt = [math.sqrt(bc) for bc in bias_correction2]
        torch._foreach_div_(exp_avg_sq_sqrt, bias_correction_sqrt)
        denom = torch._foreach_add(exp_avg_sq_sqrt, eps)

    step_size = [(lr / bc) * -1 for bc in bias_correction1]
    torch._foreach_addcdiv_(params, exp_avg, denom, step_size)


def adamw(params: List[Tensor],
          grads: List[Tensor],
          exp_avg: List[Tensor],
          exp_avg_sq: List[Tensor],
          max_exp_avg_sq: List[Tensor],
          states: List[Dict],
          *,
          amsgrad: bool,
          beta1: float,
          beta2: float,
          lr: float,
          eps: float):
    r"""Functional API that performs Adamw algorithm computation.
    See :class:`~torch.optim.Adamw` for details.
    """

    bias_correction1 = [1 - beta1 ** state['step'] for state in states]
    bias_correction2 = [1 - beta2 ** state['step'] for state in states]

    #
    # Decay the first and second moment running average coefficient
    #
    torch._foreach_mul_(exp_avg, beta1)
    torch._foreach_add_(exp_avg, grads, alpha=1 - beta1)

    torch._foreach_mul_(exp_avg_sq, beta2)
    torch._foreach_addcmul_(exp_avg_sq, grads, grads, 1 - beta2)

    if amsgrad:
        # Maintains the maximum of all 2nd moment running avg. till now
        max_exp_avg_sq = torch._foreach_maximum(max_exp_avg_sq, exp_avg_sq)  # type: ignore[assignment]

        # Use the max. for normalizing running avg. of gradient
        max_exp_avg_sq_sqrt = torch._foreach_sqrt(max_exp_avg_sq)
        bias_correction_sqrt = [math.sqrt(bc) for bc in bias_correction2]
        torch._foreach_div_(max_exp_avg_sq_sqrt, bias_correction_sqrt)
        denom = torch._foreach_add(max_exp_avg_sq_sqrt, eps)
    else:
        exp_avg_sq_sqrt = torch._foreach_sqrt(exp_avg_sq)
        bias_correction_sqrt = [math.sqrt(bc) for bc in bias_correction2]
        torch._foreach_div_(exp_avg_sq_sqrt, bias_correction_sqrt)
        denom = torch._foreach_add(exp_avg_sq_sqrt, eps)

    step_size = [-1 * (lr / bc) for bc in bias_correction1]
    torch._foreach_addcdiv_(params, exp_avg, denom, step_size)


def sgd(params: List[Tensor],
        grads: List[Tensor],
        states: List[Dict],
        *,
        momentum : float,
        has_sparse_grad: bool,
        dampening: float,
        nesterov: bool,
        weight_decay: float,
        lr: float):
    r"""Functional API that performs SGD algorithm computation.
    See :class:`~torch.optim.SGD` for details.
    """

    if weight_decay != 0:
        grads = torch._foreach_add(grads, params, alpha=weight_decay)  # type: ignore[assignment]

    if momentum != 0:
        bufs = []

        all_states_with_momentum_buffer = True
        for i in range(len(states)):
            if 'momentum_buffer' not in states[i]:
                all_states_with_momentum_buffer = False
                break
            else:
                bufs.append(states[i]['momentum_buffer'])

        if all_states_with_momentum_buffer:
            torch._foreach_mul_(bufs, momentum)
            torch._foreach_add_(bufs, grads, alpha=1 - dampening)
        else:
            bufs = []
            for i in range(len(states)):
                if 'momentum_buffer' not in states[i]:
                    buf = states[i]['momentum_buffer'] = torch.clone(grads[i]).detach()
                else:
                    buf = states[i]['momentum_buffer']
                    buf.mul_(momentum).add_(grads[i], alpha=1 - dampening)

                bufs.append(buf)

        if nesterov:
            torch._foreach_add_(grads, bufs, alpha=momentum)
        else:
            grads = bufs

    if not has_sparse_grad:
        torch._foreach_add_(params, grads, alpha=-lr)
    else:
        # foreach APIs dont support sparse
        for i in range(len(params)):
            params[i].add_(grads[i], alpha=-lr)


def rprop(params: List[Tensor],
          grads: List[Tensor],
          states: List[Tensor],
          step_sizes: List[int],
          *,
          step_size_max: float,
          step_size_min: float,
          etaminus: float,
          etaplus: float):
    r"""Functional API that performs Rprop algorithm computation.
    See :class:`~torch.optim.Rprop` for details.
    """

    signs = torch._foreach_mul(grads, [s['prev'] for s in states])  # type: ignore[misc, index]
    signs = [s.sign() for s in signs]
    for sign in signs:
        sign[sign.gt(0)] = etaplus
        sign[sign.lt(0)] = etaminus
        sign[sign.eq(0)] = 1

    # update stepsizes with step size updates
    torch._foreach_mul_(step_sizes, signs)  # type: ignore[arg-type]
    for step_size in step_sizes:
        step_size.clamp_(step_size_min, step_size_max)  # type: ignore[attr-defined]

    # for dir<0, dfdx=0
    # for dir>=0 dfdx=dfdx
    for i in range(len(grads)):
        grads[i] = grads[i].clone(memory_format=torch.preserve_format)
        grads[i][signs[i].eq(etaminus)] = 0

    # update parameters
    grad_signs = [grad.sign() for grad in grads]
    torch._foreach_addcmul_(params, grad_signs, step_sizes, value=-1)   # type: ignore[name-defined, arg-type]

    for i in range(len(states)):
        states[i]['prev'].copy_(grads[i])  # type: ignore[index]
