r"""Functional interface"""
import math
import torch
from torch import Tensor
from typing import List, Dict

def _make_sparse(grad, grad_indices, values):
    size = grad.size()
    if grad_indices.numel() == 0 or values.numel() == 0:
        return torch.empty_like(grad)
    return torch.sparse_coo_tensor(grad_indices, values, size)


def adagrad(
    params: List[Tensor],
    grads: List[Tensor],
    state_sums: List[Tensor],
    state_steps: List[Tensor],
    has_sparse_grad: bool,
    *,
    lr: float,
    weight_decay: float,
    lr_decay: float,
    eps: float
):
    r"""Functional API that performs Adagrad algorithm computation.

    See :class:`~torch.optim.Adagrad` for details.
    """

    if not all([isinstance(t, torch.Tensor) for t in state_steps]):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    # Update steps
    torch._foreach_add_(state_steps, 1)

    if weight_decay != 0:
        if has_sparse_grad:
            raise RuntimeError(
                "weight_decay option is not compatible with sparse gradients"
            )
        torch._foreach_add_(grads, params, alpha=weight_decay)

    minus_clr = [-lr / (1 + (step.item() - 1) * lr_decay) for step in state_steps]

    if has_sparse_grad:
        # sparse is not supported by multi_tensor. Fall back to optim.adagrad
        # implementation for sparse gradients
        for i, (param, grad, state_sum, step) in enumerate(
            zip(params, grads, state_sums, state_steps)
        ):
            grad = grad.coalesce()  # the update is non-linear so indices must be unique
            grad_indices = grad._indices()
            grad_values = grad._values()
            size = grad.size()

            state_sum.add_(_make_sparse(grad, grad_indices, grad_values.pow(2)))
            std_sparse = state_sum.sparse_mask(grad)
            std_sparse_values = std_sparse._values().sqrt_().add_(eps)
            param.add_(
                _make_sparse(grad, grad_indices, grad_values / std_sparse_values),
                alpha=minus_clr[i],
            )
    else:
        grads = [torch.view_as_real(x) if torch.is_complex(x) else x for x in grads]
        state_sums = [torch.view_as_real(x) if torch.is_complex(x) else x for x in state_sums]
        torch._foreach_addcmul_(state_sums, grads, grads, value=1)
        std = torch._foreach_add(torch._foreach_sqrt(state_sums), eps)
        toAdd = torch._foreach_div(torch._foreach_mul(grads, minus_clr), std)
        toAdd = [torch.view_as_complex(x) if torch.is_complex(params[i]) else x for i, x in enumerate(toAdd)]
        torch._foreach_add_(params, toAdd)
        state_sums = [torch.view_as_complex(x) if torch.is_complex(params[i]) else x for i, x in enumerate(state_sums)]


def adamax(params: List[Tensor],
           grads: List[Tensor],
           exp_avgs: List[Tensor],
           exp_infs: List[Tensor],
           state_steps: List[Tensor],
           *,
           beta1: float,
           beta2: float,
           lr: float,
           weight_decay: float,
           eps: float):
    r"""Functional API that performs Adamax algorithm computation.

    See :class:`~torch.optim.Adamax` for details.
    """

    if not all([isinstance(t, torch.Tensor) for t in state_steps]):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    # Update steps
    torch._foreach_add_(state_steps, 1)

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

    bias_corrections = [1 - beta1 ** step.item() for step in state_steps]
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
          state_steps: List[Tensor],
          *,
          beta1: float,
          beta2: float,
          lr: float,
          weight_decay: float,
          eps: float):
    r"""Functional API that performs RAdam algorithm computation.

    See :class:`~torch.optim.RAdam` for details.
    """

    if not all([isinstance(t, torch.Tensor) for t in state_steps]):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    # Update steps
    torch._foreach_add_(state_steps, 1)

    # maximum length of the approximated SMA
    rho_inf = 2 / (1 - beta2) - 1
    # compute the length of the approximated SMA
    rho_t_list = [rho_inf - 2 * step.item() * (beta2 ** step.item()) / (1 - beta2 ** step.item()) for step in state_steps]

    bias_correction1 = [1 - beta1 ** step.item() for step in state_steps]
    bias_correction2 = [1 - beta2 ** step.item() for step in state_steps]
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
