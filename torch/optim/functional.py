r"""Functional interface"""
import math
import torch
from torch import Tensor
from typing import List

# TODO: use foreach API in optim.functional to do all the computation

def _make_sparse(grad, grad_indices, values):
    size = grad.size()
    if grad_indices.numel() == 0 or values.numel() == 0:
        return torch.empty_like(grad)
    return torch.sparse_coo_tensor(grad_indices, values, size)


def adagrad(params: List[Tensor],
            grads: List[Tensor],
            state_sums: List[Tensor],
            state_steps: List[int],
            lr: float,
            weight_decay: float,
            lr_decay: float,
            eps: float):
    r"""Functional API that performs Adagrad algorithm computation.

    See :class:`~torch.optim.Adagrad` for details.
    """

    for (param, grad, state_sum, step) in zip(params, grads, state_sums, state_steps):
        if weight_decay != 0:
            if grad.is_sparse:
                raise RuntimeError("weight_decay option is not compatible with sparse gradients")
            grad = grad.add(param, alpha=weight_decay)

        clr = lr / (1 + (step - 1) * lr_decay)

        if grad.is_sparse:
            grad = grad.coalesce()  # the update is non-linear so indices must be unique
            grad_indices = grad._indices()
            grad_values = grad._values()
            size = grad.size()

            state_sum.add_(_make_sparse(grad, grad_indices, grad_values.pow(2)))
            std = state_sum.sparse_mask(grad)
            std_values = std._values().sqrt_().add_(eps)
            param.add_(_make_sparse(grad, grad_indices, grad_values / std_values), alpha=-clr)
        else:
            state_sum.addcmul_(grad, grad, value=1)
            std = state_sum.sqrt().add_(eps)
            param.addcdiv_(grad, std, value=-clr)


def adam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[int],
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float):
    r"""Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """

    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]
        if amsgrad:
            max_exp_avg_sq = max_exp_avg_sqs[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)
