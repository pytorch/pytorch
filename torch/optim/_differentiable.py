r"""Differentiable Functional interface"""
import math
import torch
from torch import Tensor
from typing import List, Optional

def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool):
    r"""Differentiable Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = d_p
                momentum_buffer_list[i] = buf
            else:
                buf = buf * momentum + d_p * (1 - dampening)
                momentum_buffer_list[i] = buf

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        params[i] = params[i] - d_p * lr


def adam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[int],
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float):
    r"""Differentiable Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """

    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg = exp_avg * beta1 + (1 - beta1) * grad
        exp_avg_sq = exp_avg_sq * beta2 + (1 - beta2) * grad * grad
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        params[i] = params[i] - step_size * exp_avg / denom


def adamw(params: List[Tensor],
          grads: List[Tensor],
          exp_avgs: List[Tensor],
          exp_avg_sqs: List[Tensor],
          max_exp_avg_sqs: List[Tensor],
          state_steps: List[int],
          *,
          amsgrad: bool,
          beta1: float,
          beta2: float,
          lr: float,
          weight_decay: float,
          eps: float):
    r"""Differentiable Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """

    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        params[i] = params[i] * (1 - lr * weight_decay)

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg = exp_avg * beta1 + (1 - beta1) * grad
        exp_avg_sq = exp_avg_sq * beta2 + (1 - beta2) * grad * grad
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        params[i] = params[i] - step_size * exp_avg / denom


def adadelta(params: List[Tensor],
             grads: List[Tensor],
             square_avgs: List[Tensor],
             acc_deltas: List[Tensor],
             *,
             lr: float,
             rho: float,
             eps: float,
             weight_decay: float):
    r"""Differentiable Functional API that performs Adadelta algorithm computation.

    See :class:`~torch.optim.Adadelta` for details.
    """

    for i, (param, grad, square_avg, acc_delta) in enumerate(zip(params, grads, square_avgs, acc_deltas)):
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        square_avg = square_avg * rho + (1 - rho) * grad * grad
        std = square_avg.add(eps).sqrt_()
        delta = acc_delta.add(eps).sqrt_().div_(std).mul_(grad)
        params[i] = params[i] - lr * delta
        acc_delta = rho * acc_delta + (1 - rho) * delta * delta
