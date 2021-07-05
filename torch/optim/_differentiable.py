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


def adagrad(params: List[Tensor],
            grads: List[Tensor],
            state_sums: List[Tensor],
            state_steps: List[int],
            *,
            lr: float,
            weight_decay: float,
            lr_decay: float,
            eps: float):
    r"""Differentiable Functional API that performs Adagrad algorithm computation.

    See :class:`~torch.optim.Adagrad` for details.
    """

    for i, (param, grad, state_sum, step) in enumerate(zip(params, grads, state_sums, state_steps)):
        if weight_decay != 0:
            if grad.is_sparse:
                raise RuntimeError("weight_decay option is not compatible with sparse gradients")
            grad = grad + weight_decay * param

        clr = lr / (1 + (step - 1) * lr_decay)

        if grad.is_sparse:
            grad = grad.coalesce()  # the update is non-linear so indices must be unique
            grad_indices = grad._indices()
            grad_values = grad._values()
            size = grad.size()

            state_sum = state_sum.add(_make_sparse(grad, grad_indices, grad_values.pow(2)))
            std = state_sum.sparse_mask(grad)
            std_values = std._values().sqrt().add(eps)
            params[i] = params[i] - clr * _make_sparse(grad, grad_indices, grad_values / std_values)
        else:
            state_sum = state_sum + grad * grad
            std = state_sum.sqrt() + eps
            params[i] = params[i] - clr * grad / std


def rmsprop(params: List[Tensor],
            grads: List[Tensor],
            square_avgs: List[Tensor],
            grad_avgs: List[Tensor],
            momentum_buffer_list: List[Tensor],
            *,
            lr: float,
            alpha: float,
            eps: float,
            weight_decay: float,
            momentum: float,
            centered: bool):
    r"""Differentiable Functional API that performs rmsprop algorithm computation.

    See :class:`~torch.optim.RMSProp` for details.
    """

    for i, param in enumerate(params):
        grad = grads[i]
        square_avg = square_avgs[i]

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        square_avg = square_avg * alpha + (1 - alpha) * grad * grad

        if centered:
            grad_avg = grad_avgs[i]
            grad_avg = grad_avg * alpha + (1 - alpha) * grad
            avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt().add(eps)
        else:
            avg = square_avg.sqrt().add(eps)

        if momentum > 0:
            buf = momentum_buffer_list[i]
            buf = buf * momentum + grad / avg
            params[i] = params[i] - lr * buf
        else:
            params[i] = params[i] - lr * grad / avg


def radam(params: List[Tensor],
          grads: List[Tensor],
          exp_avgs: List[Tensor],
          exp_avg_sqs: List[Tensor],
          state_steps: List[int],
          *,
          beta1: float,
          beta2: float,
          lr: float,
          weight_decay: float,
          eps: float):
    r"""Differentiable Functional API that performs RAdam algorithm computation.

    See :class:`~torch.optim.RAdam` for details.
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

        # correcting bias for the first moving moment
        bias_corrected_exp_avg = exp_avg / bias_correction1

        # maximum length of the approximated SMA
        rho_inf = 2 / (1 - beta2) - 1
        # compute the length of the approximated SMA
        rho_t = rho_inf - 2 * step * (beta2 ** step) / bias_correction2

        if rho_t > 5.:
            # Compute the variance rectification term and update parameters accordingly
            rect = math.sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
            adaptive_lr = math.sqrt(bias_correction2) / exp_avg_sq.sqrt().add(eps)

            params[i] = params[i] - bias_corrected_exp_avg * lr * adaptive_lr * rect
        else:
            params[i] = params[i] - bias_corrected_exp_avg * lr


def nadam(params: List[Tensor],
          grads: List[Tensor],
          exp_avgs: List[Tensor],
          exp_avg_sqs: List[Tensor],
          mu_products: List[float],
          state_steps: List[int],
          *,
          beta1: float,
          beta2: float,
          lr: float,
          weight_decay: float,
          momentum_decay: float,
          eps: float):
    r"""Differentiable Functional API that performs NAdam algorithm computation.

    See :class:`~torch.optim.NAdam` for details.
    """

    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        mu_product = mu_products[i]
        step = state_steps[i]

        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # calculate the momentum cache \mu^{t} and \mu^{t+1}
        mu = beta1 * (1. - 0.5 * (0.96 ** (step * momentum_decay)))
        mu_next = beta1 * (1. - 0.5 * (0.96 ** ((step + 1) * momentum_decay)))
        mu_product = mu_product * mu
        mu_product_next = mu_product * mu * mu_next

        # decay the first and second moment running average coefficient
        exp_avg = exp_avg * beta1 + (1 - beta1) * grad
        exp_avg_sq = exp_avg_sq * beta2 + (1 - beta2) * grad * grad

        denom = exp_avg_sq.div(bias_correction2).sqrt().add(eps)
        params[i] = params[i] - lr * grad / denom * (1. - mu) / (1. - mu_product)
        params[i] = params[i] - lr * exp_avg / denom * lr * mu_next / (1. - mu_product_next)
