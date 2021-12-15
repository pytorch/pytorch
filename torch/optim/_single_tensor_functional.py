r"""Functional interface"""
import torch
from torch import Tensor
from typing import List

def _make_sparse(grad, grad_indices, values):
    size = grad.size()
    if grad_indices.numel() == 0 or values.numel() == 0:
        return torch.empty_like(grad)
    return torch.sparse_coo_tensor(grad_indices, values, size)


def single_tensor_adagrad(params: List[Tensor],
                          grads: List[Tensor],
                          state_sums: List[Tensor],
                          state_steps: List[int],
                          *,
                          lr: float,
                          weight_decay: float,
                          lr_decay: float,
                          eps: float):
    r"""Functional API that performs single tensor Adagrad algorithm computation.

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
            is_complex = torch.is_complex(param)
            if is_complex:
                grad = torch.view_as_real(grad)
                state_sum = torch.view_as_real(state_sum)
                param = torch.view_as_real(param)
            state_sum.addcmul_(grad, grad, value=1)
            std = state_sum.sqrt().add_(eps)
            param.addcdiv_(grad, std, value=-clr)
            if is_complex:
                param = torch.view_as_complex(param)
                state_sum = torch.view_as_complex(state_sum)


def single_tensor_adadelta(params: List[Tensor],
                           grads: List[Tensor],
                           square_avgs: List[Tensor],
                           acc_deltas: List[Tensor],
                           *,
                           lr: float,
                           rho: float,
                           eps: float,
                           weight_decay: float):
    r"""Functional API that performs single tensor Adadelta algorithm computation.

    See :class:`~torch.optim.Adadelta` for details.
    """

    for (param, grad, square_avg, acc_delta) in zip(params, grads, square_avgs, acc_deltas):
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        if torch.is_complex(param):
            square_avg = torch.view_as_real(square_avg)
            acc_delta = torch.view_as_real(acc_delta)
            grad = torch.view_as_real(grad)

        square_avg.mul_(rho).addcmul_(grad, grad, value=1 - rho)
        std = square_avg.add(eps).sqrt_()
        delta = acc_delta.add(eps).sqrt_().div_(std).mul_(grad)
        acc_delta.mul_(rho).addcmul_(delta, delta, value=1 - rho)
        if torch.is_complex(param):
            delta = torch.view_as_complex(delta)
        param.add_(delta, alpha=-lr)


def single_tensor_adamax(params: List[Tensor],
                         grads: List[Tensor],
                         exp_avgs: List[Tensor],
                         exp_infs: List[Tensor],
                         state_steps: List[int],
                         *,
                         eps: float,
                         beta1: float,
                         beta2: float,
                         lr: float,
                         weight_decay: float):
    r"""Functional API that performs single tensor adamax algorithm computation.

    See :class:`~torch.optim.Adamax` for details.
    """

    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_inf = exp_infs[i]
        step = state_steps[i]

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Update biased first moment estimate.
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        # Update the exponentially weighted infinity norm.
        norm_buf = torch.cat([
            exp_inf.mul_(beta2).unsqueeze(0),
            grad.abs().add_(eps).unsqueeze_(0)
        ], 0)
        torch.amax(norm_buf, 0, keepdim=False, out=exp_inf)

        bias_correction = 1 - beta1 ** step
        clr = lr / bias_correction

        param.addcdiv_(exp_avg, exp_inf, value=-clr)
