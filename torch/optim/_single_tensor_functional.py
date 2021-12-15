r"""Functional interface"""
import torch
from torch import Tensor
from typing import List


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
