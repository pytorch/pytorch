r"""Functional interface"""
import math
import torch
from torch import Tensor
from typing import List

from .adadelta import adadelta  # type: ignore[attr-defined] # noqa: F401
from .adagrad import adagrad  # type: ignore[attr-defined] # noqa: F401
from .adam import adam  # type: ignore[attr-defined] # noqa: F401
from .adamax import adamax  # type: ignore[attr-defined] # noqa: F401
from .asgd import asgd  # type: ignore[attr-defined] # noqa: F401
from .nadam import nadam  # type: ignore[attr-defined] # noqa: F401
from .radam import radam  # type: ignore[attr-defined] # noqa: F401
from .sgd import sgd  # type: ignore[attr-defined] # noqa: F401


# TODO: use foreach API in optim._functional to do all the computation

def adamw(params: List[Tensor],
          grads: List[Tensor],
          exp_avgs: List[Tensor],
          exp_avg_sqs: List[Tensor],
          max_exp_avg_sqs: List[Tensor],
          state_steps: List[Tensor],
          *,
          amsgrad: bool,
          beta1: float,
          beta2: float,
          lr: float,
          weight_decay: float,
          eps: float,
          maximize: bool):
    r"""Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """

    if not all([isinstance(t, torch.Tensor) for t in state_steps]):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]
        # update step
        step_t += 1
        step = step_t.item()

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)


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
    r"""Functional API that performs rmsprop algorithm computation.

    See :class:`~torch.optim.RMSProp` for details.
    """

    for i, param in enumerate(params):
        grad = grads[i]
        square_avg = square_avgs[i]

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

        if centered:
            grad_avg = grad_avgs[i]
            grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
            avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(eps)
        else:
            avg = square_avg.sqrt().add_(eps)

        if momentum > 0:
            buf = momentum_buffer_list[i]
            buf.mul_(momentum).addcdiv_(grad, avg)
            param.add_(buf, alpha=-lr)
        else:
            param.addcdiv_(grad, avg, value=-lr)


def rprop(params: List[Tensor],
          grads: List[Tensor],
          prevs: List[Tensor],
          step_sizes: List[Tensor],
          *,
          step_size_min: float,
          step_size_max: float,
          etaminus: float,
          etaplus: float):
    r"""Functional API that performs rprop algorithm computation.

    See :class:`~torch.optim.Rprop` for details.
    """

    for i, param in enumerate(params):
        grad = grads[i]
        prev = prevs[i]
        step_size = step_sizes[i]

        sign = grad.mul(prev).sign()
        sign[sign.gt(0)] = etaplus
        sign[sign.lt(0)] = etaminus
        sign[sign.eq(0)] = 1

        # update stepsizes with step size updates
        step_size.mul_(sign).clamp_(step_size_min, step_size_max)

        # for dir<0, dfdx=0
        # for dir>=0 dfdx=dfdx
        grad = grad.clone(memory_format=torch.preserve_format)
        grad[sign.eq(etaminus)] = 0

        # update parameters
        param.addcmul_(grad.sign(), step_size, value=-1)

        prev.copy_(grad)


def sparse_adam(params: List[Tensor],
                grads: List[Tensor],
                exp_avgs: List[Tensor],
                exp_avg_sqs: List[Tensor],
                state_steps: List[int],
                *,
                eps: float,
                beta1: float,
                beta2: float,
                lr: float):
    r"""Functional API that performs Sparse Adam algorithm computation.

    See :class:`~torch.optim.SparseAdam` for details.
    """
    for i, param in enumerate(params):
        grad = grads[i]
        grad = grad.coalesce()  # the update is non-linear so indices must be unique
        grad_indices = grad._indices()
        grad_values = grad._values()
        size = grad.size()

        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]


        def make_sparse(values):
            constructor = grad.new
            if grad_indices.dim() == 0 or values.dim() == 0:
                return constructor().resize_as_(grad)
            return constructor(grad_indices, values, size)

        # Decay the first and second moment running average coefficient
        #      old <- b * old + (1 - b) * new
        # <==> old += (1 - b) * (new - old)
        old_exp_avg_values = exp_avg.sparse_mask(grad)._values()
        exp_avg_update_values = grad_values.sub(old_exp_avg_values).mul_(1 - beta1)
        exp_avg.add_(make_sparse(exp_avg_update_values))
        old_exp_avg_sq_values = exp_avg_sq.sparse_mask(grad)._values()
        exp_avg_sq_update_values = grad_values.pow(2).sub_(old_exp_avg_sq_values).mul_(1 - beta2)
        exp_avg_sq.add_(make_sparse(exp_avg_sq_update_values))

        # Dense addition again is intended, avoiding another sparse_mask
        numer = exp_avg_update_values.add_(old_exp_avg_values)
        exp_avg_sq_update_values.add_(old_exp_avg_sq_values)
        denom = exp_avg_sq_update_values.sqrt_().add_(eps)
        del exp_avg_update_values, exp_avg_sq_update_values

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1

        param.add_(make_sparse(-step_size * numer.div_(denom)))
