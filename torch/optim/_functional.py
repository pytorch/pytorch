r"""Functional interface"""
import math
import torch
from torch import Tensor
from typing import List, Optional

from .adadelta import adadelta as adadelta_fn
from .adagrad import adagrad as adagrad_fn
from .adamax import adamax as adamax_fn
from .nadam import nadam as nadam_fn
from .radam import radam as radam_fn

# TODO: use foreach API in optim._functional to do all the computation

def adagrad(params: List[Tensor],
            grads: List[Tensor],
            state_sums: List[Tensor],
            state_steps: List[int],
            has_sparse_grad: bool = False,
            foreach: bool = False,
            *,
            lr: float,
            weight_decay: float,
            lr_decay: float,
            eps: float):
    r"""Functional API that performs Adagrad algorithm computation.

    See :class:`~torch.optim.Adagrad` for details.
    """

    adagrad_fn(params,
               grads,
               state_sums,
               state_steps,
               has_sparse_grad=has_sparse_grad,
               foreach=foreach,
               lr=lr,
               weight_decay=weight_decay,
               lr_decay=lr_decay,
               eps=eps)


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
    r"""Functional API that performs Adam algorithm computation.
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
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)


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
    r"""Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

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


def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        alpha = lr if maximize else -lr
        param.add_(d_p, alpha=alpha)


def adadelta(params: List[Tensor],
             grads: List[Tensor],
             square_avgs: List[Tensor],
             acc_deltas: List[Tensor],
             foreach: bool = False,
             *,
             lr: float,
             rho: float,
             eps: float,
             weight_decay: float):
    r"""Functional API that performs Adadelta algorithm computation.

    See :class:`~torch.optim.Adadelta` for details.
    """

    adadelta_fn(params,
                grads,
                square_avgs,
                acc_deltas,
                foreach=foreach,
                lr=lr,
                rho=rho,
                eps=eps,
                weight_decay=weight_decay)


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


def adamax(params: List[Tensor],
           grads: List[Tensor],
           exp_avgs: List[Tensor],
           exp_infs: List[Tensor],
           state_steps: List[int],
           foreach: bool = False,
           *,
           eps: float,
           beta1: float,
           beta2: float,
           lr: float,
           weight_decay: float):
    r"""Functional API that performs adamax algorithm computation.

    See :class:`~torch.optim.Adamax` for details.
    """

    adamax_fn(params,
              grads,
              exp_avgs,
              exp_infs,
              state_steps,
              foreach=foreach,
              eps=eps,
              beta1=beta1,
              beta2=beta2,
              lr=lr,
              weight_decay=weight_decay)


def asgd(params: List[Tensor],
         grads: List[Tensor],
         axs: List[Tensor],
         mus: List[float],
         etas: List[float],
         *,
         weight_decay: float,
         lambd: float):
    r"""Functional API that performs asgd algorithm computation.

    See :class:`~torch.optim.ASGD` for details.
    """

    for i, param in enumerate(params):
        grad = grads[i]
        mu = mus[i]
        ax = axs[i]
        eta = etas[i]

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # decay term
        param.mul_(1 - lambd * eta)

        # update parameter
        param.add_(grad, alpha=-eta)

        # averaging
        if mu != 1:
            ax.add_(param.sub(ax).mul(mu))
        else:
            ax.copy_(param)


def nadam(params: List[Tensor],
          grads: List[Tensor],
          exp_avgs: List[Tensor],
          exp_avg_sqs: List[Tensor],
          mu_products: List[float],
          state_steps: List[int],
          foreach: bool = False,
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

    nadam_fn(params,
             grads,
             exp_avgs,
             exp_avg_sqs,
             mu_products,
             state_steps,
             foreach=foreach,
             beta1=beta1,
             beta2=beta2,
             lr=lr,
             weight_decay=weight_decay,
             momentum_decay=momentum_decay,
             eps=eps)


def radam(params: List[Tensor],
          grads: List[Tensor],
          exp_avgs: List[Tensor],
          exp_avg_sqs: List[Tensor],
          state_steps: List[int],
          foreach: bool = False,
          *,
          beta1: float,
          beta2: float,
          lr: float,
          weight_decay: float,
          eps: float):
    r"""Functional API that performs RAdam algorithm computation.

    See :class:`~torch.optim.RAdam` for details.
    """

    radam_fn(params,
             grads,
             exp_avgs,
             exp_avg_sqs,
             state_steps,
             foreach=foreach,
             beta1=beta1,
             beta2=beta2,
             lr=lr,
             weight_decay=weight_decay,
             eps=eps)



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
