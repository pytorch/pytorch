from torch import Tensor
from typing import List


def adagrad(params: List[Tensor],
            grads: List[Tensor],
            state_sums: List[Tensor],
            step: int,
            lr: float,
            weight_decay: float,
            lr_decay: float,
            eps: float): 
    # TODO: use foreach API to do the computation
    for (param, grad, state_sum) in zip(params, grads, state_sums):
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

            def make_sparse(values):
                constructor = grad.new
                if grad_indices.dim() == 0 or values.dim() == 0:
                    return constructor().resize_as_(grad)
                return constructor(grad_indices, values, size)
            state_sum.add_(make_sparse(grad_values.pow(2)))
            std = state_sum.sparse_mask(grad)
            std_values = std._values().sqrt_().add_(eps)
            param.add_(make_sparse(grad_values / std_values), alpha=-clr)
        else:
            state_sum.addcmul_(grad, grad, value=1)
            std = state_sum.sqrt().add_(eps)
            param.addcdiv_(grad, std, value=-clr)
