import math
import torch
from .optimizer import Optimizer


class SparseAdam(Optimizer):
    """Implements lazy version of Adam algorithm suitable for sparse tensors.

    In this variant, only moments that show up in the gradient get updated, and
    only those portions of the gradient get applied to the parameters.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(SparseAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if not grad.is_sparse:
                    raise RuntimeError('SparseAdam does not support dense gradients, please consider Adam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['step'] += 1

                grad = grad.coalesce()  # the update is non-linear so indices must be unique
                grad_indices = grad._indices()
                grad_values = grad._values()
                size = grad.size()

                def make_sparse(values):
                    constructor = grad.new
                    if grad_indices.dim() == 0 or values.dim() == 0:
                        return constructor().resize_as_(grad)
                    return constructor(grad_indices, values, size)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # Decay the first and second moment running average coefficient
                #      old <- b * old + (1 - b) * new
                # <==> old += (1 - b) * (new - old)
                old_exp_avg_values = exp_avg._sparse_mask(grad)._values()
                exp_avg_update_values = grad_values.sub(old_exp_avg_values).mul_(1 - beta1)
                exp_avg.add_(make_sparse(exp_avg_update_values))
                old_exp_avg_sq_values = exp_avg_sq._sparse_mask(grad)._values()
                exp_avg_sq_update_values = grad_values.pow(2).sub_(old_exp_avg_sq_values).mul_(1 - beta2)
                exp_avg_sq.add_(make_sparse(exp_avg_sq_update_values))

                # Dense addition again is intended, avoiding another _sparse_mask
                numer = exp_avg_update_values.add_(old_exp_avg_values)
                exp_avg_sq_update_values.add_(old_exp_avg_sq_values)
                denom = exp_avg_sq_update_values.sqrt_().add_(group['eps'])
                del exp_avg_update_values, exp_avg_sq_update_values

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.add_(make_sparse(-step_size * numer.div_(denom)))

        return loss
