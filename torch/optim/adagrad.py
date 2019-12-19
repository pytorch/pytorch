import torch
from .optimizer import Optimizer


class Adagrad(Optimizer):
    """Implements Adagrad algorithm.

    It has been proposed in `Adaptive Subgradient Methods for Online Learning
    and Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)

    .. _Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization: http://jmlr.org/papers/v12/duchi11a.html
    """

    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError("Invalid initial_accumulator_value value: {}".format(initial_accumulator_value))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr, lr_decay=lr_decay, eps=eps, weight_decay=weight_decay,
                        initial_accumulator_value=initial_accumulator_value)
        super(Adagrad, self).__init__(params, defaults)
        self.reset_state()

    def reset_state(self):
        for group in self.param_groups:
            init_acc = group['initial_accumulator_value']
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['sum'] = torch.full_like(p, init_acc, memory_format=torch.preserve_format)

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()

    def get_update(self, par, grad, lr_decay=0, eps=1e-10, **_):
        state = self.state[par]

        c = 1 + state['step'] * lr_decay
        state['step'] += 1
        state['sum'].addcmul_(1, grad, grad)
        std = state['sum'].sqrt().add_(eps)
        return grad.div_(std) / c

    def get_sparse_update(self, par, lr_decay=0, eps=1e-10, **_):
        grad = par.grad.coalesce()  # the update is non-linear so indices must be unique
        state = self.state[par]

        grad_indices = grad._indices()
        grad_values = grad._values()
        size = grad.size()

        def make_sparse(values):
            constructor = grad.new
            if grad_indices.dim() == 0 or values.dim() == 0:
                return constructor().resize_as_(grad)
            return constructor(grad_indices, values, size)

        c = 1 + state['step'] * lr_decay
        state['step'] += 1
        state['sum'].add_(make_sparse(grad_values.pow(2)))
        std = state['sum'].sparse_mask(grad)
        std_values = std._values().sqrt_().add_(eps)
        return make_sparse(grad_values / std_values) / c
