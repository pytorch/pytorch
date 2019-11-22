import torch
from .optimizer import Optimizer


class Adamax(Optimizer):
    """Implements Adamax algorithm (a variant of Adam based on infinity norm).

    It has been proposed in `Adam: A Method for Stochastic Optimization`__.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 2e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    __ https://arxiv.org/abs/1412.6980
    """

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Adamax, self).__init__(params, defaults)

    def reset_state(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['exp_inf'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    def get_update(self, p, betas=(.9, .999), eps=1e-8, weight_decay=0, **_):
        grad = p.grad
        state = self.state[p]

        if weight_decay > 0:
            grad = grad.add(weight_decay, p)

        beta1, beta2 = betas
        state['step'] += 1
        bias_corr = 1 - beta1 ** state['step']

        exp_avg, exp_inf = state['exp_avg'], state['exp_inf']
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_inf = torch.max(exp_inf.mul_(beta2), grad.abs(), out=exp_inf)

        mean = exp_avg.div(bias_corr)
        return mean.div_(exp_inf.add(eps))
