import torch
from .optimizer import Optimizer


class Adamax(Optimizer):
    """Implements Adamax algorithm (a variant of Adam based on infinity norm).

    It has been proposed in `Adam: A Method for Stochastic Optimization`__.

    Args:
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

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            grads = []
            params_with_grad = []
            states = []
            exp_avgs = []
            exp_infs = []

            beta1, beta2 = group['betas']
            eps = group['eps']

            for p in group['params']:
                if p.grad is not None:
                    if p.grad.is_sparse:
                        raise RuntimeError('Adamax does not support sparse gradients')

                    grads.append(p.grad)
                    params_with_grad.append(p)

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_inf'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_infs.append(state['exp_inf'])

                    state['step'] += 1
                    states.append(state)

            if group['weight_decay'] != 0:
                torch._foreach_add_(grads, params_with_grad, alpha=group['weight_decay'])

            # Update biased first moment estimate.
            torch._foreach_mul_(exp_avgs, beta1)
            torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)

            # Update the exponentially weighted infinity norm.
            torch._foreach_mul_(exp_infs, beta2)

            for exp_inf, grad in zip(exp_infs, grads):
                norm_buf = torch.cat([
                    exp_inf.unsqueeze(0),
                    grad.abs().add_(eps).unsqueeze_(0)
                ], 0)
                torch.max(norm_buf, 0, keepdim=False, out=(exp_inf, exp_inf.new().long()))

            bias_corrections = [1 - beta1 ** state['step'] for state in states]
            clr = [-1 * (group['lr'] / bias_correction) for bias_correction in bias_corrections]
            torch._foreach_addcdiv_(params_with_grad, exp_avgs, exp_infs, clr)

        return loss
