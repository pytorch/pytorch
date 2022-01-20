import math
import torch
from ..optimizer import Optimizer

class AdamW(Optimizer):
    r"""Implements AdamW algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, *, maximize: bool = False):
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
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize, foreach=True)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)

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
            amsgrad = group['amsgrad']

            grads = []
            state_steps = []
            exp_avg = []
            exp_avg_sq = []
            max_exp_avg_sq = []
            params_with_grad = []

            for p in group['params']:
                if p.grad is not None:
                    if p.grad.is_sparse:
                        raise RuntimeError('AdamW does not support sparse gradients')

                    # Perform stepweight decay
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                    params_with_grad.append(p)
                    grads.append(p.grad)

            if group['maximize']:
                grads = torch._foreach_neg(tuple(grads))

            for p in params_with_grad:
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = torch.tensor(0.)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg.append(state['exp_avg'])
                exp_avg_sq.append(state['exp_avg_sq'])

                if amsgrad:
                    max_exp_avg_sq.append(state['max_exp_avg_sq'])

                state_steps.append(state['step'])


            beta1, beta2 = group['betas']

            # update steps
            torch._foreach_add_(state_steps, 1)

            bias_correction1 = [1 - beta1 ** step.item() for step in state_steps]
            bias_correction2 = [1 - beta2 ** step.item() for step in state_steps]

            #
            # Decay the first and second moment running average coefficient
            #
            torch._foreach_mul_(exp_avg, beta1)
            torch._foreach_add_(exp_avg, grads, alpha=1 - beta1)

            torch._foreach_mul_(exp_avg_sq, beta2)
            torch._foreach_addcmul_(exp_avg_sq, grads, grads, 1 - beta2)

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                max_exp_avg_sq = torch._foreach_maximum(max_exp_avg_sq, exp_avg_sq)

                # Use the max. for normalizing running avg. of gradient
                max_exp_avg_sq_sqrt = torch._foreach_sqrt(max_exp_avg_sq)
                bias_correction_sqrt = [math.sqrt(bc) for bc in bias_correction2]
                torch._foreach_div_(max_exp_avg_sq_sqrt, bias_correction_sqrt)
                denom = torch._foreach_add(max_exp_avg_sq_sqrt, group['eps'])
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(exp_avg_sq)
                bias_correction_sqrt = [math.sqrt(bc) for bc in bias_correction2]
                torch._foreach_div_(exp_avg_sq_sqrt, bias_correction_sqrt)
                denom = torch._foreach_add(exp_avg_sq_sqrt, group['eps'])

            step_size = [-1 * (group['lr'] / bc) for bc in bias_correction1]
            torch._foreach_addcdiv_(params_with_grad, exp_avg, denom, step_size)

        return loss
