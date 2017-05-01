import math
from .optimizer import Optimizer


class Nadam(Optimizer):
    """Implements Nadam algorithm.

    It has been proposed in `Incorporating Nesterov Momentum into Adam`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 2e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.975, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        schedule_decay (float, optional): beta1 decay factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. _Incorporating Nesterov Momentum into Adam
        https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
    """

    def __init__(self, params, lr=2e-3, betas=(0.975, 0.999), eps=1e-8,
                 schedule_decay=0, weight_decay=0):
        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], eps=eps,
                        schedule_decay=1 - schedule_decay, weight_decay=weight_decay,
                        prod_beta1=betas[0])
        super(Nadam, self).__init__(params, defaults)

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
            beta1, beta2 = group['beta1'], group['beta2']
            prod_beta1 = group['prod_beta1']
            next_beta1 = beta1 * group['schedule_decay']
            next_prod_beta1 = prod_beta1 * next_beta1
            bias_correction1 = (1 - beta1) / (1 - prod_beta1)
            next_bias_correction1 = next_beta1 / (1 - next_prod_beta1)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                sqrt_bias_correction2 = math.sqrt((1 - beta2 ** state['step']) / beta2)
                step_size = group['lr'] * sqrt_bias_correction2

                denom = exp_avg_sq.add(group['eps'] * sqrt_bias_correction2).sqrt_()

                # For memory efficiency, separate update into two
                p.data.addcdiv_(-step_size * next_bias_correction1, exp_avg, denom)
                p.data.addcdiv_(-step_size * bias_correction1, grad, denom)

            # update beta1
            group['beta1'] = next_beta1
            group['prod_beta1'] = next_prod_beta1

        return loss
