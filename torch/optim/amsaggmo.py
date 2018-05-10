
import math
import torch
from .optimizer import Optimizer


class AMSAggMo(Optimizer):
    """Implements AMSgrad algorithm with Aggregated Momentum.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, beta1=[0, 0.9, 0.99], beta2=0.999, eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        for i, beta in enumerate(beta1):
            if not 0.0 <= beta < 1.0:
                raise ValueError("Invalid beta1 parameter at index {}: {}".format(i, beta))
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid beta2 parameter: {}".format(beta2))
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AMSAggMo, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AMSAggMo, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

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
                            
            beta1 = group['beta1']
            beta2 = group['beta2']
            
            max_beta = max(beta1)
            
            #print(avg_mom)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = {}
                    for beta in beta1:
                        state['exp_avg'][beta] =  torch.zeros_like(p.data)
                        
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                    
                exp_avg_sq = state['exp_avg_sq']
                max_exp_avg_sq = state['max_exp_avg_sq']
                
                state['step'] += 1

                exp_avg = torch.zeros_like(p.data)
                
                # Decay the first and second moment running average coefficient
                bias_correction1 = 0
                for beta in beta1:
                    buf = state['exp_avg'][beta]   
                    buf.mul_(beta).add_(1 - beta, grad)
                    exp_avg += buf/len(beta1)
                    bias_correction1 += (1 - beta**state['step'])/len(beta1)

                
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                # Use the max. for normalizing running avg. of gradient
                denom = max_exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                #print('exp_avg')
                #print(exp_avg)
                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
