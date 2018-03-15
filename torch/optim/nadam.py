'''
Created on Mar 14, 2018

@author: jyzhang
'''

import math
import torch
from .optimizer import Optimizer


class NAdam(torch.optim.Optimizer):
    """Implements Nesterov-accelerated Adam algorithm according to Keras.
    
    parameter name alias in different algorithms
    NAdam                           Keras                         054_report
    exp_avg                         m_t                            m_t
    exp_avg_prime              \prime{m}_t              \prime{m}_t
    exp_avg_bar                  \bar{m}_t                  \bar{m}_t
    exp_avg_sq                    v_t                             n_t
    exp_avg_sq_prime         \prime{v}_t               \prime{n}_t
    beta1                              beta_1                       \mu
    beta2                              beta_2                       v=0.999                            
    
    It has been proposed in `Incorporating Nesterov Momentum into Adam`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0), 
            but not used in NAdam
        schedule_decay (float, optional): coefficients used for computing
            moment schedule (default: 0.004)
    .. _Incorporating Nesterov Momentum into Adam
        http://cs229.stanford.edu/proj2015/054_report.pdf
    .. _On the importance of initialization and momentum in deep learning
        http://www.cs.toronto.edu/~fritz/absps/momentum.pdf
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, schedule_decay=0.004):
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, schedule_decay=schedule_decay)
        super(NAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NAdam, self).__setstate__(state)

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
                if grad.is_sparse:
                    raise RuntimeError('NAdam does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # \mu^{t} 
                    state['m_schedule'] = 1.

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']
                
                schedule_decay = group['schedule_decay']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # calculate the momentum cache \mu^{t} and \mu^{t+1}
                momentum_cache_t = beta1 * ( \
                    1. - 0.5 * (pow(0.96, state['step'] * schedule_decay)))
                momentum_cache_t_1 = beta1 * ( \
                    1. - 0.5 * (pow(0.96, (state['step'] + 1) * schedule_decay)))
                m_schedule_new = state['m_schedule'] * momentum_cache_t
                m_schedule_next = state['m_schedule'] * momentum_cache_t * momentum_cache_t_1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                        
                g_prime = torch.div( grad, 1. - m_schedule_new)
                exp_avg_prime = torch.div( exp_avg,  1. - m_schedule_next )
                exp_avg_sq_prime = torch.div(exp_avg_sq,  1. - pow(beta2, state['step']))
                
                exp_avg_bar = torch.add( (1. - momentum_cache_t) * g_prime, \
                                         momentum_cache_t_1,  exp_avg_prime )

                denom = exp_avg_sq_prime.sqrt().add_(group['eps'])

                step_size = group['lr'] 

                p.data.addcdiv_(-step_size, exp_avg_bar, denom)
                                      
        return loss