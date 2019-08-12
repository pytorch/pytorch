import math
import os
from distutils.util import strtobool
import torch
from torch.optim.optimizer import Optimizer
from torch.hub import _check_module_exists

NUMBA_CUDA_EXIST = False
NUMBA_CUDA_THREAD_PER_BLOCK = 512
if not strtobool(os.environ.get('NO_NUMBA', 'n')) and _check_module_exists("numba.cuda"):
    import numba.cuda
    NUMBA_CUDA_EXIST = numba.cuda.is_available()

    @numba.cuda.jit()
    def numba_cuda_kernel(param, grad, exp_avg, exp_avg_sq, beta1,
                          beta2, step_size, bias_correction2, eps,
                          weight_decay):
        i = numba.cuda.grid(1)
        if i >= param.size:
            return

        if weight_decay != 0:
            grad[i] += weight_decay * param[i]

        exp_avg[i] = exp_avg[i] * beta1 + (1 - beta1) * grad[i]
        exp_avg_sq[i] = exp_avg_sq[i] * beta2 + (1 - beta2) * grad[i] * grad[i]

        denom = math.sqrt(exp_avg_sq[i]) / bias_correction2 + eps
        param[i] = param[i] + (-step_size) * (exp_avg[i] / denom)


class Adam(Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

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
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        # In order to reduce Numba overhead, we save the device arrays
        # between calls to `step()` in `_nbstate`.
        self._nbstate = getattr(self, '_nbstate', {})

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue

                # Perform optimization step
                grad = param.grad.data
                p = param.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients,'
                                       'please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[param]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p)
                    elif NUMBA_CUDA_EXIST and numba.cuda.is_cuda_array(p.data):
                        self._nbstate[param] = {
                            'param': numba.cuda.as_cuda_array(p.data.flatten()),
                            'grad': numba.cuda.as_cuda_array(grad.flatten()),
                            'exp_avg': numba.cuda.as_cuda_array(state['exp_avg'].data.flatten()),
                            'exp_avg_sq': numba.cuda.as_cuda_array(state['exp_avg_sq'].
                                                                   data.flatten()),
                            'blockspergrid': math.ceil(p.data.numel() / NUMBA_CUDA_THREAD_PER_BLOCK)
                        }

                weight_decay = group['weight_decay']
                eps = group['eps']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = math.sqrt(1 - beta2 ** state['step'])
                step_size = group['lr'] / bias_correction1

                if param in self._nbstate:
                    s = self._nbstate[param]
                    numba_cuda_kernel[s['blockspergrid'],
                                      NUMBA_CUDA_THREAD_PER_BLOCK](s['param'],
                                                                   s['grad'],
                                                                   s['exp_avg'],
                                                                   s['exp_avg_sq'],
                                                                   beta1, beta2,
                                                                   step_size,
                                                                   bias_correction2,
                                                                   eps, weight_decay)
                else:
                    if weight_decay != 0:
                        grad.add_(weight_decay, p.data)
                    exp_avg = state['exp_avg'].data
                    exp_avg_sq = state['exp_avg_sq'].data
                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(1 - beta1, grad)
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                    if amsgrad:
                        max_exp_avg_sq = state['max_exp_avg_sq']
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = (max_exp_avg_sq.sqrt() / bias_correction2).add_(eps)
                    else:
                        denom = (exp_avg_sq.sqrt() / bias_correction2).add_(eps)
                    p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
