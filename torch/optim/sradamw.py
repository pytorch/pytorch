import torch
from torch.cuda.amp.grad_scaler import _MultiDeviceReplicator
from .adamw import AdamW
from ._amp_helper import _combined_found_inf_helper


def _apply_square_to_state_dict(state_dict):
    with torch.no_grad():
        for state_per_param in state_dict['state'].values():
            state_per_param['exp_avg_sq'].square_()
            state_per_param['max_exp_avg_sq'].square()
    return state_dict


def _apply_sqrt_to_state_dict(state_dict):
    with torch.no_grad():
        for state_per_param in state_dict['state'].values():
            state_per_param['exp_avg_sq'].sqrt_()
            if 'max_exp_avg_sq' not in state_per_param:
                state_per_param['max_exp_avg_sq'] = torch.zeros_like(state_per_param['exp_avg_sq'])
            else:
                state_per_param['max_exp_avg_sq'].sqrt_()
    return state_dict


class SRAdamW(AdamW):
    r"""Implements AdamW algorithm with Stochastic Rounding.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    With Stochastic Rounding, param, `exp_avg`, `exp_avg_sq`, and optionally `max_exp_avg_sq`
    can be represented with 16 bits. See :func:`torch.stochastic_rounding` for details.
    This optimizer requires CUDA.

    Arguments:
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

    _step_supports_amp_scaling = True

    def state_dict(self):
        return _apply_square_to_state_dict(super().state_dict())

    def load_state_dict(self, state_dict):
        super().load_state_dict(_apply_sqrt_to_state_dict(state_dict))

    @torch.no_grad()
    def step(self, closure=None, grad_scaler=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if grad_scaler is not None:
            inv_scale = grad_scaler._get_scale_async().double().reciprocal().float()
            found_inf = _combined_found_inf_helper(self, grad_scaler, inv_scale.device)
        else:
            inv_scale = torch.ones((1,), dtype=torch.float, device=torch.cuda.current_device())
            found_inf = _MultiDeviceReplicator(
                torch.zeros((1,), dtype=torch.float, device=torch.cuda.current_device()))

        inv_scale = _MultiDeviceReplicator(inv_scale)

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError('SRAdamW does not support sparse gradients')

                state = self.state[param]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                beta1, beta2 = group['betas']

                state['step'] += 1

                torch.stochastic_rounding_adam_step(
                    param, grad,
                    state['exp_avg'], state['exp_avg_sq'], state['max_exp_avg_sq'],
                    inv_scale.get(param.device), found_inf.get(param.device),
                    group['lr'], beta1, beta2,
                    group['weight_decay'], group['eps'], state['step'],
                    True, group['amsgrad'])

        return loss
