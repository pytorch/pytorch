import itertools
import math
from copy import deepcopy
import warnings

import torch
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.utils._foreach_utils import _get_foreach_kernels_supported_devices

__all__ = [
    'AveragedModel',
    'update_bn',
    'SWALR',
    'get_ema_multi_avg_fn',
    'get_swa_multi_avg_fn',
    'get_ema_avg_fn',
    'get_swa_avg_fn'
]

from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype


def get_ema_multi_avg_fn(decay=0.999):
    @torch.no_grad()
    def ema_update(ema_param_list, current_param_list, _):
        # foreach lerp only handles float and complex
        if torch.is_floating_point(ema_param_list[0]) or torch.is_complex(ema_param_list[0]):
            torch._foreach_lerp_(ema_param_list, current_param_list, 1 - decay)
        else:
            for p_ema, p_model in zip(ema_param_list, current_param_list):
                p_ema.copy_(p_ema * decay + p_model * (1 - decay))

    return ema_update


def get_swa_multi_avg_fn():
    @torch.no_grad()
    def swa_update(averaged_param_list, current_param_list, num_averaged):
        # foreach lerp only handles float and complex
        if torch.is_floating_point(averaged_param_list[0]) or torch.is_complex(averaged_param_list[0]):
            torch._foreach_lerp_(averaged_param_list, current_param_list, 1 / (num_averaged + 1))
        else:
            diffs = torch._foreach_sub(current_param_list, averaged_param_list)
            torch._foreach_addcdiv_(averaged_param_list, diffs, [num_averaged + 1] * len(averaged_param_list))

    return swa_update


def get_ema_avg_fn(decay=0.999):
    @torch.no_grad()
    def ema_update(ema_param, current_param, num_averaged):
        return decay * ema_param + (1 - decay) * current_param

    return ema_update


def get_swa_avg_fn():
    @torch.no_grad()
    def swa_update(averaged_param, current_param, num_averaged):
        return averaged_param + (current_param - averaged_param) / (num_averaged + 1)

    return swa_update


class AveragedModel(Module):
    r"""Implements averaged model for Stochastic Weight Averaging (SWA) and Exponential Moving Average (EMA).

    Stochastic Weight Averaging was proposed in `Averaging Weights Leads to
    Wider Optima and Better Generalization`_ by Pavel Izmailov, Dmitrii
    Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson
    (UAI 2018).

    Exponential Moving Average is a variation of `Polyak averaging`_,
    but using exponential weights instead of equal weights across iterations.

    AveragedModel class creates a copy of the provided module :attr:`model`
    on the device :attr:`device` and allows to compute running averages of the
    parameters of the :attr:`model`.

    Args:
        model (torch.nn.Module): model to use with SWA/EMA
        device (torch.device, optional): if provided, the averaged model will be
            stored on the :attr:`device`
        avg_fn (function, optional): the averaging function used to update
            parameters; the function must take in the current value of the
            :class:`AveragedModel` parameter, the current value of :attr:`model`
            parameter, and the number of models already averaged; if None,
            an equally weighted average is used (default: None)
        multi_avg_fn (function, optional): the averaging function used to update
            parameters inplace; the function must take in the current values of the
            :class:`AveragedModel` parameters as a list, the current values of :attr:`model`
            parameters as a list, and the number of models already averaged; if None,
            an equally weighted average is used (default: None)
        use_buffers (bool): if ``True``, it will compute running averages for
            both the parameters and the buffers of the model. (default: ``False``)

    Example:
        >>> # xdoctest: +SKIP("undefined variables")
        >>> loader, optimizer, model, loss_fn = ...
        >>> swa_model = torch.optim.swa_utils.AveragedModel(model)
        >>> scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        >>>                                     T_max=300)
        >>> swa_start = 160
        >>> swa_scheduler = SWALR(optimizer, swa_lr=0.05)
        >>> for i in range(300):
        >>>      for input, target in loader:
        >>>          optimizer.zero_grad()
        >>>          loss_fn(model(input), target).backward()
        >>>          optimizer.step()
        >>>      if i > swa_start:
        >>>          swa_model.update_parameters(model)
        >>>          swa_scheduler.step()
        >>>      else:
        >>>          scheduler.step()
        >>>
        >>> # Update bn statistics for the swa_model at the end
        >>> torch.optim.swa_utils.update_bn(loader, swa_model)

    You can also use custom averaging functions with the `avg_fn` or `multi_avg_fn` parameters.
    If no averaging function is provided, the default is to compute
    equally-weighted average of the weights (SWA).

    Example:
        >>> # xdoctest: +SKIP("undefined variables")
        >>> # Compute exponential moving averages of the weights and buffers
        >>> ema_model = torch.optim.swa_utils.AveragedModel(model,
        >>>             torch.optim.swa_utils.get_ema_multi_avg_fn(0.9), use_buffers=True)

    .. note::
        When using SWA/EMA with models containing Batch Normalization you may
        need to update the activation statistics for Batch Normalization.
        This can be done either by using the :meth:`torch.optim.swa_utils.update_bn`
        or by setting :attr:`use_buffers` to `True`. The first approach updates the
        statistics in a post-training step by passing data through the model. The
        second does it during the parameter update phase by averaging all buffers.
        Empirical evidence has shown that updating the statistics in normalization
        layers increases accuracy, but you may wish to empirically test which
        approach yields the best results in your problem.

    .. note::
        :attr:`avg_fn` and `multi_avg_fn` are not saved in the :meth:`state_dict` of the model.

    .. note::
        When :meth:`update_parameters` is called for the first time (i.e.
        :attr:`n_averaged` is `0`) the parameters of `model` are copied
        to the parameters of :class:`AveragedModel`. For every subsequent
        call of :meth:`update_parameters` the function `avg_fn` is used
        to update the parameters.

    .. _Averaging Weights Leads to Wider Optima and Better Generalization:
        https://arxiv.org/abs/1803.05407
    .. _There Are Many Consistent Explanations of Unlabeled Data: Why You Should
        Average:
        https://arxiv.org/abs/1806.05594
    .. _SWALP: Stochastic Weight Averaging in Low-Precision Training:
        https://arxiv.org/abs/1904.11943
    .. _Stochastic Weight Averaging in Parallel: Large-Batch Training That
        Generalizes Well:
        https://arxiv.org/abs/2001.02312
    .. _Polyak averaging:
        https://paperswithcode.com/method/polyak-averaging
    """

    def __init__(self, model, device=None, avg_fn=None, multi_avg_fn=None, use_buffers=False):
        super().__init__()
        assert avg_fn is None or multi_avg_fn is None, 'Only one of avg_fn and multi_avg_fn should be provided'
        self.module = deepcopy(model)
        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer('n_averaged',
                             torch.tensor(0, dtype=torch.long, device=device))
        self.avg_fn = avg_fn
        self.multi_avg_fn = multi_avg_fn
        self.use_buffers = use_buffers

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def update_parameters(self, model):
        self_param = (
            itertools.chain(self.module.parameters(), self.module.buffers())
            if self.use_buffers else self.parameters()
        )
        model_param = (
            itertools.chain(model.parameters(), model.buffers())
            if self.use_buffers else model.parameters()
        )
        self_param_detached = []
        model_param_detached = []
        for p_averaged, p_model in zip(self_param, model_param):
            p_model_ = p_model.detach().to(p_averaged.device)
            self_param_detached.append(p_averaged.detach())
            model_param_detached.append(p_model_)
            if self.n_averaged == 0:
                p_averaged.detach().copy_(p_model_)

        if self.n_averaged > 0:
            if self.multi_avg_fn is not None or self.avg_fn is None:
                grouped_tensors = _group_tensors_by_device_and_dtype([self_param_detached, model_param_detached])
                for ((device, _), ([self_params, model_params], _)) in grouped_tensors.items():
                    if self.multi_avg_fn:
                        self.multi_avg_fn(self_params, model_params, self.n_averaged.to(device))
                    elif device.type in _get_foreach_kernels_supported_devices():
                        multi_avg_fn = get_swa_multi_avg_fn()
                        multi_avg_fn(self_params, model_params, self.n_averaged.to(device))
                    else:
                        avg_fn = get_swa_avg_fn()
                        n_averaged = self.n_averaged.to(device)
                        for p_averaged, p_model in zip(self_params, model_params):
                            p_averaged.copy_(avg_fn(p_averaged, p_model, n_averaged))
            else:
                for p_averaged, p_model in zip(self_param_detached, model_param_detached):
                    n_averaged = self.n_averaged.to(p_averaged.device)
                    p_averaged.detach().copy_(self.avg_fn(p_averaged.detach(), p_model, n_averaged))

        if not self.use_buffers:
            # If not apply running averages to the buffers,
            # keep the buffers in sync with the source model.
            for b_swa, b_model in zip(self.module.buffers(), model.buffers()):
                b_swa.detach().copy_(b_model.detach().to(b_swa.device))
        self.n_averaged += 1


@torch.no_grad()
def update_bn(loader, model, device=None):
    r"""Update BatchNorm running_mean, running_var buffers in the model.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.

    Example:
        >>> # xdoctest: +SKIP("Undefined variables")
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)

    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None

    for input in loader:
        if isinstance(input, (list, tuple)):
            input = input[0]
        if device is not None:
            input = input.to(device)

        model(input)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


class SWALR(LRScheduler):
    r"""Anneals the learning rate in each parameter group to a fixed value.

    This learning rate scheduler is meant to be used with Stochastic Weight
    Averaging (SWA) method (see `torch.optim.swa_utils.AveragedModel`).

    Args:
        optimizer (torch.optim.Optimizer): wrapped optimizer
        swa_lrs (float or list): the learning rate value for all param groups
            together or separately for each group.
        annealing_epochs (int): number of epochs in the annealing phase
            (default: 10)
        annealing_strategy (str): "cos" or "linear"; specifies the annealing
            strategy: "cos" for cosine annealing, "linear" for linear annealing
            (default: "cos")
        last_epoch (int): the index of the last epoch (default: -1)

    The :class:`SWALR` scheduler can be used together with other
    schedulers to switch to a constant learning rate late in the training
    as in the example below.

    Example:
        >>> # xdoctest: +SKIP("Undefined variables")
        >>> loader, optimizer, model = ...
        >>> lr_lambda = lambda epoch: 0.9
        >>> scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer,
        >>>        lr_lambda=lr_lambda)
        >>> swa_scheduler = torch.optim.swa_utils.SWALR(optimizer,
        >>>        anneal_strategy="linear", anneal_epochs=20, swa_lr=0.05)
        >>> swa_start = 160
        >>> for i in range(300):
        >>>      for input, target in loader:
        >>>          optimizer.zero_grad()
        >>>          loss_fn(model(input), target).backward()
        >>>          optimizer.step()
        >>>      if i > swa_start:
        >>>          swa_scheduler.step()
        >>>      else:
        >>>          scheduler.step()

    .. _Averaging Weights Leads to Wider Optima and Better Generalization:
        https://arxiv.org/abs/1803.05407
    """

    def __init__(self, optimizer, swa_lr, anneal_epochs=10, anneal_strategy='cos', last_epoch=-1):
        swa_lrs = self._format_param(optimizer, swa_lr)
        for swa_lr, group in zip(swa_lrs, optimizer.param_groups):
            group['swa_lr'] = swa_lr
        if anneal_strategy not in ['cos', 'linear']:
            raise ValueError("anneal_strategy must by one of 'cos' or 'linear', "
                             f"instead got {anneal_strategy}")
        elif anneal_strategy == 'cos':
            self.anneal_func = self._cosine_anneal
        elif anneal_strategy == 'linear':
            self.anneal_func = self._linear_anneal
        if not isinstance(anneal_epochs, int) or anneal_epochs < 0:
            raise ValueError(f"anneal_epochs must be equal or greater than 0, got {anneal_epochs}")
        self.anneal_epochs = anneal_epochs
        super().__init__(optimizer, last_epoch)

    @staticmethod
    def _format_param(optimizer, swa_lrs):
        if isinstance(swa_lrs, (list, tuple)):
            if len(swa_lrs) != len(optimizer.param_groups):
                raise ValueError("swa_lr must have the same length as "
                                 f"optimizer.param_groups: swa_lr has {len(swa_lrs)}, "
                                 f"optimizer.param_groups has {len(optimizer.param_groups)}")
            return swa_lrs
        else:
            return [swa_lrs] * len(optimizer.param_groups)

    @staticmethod
    def _linear_anneal(t):
        return t

    @staticmethod
    def _cosine_anneal(t):
        return (1 - math.cos(math.pi * t)) / 2

    @staticmethod
    def _get_initial_lr(lr, swa_lr, alpha):
        if alpha == 1:
            return swa_lr
        return (lr - alpha * swa_lr) / (1 - alpha)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        step = self._step_count - 1
        if self.anneal_epochs == 0:
            step = max(1, step)
        prev_t = max(0, min(1, (step - 1) / max(1, self.anneal_epochs)))
        prev_alpha = self.anneal_func(prev_t)
        prev_lrs = [self._get_initial_lr(group['lr'], group['swa_lr'], prev_alpha)
                    for group in self.optimizer.param_groups]
        t = max(0, min(1, step / max(1, self.anneal_epochs)))
        alpha = self.anneal_func(t)
        return [group['swa_lr'] * alpha + lr * (1 - alpha)
                for group, lr in zip(self.optimizer.param_groups, prev_lrs)]
