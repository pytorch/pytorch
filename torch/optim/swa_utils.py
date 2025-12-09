# mypy: allow-untyped-defs
r"""Implementation for Stochastic Weight Averaging implementation."""

import itertools
import math
import warnings
from collections.abc import Callable, Iterable
from copy import deepcopy
from typing import Any, cast, Literal, Union
from typing_extensions import override

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim.lr_scheduler import _format_param, LRScheduler
from torch.utils._foreach_utils import _get_foreach_kernels_supported_devices

from .optimizer import Optimizer


__all__ = [
    "AveragedModel",
    "update_bn",
    "SWALR",
    "get_ema_multi_avg_fn",
    "get_swa_multi_avg_fn",
    "get_ema_avg_fn",
    "get_swa_avg_fn",
]

from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype


PARAM_LIST = Union[tuple[Tensor, ...], list[Tensor]]


def get_ema_multi_avg_fn(decay=0.999):
    r"""Get function for exponential moving average across multiple params.

    Returns a function that applies exponential moving average (EMA) to
    update parameters. The EMA update follows the formula:

    .. math::
        \text{ema\_param}_t = \text{decay} \times \text{ema\_param}_{t-1}
        + (1 - \text{decay}) \times \text{current\_param}_t

    where :math:`\text{ema\_param}_t` is the exponentially averaged
    parameter at step t, and :math:`\text{current\_param}_t` is the
    current model parameter at step t.

    Args:
        decay (float, optional): Decay rate for exponential moving average.
            Higher values give more weight to previous parameters. Must be
            in range [0, 1]. Default: 0.999

    Returns:
        Callable: Function that updates EMA parameters in-place. The
            returned function has signature:
            ``ema_update(ema_param_list, current_param_list, num_averaged)``

    Raises:
        ValueError: If decay is not in range [0, 1].

    Example:
        >>> ema_fn = get_ema_multi_avg_fn(decay=0.99)
        >>> # ema_params will be updated as:
        >>> # ema_params = 0.99 * ema_params + 0.01 * current_params
    """

    if decay < 0.0 or decay > 1.0:
        raise ValueError(
            f"Invalid decay value {decay} provided. Please provide a value in [0,1] range."
        )

    @torch.no_grad()
    def ema_update(
        ema_param_list: PARAM_LIST, current_param_list: PARAM_LIST, _
    ) -> None:
        # foreach lerp only handles float and complex
        if torch.is_floating_point(ema_param_list[0]) or torch.is_complex(
            ema_param_list[0]
        ):
            torch._foreach_lerp_(ema_param_list, current_param_list, 1 - decay)
        else:
            for p_ema, p_model in zip(ema_param_list, current_param_list, strict=True):
                p_ema.copy_(p_ema * decay + p_model * (1 - decay))

    return ema_update


def get_swa_multi_avg_fn():
    """Get the function applying stochastic weight average (SWA) across multiple params."""

    @torch.no_grad()
    def swa_update(
        averaged_param_list: PARAM_LIST,
        current_param_list: PARAM_LIST,
        num_averaged: Tensor | int,
    ) -> None:
        # foreach lerp only handles float and complex
        if torch.is_floating_point(averaged_param_list[0]) or torch.is_complex(
            averaged_param_list[0]
        ):
            torch._foreach_lerp_(
                averaged_param_list,
                current_param_list,
                cast(float, 1 / (num_averaged + 1)),
            )
        else:
            diffs = torch._foreach_sub(current_param_list, averaged_param_list)
            if isinstance(num_averaged, Tensor):
                torch._foreach_addcdiv_(
                    averaged_param_list,
                    diffs,
                    [num_averaged + 1] * len(averaged_param_list),
                )
            else:
                torch._foreach_add_(
                    averaged_param_list, diffs, alpha=1.0 / (num_averaged + 1)
                )

    return swa_update


def get_ema_avg_fn(decay=0.999):
    r"""Get function for exponential moving average across a single param.

    Returns a function that applies exponential moving average (EMA) to
    update a single parameter. The EMA update follows the formula:

    .. math::
        \text{ema\_param}_t = \text{decay} \times \text{ema\_param}_{t-1}
        + (1 - \text{decay}) \times \text{current\_param}_t

    where :math:`\text{ema\_param}_t` is the exponentially averaged
    parameter at step t, and :math:`\text{current\_param}_t` is the
    current model parameter at step t.

    Args:
        decay (float, optional): Decay rate for exponential moving average.
            Higher values give more weight to previous parameters. Must be
            in range [0, 1]. Default: 0.999

    Returns:
        Callable: Function that computes and returns the updated EMA
            parameter. The returned function has signature:
            ``ema_update(ema_param, current_param, num_averaged)``

    Raises:
        ValueError: If decay is not in range [0, 1].

    Example:
        >>> ema_fn = get_ema_avg_fn(decay=0.99)
        >>> ema_param = torch.tensor([1.0, 2.0, 3.0])
        >>> current_param = torch.tensor([1.5, 2.5, 3.5])
        >>> # Result: 0.99 * [1.0, 2.0, 3.0] + 0.01 * [1.5, 2.5, 3.5]
        >>> updated = ema_fn(ema_param, current_param, None)
    """

    if decay < 0.0 or decay > 1.0:
        raise ValueError(
            f"Invalid decay value {decay} provided. Please provide a value in [0,1] range."
        )

    @torch.no_grad()
    def ema_update(ema_param: Tensor, current_param: Tensor, num_averaged):
        return decay * ema_param + (1 - decay) * current_param

    return ema_update


def get_swa_avg_fn():
    """Get the function applying stochastic weight average (SWA) across a single param."""

    @torch.no_grad()
    def swa_update(
        averaged_param: Tensor, current_param: Tensor, num_averaged: Tensor | int
    ):
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
            both parameters and the buffers of the model. (default: ``False``)

    Example:
        >>> # xdoctest: +SKIP("undefined variables")
        >>> loader, optimizer, model, loss_fn = ...
        >>> swa_model = torch.optim.swa_utils.AveragedModel(model)
        >>> scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        >>>                                     T_max=300)
        >>> swa_start = 160
        >>> swa_scheduler = SWALR(optimizer, swa_lr=0.05)
        >>>
        >>> for epoch in range(300):
        >>>       for input, target in loader:
        >>>           optimizer.zero_grad()
        >>>           loss_fn(model(input), target).backward()
        >>>           optimizer.step()
        >>>       if epoch > swa_start:
        >>>           swa_model.update_parameters(model)
        >>>           swa_scheduler.step()
        >>>       else:
        >>>           scheduler.step()
        >>>
        >>> # Update bn statistics for the swa_model at the end
        >>> torch.optim.swa_utils.update_bn(loader, swa_model)

    You can also use custom averaging functions with `avg_fn` or `multi_avg_fn` parameters.
    If no averaging function is provided, the default is to compute an equally-weighted
    average of the weights.

    Example:
        >>> # xdoctest: +SKIP("undefined variables")
        >>> # Compute exponential moving averages of the weights
        >>> ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
        >>>         0.9 * averaged_model_parameter + 0.1 * model_parameter
        >>> ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)

    .. note::
        When using SWA/EMA with models containing Batch Normalization you may
        need to update the activation statistics for Batch Normalization.
        You can do so by using :meth:`torch.optim.swa_utils.update_bn` utility.

    .. note::
        :attr:`avg_fn` and :attr:`multi_avg_fn` are not saved in the
        :meth:`state_dict` of the model.

    .. note::
        When :meth:`update_parameters` is called for the first time (i.e.
        :attr:`n_averaged` is `0`) the parameters of `model` are copied
        to the parameters of :class:`AveragedModel`. For every subsequent
        call of :meth:`update_parameters` the function `avg_fn` is used
        to update the parameters.

    .. _Averaging Weights Leads to Wider Optima and Better Generalization:
        https://arxiv.org/abs/1803.05407
    .. _Polyak averaging:
        https://paperswithcode.com/method/polyak-averaging
    .. _There Are Many Consistent Explanations of Unlabeled Data: Why You Should
        Average:
        https://arxiv.org/abs/1806.05594
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    .. _Improving Consistency-Based Semi-Supervised Learning with Weight
        Averaging:
        https://arxiv.org/abs/1806.05594
    """

    def __init__(
        self,
        model: Module,
        device: torch.device | str | None = None,
        avg_fn: Callable[[Tensor, Tensor, Tensor | int], Tensor] | None = None,
        multi_avg_fn: Callable[[PARAM_LIST, PARAM_LIST, Tensor | int], None]
        | None = None,
        use_buffers: bool = False,
    ):
        super().__init__()
        self.module = deepcopy(model)
        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer("n_averaged", torch.tensor(0, dtype=torch.long))
        self.avg_fn = avg_fn
        self.multi_avg_fn = multi_avg_fn
        self.use_buffers = use_buffers

    @override
    def forward(self, *args: Any, **kwargs: Any):
        return self.module(*args, **kwargs)

    def update_parameters(self, model: Module) -> None:
        """Update the parameters of the averaged model.

        Args:
            model (torch.nn.Module): model whose parameters will be used to
                update the averaged model parameters
        """
        self_param = (
            itertools.chain(self.module.parameters(), self.module.buffers())
            if self.use_buffers
            else self.parameters()
        )
        model_param = (
            itertools.chain(model.parameters(), model.buffers())
            if self.use_buffers
            else model.parameters()
        )
        self_param_detached = []
        model_param_detached = []
        for p_averaged, p_model in zip(self_param, model_param, strict=False):
            p_model_ = p_model.detach().to(p_averaged.device)
            self_param_detached.append(p_averaged.detach())
            model_param_detached.append(p_model_)
            if self.n_averaged == 0:
                p_averaged.copy_(p_model_)

        if self.n_averaged > 0:
            if self.multi_avg_fn is not None:
                self.multi_avg_fn(
                    self_param_detached, model_param_detached, self.n_averaged
                )
            elif self.avg_fn is not None:
                for p_averaged, p_model in zip(
                    self_param_detached, model_param_detached, strict=True
                ):
                    p_averaged.copy_(self.avg_fn(p_averaged, p_model, self.n_averaged))
            else:
                for p_averaged, p_model in zip(
                    self_param_detached, model_param_detached, strict=True
                ):
                    p_averaged.copy_(
                        p_averaged * self.n_averaged / (self.n_averaged + 1.0)
                        + p_model / (self.n_averaged + 1.0)
                    )
        self.n_averaged += 1


def update_bn(
    loader: Iterable,
    model: Module,
    device: torch.device | str | None = None,
) -> None:
    r"""Update BatchNorm running statistics.

    Updates BatchNorm running_mean, running_var and num_batches_tracked
    buffers in the model.

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
        >>> # xdoctest: +SKIP("undefined variables")
        >>> loader, swa_model = ...
        >>> torch.optim.swa_utils.update_bn(loader, swa_model)

    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta:
        module.momentum = None
        module.num_batches_tracked *= 0

    for input in loader:
        if isinstance(input, (list, tuple)):
            input = input[0]
        if device is not None:
            input = input.to(device)

        model(input)

    for bn_module in momenta:
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


class SWALR(LRScheduler):
    r"""Anneals the learning rate for Stochastic Weight Averaging (SWA).

    This scheduler is used together with other learning rate schedulers to
    switch to a constant learning rate late in the training as proposed in
    `Averaging Weights Leads to Wider Optima and Better Generalization`_.

    Args:
        optimizer (torch.optim.Optimizer): wrapped optimizer
        swa_lr (float): the constant learning rate to use for SWA
        anneal_epochs (int): number of epochs in the annealing phase
            (default: 10)
        anneal_strategy (str): "cos" or "linear"; specifies the annealing
            strategy: "cos" for cosine annealing, "linear" for linear
            annealing (default: "cos")
        last_epoch (int): the index of the last epoch (default: -1)

    The :class:`SWALR` scheduler can be used together with other
    schedulers to switch to a constant learning rate late in the training
    as in the example below.

    Example:
        >>> # xdoctest: +SKIP("undefined variables")
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

    def __init__(
        self,
        optimizer: Optimizer,
        swa_lr: float,
        anneal_epochs: int = 10,
        anneal_strategy: Literal["cos", "linear"] = "cos",
        last_epoch: int = -1,
    ):
        swa_lrs = self._format_param(optimizer, swa_lr)
        for swa_lr, group in zip(swa_lrs, optimizer.param_groups, strict=True):
            group["swa_lr"] = swa_lr
        if anneal_strategy not in ["cos", "linear"]:
            raise ValueError(
                "anneal_strategy must by one of 'cos' or 'linear', "
                f"instead got {anneal_strategy}"
            )
        if not isinstance(anneal_epochs, int) or anneal_epochs < 0:
            raise ValueError(
                f"anneal_epochs must be equal or greater than 0, got {anneal_epochs}"
            )
        self.anneal_epochs = anneal_epochs
        self.anneal_strategy = anneal_strategy
        super().__init__(optimizer, last_epoch)

    @staticmethod
    def _format_param(optimizer: Optimizer, swa_lrs: float | list) -> list:
        if isinstance(swa_lrs, (list, tuple)):
            if len(swa_lrs) != len(optimizer.param_groups):
                raise ValueError(
                    f"swa_lr must have {len(optimizer.param_groups)} elements, got {len(swa_lrs)}"
                )
            return list(swa_lrs)
        return [swa_lrs] * len(optimizer.param_groups)

    @staticmethod
    def _annealed_lr(
        swa_lr: float, lr: float, pct: float, anneal_strategy: Literal["cos", "linear"]
    ) -> float:
        if anneal_strategy == "cos":
            return swa_lr + (lr - swa_lr) * (1 + math.cos(math.pi * pct)) / 2
        return (1 - pct) * lr + pct * swa_lr

    @override
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
                stacklevel=2,
            )
        step = self._step_count - 1
        prev_lrs = [group["lr"] for group in self.optimizer.param_groups]
        if step < self.anneal_epochs:
            pct = step / self.anneal_epochs
            return [
                self._annealed_lr(
                    group["swa_lr"], lr, pct, self.anneal_strategy
                )
                for group, lr in zip(
                    self.optimizer.param_groups, prev_lrs, strict=True
                )
            ]
        return [group["swa_lr"] for group in self.optimizer.param_groups]
