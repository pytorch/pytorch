from .optimizer import Optimizer, required, _use_grad_for_differentiable, _differentiable_doc, _maximize_doc  # type: ignore[attr-defined]
import torch
from typing import List, Optional
from torch import Tensor

__all__ = ["LARS", "lars"]


class LARS(Optimizer):
    def __init__(
        self,
        params,
        lr=required,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        *,
        trust_coefficient: float = 0.001,
        eps: float = 1e-8,
        maximize: bool = False,
        differentiable: bool = False,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            trust_coefficient=trust_coefficient,
            eps=eps,
            maximize=maximize,
            differentiable=differentiable,
        )

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)

    def _init_group(self, group, params_with_grad, grads, momentum_buffer_list):
        for p in group["params"]:
            if p.grad is None:
                continue
            params_with_grad.append(p)

            grads.append(p.grad)

            state = self.state[p]

            if group["momentum"] > 0:
                momentum_buffer_list.append(state["momentum_buffer"])

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            momentum_buffer_list = []

            lars(
                params_with_grad,
                grads,
                momentum_buffer_list,
                lr=group["lr"],
                momentum=group["momentum"],
                dampening=group["dampening"],
                weight_decay=group["weight_decay"],
                nesterov=group["nesterov"],
                trust_coefficient=group["trust_coefficient"],
                eps=group["eps"],
                maximize=group["maximize"],
            )

            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

        return loss

LARS.__doc__ = r"""Implements LARS algorithm.

    For further details regarding the algorithm we refer to `Large Batch Training of Convolutional Networks`_.
    """ + r"""
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        trust_coefficient (float, optional): coefficient for computing LR (default: 0.001)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        {maximize}
        {differentiable}

    .. _Large Batch Training of Convolutional Networks:
        https://arxiv.org/abs/1708.03888

    """.format(maximize=_maximize_doc, differentiable=_differentiable_doc) + r"""

    Example:
        >>> # xdoctest: +SKIP
        >>> optimizer = torch.optim.LARS(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

def lars(
    params: List[Tensor],
    grads: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    *,
    lr: float,
    momentum: float,
    dampening: float,
    weight_decay: float,
    nesterov: bool,
    trust_coefficient: float,
    eps: float,
    maximize: bool,
):
    r"""Functional API that performs LARS algorithm computation.
    See :class:`~torch.optim.LARS` for details.
    """
    if torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if not torch.jit.is_scripting():
        func = _single_tensor_lars

    func(
        params,
        grads,
        momentum_buffer_list,
        lr=lr,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        nesterov=nesterov,
        trust_coefficient=trust_coefficient,
        eps=eps,
        maximize=maximize,
    )


def _single_tensor_lars(
    params: List[Tensor],
    grads: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    *,
    lr: float,
    momentum: float,
    dampening: float,
    weight_decay: float,
    nesterov: bool,
    trust_coefficient: float,
    eps: float,
    maximize: bool,
):
    for i, param in enumerate(params):
        d_p = grads[i] if not maximize else -grads[i]

        p_norm = torch.norm(param.data)
        g_norm = torch.norm(d_p.data)

        if weight_decay != 0:
            # LARS scaling:
            if p_norm * g_norm > 0:
                lars_lr = trust_coefficient * p_norm / (g_norm + p_norm * weight_decay + eps)

                d_p = d_p.add(param, alpha=weight_decay)
                d_p.mul_(lars_lr)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            d_p = d_p.add(buf, alpha=momentum) if nesterov else buf

        param.add_(d_p, alpha=-lr)
