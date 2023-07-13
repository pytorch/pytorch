import torch
from torch import Tensor

from .optimizer import (Optimizer, _use_grad_for_differentiable, _get_value, _default_to_fused_or_foreach,
                        _differentiable_doc, _foreach_doc, _maximize_doc)
from torch._utils import is_compiling
from typing import List, Optional

__all__ = ["ASGD", "asgd"]

def _to_tensor(x):
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x)

    return x

class ASGD(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-2,
        lambd=1e-4,
        alpha=0.75,
        t0=1e6,
        weight_decay=0,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        differentiable: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            lambd=lambd,
            alpha=alpha,
            t0=t0,
            weight_decay=weight_decay,
            foreach=foreach,
            maximize=maximize,
            differentiable=differentiable,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))
        eta_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["eta"]
        )
        if not eta_is_tensor:
            for s in state_values:
                s["eta"] = torch.tensor(s["eta"])
        mu_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["mu"]
        )
        if not mu_is_tensor:
            for s in state_values:
                s["mu"] = torch.tensor(float(s["mu"]))

    def _init_group(self, group, params_with_grad, grads, mus, axs, etas, state_steps):
        for p in group["params"]:
            if p.grad is not None:
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError("ASGD does not support sparse gradients")
                grads.append(p.grad)

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    state["eta"] = torch.tensor(group["lr"])
                    state["mu"] = torch.tensor(1.0)
                    state["ax"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                mus.append(state["mu"])
                axs.append(state["ax"])
                etas.append(state["eta"])
                state_steps.append(state["step"])

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
            mus = []
            axs = []
            etas = []
            state_steps = []

            self._init_group(group, params_with_grad, grads, mus, axs, etas, state_steps)

            asgd(
                params_with_grad,
                grads,
                axs,
                mus,
                etas,
                state_steps,
                lambd=group["lambd"],
                lr=group["lr"],
                t0=group["t0"],
                alpha=group["alpha"],
                weight_decay=group["weight_decay"],
                foreach=group["foreach"],
                maximize=group["maximize"],
                differentiable=group["differentiable"],
            )

        return loss


ASGD.__doc__ = r"""Implements Averaged Stochastic Gradient Descent.

    It has been proposed in `Acceleration of stochastic approximation by
    averaging`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lambd (float, optional): decay term (default: 1e-4)
        alpha (float, optional): power for eta update (default: 0.75)
        t0 (float, optional): point at which to start averaging (default: 1e6)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        {foreach}
        {maximize}
        {differentiable}

    .. _Acceleration of stochastic approximation by averaging:
        https://dl.acm.org/citation.cfm?id=131098

    """.format(foreach=_foreach_doc, maximize=_maximize_doc, differentiable=_differentiable_doc)


def asgd(
    params: List[Tensor],
    grads: List[Tensor],
    axs: List[Tensor],
    mus: List[Tensor],
    etas: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: Optional[bool] = None,
    maximize: bool = False,
    differentiable: bool = False,
    *,
    lambd: float,
    lr: float,
    t0: float,
    alpha: float,
    weight_decay: float,
):
    r"""Functional API that performs asgd algorithm computation.

    See :class:`~torch.optim.ASGD` for details.
    """

    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_asgd
    else:
        func = _single_tensor_asgd

    func(
        params,
        grads,
        axs,
        mus,
        etas,
        state_steps,
        lambd=lambd,
        lr=lr,
        t0=t0,
        alpha=alpha,
        weight_decay=weight_decay,
        maximize=maximize,
        differentiable=differentiable,
    )


def _single_tensor_asgd(
    params: List[Tensor],
    grads: List[Tensor],
    axs: List[Tensor],
    mus: List[Tensor],
    etas: List[Tensor],
    state_steps: List[Tensor],
    *,
    lambd: float,
    lr: float,
    t0: float,
    alpha: float,
    weight_decay: float,
    maximize: bool,
    differentiable: bool,
):
    def _to_tensor(x):
        if not isinstance(x, torch.Tensor):
            return torch.tensor(x)
        return x

    for i, param in enumerate(params):
        grad = grads[i]
        grad = grad if not maximize else -grad
        mu = mus[i]
        ax = axs[i]
        eta = etas[i]
        step_t = state_steps[i]

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            param = torch.view_as_real(param)
            ax = torch.view_as_real(ax)

        # update step
        step_t += 1
        step = _get_value(step_t)

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        eta_value = _get_value(eta)
        # decay term
        param.mul_(1 - lambd * eta_value)

        # update parameter
        param.add_(grad, alpha=-eta_value)

        # averaging
        if is_compiling() or mu.item() != 1:
            ax.add_(param.sub(ax).mul(mu))
        else:
            ax.copy_(param)

        new_eta = _to_tensor(lr / ((1 + lambd * lr * step) ** alpha))
        eta.copy_(new_eta)
        new_mu = _to_tensor(1 / max(1, step - t0))
        mu.copy_(new_mu)


def _multi_tensor_asgd(
    params: List[Tensor],
    grads: List[Tensor],
    axs: List[Tensor],
    mus: List[Tensor],
    etas: List[Tensor],
    state_steps: List[Tensor],
    *,
    lambd: float,
    lr: float,
    t0: float,
    alpha: float,
    weight_decay: float,
    maximize: bool,
    differentiable: bool,
):

    if len(params) == 0:
        return

    assert not differentiable, "_foreach ops don't support autograd"

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([params, grads, axs, mus, etas, state_steps])
    for ((grouped_params, grouped_grads, grouped_axs, grouped_mus,
         grouped_etas, grouped_state_steps), _) in grouped_tensors.values():
        if maximize:
            grouped_grads = torch._foreach_neg(grouped_grads)

        def _view_complex_as_real(tensor_list):
            return [
                torch.view_as_real(t) if torch.is_complex(t) else t for t in tensor_list
            ]

        grouped_grads = _view_complex_as_real(grouped_grads)
        grouped_params = _view_complex_as_real(grouped_params)
        grouped_axs = _view_complex_as_real(grouped_axs)

        # update step
        torch._foreach_add_(grouped_state_steps, 1)

        if weight_decay != 0:
            grouped_grads = torch._foreach_add(grouped_grads, grouped_params, alpha=weight_decay)

        # decay term
        eta = _get_value(grouped_etas[0])
        torch._foreach_mul_(grouped_params, 1 - lambd * eta)

        # update parameter
        torch._foreach_add_(grouped_params, grouped_grads, alpha=-eta)

        # averaging
        for i in range(len(grouped_axs)):
            if is_compiling() or grouped_mus[i].item() != 1:
                grouped_axs[i].add_(grouped_params[i].sub(grouped_axs[i]).mul(grouped_mus[i]))
            else:
                grouped_axs[i].copy_(grouped_params[i])

        # update eta and mu
        for i in range(len(grouped_mus)):
            new_eta = _to_tensor(
                lr / (1 + lambd * lr * _get_value(grouped_state_steps[i]) ** alpha)
            )
            grouped_etas[i].copy_(new_eta)
            new_mu = _to_tensor(1 / max(1, _get_value(grouped_state_steps[i]) - t0))
            grouped_mus[i].copy_(new_mu)
