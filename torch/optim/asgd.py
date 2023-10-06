import torch
from torch import Tensor

from .optimizer import (Optimizer, _use_grad_for_differentiable, _get_value, _default_to_fused_or_foreach,
                        _differentiable_doc, _foreach_doc, _maximize_doc, _capturable_doc)
from torch._utils import is_compiling
from typing import List, Optional

__all__ = ["ASGD", "asgd"]

def _to_tensor(x, device=None):
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, device=device)

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
        capturable: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        if foreach is False and capturable:
            raise ValueError("Capturable not supported with single tensor ASGD")

        defaults = dict(
            lr=lr,
            lambd=lambd,
            alpha=alpha,
            t0=t0,
            weight_decay=weight_decay,
            foreach=foreach,
            maximize=maximize,
            differentiable=differentiable,
            capturable=capturable,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)
            group.setdefault("capturable", False)
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
                    state["step"] = torch.zeros((), device=p.device)
                    state["eta"] = torch.tensor(group["lr"], device=p.device)
                    state["mu"] = torch.ones((), device=p.device)
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
                capturable=group["capturable"],
            )

        return loss


ASGD.__doc__ = fr"""Implements Averaged Stochastic Gradient Descent.

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
        {_foreach_doc}
        {_maximize_doc}
        {_differentiable_doc}
        {_capturable_doc} For ASGD, capturable is only supported when foreach is True.

    .. _Acceleration of stochastic approximation by averaging:
        https://dl.acm.org/citation.cfm?id=131098

    """


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
    capturable: bool = False,
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
        if capturable and not is_compiling():
            raise RuntimeError("Capturable not supported with single tensor ASGD")
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
        capturable=capturable,
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
    capturable: bool,
):
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
    capturable: bool,
):

    if len(params) == 0:
        return

    assert not differentiable, "_foreach ops don't support autograd"

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([params, grads, axs, mus, etas, state_steps])
    for ((device, _), ((grouped_params, grouped_grads, grouped_axs, grouped_mus,
         grouped_etas, grouped_state_steps), _)) in grouped_tensors.items():
        if maximize:
            grouped_grads = torch._foreach_neg(grouped_grads)

        grouped_grads = list(grouped_grads)
        for i in range(len(grouped_params)):
            if torch.is_complex(grouped_params[i]):
                grouped_params[i] = torch.view_as_real(grouped_params[i])
                grouped_grads[i] = torch.view_as_real(grouped_grads[i])
                grouped_axs[i] = torch.view_as_real(grouped_axs[i])

        # update step
        torch._foreach_add_(grouped_state_steps, 1)

        # intermediate = grad + param * lambd
        if weight_decay != 0:
            if maximize:
                torch._foreach_add_(grouped_grads, grouped_params, alpha=weight_decay)
                intermediate = grouped_grads
            else:
                intermediate = torch._foreach_add(grouped_grads, grouped_params, alpha=weight_decay)

            torch._foreach_add_(intermediate, grouped_params, alpha=lambd)
        else:
            intermediate = torch._foreach_add(grouped_grads, grouped_params, alpha=lambd)

        # update param
        # param * (1 - lambd * eta) - eta * grad
        # => param - param * lambd * eta - eta * grad
        # => param - eta * intermediate
        torch._foreach_addcmul_(grouped_params, intermediate, grouped_etas, value=-1)
        del intermediate

        # update grouped_axs
        # averaging: ax = ax + mu * (param - ax)
        # Note (mlazos): We can't use lerp here since it requires weight to be float64
        # and our grouping code requires dtypes to match for all tensors in a group (and it should, since
        # we use the mus in other places)
        # all dtypes need to match, so we could introduce a cast in a loop
        # but since this only adds one additional kernel launch, this looks like the cleaner
        # and faster solution
        intermediate = torch._foreach_sub(grouped_params, grouped_axs)
        torch._foreach_addcmul_(grouped_axs, intermediate, grouped_mus)
        del intermediate

        if capturable:
            # update grouped_mus
            new_mus = torch._foreach_sub(grouped_state_steps, t0)
            torch._foreach_maximum_(new_mus, 1.0)
            torch._foreach_reciprocal_(new_mus)
            torch._foreach_copy_(grouped_mus, new_mus)
            del new_mus

            # update eta = lr / (1 + lambd * lr * step^alpha)
            new_etas = torch._foreach_pow(grouped_state_steps, alpha)
            torch._foreach_mul_(new_etas, lambd)
            torch._foreach_mul_(new_etas, lr)
            torch._foreach_add_(new_etas, 1)
            torch._foreach_reciprocal_(new_etas)
            torch._foreach_mul_(new_etas, lr)
            torch._foreach_copy_(grouped_etas, new_etas)
        else:
            step = grouped_state_steps[0].item()
            new_etas = []
            new_mus = []

            for i in range(len(grouped_mus)):
                new_eta = _to_tensor(
                    lr / (1 + lambd * lr * step ** alpha), device=device
                )
                new_etas.append(new_eta)
                new_mu = _to_tensor(1 / max(1, step - t0), device=device)
                new_mus.append(new_mu)

            torch._foreach_copy_(grouped_etas, new_etas)
            torch._foreach_copy_(grouped_mus, new_mus)
