# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
from typing import cast, List, Optional, Tuple, Union

import torch
from torch import Tensor

from .optimizer import (
    _capturable_doc,
    _default_to_fused_or_foreach,
    _differentiable_doc,
    _disable_dynamo_if_unsupported,
    _foreach_doc,
    _get_capturable_supported_devices,
    _get_scalar_dtype,
    _get_value,
    _maximize_doc,
    _params_doc,
    _use_grad_for_differentiable,
    _view_as_real,
    Optimizer,
    ParamsT,
)


__all__ = ["ASGD", "asgd"]


class ASGD(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-2,
        lambd: float = 1e-4,
        alpha: float = 0.75,
        t0: float = 1e6,
        weight_decay: float = 0,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        differentiable: bool = False,
        capturable: bool = False,
    ):
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

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
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0:
                    if not torch.is_tensor(p_state["step"]):
                        step_val = float(p_state["step"])
                        p_state["step"] = torch.tensor(
                            step_val, dtype=_get_scalar_dtype(), device=p.device
                        )
                    if not torch.is_tensor(p_state["eta"]):
                        p_state["eta"] = torch.tensor(
                            p_state["eta"], dtype=_get_scalar_dtype(), device=p.device
                        )
                    if not torch.is_tensor(p_state["mu"]):
                        p_state["mu"] = torch.tensor(
                            p_state["mu"], dtype=_get_scalar_dtype(), device=p.device
                        )

    def _init_group(self, group, params_with_grad, grads, mus, axs, etas, state_steps):
        has_complex = False
        for p in group["params"]:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError("ASGD does not support sparse gradients")
                grads.append(p.grad)

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = torch.zeros(
                        (), device=p.device, dtype=_get_scalar_dtype()
                    )
                    state["eta"] = (
                        torch.as_tensor(
                            group["lr"], device=p.device, dtype=_get_scalar_dtype()
                        )
                        .clone()
                        .detach()
                    )
                    state["mu"] = torch.ones(
                        (), device=p.device, dtype=_get_scalar_dtype()
                    )
                    state["ax"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                mus.append(state["mu"])
                axs.append(state["ax"])
                etas.append(state["eta"])
                state_steps.append(state["step"])
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            mus: List[Tensor] = []
            axs: List[Tensor] = []
            etas: List[Tensor] = []
            state_steps: List[Tensor] = []

            has_complex = self._init_group(
                group, params_with_grad, grads, mus, axs, etas, state_steps
            )

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
                has_complex=has_complex,
            )

        return loss


ASGD.__doc__ = rf"""Implements Averaged Stochastic Gradient Descent.

    It has been proposed in `Acceleration of stochastic approximation by
    averaging`_.

    Args:
        {_params_doc}
        lr (float, Tensor, optional): learning rate (default: 1e-2)
        lambd (float, optional): decay term (default: 1e-4)
        alpha (float, optional): power for eta update (default: 0.75)
        t0 (float, optional): point at which to start averaging (default: 1e6)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        {_foreach_doc}
        {_maximize_doc}
        {_differentiable_doc}
        {_capturable_doc}

    .. _Acceleration of stochastic approximation by averaging:
        https://dl.acm.org/citation.cfm?id=131098

    """


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
    has_complex: bool,
):
    for i, param in enumerate(params):
        grad = grads[i]
        grad = grad if not maximize else -grad
        mu = mus[i]
        ax = axs[i]
        eta = etas[i]
        step_t = state_steps[i]

        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if not torch.compiler.is_compiling() and capturable:
            capturable_supported_devices = _get_capturable_supported_devices()
            assert (
                param.device.type
                == mu.device.type
                == eta.device.type
                == step_t.device.type
                and param.device.type in capturable_supported_devices
            ), (
                f"If capturable=True, params, mus, etas, and state_steps must be "
                f"on supported devices: {capturable_supported_devices}."
            )

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            param = torch.view_as_real(param)
            ax = torch.view_as_real(ax)

        # update step
        step_t += 1

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        if capturable:
            param.mul_(1 - lambd * eta)
            param.addcmul_(grad, eta, value=-1)  # update parameter
        else:
            eta_value = _get_value(eta)
            param.mul_(1 - lambd * eta_value)  # decay term
            param.add_(grad, alpha=-eta_value)  # update parameter

        # averaging
        if capturable or mu.item() != 1:
            ax.add_(param.sub(ax).mul_(mu))
        else:
            ax.copy_(param)

        if capturable:
            eta.copy_(lr / ((1 + lambd * lr * step_t) ** alpha))
            mu.copy_(1 / torch.maximum(step_t - t0, torch.ones_like(step_t)))
        else:
            step = _get_value(step_t)
            new_eta = torch.as_tensor(lr / ((1 + lambd * lr * step) ** alpha))
            eta.copy_(new_eta)
            new_mu = torch.as_tensor(1 / max(1, step - t0))
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
    has_complex: bool,
):
    if len(params) == 0:
        return

    assert not differentiable, "_foreach ops don't support autograd"

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if not torch.compiler.is_compiling() and capturable:
        capturable_supported_devices = _get_capturable_supported_devices(
            supports_xla=False
        )
        assert all(
            p.device.type == mu.device.type == eta.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, mu, eta, step in zip(params, mus, etas, state_steps)
        ), f"If capturable=True, params, mus, etas, and state_steps must be on supported devices: {capturable_supported_devices}."

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, axs, mus, etas, state_steps]  # type: ignore[list-item]
    )
    for (device, _), (
        (
            grouped_params_,
            grouped_grads_,
            grouped_axs_,
            grouped_mus_,
            grouped_etas_,
            grouped_state_steps_,
        ),
        _,
    ) in grouped_tensors.items():
        grouped_params = cast(List[Tensor], grouped_params_)
        grouped_grads = cast(List[Tensor], grouped_grads_)
        grouped_axs = cast(List[Tensor], grouped_axs_)
        grouped_mus = cast(List[Tensor], grouped_mus_)
        grouped_etas = cast(List[Tensor], grouped_etas_)
        grouped_state_steps = cast(List[Tensor], grouped_state_steps_)

        if has_complex:
            _view_as_real(grouped_params, grouped_grads, grouped_axs)

        if maximize:
            grouped_grads = torch._foreach_neg(grouped_grads)  # type: ignore[assignment]

        # Update steps
        # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
        # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
        # wrapped it once now. The alpha is required to assure we go to the right overload.
        if not torch.compiler.is_compiling() and grouped_state_steps[0].is_cpu:
            torch._foreach_add_(
                grouped_state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0
            )
        else:
            torch._foreach_add_(grouped_state_steps, 1)

        # intermediate = grad + param * lambd
        intermediate: Union[Tuple[Tensor, ...], List[Tensor]]
        if weight_decay != 0:
            if maximize:
                torch._foreach_add_(grouped_grads, grouped_params, alpha=weight_decay)
                intermediate = grouped_grads
            else:
                intermediate = torch._foreach_add(
                    grouped_grads, grouped_params, alpha=weight_decay
                )

            torch._foreach_add_(intermediate, grouped_params, alpha=lambd)
        else:
            intermediate = torch._foreach_add(
                grouped_grads, grouped_params, alpha=lambd
            )

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

        new_etas: Union[Tuple[Tensor, ...], List[Tensor]]
        new_mus: Union[Tuple[Tensor, ...], List[Tensor]]
        if capturable:
            # update grouped_mus
            new_mus = torch._foreach_sub(grouped_state_steps, t0)
            torch._foreach_maximum_(new_mus, 1.0)
            torch._foreach_reciprocal_(new_mus)
            torch._foreach_copy_(grouped_mus, new_mus)
            del new_mus

            # update eta = lr / ((1 + lambd * lr * step)^alpha)
            new_etas = torch._foreach_mul(grouped_state_steps, lambd)
            torch._foreach_mul_(new_etas, lr)
            torch._foreach_add_(new_etas, 1)
            torch._foreach_pow_(new_etas, alpha)
            torch._foreach_reciprocal_(new_etas)
            torch._foreach_mul_(new_etas, lr)
            torch._foreach_copy_(grouped_etas, new_etas)
        else:
            new_etas = [
                torch.as_tensor(lr / ((1 + lambd * lr * step) ** alpha), device=device)
                for step in grouped_state_steps
            ]
            new_mus = [
                torch.as_tensor(1 / max(1, _get_value(step) - t0), device=device)
                for step in grouped_state_steps
            ]
            torch._foreach_copy_(grouped_etas, new_etas)
            torch._foreach_copy_(grouped_mus, new_mus)


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_asgd)
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
    has_complex: bool = False,
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
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )

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
        capturable=capturable,
        has_complex=has_complex,
    )
