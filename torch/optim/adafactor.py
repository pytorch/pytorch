# mypy: allow-untyped-defs
from typing import cast, List, Optional, Tuple, Union

import torch
from torch import Tensor
from .optimizer import (
    _disable_dynamo_if_unsupported,
    _get_scalar_dtype,
    _use_grad_for_differentiable,
    Optimizer,
    ParamsT,
)

__all__ = ["Adafactor", "adafactor"]


class Adafactor(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0, 0.999),
        eps: Tuple[float, float] = (1e-30, 1e-3),
        d: float = 1,
        weight_decay: float = 1e-2,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if not 0.0 <= eps[0]:
            raise ValueError(f"epsilon1 should be >= 0 but is: {eps[0]}")
        if not 0.0 <= eps[1]:
            raise ValueError(f"epsilon2 should be >= 0 but is: {eps[1]}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"beta1 should be between 0 and 1 but is: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"beta2 should be between 0 and 1 but is: {betas[1]}")
        if not 1.0 <= d:
            raise ValueError(f"Clipping threshold d should be >= 1 but is: {d}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"weight_decay should be >= 0 but is: {weight_decay}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            d=d,
            weight_decay=weight_decay,
            differentiable=False,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("differentiable", False)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = torch.tensor(step_val, dtype=_get_scalar_dtype())

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        row_vars,
        col_vars,
        state_steps,
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is None:
                continue
            has_complex |= torch.is_complex(p)
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("Adafactor does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.
                # This is because kernel launches are costly on CUDA and XLA.
                state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                # Row factor of variance, NOT the same shape as grads (will be reduced along last dim)
                state["row_var"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                ).sum(dim=-1)
                # Col factor of variance, NOT the same shape as grads (will be reduced along penultimate dim)
                state["col_var"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                ).sum(dim=-2)

            exp_avgs.append(state["exp_avg"])
            row_vars.append(state["row_var"])
            col_vars.append(state["col_var"])
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
            exp_avgs: List[Tensor] = []
            row_vars: List[Tensor] = []
            col_vars: List[Tensor] = []
            state_steps: List[Tensor] = []
            beta1, beta2 = cast(Tuple[float, float], group["betas"])
            eps1, eps2 = cast(Tuple[float, float], group["eps"])

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                row_vars,
                col_vars,
                state_steps,
            )

            adafactor(
                params_with_grad,
                grads,
                exp_avgs,
                row_vars,
                col_vars,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                d=group["d"],
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps1=eps1,
                eps2=eps2,
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                has_complex=has_complex,
                maximize=False,
            )

        return loss


Adafactor.__doc__ = (
    r"""Implements Adafactor algorithm.

    The docs are currently wrong. Will fill in later.

    For further details regarding the algorithm we refer to `Adafactor: Adaptive Learning Rates with Sublinear Memory Cost`_.
    """
    + r"""
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, Tensor, optional): learning rate (default: 1e-3). A tensor LR
            is not yet supported for all our implementations. Please use a float
            LR if you are not also specifying fused=True or capturable=True.
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)

    .. _Adafactor: Adaptive Learning Rates with Sublinear Memory Cost:
        https://arxiv.org/pdf/1804.04235

    """
)


def _single_tensor_adafactor(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    row_vars: List[Tensor],
    col_vars: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    beta1: float,
    beta2: float,
    d: float,
    lr: Union[Tensor, float],
    weight_decay: float,
    eps1: float,
    eps2: float,
    has_complex: bool,
):
    assert grad_scale is None and found_inf is None

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    for i, param in enumerate(params):
        grad = grads[i]
        if beta1 != 0:
            exp_avg = exp_avgs[i]
        step_t = state_steps[i]
        row_var = row_vars[i]
        col_var = col_vars[i]

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            if beta1 != 0:
                exp_avg = torch.view_as_real(exp_avg)
            param = torch.view_as_real(param)

        # update step
        step_t += 1
        step_float = step_t.item()

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        beta2_t = 1 - step_float**-0.8
        rho_t = min(0.01, 1 / (step_float**0.5))  # keras uses lr instead of 0.01
        alpha = max(eps2, param.norm(2).item() / (param.numel() ** 0.5)) * rho_t

        grad_squared_eps = grad * grad + eps1
        row_mean = grad_squared_eps.mean(dim=-1)
        row_var.lerp_(row_mean, 1 - beta2_t)
        col_mean = grad_squared_eps.mean(dim=-2)
        col_var.lerp_(col_mean, 1 - beta2_t)

        var_estimate = row_var.unsqueeze(-1) @ col_var.unsqueeze(-2)
        var_estimate.div_(row_var.mean(dim=-1))
        var_estimate.rsqrt_()

        update = var_estimate.mul_(grad)
        denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * d))
        param.add_(update, alpha=-alpha / denom)


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adafactor)
def adafactor(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    row_vars: List[Tensor],
    col_vars: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    has_complex: bool = False,
    *,
    beta1: float,
    beta2: float,
    d: float,
    lr: Union[float, Tensor],
    weight_decay: float,
    eps1: float,
    eps2: float,
    maximize: bool,
):
    r"""Functional API that performs Adafactor algorithm computation.

    See :class:`~torch.optim.Adafactor` for details.
    """
    if not torch._utils.is_compiling() and not all(
        isinstance(t, torch.Tensor) for t in state_steps
    ):
        raise RuntimeError(
            "`state_steps` argument must contain a list of singleton tensors"
        )

    func = _single_tensor_adafactor

    func(
        params,
        grads,
        exp_avgs,
        row_vars,
        col_vars,
        state_steps,
        beta1=beta1,
        beta2=beta2,
        d=d,
        lr=lr,
        weight_decay=weight_decay,
        eps1=eps1,
        eps2=eps2,
        grad_scale=grad_scale,
        found_inf=found_inf,
        has_complex=has_complex,
    )
