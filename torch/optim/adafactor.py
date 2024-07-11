# mypy: allow-untyped-defs
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from .optimizer import (
    _disable_dynamo_if_unsupported,
    _get_scalar_dtype,
    Optimizer,
    ParamsT,
)

__all__ = ["Adafactor", "adafactor"]


class Adafactor(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-2,
        beta2_decay: float = -0.8,
        eps: Tuple[Optional[float], float] = (None, 1e-3),
        d: float = 1,
        weight_decay: float = 0,
        *,
        maximize: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if not 0.0 >= beta2_decay:
            raise ValueError(f"beta2_decay should be <= 0 but is: {beta2_decay}")
        if eps[0] is not None and not 0.0 <= eps[0]:
            raise ValueError(f"epsilon1 should be >= 0 but is: {eps[0]}")
        if not 0.0 <= eps[1]:
            raise ValueError(f"epsilon2 should be >= 0 but is: {eps[1]}")
        if not 1.0 <= d:
            raise ValueError(f"Clipping threshold d should be >= 1 but is: {d}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"weight_decay should be >= 0 but is: {weight_decay}")
        defaults = dict(
            lr=lr,
            beta2_decay=beta2_decay,
            eps=eps,
            d=d,
            weight_decay=weight_decay,
            maximize=maximize,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
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
        row_vars,
        col_vars,
        variances,
        state_steps,
    ):
        for p in group["params"]:
            if p.grad is None:
                continue
            if torch.is_complex(p):
                raise RuntimeError("Adafactor does not support complex parameters")
            if p.grad.is_sparse:
                raise RuntimeError("Adafactor does not support sparse gradients")

            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.
                # This is because kernel launches are costly on CUDA and XLA.
                state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())

                if p.grad.dim() > 1:
                    # Row factor of variance, NOT the same shape as grads (will be reduced along last dim)
                    state["row_var"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    ).sum(dim=-1)
                    # Col factor of variance, NOT the same shape as grads (will be reduced along penultimate dim)
                    state["col_var"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    ).sum(dim=-2)
                else:
                    state["variance"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

            row_vars.append(state.get("row_var", None))
            col_vars.append(state.get("col_var", None))
            variances.append(state.get("variance", None))
            state_steps.append(state["step"])
        return False  # has_complex

    @torch.no_grad()
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
            row_vars: List[Tensor] = []
            col_vars: List[Tensor] = []
            variances: List[Tensor] = []
            state_steps: List[Tensor] = []
            eps1, eps2 = group["eps"]

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                row_vars,
                col_vars,
                variances,
                state_steps,
            )

            adafactor(
                params_with_grad,
                grads,
                row_vars,
                col_vars,
                variances,
                state_steps,
                d=group["d"],
                lr=group["lr"],
                beta2_decay=group["beta2_decay"],
                weight_decay=group["weight_decay"],
                eps1=eps1,
                eps2=eps2,
                maximize=group["maximize"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                has_complex=has_complex,
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
        lr (float, Tensor, optional): learning rate (default: 1e-3). Adafactor only
            uses lr for weight_decay application. A tensor LR
            is not yet supported for all our implementations. Please use a float
            LR if you are not also specifying fused=True or capturable=True.
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (Tuple[float, float], optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)

    .. _Adafactor: Adaptive Learning Rates with Sublinear Memory Cost:
        https://arxiv.org/pdf/1804.04235

    """
)


def _single_tensor_adafactor(
    params: List[Tensor],
    grads: List[Tensor],
    row_vars: List[Tensor],
    col_vars: List[Tensor],
    variances: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    d: float,
    lr: Union[Tensor, float],
    beta2_decay: float,
    weight_decay: float,
    eps1: Optional[float],
    eps2: float,
    maximize: bool,
    has_complex: bool,
):
    assert grad_scale is None and found_inf is None

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        step_t = state_steps[i]
        row_var = row_vars[i]
        col_var = col_vars[i]
        variance = variances[i]
        if eps1 is None:
            eps1 = torch.finfo(param.dtype).eps

        # update step
        step_t += 1
        step_float = step_t.item()

        # Perform stepweight decay
        if weight_decay != 0:
            param.mul_(1 - lr * weight_decay)

        beta2_t = 1 - step_float**beta2_decay
        rho_t = min(
            lr, 1 / (step_float**0.5)
        )  # keras uses lr instead of 0.01 => and US TOO
        alpha = max(eps2, param.norm(2).item() / (param.numel() ** 0.5)) * rho_t

        if grad.dim() > 1:
            # same as (g * g).mean(dim=-1) w/o materializing an intermediate size g
            row_mean = torch.norm(grad, dim=-1).square_().div_(grad.size(-1))
            row_var.lerp_(row_mean, 1 - beta2_t)
            # same as (g * g).mean(dim=-2) w/o materializing an intermediate size g
            col_mean = torch.norm(grad, dim=-2).square_().div_(grad.size(-2))
            col_var.lerp_(col_mean, 1 - beta2_t)
            var_estimate = row_var.unsqueeze(-1) @ col_var.unsqueeze(-2)
            # Include in docs later: we move epsilon1 to these 2 lines
            # as an attempt to follow the intention of the paper authors
            # in using eps1 for stability.
            var_est_denom = (
                row_var.mean(dim=-1).clamp_(min=eps1).unsqueeze(-1).unsqueeze(-1)
            )
            var_estimate.div_(var_est_denom)
            update = var_estimate.sqrt_().clamp_(min=eps1).reciprocal_()
        else:
            grad_squared_eps = grad * grad + eps1
            variance.lerp_(grad_squared_eps, 1 - beta2_t)
            update = variance.rsqrt()

        update.mul_(grad)
        denom = max(1.0, update.norm(2).item() / ((update.numel() ** 0.5) * d))
        param.add_(update, alpha=-alpha / denom)


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adafactor)
def adafactor(
    params: List[Tensor],
    grads: List[Tensor],
    row_vars: List[Tensor],
    col_vars: List[Tensor],
    variances: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    has_complex: bool = False,
    *,
    d: float,
    lr: Union[float, Tensor],
    beta2_decay: float,
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
        row_vars,
        col_vars,
        variances,
        state_steps,
        d=d,
        lr=lr,
        beta2_decay=beta2_decay,
        weight_decay=weight_decay,
        eps1=eps1,
        eps2=eps2,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
        has_complex=has_complex,
    )
