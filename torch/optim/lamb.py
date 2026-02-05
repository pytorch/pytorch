# mypy: allow-untyped-defs
r"""LAMB Optimizer Implementation.

LAMB (Layer-wise Adaptive Moments) optimizer for large batch training.
Reference: https://arxiv.org/abs/1904.00962
"""

from typing import cast, Optional

import torch
from torch import Tensor

from .optimizer import (
    _default_to_fused_or_foreach,
    _disable_dynamo_if_unsupported,
    _foreach_doc,
    _get_scalar_dtype,
    _get_value,
    _maximize_doc,
    _params_doc,
    _use_grad_for_differentiable,
    _view_as_real,
    Optimizer,
    ParamsT,
)


__all__ = ["LAMB", "lamb"]


class LAMB(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.01,
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        differentiable: bool = False,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "foreach": foreach,
            "maximize": maximize,
            "differentiable": differentiable,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError("LAMB does not support sparse gradients")
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                if group["differentiable"] and state["step"].requires_grad:
                    raise RuntimeError(
                        "`requires_grad` is not supported for `step` in differentiable mode"
                    )

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
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            state_steps: list[Tensor] = []
            beta1, beta2 = group["betas"]

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
            )

            lamb(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                has_complex=has_complex,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                differentiable=group["differentiable"],
            )

        return loss


LAMB.__doc__ = (
    r"""Implements LAMB algorithm.

    LAMB (Layer-wise Adaptive Moments) is designed for large batch training.
    It computes a trust ratio per layer that scales the update based on the
    ratio of parameter norm to update norm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \beta_1, \beta_2
                \text{ (betas)}, \theta_0 \text{ (params)}, f(\theta) \text{ (objective)}        \\
            &\hspace{13mm}      \lambda \text{ (weight decay)}, \epsilon                         \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ (first moment)},
                v_0 \leftarrow 0 \text{ (second moment)}                                  \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}u_t \leftarrow \widehat{m_t}/(\sqrt{\widehat{v_t}} + \epsilon)
                + \lambda \theta_{t-1}                                                           \\
            &\hspace{5mm}r_1 \leftarrow \|\theta_{t-1}\|_2                                       \\
            &\hspace{5mm}r_2 \leftarrow \|u_t\|_2                                                \\
            &\hspace{5mm}\phi \leftarrow r_1 / r_2 \text{ if } r_1 > 0 \text{ and } r_2 > 0
                \text{ else } 1                                                                  \\
            &\hspace{5mm}\theta_t \leftarrow \theta_{t-1} - \gamma \phi u_t                      \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to
    `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.
    """
    + rf"""
    Args:
        {_params_doc}
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-6)
        weight_decay (float, optional): weight decay (default: 0.01)
        {_foreach_doc}
        {_maximize_doc}

    .. _Large Batch Optimization for Deep Learning\: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962

    """
)


def _single_tensor_lamb(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    *,
    has_complex: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
    differentiable: bool,
) -> None:
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # Update step
        step_t += 1
        step = _get_value(step_t)

        # Handle complex parameters
        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            param = torch.view_as_real(param)

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # Bias correction
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        # Compute the Adam update (bias-corrected)
        exp_avg_corrected = exp_avg / bias_correction1
        exp_avg_sq_corrected = exp_avg_sq / bias_correction2

        # u_t = m_hat / (sqrt(v_hat) + eps) + weight_decay * param
        update = exp_avg_corrected / (exp_avg_sq_corrected.sqrt() + eps)
        if weight_decay != 0:
            update = update.add(param, alpha=weight_decay)

        # Compute trust ratio (layer-wise)
        # r1 = ||param||, r2 = ||update||
        param_norm = param.norm()
        update_norm = update.norm()

        # Trust ratio: phi = r1 / r2 if both > 0, else 1.0
        if param_norm > 0 and update_norm > 0:
            trust_ratio = param_norm / update_norm
        else:
            trust_ratio = 1.0

        # Apply update with trust ratio scaling
        param.add_(update, alpha=-lr * trust_ratio)


def _multi_tensor_lamb(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    *,
    has_complex: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
    differentiable: bool,
) -> None:
    if len(params) == 0:
        return

    if differentiable:
        raise RuntimeError("_foreach ops don't support autograd")

    # Handle complex parameters
    if has_complex:
        _view_as_real(params, grads, exp_avgs, exp_avg_sqs)

    if maximize:
        grads = torch._foreach_neg(grads)  # type: ignore[assignment]

    # Group tensors by device and dtype
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, state_steps]  # type: ignore[list-item]
    )

    for (
        device_params_,
        device_grads_,
        device_exp_avgs_,
        device_exp_avg_sqs_,
        device_state_steps_,
    ), _ in grouped_tensors.values():
        device_params = cast(list[Tensor], device_params_)
        device_grads = cast(list[Tensor], device_grads_)
        device_exp_avgs = cast(list[Tensor], device_exp_avgs_)
        device_exp_avg_sqs = cast(list[Tensor], device_exp_avg_sqs_)
        device_state_steps = cast(list[Tensor], device_state_steps_)

        # Update steps
        # If steps are on CPU, foreach will fall back to the slow path
        if not torch.compiler.is_compiling() and device_state_steps[0].is_cpu:
            torch._foreach_add_(
                device_state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0
            )
        else:
            torch._foreach_add_(device_state_steps, 1)

        # Decay the first and second moment running average coefficient
        torch._foreach_lerp_(device_exp_avgs, device_grads, 1 - beta1)
        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(
            device_exp_avg_sqs, device_grads, device_grads, value=1 - beta2
        )

        # Compute bias-corrected estimates and updates per parameter
        # Note: Trust ratio must be computed per-parameter, so we can't fully vectorize
        for i, (param, exp_avg, exp_avg_sq, step_t) in enumerate(
            zip(
                device_params,
                device_exp_avgs,
                device_exp_avg_sqs,
                device_state_steps,
                strict=True,
            )
        ):
            step = _get_value(step_t)
            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step

            # Compute update
            exp_avg_corrected = exp_avg / bias_correction1
            exp_avg_sq_corrected = exp_avg_sq / bias_correction2
            update = exp_avg_corrected / (exp_avg_sq_corrected.sqrt() + eps)

            if weight_decay != 0:
                update = update.add(param, alpha=weight_decay)

            # Compute trust ratio
            param_norm = param.norm()
            update_norm = update.norm()

            if param_norm > 0 and update_norm > 0:
                trust_ratio = param_norm / update_norm
            else:
                trust_ratio = 1.0

            # Apply update
            param.add_(update, alpha=-lr * trust_ratio)


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_lamb)
def lamb(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: Optional[bool] = None,
    differentiable: bool = False,
    has_complex: bool = False,
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
) -> None:
    r"""Functional API that performs LAMB algorithm computation.

    See :class:`~torch.optim.LAMB` for details.
    """
    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )
    if foreach is None:
        foreach = False

    # this check is slow during compilation, so we skip it
    # if it's strictly needed we can add this check back in dynamo
    if not torch.compiler.is_compiling() and not all(
        isinstance(t, torch.Tensor) for t in state_steps
    ):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_lamb
    else:
        func = _single_tensor_lamb

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        has_complex=has_complex,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        differentiable=differentiable,
    )
