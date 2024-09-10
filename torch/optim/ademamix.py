# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import time
from typing import List, Optional, Tuple, Union, cast

import torch
from torch import Tensor
from torch.optim.optimizer import (
    _default_to_fused_or_foreach,
    _disable_dynamo_if_unsupported,
    _get_capturable_supported_devices,
    _get_value,
    _use_grad_for_differentiable,
    Optimizer,
    ParamsT,
    _get_scalar_dtype, _dispatch_sqrt, _capturable_doc, _maximize_doc, _differentiable_doc,
)

__all__ = ["AdEMAMix", "ademamix"]


class AdEMAMix(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, torch.Tensor] = 1e-3,
        betas: Tuple[float, float, float] = (0.9, 0.999, 0.9999),
        alpha: float = 5.0,
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        # T_alpha and T_beta3 are typically set to the total number of training iterations.
        T_alpha: Optional[int] = None,
        T_beta3: Optional[int] = None,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        if isinstance(lr, torch.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 2: {betas[2]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if T_alpha is not None and T_alpha <= 0:
            raise ValueError(f"Invalid T_alpha value: {T_alpha}")
        if T_beta3 is not None and T_beta3 <= 0:
            raise ValueError(f"Invalid T_beta3 value: {T_beta3}")

        defaults = dict(
            lr=lr,
            betas=betas,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            T_alpha=T_alpha,
            T_beta3=T_beta3,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            amsgrad=amsgrad,
            fused=fused,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        # self.param_groups correspond to
        # different learnable parameters in the model
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            fused = group.setdefault("fused", None)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = (
                        torch.tensor(
                            step_val,
                            dtype=_get_scalar_dtype(is_fused=fused),
                            device=p.device,
                        )
                        if group["capturable"] or group["fused"]
                        else torch.tensor(step_val, dtype=_get_scalar_dtype())
                    )

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        amsgrad,
        exp_avgs1,
        exp_avgs2,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError("AdEMAMix does not support sparse gradients")
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = (
                        torch.zeros(
                            (),
                            dtype=_get_scalar_dtype(is_fused=group["fused"]),
                            device=p.device,
                        )
                        if group["capturable"] or group["fused"]
                        else torch.tensor(0.0, dtype=_get_scalar_dtype())
                    )
                    state["exp_avg1"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg2"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                exp_avgs1.append(state["exp_avg1"])
                exp_avgs2.append(state["exp_avg2"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                state_steps.append(state["step"])
                if amsgrad:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs1 = []
            exp_avgs2 = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad: bool = group["amsgrad"]

            beta1, beta2, beta3 = cast(Tuple[float, float, float], group["betas"])

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                amsgrad,
                exp_avgs1,
                exp_avgs2,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            ademamix(
                params_with_grad,
                grads,
                exp_avgs1,
                exp_avgs2,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                alpha=group["alpha"],
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                has_complex=has_complex,
                fused=group["fused"],
                amsgrad=group["amsgrad"],
                T_alpha=group["T_alpha"],
                T_beta3=group["T_beta3"],
            )

        return loss

AdEMAMix.__doc__ = (
    r"""Implements AdEMAMix algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{(lr)}, \: \beta_1, \beta_2, \beta_3
                \text{(betas)}, \: \alpha \text{(alpha)}, \: \theta_0 \text{(params)}, \: f(\theta) \text{(objective)},
                \: \epsilon \text{ (epsilon)}                                                    \\
            &\hspace{13mm}      \lambda \text{(weight decay)},  \: \textit{amsgrad},
                \: \textit{maximize}, \: T_{\alpha}, \: T_{\beta_3}                              \\
            &\textbf{initialize} : m_0 \leftarrow 0 \text{ (first moment)}, v_0 \leftarrow 0
                \text{ (second moment)}, \: \widehat{v_0}^{max}\leftarrow 0,
                \: e_0 \leftarrow 0 \text{ (EMA of gradient)}                                    \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\

            &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})         \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{5mm} \theta_t \leftarrow \theta_{t-1} - \gamma \lambda \theta_{t-1}         \\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}e_t           \leftarrow   \beta_{3,t} e_{t-1} + (1-\beta_{3,t}) g_t    \\
            &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}\alpha_t      \leftarrow   \alpha \cdot \min(t/T_{\alpha}, 1)           \\
            &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
            &\hspace{10mm}\widehat{v_t}^{max} \leftarrow \mathrm{max}(\widehat{v_t}^{max},
                \widehat{v_t})                                                                   \\
            &\hspace{10mm}\theta_t \leftarrow \theta_t - \gamma (\widehat{m_t} + \alpha_t e_t)/
                \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)                                 \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\theta_t \leftarrow \theta_t - \gamma (\widehat{m_t} + \alpha_t e_t)/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to [AdEMAMix paper link].
    """ + rf"""
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, Tensor, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float, float], optional): coefficients used for computing
            running averages of gradient and its square, and EMA of gradient
            (default: (0.9, 0.999, 0.9999))
        alpha (float, optional): coefficient for mixing two types of momentum (default: 5.0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (bool, optional): whether to use the AMSGrad variant of this
            algorithm (default: False)
        T_alpha (int, optional): total number of training iterations for alpha scheduling
        T_beta3 (int, optional): total number of training iterations for beta3 scheduling
        {_maximize_doc}
        {_capturable_doc}
        {_differentiable_doc}
        Currently, AdEMAMix does not support foreach and fused operations.
    ..  _The AdEMAMix Optimizer: Better, Faster, Older
        https://arxiv.org/pdf/2409.03137\
        
    """

)

def _single_tensor_ademamix(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs1: List[Tensor],
    exp_avgs2: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    found_inf: Optional[Tensor] = None,
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    beta3: float,
    alpha: float,
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
    T_alpha: Optional[int],
    T_beta3: Optional[int],
    capturable: bool,
    differentiable: bool,
    has_complex: bool,
):
    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float), "lr must be a float in JIT script mode"

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg1 = exp_avgs1[i]
        exp_avg2 = exp_avgs2[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        if not torch._utils.is_compiling() and capturable:
            capturable_supported_device = _get_capturable_supported_devices()
            assert (
                param.device.type == step_t.device.type
                and param.device.type in capturable_supported_device
            ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_device}."

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg1 = torch.view_as_real(exp_avg1)
            exp_avg2 = torch.view_as_real(exp_avg2)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            if amsgrad:
                param = torch.view_as_real(param)

        step_t += 1
        param.mul_(1 - lr * weight_decay)

        # If T_alpha is None, scheduler function is not used.
        if T_alpha is None:
            alpha_t = alpha
        else:
            alpha_t = alpha * min((step_t / T_alpha).item(), 1)

        # If T_beta3 is None, scheduler function is not used.
        if T_beta3 is None:
            beta3_t = beta3
        else:
            t_ratio = step_t / T_beta3
            log_ratio = torch.log(torch.tensor(beta3)) / torch.log(torch.tensor(beta1))
            exponent = torch.log(torch.tensor(beta1)) * (
                (1 - t_ratio) * log_ratio + t_ratio
            )
            beta3_t = torch.exp(
                torch.min(exponent, torch.log(torch.tensor(beta3)))
            ).item()

        exp_avg1.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        exp_avg2.mul_(beta3_t).add_(grad, alpha=1 - beta3_t)
        if capturable or differentiable:
            step = step_t

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step

            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()

            bias_correction2_sqrt = bias_correction2.sqrt()
            if amsgrad:
                if differentiable:
                    max_exp_avg_sq = max_exp_avg_sqs[i].clone()
                else:
                    max_exp_avg_sq = max_exp_avg_sqs[i]

                max_exp_avg_sqs[i].copy_(torch.maximum(max_exp_avg_sq, exp_avg_sq))
                denom = (max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt)).add_(
                    eps / step_size_neg
                )
            else:
                denom = (
                    exp_avg_sq.sqrt() / bias_correction2_sqrt * step_size_neg
                ).add_(eps)

            update = exp_avg1.add(alpha_t * exp_avg2)
            param.addcdiv_(update, denom)

        else:
            step = int(_get_value(step_t))

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step

            step_size = lr / bias_correction1

            bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)

            if amsgrad:
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                denom = (max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

            update = exp_avg1.add(alpha_t * exp_avg2)
            param.addcdiv_(update, denom, value=-step_size)

        if torch.is_complex(params[i]):
            params[i] = torch.view_as_complex(param)
            if amsgrad:
                max_exp_avg_sqs[i] = torch.view_as_complex(max_exp_avg_sqs[i])


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_ademamix)
def ademamix(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs1: List[Tensor],
    exp_avgs2: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    has_complex: bool = False,
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    beta3: float,
    alpha: float,
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
    T_alpha: Optional[int],
    T_beta3: Optional[int],
):
    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused optimizers")

    if fused and not torch.jit.is_scripting():
        raise RuntimeError("Fused AdEMAMix has not been implemented yet")
    elif foreach and not torch.jit.is_scripting():
        raise RuntimeError("Foreach AdEMAMix has not been implemented yet")
    else:
        func = _single_tensor_ademamix

    func(
        params,
        grads,
        exp_avgs1,
        exp_avgs2,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        beta3=beta3,
        alpha=alpha,
        has_complex=has_complex,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        T_alpha=T_alpha,
        T_beta3=T_beta3,
    )
