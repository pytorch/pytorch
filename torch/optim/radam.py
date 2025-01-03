# mypy: allow-untyped-defs
r"""Implementation for the RAdam algorithm."""
from typing import cast, Optional, Union

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


__all__ = ["RAdam", "radam"]


class RAdam(Optimizer):  # noqa: D101
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        decoupled_weight_decay: bool = False,
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
    ):  # noqa: D107
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
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

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            decoupled_weight_decay=decoupled_weight_decay,
            differentiable=differentiable,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):  # noqa: D105
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)
            group.setdefault("decoupled_weight_decay", False)
            group.setdefault("capturable", False)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = (
                        torch.tensor(
                            step_val, dtype=_get_scalar_dtype(), device=p.device
                        )
                        if group["capturable"]
                        else torch.tensor(step_val, dtype=_get_scalar_dtype())
                    )

    def _init_group(
        self, group, params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state["step"] = (
                        torch.zeros((), dtype=_get_scalar_dtype(), device=p.device)
                        if group["capturable"]
                        else torch.tensor(0.0, dtype=_get_scalar_dtype())
                    )
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
            beta1, beta2 = cast(tuple[float, float], group["betas"])

            has_complex = self._init_group(
                group, params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps
            )

            radam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                decoupled_weight_decay=group["decoupled_weight_decay"],
                has_complex=has_complex,
            )

        return loss


RAdam.__doc__ = (
    r"""Implements RAdam algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \beta_1, \beta_2
                \text{ (betas)}, \: \theta_0 \text{ (params)}, \:f(\theta) \text{ (objective)}, \:
                \lambda \text{ (weightdecay)}, \:\textit{maximize}                               \\
            &\hspace{13mm} \epsilon \text{ (epsilon)}, \textit{decoupled\_weight\_decay}         \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                v_0 \leftarrow 0 \text{ ( second moment)},                                       \\
            &\hspace{18mm} \rho_{\infty} \leftarrow 2/(1-\beta_2) -1                      \\[-1.ex]
            &\rule{110mm}{0.4pt}  \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{6mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{12mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})         \\
            &\hspace{6mm}\textbf{else}                                                           \\
            &\hspace{12mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{6mm} \theta_t \leftarrow \theta_{t-1}                                       \\
            &\hspace{6mm} \textbf{if} \: \lambda \neq 0                                          \\
            &\hspace{12mm}\textbf{if} \: \textit{decoupled\_weight\_decay}                       \\
            &\hspace{18mm} \theta_t \leftarrow \theta_{t} - \gamma \lambda \theta_{t}            \\
            &\hspace{12mm}\textbf{else}                                                          \\
            &\hspace{18mm} g_t \leftarrow g_t + \lambda \theta_{t}                               \\
            &\hspace{6mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{6mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{6mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{6mm}\rho_t \leftarrow \rho_{\infty} -
                2 t \beta^t_2 /\big(1-\beta_2^t \big)                                    \\[0.1.ex]
            &\hspace{6mm}\textbf{if} \: \rho_t > 5                                               \\
            &\hspace{12mm} l_t \leftarrow \frac{\sqrt{ (1-\beta^t_2) }}{ \sqrt{v_t} +\epsilon  } \\
            &\hspace{12mm} r_t \leftarrow
      \sqrt{\frac{(\rho_t-4)(\rho_t-2)\rho_{\infty}}{(\rho_{\infty}-4)(\rho_{\infty}-2) \rho_t}} \\
            &\hspace{12mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t} r_t l_t        \\
            &\hspace{6mm}\textbf{else}                                                           \\
            &\hspace{12mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}                \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `On the variance of the adaptive learning rate and beyond`_.

    This implementation provides an option to use either the original weight_decay implementation as in Adam
    (where the weight_decay is applied to the gradient) or the one from AdamW (where weight_decay is applied
    to the weight) through the decoupled_weight_decay option. When decoupled_weight_decay is set to False
    (default), it uses the original Adam style weight decay, otherwise, it uses the AdamW style which
    corresponds more closely to the `author's implementation`_ in the RAdam paper. Further information
    about decoupled weight decay can be found in `Decoupled Weight Decay Regularization`_.

    """
    + rf"""
    Args:
        {_params_doc}
        lr (float, Tensor, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        decoupled_weight_decay (bool, optional): whether to decouple the weight
            decay as in AdamW to obtain RAdamW. If True, the algorithm does not
            accumulate weight decay in the momentum nor variance. (default: False)
        {_foreach_doc}
        {_maximize_doc}
        {_capturable_doc}
        {_differentiable_doc}

    .. _On the variance of the adaptive learning rate and beyond:
        https://arxiv.org/abs/1908.03265
    .. _author's implementation:
        https://github.com/LiyuanLucasLiu/RAdam
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101

    """
)


def _single_tensor_radam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    decoupled_weight_decay: bool,
    differentiable: bool,
    maximize: bool,
    capturable: bool,
    has_complex: bool,
):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if not torch.compiler.is_compiling() and capturable:
            capturable_supported_devices = _get_capturable_supported_devices()
            assert (
                param.device.type == step_t.device.type
                and param.device.type in capturable_supported_devices
            ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

        if torch.is_complex(param):
            param = torch.view_as_real(param)
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)

        # update step
        step_t += 1
        step = step_t if capturable else _get_value(step_t)

        if weight_decay != 0:
            if decoupled_weight_decay:
                param.mul_(1 - lr * weight_decay)
            else:
                grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        # correcting bias for the first moving moment
        bias_corrected_exp_avg = exp_avg / bias_correction1

        # maximum length of the approximated SMA
        rho_inf = 2 / (1 - beta2) - 1
        # compute the length of the approximated SMA
        rho_t = rho_inf - 2 * step * (beta2**step) / bias_correction2

        def _compute_rect():
            return (
                (rho_t - 4)
                * (rho_t - 2)
                * rho_inf
                / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
            ) ** 0.5

        def _compute_adaptive_lr():
            exp_avg_sq_sqrt = exp_avg_sq.sqrt()
            if differentiable:
                exp_avg_sq_sqrt = exp_avg_sq_sqrt.add(eps)
            else:
                exp_avg_sq_sqrt = exp_avg_sq_sqrt.add_(eps)

            return (bias_correction2**0.5) / exp_avg_sq_sqrt

        # Compute the variance rectification term and update parameters accordingly
        if capturable:
            update = torch.where(
                rho_t > 5.0, _compute_rect() * _compute_adaptive_lr(), 1.0
            )
            param.add_(bias_corrected_exp_avg * lr * update, alpha=-1.0)
        else:
            if rho_t > 5.0:
                param.add_(
                    bias_corrected_exp_avg
                    * lr
                    * _compute_adaptive_lr()
                    * _compute_rect(),
                    alpha=-1.0,
                )
            else:
                param.add_(bias_corrected_exp_avg * lr, alpha=-1.0)


def _multi_tensor_radam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    decoupled_weight_decay: bool,
    differentiable: bool,
    maximize: bool,
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
            p.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, step in zip(params, state_steps)
        ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, state_steps]  # type: ignore[list-item]
    )
    for (
        grouped_params_,
        grouped_grads_,
        grouped_exp_avgs_,
        grouped_exp_avg_sqs_,
        grouped_state_steps_,
    ), _ in grouped_tensors.values():
        grouped_params = cast(list[Tensor], grouped_params_)
        grouped_grads = cast(list[Tensor], grouped_grads_)
        grouped_exp_avgs = cast(list[Tensor], grouped_exp_avgs_)
        grouped_exp_avg_sqs = cast(list[Tensor], grouped_exp_avg_sqs_)
        grouped_state_steps = cast(list[Tensor], grouped_state_steps_)

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

        if has_complex:
            _view_as_real(
                grouped_params, grouped_grads, grouped_exp_avgs, grouped_exp_avg_sqs
            )

        if maximize:
            grouped_grads = torch._foreach_neg(grouped_grads)  # type: ignore[assignment]

        # maximum length of the approximated SMA
        rho_inf = 2 / (1 - beta2) - 1
        # compute the length of the approximated SMA
        bias_correction1: Union[tuple[Tensor, ...], list[Tensor]]
        bias_correction2: Union[tuple[Tensor, ...], list[Tensor]]
        rho_t_list: Union[tuple[Tensor, ...], list[Tensor]]
        if capturable:
            bias_correction1 = torch._foreach_pow(beta2, grouped_state_steps)
            torch._foreach_neg_(bias_correction1)
            torch._foreach_add_(bias_correction1, 1)
            bias_correction2 = torch._foreach_pow(beta2, grouped_state_steps)
            torch._foreach_mul_(bias_correction2, grouped_state_steps)
            torch._foreach_mul_(bias_correction2, 2)
            torch._foreach_div_(bias_correction2, bias_correction1)
            torch._foreach_neg_(bias_correction2)
            torch._foreach_add_(bias_correction2, rho_inf)
            rho_t_list = bias_correction2
        else:
            rho_t_list = [
                rho_inf
                - 2
                * _get_value(step)
                * (beta2 ** _get_value(step))
                / (1 - beta2 ** _get_value(step))
                for step in grouped_state_steps
            ]

        if weight_decay != 0:
            if decoupled_weight_decay:
                torch._foreach_mul_(grouped_params, 1 - lr * weight_decay)
            else:
                # Re-use the intermediate memory (grouped_grads) already allocated for maximize
                if maximize:
                    torch._foreach_add_(
                        grouped_grads, grouped_params, alpha=weight_decay
                    )
                else:
                    grouped_grads = torch._foreach_add(  # type: ignore[assignment]
                        grouped_grads, grouped_params, alpha=weight_decay
                    )

        # Decay the first and second moment running average coefficient
        torch._foreach_lerp_(grouped_exp_avgs, grouped_grads, 1 - beta1)

        torch._foreach_mul_(grouped_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(
            grouped_exp_avg_sqs, grouped_grads, grouped_grads, 1 - beta2
        )

        # Delete the local intermediate since it won't be used anymore to save on peak memory
        del grouped_grads

        if capturable:
            num = torch._foreach_sub(rho_t_list, 4)
            sub2 = torch._foreach_sub(rho_t_list, 2)
            torch._foreach_mul_(num, sub2)
            del sub2
            torch._foreach_mul_(num, rho_inf)
            rho_inf = (rho_inf - 4) * (rho_inf - 2)
            denom = torch._foreach_mul(rho_t_list, rho_inf)
            torch._foreach_div_(num, denom)
            del denom
            torch._foreach_sqrt_(num)

            # TODO(mlazos): we should try and get a foreach_where op https://github.com/pytorch/pytorch/issues/117884
            rect = [
                torch.where(rho_t > 5.0, n, 0.0) for n, rho_t in zip(num, rho_t_list)
            ]
            del num
            del rho_t_list
            unrect_step_size = [torch.where(rect > 0, 0.0, 1.0) for rect in rect]
            torch._foreach_mul_(unrect_step_size, lr)

            bias_correction1 = torch._foreach_pow(beta1, grouped_state_steps)
            torch._foreach_neg_(bias_correction1)
            torch._foreach_add_(bias_correction1, 1)

            torch._foreach_div_(unrect_step_size, bias_correction1)
            torch._foreach_neg_(unrect_step_size)

            bias_correction2 = torch._foreach_pow(beta2, grouped_state_steps)
            torch._foreach_neg_(bias_correction2)
            torch._foreach_add_(bias_correction2, 1)
            torch._foreach_sqrt_(bias_correction2)
            torch._foreach_mul_(bias_correction2, lr)
            torch._foreach_mul_(bias_correction2, rect)
            del rect
            torch._foreach_neg_(bias_correction2)
            torch._foreach_div_(bias_correction2, bias_correction1)
            del bias_correction1
        else:
            rect = [
                (  # type: ignore[misc]
                    (rho_t - 4)  # type: ignore[arg-type]
                    * (rho_t - 2)
                    * rho_inf
                    / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
                )
                ** 0.5
                if rho_t > 5
                else 0
                for rho_t in rho_t_list
            ]
            unrectified = [0 if rect > 0 else 1.0 for rect in rect]

            bias_correction1 = [
                1 - beta1 ** _get_value(step) for step in grouped_state_steps
            ]
            unrect_step_size = [
                (lr * rect / bc) * -1 for rect, bc in zip(unrectified, bias_correction1)
            ]
            bias_correction2 = [
                ((1 - beta2 ** _get_value(step)) ** 0.5) * (lr * rect / bc) * -1
                for step, rect, bc in zip(grouped_state_steps, rect, bias_correction1)
            ]

        buffer = torch._foreach_sqrt(grouped_exp_avg_sqs)
        torch._foreach_add_(buffer, eps)
        torch._foreach_div_(buffer, bias_correction2)
        torch._foreach_reciprocal_(buffer)
        torch._foreach_add_(buffer, unrect_step_size)

        # Here, buffer = sqrt(1 - beta2^t) * rect_step_size / (sqrt(v) + eps) + unrect_step_size
        torch._foreach_addcmul_(grouped_params, grouped_exp_avgs, buffer)


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_radam)
def radam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    decoupled_weight_decay: bool = False,
    foreach: Optional[bool] = None,
    differentiable: bool = False,
    capturable: bool = False,
    has_complex: bool = False,
    maximize: bool = False,
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
):
    r"""Functional API that performs RAdam algorithm computation.

    See :class:`~torch.optim.RAdam` for details.
    """
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

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_radam
    else:
        func = _single_tensor_radam

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        decoupled_weight_decay=decoupled_weight_decay,
        differentiable=differentiable,
        capturable=capturable,
        has_complex=has_complex,
    )
