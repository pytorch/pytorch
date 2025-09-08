# mypy: allow-untyped-defs
from typing import Any, cast, Optional, Union

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
    _maximize_doc,
    _params_doc,
    _to_scalar,
    _use_grad_for_differentiable,
    _view_as_real,
    Optimizer,
    ParamsT,
)


__all__ = ["Adadelta", "adadelta"]


class Adadelta(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0,
        foreach: Optional[bool] = None,
        *,
        capturable: bool = False,
        maximize: bool = False,
        differentiable: bool = False,
    ):
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"Invalid rho value: {rho}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "rho": rho,
            "eps": eps,
            "weight_decay": weight_decay,
            "maximize": maximize,
            "capturable": capturable,
            "foreach": foreach,
            "differentiable": differentiable,
        }
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
        self,
        group: dict[str, Any],
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        square_avgs: list[Tensor],
        acc_deltas: list[Tensor],
        state_steps: list[Tensor],
    ):
        has_complex = False
        p: Tensor
        for p in group["params"]:
            if p.grad is None:
                continue
            has_complex |= torch.is_complex(p)
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("Adadelta does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]

            # Lazy state initialization
            if len(state) == 0:
                state["step"] = (
                    torch.zeros((), dtype=_get_scalar_dtype(), device=p.device)
                    if group["capturable"]
                    else torch.zeros((), dtype=_get_scalar_dtype())
                )

                state["square_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                state["acc_delta"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )

            square_avgs.append(state["square_avg"])
            acc_deltas.append(state["acc_delta"])
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
            square_avgs: list[Tensor] = []
            acc_deltas: list[Tensor] = []
            state_steps: list[Tensor] = []
            (
                lr,
                rho,
                eps,
                weight_decay,
                foreach,
                maximize,
                differentiable,
                capturable,
            ) = (
                group["lr"],
                group["rho"],
                group["eps"],
                group["weight_decay"],
                group["foreach"],
                group["maximize"],
                group["differentiable"],
                group["capturable"],
            )

            has_complex = self._init_group(
                group, params_with_grad, grads, square_avgs, acc_deltas, state_steps
            )

            adadelta(
                params_with_grad,
                grads,
                square_avgs,
                acc_deltas,
                state_steps,
                lr=lr,
                rho=rho,
                eps=eps,
                weight_decay=weight_decay,
                foreach=foreach,
                maximize=maximize,
                differentiable=differentiable,
                capturable=capturable,
                has_complex=has_complex,
            )

        return loss


Adadelta.__doc__ = (
    r"""Implements Adadelta algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)},
                \: f(\theta) \text{ (objective)}, \: \rho \text{ (decay)},
                \: \lambda \text{ (weight decay)}                                                \\
            &\textbf{initialize} :  v_0  \leftarrow 0 \: \text{ (square avg)},
                \: u_0 \leftarrow 0 \: \text{ (accumulate variables)}                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}if \: \lambda \neq 0                                                    \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm} v_t      \leftarrow v_{t-1} \rho + g^2_t (1 - \rho)                    \\
            &\hspace{5mm}\Delta x_t    \leftarrow   \frac{\sqrt{u_{t-1} +
                \epsilon }}{ \sqrt{v_t + \epsilon}  }g_t \hspace{21mm}                           \\
            &\hspace{5mm} u_t  \leftarrow   u_{t-1}  \rho +
                 \Delta x^2_t  (1 - \rho)                                                        \\
            &\hspace{5mm}\theta_t      \leftarrow   \theta_{t-1} - \gamma  \Delta x_t            \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `ADADELTA: An Adaptive Learning Rate Method`_.
    """
    + rf"""
    Args:
        {_params_doc}
        lr (float, Tensor, optional): coefficient that scale delta before it is applied
            to the parameters (default: 1.0)
        rho (float, optional): coefficient used for computing a running average
            of squared gradients (default: 0.9). A higher value of `rho` will
            result in a slower average, which can be helpful for preventing
            oscillations in the learning process.
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-6).
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        {_foreach_doc}
        {_capturable_doc}
        {_maximize_doc}
        {_differentiable_doc}

    .. _ADADELTA\: An Adaptive Learning Rate Method:
        https://arxiv.org/abs/1212.5701

    """
)


def _single_tensor_adadelta(
    params: list[Tensor],
    grads: list[Tensor],
    square_avgs: list[Tensor],
    acc_deltas: list[Tensor],
    state_steps: list[Tensor],
    *,
    lr: float,
    rho: float,
    eps: float,
    weight_decay: float,
    maximize: bool,
    differentiable: bool,
    capturable: bool,
    has_complex: bool,
):
    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if not torch.compiler.is_compiling() and capturable:
        capturable_supported_devices = _get_capturable_supported_devices(
            supports_xla=False
        )
        assert all(
            p.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, step in zip(params, state_steps)
        ), (
            f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."
        )

    if not torch.jit.is_scripting():
        lr = _to_scalar(lr)

    for param, grad, square_avg, acc_delta, step in zip(
        params, grads, square_avgs, acc_deltas, state_steps
    ):
        step += 1
        grad = grad if not maximize else -grad

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        if torch.is_complex(param):
            square_avg = torch.view_as_real(square_avg)
            acc_delta = torch.view_as_real(acc_delta)
            grad = torch.view_as_real(grad)

        square_avg.mul_(rho).addcmul_(grad, grad, value=1 - rho)
        std = square_avg.add(eps).sqrt_()
        delta = acc_delta.add(eps).sqrt_()
        if differentiable:
            delta = delta.clone()
        delta.div_(std).mul_(grad)
        acc_delta.mul_(rho).addcmul_(delta, delta, value=1 - rho)

        if torch.is_complex(param):
            delta = torch.view_as_complex(delta)
        param.add_(delta, alpha=-lr)


def _multi_tensor_adadelta(
    params: list[Tensor],
    grads: list[Tensor],
    square_avgs: list[Tensor],
    acc_deltas: list[Tensor],
    state_steps: list[Tensor],
    *,
    lr: float,
    rho: float,
    eps: float,
    weight_decay: float,
    maximize: bool,
    differentiable: bool,
    capturable: bool,
    has_complex: bool,
):
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
        ), (
            f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."
        )

    if len(params) == 0:
        return

    lr = _to_scalar(lr)

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, square_avgs, acc_deltas, state_steps]  # type: ignore[list-item]
    )
    for (
        device_params_,
        device_grads_,
        device_square_avgs_,
        device_acc_deltas_,
        device_state_steps_,
    ), _ in grouped_tensors.values():
        device_params = cast(list[Tensor], device_params_)
        device_grads = cast(list[Tensor], device_grads_)
        device_square_avgs = cast(list[Tensor], device_square_avgs_)
        device_acc_deltas = cast(list[Tensor], device_acc_deltas_)
        device_state_steps = cast(list[Tensor], device_state_steps_)
        if has_complex:
            _view_as_real(
                device_params, device_grads, device_square_avgs, device_acc_deltas
            )

        # Update steps
        # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
        # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
        # wrapped it once now. The alpha is required to assure we go to the right overload.
        if not torch.compiler.is_compiling() and device_state_steps[0].is_cpu:
            torch._foreach_add_(
                device_state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0
            )
        else:
            torch._foreach_add_(device_state_steps, 1)

        if maximize:
            device_grads = torch._foreach_neg(device_grads)  # type: ignore[assignment]

        if weight_decay != 0:
            # Reuse the intermediate memory (device_grads) already allocated for maximize
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = torch._foreach_add(  # type: ignore[assignment]
                    device_grads, device_params, alpha=weight_decay
                )

        torch._foreach_mul_(device_square_avgs, rho)
        torch._foreach_addcmul_(
            device_square_avgs, device_grads, device_grads, value=1 - rho
        )

        std = torch._foreach_add(device_square_avgs, eps)
        torch._foreach_sqrt_(std)

        deltas = torch._foreach_add(device_acc_deltas, eps)
        torch._foreach_sqrt_(deltas)
        torch._foreach_div_(deltas, std)
        torch._foreach_mul_(deltas, device_grads)

        torch._foreach_mul_(device_acc_deltas, rho)
        torch._foreach_addcmul_(device_acc_deltas, deltas, deltas, value=1 - rho)

        # If LR is a tensor, the else branch will internally call item()
        # which will cause silent incorrectness if we are capturing
        if capturable and isinstance(lr, torch.Tensor):
            torch._foreach_mul_(deltas, -lr)
            torch._foreach_add_(device_params, deltas)
        else:
            torch._foreach_add_(device_params, deltas, alpha=-lr)


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adadelta)
def adadelta(
    params: list[Tensor],
    grads: list[Tensor],
    square_avgs: list[Tensor],
    acc_deltas: list[Tensor],
    state_steps: list[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    capturable: bool = False,
    foreach: Optional[bool] = None,
    differentiable: bool = False,
    has_complex: bool = False,
    *,
    lr: float,
    rho: float,
    eps: float,
    weight_decay: float,
    maximize: bool,
):
    r"""Functional API that performs Adadelta algorithm computation.

    See :class:`~torch.optim.Adadelta` for details.
    """

    # this check is slow during compilation, so we skip it
    # if it's strictly needed we can add this check back in dynamo
    if not torch.compiler.is_compiling() and not all(
        isinstance(t, torch.Tensor) for t in state_steps
    ):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    # We still respect when the user inputs False for foreach.
    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adadelta
    else:
        func = _single_tensor_adadelta

    func(
        params,
        grads,
        square_avgs,
        acc_deltas,
        state_steps,
        lr=lr,
        rho=rho,
        eps=eps,
        weight_decay=weight_decay,
        maximize=maximize,
        differentiable=differentiable,
        capturable=capturable,
        has_complex=has_complex,
    )
