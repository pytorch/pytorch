# mypy: allow-untyped-defs
from typing import cast, Optional, Union

import torch
from torch import Tensor

from .optimizer import (
    _default_to_fused_or_foreach,
    _device_dtype_check_for_fused,
    _differentiable_doc,
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


__all__ = ["Adagrad", "adagrad"]


class Adagrad(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-2,
        lr_decay: float = 0,
        weight_decay: float = 0,
        initial_accumulator_value: float = 0,
        eps: float = 1e-10,
        foreach: Optional[bool] = None,
        *,
        maximize: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= lr_decay:
            raise ValueError(f"Invalid lr_decay value: {lr_decay}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= initial_accumulator_value:
            raise ValueError(
                f"Invalid initial_accumulator_value value: {initial_accumulator_value}"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(
            lr=lr,
            lr_decay=lr_decay,
            eps=eps,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            foreach=foreach,
            maximize=maximize,
            differentiable=differentiable,
            fused=fused,
        )
        super().__init__(params, defaults)

        if fused:
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")
            self._need_device_dtype_check_for_fused = True

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = (
                    torch.zeros(
                        (),
                        dtype=_get_scalar_dtype(is_fused=group["fused"]),
                        device=p.device,
                    )
                    if group["fused"]
                    else torch.tensor(0.0, dtype=_get_scalar_dtype())
                )
                init_value = (
                    complex(initial_accumulator_value, initial_accumulator_value)
                    if torch.is_complex(p)
                    else initial_accumulator_value
                )
                state["sum"] = torch.full_like(
                    p, init_value, memory_format=torch.preserve_format
                )

    def __setstate__(self, state):
        super().__setstate__(state)
        #  define "fused" for
        #  MYPY error: Name "fused" may be undefined
        fused = None
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)
            fused = group.setdefault("fused", None)

        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(
                    float(s["step"]), dtype=_get_scalar_dtype(is_fused=fused)
                )

    def share_memory(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["sum"].share_memory_()

    def _init_group(self, group, params_with_grad, grads, state_sums, state_steps):
        has_sparse_grad, has_complex = False, False
        for p in group["params"]:
            if p.grad is not None:
                if group["fused"] and getattr(
                    self,
                    "_need_device_dtype_check_for_fused",
                    True,
                ):
                    _device_dtype_check_for_fused(p, cuda_unsupported=True)
                    self._need_device_dtype_check_for_fused = False
                has_sparse_grad |= p.grad.is_sparse
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                grads.append(p.grad)
                state = self.state[p]
                state_sums.append(state["sum"])
                state_steps.append(state["step"])

        return has_sparse_grad, has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            state_sums: list[Tensor] = []
            state_steps: list[Tensor] = []

            has_sparse_grad, has_complex = self._init_group(
                group, params_with_grad, grads, state_sums, state_steps
            )

            adagrad(
                params_with_grad,
                grads,
                state_sums,
                state_steps,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                lr_decay=group["lr_decay"],
                eps=group["eps"],
                has_sparse_grad=has_sparse_grad,
                foreach=group["foreach"],
                maximize=group["maximize"],
                differentiable=group["differentiable"],
                has_complex=has_complex,
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        return loss


Adagrad.__doc__ = (
    r"""Implements Adagrad algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)}, \: f(\theta)
                \text{ (objective)}, \: \lambda \text{ (weight decay)},                          \\
            &\hspace{12mm}    \tau \text{ (initial accumulator value)}, \: \eta\text{ (lr decay)}\\
            &\textbf{initialize} :  state\_sum_0 \leftarrow \tau                          \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm} \tilde{\gamma}    \leftarrow \gamma / (1 +(t-1) \eta)                  \\
            &\hspace{5mm} \textbf{if} \: \lambda \neq 0                                          \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda \theta_{t-1}                             \\
            &\hspace{5mm}state\_sum_t  \leftarrow  state\_sum_{t-1} + g^2_t                      \\
            &\hspace{5mm}\theta_t \leftarrow
                \theta_{t-1}- \tilde{\gamma} \frac{g_t}{\sqrt{state\_sum_t}+\epsilon}            \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `Adaptive Subgradient Methods for Online Learning
    and Stochastic Optimization`_.
    """
    + rf"""
    Args:
        {_params_doc}
        lr (float, Tensor, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        initial_accumulator_value (float, optional): initial value of the
            sum of squares of gradients (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
        {_foreach_doc}
        {_maximize_doc}
        {_differentiable_doc}
        fused (bool, optional): whether the fused implementation (CPU only) is used.
            Currently, `torch.float64`, `torch.float32`, `torch.float16`, and `torch.bfloat16`
            are supported. (default: None). Please note that the fused implementations does not
            support sparse or complex gradients.
    .. _Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization: http://jmlr.org/papers/v12/duchi11a.html

    """
)


def adagrad(
    params: list[Tensor],
    grads: list[Tensor],
    state_sums: list[Tensor],
    state_steps: list[Tensor],
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting these as kwargs for now as functional API is compiled by torch/distributed/optim
    has_sparse_grad: bool = False,
    foreach: Optional[bool] = None,
    differentiable: bool = False,
    has_complex: bool = False,
    *,
    lr: float,
    weight_decay: float,
    lr_decay: float,
    eps: float,
    maximize: bool,
):
    r"""Functional API that performs Adagrad algorithm computation.

    See :class:`~torch.optim.Adagrad` for details.
    """
    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    # Respect when the user inputs False/True for foreach or fused. We only want to change
    # the default when neither have been user-specified. Note that we default to foreach
    # and pass False to use_fused. This is not a mistake--we want to give the fused impl
    # bake-in time before making it the default, even if it is typically faster.
    if fused is None and foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )

    if fused is None:
        fused = False
    if foreach is None:
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused optimizers")

    if fused and not torch.jit.is_scripting():
        func = _fused_adagrad
    elif foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adagrad
    else:
        func = _single_tensor_adagrad

    func(
        params,
        grads,
        state_sums,
        state_steps,
        lr=lr,
        weight_decay=weight_decay,
        lr_decay=lr_decay,
        eps=eps,
        has_sparse_grad=has_sparse_grad,
        maximize=maximize,
        differentiable=differentiable,
        has_complex=has_complex,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )


def _make_sparse(grad, grad_indices, values):
    size = grad.size()
    return torch.sparse_coo_tensor(grad_indices, values, size)


def _single_tensor_adagrad(
    params: list[Tensor],
    grads: list[Tensor],
    state_sums: list[Tensor],
    state_steps: list[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    lr: float,
    weight_decay: float,
    lr_decay: float,
    eps: float,
    has_sparse_grad: bool,
    maximize: bool,
    differentiable: bool,
    has_complex: bool,
):
    assert grad_scale is None and found_inf is None
    for param, grad, state_sum, step_t in zip(params, grads, state_sums, state_steps):
        # update step
        step_t += 1
        step = _get_value(step_t)
        grad = grad if not maximize else -grad

        if weight_decay != 0:
            if grad.is_sparse:
                raise RuntimeError(
                    "weight_decay option is not compatible with sparse gradients"
                )
            grad = grad.add(param, alpha=weight_decay)

        clr = lr / (1 + (step - 1) * lr_decay)

        if grad.is_sparse:
            grad = grad.coalesce()  # the update is non-linear so indices must be unique
            grad_indices = grad._indices()
            grad_values = grad._values()

            state_sum.add_(_make_sparse(grad, grad_indices, grad_values.pow(2)))
            std = state_sum.sparse_mask(grad)
            std_values = std._values().sqrt_().add_(eps)
            param.add_(
                _make_sparse(grad, grad_indices, grad_values / std_values), alpha=-clr
            )
        else:
            is_complex = torch.is_complex(param)
            if is_complex:
                grad = torch.view_as_real(grad)
                state_sum = torch.view_as_real(state_sum)
                param = torch.view_as_real(param)
            state_sum.addcmul_(grad, grad, value=1)
            if differentiable:
                std = state_sum.sqrt() + eps
            else:
                std = state_sum.sqrt().add_(eps)
            param.addcdiv_(grad, std, value=-clr)
            if is_complex:
                param = torch.view_as_complex(param)
                state_sum = torch.view_as_complex(state_sum)


def _multi_tensor_adagrad(
    params: list[Tensor],
    grads: list[Tensor],
    state_sums: list[Tensor],
    state_steps: list[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    lr: float,
    weight_decay: float,
    lr_decay: float,
    eps: float,
    has_sparse_grad: bool,
    maximize: bool,
    differentiable: bool,
    has_complex: bool,
):
    assert not differentiable, "_foreach ops don't support autograd"
    assert grad_scale is None and found_inf is None

    # Foreach functions will throw errors if given empty lists
    if len(params) == 0:
        return

    grouped_tensorlists = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, state_sums, state_steps]  # type: ignore[list-item]
    )
    for (
        device_params_,
        device_grads_,
        device_state_sums_,
        device_state_steps_,
    ), _ in grouped_tensorlists.values():
        device_params = cast(list[Tensor], device_params_)
        device_grads = cast(list[Tensor], device_grads_)
        device_state_sums = cast(list[Tensor], device_state_sums_)
        device_state_steps = cast(list[Tensor], device_state_steps_)

        device_has_sparse_grad = has_sparse_grad and any(
            grad.is_sparse for grad in device_grads
        )

        if device_has_sparse_grad:
            _single_tensor_adagrad(
                device_params,
                device_grads,
                device_state_sums,
                device_state_steps,
                lr=lr,
                weight_decay=weight_decay,
                lr_decay=lr_decay,
                eps=eps,
                has_sparse_grad=True,
                maximize=maximize,
                differentiable=differentiable,
                has_complex=has_complex,
                grad_scale=grad_scale,
                found_inf=found_inf,
            )
            continue

        # Handle complex parameters
        if has_complex:
            _view_as_real(device_params, device_grads, device_state_sums)

        if maximize:
            device_grads = torch._foreach_neg(device_grads)  # type: ignore[assignment]

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

        if weight_decay != 0:
            # Re-use the intermediate memory (device_grads) already allocated for maximize
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = torch._foreach_add(  # type: ignore[assignment]
                    device_grads, device_params, alpha=weight_decay
                )

        minus_clr = [
            -lr / (1 + (_get_value(step) - 1) * lr_decay) for step in device_state_steps
        ]

        torch._foreach_addcmul_(device_state_sums, device_grads, device_grads, value=1)

        std = torch._foreach_sqrt(device_state_sums)
        torch._foreach_add_(std, eps)

        if weight_decay != 0 or maximize:
            # Again, re-use the intermediate memory (device_grads) already allocated
            torch._foreach_mul_(device_grads, minus_clr)
            numerator = device_grads
        else:
            numerator = torch._foreach_mul(device_grads, minus_clr)  # type: ignore[assignment]

        torch._foreach_addcdiv_(device_params, numerator, std)


def _fused_adagrad(
    params: list[Tensor],
    grads: list[Tensor],
    state_sums: list[Tensor],
    state_steps: list[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    lr: float,
    weight_decay: float,
    lr_decay: float,
    eps: float,
    has_sparse_grad: bool,
    maximize: bool,
    differentiable: bool,
    has_complex: bool,
) -> None:
    if not params:
        return
    if has_sparse_grad or has_complex:
        raise RuntimeError("`fused` does not support sparse grad or complex param")

    if differentiable:
        raise RuntimeError(
            "adagrad with fused=True does not support differentiable=True"
        )

    grad_scale_dict = (
        {grad_scale.device: grad_scale} if grad_scale is not None else None
    )
    found_inf_dict = {found_inf.device: found_inf} if found_inf is not None else None

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, state_sums, state_steps]  # type: ignore[list-item]
    )
    for (device, _), (
        (
            device_params_,
            device_grads_,
            device_state_sums_,
            device_state_steps_,
        ),
        _,
    ) in grouped_tensors.items():
        device_params = cast(list[Tensor], device_params_)
        device_grads = cast(list[Tensor], device_grads_)
        device_state_sums = cast(list[Tensor], device_state_sums_)
        device_state_steps = cast(list[Tensor], device_state_steps_)

        device_grad_scale, device_found_inf = None, None
        if grad_scale is not None and grad_scale_dict is not None:
            if device not in grad_scale_dict:
                grad_scale_dict[device] = grad_scale.to(device, non_blocking=True)  # type: ignore[index]
            device_grad_scale = grad_scale_dict[device]  # type: ignore[index]
        if found_inf is not None and found_inf_dict is not None:
            if found_inf not in found_inf_dict:
                found_inf_dict[device] = found_inf.to(device, non_blocking=True)  # type: ignore[index]
            device_found_inf = found_inf_dict[device]  # type: ignore[index]
        torch._foreach_add_(device_state_steps, 1)
        torch._fused_adagrad_(
            device_params,
            device_grads,
            device_state_sums,
            device_state_steps,
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            eps=eps,
            maximize=maximize,
            grad_scale=device_grad_scale,
            found_inf=device_found_inf,
        )
        if device_found_inf is not None:
            torch._foreach_sub_(
                device_state_steps, [device_found_inf] * len(device_state_steps)
            )
