# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
r"""Implementation for the Resilient backpropagation."""
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
    _maximize_doc,
    _params_doc,
    _use_grad_for_differentiable,
    _view_as_real,
    Optimizer,
    ParamsT,
)


__all__ = ["Rprop", "rprop"]


class Rprop(Optimizer):  # noqa: D101
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-2,
        etas: Tuple[float, float] = (0.5, 1.2),
        step_sizes: Tuple[float, float] = (1e-6, 50),
        *,
        capturable: bool = False,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        differentiable: bool = False,
    ):  # noqa: D107
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 < etas[0] < 1.0 < etas[1]:
            raise ValueError(f"Invalid eta values: {etas[0]}, {etas[1]}")

        defaults = dict(
            lr=lr,
            etas=etas,
            step_sizes=step_sizes,
            foreach=foreach,
            maximize=maximize,
            differentiable=differentiable,
            capturable=capturable,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):  # noqa: D105
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

    def _init_group(self, group, params, grads, prevs, step_sizes, state_steps):
        has_complex = False
        for p in group["params"]:
            if p.grad is None:
                continue
            has_complex |= torch.is_complex(p)
            params.append(p)
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("Rprop does not support sparse gradients")

            grads.append(grad)
            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state["step"] = (
                    torch.zeros((), dtype=_get_scalar_dtype(), device=p.device)
                    if group["capturable"]
                    else torch.zeros((), dtype=_get_scalar_dtype())
                )

                state["prev"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if p.dtype.is_complex:
                    # Complex Number should be as if they are two independent real numbers.
                    # Hence the step_size shouldn't be zero for imaginary part.
                    state["step_size"] = torch.full_like(
                        grad, complex(group["lr"], group["lr"])
                    )
                else:
                    state["step_size"] = torch.full_like(grad, group["lr"])

            prevs.append(state["prev"])
            step_sizes.append(state["step_size"])
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
            params: List[Tensor] = []
            grads: List[Tensor] = []
            prevs: List[Tensor] = []
            step_sizes: List[Tensor] = []
            state_steps: List[Tensor] = []

            etaminus, etaplus = group["etas"]
            step_size_min, step_size_max = group["step_sizes"]
            foreach = group["foreach"]
            maximize = group["maximize"]

            has_complex = self._init_group(
                group, params, grads, prevs, step_sizes, state_steps
            )

            rprop(
                params,
                grads,
                prevs,
                step_sizes,
                state_steps,
                step_size_min=step_size_min,
                step_size_max=step_size_max,
                etaminus=etaminus,
                etaplus=etaplus,
                foreach=foreach,
                maximize=maximize,
                differentiable=group["differentiable"],
                capturable=group["capturable"],
                has_complex=has_complex,
            )

        return loss


Rprop.__doc__ = (
    r"""Implements the resilient backpropagation algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \theta_0 \in \mathbf{R}^d \text{ (params)},f(\theta)
                \text{ (objective)},                                                             \\
            &\hspace{13mm}      \eta_{+/-} \text{ (etaplus, etaminus)}, \Gamma_{max/min}
                \text{ (step sizes)}                                                             \\
            &\textbf{initialize} :   g^0_{prev} \leftarrow 0,
                \: \eta_0 \leftarrow \text{lr (learning rate)}                                   \\
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm} \textbf{for} \text{  } i = 0, 1, \ldots, d-1 \: \mathbf{do}            \\
            &\hspace{10mm}  \textbf{if} \:   g^i_{prev} g^i_t  > 0                               \\
            &\hspace{15mm}  \eta^i_t \leftarrow \mathrm{min}(\eta^i_{t-1} \eta_{+},
                \Gamma_{max})                                                                    \\
            &\hspace{10mm}  \textbf{else if}  \:  g^i_{prev} g^i_t < 0                           \\
            &\hspace{15mm}  \eta^i_t \leftarrow \mathrm{max}(\eta^i_{t-1} \eta_{-},
                \Gamma_{min})                                                                    \\
            &\hspace{15mm}  g^i_t \leftarrow 0                                                   \\
            &\hspace{10mm}  \textbf{else}  \:                                                    \\
            &\hspace{15mm}  \eta^i_t \leftarrow \eta^i_{t-1}                                     \\
            &\hspace{5mm}\theta_t \leftarrow \theta_{t-1}- \eta_t \mathrm{sign}(g_t)             \\
            &\hspace{5mm}g_{prev} \leftarrow  g_t                                                \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to the paper
    `A Direct Adaptive Method for Faster Backpropagation Learning: The RPROP Algorithm
    <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.21.1417>`_.
    """
    + rf"""
    Args:
        {_params_doc}
        lr (float, optional): learning rate (default: 1e-2)
        etas (Tuple[float, float], optional): pair of (etaminus, etaplus), that
            are multiplicative increase and decrease factors
            (default: (0.5, 1.2))
        step_sizes (Tuple[float, float], optional): a pair of minimal and
            maximal allowed step sizes (default: (1e-6, 50))
        {_capturable_doc}
        {_foreach_doc}
        {_maximize_doc}
        {_differentiable_doc}

    """
)


def _single_tensor_rprop(
    params: List[Tensor],
    grads: List[Tensor],
    prevs: List[Tensor],
    step_sizes: List[Tensor],
    state_steps: List[Tensor],
    *,
    step_size_min: float,
    step_size_max: float,
    etaminus: float,
    etaplus: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    has_complex: bool,
):
    for i, param in enumerate(params):
        grad = grads[i]
        grad = grad if not maximize else -grad
        prev = prevs[i]
        step_size = step_sizes[i]
        step = state_steps[i]

        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if not torch.compiler.is_compiling() and capturable:
            capturable_supported_devices = _get_capturable_supported_devices()
            assert (
                param.device.type == step.device.type
                and param.device.type in capturable_supported_devices
            ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

        step += 1

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            prev = torch.view_as_real(prev)
            param = torch.view_as_real(param)
            step_size = torch.view_as_real(step_size)
        if differentiable:
            sign = grad.mul(prev.clone()).sign()
        else:
            sign = grad.mul(prev).sign()

        if capturable:
            sign.copy_(torch.where(sign.gt(0), etaplus, sign))
            sign.copy_(torch.where(sign.lt(0), etaminus, sign))
            sign.copy_(torch.where(sign.eq(0), 1, sign))
        else:
            sign[sign.gt(0)] = etaplus
            sign[sign.lt(0)] = etaminus
            sign[sign.eq(0)] = 1

        # update stepsizes with step size updates
        step_size.mul_(sign).clamp_(step_size_min, step_size_max)

        # for dir<0, dfdx=0
        # for dir>=0 dfdx=dfdx
        grad = grad.clone(memory_format=torch.preserve_format)
        if capturable:
            grad.copy_(torch.where(sign.eq(etaminus), 0, grad))
        else:
            grad[sign.eq(etaminus)] = 0

        # update parameters
        param.addcmul_(grad.sign(), step_size, value=-1)
        prev.copy_(grad)


def _multi_tensor_rprop(
    params: List[Tensor],
    grads: List[Tensor],
    prevs: List[Tensor],
    step_sizes: List[Tensor],
    state_steps: List[Tensor],
    *,
    step_size_min: float,
    step_size_max: float,
    etaminus: float,
    etaplus: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    has_complex: bool,
):
    if len(params) == 0:
        return

    assert not differentiable, "_foreach ops don't support autograd"

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if not torch.compiler.is_compiling() and capturable:
        capturable_supported_devices = _get_capturable_supported_devices()
        assert all(
            p.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, step in zip(params, state_steps)
        ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, prevs, step_sizes, state_steps]  # type: ignore[list-item]
    )
    for (
        grouped_params_,
        grouped_grads_,
        grouped_prevs_,
        grouped_step_sizes_,
        grouped_state_steps_,
    ), _ in grouped_tensors.values():
        grouped_params = cast(List[Tensor], grouped_params_)
        grouped_grads = cast(List[Tensor], grouped_grads_)
        grouped_prevs = cast(List[Tensor], grouped_prevs_)
        grouped_step_sizes = cast(List[Tensor], grouped_step_sizes_)
        grouped_state_steps = cast(List[Tensor], grouped_state_steps_)

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

        # Handle complex params
        if has_complex:
            _view_as_real(
                grouped_params, grouped_grads, grouped_prevs, grouped_step_sizes
            )

        signs = torch._foreach_mul(grouped_grads, grouped_prevs)
        if maximize:
            torch._foreach_neg_(signs)

        # At the end of the step, grouped_prevs will contain the current grads, so we reuse
        # grouped_prevs memory instead of creating a new buffer, but, for clarity, we reassign
        # to keep referring to the buffer as grouped_grads.
        torch._foreach_copy_(grouped_prevs, grouped_grads)
        if maximize:
            torch._foreach_neg_(grouped_prevs)
        grouped_grads = grouped_prevs

        torch._foreach_sign_(signs)
        if capturable:
            for sign in signs:
                sign.copy_(torch.where(sign.gt(0), etaplus, sign))
                sign.copy_(torch.where(sign.lt(0), etaminus, sign))
                sign.copy_(torch.where(sign.eq(0), 1, sign))
        else:
            for sign in signs:
                sign[sign.gt(0)] = etaplus
                sign[sign.lt(0)] = etaminus
                sign[sign.eq(0)] = 1

        # update stepsizes with step size updates
        torch._foreach_mul_(grouped_step_sizes, signs)
        for step_size in grouped_step_sizes:
            step_size.clamp_(step_size_min, step_size_max)

        # for dir<0, dfdx=0
        # for dir>=0 dfdx=dfdx
        grouped_grads = list(grouped_grads)
        for i in range(len(grouped_grads)):
            grouped_grads[i].copy_(
                torch.where(signs[i].eq(etaminus), 0, grouped_grads[i])
            )

        # explicitly del signs as it's not used after here to save memory
        del signs

        # update parameters
        grad_signs = [grad.sign() for grad in grouped_grads]
        torch._foreach_addcmul_(
            grouped_params, grad_signs, grouped_step_sizes, value=-1
        )

        # Logically, you may expect grouped_prevs to get updated to grouped_grads, but that's
        # basically already happened since we've been using grouped_prevs' memory to store
        # updated grouped_grads!


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_rprop)
def rprop(
    params: List[Tensor],
    grads: List[Tensor],
    prevs: List[Tensor],
    step_sizes: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: Optional[bool] = None,
    capturable: bool = False,
    maximize: bool = False,
    differentiable: bool = False,
    has_complex: bool = False,
    *,
    step_size_min: float,
    step_size_max: float,
    etaminus: float,
    etaplus: float,
):
    r"""Functional API that performs rprop algorithm computation.

    See :class:`~torch.optim.Rprop` for details.
    """
    # this check is slow during compilation, so we skip it
    # if it's strictly needed we can add this check back in dynamo
    if not torch.compiler.is_compiling() and not all(
        isinstance(t, torch.Tensor) for t in state_steps
    ):
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
        func = _multi_tensor_rprop
    else:
        func = _single_tensor_rprop

    func(
        params,
        grads,
        prevs,
        step_sizes,
        state_steps,
        step_size_min=step_size_min,
        step_size_max=step_size_max,
        etaminus=etaminus,
        etaplus=etaplus,
        capturable=capturable,
        maximize=maximize,
        differentiable=differentiable,
        has_complex=has_complex,
    )
