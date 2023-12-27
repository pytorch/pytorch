import torch
from torch import Tensor

from .optimizer import (Optimizer, _use_grad_for_differentiable, _default_to_fused_or_foreach,
                        _differentiable_doc, _foreach_doc, _maximize_doc, _view_as_real)
from typing import List, Optional

__all__ = ["Adadelta", "adadelta"]


class Adadelta(Optimizer):
    def __init__(
        self,
        params,
        lr=1.0,
        rho=0.9,
        eps=1e-6,
        weight_decay=0,
        foreach: Optional[bool] = None,
        *,
        maximize: bool = False,
        differentiable: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"Invalid rho value: {rho}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            rho=rho,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)

    def _init_group(self, group, params_with_grad, grads, square_avgs, acc_deltas):
        has_complex = False
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
                state["step"] = 0
                state["square_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                state["acc_delta"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )

            square_avgs.append(state["square_avg"])
            acc_deltas.append(state["acc_delta"])

            state["step"] += 1
        return has_complex

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
            params_with_grad = []
            grads = []
            square_avgs = []
            acc_deltas = []
            lr, rho, eps, weight_decay, foreach, maximize, differentiable = (
                group["lr"],
                group["rho"],
                group["eps"],
                group["weight_decay"],
                group["foreach"],
                group["maximize"],
                group["differentiable"],
            )

            has_complex = self._init_group(group, params_with_grad, grads, square_avgs, acc_deltas)

            adadelta(
                params_with_grad,
                grads,
                square_avgs,
                acc_deltas,
                lr=lr,
                rho=rho,
                eps=eps,
                weight_decay=weight_decay,
                foreach=foreach,
                maximize=maximize,
                differentiable=differentiable,
                has_complex=has_complex,
            )

        return loss


Adadelta.__doc__ = r"""Implements Adadelta algorithm.

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
    """ + fr"""
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rho (float, optional): coefficient used for computing a running average
            of squared gradients (default: 0.9). A higher value of `rho` will
            result in a slower average, which can be helpful for preventing
            oscillations in the learning process.
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-6).
        lr (float, optional): coefficient that scale delta before it is applied
            to the parameters (default: 1.0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        {_foreach_doc}
        {_maximize_doc}
        {_differentiable_doc}

    .. _ADADELTA\: An Adaptive Learning Rate Method:
        https://arxiv.org/abs/1212.5701

    """


def adadelta(
    params: List[Tensor],
    grads: List[Tensor],
    square_avgs: List[Tensor],
    acc_deltas: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
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
    # We still respect when the user inputs False for foreach.
    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)

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
        lr=lr,
        rho=rho,
        eps=eps,
        weight_decay=weight_decay,
        maximize=maximize,
        differentiable=differentiable,
        has_complex=has_complex,
    )


def _single_tensor_adadelta(
    params: List[Tensor],
    grads: List[Tensor],
    square_avgs: List[Tensor],
    acc_deltas: List[Tensor],
    *,
    lr: float,
    rho: float,
    eps: float,
    weight_decay: float,
    maximize: bool,
    differentiable: bool,
    has_complex: bool,
):

    for (param, grad, square_avg, acc_delta) in zip(
        params, grads, square_avgs, acc_deltas
    ):
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
    params: List[Tensor],
    grads: List[Tensor],
    square_avgs: List[Tensor],
    acc_deltas: List[Tensor],
    *,
    lr: float,
    weight_decay: float,
    rho: float,
    eps: float,
    maximize: bool,
    differentiable: bool,
    has_complex: bool,
):

    assert not differentiable, "_foreach ops don't support autograd"

    if len(params) == 0:
        return

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([params, grads, square_avgs, acc_deltas])
    for ((device_params, device_grads, device_square_avgs, device_acc_deltas), _) in grouped_tensors.values():
        if maximize:
            device_grads = torch._foreach_neg(device_grads)

        if has_complex:
            _view_as_real(device_params, device_grads, device_square_avgs, device_acc_deltas)

        if weight_decay != 0:
            # Re-use the intermediate memory (device_grads) already allocated for maximize
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = torch._foreach_add(device_grads, device_params, alpha=weight_decay)

        torch._foreach_mul_(device_square_avgs, rho)
        torch._foreach_addcmul_(device_square_avgs, device_grads, device_grads, value=1 - rho)

        std = torch._foreach_add(device_square_avgs, eps)
        torch._foreach_sqrt_(std)

        deltas = torch._foreach_add(device_acc_deltas, eps)
        torch._foreach_sqrt_(deltas)
        torch._foreach_div_(deltas, std)
        torch._foreach_mul_(deltas, device_grads)

        torch._foreach_add_(device_params, deltas, alpha=-lr)

        torch._foreach_mul_(device_acc_deltas, rho)
        torch._foreach_addcmul_(device_acc_deltas, deltas, deltas, value=1 - rho)
