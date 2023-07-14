import torch
from torch import Tensor

from .optimizer import (Optimizer, _use_grad_for_differentiable, _get_value, _stack_if_compiling,
                        _default_to_fused_or_foreach, _differentiable_doc, _maximize_doc, _foreach_doc)
from typing import List, Optional

__all__ = ["Adamax", "adamax"]


class Adamax(Optimizer):
    def __init__(
        self,
        params,
        lr=2e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        foreach: Optional[bool] = None,
        *,
        maximize: bool = False,
        differentiable: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            foreach=foreach,
            maximize=maximize,
            differentiable=differentiable,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

    def _init_group(self, group, params_with_grad, grads, exp_avgs, exp_infs, state_steps):
        for p in group["params"]:
            if p.grad is None:
                continue
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("Adamax does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state["step"] = torch.tensor(0.0)
                state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                state["exp_inf"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )

            exp_avgs.append(state["exp_avg"])
            exp_infs.append(state["exp_inf"])
            state_steps.append(state["step"])

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

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
            exp_avgs = []
            exp_infs = []
            state_steps = []

            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            foreach = group["foreach"]
            maximize = group["maximize"]
            differentiable = group["differentiable"]

            self._init_group(group, params_with_grad, grads, exp_avgs, exp_infs, state_steps)

            adamax(
                params_with_grad,
                grads,
                exp_avgs,
                exp_infs,
                state_steps,
                eps=eps,
                beta1=beta1,
                beta2=beta2,
                lr=lr,
                weight_decay=weight_decay,
                foreach=foreach,
                maximize=maximize,
                differentiable=differentiable,
            )

        return loss


Adamax.__doc__ = r"""Implements Adamax algorithm (a variant of Adam based on infinity norm).

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \beta_1, \beta_2
                \text{ (betas)},\theta_0 \text{ (params)},f(\theta) \text{ (objective)},
                \: \lambda \text{ (weight decay)},                                                \\
            &\hspace{13mm}    \epsilon \text{ (epsilon)}                                          \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                u_0 \leftarrow 0 \text{ ( infinity norm)}                                 \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}if \: \lambda \neq 0                                                    \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}m_t      \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t               \\
            &\hspace{5mm}u_t      \leftarrow   \mathrm{max}(\beta_2 u_{t-1}, |g_{t}|+\epsilon)   \\
            &\hspace{5mm}\theta_t \leftarrow \theta_{t-1} - \frac{\gamma m_t}{(1-\beta^t_1) u_t} \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `Adam: A Method for Stochastic Optimization`_.
    """ + r"""
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 2e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        {foreach}
        {maximize}
        {differentiable}

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980

    """.format(foreach=_foreach_doc, maximize=_maximize_doc, differentiable=_differentiable_doc)


def adamax(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_infs: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: Optional[bool] = None,
    maximize: bool = False,
    differentiable: bool = False,
    *,
    eps: float,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
):
    r"""Functional API that performs adamax algorithm computation.

    See :class:`~torch.optim.Adamax` for details.
    """

    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adamax
    else:
        func = _single_tensor_adamax

    func(
        params,
        grads,
        exp_avgs,
        exp_infs,
        state_steps,
        eps=eps,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        maximize=maximize,
        differentiable=differentiable,
    )


def _single_tensor_adamax(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_infs: List[Tensor],
    state_steps: List[Tensor],
    *,
    eps: float,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    maximize: bool,
    differentiable: bool,
):

    for i, param in enumerate(params):
        grad = grads[i]
        grad = grad if not maximize else -grad
        exp_avg = exp_avgs[i]
        exp_inf = exp_infs[i]
        step_t = state_steps[i]
        # update step
        step_t += 1

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        if torch.is_complex(param):
            param = torch.view_as_real(param)
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_inf = torch.view_as_real(exp_inf)

        # Update biased first moment estimate.
        exp_avg.lerp_(grad, 1 - beta1)
        # Update the exponentially weighted infinity norm.
        norm_buf = torch.cat(
            [exp_inf.mul_(beta2).unsqueeze(0), grad.abs().add_(eps).unsqueeze_(0)], 0
        )

        if not differentiable:
            torch.amax(norm_buf, 0, keepdim=False, out=exp_inf)
        else:
            exp_inf.copy_(torch.amax(norm_buf, 0, keepdim=False))

        bias_correction = 1 - beta1 ** _get_value(step_t)
        clr = lr / bias_correction

        param.addcdiv_(exp_avg, exp_inf, value=-clr)


def _multi_tensor_adamax(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_infs: List[Tensor],
    state_steps: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
    differentiable: bool,
):

    assert not differentiable, "_foreach ops don't support autograd"

    if len(params) == 0:
        return

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([params, grads, exp_avgs, exp_infs, state_steps])
    for ((grouped_params, grouped_grads, grouped_exp_avgs, grouped_exp_infs, grouped_state_steps), _) in grouped_tensors.values():
        if maximize:
            grouped_grads = torch._foreach_neg(grouped_grads)

        grouped_params = [torch.view_as_real(x) if torch.is_complex(x) else x for x in grouped_params]
        grouped_grads = [torch.view_as_real(x) if torch.is_complex(x) else x for x in grouped_grads]
        grouped_exp_avgs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in grouped_exp_avgs]
        grouped_exp_infs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in grouped_exp_infs]

        # Update steps
        torch._foreach_add_(grouped_state_steps, 1)

        if weight_decay != 0:
            if maximize:
                # Re-use the intermediate memory (device_grads) already allocated for maximize
                torch._foreach_add_(grouped_grads, grouped_params, alpha=weight_decay)
            else:
                grouped_grads = torch._foreach_add(grouped_grads, grouped_params, alpha=weight_decay)

        # Update biased first moment estimate.
        torch._foreach_lerp_(grouped_exp_avgs, grouped_grads, 1 - beta1)

        # Update the exponentially weighted infinity norm.
        torch._foreach_mul_(grouped_exp_infs, beta2)

        for exp_inf, grad in zip(grouped_exp_infs, grouped_grads):
            norm_buf = torch.cat(
                [exp_inf.unsqueeze(0), grad.abs().add_(eps).unsqueeze_(0)], 0
            )
            torch.max(norm_buf, 0, keepdim=False, out=(exp_inf, exp_inf.new().long()))

        bias_corrections = [1 - beta1 ** _get_value(step) for step in grouped_state_steps]
        clr = _stack_if_compiling([-1 * (lr / bias_correction) for bias_correction in bias_corrections])

        torch._foreach_addcdiv_(grouped_params, grouped_exp_avgs, grouped_exp_infs, clr)
