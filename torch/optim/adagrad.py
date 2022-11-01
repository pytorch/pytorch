import torch
from torch import Tensor

from .optimizer import Optimizer, _use_grad_for_differentiable
from typing import List, Optional

__all__ = ['Adagrad', 'adagrad']

class Adagrad(Optimizer):
    r"""Implements Adagrad algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)}, \: f(\theta)
                \text{ (objective)}, \: \lambda \text{ (weight decay)},                          \\
            &\hspace{12mm}    \tau \text{ (initial accumulator value)}, \: \eta\text{ (lr decay)}\\
            &\textbf{initialize} :  state\_sum_0 \leftarrow 0                             \\[-1.ex]
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

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
        foreach (bool, optional): whether foreach implementation of optimizer is used (default: None)
        maximize (bool, optional): maximize the params based on the objective, instead of
            minimizing (default: False)

    .. _Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization: http://jmlr.org/papers/v12/duchi11a.html
    """

    def __init__(
        self,
        params,
        lr=1e-2,
        lr_decay=0,
        weight_decay=0,
        initial_accumulator_value=0,
        eps=1e-10,
        foreach: Optional[bool] = None,
        *,
        maximize: bool = False,
        differentiable: bool = False
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError(
                "Invalid initial_accumulator_value value: {}".format(
                    initial_accumulator_value
                )
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(
            lr=lr,
            lr_decay=lr_decay,
            eps=eps,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            foreach=foreach,
            maximize=maximize,
            differentiable=differentiable
        )
        super(Adagrad, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.tensor(0.0)
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

    def share_memory(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["sum"].share_memory_()

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
            state_sums = []
            state_steps = []

            has_sparse_grad = False
            for p in group["params"]:
                if p.grad is not None:
                    if p.grad.is_sparse:
                        has_sparse_grad = True
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    state = self.state[p]
                    state_sums.append(state["sum"])
                    state_steps.append(state["step"])

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
                differentiable=group["differentiable"]
            )

        return loss


def adagrad(
    params: List[Tensor],
    grads: List[Tensor],
    state_sums: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting these as kwargs for now as functional API is compiled by torch/distributed/optim
    has_sparse_grad: bool = None,
    foreach: bool = None,
    differentiable: bool = False,
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

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    if foreach and not torch.jit.is_scripting():
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
        differentiable=differentiable
    )


def _make_sparse(grad, grad_indices, values):
    size = grad.size()
    if grad_indices.numel() == 0 or values.numel() == 0:
        return torch.empty_like(grad)
    return torch.sparse_coo_tensor(grad_indices, values, size)


def _single_tensor_adagrad(
    params: List[Tensor],
    grads: List[Tensor],
    state_sums: List[Tensor],
    state_steps: List[Tensor],
    *,
    lr: float,
    weight_decay: float,
    lr_decay: float,
    eps: float,
    has_sparse_grad: bool,
    maximize: bool,
    differentiable: bool
):

    for (param, grad, state_sum, step_t) in zip(params, grads, state_sums, state_steps):
        # update step
        step_t += 1
        grad = grad if not maximize else -grad

        if weight_decay != 0:
            if grad.is_sparse:
                raise RuntimeError(
                    "weight_decay option is not compatible with sparse gradients"
                )
            grad = grad.add(param, alpha=weight_decay)

        clr = lr / (1 + (step_t - 1) * lr_decay)

        if grad.is_sparse:
            grad = grad.coalesce()  # the update is non-linear so indices must be unique
            grad_indices = grad._indices()
            grad_values = grad._values()
            size = grad.size()

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
    params: List[Tensor],
    grads: List[Tensor],
    state_sums: List[Tensor],
    state_steps: List[Tensor],
    *,
    lr: float,
    weight_decay: float,
    lr_decay: float,
    eps: float,
    has_sparse_grad: bool,
    maximize: bool,
    differentiable: bool
):

    assert not differentiable, "_foreach ops don't support autograd"

    # Foreach functions will throw errors if given empty lists
    if len(params) == 0:
        return

    if maximize:
        grads = torch._foreach_neg(grads)

    if has_sparse_grad is None:
        has_sparse_grad = any(grad.is_sparse for grad in grads)

    if has_sparse_grad:
        return _single_tensor_adagrad(
            params,
            grads,
            state_sums,
            state_steps,
            lr=lr,
            weight_decay=weight_decay,
            lr_decay=lr_decay,
            eps=eps,
            has_sparse_grad=has_sparse_grad,
            maximize=False,
            differentiable=differentiable,
        )

    # Update steps
    torch._foreach_add_(state_steps, 1)

    if weight_decay != 0:
        torch._foreach_add_(grads, params, alpha=weight_decay)

    minus_clr = [-lr / (1 + (step - 1) * lr_decay) for step in state_steps]

    grads = [torch.view_as_real(x) if torch.is_complex(x) else x for x in grads]
    state_sums = [
        torch.view_as_real(x) if torch.is_complex(x) else x for x in state_sums
    ]
    torch._foreach_addcmul_(state_sums, grads, grads, value=1)
    std = torch._foreach_add(torch._foreach_sqrt(state_sums), eps)
    toAdd = torch._foreach_div(torch._foreach_mul(grads, minus_clr), std)
    toAdd = [
        torch.view_as_complex(x) if torch.is_complex(params[i]) else x
        for i, x in enumerate(toAdd)
    ]
    torch._foreach_add_(params, toAdd)
    state_sums = [
        torch.view_as_complex(x) if torch.is_complex(params[i]) else x
        for i, x in enumerate(state_sums)
    ]
