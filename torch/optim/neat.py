# mypy: allow-untyped-defs
from typing import cast, Optional, Union
import torch
from torch import Tensor
from .optimizer import (
    _capturable_doc,
    _default_to_fused_or_foreach,
    _differentiable_doc,
    _disable_dynamo_if_unsupported,
    _foreach_doc,
    _get_scalar_dtype,
    _get_value,
    _maximize_doc,
    _params_doc,
    _to_scalar,
    _use_grad_for_differentiable,
    Optimizer,
    ParamsT,
)

__all__ = ["NEAT", "neat"]


class NEAT(Optimizer):
    """Implements NEAT (Nash-Equilibrium Adaptive Training) optimizer.
    
    This optimizer applies game-theoretic principles to neural network optimization,
    treating each layer as a rational agent in a cooperative game. It extends Adam
    with Nash Gradient Equilibrium to reduce gradient conflicts between layers.
    
    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-3)
        betas: coefficients for computing running averages (default: (0.9, 0.999))
        eps: term added to denominator for numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
        mu: equilibrium parameter controlling cooperative strength (default: 0.5)
        foreach: whether to use foreach implementation (default: None)
        maximize: maximize the parameters based on objective (default: False)
        capturable: whether algorithm can be captured in CUDA graph (default: False)
        differentiable: whether autograd should record operations (default: False)
    """

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        mu: float = 0.5,
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = False,
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
        if not 0.0 <= mu <= 1.0:
            raise ValueError(f"Invalid mu parameter: {mu}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "mu": mu,
            "maximize": maximize,
            "foreach": foreach,
            "capturable": capturable,
            "differentiable": differentiable,
        }
        super().__init__(params, defaults)
        self._step_supports_amp_scaling = True

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("mu", 0.5)

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
    ):
        for p in group["params"]:
            if p.grad is not None:
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError(
                        "NEAT does not support sparse gradients"
                    )
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = (
                        torch.zeros(
                            (),
                            dtype=_get_scalar_dtype(),
                            device=p.device,
                        )
                        if group["capturable"]
                        else torch.tensor(0.0, dtype=_get_scalar_dtype())
                    )
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                state_steps.append(state["step"])

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step."""
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

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
            )

            neat(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                mu=group["mu"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
            )

        return loss


def _construct_laplacian(num_params: int, device) -> Tensor:
    """Construct graph Laplacian for layer connectivity."""
    # Create adjacency matrix for sequential layer connections
    adj = torch.zeros(num_params, num_params, device=device)
    for i in range(num_params - 1):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    
    # Compute degree matrix
    degree = torch.diag(adj.sum(dim=1))
    
    # Laplacian = D - A
    laplacian = degree - adj
    return laplacian


def _single_tensor_neat(
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
    mu: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
) -> None:
    """Single tensor NEAT update."""
    # Apply Nash Gradient Equilibrium
    if len(grads) > 1 and mu > 0:
        # Stack gradients
        grad_stack = torch.stack([g.flatten() for g in grads])
        
        # Construct Laplacian
        laplacian = _construct_laplacian(len(grads), grad_stack.device)
        
        # Apply equilibrium correction: (I - mu * L) @ G
        identity = torch.eye(len(grads), device=grad_stack.device)
        equilibrium_matrix = identity - mu * laplacian
        corrected_grads = equilibrium_matrix @ grad_stack
        
        # Reshape back to original shapes
        for i, (grad, orig_grad) in enumerate(zip(grads, grads)):
            grads[i] = corrected_grads[i].reshape(orig_grad.shape)
    
    # Standard Adam update with equilibrium-corrected gradients
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        step_t += 1

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if capturable or differentiable:
            step = step_t
            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step
            step_size = lr / bias_correction1
            bias_correction2_sqrt = bias_correction2.sqrt()
            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
            param.addcdiv_(exp_avg, denom, value=-step_size)
        else:
            step = _get_value(step_t)
            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step
            step_size = lr / bias_correction1
            bias_correction2_sqrt = bias_correction2**0.5
            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
            param.addcdiv_(exp_avg, denom, value=-step_size)


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_neat)
def neat(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    mu: float,
    eps: float,
    maximize: bool,
) -> None:
    r"""Functional API for NEAT algorithm."""
    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )
    
    if foreach is None:
        foreach = False

    if not torch.compiler.is_compiling() and not all(
        isinstance(t, torch.Tensor) for t in state_steps
    ):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    func = _single_tensor_neat

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
        mu=mu,
        eps=eps,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
    )
