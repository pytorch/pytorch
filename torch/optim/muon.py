# mypy: allow-untyped-defs
from typing import Optional, Union

import torch
from torch import Tensor

from .optimizer import (
    _disable_dynamo_if_unsupported,
    _get_scalar_dtype,  
    _maximize_doc,
    _params_doc,
    _to_scalar,
    Optimizer,
    ParamsT,
)

__all__ = ["Muon", "muon"]


def _zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 0.02,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
    ):
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not isinstance(ns_steps, int) or ns_steps < 1:
            raise ValueError(f"Invalid ns_steps value: {ns_steps}")

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "foreach": foreach,
            "maximize": maximize,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", True)
            group.setdefault("ns_steps", 5)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = torch.tensor(step_val, dtype=_get_scalar_dtype())

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        momentum_bufs,
        state_steps,
    ):
        for p in group["params"]:
            if p.grad is None:
                continue
            if torch.is_complex(p):
                raise RuntimeError("Muon does not support complex parameters")
            if p.grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")

            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.
                # This is because kernel launches are costly on CUDA and XLA.
                state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())
                # Momentum buffer
                state["momentum_buffer"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )

            momentum_bufs.append(state["momentum_buffer"])
            state_steps.append(state["step"])
        return False  # has_complex

    @torch.no_grad()
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
            momentum_bufs: list[Tensor] = []
            state_steps: list[Tensor] = []

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                momentum_bufs,
                state_steps,
            )

            muon(
                params_with_grad,
                grads,
                momentum_bufs,
                state_steps,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                nesterov=group["nesterov"],
                ns_steps=group["ns_steps"],
                foreach=group["foreach"],
                maximize=group["maximize"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                has_complex=has_complex,
            )

        return loss


def muon(
    params: list[Tensor],
    grads: list[Tensor],
    momentum_bufs: list[Tensor],
    state_steps: list[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    has_complex: bool = False,
    *,
    lr: Union[float, Tensor],
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_steps: int,
    maximize: bool,
):
    r"""Functional API that performs Muon algorithm computation.
    See :class:`~torch.optim.Muon` for details.
    """
    if not torch.compiler.is_compiling() and not all(
        isinstance(t, torch.Tensor) for t in state_steps
    ):
        raise RuntimeError(
            "`state_steps` argument must contain a list of singleton tensors"
        )

    if foreach is None:
        if torch.jit.is_scripting():
            foreach = False
        else:
            foreach = len(params) > 0 and all(
                torch.is_same_size(params[0], p) for p in params[1:]
            )

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    
    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_muon
    else:
        func = _single_tensor_muon

    func(
        params,
        grads,
        momentum_bufs,
        state_steps,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        ns_steps=ns_steps,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
        has_complex=has_complex,
    )


def _single_tensor_muon(
    params: list[Tensor],
    grads: list[Tensor],
    momentum_bufs: list[Tensor],
    state_steps: list[Tensor],
    *,
    lr: Union[float, Tensor],
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_steps: int,
    maximize: bool,
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    has_complex: bool,
):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        momentum_buf = momentum_bufs[i]
        step_t = state_steps[i]
        
        # update step
        step_t += 1

        # Apply weight decay
        if weight_decay != 0:
            param.mul_(1 - lr * weight_decay)

        # Update momentum buffer
        momentum_buf.lerp_(grad, 1 - momentum)

        # Choose update based on nesterov
        if nesterov:
            update = grad.lerp_(momentum_buf, momentum)
        else:
            update = momentum_buf

        # Handle different tensor dimensions
        original_shape = update.shape
        if update.ndim == 4:
            # 4D convolutional filters: flatten last 3 dimensions
            update = update.view(len(update), -1)
        elif update.ndim < 2:
            # 1D parameters: skip orthogonalization
            param.add_(update, alpha=-lr)
            continue

        # Newton-Schulz orthogonalization for 2D+ tensors
        if update.ndim >= 2:
            update = _zeropower_via_newtonschulz5(update, steps=ns_steps)
            # Scale factor
            scale = max(1, update.size(-2) / update.size(-1))**0.5
            update = update * scale

        # Reshape back to original shape
        update = update.reshape(original_shape)

        # Apply update
        param.add_(update, alpha=-lr)


def _multi_tensor_muon(
    params: list[Tensor],
    grads: list[Tensor],
    momentum_bufs: list[Tensor],
    state_steps: list[Tensor],
    *,
    lr: Union[float, Tensor],
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_steps: int,
    maximize: bool,
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    has_complex: bool,
):
    if len(params) == 0:
        return

    # Update steps
    torch._foreach_add_(state_steps, 1)

    # Handle maximize
    if maximize:
        grads = torch._foreach_neg(grads)

    # Apply weight decay
    if weight_decay != 0:
        torch._foreach_mul_(params, 1 - lr * weight_decay)

    # Update momentum buffers
    torch._foreach_lerp_(momentum_bufs, grads, 1 - momentum)

    # Choose updates based on nesterov
    if nesterov:
        updates = torch._foreach_lerp(grads, momentum_bufs, momentum)
    else:
        updates = [buf.clone() for buf in momentum_bufs]

    # Process each parameter individually for orthogonalization
    for i, (param, update) in enumerate(zip(params, updates)):
        original_shape = update.shape
        if update.ndim == 4:
            # 4D convolutional filters: flatten last 3 dimensions
            update = update.view(len(update), -1)
        elif update.ndim < 2:
            # 1D parameters: skip orthogonalization
            param.add_(update, alpha=-lr)
            continue

        # Newton-Schulz orthogonalization for 2D+ tensors
        if update.ndim >= 2:
            update = _zeropower_via_newtonschulz5(update, steps=ns_steps)
            # Scale factor
            scale = max(1, update.size(-2) / update.size(-1))**0.5
            update = update * scale

        # Reshape back to original shape
        update = update.reshape(original_shape)
        updates[i] = update

    # Apply updates
    torch._foreach_add_(params, updates, alpha=-lr)


Muon.__doc__ = (
    r"""Implements Momentum Orthogonalized by Newton-schulz (Muon).

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{Require:} \: \eta \text{ (learning rate)}, \: \mu \text{ (momentum)}, \: \lambda \text{ (weight decay)}, 
                \: s \text{ (ns\_steps)}  \\
            &\textbf{Initialize:} \: B_0 \leftarrow 0                                            \\
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}\text{Compute gradient} \: G_t \leftarrow \nabla_{\theta}\mathcal{L}_t(\theta_{t-1})  \\
            &\hspace{5mm}B_t \leftarrow \mu B_{t-1} + (1-\mu) G_t \: \text{(or Nesterov variant if enabled)}          \\
            &\hspace{5mm}O_t \leftarrow \text{NewtonSchulz5}(B_t)                                \\
            &\hspace{5mm}\text{Update parameters} \: \theta_t \leftarrow \theta_{t-1}(1 - \eta\lambda) - \eta \beta O_t  \\
            &\hspace{5mm}\text{where} \: \beta = \sqrt{\max(1, \frac{d_{\text{row}}}{d_{\text{col}}})}        \\
            &\textbf{end for}                                                                    \\
            &\textbf{return} \: \theta_t                                                         \\
            &\rule{110mm}{0.4pt}                                                                 \\
       \end{aligned}

    Muon is an optimizer designed for 2D parameters of neural network hidden layers. It internally runs 
    standard SGD-momentum, and then performs an orthogonalization post-processing step using 
    Newton-Schulz iterations, replacing each 2D parameter's update with the nearest orthogonal matrix.
    This helps maintain the orthogonality of weight matrices during training.
    """
    + rf"""
    Args:
        {_params_doc}
        lr (float, Tensor, optional): learning rate (default: 0.02)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.01)
        momentum (float, optional): momentum factor (default: 0.95)
        nesterov (bool, optional): enables Nesterov momentum (default: True)
        ns_steps (int, optional): number of Newton-Schulz iteration steps (default: 5)
        {_maximize_doc}
        foreach (bool, optional): whether to use the foreach implementation
            (default: None)
    """
    + r"""

    .. warning::
        This optimizer should not be used for the embedding layer, the final fully connected layer,
        or any 1-D parameters; those should all be optimized by a standard method (e.g., AdamW).
        For 4D convolutional filters, the optimizer handles them by flattening their last 3 dimensions.

    .. note::
        Reference paper: `Muon: MomentUm Orthogonalized by Newton-schulz <https://kellerjordan.github.io/posts/muon/>`_

    Example:
        >>> # xdoctest: +SKIP
        >>> optimizer = torch.optim.Muon(model.parameters(), lr=0.02)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """
)