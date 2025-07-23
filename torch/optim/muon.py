# mypy: allow-untyped-defs
from typing import Optional, Union

import torch
from torch import Tensor

from .optimizer import (
    _capturable_doc,
    _default_to_fused_or_foreach,
    _differentiable_doc,
    _disable_dynamo_if_unsupported,
    _foreach_doc,
    _fused_doc,
    _get_scalar_dtype,
    _get_value,
    _maximize_doc,
    _params_doc,
    _stack_if_compiling,
    _to_scalar,
    _use_grad_for_differentiable,
    _view_as_real,
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


def _single_tensor_muon(
    param: Tensor,
    grad: Tensor,
    momentum_buf: Tensor,
    *,
    lr: Union[float, Tensor],
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_steps: int,
    maximize: bool,
):
    r"""Functional API that performs a single tensor Muon algorithm computation."""
    
    grad = grad if not maximize else -grad
    
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
    
    # Reshape for 2D if needed (conv filters)
    original_shape = update.shape
    if update.ndim == 4:
        update = update.view(len(update), -1)
    elif update.ndim < 2:
        # Skip orthogonalization for 1D parameters
        param.add_(update, alpha=-lr)
        return
    
    # Newton-Schulz orthogonalization
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
    *,
    lr: Union[float, Tensor],
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_steps: int,
    maximize: bool,
):
    r"""Functional API that performs Muon algorithm computation in a multi-tensor fashion."""
    
    if len(params) == 0:
        return
    
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        momentum_buf = momentum_bufs[i]
        
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
        
        # Reshape for 2D if needed (conv filters)
        original_shape = update.shape
        if update.ndim == 4:
            update = update.view(len(update), -1)
        elif update.ndim < 2:
            # Skip orthogonalization for 1D parameters
            param.add_(update, alpha=-lr)
            continue
        
        # Newton-Schulz orthogonalization
        if update.ndim >= 2:
            update = _zeropower_via_newtonschulz5(update, steps=ns_steps)
            # Scale factor
            scale = max(1, update.size(-2) / update.size(-1))**0.5
            update = update * scale
        
        # Reshape back to original shape
        update = update.reshape(original_shape)
        
        # Apply update
        param.add_(update, alpha=-lr)

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
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
    ):
        if isinstance(lr, Tensor):
            if foreach and not capturable:
                raise ValueError(
                    "lr as a Tensor is not supported for capturable=False and foreach=True"
                )
            if lr.numel() != 1:
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
            "maximize": maximize,
            "foreach": foreach,
            "capturable": capturable,
            "differentiable": differentiable,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", True)
            group.setdefault("ns_steps", 5)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        momentum_bufs,
    ):
        for p in group["params"]:
            if p.grad is not None:
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                
                momentum_bufs.append(state["momentum_buffer"])

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
            momentum_bufs: list[Tensor] = []

            self._init_group(
                group,
                params_with_grad,
                grads,
                momentum_bufs,
            )

            muon(
                params_with_grad,
                grads,
                momentum_bufs,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                nesterov=group["nesterov"],
                ns_steps=group["ns_steps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
            )

        return loss


def muon(
    params: list[Tensor],
    grads: list[Tensor],
    momentum_bufs: list[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
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
    
    # Respect when the user inputs False/True for foreach. We only want to change
    # the default when it has not been user-specified.
    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )
        # Do not flip on foreach for the unsupported case where lr is a Tensor and capturable=False.
        if foreach and isinstance(lr, Tensor) and not capturable:
            foreach = False
    if foreach is None:
        foreach = False

    # this check is slow during compilation, so we skip it
    # if it's strictly needed we can add this check back in dynamo
    if not torch.compiler.is_compiling() and not all(
        isinstance(t, torch.Tensor) for t in momentum_bufs
    ):
        raise RuntimeError(
            "API has changed, `momentum_bufs` argument must be a list of tensors"
        )

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    
    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_muon
    else:
        func = _single_tensor_muon

    if len(params) == 1:
        func(
            params[0],
            grads[0],
            momentum_bufs[0],
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            maximize=maximize,
        )
    else:
        func(
            params,
            grads,
            momentum_bufs,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            maximize=maximize,
        )


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
        {_foreach_doc}
        {_capturable_doc}
        {_differentiable_doc}
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
