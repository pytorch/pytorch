# mypy: allow-untyped-defs
import torch
import torch.distributed as dist
from torch import Tensor
from .optimizer import (
    Optimizer,
    _params_doc,
    _maximize_doc,
    _foreach_doc,
    _capturable_doc,
    _differentiable_doc,
    _fused_doc,
)

__all__ = ["Muon"]

def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(
        self,
        params,
        lr=0.02,
        weight_decay=0.01,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        rank=None,
        world_size=None,
    ):
        if (rank is None) or (world_size is None):
            raise Exception("world_size and rank params required, if you want to use this optimizer on a single GPU, pass rank=0 and world_size=1.")
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(params=[p for p in params if p.numel() == size],
                         update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None
            def update_prev(): # optimized Muon implementation contributed by @YouJiacheng
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.mul_(1 - group["lr"] * group["weight_decay"])
                    p_world.add_(g_world.view_as(p_world),
                                 alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5)
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    if g.ndim == 4: # for the case of conv filters
                        g = g.view(len(g), -1)
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()

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
            &\hspace{5mm}B_t \leftarrow \mu B_{t-1} + G_t \: \text{(or Nesterov variant if enabled)}          \\
            &\hspace{5mm}O_t \leftarrow \text{NewtonSchulz5}(B_t)                                \\
            &\hspace{5mm}\text{Update parameters} \: \theta_t \leftarrow \theta_{t-1}(1 - \eta\lambda) - \eta \beta O_t  \\
            &\hspace{5mm}\text{where} \: \beta = \sqrt{\max(1, \frac{d_{\text{row}}}{d_{\text{col}}})}        \\
            &\textbf{end for}                                                                    \\
            &\textbf{return} \: \theta_t                                                         \\
            &\rule{110mm}{0.4pt}                                                                 \\
       \end{aligned}

    Muon is an optimizer for 2D parameters of neural network hidden layers. It internally runs 
    standard SGD-momentum, and then performs an orthogonalization post-processing step using 
    Newton-Schulz iterations, replacing each 2D parameter's update with the nearest orthogonal matrix.
    """
    + rf"""
    Args:
        {_params_doc}
        lr (float): learning rate (default: 0.02)
        weight_decay (float): weight decay (L2 penalty) (default: 0.01)
        momentum (float): momentum factor (default: 0.95)
        nesterov (bool): enables Nesterov momentum (default: True)
        ns_steps (int): number of Newton-Schulz iteration steps (default: 5)
        rank (int): process rank for distributed training
        world_size (int): number of processes for distributed training
        {_maximize_doc}
        {_foreach_doc}
        {_capturable_doc}
        {_differentiable_doc}
        {_fused_doc}
    """
    + r"""

    Warning:
        This optimizer should not be used for the embedding layer, the final fully connected layer,
        or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
        To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Example:
        >>> # xdoctest: +SKIP
        >>> optimizer = torch.optim.Muon(model.parameters(), lr=0.02, momentum=0.95, 
        ...                              rank=0, world_size=1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    Reference:
        https://kellerjordan.github.io/posts/muon/
    """
)
