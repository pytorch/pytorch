# mypy: allow-untyped-defs
# mypy: disable-error-code=arg-type
"""Implementation of the Muon optimizer."""

import math
from collections.abc import MutableMapping

import torch
from torch import Tensor

from .optimizer import (
    _disable_dynamo_if_unsupported,
    _params_doc,
    _to_scalar,
    Optimizer,
    ParamsT,
)


__all__ = ["Muon"]

# Constants from Keller Jordan's Muon post: https://kellerjordan.github.io/posts/muon/
# github permlink: https://github.com/KellerJordan/Muon/blob/f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd/muon.py#L16
EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315
DEFAULT_NS_STEPS = 5


def _zeropower_via_newtonschulz(
    grad: Tensor, ns_coefficients: tuple[float, float, float], ns_steps: int, eps: float
) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.

    Implementation reference: https://github.com/KellerJordan/Muon/blob/master/muon.py
    with suggestions by @jxbz, @leloykun, and @YouJiacheng.
    """
    if ns_steps >= 100:
        raise ValueError(
            "Number of steps must be less than 100 for computational efficiency"
        )
    if len(grad.shape) != 2:
        raise ValueError("Input tensor gradient must be a 2D matrix")
    if len(ns_coefficients) != 3:
        raise ValueError("Coefficients must be a tuple of exactly 3 values")
    a, b, c = ns_coefficients
    ortho_grad = grad.bfloat16()
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    # Ensure spectral norm is at most 1
    ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
    # Perform the NS iterations
    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(
            gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c
        )
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)

    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    return ortho_grad


def _adjust_lr(lr: float, adjust_lr_fn: str | None, param_shape: torch.Size) -> float:
    """Default learning rate adjustment used by Muon."""
    A, B = param_shape[:2]

    if adjust_lr_fn is None or adjust_lr_fn == "original":
        # pyrefly: ignore [no-matching-overload]
        adjusted_ratio = math.sqrt(max(1, A / B))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    elif adjust_lr_fn == "spectral_unclamped":
        adjusted_ratio = math.sqrt(A / B)
    else:
        adjusted_ratio = 1.0
    return lr * adjusted_ratio


class Muon(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
        eps: float = EPS,
        ns_steps: int = DEFAULT_NS_STEPS,
        adjust_lr_fn: str | None = None,
        foreach: bool | None = None,
    ) -> None:
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")
        if adjust_lr_fn is not None and adjust_lr_fn not in [
            "original",
            "match_rms_adamw",
            "spectral_unclamped",
        ]:
            raise ValueError(
                f"Adjust learning rate function {adjust_lr_fn} is not supported"
            )

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
            "foreach": foreach,
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon only supports 2D parameters whereas we found a parameter with size: {p.size()}"
                    )

    def __setstate__(self, state):
        super().__setstate__(state)
        # Backfill defaults for flags introduced after the original release so
        # that state dicts saved by older versions of Muon load cleanly.
        for group in self.param_groups:
            group.setdefault("foreach", None)

    def _init_group(
        self,
        group: MutableMapping,
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        muon_momentum_bufs: list[Tensor],
    ) -> bool:
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

            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(
                    p.grad, memory_format=torch.preserve_format
                )
            muon_momentum_bufs.append(state["momentum_buffer"])

        return False  # has_complex

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            muon_momentum_bufs: list[Tensor] = []

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                muon_momentum_bufs,
            )

            muon(
                params_with_grad,
                grads,
                muon_momentum_bufs,
                foreach=group["foreach"],
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                nesterov=group["nesterov"],
                ns_coefficients=group["ns_coefficients"],
                eps=group["eps"],
                ns_steps=group["ns_steps"],
                adjust_lr_fn=group["adjust_lr_fn"],
                has_complex=has_complex,
            )
        return loss


Muon.__doc__ = (
    r"""Implements Muon algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt} \\
            &\textbf{input}      : \gamma \text{ (lr)},\ \lambda \text{ (weight decay)},\
               \mu \text{ (momentum)},\ \textit{nesterov}\in\{True,False\},\\
            &\hspace{13mm}(a,b,c)\ \text{ (NS coefficients)},\
               \varepsilon \text{ (epsilon)},\ k \text{ (NS steps)},\
               \theta_0 \text{ (params)},\ f(\theta) \text{ (objective)} \\
            &\textbf{initialize} : B_0 \leftarrow 0 \text{ (momentum buffer)} \\[-1.ex]
            &\rule{110mm}{0.4pt} \\
            &\textbf{for}\ t=1\ \textbf{to}\ \ldots\ \textbf{do} \\[0.25ex]
            &\hspace{5mm} g_t \leftarrow \nabla_{\theta} f_t(\theta_{t-1}) \\[0.25ex]
            &\hspace{5mm} B_t \leftarrow \mu B_{t-1} + g_t \\[0.25ex]
            &\hspace{5mm} \widetilde{B}_t \leftarrow
                \begin{cases}
                   g_t + \mu B_t, & \text{if nesterov}=True \\
                   B_t,           & \text{if nesterov}=False
                \end{cases} \\[1.0ex]
            &\hspace{5mm} O_t \leftarrow \mathrm{NS}^{(a,b,c)}_{k}\!\big(\widetilde{B}_t;\ \varepsilon\big) \\[0.5ex]
            &\hspace{5mm} \theta_t \leftarrow \theta_{t-1} - \gamma\,\lambda\,\theta_{t-1}
               \quad\text{(decoupled weight decay)} \\[0.25ex]

            &\hspace{5mm} \gamma \leftarrow \mathrm{AdjustLR}\!\big(\gamma;\ \mathrm{shape}\!\big(\theta_t \big) \big) \\[0.25ex]
            &\hspace{5mm} \theta_t \leftarrow \theta_t - \gamma\, O_t \\
            &\rule{110mm}{0.4pt} \\[-1.ex]
            &\mathbf{return}\ \theta_t \\[-1.ex]
            &\rule{110mm}{0.4pt}s
       \end{aligned}

    Here, :math:`\mathrm{NS}^{(a,b,c)}_{k}(\cdot;\varepsilon)` denotes :math:`k` iterations of the
    Newton–Schulz orthogonalization operator parameterized by coefficients :math:`(a,b,c)`
    with numerical stabilization :math:`\varepsilon`.

    The purpose for :math:`\mathrm{AdjustLR}\!\big(\gamma;\ \mathrm{shape}\!\big(\theta_t \big) \big)`
    is to make the orthogonalized update scale consistently across rectangular matrices.

    Keller's original implementation scales the update by :math:`\sqrt{\max\!\left(1, \frac{A}{B}\right)}`,
    where :math:`A` and :math:`B` are dimensions of the matrix being optimized, which represent fan-out
    and fan-in for a Linear weight matrix.

    Moonshot's implementation focuses on matching :math:`RMS` of AdamW. The adjustment is computed as:
    :math:`\gamma \leftarrow {0.2}\gamma\,\sqrt{\max\!\left({A}, {B}\right)}`
    The method is adopted from `Muon is Scalable for LLM Training`_. Research
    results show that with this adjustment Muon can directly reuse the learning rate
    and weight decay tuned for AdamW.

    Jeremy Bernstein in `Deriving Muon`_ proposes a scaling condition on the spectral norm, which
    scales the update by :math:`\sqrt{\frac{A}{B}}`. This is similar to the Keller's "original"
    implementation but removes clamping down to 1.

    We provide these options for the learning rate adjustment: "original", which follows Keller's
    implementation, "match_rms_adamw", which refers to Moonshot's implementation, and "spectral_unclamped",
    which matches Bernstein's implementation. If `adjust_lr_fn` is not specified, the default is "original".

    For further details regarding the algorithm we refer to `Muon: An optimizer for hidden layers in neural networks`_,
    `Muon is Scalable for LLM Training`_, and `Deriving Muon`_.
    """
    + rf"""
    Args:
        {_params_doc}. Note that Muon is an optimizer for 2D parameters of neural network hidden layers. Other
            parameters, such as bias, and embedding, should be optimized by a standard method such as AdamW.
        lr (float, Tensor, optional): learning rate (default: 1e-3).
        weight_decay (float, optional): weight decay (L2 penalty). (default: 0.1)
        momentum (float, optional): momentum factor (default: 0.95)
        nesterov (bool, optional): enables Nesterov momentum. Only applicable
            when momentum is non-zero
        ns_coefficients (tuple of three floats, optional): coefficients \(a,b,c\) for the
            Newton–Schulz orthogonalization polynomial (default: ({DEFAULT_A}, {DEFAULT_B}, {DEFAULT_C}))
        eps (float, optional): term added to the denominator for numerical stability. (default: {EPS})
        ns_steps (int, optional): number of Newton–Schulz iteration steps. (default: {DEFAULT_NS_STEPS})
        adjust_lr_fn (str, optional): function to adjust learning rate. One of "original", "match_rms_adamw", and "spectral_unclamped".
            If not specified, we will default to use "original". (default: None)
        foreach (bool, optional): whether to use the multi-tensor (``torch._foreach_*``) implementation. The
            multi-tensor path fuses the momentum and parameter updates across same-shape 2D params; the
            Newton-Schulz iteration itself is still performed per-tensor, so numerics match the single-tensor
            path. If ``None`` (the default), the single-tensor path is used to preserve historical behavior;
            pass ``foreach=True`` to opt in. (default: None)

    Example:
        >>> # xdoctest: +SKIP
        >>> # Muon only supports 2D params; use a standard optimizer
        >>> # such as AdamW for biases, embeddings, and other non-2D
        >>> # parameters.
        >>> muon_params = [
        ...     p for p in model.parameters() if p.ndim == 2
        ... ]
        >>> other_params = [
        ...     p for p in model.parameters() if p.ndim != 2
        ... ]
        >>> optim_muon = torch.optim.Muon(
        ...     muon_params, lr=0.02, momentum=0.95
        ... )
        >>> optim_adamw = torch.optim.AdamW(
        ...     other_params, lr=3e-4, weight_decay=0.01
        ... )
        >>> optim_muon.zero_grad()
        >>> optim_adamw.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optim_muon.step()
        >>> optim_adamw.step()

    .. _Muon\: An optimizer for hidden layers in neural networks:
        https://kellerjordan.github.io/posts/muon/
    .. _Muon is Scalable for LLM Training:
        https://arxiv.org/pdf/2502.16982
    .. _Deriving Muon:
        https://jeremybernste.in/writing/deriving-muon

    """
)


def _single_tensor_muon(
    params: list[Tensor],
    grads: list[Tensor],
    muon_momentum_bufs: list[Tensor],
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: str | None,
    has_complex: bool,
) -> None:
    lr = _to_scalar(lr)
    if has_complex:
        raise ValueError("Complex parameters are not supported")

    for i, param in enumerate(params):
        grad = grads[i]
        if grad.ndim != 2:
            raise ValueError("Param gradient must be a 2D matrix")

        buf = muon_momentum_bufs[i]
        buf.lerp_(grad, 1 - momentum)
        update = grad.lerp(buf, momentum) if nesterov else buf

        update = _zeropower_via_newtonschulz(update, ns_coefficients, ns_steps, eps)

        adjusted_lr = _adjust_lr(lr, adjust_lr_fn, param.shape)

        param.mul_(1 - lr * weight_decay)
        param.add_(update, alpha=-adjusted_lr)


def _multi_tensor_muon(
    params: list[Tensor],
    grads: list[Tensor],
    muon_momentum_bufs: list[Tensor],
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: str | None,
    has_complex: bool,
) -> None:
    lr = _to_scalar(lr)
    if has_complex:
        raise ValueError("Complex parameters are not supported")
    if not params:
        return
    for g in grads:
        if g.ndim != 2:
            raise ValueError("Param gradient must be a 2D matrix")

    # Phase 1: momentum buffer update fused across all params.
    # buf <- lerp(buf, grad, 1 - momentum) for every param in one call.
    torch._foreach_lerp_(muon_momentum_bufs, grads, 1 - momentum)
    if nesterov:
        updates = list(torch._foreach_lerp(grads, muon_momentum_bufs, momentum))
    else:
        updates = list(muon_momentum_bufs)

    # Phase 2: Newton-Schulz per tensor. Standard NS has no natural batching,
    # so per-tensor preserves bit-for-bit numerics with _single_tensor_muon.
    updates = [
        _zeropower_via_newtonschulz(u, ns_coefficients, ns_steps, eps) for u in updates
    ]

    # Phase 3: weight-decay shrink (uniform scale across all params) followed
    # by the shape-grouped negative-lr add. Group by shape, dtype, and device
    # so each _foreach_add_ call preserves per-tensor dtype semantics while
    # still sharing the shape-specific adjusted_lr.
    torch._foreach_mul_(params, 1 - lr * weight_decay)

    add_groups: dict[
        tuple[int, int, torch.dtype, torch.dtype, torch.device, torch.device], list[int]
    ] = {}
    for i, (p, update) in enumerate(zip(params, updates, strict=True)):
        add_groups.setdefault(
            (p.size(0), p.size(1), p.dtype, update.dtype, p.device, update.device),
            [],
        ).append(i)
    for indices in add_groups.values():
        group_params = [params[i] for i in indices]
        group_updates = [updates[i] for i in indices]
        # _foreach_add_ silently falls back to per-tensor add_ when self/other
        # dtypes differ. NS outputs bf16; params are typically fp32. Cast once
        # via a fused _foreach_copy_ so _foreach_add_ stays on the multi-tensor
        # fast path. The cast is bit-equivalent to the per-tensor add_ that the
        # single-tensor path performs internally.
        target_dtype = group_params[0].dtype
        if group_updates[0].dtype != target_dtype:
            cast = [torch.empty_like(p) for p in group_params]
            torch._foreach_copy_(cast, group_updates)
            group_updates = cast
        adjusted_lr = _adjust_lr(lr, adjust_lr_fn, group_params[0].shape)
        torch._foreach_add_(group_params, group_updates, alpha=-adjusted_lr)


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_muon)
def muon(
    params: list[Tensor],
    grads: list[Tensor],
    muon_momentum_bufs: list[Tensor],
    *,
    foreach: bool | None = None,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: str | None,
    has_complex: bool,
) -> None:
    r"""Functional API that performs Muon algorithm computation.

    See :class:`~torch.optim.Muon` for details.
    """
    # foreach=None preserves the historical single-tensor default; only an
    # explicit True opts into the multi-tensor (_foreach_*) path.
    func = _multi_tensor_muon if foreach else _single_tensor_muon

    func(
        params,
        grads,
        muon_momentum_bufs,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        ns_coefficients=ns_coefficients,
        ns_steps=ns_steps,
        eps=eps,
        adjust_lr_fn=adjust_lr_fn,
        has_complex=has_complex,
    )
