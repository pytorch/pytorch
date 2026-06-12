# mypy: allow-untyped-defs
# mypy: disable-error-code=arg-type
"""Implementation of the Muon optimizer."""

import math
from collections.abc import MutableMapping
from itertools import repeat

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

EPS = 1e-7
DEFAULT_NS_STEPS = 5

# Constants from Keller Jordan's Muon post: https://kellerjordan.github.io/posts/muon/
# github permlink: https://github.com/KellerJordan/Muon/blob/f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd/muon.py#L16
JORDAN_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
# constants from https://arxiv.org/abs/2505.16932 and https://arxiv.org/abs/2506.10935 (same coefficients by two independent teams)
# code to compute coefficients can be found in https://github.com/NoahAmsel/PolarExpress/blob/main/polar_express.py#L74
PE_COEFFICIENTS = (
    (8.237312, -23.157747, 16.680568),
    (4.082442, -2.893048, 0.525285),
    (3.926348, -2.854747, 0.531802),
    (3.298219, -2.424542, 0.486320),
    (2.297037, -1.636626, 0.400263),
    (1.876381, -1.234790, 0.358919),
    (1.856442, -1.213245, 0.356800),
    (1.856435, -1.213237, 0.356799),
    (1.856431, -1.213229, 0.356795),
    (1.874995, -1.249991, 0.374995),
)


# A single (a, b, c) tuple or a sequence of per-step (a, b, c) tuples.
NSCoefficients = (
    str | tuple[float, float, float] | tuple[tuple[float, float, float], ...]
)


def _zeropower_via_newtonschulz(
    grad: Tensor,
    ns_coefficients: tuple[tuple[float, float, float], ...],
    ns_steps: int,
    eps: float,
    normalization: str = "schatten",
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
    if normalization not in ("fro", "schatten", "aol"):
        raise ValueError(
            f"Unsupported normalization {normalization}, expected one of 'fro', 'schatten', or 'aol'"
        )
    ortho_grad = grad.bfloat16()
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T

    coefficients = ns_coefficients[:ns_steps] + tuple(
        repeat(ns_coefficients[-1], ns_steps - len(ns_coefficients))
    )
    # Ensure spectral norm is at most 1
    if normalization == "fro":
        ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
        gram_matrix = ortho_grad @ ortho_grad.T
    elif normalization == "schatten":
        gram_matrix = ortho_grad @ ortho_grad.T
        s = gram_matrix.norm().clamp(min=eps)
        ortho_grad.mul_(s.rsqrt())  # normalize input
        gram_matrix.div_(s)  # update gram matrix without recomputing
    elif normalization == "aol":
        gram_matrix = ortho_grad @ ortho_grad.T
        s_vec = torch.rsqrt(
            torch.clamp_min(gram_matrix.abs().sum(dim=-1, keepdim=False), min=eps)
        )
        ortho_grad.mul_(s_vec.unsqueeze(-1))  # normalize input
        gram_matrix.mul_(s_vec.unsqueeze(-1) * s_vec.unsqueeze(-2))
    # setting ns_steps to 0 will only perform the normalization
    if ns_steps > 0:
        # perform the first iteration reusing the gram matrix computed for normalization
        a0, b0, c0 = coefficients[0]
        gram_update = torch.addmm(
            gram_matrix, gram_matrix, gram_matrix, beta=b0, alpha=c0
        )
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a0)
        # Perform the NS remaining iterations
        for a, b, c in coefficients[1:]:
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
        ns_coefficients: NSCoefficients = "polar_express",
        normalization: str = "schatten",
        eps: float = EPS,
        ns_steps: int = DEFAULT_NS_STEPS,
        adjust_lr_fn: str | None = None,
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
        ]:
            raise ValueError(
                f"Adjust learning rate function {adjust_lr_fn} is not supported"
            )
        if isinstance(ns_coefficients, str):
            if ns_coefficients == "jordan":
                ns_coefficients = JORDAN_COEFFICIENTS
            elif ns_coefficients == "polar_express":
                ns_coefficients = PE_COEFFICIENTS
            else:
                raise ValueError(
                    f"Unsupported NS coefficients preset: {ns_coefficients}"
                )
        # Normalize a single (a, b, c) tuple into a tuple of tuples.
        if ns_coefficients and not isinstance(ns_coefficients[0], tuple):
            ns_coefficients = (ns_coefficients,)  # type: ignore[assignment]
        if normalization not in ("fro", "schatten", "aol"):
            raise ValueError(
                f"Unsupported normalization {normalization}, expected one of 'fro', 'schatten', or 'aol'"
            )

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "normalization": normalization,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon only supports 2D parameters whereas we found a parameter with size: {p.size()}"
                    )

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
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                nesterov=group["nesterov"],
                ns_coefficients=group["ns_coefficients"],
                normalization=group["normalization"],
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
    is to make the orthogonalized update have a consistent :math:`RMS` across rectangular matrices.

    Keller's original implementation scales the update by :math:`\sqrt{\max\!\left(1, \frac{A}{B}\right)}`,
    where :math:`A` and :math:`B` are dimension of the matrix being optimized.

    Moonshot's implementation also focuses on matching :math:`RMS` of AdamW. The adjustment is computed as:
    :math:`\gamma \leftarrow {0.2}\gamma\,\sqrt{\max\!\left({A}, {B}\right)}`
    The method is adopted from `Muon is Scalable for LLM Training`_. Research
    results show that with this adjustment Muon can directly reuse the learning rate
    and weight decay tuned for AdamW.

    We provide two options for the learning rate adjustment: "original", which follows Keller's
    implementation, and "match_rms_adamw", which refers to Moonshot's implementation. This gives users the
    flexibility to choose between the two. If `adjust_lr_fn` is not specified, the default is "original".

    We also provide two options for the Newton–Schulz coefficients: "jordan", which corresponds
    to the coefficients used in Keller's original implementation, and "polar_express", which corresponds
    to the coefficients derived in `Polar Express coefficients`_ and
    `Accelerating Newton-Schulz Iteration for Orthogonalization via Chebyshev-type Polynomials`_.

    For further details regarding the algorithm we refer to `Muon: An optimizer for hidden layers in neural networks`_
    and `Muon is Scalable for LLM Training`_.
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
        ns_coefficients (string or tuple of three floats or tuple of tuples, optional): coefficients \(a,b,c\) for the
            Newton–Schulz orthogonalization polynomial. If a string is provided, it must be one of "jordan" or
            "polar_express". Tuple of three floats corresponds to a single (a, b, c) tuple for all iterations,
            while a tuple of tuples corresponds to per-step (a, b, c). If not specified, we will default
            to use the "polar_express" coefficients. (default: "polar_express")
        normalization (str, optional): method to normalize the input matrix before applying Newton–Schulz iteration.
            A spectral norm of the input matrix at most 1 is required to ensure convergence, and tighter estimation
            yields a faster convergence. Options are "fro", "schatten", or "aol". "fro" corresponds to normalizing
            the input matrix by its Frobenius norm. "schatten" corresponds
            to normalizing the input matrix by its Schatten p-norm (see
            `Accelerating Newton-Schulz Iteration for Orthogonalization via Chebyshev-type Polynomials`_). "aol"
            corresponds to normalizing the input matrix as done in
            `Turbo-Muon: Accelerating Orthogonality-Based Optimization with Pre-Conditioning`_ . If not specified,
            we will default to use "schatten". (default: "schatten")
        eps (float, optional): term added to the denominator for numerical stability. (default: {EPS})
        ns_steps (int, optional): number of Newton–Schulz iteration steps. (default: {DEFAULT_NS_STEPS})
        adjust_lr_fn (str, optional): function to adjust learning rate. One of "original" and "match_rms_adamw".
            If not specified, we will default to use "original". (default: None)

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
    .. _Polar Express coefficients:
        https://arxiv.org/pdf/2505.16932
    .. _Accelerating Newton-Schulz Iteration for Orthogonalization via Chebyshev-type Polynomials:
        https://arxiv.org/pdf/2506.10935
    .. _Turbo-Muon\: Accelerating Orthogonality-Based Optimization with Pre-Conditioning:
        https://arxiv.org/pdf/2512.04632

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
    ns_coefficients: tuple[tuple[float, float, float], ...],
    normalization: str,
    ns_steps: int,
    eps: float,
    adjust_lr_fn: str | None,
    has_complex: bool,
) -> None:
    lr = _to_scalar(lr)
    if has_complex:
        raise ValueError("Complex parameters are not supported")
    if normalization not in ("fro", "schatten", "aol"):
        raise ValueError(
            f"Unsupported normalization {normalization}, expected one of 'fro', 'schatten', or 'aol'"
        )

    for i, param in enumerate(params):
        grad = grads[i]
        if grad.ndim != 2:
            raise ValueError("Param gradient must be a 2D matrix")

        buf = muon_momentum_bufs[i]
        buf.lerp_(grad, 1 - momentum)
        update = grad.lerp(buf, momentum) if nesterov else buf

        update = _zeropower_via_newtonschulz(
            update, ns_coefficients, ns_steps, eps, normalization
        )

        adjusted_lr = _adjust_lr(lr, adjust_lr_fn, param.shape)

        param.mul_(1 - lr * weight_decay)
        param.add_(update, alpha=-adjusted_lr)


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
    ns_coefficients: tuple[tuple[float, float, float], ...],
    normalization: str = "schatten",
    ns_steps: int,
    eps: float,
    adjust_lr_fn: str | None,
    has_complex: bool,
) -> None:
    r"""Functional API that performs Muon algorithm computation.

    See :class:`~torch.optim.Muon` for details.
    """
    if foreach is not None and foreach:
        raise RuntimeError("Foreach is not supported for Muon yet")

    func = _single_tensor_muon

    func(
        params,
        grads,
        muon_momentum_bufs,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        ns_coefficients=ns_coefficients,
        normalization=normalization,
        ns_steps=ns_steps,
        eps=eps,
        adjust_lr_fn=adjust_lr_fn,
        has_complex=has_complex,
    )
