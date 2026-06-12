# mypy: allow-untyped-defs
# mypy: disable-error-code=arg-type
"""Implementation of the Muon optimizer with optional Gram Newton-Schulz support."""

import math
from collections.abc import Callable, MutableMapping
from enum import Enum
from typing import Any, NamedTuple, TypedDict

import torch
from torch import Tensor

from .optimizer import (
    _disable_dynamo_if_unsupported,
    _params_doc,
    _to_scalar,
    Optimizer,
    ParamsT,
)


__all__ = ["GramNewtonSchulzConfig", "Muon", "NewtonSchulzAlgorithm"]


class NewtonSchulzAlgorithm(str, Enum):
    """Orthogonalization algorithm used inside Muon."""

    STANDARD = "standard"
    GRAM = "gram"


# Allow Muon state dicts containing the enum to deserialize under
# torch.load(weights_only=True).
torch.serialization.add_safe_globals([NewtonSchulzAlgorithm])


class GramNewtonSchulzConfig(TypedDict, total=False):
    """Schema for the ``gram_newton_schulz_config`` dict consumed by Muon.

    All keys are optional; missing keys fall back to module-level defaults
    (see ``DEFAULT_GRAM_NS_COEFFICIENTS``, ``DEFAULT_GRAM_NS_RESET_ITERATIONS``).
    """

    gram_ns_coefficients: list[list[float]]
    gram_ns_reset_iterations: list[int]


# Constants from Keller Jordan's Muon post: https://kellerjordan.github.io/posts/muon/
# github permlink: https://github.com/KellerJordan/Muon/blob/f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd/muon.py#L16
EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315
DEFAULT_NS_STEPS = 5
DEFAULT_GRAM_NS_RESET_ITERATIONS: list[int] = []
# Default per-iteration (a, b, c) for Gram NS: same constants as vanilla Muon
# repeated for every NS step. Schedules like YOU_COEFFICIENTS can be opted into
# via gram_newton_schulz_config["gram_ns_coefficients"], but they typically
# require a non-empty gram_ns_reset_iterations to stay numerically stable.
# Keeping the default identical to vanilla Muon makes Gram NS a drop-in
# replacement that does not depend on a reset schedule.
DEFAULT_GRAM_NS_COEFFICIENTS: list[list[float]] = [
    [DEFAULT_A, DEFAULT_B, DEFAULT_C] for _ in range(DEFAULT_NS_STEPS)
]

# Valid keys in gram_newton_schulz_config, derived from the TypedDict schema.
_GRAM_CONFIG_KEYS: frozenset[str] = frozenset(GramNewtonSchulzConfig.__annotations__)


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


def _zeropower_via_newtonschulz_batched(
    grads: list[Tensor],
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> list[Tensor]:
    """Batched standard NS for a list of same-shape square 2D gradients.

    Stacks gradients into a 3D batch and runs the quintic NS iteration with
    bmm/baddbmm so same-shape squares fuse into one batched matmul per op
    instead of sequential per-tensor mm/addmm. Caller guarantees all entries
    are 2D and square with identical shape.
    """
    a, b, c = ns_coefficients
    X = torch.stack([g.bfloat16() for g in grads])  # (batch, M, M)
    X = X / X.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    for _ in range(ns_steps):
        gram = X @ X.mT
        gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        X = torch.baddbmm(X, gram_update, X, beta=a)
    return list(X.unbind(0))


def _gram_newton_schulz(
    X: Tensor,
    gram_ns_coefficients: list[list[float]],
    gram_ns_reset_iterations: list[int],
) -> Tensor:
    """Gram Newton-Schulz orthogonalization for a single 2D matrix.

    Works in the Gram space (R = X @ X^T), accumulating Q and applying
    X = Q @ X once at the end. More efficient for rectangular matrices
    where M << N or M >> N because the iteration body operates on the
    smaller (M, M) Gram matrix rather than the full (M, N) input.
    """
    R = X @ X.T
    I = torch.eye(R.size(0), device=X.device, dtype=X.dtype)  # noqa: E741
    Q: Tensor = I

    for i, (a, b, c) in enumerate(gram_ns_coefficients):
        if i in gram_ns_reset_iterations and i != 0:
            X = Q @ X
            R = X @ X.T
            Q = I

        Z = torch.addmm(R, R, R, beta=b, alpha=c)
        if i == 0 or i in gram_ns_reset_iterations:
            Q = Z + a * I
        else:
            Q = torch.addmm(Q, Q, Z, beta=a)
        if (
            i < len(gram_ns_coefficients) - 1
            and i + 1 not in gram_ns_reset_iterations
        ):
            RZ = torch.addmm(R, R, Z, beta=a)
            R = torch.addmm(RZ, Z, RZ, beta=a)

    return Q @ X


def _gram_newton_schulz_batched(
    X: Tensor,
    gram_ns_coefficients: list[list[float]],
    gram_ns_reset_iterations: list[int],
) -> Tensor:
    """Batched Gram Newton-Schulz orthogonalization.

    X must be a 3D tensor of shape (batch, M, N) where M <= N. Processes
    all matrices in the batch simultaneously using batched matmuls.
    """
    R = X @ X.mT  # (batch, M, M)
    batch_size = R.size(0)
    I = (  # noqa: E741
        torch.eye(R.size(-1), device=X.device, dtype=X.dtype)
        .unsqueeze(0)
        .expand(batch_size, -1, -1)
        .contiguous()
    )
    Q: Tensor = I

    for i, (a, b, c) in enumerate(gram_ns_coefficients):
        if i in gram_ns_reset_iterations and i != 0:
            X = Q @ X
            R = X @ X.mT
            Q = I

        Z = torch.baddbmm(R, R, R, beta=b, alpha=c)
        if i == 0 or i in gram_ns_reset_iterations:
            Q = Z + a * I
        else:
            Q = torch.baddbmm(Q, Q, Z, beta=a)
        if (
            i < len(gram_ns_coefficients) - 1
            and i + 1 not in gram_ns_reset_iterations
        ):
            RZ = torch.baddbmm(R, R, Z, beta=a)
            R = torch.baddbmm(RZ, Z, RZ, beta=a)

    return Q @ X


def _zeropower_via_gram_newtonschulz(
    grad: Tensor,
    gram_ns_coefficients: list[list[float]],
    gram_ns_reset_iterations: list[int],
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> Tensor:
    """Orthogonalize a 2D gradient using Gram Newton-Schulz iteration.

    Uses the Gram NS method for rectangular matrices. For square matrices
    (where the Gram reformulation has no asymptotic advantage), falls back
    to the standard NS path so the caller does not need to pre-route.
    """
    if len(grad.shape) != 2:
        raise ValueError("Input tensor gradient must be a 2D matrix")
    if grad.size(0) == grad.size(1):
        return _zeropower_via_newtonschulz(grad, ns_coefficients, ns_steps, eps)

    X = grad.float()
    should_transpose = X.size(0) > X.size(1)
    if should_transpose:
        X = X.T
    X = (X / (X.norm() + eps)).half()
    X = _gram_newton_schulz(X, gram_ns_coefficients, gram_ns_reset_iterations)
    if should_transpose:
        X = X.T
    return X


def _zeropower_via_gram_newtonschulz_batched(
    grads: list[Tensor],
    gram_ns_coefficients: list[list[float]],
    gram_ns_reset_iterations: list[int],
    eps: float,
) -> list[Tensor]:
    """Batched Gram NS orthogonalization for same-shape 2D gradients.

    Stacks gradients into a 3D batch, runs batched Gram NS, then unstacks.
    Caller guarantees all entries are 2D, identically-shaped, and rectangular
    (square shapes should be routed to the standard NS path instead).
    """
    should_transpose = grads[0].size(0) > grads[0].size(1)
    X = torch.stack([g.float() for g in grads])  # (batch, M, N)
    if should_transpose:
        X = X.mT  # (batch, N, M) so rows <= cols
    X = (X / (X.norm(dim=(-2, -1), keepdim=True) + eps)).half()
    X = _gram_newton_schulz_batched(
        X, gram_ns_coefficients, gram_ns_reset_iterations
    )
    if should_transpose:
        X = X.mT
    return list(X.unbind(0))


def _parse_gram_newton_schulz_config(
    gram_newton_schulz_config: GramNewtonSchulzConfig | None,
) -> tuple[list[list[float]], list[int]]:
    """Validate and unpack a ``gram_newton_schulz_config`` dict.

    Returns ``(gram_ns_coefficients, gram_ns_reset_iterations)``, filling in
    defaults for missing keys. ``None`` or ``{}`` yields all defaults.
    """
    cfg = gram_newton_schulz_config or {}
    unknown = set(cfg) - _GRAM_CONFIG_KEYS
    if unknown:
        raise ValueError(
            f"Unknown keys in gram_newton_schulz_config: {sorted(unknown)}. "
            f"Supported keys: {sorted(_GRAM_CONFIG_KEYS)}"
        )
    gram_ns_coefficients = cfg.get("gram_ns_coefficients")
    if gram_ns_coefficients is None:
        gram_ns_coefficients = [list(c) for c in DEFAULT_GRAM_NS_COEFFICIENTS]
    gram_ns_reset_iterations = cfg.get("gram_ns_reset_iterations")
    if gram_ns_reset_iterations is None:
        gram_ns_reset_iterations = list(DEFAULT_GRAM_NS_RESET_ITERATIONS)
    return gram_ns_coefficients, gram_ns_reset_iterations


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
        ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
        eps: float = EPS,
        ns_steps: int = DEFAULT_NS_STEPS,
        adjust_lr_fn: str | None = None,
        foreach: bool | None = None,
        ns_algorithm: NewtonSchulzAlgorithm = NewtonSchulzAlgorithm.STANDARD,
        gram_newton_schulz_config: GramNewtonSchulzConfig | None = None,
        use_cuda_graph: bool = False,
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
        ns_algorithm = NewtonSchulzAlgorithm(ns_algorithm)
        if (
            ns_algorithm is not NewtonSchulzAlgorithm.GRAM
            and gram_newton_schulz_config is not None
        ):
            # Reject the config when gram NS is off so callers cannot silently
            # configure gram-only knobs that would have no effect.
            raise ValueError(
                "gram_newton_schulz_config is only valid when "
                "ns_algorithm=NewtonSchulzAlgorithm.GRAM; got "
                f"gram_newton_schulz_config={gram_newton_schulz_config!r} "
                f"with ns_algorithm={ns_algorithm}."
            )
        gram_ns_coefficients, gram_ns_reset_iterations = (
            _parse_gram_newton_schulz_config(gram_newton_schulz_config)
        )
        if use_cuda_graph and not foreach:
            # CUDA graph capture is an amortization of per-shape NS launches
            # across a shape group; it only makes sense in the multi-tensor
            # (foreach) path, which is also where the per-instance cache is
            # consulted.
            raise ValueError(
                "use_cuda_graph=True requires foreach=True; got "
                f"foreach={foreach!r}."
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
            "ns_algorithm": ns_algorithm,
            "gram_ns_coefficients": gram_ns_coefficients,
            "gram_ns_reset_iterations": gram_ns_reset_iterations,
            "use_cuda_graph": use_cuda_graph,
        }
        super().__init__(params, defaults)

        # Per-instance cache for CUDA-graph-captured NS computations. Shared
        # across param groups so two groups with identical (count, shape,
        # dtype, device, algorithm, coefficients) reuse the same graph; see
        # _NsCacheKey for the full discriminator set.
        self._cuda_graph_cache: dict[
            _NsCacheKey,
            tuple[torch.cuda.CUDAGraph, list[Tensor], list[Tensor]],
        ] = {}

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
            group.setdefault("ns_algorithm", NewtonSchulzAlgorithm.STANDARD)
            group.setdefault(
                "gram_ns_coefficients",
                [list(c) for c in DEFAULT_GRAM_NS_COEFFICIENTS],
            )
            group.setdefault(
                "gram_ns_reset_iterations",
                list(DEFAULT_GRAM_NS_RESET_ITERATIONS),
            )
            group.setdefault("use_cuda_graph", False)
            # ns_algorithm round-trips through JSON / pickle as a plain string;
            # normalize back to the enum so dispatch comparisons are stable.
            group["ns_algorithm"] = NewtonSchulzAlgorithm(group["ns_algorithm"])
        # The CUDA graph cache itself is *not* part of the state dict -- the
        # captured graphs reference live tensors that are not portable across
        # processes. Loading state always starts with a fresh, empty cache;
        # graphs will be re-captured on the first step() with each shape.
        if not hasattr(self, "_cuda_graph_cache"):
            self._cuda_graph_cache = {}

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
                ns_algorithm=group["ns_algorithm"],
                gram_ns_coefficients=group["gram_ns_coefficients"],
                gram_ns_reset_iterations=group["gram_ns_reset_iterations"],
                use_cuda_graph=group["use_cuda_graph"],
                cuda_graph_cache=self._cuda_graph_cache,
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

    Setting ``ns_algorithm=NewtonSchulzAlgorithm.GRAM`` selects an alternate
    orthogonalization path that computes the Newton-Schulz iteration in
    Gram space (``R = X @ X^T``) for rectangular matrices, falling back to
    the standard iteration on square matrices. The Gram path is typically
    faster on tall or wide rectangular params (``M << N`` or ``M >> N``)
    because the iteration body operates on the smaller (M, M) Gram matrix
    rather than the full (M, N) input. Note that Gram NS is *not* bit-equivalent
    to standard NS on rectangular params -- it converges to a different (but
    still valid) orthogonalization; end-to-end model quality is comparable in
    practice. Pass ``gram_newton_schulz_config`` (only when ``ns_algorithm``
    is GRAM) to override the gram-specific coefficients / reset schedule.

    When ``foreach=True``, same-shape parameter groups are batched through
    the NS routine: for STANDARD this routes same-shape square groups
    (with more than one tensor) through ``bmm`` / ``baddbmm`` so they fuse
    into a single batched matmul per op; for GRAM rectangular groups
    likewise go through batched Gram NS. The multi-tensor path therefore
    diverges slightly from the single-tensor path on those configurations
    (different reduction order across the batch dim), but the difference
    stays within float-precision tolerance and is irrelevant for training.

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
        ns_coefficients (tuple of three floats, optional): coefficients \(a,b,c\) for the
            Newton–Schulz orthogonalization polynomial (default: ({DEFAULT_A}, {DEFAULT_B}, {DEFAULT_C}))
        eps (float, optional): term added to the denominator for numerical stability. (default: {EPS})
        ns_steps (int, optional): number of Newton–Schulz iteration steps. (default: {DEFAULT_NS_STEPS})
        adjust_lr_fn (str, optional): function to adjust learning rate. One of "original" and "match_rms_adamw".
            If not specified, we will default to use "original". (default: None)
        foreach (bool, optional): whether to use the multi-tensor (``torch._foreach_*``) implementation. The
            multi-tensor path fuses the momentum and parameter updates across same-shape 2D params; with the
            standard NS algorithm and only rectangular shapes (or square shapes with one tensor each), numerics
            match the single-tensor path. Same-shape square groups with more than one tensor go through batched
            ``bmm`` / ``baddbmm`` when ``foreach=True``, which is within float-precision tolerance of per-tensor
            execution but not bit-identical. If ``None`` (the default), the single-tensor path is used to
            preserve historical behavior; pass ``foreach=True`` to opt in. (default: None)
        ns_algorithm (NewtonSchulzAlgorithm, optional): which orthogonalization algorithm to use. One of
            ``NewtonSchulzAlgorithm.STANDARD`` (the original Muon iteration) or ``NewtonSchulzAlgorithm.GRAM``
            (the Gram-space variant; see the class description above for when this is faster).
            (default: ``NewtonSchulzAlgorithm.STANDARD``)
        gram_newton_schulz_config (dict, optional): configuration dict for the Gram NS path. Only valid when
            ``ns_algorithm=NewtonSchulzAlgorithm.GRAM``; passing it with any other algorithm raises
            ``ValueError``. Supported keys (all optional):

            - ``"gram_ns_coefficients"`` (list[list[float]]): per-iteration ``(a, b, c)`` triples; one inner
              list per NS step.
            - ``"gram_ns_reset_iterations"`` (list[int]): iteration indices at which to materialize ``X``
              from the accumulated ``Q`` and reset ``Q`` to the identity.

            (default: None)
        use_cuda_graph (bool, optional): if True, capture each per-shape NS computation into a CUDA graph and
            replay on subsequent ``step()`` calls. The graph is keyed by (count, shape, dtype, device,
            algorithm, coefficients), so distinct shape groups and distinct param groups with different
            configs get independent graphs. The captured static-output buffers are returned by reference --
            the immediate Phase-3 ``_foreach_add_`` consumes them in the same ``step()``, so they are safe
            to overwrite on the next replay. Requires ``foreach=True`` (CUDA graph capture is an
            amortization across the multi-tensor NS dispatch); passing ``use_cuda_graph=True`` with
            ``foreach`` left at its ``None`` default or set to ``False`` raises ``ValueError``. The flag is
            silently a no-op on CPU. (default: False)

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

    """
)


def _ns_eager(
    inputs: list[Tensor],
    *,
    ns_algorithm: NewtonSchulzAlgorithm,
    is_square: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    gram_ns_coefficients: list[list[float]],
    gram_ns_reset_iterations: list[int],
) -> list[Tensor]:
    """Eager NS dispatch for one same-shape group of tensors.

    Routes by ``(ns_algorithm, is_square, len(inputs))``.

    - STANDARD, square, len > 1: batched bmm/baddbmm path.
    - STANDARD, otherwise: per-tensor standard NS (rectangular standard NS
      has no natural batching).
    - GRAM, rectangular, len > 1: batched Gram NS.
    - GRAM, rectangular, len == 1: single Gram NS.
    - GRAM, square: fall back to the standard NS routing (Gram offers no
      advantage on square shapes and ``_zeropower_via_gram_newtonschulz``
      already falls back internally for the single-tensor case).
    """
    if ns_algorithm is NewtonSchulzAlgorithm.STANDARD or is_square:
        if is_square and len(inputs) > 1:
            return _zeropower_via_newtonschulz_batched(
                inputs, ns_coefficients, ns_steps, eps
            )
        return [
            _zeropower_via_newtonschulz(t, ns_coefficients, ns_steps, eps)
            for t in inputs
        ]
    if len(inputs) == 1:
        return [
            _zeropower_via_gram_newtonschulz(
                inputs[0],
                gram_ns_coefficients,
                gram_ns_reset_iterations,
                ns_coefficients,
                ns_steps,
                eps,
            )
        ]
    return _zeropower_via_gram_newtonschulz_batched(
        inputs, gram_ns_coefficients, gram_ns_reset_iterations, eps
    )


class _NsCacheKey(NamedTuple):
    """Hashable cache key for CUDA-graph-captured NS computations.

    Distinct entries are needed per `(count, shape, device, dtype)` of the
    inputs and per `(algorithm, coefficients, ns_steps, eps)` of the
    computation, so two param groups with different configs do not collide.
    """

    count: int
    shape: tuple[int, int]
    device: torch.device
    dtype: torch.dtype
    ns_algorithm: NewtonSchulzAlgorithm
    gram_ns_coefficients: tuple[tuple[float, float, float], ...]
    gram_ns_reset_iterations: tuple[int, ...]
    ns_coefficients: tuple[float, float, float]
    ns_steps: int
    eps: float


def _ns_cudagraph(
    inputs: list[Tensor],
    compute_fn: Callable[[list[Tensor]], list[Tensor]],
    cache_key: _NsCacheKey,
    cuda_graph_cache: dict[
        _NsCacheKey,
        tuple["torch.cuda.CUDAGraph", list[Tensor], list[Tensor]],
    ],
) -> list[Tensor]:
    """Run an NS-style computation through CUDA graph capture/replay.

    Wraps ``compute_fn`` (which must be safe to call inside a capture region
    and must return a list of newly-allocated tensors) with cache lookup,
    static-buffer allocation, warmup, capture, and replay.

    Returns ``static_outputs`` directly (no clone). Callers must consume the
    returned tensors before the next call with the same ``cache_key`` --
    that next call will overwrite the static output buffers in place. This
    contract is safe inside Muon's step() because the immediate Phase-3
    `_foreach_add_` (or per-tensor `add_`) consumes the NS outputs before
    `_run_ns` is invoked again, even across param groups.
    """
    if cache_key not in cuda_graph_cache:
        # Static input buffers: the exact tensors the graph reads from.
        static_inputs = [t.clone() for t in inputs]
        # Warmup is required before capture.
        compute_fn(static_inputs)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_outputs = compute_fn(static_inputs)
        cuda_graph_cache[cache_key] = (graph, static_inputs, static_outputs)

    graph, static_inputs, static_outputs = cuda_graph_cache[cache_key]
    for si, t in zip(static_inputs, inputs, strict=True):
        si.copy_(t)
    graph.replay()
    return list(static_outputs)


def _dispatch_ns_group(
    inputs: list[Tensor],
    *,
    ns_algorithm: NewtonSchulzAlgorithm,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    gram_ns_coefficients: list[list[float]],
    gram_ns_reset_iterations: list[int],
    use_cuda_graph: bool,
    cuda_graph_cache: dict[_NsCacheKey, Any] | None,
) -> list[Tensor]:
    """Dispatch one same-shape NS group through CUDA graph or eager.

    CUDA graph capture only engages when ``use_cuda_graph`` is True, a
    cache was supplied, and the inputs live on CUDA. Otherwise the call
    falls through to the plain eager dispatch.
    """
    is_square = inputs[0].size(0) == inputs[0].size(1)

    def _compute(buffers: list[Tensor]) -> list[Tensor]:
        return _ns_eager(
            buffers,
            ns_algorithm=ns_algorithm,
            is_square=is_square,
            ns_coefficients=ns_coefficients,
            ns_steps=ns_steps,
            eps=eps,
            gram_ns_coefficients=gram_ns_coefficients,
            gram_ns_reset_iterations=gram_ns_reset_iterations,
        )

    graph_eligible = (
        use_cuda_graph and cuda_graph_cache is not None and inputs[0].is_cuda
    )
    if not graph_eligible:
        return _compute(inputs)

    cache_key = _NsCacheKey(
        count=len(inputs),
        shape=(inputs[0].size(0), inputs[0].size(1)),
        device=inputs[0].device,
        dtype=inputs[0].dtype,
        ns_algorithm=ns_algorithm,
        gram_ns_coefficients=tuple(
            tuple(c) for c in gram_ns_coefficients  # type: ignore[misc]
        ),
        gram_ns_reset_iterations=tuple(gram_ns_reset_iterations),
        ns_coefficients=tuple(ns_coefficients),  # type: ignore[arg-type]
        ns_steps=ns_steps,
        eps=eps,
    )
    return _ns_cudagraph(inputs, _compute, cache_key, cuda_graph_cache)


def _run_ns(
    updates: list[Tensor],
    *,
    ns_algorithm: NewtonSchulzAlgorithm,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    gram_ns_coefficients: list[list[float]],
    gram_ns_reset_iterations: list[int],
    use_cuda_graph: bool = False,
    cuda_graph_cache: dict[_NsCacheKey, Any] | None = None,
) -> list[Tensor]:
    """Top-level NS dispatcher.

    Groups updates by shape so same-shape tensors can be batched through the
    appropriate NS routine, then writes the orthogonalized outputs back into
    a list aligned with the input order. Each group is routed through
    ``_dispatch_ns_group`` which optionally captures the per-group NS into a
    CUDA graph keyed on shape, dtype, algorithm, and coefficients.
    """
    shape_groups: dict[tuple[int, int], list[int]] = {}
    for i, update in enumerate(updates):
        shape_groups.setdefault((update.size(0), update.size(1)), []).append(i)

    results: list[Tensor] = list(updates)
    for indices in shape_groups.values():
        group_inputs = [updates[idx] for idx in indices]
        group_results = _dispatch_ns_group(
            group_inputs,
            ns_algorithm=ns_algorithm,
            ns_coefficients=ns_coefficients,
            ns_steps=ns_steps,
            eps=eps,
            gram_ns_coefficients=gram_ns_coefficients,
            gram_ns_reset_iterations=gram_ns_reset_iterations,
            use_cuda_graph=use_cuda_graph,
            cuda_graph_cache=cuda_graph_cache,
        )
        for idx, r in zip(indices, group_results, strict=True):
            results[idx] = r
    return results


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
    ns_algorithm: NewtonSchulzAlgorithm = NewtonSchulzAlgorithm.STANDARD,
    gram_ns_coefficients: list[list[float]] | None = None,
    gram_ns_reset_iterations: list[int] | None = None,
) -> None:
    lr = _to_scalar(lr)
    if has_complex:
        raise ValueError("Complex parameters are not supported")
    if gram_ns_coefficients is None:
        gram_ns_coefficients = [list(c) for c in DEFAULT_GRAM_NS_COEFFICIENTS]
    if gram_ns_reset_iterations is None:
        gram_ns_reset_iterations = list(DEFAULT_GRAM_NS_RESET_ITERATIONS)

    for i, param in enumerate(params):
        grad = grads[i]
        if grad.ndim != 2:
            raise ValueError("Param gradient must be a 2D matrix")

        buf = muon_momentum_bufs[i]
        buf.lerp_(grad, 1 - momentum)
        update = grad.lerp(buf, momentum) if nesterov else buf

        if ns_algorithm is NewtonSchulzAlgorithm.GRAM:
            update = _zeropower_via_gram_newtonschulz(
                update,
                gram_ns_coefficients,
                gram_ns_reset_iterations,
                ns_coefficients,
                ns_steps,
                eps,
            )
        else:
            update = _zeropower_via_newtonschulz(
                update, ns_coefficients, ns_steps, eps
            )

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
    ns_algorithm: NewtonSchulzAlgorithm = NewtonSchulzAlgorithm.STANDARD,
    gram_ns_coefficients: list[list[float]] | None = None,
    gram_ns_reset_iterations: list[int] | None = None,
    use_cuda_graph: bool = False,
    cuda_graph_cache: dict[_NsCacheKey, Any] | None = None,
) -> None:
    lr = _to_scalar(lr)
    if has_complex:
        raise ValueError("Complex parameters are not supported")
    if not params:
        return
    for g in grads:
        if g.ndim != 2:
            raise ValueError("Param gradient must be a 2D matrix")
    if gram_ns_coefficients is None:
        gram_ns_coefficients = [list(c) for c in DEFAULT_GRAM_NS_COEFFICIENTS]
    if gram_ns_reset_iterations is None:
        gram_ns_reset_iterations = list(DEFAULT_GRAM_NS_RESET_ITERATIONS)

    # Phase 1: momentum buffer update fused across all params.
    # buf <- lerp(buf, grad, 1 - momentum) for every param in one call.
    torch._foreach_lerp_(muon_momentum_bufs, grads, 1 - momentum)
    if nesterov:
        updates = list(torch._foreach_lerp(grads, muon_momentum_bufs, momentum))
    else:
        updates = list(muon_momentum_bufs)

    # Phase 2: shape-grouped NS. For STANDARD this batches same-shape square
    # groups (len > 1) via bmm/baddbmm; rectangular shapes still run per-tensor.
    # For GRAM, rectangular shapes batch into the Gram NS routine; square groups
    # fall back to standard (with the same batching when len > 1). When
    # use_cuda_graph is enabled, each per-shape NS group is captured into a
    # CUDA graph keyed on (count, shape, dtype, device, algorithm, coeffs).
    updates = _run_ns(
        updates,
        ns_algorithm=ns_algorithm,
        ns_coefficients=ns_coefficients,
        ns_steps=ns_steps,
        eps=eps,
        gram_ns_coefficients=gram_ns_coefficients,
        gram_ns_reset_iterations=gram_ns_reset_iterations,
        use_cuda_graph=use_cuda_graph,
        cuda_graph_cache=cuda_graph_cache,
    )

    # Phase 3: weight-decay shrink (uniform scale across all params) followed
    # by the shape-grouped negative-lr add. Group by shape so we can issue one
    # _foreach_add_ per shape group with the shape-specific adjusted_lr.
    torch._foreach_mul_(params, 1 - lr * weight_decay)

    shape_groups: dict[tuple[int, int], list[int]] = {}
    for i, p in enumerate(params):
        shape_groups.setdefault((p.size(0), p.size(1)), []).append(i)
    for indices in shape_groups.values():
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
    ns_algorithm: NewtonSchulzAlgorithm = NewtonSchulzAlgorithm.STANDARD,
    gram_ns_coefficients: list[list[float]] | None = None,
    gram_ns_reset_iterations: list[int] | None = None,
    use_cuda_graph: bool = False,
    cuda_graph_cache: dict[_NsCacheKey, Any] | None = None,
) -> None:
    r"""Functional API that performs Muon algorithm computation.

    See :class:`~torch.optim.Muon` for details.
    """
    # foreach=None preserves the historical single-tensor default; only an
    # explicit True opts into the multi-tensor (_foreach_*) path.
    if foreach:
        _multi_tensor_muon(
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
            ns_algorithm=ns_algorithm,
            gram_ns_coefficients=gram_ns_coefficients,
            gram_ns_reset_iterations=gram_ns_reset_iterations,
            use_cuda_graph=use_cuda_graph,
            cuda_graph_cache=cuda_graph_cache,
        )
        return

    # Single-tensor path does not support CUDA graph capture (graph capture
    # is intrinsically a multi-tensor amortization); the constructor rejects
    # `use_cuda_graph=True` unless `foreach=True`, so we never get here with
    # the flag set.
    _single_tensor_muon(
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
        ns_algorithm=ns_algorithm,
        gram_ns_coefficients=gram_ns_coefficients,
        gram_ns_reset_iterations=gram_ns_reset_iterations,
    )
