# mypy: allow-untyped-defs
# mypy: disable-error-code=arg-type
"""Implementation of the Muon optimizer with optional Gram Newton-Schulz support.

By default this optimizer is API-compatible with prior versions of
torch.optim.Muon and uses the standard Newton-Schulz orthogonalization. Set
use_gram_newton_schulz=True to opt into the Gram Newton-Schulz path for
rectangular matrices (with automatic fallback to standard NS for square
matrices). Pass a GramNewtonSchulzConfig TypedDict via gram_newton_schulz_config
to override gram-specific settings.
"""

import math
from collections.abc import Callable, MutableMapping
from typing import Any, TypedDict

import torch
from torch import Tensor

from .optimizer import (
    _disable_dynamo_if_unsupported,
    _params_doc,
    _to_scalar,
    Optimizer,
    ParamsT,
)


__all__ = ["GramNewtonSchulzConfig", "Muon"]


class GramNewtonSchulzConfig(TypedDict, total=False):
    """Schema for the gram_newton_schulz_config dict consumed by Muon.

    All keys are optional; missing keys fall back to module-level defaults
    (see DEFAULT_GRAM_NS_COEFFICIENTS, DEFAULT_GRAM_NS_RESET_ITERATIONS).
    """

    gram_ns_coefficients: list[list[float]]
    gram_ns_reset_iterations: list[int]
    use_cuda_graph: bool


# Constants from Keller Jordan's Muon post: https://kellerjordan.github.io/posts/muon/
# github permlink: https://github.com/KellerJordan/Muon/blob/f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd/muon.py#L16
EPS: float = 1e-7
DEFAULT_A: float = 3.4445
DEFAULT_B: float = -4.7750
DEFAULT_C: float = 2.0315
DEFAULT_NS_STEPS: int = 5
DEFAULT_GRAM_NS_RESET_ITERATIONS: list[int] = []
# Default per-iteration (a, b, c) for Gram NS: same constants as vanilla Muon
# repeated for every NS step. Schedules like YOU_COEFFICIENTS can be opted into
# via gram_newton_schulz_config["gram_ns_coefficients"], but they typically
# require a non-empty gram_ns_reset_iterations (e.g. [2]) to stay numerically
# stable. Keeping the default identical to vanilla Muon makes Gram NS a drop-in
# replacement that does not depend on a reset schedule.
DEFAULT_GRAM_NS_COEFFICIENTS: list[list[float]] = [
    [DEFAULT_A, DEFAULT_B, DEFAULT_C] for _ in range(DEFAULT_NS_STEPS)
]

# Valid keys in gram_newton_schulz_config, derived from the TypedDict schema.
_GRAM_CONFIG_KEYS: frozenset[str] = frozenset(GramNewtonSchulzConfig.__annotations__)


def _gram_newton_schulz(
    X: Tensor,
    gram_ns_coefficients: list[list[float]],
    gram_ns_reset_iterations: list[int],
) -> Tensor:
    """Gram Newton-Schulz orthogonalization.

    Works in the Gram space (R = X @ X^T), accumulating Q and applying X = Q @ X
    once at the end. More efficient for rectangular matrices where M << N or M >> N.

    X must be a 2D matrix.
    """
    R = X @ X.T

    I = torch.eye(R.size(0), device=X.device, dtype=X.dtype)  # noqa: E741
    Q: Tensor = I

    reset_iterations = gram_ns_reset_iterations

    for i, (a, b, c) in enumerate(gram_ns_coefficients):
        if i in reset_iterations and i != 0:
            X = Q @ X
            R = X @ X.T
            Q = I

        Z = torch.addmm(R, R, R, beta=b, alpha=c)
        if i == 0 or i in reset_iterations:
            Q = Z + a * I
        else:
            Q = torch.addmm(Q, Q, Z, beta=a)
        if i < len(gram_ns_coefficients) - 1 and i + 1 not in reset_iterations:
            RZ = torch.addmm(R, R, Z, beta=a)
            R = torch.addmm(RZ, Z, RZ, beta=a)

    X = Q @ X

    return X


def _gram_newton_schulz_batched(
    X: Tensor,
    gram_ns_coefficients: list[list[float]],
    gram_ns_reset_iterations: list[int],
) -> Tensor:
    """Batched Gram Newton-Schulz orthogonalization.

    X must be a 3D tensor of shape (batch, M, N) where M <= N.
    Processes all matrices in the batch simultaneously using batched matmuls.
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

    reset_iterations = gram_ns_reset_iterations

    for i, (a, b, c) in enumerate(gram_ns_coefficients):
        if i in reset_iterations and i != 0:
            X = Q @ X
            R = X @ X.mT
            Q = I

        Z = torch.baddbmm(R, R, R, beta=b, alpha=c)
        if i == 0 or i in reset_iterations:
            Q = Z + a * I
        else:
            Q = torch.baddbmm(Q, Q, Z, beta=a)
        if i < len(gram_ns_coefficients) - 1 and i + 1 not in reset_iterations:
            RZ = torch.baddbmm(R, R, Z, beta=a)
            R = torch.baddbmm(RZ, Z, RZ, beta=a)

    X = Q @ X

    return X


def _zeropower_via_gram_newtonschulz_batched(
    grads: list[Tensor],
    gram_ns_coefficients: list[list[float]],
    gram_ns_reset_iterations: list[int],
    eps: float,
) -> list[Tensor]:
    """Batched Gram NS orthogonalization for a list of same-shape 2D gradients.

    Stacks gradients into a 3D batch, runs batched Gram NS, then unstacks.
    """
    # Check if transposing is needed (all same shape, so check first)
    should_transpose = grads[0].size(0) > grads[0].size(1)

    # Stack into (batch, M, N)
    X = torch.stack([g.float() for g in grads])  # (batch, M, N)

    if should_transpose:
        X = X.mT  # (batch, N, M) -> now rows <= cols

    # Normalize each matrix in the batch
    norms = X.norm(dim=(-2, -1), keepdim=True)  # (batch, 1, 1)
    X = X / (norms + eps)
    X = X.half()

    X = _gram_newton_schulz_batched(X, gram_ns_coefficients, gram_ns_reset_iterations)

    if should_transpose:
        X = X.mT

    return list(X.unbind(0))


def _zeropower_via_newtonschulz(
    grad: Tensor,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
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


def _zeropower_via_gram_newtonschulz(
    grad: Tensor,
    gram_ns_coefficients: list[list[float]],
    gram_ns_reset_iterations: list[int],
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> Tensor:
    """Orthogonalize a gradient matrix using Gram Newton-Schulz iteration.

    Uses the Gram NS method for rectangular matrices and falls back to
    standard NS for square matrices.
    """
    if len(grad.shape) != 2:
        raise ValueError("Input tensor gradient must be a 2D matrix")

    X = grad.float()

    if should_transpose := (X.size(0) > X.size(1)):
        X = X.T

    X /= X.norm() + eps
    X = X.half()

    if X.size(0) == X.size(1):
        return _zeropower_via_newtonschulz(grad, ns_coefficients, ns_steps, eps)

    X = _gram_newton_schulz(X, gram_ns_coefficients, gram_ns_reset_iterations)

    if should_transpose:
        X = X.T

    return X


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


def _parse_gram_newton_schulz_config(
    gram_newton_schulz_config: GramNewtonSchulzConfig | None,
) -> tuple[list[list[float]], list[int], bool]:
    """Validate and unpack a gram_newton_schulz_config dict.

    Returns (gram_ns_coefficients, gram_ns_reset_iterations, use_cuda_graph),
    filling in defaults for missing keys. None or {} yields all defaults.
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
    use_cuda_graph = bool(cfg.get("use_cuda_graph", False))
    return gram_ns_coefficients, gram_ns_reset_iterations, use_cuda_graph


class Muon(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (
            DEFAULT_A,
            DEFAULT_B,
            DEFAULT_C,
        ),
        eps: float = EPS,
        ns_steps: int = DEFAULT_NS_STEPS,
        adjust_lr_fn: str | None = None,
        use_gram_newton_schulz: bool = False,
        gram_newton_schulz_config: GramNewtonSchulzConfig | None = None,
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

        if not use_gram_newton_schulz and gram_newton_schulz_config is not None:
            # When gram NS is disabled, the optimizer is bit-compatible with the
            # vanilla pytorch path - no gram-specific knobs (including CUDA graph)
            # apply. Reject the dict so callers can't silently configure a
            # setting that has no effect.
            raise ValueError(
                "gram_newton_schulz_config is only valid when "
                "use_gram_newton_schulz=True; got "
                f"gram_newton_schulz_config={gram_newton_schulz_config!r} "
                "with use_gram_newton_schulz=False."
            )

        gram_ns_coefficients, gram_ns_reset_iterations, use_cuda_graph = (
            _parse_gram_newton_schulz_config(gram_newton_schulz_config)
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
            "use_gram_newton_schulz": use_gram_newton_schulz,
            "gram_ns_coefficients": gram_ns_coefficients,
            "gram_ns_reset_iterations": gram_ns_reset_iterations,
            "use_cuda_graph": use_cuda_graph,
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon only supports 2D parameters whereas we found a parameter with size: {p.size()}"
                    )

        # CUDA graph cache: keyed by NS config, stores
        # (graph, static_input_bufs, static_output_bufs).
        self._cuda_graph_cache: dict[
            tuple[Any, ...],
            tuple[torch.cuda.CUDAGraph, list[Tensor], list[Tensor]],
        ] = {}

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
    def step(self, closure: Callable | None = None) -> float | None:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
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
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                nesterov=group["nesterov"],
                ns_coefficients=group["ns_coefficients"],
                ns_steps=group["ns_steps"],
                eps=group["eps"],
                adjust_lr_fn=group["adjust_lr_fn"],
                has_complex=has_complex,
                use_gram_newton_schulz=group["use_gram_newton_schulz"],
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

    Setting ``use_gram_newton_schulz=True`` selects an alternate orthogonalization
    path that computes the Newton-Schulz iteration in Gram space (R = X @ X^T)
    for rectangular matrices, with automatic fallback to the standard iteration
    on square matrices. The Gram path also fuses momentum and parameter updates
    across same-shape 2D parameters via ``torch._foreach_*`` ops, and can
    optionally capture each shape group's NS computation into a CUDA graph for
    replay on subsequent steps.

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
        use_gram_newton_schulz (bool, optional): if True, use the Gram Newton-Schulz
            iteration for orthogonalization on rectangular matrices, with automatic
            fallback to standard Newton-Schulz on square matrices. The Gram path
            also fuses the per-step momentum and parameter updates across
            same-shape 2D params via ``torch._foreach_*`` ops, and may optionally
            capture the orthogonalization kernels into a CUDA graph. (default: False)
        gram_newton_schulz_config (dict, optional): configuration dict for the
            Gram Newton-Schulz path. Only valid when ``use_gram_newton_schulz=True``;
            passing it with the flag off raises ``ValueError``. Supported keys
            (all optional, with defaults applied per key):

            - ``"gram_ns_coefficients"`` (list[list[float]]): per-iteration
              ``(a, b, c)`` triples; one inner list per NS step.
            - ``"gram_ns_reset_iterations"`` (list[int]): iteration indices at
              which to materialize ``X`` from the accumulated ``Q`` and reset
              ``Q`` to identity.
            - ``"use_cuda_graph"`` (bool): if True, capture each per-shape NS
              computation into a CUDA graph and replay on subsequent steps.

            (default: None)

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


def _ns_cudagraph(
    inputs: list[Tensor],
    compute_fn: Callable[[list[Tensor]], list[Tensor]],
    cache_key: tuple[Any, ...],
    cuda_graph_cache: dict[
        tuple[Any, ...],
        tuple[torch.cuda.CUDAGraph, list[Tensor], list[Tensor]],
    ],
) -> list[Tensor]:
    """Run an NS-style computation with CUDA graph capture/replay.

    Wraps an arbitrary `compute_fn` (which must be safe to call inside a
    CUDA graph capture region and must return a list of newly-allocated
    tensors) with cache lookup, static-buffer allocation, warmup, capture,
    and replay. Returns clones of the static output buffers so callers may
    safely retain the results across subsequent calls.
    """
    if cache_key not in cuda_graph_cache:
        # Static input buffers - the exact tensors captured by the graph.
        static_inputs = [t.clone() for t in inputs]
        # Warmup is required before capture.
        compute_fn(static_inputs)
        torch.cuda.synchronize()

        # Capture: static_inputs are read, static_outputs are written.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_outputs = compute_fn(static_inputs)
        cuda_graph_cache[cache_key] = (graph, static_inputs, static_outputs)

    graph, static_inputs, static_outputs = cuda_graph_cache[cache_key]
    # Copy new data into the static input buffers the graph reads from.
    for si, t in zip(static_inputs, inputs):
        si.copy_(t)
    graph.replay()
    return [so.clone() for so in static_outputs]


def _ns_eager(
    inputs: list[Tensor],
    *,
    is_square: bool,
    gram_ns_coefficients: list[list[float]],
    gram_ns_reset_iterations: list[int],
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> list[Tensor]:
    """Eager (non-CUDA-graph) NS dispatch for a same-shape group of tensors.

    Routes based on shape:
    - Square shapes: standard NS per tensor.
    - Rectangular, len > 1: batched gram NS.
    - Rectangular, len == 1: single gram NS.
    """
    if is_square:
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
        inputs,
        gram_ns_coefficients,
        gram_ns_reset_iterations,
        eps,
    )


def _dispatch_ns_group(
    inputs: list[Tensor],
    *,
    use_cuda_graph: bool,
    cuda_graph_cache: dict[
        tuple[Any, ...],
        tuple[torch.cuda.CUDAGraph, list[Tensor], list[Tensor]],
    ]
    | None,
    gram_ns_coefficients: list[list[float]],
    gram_ns_reset_iterations: list[int],
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> list[Tensor]:
    """Dispatch one same-shape group through CUDA graph or eager.

    Square groups fall back to standard NS individually and are not worth
    graph capture, so they always run eager. Rectangular groups use CUDA
    graph when enabled.
    """
    is_square: bool = inputs[0].size(0) == inputs[0].size(1)

    def _compute(buffers: list[Tensor]) -> list[Tensor]:
        return _ns_eager(
            buffers,
            is_square=is_square,
            gram_ns_coefficients=gram_ns_coefficients,
            gram_ns_reset_iterations=gram_ns_reset_iterations,
            ns_coefficients=ns_coefficients,
            ns_steps=ns_steps,
            eps=eps,
        )

    graph_eligible = use_cuda_graph and cuda_graph_cache is not None and not is_square
    if not graph_eligible:
        return _compute(inputs)

    cache_key = (
        len(inputs),
        tuple(inputs[0].shape),
        inputs[0].device,
        inputs[0].dtype,
        tuple(tuple(c) for c in gram_ns_coefficients),
        tuple(gram_ns_reset_iterations),
        ns_coefficients,
        ns_steps,
        eps,
    )
    assert cuda_graph_cache is not None  # narrowed by graph_eligible check
    return _ns_cudagraph(inputs, _compute, cache_key, cuda_graph_cache)


def _run_ns(
    updates: list[Tensor],
    *,
    use_cuda_graph: bool,
    cuda_graph_cache: dict[
        tuple[Any, ...],
        tuple[torch.cuda.CUDAGraph, list[Tensor], list[Tensor]],
    ]
    | None,
    gram_ns_coefficients: list[list[float]],
    gram_ns_reset_iterations: list[int],
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> list[Tensor]:
    """Top-level NS dispatcher.

    Groups updates by shape so same-shape rectangular tensors can be batched
    through Gram NS. Each group is routed through _dispatch_ns_group.
    """
    shape_groups: dict[tuple[int, int], list[int]] = {}
    for i, update in enumerate(updates):
        shape_groups.setdefault((update.size(0), update.size(1)), []).append(i)
    index_groups = list(shape_groups.values())

    results = list(updates)
    for indices in index_groups:
        group_inputs = [updates[idx] for idx in indices]
        group_results = _dispatch_ns_group(
            group_inputs,
            use_cuda_graph=use_cuda_graph,
            cuda_graph_cache=cuda_graph_cache,
            gram_ns_coefficients=gram_ns_coefficients,
            gram_ns_reset_iterations=gram_ns_reset_iterations,
            ns_coefficients=ns_coefficients,
            ns_steps=ns_steps,
            eps=eps,
        )
        for idx, r in zip(indices, group_results):
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


def _gram_muon_impl(
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
    gram_ns_coefficients: list[list[float]],
    gram_ns_reset_iterations: list[int],
    use_cuda_graph: bool,
    cuda_graph_cache: dict[
        tuple[Any, ...],
        tuple[torch.cuda.CUDAGraph, list[Tensor], list[Tensor]],
    ]
    | None,
) -> None:
    """Gram NS Muon implementation.

    Multi-tensor (foreach) variant of muon: Phase 1 fuses momentum updates
    across all params with torch._foreach_lerp_, Phase 2 groups by shape and
    runs the Gram NS orthogonalizer per group (optionally captured into a
    CUDA graph; square shapes fall back to standard NS per tensor), and
    Phase 3 fuses the parameter update with torch._foreach_mul_ for the
    uniform weight-decay scale and one torch._foreach_add_ per shape group
    for the shape-specific adjusted learning rate.
    """
    lr = _to_scalar(lr)
    if has_complex:
        raise ValueError("Complex parameters are not supported")

    if not params:
        return

    for g in grads:
        if g.ndim != 2:
            raise ValueError("Param gradient must be a 2D matrix")

    # Phase 1: foreach momentum-applied updates.
    # buf <- lerp(buf, grad, 1 - momentum) for every param in one fused call.
    torch._foreach_lerp_(muon_momentum_bufs, grads, 1 - momentum)
    if nesterov:
        # update = lerp(grad, buf, momentum); returns a fresh list of tensors.
        updates = list(torch._foreach_lerp(grads, muon_momentum_bufs, momentum))
    else:
        updates = list(muon_momentum_bufs)

    # Phase 2: shape-grouped (and optionally graph-captured) NS dispatch.
    updates = _run_ns(
        updates,
        use_cuda_graph=use_cuda_graph,
        cuda_graph_cache=cuda_graph_cache,
        gram_ns_coefficients=gram_ns_coefficients,
        gram_ns_reset_iterations=gram_ns_reset_iterations,
        ns_coefficients=ns_coefficients,
        ns_steps=ns_steps,
        eps=eps,
    )

    # Phase 3: foreach parameter update. Weight-decay scale (1 - lr*wd) is
    # uniform across params so it fuses across the whole list. The
    # learning-rate add uses shape-specific alpha (via _adjust_lr) so we
    # group params by shape and issue one foreach_add per shape group.
    torch._foreach_mul_(params, 1 - lr * weight_decay)

    shape_groups: dict[tuple[int, int], list[int]] = {}
    for i, p in enumerate(params):
        shape_groups.setdefault((p.size(0), p.size(1)), []).append(i)
    for indices in shape_groups.values():
        adjusted_lr = _adjust_lr(lr, adjust_lr_fn, params[indices[0]].shape)
        torch._foreach_add_(
            [params[i] for i in indices],
            [updates[i] for i in indices],
            alpha=-adjusted_lr,
        )


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
    use_gram_newton_schulz: bool = False,
    gram_ns_coefficients: list[list[float]],
    gram_ns_reset_iterations: list[int],
    use_cuda_graph: bool = False,
    cuda_graph_cache: dict[
        tuple[Any, ...],
        tuple[torch.cuda.CUDAGraph, list[Tensor], list[Tensor]],
    ]
    | None = None,
) -> None:
    r"""Functional API that performs Muon algorithm computation.

    Dispatches to the vanilla pytorch path (`_single_tensor_muon`) when
    `use_gram_newton_schulz=False` and to the gram path (`_gram_muon_impl`)
    when True. `gram_ns_coefficients` and `gram_ns_reset_iterations` are
    required and only consulted on the gram path; pre-validated values come
    from `_parse_gram_newton_schulz_config` (called in `Muon.__init__`).

    See :class:`~torch.optim.Muon` for details.
    """
    if foreach is not None and foreach:
        raise RuntimeError("Foreach is not supported for Muon yet")

    if not use_gram_newton_schulz:
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
        )
        return

    _gram_muon_impl(
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
        gram_ns_coefficients=gram_ns_coefficients,
        gram_ns_reset_iterations=gram_ns_reset_iterations,
        use_cuda_graph=use_cuda_graph,
        cuda_graph_cache=cuda_graph_cache,
    )
