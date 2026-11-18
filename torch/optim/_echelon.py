# mypy: allow-untyped-defs
# mypy: disable-error-code=arg-type
"""Implementation of the Echelon optimizer."""

import math
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager

import torch
from torch import Tensor

from ._muon import (
    _adjust_lr,
    _validate_adjust_lr_fn,
    DEFAULT_A,
    DEFAULT_B,
    DEFAULT_C,
    DEFAULT_NS_STEPS,
    EPS,
)
from .optimizer import (
    _disable_dynamo_if_unsupported,
    _functional_api_doc,
    _params_doc,
    _to_scalar,
    Optimizer,
    ParamsT,
)


__all__ = ["Echelon", "echelon"]

# Reference Echelon specification: https://evolvingmachines.ai/blog/echelon
DEFAULT_RANK = 64
DEFAULT_TAU_ITERS = 2
DEFAULT_MARGIN = 1.05
_WARMUP_STEPS = 3
_TAU_FLOOR_MULT = 16.0
_TINY = 1e-30
_FP32_EPS = torch.finfo(torch.float32).eps


@contextmanager
def _disable_autocast(device_type: str) -> Iterator[None]:
    available = torch.amp.autocast_mode.is_autocast_available(device_type)
    if available and torch.is_autocast_enabled(device_type):
        with torch.autocast(device_type=device_type, enabled=False):
            yield
    else:
        yield


def _quintic_newton_schulz(
    X: Tensor,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
) -> Tensor:
    """Apply `ns_steps` odd-quintic Newton-Schulz iterations on pre-normalized `X` (`rows <= cols`)."""
    a, b, c = ns_coefficients
    for _ in range(ns_steps):
        A = X @ X.mT
        B = torch.addmm(A, A, A, beta=b, alpha=c)
        X = torch.addmm(X, B, X, beta=a)
    return X


def _subspace_step(X: Tensor, U: Tensor) -> Tensor:
    """One subspace power step `U <- orth(X X^T U)` as two QR half-steps."""
    V, _ = torch.linalg.qr(X.mT @ U, mode="reduced")
    U_next, _ = torch.linalg.qr(X @ V, mode="reduced")
    return U_next


def _echelon_step(
    X: Tensor,
    U: Tensor | None,
    z: Tensor | None,
    *,
    rank: int,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    tau_iters: int,
    margin: float,
    eps: float,
) -> Tensor:
    """Compute Echelon update for short-side-first momentum `X` (`s x l`, `s <= l`), updating `U` and `z` in place."""
    with _disable_autocast(X.device.type):
        Xf = X.to(dtype=torch.float32)
        s = Xf.size(0)
        x_norm = torch.linalg.vector_norm(Xf)
        denom = x_norm.clamp_min(eps) if eps > 0.0 else x_norm
        if rank >= s:
            return _quintic_newton_schulz(
                Xf / denom.clamp_min(_TINY), ns_coefficients, ns_steps
            )

        if U is None or z is None:
            raise ValueError(
                f"subspace_frame and tail_vec must be provided when rank ({rank}) < short_side ({s})"
            )

        active = torch.isfinite(x_norm) & (x_norm > 0)
        scale = torch.where(active, x_norm / denom.clamp_min(_TINY), 0.0)
        X_unit = Xf / x_norm.clamp_min(_TINY)
        Uf = U.to(dtype=torch.float32)

        # 1. Refresh tracked top-r subspace: V = orth(X^T U), U = orth(X V)
        Uf = torch.where(active, _subspace_step(X_unit, Uf), Uf)

        # 2. Head: H = U^T X, P = NS5(H)
        H = Uf.mT @ X_unit
        H_norm = torch.linalg.vector_norm(H)
        P = _quintic_newton_schulz(
            (H / H_norm.clamp_min(_TINY)) * scale, ns_coefficients, ns_steps
        )

        # 3. Tail: T = X - U H (re-projected once to keep U^T T = 0 to fp32 precision)
        T = X_unit - Uf @ H
        T = T - Uf @ (Uf.mT @ T)
        t_frob = torch.linalg.vector_norm(T)
        tau_floor = _TAU_FLOOR_MULT * math.sqrt(s) * _FP32_EPS * H_norm

        zf = z.to(dtype=torch.float32)
        z_cur = zf / torch.linalg.vector_norm(zf).clamp_min(_TINY)
        est = torch.zeros_like(x_norm)
        for _ in range(tau_iters):
            w = T @ z_cur
            est = torch.linalg.vector_norm(w)
            z_next = T.mT @ w
            z_next_norm = torch.linalg.vector_norm(z_next)
            z_cur = torch.where(
                z_next_norm > 0, z_next / z_next_norm.clamp_min(_TINY), z_cur
            )

        U.copy_(Uf)
        z.copy_(torch.where(active & (t_frob > tau_floor), z_cur, zf))

        # 4. Output: O = U P + T / tau (with lower bound ||T||_F / sqrt(s - r) and ker(T) fallback ||T||_F)
        tail_lb = t_frob / math.sqrt(s - rank)
        est = torch.where(est > 0, torch.maximum(est, tail_lb), t_frob)
        tau = margin * torch.maximum(est, tau_floor).clamp_min(_TINY)
        return Uf @ P + (T / tau) * scale


def _validate_echelon_hyperparameters(
    *,
    lr: float | Tensor,
    momentum: float,
    weight_decay: float,
    adjust_lr_fn: str | None,
) -> None:
    if isinstance(lr, Tensor) and lr.numel() != 1:
        raise ValueError("Tensor lr must be 1-element")
    if not 0.0 <= lr:
        raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
    if not 0.0 <= momentum:
        raise ValueError(f"momentum should be >= 0 but is: {momentum}")
    if not 0.0 <= weight_decay:
        raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")
    _validate_adjust_lr_fn(adjust_lr_fn)


def _validate_echelon_options(
    *,
    rank: int,
    tau_iters: int,
    margin: float,
    ns_steps: int,
    ns_coefficients: tuple[float, float, float],
    eps: float,
) -> None:
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError(f"rank must be an integer >= 1, got: {rank!r}")
    if isinstance(tau_iters, bool) or not isinstance(tau_iters, int) or tau_iters < 1:
        raise ValueError(f"tau_iters must be an integer >= 1, got: {tau_iters!r}")
    if not margin > 0.0:
        raise ValueError(f"margin must be > 0, got: {margin}")
    if (
        isinstance(ns_steps, bool)
        or not isinstance(ns_steps, int)
        or not 1 <= ns_steps < 100
    ):
        raise ValueError(
            f"ns_steps must be an integer in [1, 100) for computational efficiency, got: {ns_steps!r}"
        )
    if len(ns_coefficients) != 3:
        raise ValueError("Coefficients must be a tuple of exactly 3 values")
    if not 0.0 <= eps:
        raise ValueError(f"eps should be >= 0 but is: {eps}")


class Echelon(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        *,
        rank: int = DEFAULT_RANK,
        tau_iters: int = DEFAULT_TAU_ITERS,
        margin: float = DEFAULT_MARGIN,
        ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
        eps: float = EPS,
        ns_steps: int = DEFAULT_NS_STEPS,
        adjust_lr_fn: str | None = None,
    ) -> None:
        _validate_echelon_hyperparameters(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            adjust_lr_fn=adjust_lr_fn,
        )
        _validate_echelon_options(
            rank=rank,
            tau_iters=tau_iters,
            margin=margin,
            ns_steps=ns_steps,
            ns_coefficients=ns_coefficients,
            eps=eps,
        )

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "rank": rank,
            "tau_iters": tau_iters,
            "margin": margin,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        super().__init__(params, defaults)

    @torch._disable_dynamo
    def add_param_group(self, param_group: dict) -> None:
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.

        As :meth:`torch.optim.Optimizer.add_param_group`, but the group's hyperparameters get the
        same checks as the constructor's and all of its parameters must be 2D; a group that fails
        a check raises and is not added.

        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group
                specific optimization options.
        """
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        try:
            _validate_echelon_hyperparameters(
                lr=group["lr"],
                momentum=group["momentum"],
                weight_decay=group["weight_decay"],
                adjust_lr_fn=group["adjust_lr_fn"],
            )
            _validate_echelon_options(
                rank=group["rank"],
                tau_iters=group["tau_iters"],
                margin=group["margin"],
                ns_steps=group["ns_steps"],
                ns_coefficients=group["ns_coefficients"],
                eps=group["eps"],
            )
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Echelon only supports 2D parameters whereas we found a parameter with size: {p.size()}"
                    )
        except Exception:
            self.param_groups.pop()
            raise

    def _init_group(
        self,
        group: MutableMapping,
        group_idx: int,
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        momentum_bufs: list[Tensor],
        subspace_frames: list[Tensor | None],
        tail_vecs: list[Tensor | None],
    ) -> bool:
        rank = group["rank"]
        for param_idx, p in enumerate(group["params"]):
            s, l = min(p.size(0), p.size(1)), max(p.size(0), p.size(1))
            low_rank = p.numel() > 0 and rank < s
            if not low_rank and p in self.state:
                # A frame kept from a smaller `rank` would later be reused without a warmup.
                self.state[p].pop("subspace_frame", None)
                self.state[p].pop("tail_vec", None)
            if p.grad is None:
                continue
            if torch.is_complex(p):
                raise RuntimeError("Echelon does not support complex parameters")
            if p.grad.is_sparse:
                raise RuntimeError("Echelon does not support sparse gradients")

            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(
                    p.grad, memory_format=torch.preserve_format
                )
            momentum_bufs.append(state["momentum_buffer"])

            if low_rank:
                frame = state.get("subspace_frame")
                # Initialize on the first step and after a change of `rank`.
                if frame is None or frame.shape != (s, rank):
                    # A CPU generator gives the same initial frame on every device.
                    gen = torch.Generator(device="cpu")
                    gen.manual_seed(group_idx * 1000003 + param_idx)
                    f32_cpu = {"device": "cpu", "dtype": torch.float32}
                    u0 = torch.randn((s, rank), generator=gen, **f32_cpu)
                    z0 = torch.randn((l, 1), generator=gen, **f32_cpu)
                    with _disable_autocast(p.device.type):
                        U0, _ = torch.linalg.qr(u0.to(device=p.device), mode="reduced")
                    # linalg.qr returns a column-major Q. Store the frame row-major: later steps
                    # copy into it in place, and some checkpointers write state as raw storage.
                    U0 = U0.contiguous()
                    z0 = z0.to(device=p.device)
                    m0 = state["momentum_buffer"].lerp(p.grad, 1 - group["momentum"])
                    if group["nesterov"]:
                        m0 = p.grad.lerp(m0, group["momentum"])
                    X0 = m0.T if p.size(0) > p.size(1) else m0
                    # Warm up the frame and the tail vector on the update this step orthogonalizes,
                    # discarding the updates. A zero or non-finite update leaves them at their
                    # random start.
                    for _ in range(_WARMUP_STEPS):
                        _echelon_step(
                            X0,
                            U0,
                            z0,
                            rank=rank,
                            ns_coefficients=group["ns_coefficients"],
                            ns_steps=group["ns_steps"],
                            tau_iters=group["tau_iters"],
                            margin=group["margin"],
                            eps=group["eps"],
                        )
                    state["subspace_frame"] = U0
                    state["tail_vec"] = z0
                elif frame.dtype != torch.float32:
                    # load_state_dict casts state to the parameter dtype by default.
                    state["subspace_frame"] = frame.to(torch.float32)
                    state["tail_vec"] = state["tail_vec"].to(torch.float32)
                subspace_frames.append(state["subspace_frame"])
                tail_vecs.append(state["tail_vec"])
            else:
                subspace_frames.append(None)
                tail_vecs.append(None)

        return False  # has_complex

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        # Validate every group before any state is touched: an invalid `rank` would otherwise
        # replace the frames, or leave earlier groups stepped.
        for group in self.param_groups:
            _validate_echelon_options(
                rank=group["rank"],
                tau_iters=group["tau_iters"],
                margin=group["margin"],
                ns_steps=group["ns_steps"],
                ns_coefficients=group["ns_coefficients"],
                eps=group["eps"],
            )
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group_idx, group in enumerate(self.param_groups):
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            momentum_bufs: list[Tensor] = []
            subspace_frames: list[Tensor | None] = []
            tail_vecs: list[Tensor | None] = []

            has_complex = self._init_group(
                group,
                group_idx,
                params_with_grad,
                grads,
                momentum_bufs,
                subspace_frames,
                tail_vecs,
            )

            if not params_with_grad:
                continue

            echelon(
                params_with_grad,
                grads,
                momentum_bufs,
                subspace_frames,
                tail_vecs,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                nesterov=group["nesterov"],
                rank=group["rank"],
                tau_iters=group["tau_iters"],
                margin=group["margin"],
                ns_coefficients=group["ns_coefficients"],
                eps=group["eps"],
                ns_steps=group["ns_steps"],
                adjust_lr_fn=group["adjust_lr_fn"],
                has_complex=has_complex,
            )
        return loss


Echelon.__doc__ = (
    r"""Implements the Echelon optimizer: rank-anchored spectral clipping in :math:`O(mnr)` FLOPs.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt} \\
            &\textbf{input}      : \gamma \text{ (lr)},\ \lambda \text{ (weight decay)},\
               \mu \text{ (momentum)},\ \textit{nesterov}\in\{\text{True},\text{False}\},\\
            &\hspace{13mm} r \text{ (rank)},\ k_{\tau} \text{ (tail iters)},\ c_{\tau} \text{ (margin)},\
               \theta_0 \in \mathbb{R}^{m \times n} \\
            &\textbf{initialize} : B_0 \leftarrow 0,\ U_0 \in \mathbb{R}^{s \times r}\ (U_0^\top U_0 = I_r),\
               z_0 \in \mathbb{R}^{l \times 1} \\[-1.ex]
            &\rule{110mm}{0.4pt} \\
            &\textbf{for}\ t=1\ \textbf{to}\ \ldots\ \textbf{do} \\[0.25ex]
            &\hspace{5mm} g_t \leftarrow \nabla_{\theta} f_t(\theta_{t-1}) \\[0.25ex]
            &\hspace{5mm} B_t \leftarrow \mu B_{t-1} + (1 - \mu) g_t \\[0.25ex]
            &\hspace{5mm} \widetilde{B}_t \leftarrow
                \begin{cases}
                   (1 - \mu) g_t + \mu B_t, & \text{if nesterov}=\text{True} \\
                   B_t,                     & \text{if nesterov}=\text{False}
                \end{cases} \\[0.75ex]
            &\hspace{5mm} X_t \leftarrow \widetilde{B}_t \text{ or } \widetilde{B}_t^\top \in \mathbb{R}^{s \times l}\
               (s = \min(m, n),\ l = \max(m, n)) \\[0.25ex]
            &\hspace{5mm} V_t \leftarrow \mathrm{orth}(X_t^\top U_{t-1}),\quad
               U_t \leftarrow \mathrm{orth}(X_t V_t) \\[0.25ex]
            &\hspace{5mm} H_t \leftarrow U_t^\top X_t,\quad
               P_t \leftarrow \mathrm{NS}_k(H_t),\quad
               T_t \leftarrow X_t - U_t H_t \\[0.25ex]
            &\hspace{5mm} \tau_t \leftarrow c_{\tau}\max\big(\widehat{\sigma}_{\max}(T_t),\
               \lVert T_t \rVert_F / \sqrt{s - r},\ \epsilon_{\tau}\big)\
               \text{(} \widehat{\sigma}_{\max} \text{ via } k_{\tau} \text{ power steps)} \\[0.25ex]
            &\hspace{5mm} O_t \leftarrow U_t P_t + T_t / \tau_t\ \text{(transposed back if } m > n\text{)} \\[0.25ex]
            &\hspace{5mm} \theta_t \leftarrow (1 - \gamma\lambda)\theta_{t-1}
               - \mathrm{AdjustLR}(\gamma; \mathrm{shape}(\theta_{t-1}))\, O_t \\
            &\rule{110mm}{0.4pt}
       \end{aligned}

    With an exact top-:math:`r` frame, an exact head polar factor, and :math:`\tau_t = c_{\tau}\sigma_{r+1}`,
    the update equalizes the top-:math:`r` singular values and scales the tail by :math:`1 / (c_{\tau}\sigma_{r+1})`.
    When :math:`r \ge \min(m, n)`, the tail vanishes and Echelon reduces directly to full-rank Newton-Schulz.
    :math:`\mathrm{AdjustLR}` is the learning-rate scaling selected by ``adjust_lr_fn``, as in
    :class:`torch.optim.Muon`, and :math:`\epsilon` (``eps``) floors the Frobenius norm that
    normalizes :math:`X_t` before these steps.

    .. note::
        This reference implementation computes reduced QR, head Newton-Schulz, and explicit tail
        deflation in ``torch.float32`` per parameter, with autocast disabled inside the step (rather
        than the blog post's batched ``bfloat16`` Cholesky-QR / fused kernel). The roundoff floor
        :math:`\epsilon_{\tau} = 16\sqrt{s}\,\epsilon_{\text{fp32}}\lVert H_t \rVert_F` keeps
        rank-deficient momenta from amplifying float32 cancellation noise, and the lower bound
        :math:`\lVert T_t \rVert_F / \sqrt{s - r}` guards against an underestimated
        :math:`\widehat{\sigma}_{\max}(T_t)`; if the power iteration returns zero,
        :math:`\lVert T_t \rVert_F` replaces it. The floor assumes IEEE float32 matmuls. As for other
        optimizers, matmul precision follows the caller's settings
        (``torch.backends.cuda.matmul.fp32_precision``, ``torch.backends.mkldnn.matmul.fp32_precision``
        and :func:`torch.set_float32_matmul_precision`); with TF32 or bfloat16 matmuls, the roundoff
        tail of a rank-deficient momentum is amplified (update spectral norm up to about 1.5 instead
        of 1.1). The first step warms up :math:`U` and :math:`z` on :math:`X_t`, discarding the
        updates (a zero or non-finite first update leaves them at their random start); changing a
        group's ``rank`` re-initializes them the same way at the next step, and parameters with
        ``rank >= s`` drop them. ``subspace_frame`` and ``tail_vec`` are kept in
        ``torch.float32``: if :meth:`load_state_dict` casts them to a low-precision parameter dtype,
        the next step converts them back (:func:`torch.optim.preserve_state_dtype_on_load` avoids the
        round trip). DTensor parameters, for example under FSDP2, are not supported yet.
        Defaults (``lr=1e-3``, ``weight_decay=0.1``) mirror :class:`torch.optim.Muon`;
        the Echelon blog post uses ``lr=0.02``, ``weight_decay=0.0``, and ``rank = max(64, width // 16)``.

    For further details see `The Echelon Optimizer`_.
    """
    + rf"""
    Args:
        {_params_doc}. Echelon supports 2D matrix parameters and is intended for hidden-layer weights.
            Non-2D parameters (biases, norms), as well as 2D embeddings and output heads, should use
            a standard optimizer such as AdamW.
        lr (float, Tensor, optional): learning rate (default: 1e-3).
        weight_decay (float, optional): decoupled weight decay penalty (default: 0.1).
        momentum (float, optional): momentum factor (default: 0.95).
        nesterov (bool, optional): enables Nesterov momentum (default: True).
        rank (int, optional): tracked top singular subspace rank :math:`r`, a Python ``int`` >= 1
            (default: {DEFAULT_RANK}).
        tau_iters (int, optional): power iterations used to estimate the tail spectral norm, a Python
            ``int`` >= 1 (default: {DEFAULT_TAU_ITERS}).
        margin (float, optional): multiplicative safety margin on the tail spectral norm estimate (default: {DEFAULT_MARGIN}).
        ns_coefficients (tuple of three floats, optional): coefficients :math:`(a, b, c)` for the
            Newton-Schulz orthogonalization of the head, or of the whole update when
            :math:`r \ge \min(m, n)` (default: ({DEFAULT_A}, {DEFAULT_B}, {DEFAULT_C})).
        eps (float, optional): floor on the Frobenius norm used to normalize the momentum
            (default: {EPS}).
        ns_steps (int, optional): number of those Newton-Schulz steps, a Python ``int`` in [1, 100)
            (default: {DEFAULT_NS_STEPS}).
        adjust_lr_fn (str, optional): learning-rate scaling rule. One of ``"original"``,
            ``"match_rms_adamw"``, or ``"spectral_unclamped"`` (default: None, equivalent to ``"original"``).

    Example:
        >>> # xdoctest: +SKIP
        >>> hidden_weights = [
        ...     p for n, p in model.named_parameters()
        ...     if p.ndim == 2 and "embed" not in n and "head" not in n
        ... ]
        >>> other_params = [
        ...     p for n, p in model.named_parameters()
        ...     if p.ndim != 2 or "embed" in n or "head" in n
        ... ]
        >>> optim_echelon = torch.optim.Echelon(hidden_weights, lr=0.02, weight_decay=0.0, rank=64)
        >>> optim_adamw = torch.optim.AdamW(other_params, lr=3e-4, weight_decay=0.01)

    .. _The Echelon Optimizer:
        https://evolvingmachines.ai/blog/echelon
    """
)


def _single_tensor_echelon(
    params: list[Tensor],
    grads: list[Tensor],
    momentum_bufs: list[Tensor],
    subspace_frames: list[Tensor | None],
    tail_vecs: list[Tensor | None],
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    rank: int,
    tau_iters: int,
    margin: float,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: str | None,
    has_complex: bool,
) -> None:
    if has_complex:
        raise RuntimeError("Echelon does not support complex parameters")
    if not (
        len(params)
        == len(grads)
        == len(momentum_bufs)
        == len(subspace_frames)
        == len(tail_vecs)
    ):
        raise ValueError(
            "params, grads, momentum_bufs, subspace_frames, and tail_vecs must have matching lengths"
        )
    _validate_echelon_options(
        rank=rank,
        tau_iters=tau_iters,
        margin=margin,
        ns_steps=ns_steps,
        ns_coefficients=ns_coefficients,
        eps=eps,
    )
    for i, (param, grad, buf) in enumerate(zip(params, grads, momentum_bufs)):
        if torch.is_complex(param) or torch.is_complex(grad) or torch.is_complex(buf):
            raise RuntimeError("Echelon does not support complex parameters")
        if grad.is_sparse:
            raise RuntimeError("Echelon does not support sparse gradients")
        if param.ndim != 2 or grad.ndim != 2:
            raise ValueError("Param and gradient must be 2D matrices")
        s, l = min(param.size(0), param.size(1)), max(param.size(0), param.size(1))
        if param.numel() > 0 and rank < s:
            u_i, z_i = subspace_frames[i], tail_vecs[i]
            if u_i is None or z_i is None:
                raise ValueError(
                    f"subspace_frames[{i}] and tail_vecs[{i}] must be provided when rank ({rank}) < short_side ({s})"
                )
            if torch.is_complex(u_i) or torch.is_complex(z_i):
                raise RuntimeError("Echelon does not support complex state tensors")
            if u_i.shape != (s, rank) or z_i.shape != (l, 1):
                raise ValueError(
                    f"Expected subspace_frames[{i}] shape {(s, rank)} and tail_vecs[{i}] shape {(l, 1)}, "
                    f"got {tuple(u_i.shape)} and {tuple(z_i.shape)}"
                )

    lr_val = _to_scalar(lr)
    for i, param in enumerate(params):
        grad = grads[i]
        buf = momentum_bufs[i]
        buf.lerp_(grad, 1 - momentum)
        if param.numel() == 0:
            continue
        update = grad.lerp(buf, momentum) if nesterov else buf

        transposed = update.size(0) > update.size(1)
        X = update.T if transposed else update
        O = _echelon_step(
            X,
            subspace_frames[i],
            tail_vecs[i],
            rank=rank,
            ns_coefficients=ns_coefficients,
            ns_steps=ns_steps,
            tau_iters=tau_iters,
            margin=margin,
            eps=eps,
        )
        if transposed:
            O = O.T

        adjusted_lr = _adjust_lr(lr_val, adjust_lr_fn, param.shape)
        param.mul_(1 - lr_val * weight_decay)
        param.add_(O.to(dtype=param.dtype), alpha=-adjusted_lr)


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_echelon)
def echelon(
    params: list[Tensor],
    grads: list[Tensor],
    momentum_bufs: list[Tensor],
    subspace_frames: list[Tensor | None],
    tail_vecs: list[Tensor | None],
    *,
    foreach: bool | None = None,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    rank: int,
    tau_iters: int,
    margin: float,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: str | None,
    has_complex: bool,
) -> None:
    _validate_adjust_lr_fn(adjust_lr_fn)
    if foreach is not None and foreach:
        raise RuntimeError("Foreach is not supported for Echelon yet")

    _single_tensor_echelon(
        params,
        grads,
        momentum_bufs,
        subspace_frames,
        tail_vecs,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        rank=rank,
        tau_iters=tau_iters,
        margin=margin,
        ns_coefficients=ns_coefficients,
        ns_steps=ns_steps,
        eps=eps,
        adjust_lr_fn=adjust_lr_fn,
        has_complex=has_complex,
    )


echelon.__doc__ = (
    _functional_api_doc.format(optimizer="Echelon")
    + r"""

.. note::
    For each 2D parameter ``p`` with ``s = min(p.shape)`` and ``l = max(p.shape)``:
    when ``rank < s``, ``subspace_frames[i]`` must be a tensor of shape ``(s, rank)``
    with orthonormal columns and ``tail_vecs[i]`` a nonzero tensor of shape ``(l, 1)``.
    Both are updated in place and should be ``torch.float32``: a lower-precision frame is
    rounded to its dtype again at every step. Their shapes are checked, but not that the frame
    is orthonormal or that the tail vector is nonzero. Valid starting values are
    ``torch.linalg.qr(torch.randn(s, rank, device=p.device, dtype=torch.float32))[0]`` and
    ``torch.randn(l, 1, device=p.device, dtype=torch.float32)``. :class:`~torch.optim.Echelon` also warms them up
    on the first step's update; without that warmup the first updates follow a random frame
    and can overshoot. When ``rank >= s``, ``None`` may be passed for both entries.
"""
)
