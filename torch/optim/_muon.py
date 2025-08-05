# mypy: allow-untyped-defs
# mypy: disable-error-code=arg-type
"""Implementation of the Muon optimizer."""

import math
import warnings
from collections.abc import Iterable, MutableMapping
from dataclasses import dataclass
from typing import Callable, cast, Optional
from typing_extensions import TypeAlias

import torch
from torch import Tensor

from .optimizer import _get_scalar_dtype, _to_scalar, Optimizer, ParamsT


__all__ = ["Muon"]

EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315


@dataclass
class BaseMsignFnConfig:
    """Configuration used by :func:`msign_fn`."""


@dataclass
class NewtonSchulzConfig(BaseMsignFnConfig):
    # """Configuration used by :func:`zeropower_via_newtonschulz`."""

    coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C)
    ns_steps: int = 5


AdjustLrFn: TypeAlias = Callable[[float, torch.Size], float]
MsignFn: TypeAlias = Callable[[Tensor, BaseMsignFnConfig], Tensor]


def zeropower_via_newtonschulz(G: Tensor, ns_config: BaseMsignFnConfig) -> Tensor:
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
    ns_config = cast(NewtonSchulzConfig, ns_config)
    steps = ns_config.ns_steps
    coefficients = ns_config.coefficients
    assert steps < 100, (
        "Number of steps must be less than 100 for computational efficiency"
    )
    assert len(G.shape) == 2, "Input tensor gradient must be a 2D matrix"
    assert len(coefficients) == 3, "Coefficients must be a tuple of exactly 3 values"
    a, b, c = coefficients[0], coefficients[1], coefficients[2]
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + EPS)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


def default_adjust_lr(lr: float, param_shape: torch.Size) -> float:
    """Default learning rate adjustment used by Muon. Method reported in the paper https://arxiv.org/pdf/2502.16982."""
    A, B = param_shape[:2]
    adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    return lr * adjusted_ratio


class Muon(Optimizer):
    """Implements the Muon optimizer.

    This optimizer performs momentum SGD followed by an optional orthogonalization
    step computed via a user provided callable.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        wd: float = 0.1,
        muon_param_fqns: Optional[Iterable[str]] = None,
        momentum: float = 0.95,
        nesterov: bool = True,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        *,
        msign_fn: MsignFn = zeropower_via_newtonschulz,
        msign_fn_config: BaseMsignFnConfig = NewtonSchulzConfig(),
        adjust_lr_fn: AdjustLrFn = default_adjust_lr,
    ) -> None:
        named_params = list(
            cast(Iterable[tuple[str, torch.Tensor]], params)
            if params is not None
            else []
        )
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if not 0.0 <= wd:
            raise ValueError(f"wd (weight decay) should be >= 0 but is: {wd}")
        if not 0.0 <= adamw_eps:
            raise ValueError(f"AdamW epsilon should be >= 0 but is: {adamw_eps}")
        if not 0.0 <= adamw_betas[0] < 1.0:
            raise ValueError(f"Invalid AdamW beta[0]: {adamw_betas[0]}")
        if not 0.0 <= adamw_betas[1] < 1.0:
            raise ValueError(f"Invalid AdamW beta[1]: {adamw_betas[1]}")
        if not all(
            (isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str))
            or (isinstance(item, dict) and all(isinstance(k, str) for k in item))
            for item in named_params
        ):
            raise RuntimeError("Expected params to be named parameters")

        defaults = {
            "lr": lr,
            "wd": wd,
            "momentum": momentum,
            "nesterov": nesterov,
            "adamw_betas": adamw_betas,
            "adamw_eps": adamw_eps,
        }
        super().__init__(named_params, defaults)

        # Note: Muon doesn't support multiple param groups for now.
        if muon_param_fqns is not None:
            muon_param_fqns_set = set(muon_param_fqns)
            for name, p in named_params:
                self.state[p]["use_muon"] = name in muon_param_fqns_set
                if self.state[p]["use_muon"]:
                    assert p.ndim == 2, "Param optimized by Muon must be 2D."
        else:
            warnings.warn(
                "No Muon params FQNs provided. Using Muon to optimize all 2D parameters. "
                "Note that this may not be the expected behavior since some 2D parameters "
                "are not intended to be optimized with Muon, for example word embedding. "
                "Optimizing these parameters with Muon may cause model performance degradation. "
                "We recommend users to explicitly specify the muon_param_fqns for parameters "
                "to be optimized by Muon."
            )
            for _, p in named_params:
                self.state[p]["use_muon"] = p.ndim == 2

        self._msign_fn = msign_fn
        self._msign_fn_config = msign_fn_config
        self._adjust_lr_fn = adjust_lr_fn

    def _init_group(
        self,
        group: MutableMapping,
        muon_params: list[Tensor],
        muon_grads: list[Tensor],
        muon_momentum_bufs: list[Tensor],
        other_params: list[Tensor],
        other_grads: list[Tensor],
        exp_avgs: list[Tensor],
        exp_avg_sqs: list[Tensor],
        steps: list[Tensor],
    ) -> None:
        for p in group["params"]:
            if p.grad is None:
                continue

            if torch.is_complex(p):
                raise RuntimeError("Muon does not support complex parameters")
            if p.grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")

            state = self.state[p]

            if state["use_muon"]:
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
                    state["momentum_buffer"] = buf
                muon_params.append(p)
                muon_grads.append(p.grad)
                muon_momentum_bufs.append(buf)
            else:
                # for the rest of the parameters, we use AdamW to optimize.
                step = state.get("step")
                if step is None:
                    step = torch.zeros((), dtype=_get_scalar_dtype(), device=p.device)
                    state["step"] = step
                steps.append(step)
                exp_avg = state.get("moment1")
                exp_avg_sq = state.get("moment2")
                if exp_avg is None:
                    exp_avg = torch.zeros_like(p.grad)
                    exp_avg_sq = torch.zeros_like(p.grad)
                    state["moment1"] = exp_avg
                    state["moment2"] = exp_avg_sq

                other_params.append(p)
                other_grads.append(p.grad)
                exp_avgs.append(exp_avg)
                exp_avg_sqs.append(exp_avg_sq)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]

            muon_params: list[Tensor] = []
            muon_grads: list[Tensor] = []
            muon_momentum_bufs: list[Tensor] = []
            other_params: list[Tensor] = []
            other_grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            steps: list[Tensor] = []

            self._init_group(
                group,
                muon_params,
                muon_grads,
                muon_momentum_bufs,
                other_params,
                other_grads,
                exp_avgs,
                exp_avg_sqs,
                steps,
            )

            _single_tensor_muon(
                muon_params,
                muon_grads,
                muon_momentum_bufs,
                other_params,
                other_grads,
                exp_avgs,
                exp_avg_sqs,
                steps,
                lr=lr,
                wd=wd,
                momentum=momentum,
                nesterov=group["nesterov"],
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                msign_fn=self._msign_fn,
                msign_fn_config=self._msign_fn_config,
                adjust_lr_fn=self._adjust_lr_fn,
            )
        return loss


def _single_tensor_muon(
    muon_params: list[Tensor],
    muon_grads: list[Tensor],
    muon_momentum_bufs: list[Tensor],
    other_params: list[Tensor],
    other_grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    steps: list[Tensor],
    *,
    lr: float,
    wd: float,
    momentum: float,
    nesterov: bool,
    beta1: float,
    beta2: float,
    eps: float,
    msign_fn: Callable[[Tensor, BaseMsignFnConfig], Tensor],
    msign_fn_config: BaseMsignFnConfig,
    adjust_lr_fn: Callable[[float, torch.Size], float],
) -> None:
    lr = _to_scalar(lr)
    for i, param in enumerate(muon_params):
        grad = muon_grads[i]
        assert grad.ndim == 2, "Param gradient must be a 2D matrix"

        buf = muon_momentum_bufs[i]
        buf.mul_(momentum).add_(grad)
        if nesterov:
            grad = grad.add(buf, alpha=momentum)
        else:
            grad = buf

        update = msign_fn(grad, msign_fn_config)
        adjusted_lr = adjust_lr_fn(lr, param.shape)
        param.mul_(1 - lr * wd)
        param.add_(update, alpha=-adjusted_lr)

    for i, param in enumerate(other_params):
        grad = other_grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = steps[i]
        step += 1

        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.lerp_(grad.square(), 1 - beta2)

        bias_correction1 = 1 - beta1 ** step.item()
        bias_correction2 = 1 - beta2 ** step.item()
        bias_correction2_sqrt = bias_correction2**0.5
        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
        step_size = lr / bias_correction1

        param.mul_(1 - lr * wd)
        param.addcdiv_(exp_avg, denom, value=-step_size)
