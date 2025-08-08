# mypy: allow-untyped-defs
# mypy: disable-error-code=arg-type
"""Implementation of the Muon optimizer."""

import math
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Callable, cast, Optional
from typing_extensions import TypeAlias

import torch
from torch import Tensor

from .optimizer import _disable_dynamo_if_unsupported, _to_scalar, Optimizer, ParamsT


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
    eps: float = EPS
    ns_steps: int = 5


AdjustLrFn: TypeAlias = Callable[[float, torch.Size], float]
MsignFn: TypeAlias = Callable[[Tensor, BaseMsignFnConfig], Tensor]


def zeropower_via_newtonschulz(grad: Tensor, ns_config: BaseMsignFnConfig) -> Tensor:
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
    eps = ns_config.eps
    coefficients = ns_config.coefficients
    assert steps < 100, (
        "Number of steps must be less than 100 for computational efficiency"
    )
    assert len(grad.shape) == 2, "Input tensor gradient must be a 2D matrix"
    assert len(coefficients) == 3, "Coefficients must be a tuple of exactly 3 values"
    a, b, c = coefficients[0], coefficients[1], coefficients[2]
    X = grad.bfloat16()
    if grad.size(0) > grad.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + eps)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if grad.size(0) > grad.size(1):
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
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        *,
        msign_fn: MsignFn = zeropower_via_newtonschulz,
        msign_fn_config: BaseMsignFnConfig = NewtonSchulzConfig(),
        adjust_lr_fn: AdjustLrFn = default_adjust_lr,
    ) -> None:
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon only supports 2D parameters where as the parameter has size: {p.size()}"
                    )

        self._msign_fn = msign_fn
        self._msign_fn_config = msign_fn_config
        self._adjust_lr_fn = adjust_lr_fn

    def _init_group(
        self,
        group: MutableMapping,
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        muon_momentum_bufs: list[Tensor],
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

            buf = state.get("momentum_buffer")
            if buf is None:
                buf = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
                state["momentum_buffer"] = buf
            muon_momentum_bufs.append(buf)

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
                msign_fn=self._msign_fn,
                msign_fn_config=self._msign_fn_config,
                adjust_lr_fn=self._adjust_lr_fn,
                has_complex=has_complex,
            )
        return loss


def _single_tensor_muon(
    params: list[Tensor],
    grads: list[Tensor],
    muon_momentum_bufs: list[Tensor],
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    msign_fn: Callable[[Tensor, BaseMsignFnConfig], Tensor],
    msign_fn_config: BaseMsignFnConfig,
    adjust_lr_fn: Callable[[float, torch.Size], float],
    has_complex: bool,
) -> None:
    lr = _to_scalar(lr)
    assert has_complex is False, "Complex parameters are not supported"

    for i, param in enumerate(params):
        grad = grads[i]
        assert grad.ndim == 2, "Param gradient must be a 2D matrix"

        buf = muon_momentum_bufs[i]
        buf.mul_(momentum).add_(grad)
        if nesterov:
            grad = grad.add(buf, alpha=momentum)
        else:
            grad = buf

        update = msign_fn(grad, msign_fn_config)
        adjusted_lr = adjust_lr_fn(lr, param.shape)
        param.mul_(1 - lr * weight_decay)
        param.add_(update, alpha=-adjusted_lr)


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_muon)
def muon(
    params: list[Tensor],
    grads: list[Tensor],
    muon_momentum_bufs: list[Tensor],
    foreach: Optional[bool] = None,
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    msign_fn: Callable[[Tensor, BaseMsignFnConfig], Tensor],
    msign_fn_config: BaseMsignFnConfig,
    adjust_lr_fn: Callable[[float, torch.Size], float],
    has_complex: bool,
):
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
        msign_fn=msign_fn,
        msign_fn_config=msign_fn_config,
        adjust_lr_fn=adjust_lr_fn,
        has_complex=has_complex
    )
