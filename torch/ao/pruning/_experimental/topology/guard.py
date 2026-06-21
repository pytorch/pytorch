from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Mapping


class LyapunovSpectralGuard:
    r"""Tracks warmup stability and rejects compression when metrics drift."""

    def __init__(
        self,
        *,
        warmup_steps: int = 8,
        max_loss_relative_increase: float = 0.02,
        max_spectral_relative_increase: float = 0.05,
        history_size: int = 16,
    ) -> None:
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive")
        if max_loss_relative_increase < 0:
            raise ValueError("max_loss_relative_increase must be non-negative")
        if max_spectral_relative_increase < 0:
            raise ValueError("max_spectral_relative_increase must be non-negative")
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        if history_size < warmup_steps:
            raise ValueError("history_size must be greater than or equal to warmup_steps")
        self.warmup_steps = warmup_steps
        self.max_loss_relative_increase = max_loss_relative_increase
        self.max_spectral_relative_increase = max_spectral_relative_increase
        self.loss_history: deque[float] = deque(maxlen=history_size)
        self.spectral_history: deque[float] = deque(maxlen=history_size)
        self.is_ready = False

    @staticmethod
    def _as_float(value: float | torch.Tensor) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().float().item())
        return float(value)

    @staticmethod
    def _relative_increase(baseline: float, value: float) -> float:
        denominator = max(abs(baseline), 1e-12)
        return (value - baseline) / denominator

    def observe(self, *, loss: float | torch.Tensor, spectral_energy: float | torch.Tensor) -> bool:
        self.loss_history.append(self._as_float(loss))
        self.spectral_history.append(self._as_float(spectral_energy))
        self.is_ready = len(self.loss_history) >= self.warmup_steps
        return self.is_ready

    def accept(self, *, loss: float | torch.Tensor, spectral_energy: float | torch.Tensor) -> bool:
        if not self.is_ready:
            return False
        current_loss = self._as_float(loss)
        current_spectral = self._as_float(spectral_energy)
        baseline_loss = sum(self.loss_history) / len(self.loss_history)
        baseline_spectral = sum(self.spectral_history) / len(self.spectral_history)
        loss_increase = self._relative_increase(baseline_loss, current_loss)
        spectral_increase = self._relative_increase(baseline_spectral, current_spectral)
        return (
            loss_increase <= self.max_loss_relative_increase
            and spectral_increase <= self.max_spectral_relative_increase
        )

    def reject(self, *, loss: float | torch.Tensor, spectral_energy: float | torch.Tensor) -> bool:
        return not self.accept(loss=loss, spectral_energy=spectral_energy)

    @staticmethod
    def snapshot_module(module: nn.Module) -> dict[str, torch.Tensor]:
        return {name: value.detach().clone() for name, value in module.state_dict().items()}

    @staticmethod
    def rollback_module(module: nn.Module, snapshot: Mapping[str, torch.Tensor]) -> None:
        module.load_state_dict(snapshot, strict=True)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "loss_history": torch.tensor(list(self.loss_history), dtype=torch.float64),
            "spectral_history": torch.tensor(list(self.spectral_history), dtype=torch.float64),
            "is_ready": torch.tensor(self.is_ready),
        }

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        loss_history = state_dict["loss_history"].detach().cpu().tolist()
        spectral_history = state_dict["spectral_history"].detach().cpu().tolist()
        if len(loss_history) > self.loss_history.maxlen or len(spectral_history) > self.spectral_history.maxlen:
            raise ValueError("loaded history is larger than this guard's history_size")
        self.loss_history.clear()
        self.spectral_history.clear()
        self.loss_history.extend(float(value) for value in loss_history)
        self.spectral_history.extend(float(value) for value in spectral_history)
        self.is_ready = bool(state_dict["is_ready"].detach().cpu().item())
