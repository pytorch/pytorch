from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from torch.optim import Optimizer


def _optimizer_contains_parameter(optimizer: Optimizer, parameter: nn.Parameter) -> bool:
    return any(param is parameter for group in optimizer.param_groups for param in group["params"])


def _replace_optimizer_parameters(
    optimizer: Optimizer,
    old_parameter: nn.Parameter,
    new_parameters: tuple[nn.Parameter, ...],
) -> None:
    found = False
    inserted = False
    for group in optimizer.param_groups:
        params = group["params"]
        next_params = []
        for param in params:
            if param is old_parameter:
                found = True
                if not inserted:
                    next_params.extend(new_parameters)
                    inserted = True
            else:
                next_params.append(param)
        params[:] = next_params
    if not found:
        raise ValueError("optimizer does not contain the parameter being compressed")
    optimizer.state.pop(old_parameter, None)


class TopologyGatedLowRankLinear(nn.Module):
    r"""Linear layer that only switches to low-rank factors after a spectral gate.

    The module starts as an exact dense copy. Calling :meth:`try_compress` computes
    an SVD of the dense weight and replaces it with two factors only if the kept
    singular spectrum preserves enough energy and reduces parameter count by the
    requested ratio.
    """

    in_features: int
    out_features: int
    is_compressed: bool

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        energy_threshold: float = 0.98,
        max_rank: int | None = None,
        min_compression_ratio: float = 1.05,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 < energy_threshold <= 1.0:
            raise ValueError("energy_threshold must be in (0, 1]")
        if max_rank is not None and max_rank <= 0:
            raise ValueError("max_rank must be positive")
        if min_compression_ratio <= 1.0:
            raise ValueError("min_compression_ratio must be greater than 1")
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.energy_threshold = energy_threshold
        self.max_rank = max_rank
        self.min_compression_ratio = min_compression_ratio
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs)) if bias else None
        self.low_rank_left: nn.Parameter | None = None
        self.low_rank_right: nn.Parameter | None = None
        self.is_compressed = False
        self.reset_parameters()

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        energy_threshold: float = 0.98,
        max_rank: int | None = None,
        min_compression_ratio: float = 1.05,
    ) -> TopologyGatedLowRankLinear:
        module = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            energy_threshold=energy_threshold,
            max_rank=max_rank,
            min_compression_ratio=min_compression_ratio,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            module.weight.copy_(linear.weight)
            if linear.bias is not None and module.bias is not None:
                module.bias.copy_(linear.bias)
        return module

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def dense_parameter_count(self) -> int:
        bias_count = 0 if self.bias is None else self.bias.numel()
        return self.out_features * self.in_features + bias_count

    def compressed_parameter_count(self, rank: int | None = None) -> int:
        if rank is None:
            if self.low_rank_left is None or self.low_rank_right is None:
                rank = min(self.out_features, self.in_features)
            else:
                rank = self.low_rank_right.shape[0]
        bias_count = 0 if self.bias is None else self.bias.numel()
        return rank * (self.out_features + self.in_features) + bias_count

    def _target_rank(self, singular_values: torch.Tensor) -> int | None:
        if singular_values.numel() == 0:
            return None
        energy = singular_values.square()
        total = energy.sum()
        if total <= 0:
            return None
        cumulative = torch.cumsum(energy, dim=0) / total
        required_rank = int(torch.searchsorted(cumulative, torch.tensor(self.energy_threshold, device=cumulative.device)).item()) + 1
        if required_rank <= 0:
            return None
        if self.max_rank is not None and self.max_rank < required_rank:
            return None
        rank = required_rank
        if self.compressed_parameter_count(rank) * self.min_compression_ratio >= self.dense_parameter_count():
            return None
        return rank

    @torch.no_grad()
    def try_compress(self, optimizer: Optimizer | None = None) -> bool:
        if self.is_compressed:
            return True
        if optimizer is not None and not _optimizer_contains_parameter(optimizer, self.weight):
            raise ValueError("optimizer does not contain the dense weight parameter")
        weight = self.weight.detach()
        svd_input = weight.float() if weight.dtype in (torch.float16, torch.bfloat16) else weight
        u, s, vh = torch.linalg.svd(svd_input, full_matrices=False)
        rank = self._target_rank(s)
        if rank is None:
            return False
        sqrt_s = torch.sqrt(s[:rank])
        left = (u[:, :rank] * sqrt_s.unsqueeze(0)).to(dtype=weight.dtype)
        right = (sqrt_s.unsqueeze(1) * vh[:rank, :]).to(dtype=weight.dtype)
        old_weight = self.weight
        del self._parameters["weight"]
        self.weight = None  # type: ignore[assignment]
        self.low_rank_left = nn.Parameter(left.contiguous())
        self.low_rank_right = nn.Parameter(right.contiguous())
        self.is_compressed = True
        if optimizer is not None:
            _replace_optimizer_parameters(optimizer, old_weight, (self.low_rank_left, self.low_rank_right))
        return True

    def _checkpoint_factory_kwargs(
        self,
        fallback: torch.Tensor | None = None,
        *,
        assign: bool = False,
    ) -> dict[str, torch.device | torch.dtype]:
        if assign and fallback is not None:
            return {"device": fallback.device, "dtype": fallback.dtype}
        reference = None
        for candidate in (self.weight, self.low_rank_left, self.low_rank_right, self.bias):
            if isinstance(candidate, torch.Tensor):
                reference = candidate
                break
        if reference is None:
            reference = fallback
        if reference is None:
            return {}
        return {"device": reference.device, "dtype": reference.dtype}

    def _prepare_dense_parameters(self, weight: torch.Tensor | None = None, *, assign: bool = False) -> None:
        factory_kwargs = self._checkpoint_factory_kwargs(weight, assign=assign)
        if "low_rank_left" in self._parameters:
            del self._parameters["low_rank_left"]
        if "low_rank_right" in self._parameters:
            del self._parameters["low_rank_right"]
        self.low_rank_left = None
        self.low_rank_right = None
        if "weight" not in self._parameters:
            del self.__dict__["weight"]
            self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features, **factory_kwargs))
        self.is_compressed = False

    def _prepare_low_rank_parameters(self, left: torch.Tensor, right: torch.Tensor, *, assign: bool = False) -> None:
        factory_kwargs = self._checkpoint_factory_kwargs(left, assign=assign)
        if "weight" in self._parameters:
            del self._parameters["weight"]
        self.weight = None  # type: ignore[assignment]
        self.low_rank_left = nn.Parameter(torch.empty(left.shape, **factory_kwargs))
        self.low_rank_right = nn.Parameter(torch.empty(right.shape, **factory_kwargs))
        self.is_compressed = True

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        left = state_dict.get(prefix + "low_rank_left")
        right = state_dict.get(prefix + "low_rank_right")
        if left is not None and right is not None:
            if (
                left.ndim != 2
                or right.ndim != 2
                or left.shape[0] != self.out_features
                or right.shape[1] != self.in_features
                or left.shape[1] != right.shape[0]
            ):
                error_msgs.append(
                    "size mismatch for low-rank checkpoint: "
                    f"low_rank_left has shape {tuple(left.shape)} and "
                    f"low_rank_right has shape {tuple(right.shape)}, expected "
                    f"({self.out_features}, rank) and (rank, {self.in_features})"
                )
            else:
                assign = bool(local_metadata.get("assign_to_params_buffers", False))
                self._prepare_low_rank_parameters(left, right, assign=assign)
        elif prefix + "weight" in state_dict and self.is_compressed:
            assign = bool(local_metadata.get("assign_to_params_buffers", False))
            self._prepare_dense_parameters(state_dict[prefix + "weight"], assign=assign)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def extra_repr(self) -> str:
        rank = None if self.low_rank_right is None else self.low_rank_right.shape[0]
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, is_compressed={self.is_compressed}, rank={rank}"
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.is_compressed:
            if self.low_rank_left is None or self.low_rank_right is None:
                raise RuntimeError("compressed low-rank factors are missing")
            hidden = F.linear(input, self.low_rank_right)
            return F.linear(hidden, self.low_rank_left, self.bias)
        return F.linear(input, self.weight, self.bias)
