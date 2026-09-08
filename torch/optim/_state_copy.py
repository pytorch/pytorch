from collections.abc import Callable

import torch

from .optimizer import Optimizer


_original = Optimizer._process_value_according_to_param_policy


def _copy_state_value(
    param: torch.Tensor,
    value: torch.Tensor,
    param_id: int,
    param_groups: list[dict[str, object]],
    key: str | None = None,
) -> torch.Tensor:
    result = _original(param, value, param_id, param_groups, key)
    return result.clone()


def install() -> None:
    Optimizer._process_value_according_to_param_policy = staticmethod(
        _copy_state_value
    )


__all__ = ["install"]
