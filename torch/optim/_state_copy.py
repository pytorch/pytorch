import threading

import torch

from .optimizer import Optimizer


_original_state_dict = Optimizer.state_dict
_original_process_value = Optimizer._process_value_according_to_param_policy
_install_lock = threading.Lock()


def _copy_state_value(
    param: torch.Tensor,
    value: torch.Tensor,
    param_id: int,
    param_groups: list[dict[str, object]],
    key: str | None = None,
):
    result = _original_process_value(param, value, param_id, param_groups, key)
    return result.clone() if isinstance(result, torch.Tensor) else result


def _clone_tensors(value, memo=None):
    if memo is None:
        memo = {}
    if isinstance(value, torch.Tensor):
        key = id(value)
        if key not in memo:
            memo[key] = value.clone()
        return memo[key]
    if isinstance(value, dict):
        return {key: _clone_tensors(item, memo) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tensors(item, memo) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tensors(item, memo) for item in value)
    return value


def _state_dict(self):
    return _clone_tensors(_original_state_dict(self))


def install() -> None:
    if getattr(Optimizer, "_native_neo_state_copy_installed", False):
        return
    with _install_lock:
        if getattr(Optimizer, "_native_neo_state_copy_installed", False):
            return
        Optimizer.state_dict = _state_dict
        Optimizer._process_value_according_to_param_policy = staticmethod(
            _copy_state_value
        )
        Optimizer._native_neo_state_copy_installed = True


__all__ = ["install"]
