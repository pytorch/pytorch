import operator
from typing import Any

import torch


def _constant(value: Any) -> Any:
    # Config values are guarded constants, not dynamic tensor dimensions.
    if isinstance(value, (bool, torch.SymBool)):
        return bool(value)
    if isinstance(value, (int, torch.SymInt)):
        return operator.index(value)
    if isinstance(value, (float, torch.SymFloat)):
        return float(value)
    if isinstance(value, dict):
        return {key: _constant(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_constant(item) for item in value)
    return value


def _collective_config_dict(config: object | None) -> dict[str, Any] | None:
    """Collective configs are passed to ops as their attribute dictionaries."""
    if config is None:
        return None
    values = config if isinstance(config, dict) else vars(config)
    # Empty sequences are omitted: export turns them into untyped lists.
    return {
        key: _constant(value)
        for key, value in values.items()
        if not (isinstance(value, (tuple, list)) and not value)
    }
