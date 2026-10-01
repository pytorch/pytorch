import operator
from typing import Any


def _collective_config_dict(config: object | None) -> dict[str, Any] | None:
    """Collective configs are passed to ops as their attribute dictionaries."""
    if config is None:
        return None
    values = config if isinstance(config, dict) else vars(config)
    # Config integers are guarded constants, not dynamic tensor dimensions.
    # Empty sequences are omitted: export turns them into untyped lists.
    return {
        key: operator.index(value)
        if isinstance(value, int) and not isinstance(value, bool)
        else value
        for key, value in values.items()
        if not (isinstance(value, (tuple, list)) and not value)
    }
