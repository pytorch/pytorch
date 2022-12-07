import torch.nn as nn
from typing import cast, Dict, Optional


class _State:
    pass

_module_state_mapping: Dict[nn.Module, _State] = {}

def _insert_module_state(module: nn.Module, state: _State) -> None:
    global _module_state_mapping
    assert module not in _module_state_mapping
    _module_state_mapping[module] = state


def _get_module_state(module: nn.Module) -> Optional[_State]:
    global _module_state_mapping
    if isinstance(module, _State):
        return cast(_State, module)
    else:
        return _module_state_mapping.get(module, None)
