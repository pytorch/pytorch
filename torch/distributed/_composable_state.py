from typing import cast, Dict, Optional

import torch.nn as nn


class _State:
    pass


_module_state_mapping: Dict[nn.Module, _State] = {}


def _insert_module_state(module: nn.Module, state: _State) -> None:
    global _module_state_mapping
    assert module not in _module_state_mapping, f"Inserting {module} more than once."
    _module_state_mapping[module] = state


def _get_module_state(module: nn.Module) -> Optional[_State]:
    """
    Given a ``module``, this API finds out if the module is also a ``_State``
    instance or if the module is managed by a composable API. If the module
    is also a ``_State``, ``module`` will be casted to ``_State` and returned.
    If it is managed by a composable API, the corresponding ``_State`` will
    be returned.
    """

    global _module_state_mapping
    if isinstance(module, _State):
        return cast(_State, module)
    else:
        # https://github.com/pytorch/pytorch/issues/107054
        if module in _module_state_mapping:
            return _module_state_mapping[module]
        else:
            return None
