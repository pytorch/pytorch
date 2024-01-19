from typing import Optional

import torch
import torch.nn as nn

from torch.distributed._composable_state import _get_module_state, _State

from ._fsdp_param_group import FSDPParamGroup


class FSDPState(_State):
    _module: nn.Module  # permit ref cycle since module and state lifetimes are 1:1
    _device: torch.device

    def __init__(self):
        super().__init__()
        self._fsdp_param_group: Optional[FSDPParamGroup] = None


def _get_module_fsdp_state(module: nn.Module) -> Optional[FSDPState]:
    state = _get_module_state(module)
    if isinstance(state, FSDPState):
        return state
    return None
