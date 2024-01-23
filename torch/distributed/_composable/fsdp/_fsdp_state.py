import torch
import torch.nn as nn

from torch.distributed._composable_state import _insert_module_state, _State


class FSDPState(_State):
    _module: nn.Module  # permit ref cycle since module and state lifetimes are 1:1
    _device: torch.device

    def __init__(self):
        super().__init__()

    # Define a separate init since `__init__` is called in the contract
    def init(self, module: nn.Module, device: torch.device) -> None:
        _insert_module_state(module, self)
        self._module = module
        self._device = device
