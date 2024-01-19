import torch
import torch.nn as nn

from torch.distributed._composable_state import _State


class FSDPState(_State):
    _module: nn.Module  # ref cycle

    def __init__(self):
        super().__init__()
        self._device = torch.device("cpu")
