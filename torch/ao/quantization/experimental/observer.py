"""
This module implements nonuniform observers used to collect statistics about
the values observed during calibration (PTQ) or training (QAT).
"""

import torch
from torch.ao.quantization.observer import ObserverBase

class APoTObserver(ObserverBase):
    alpha = 0
    gamma = 0
    level_indices = torch.Tensor()

    def __init__(
        self,
        min_val: torch.Tensor,
        max_val: torch.Tensor,
        level_indices: torch.Tensor,
            b: int,
            k: int) -> None:
        super().__init__

    def calculate_qparams(self):
        return self._calculate_qparams()

    def _calculate_qparams(self):
        raise NotImplementedError

    def forward(self, x_orig):
        pass
