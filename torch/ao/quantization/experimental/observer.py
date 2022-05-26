"""
This module implements nonuniform observers used to collect statistics about
the values observed during calibration (PTQ) or training (QAT).
"""

import torch
from torch.ao.quantization.observer import ObserverBase
from typing import Tuple

class NonUniformQuantizationObserverBase(ObserverBase):
    def _calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

class APoTObserver(NonUniformQuantizationObserverBase):
    # def __init__(self, alpha: fp32, gamma: fp32, level_indices: tensor):
    #     self.alpha = alpha
    #     self.gamma = gamma
    #     self.level_indices = level_indices

    def calculate_qparams(self):
        NonUniformQuantizationObserverBase._calculate_qparams()

    def _calculate_qparams(self):
        raise NotImplementedError
