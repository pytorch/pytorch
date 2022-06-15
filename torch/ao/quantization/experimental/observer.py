"""
This module implements nonuniform observers used to collect statistics about
the values observed during calibration (PTQ) or training (QAT).
"""

import torch
from torch.ao.quantization.observer import ObserverBase

# TODO: Consider adding NonUniformQuantizationObserverBase class
# when more than one non-uniform method is implemented

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
        r"""Records the running maximum of ``x``."""
        max_val = self.max_val
        return x_orig

    def quant_levels_visualization(self, obs_result, filename):
        xs = [float(x) / 1000.0 for x in range(1000)]
        ys = [apot_to_float(float_to_apot(x, obs_result[1], obs_result[2]),
                            obs_result[1], obs_result[2]).item() for x in xs]

        f = plt.figure(figsize=(15, 10))

        plt.plot(xs, ys)
        plt.title("APoT Quantization Plot")
        plt.xlabel("Full Precision")
        plt.ylabel("Quantized")
        filestr = "pytorch/test/quantization/core/experimental/plots/" + filename
        plt.savefig(filestr)

r"""Converts floating point input into int4 APoT2 number
    based on quantization levels
"""
def float_to_apot(x, levels, indices):
    levels_lst = list(levels)
    indices_lst = list(indices)

    min_delta = math.inf
    best_idx = 0

    for level, idx in zip(levels_lst, indices_lst):
        cur_delta = abs(level - x)
        if cur_delta < min_delta:
            min_delta = cur_delta
            best_idx = idx

    return best_idx

r"""Converts int4 APoT2 input into floating point number
based on quantization levels
"""
def apot_to_float(x_apot, levels, indices):
    idx = list(indices).index(x_apot)
    return levels[idx]
