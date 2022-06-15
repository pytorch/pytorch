"""
This module implements nonuniform observers used to collect statistics about
the values observed during calibration (PTQ) or training (QAT).
"""

import torch
import itertools
import math
import matplotlib.pyplot as plt
from torch.ao.quantization.observer import ObserverBase

class APoTObserver(ObserverBase):
    max_val: float
    b: int
    k: int
    n: int
    alpha: float
    gamma: float
    level_indices: torch.Tensor

    def __init__(
        self,
        max_val,
        b,
        k,
            dtype=torch.quint8) -> None:
        super().__init__(dtype)
        self.max_val = max_val
        self.b = b
        self.k = k

    def calculate_qparams(self, signed):
        return self._calculate_qparams(signed)

    r""" Calculates nonuniform quantization parameters according to APoT paper:
    https://arxiv.org/pdf/1909.13144.pdf.
    Arg:
        signed: specifies whether to include signed values in quantization level calculations
    Returns:
        gamma: gamma quantization parameter, defined to ensure that alpha is the maximum of the range
        quantization_levels: non-uniform quantization levels (fp representation)
        level_indices: int representation of quantization_levels indices
    """
    def _calculate_qparams(self, signed):
        # compute alpha
        self.alpha = self.max_val

        # check for valid inputs of b, k
        assert(self.k and self.k != 0)
        assert(self.b % self.k == 0)

        # compute n and store as member variable
        self.n = self.b // self.k

        # store a tensor of subtensors (all levels)
        p_all = []

        # create levels
        for i in range(0, self.n):
            p_curr = torch.tensor([0])

            for j in range(0, 2 ** (self.k - 1) + 1):
                curr_ele = 2 ** (- (i + j * self.n))
                p_append = torch.tensor([curr_ele])
                p_curr = torch.cat((p_curr, p_append))
                # introduce signed numbers
                if signed:
                    p_curr = torch.cat((p_curr, torch.tensor([-curr_ele])))

            if signed:
                # sort tensor in reverse order before adding to list if signed
                sorted, indices = torch.sort(p_curr, descending=True)
                p_all.append(sorted)
            else:
                p_all.append(p_curr)

        # gamma calculation:
        # loop through all tensors
        # if signed, add element at index 0 for each tensor
        # else, add element at index 1 for each tensor
        # gamma defined to ensure alpha is at max of range
        p_sum = 0.0
        for tens in p_all:
            if signed:
                p_sum += float(tens[0])
            else:
                p_sum += float(tens[1])

        # assign gamma
        self.gamma = self.alpha / p_sum

        # calculate cartesian product
        cartesian_product = list(itertools.product(*p_all))

        quantization_levels_list = []

        # calculate sum of each row
        for row in cartesian_product:
            sum = 0
            for ele in row:
                sum += ele
            quantization_levels_list.append(sum)

        quantization_levels_gamma = [self.gamma * ele for ele in quantization_levels_list]
        quantization_levels = torch.tensor(quantization_levels_gamma)
        level_indices = torch.tensor([])
        quantization_levels, self.level_indices = quantization_levels.sort()

        return (self.gamma, quantization_levels, self.level_indices)

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
    x_apot = round(x_apot)
    idx = list(indices).index(x_apot)
    return levels[idx]
