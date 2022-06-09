"""
This module implements nonuniform observers used to collect statistics about
the values observed during calibration (PTQ) or training (QAT).
"""

import torch
import itertools
from torch.ao.quantization.observer import ObserverBase
from typing import Tuple

class NonUniformQuantizationObserverBase(ObserverBase):
    min_val: torch.Tensor
    max_val: torch.Tensor
    level_indices: torch.Tensor
    b: int
    k: int
    n: int
    alpha: float
    gamma: float

    def __init__(
        self,
        min_val=None,
        max_val=None,
        b=None,
        k=None,
            dtype=torch.quint8) -> None:
        super().__init__(dtype)
        self.min_val = min_val
        self.max_val = max_val
        self.level_indices = torch.tensor([])
        self.b = b
        self.k = k
        self.n = None
        self.alpha = 0.0
        self.gamma = 0.0

    r""" Calculates nonuniform quantization parameters given min and max value tensors.
    Parameters calculated according to APoT paper: https://arxiv.org/pdf/1909.13144.pdf
    Args:
        min_val: minimum values per channel
        max_val: maximum values per channel
        signed: specifies whether to include signed values in quantization level calculations
    Returns:
        gamma: gamma quantization parameter, defined to ensure that alpha is the maximum of the range
        quantization_levels: non-uniform quantization levels
        level_indices: int representation of quantization_levels indices
    """
    def _calculate_qparams(
        self,
        min_val: torch.Tensor,
        max_val: torch.Tensor,
            signed: bool) -> Tuple[float, torch.Tensor, torch.Tensor]:
        # compute alpha
        self.alpha = max_val

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
        # loop through all tensors, add element at index 1 for each tensor
        p_sum = 0
        for tens in p_all:
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

        quantization_levels = [self.gamma * ele for ele in quantization_levels_list]
        quantization_levels = torch.tensor(quantization_levels)
        quantization_levels, level_indices = quantization_levels.sort()

        return (self.gamma, quantization_levels, level_indices)

class APoTObserver(NonUniformQuantizationObserverBase):
    def __init__(
        self,
        min_val=torch.Tensor,
        max_val=torch.Tensor,
        b=0,
            k=0) -> None:
        super(APoTObserver, self).__init__(min_val, max_val, b, k)

    def calculate_qparams(self, signed):
        return self._calculate_qparams(self.min_val, self.max_val, signed)

    def _calculate_qparams(self, min_val, max_val, signed):
        return super(APoTObserver, self)._calculate_qparams(min_val, max_val, signed)

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()
        x = x.to(self.min_val.dtype)
        min_val_cur, max_val_cur = torch.aminmax(x)
        min_val = torch.min(min_val_cur, self.min_val)
        max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig
