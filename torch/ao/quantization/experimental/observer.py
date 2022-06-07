"""
This module implements nonuniform observers used to collect statistics about
the values observed during calibration (PTQ) or training (QAT).
"""

import torch
import itertools
import math
from torch.ao.quantization.observer import ObserverBase
from typing import Tuple
from torch.ao.quantization.utils import check_min_max_valid, calculate_qmin_qmax

class NonUniformQuantizationObserverBase(ObserverBase):
    # quant_min = None
    # quant_max = None
    min_val: torch.Tensor
    max_val: torch.Tensor
    # level_indices: torch.Tensor
    b: int
    k: int
    n: int
    alpha: float
    gamma: float

    def __init__(
        self,
        min_val = None,
        max_val = None,
        b = 0,
        k = 0,
        dtype = torch.quint8) -> None:
        super().__init__(dtype)
        self.min_val = min_val
        self.max_val = max_val
        # self.level_indices = level_indices
        self.b = b
        self.k = k
        self.n = 0

        # self.has_customized_qrange = (quant_min is not None) and (quant_max is not None)
        # self.quant_min, self.quant_max = calculate_qmin_qmax(quant_min, quant_max, self.has_customized_qrange, self.dtype, self.reduce_range)

    # introduce signed numbers
    r""" Calculates nonuniform quantization parameters given min and max value tensors.

    Args:
        min_val: Minimum values per channel
        max_val: Maximum values per channel

    Returns:
        gamma: gamma quantization parameter, defined to ensure that alpha is the maximum of the range
        quantization_levels: non-uniform quantization levels, calculated according to APoT paper: (https://arxiv.org/pdf/1909.13144.pdf)
        level_indices: int representation of quantization_levels
    """
    def _calculate_qparams(self, min_val: torch.Tensor, max_val: torch.Tensor, signed: bool) -> Tuple[float, torch.tensor, torch.tensor]:
        # compute alpha
        self.alpha = max_val

        # compute n and store as member variable
        if self.k:
            self.n = self.b // self.k

        if self.n == 0:
            return (0.0, torch.tensor([]), torch.tensor([]))

        print("n", self.n)
        print("b", self.b)
        print("k", self.k)

        # store a tensor of subtensors (all levels)
        p_all = []

        # create levels
        for i in range(0, self.n):
            p_curr = torch.tensor([0])

            for j in range(0, 2**(self.k-1) + 1):
                curr_ele = 2**(-(i+j*self.n))
                p_append = torch.tensor([curr_ele])
                p_curr = torch.cat((p_curr, p_append))
                # introduce signed numbers
                if signed:
                    p_curr = torch.cat((p_curr, torch.tensor([-curr_ele])))

            if signed:
                # sort tensor before adding to list
                sorted, indices = torch.sort(p_curr)
                p_all.append(sorted)
            else:
                p_all.append(p_curr)

        # gamma calculation:
        # loop through all tensors, add element at index 1 for each tensor
        p_sum = 0
        for tens in p_all:
            p_sum += tens[1]

        print("p sum: ", p_sum)

        # assign gamma
        self.gamma = self.alpha / p_sum

        print("alpha: ", self.alpha)
        print("gamma: ", self.gamma)

        # hard code this to test
        # p_all.append(torch.Tensor([0, 1, 0.25, 0.0625]))

        # calculate cartesian product
        cartesian_product = list(itertools.product(*p_all))

        print("**************")

        print(cartesian_product)
        print(cartesian_product[0][0])

        quantization_levels = []

        # calculate sum of each row
        for row in cartesian_product:
            sum = 0
            for ele in row:
                sum += ele
            quantization_levels.append(sum)

        quantization_levels = [self.gamma * ele for ele in quantization_levels]

        print(quantization_levels)
        quantization_levels = torch.Tensor(quantization_levels)

        quantization_levels, indices = quantization_levels.sort()

        print("indices", indices)

        # level_indices = float_to_apot(0.5686, quantization_levels)

        print("tensors: ")
        for t in p_all:
            print(t)

        print("quantization levels", quantization_levels)

        level_indices = self.float_to_apot(0.5686, quantization_levels, indices)

        print(level_indices)

        return (self.gamma, quantization_levels, torch.tensor([]))

    # from https://www.internalfb.com/intern/anp/view/?id=1964753
    # generalize this to work for levels
    def float_to_apot(self, x, levels, indices):
        assert 0.0 <= x < 1.0

        # brute force search for the right combination of levels
        min_delta = math.inf

        # min_index is a list to represent the index coordinate of min_delta within the n-dimensional matrix.
        # min_index will have length n.
        min_index = 0
        index_to_base2 = []

        zip_matrices = zip(levels, indices)

        # look for the smallest difference between x and the boundary of the quantization level
        for cur_val, index in zip_matrices:
            cur_delta = abs(cur_val - x)
            if cur_delta < min_delta:
                min_delta = cur_delta
                min_index = index

        print("min index: ", min_index)

        # convert min_index to base 2
        res = bin(min_index)

        return res

class APoTObserver(NonUniformQuantizationObserverBase):
    # !!!!! fix this !!!!!
    alpha = 0
    gamma = 0
    min_val = torch.Tensor()
    max_val = torch.Tensor()

    def __init__(
        self,
        min_val = torch.Tensor(),
        max_val = torch.Tensor(),
        b = 0,
        k = 0) -> None:
        super(APoTObserver, self).__init__(min_val, max_val, b, k)

    def calculate_qparams(self, signed):
        return self._calculate_qparams(self.min_val, self.max_val, signed)

    def _calculate_qparams(self, min_val, max_val, signed):
        return super(APoTObserver, self)._calculate_qparams(min_val, max_val, signed)

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val_cur, max_val_cur = torch.aminmax(x)
        min_val = torch.min(min_val_cur, self.min_val)
        max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig
