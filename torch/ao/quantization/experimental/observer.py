"""
This module implements nonuniform observers used to collect statistics about
the values observed during calibration (PTQ) or training (QAT).
"""

import torch
import itertools
import matplotlib.pyplot as plt
from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.experimental.apot_utils import float_to_apot, apot_to_float

# TODO: Consider adding NonUniformQuantizationObserverBase class
# when more than one non-uniform method is implemented

class APoTObserver(ObserverBase):
    b: int
    k: int
    n: int
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(
        self,
        b,
        k,
            dtype=torch.quint8) -> None:
        super().__init__(dtype)
        self.b = b
        self.k = k

        self.min_val = torch.tensor([])
        self.max_val = torch.tensor([])

    # min_val and max_val are optional args to override
    # the min_val and max_val observed by forward
    def calculate_qparams(self, signed):
        return self._calculate_qparams(signed, self.min_val, self.max_val)

    r""" Calculates nonuniform quantization parameters according to APoT paper:
    https://arxiv.org/pdf/1909.13144.pdf.
    Arg:
        signed: specifies whether to include signed values in quantization level calculations
        min_val: optional arg that can override min_val internal attribute
        max_val: optional arg that can override max_val internal attribute
    Returns:
        alpha: alpha quantization parameter, max of abs value of observed values
        gamma: gamma quantization parameter, defined to ensure that alpha is the maximum of the range
        quantization_levels: non-uniform quantization levels (fp representation)
        level_indices: int representation of quantization_levels indices
    """
    def _calculate_qparams(self, signed: bool, min_val=None, max_val=None):
        if min_val is not None:
            self.min_val = min_val
        if max_val is not None:
            self.max_val = max_val

        # compute alpha
        alpha = torch.max(-self.min_val, self.max_val)

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

            for j in range(0, (2 ** self.k - 2) + 1):
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
        gamma = alpha / p_sum

        # calculate cartesian product
        cartesian_product = list(itertools.product(*p_all))

        quantization_levels_list = []

        # calculate sum of each row
        for row in cartesian_product:
            sum = 0.0
            for ele in row:
                sum += ele
            quantization_levels_list.append(sum)

        quantization_levels_gamma = [float(gamma) * ele for ele in quantization_levels_list]
        quantization_levels = torch.tensor(quantization_levels_gamma)
        level_indices = torch.tensor([])
        quantization_levels, level_indices = quantization_levels.sort()

        return (alpha, gamma, quantization_levels, level_indices)

    r"""Records the running minimum and maximum of ``x``.
        Args:
            x_orig: Tensor to be observed for min and max val"""
    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()
        min_val, max_val = torch.aminmax(x)
        if self.min_val.numel():
            min_val = torch.min(min_val, self.min_val)
        if self.max_val.numel():
            max_val = torch.max(max_val, self.max_val)
        self.min_val = min_val
        self.max_val = max_val
        return x_orig

    r"""Displays visualization of APoT quantization levels
        Args:
            observer: APoTObserver to calculate qparams
            signed: bool to indicate if qparams should be signed/unsigned
    """
    def quant_levels_visualization(self, signed=False):
        alpha, gamma, quantization_levels, level_indices = self.calculate_qparams(signed)

        xs = [float(x) / 1000.0 for x in range(1000)]
        ys = [apot_to_float(float_to_apot(x, quantization_levels, level_indices, alpha),
                            quantization_levels, level_indices).item() for x in xs]

        f = plt.figure(figsize=(15, 10))

        plt.plot(xs, ys)
        plt.title("APoT Quantization Plot")
        plt.xlabel("Full Precision")
        plt.ylabel("Quantized")
        plt.show()
