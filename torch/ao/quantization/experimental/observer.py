"""
This module implements nonuniform observers used to collect statistics about
the values observed during calibration (PTQ) or training (QAT).
"""

import torch
from torch.ao.quantization.observer import ObserverBase
from typing import Tuple
from torch.ao.quantization.utils import check_min_max_valid, calculate_qmin_qmax

class NonUniformQuantizationObserverBase(ObserverBase):
    # quant_min = None
    # quant_max = None
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
        min_val = None,
        max_val = None,
        b = 0,
        k = 0,
        dtype = torch.quint8) -> None:
        super().__init__(dtype)
        self.min_val = min_val
        self.max_val = max_val
        self.level_indices = level_indices
        self.b = b
        self.k = k

        # self.has_customized_qrange = (quant_min is not None) and (quant_max is not None)
        # self.quant_min, self.quant_max = calculate_qmin_qmax(quant_min, quant_max, self.has_customized_qrange, self.dtype, self.reduce_range)

    # introduce signed numbers
    def _calculate_qparams(self, min_val: torch.Tensor, max_val: torch.Tensor, signed: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        # compute alpha
        self.alpha = max_val

        # compute n and store as member variable
        self.n = b // k

        # store a tensor of subtensors (all levels)
        p_all = []

        # create levels
        for i in range(0, self.n):
            p_curr = torch.tensor([0])

            for j in range(0, 2**(self.k-1) + 1):
                curr_ele = 2**(-(i+j*n))
                p_append = torch.tensor([curr_ele])
                p_curr = torch.cat((p_curr, p_append))
                # introduce signed numbers
                if signed:
                    p_curr = torch.cat((p_curr, torch.tensor([-curr_ele])))

            # sort tensor before adding to list
            sorted, indices = torch.sort(p_curr)
            print(sorted)
            p_all.append(sorted)

        # gamma calculation:
        # loop through all tensors, add element at index 1 for each tensor
        p_sum = 0
        for tens in p_all:
            p_sum += tens[1]

        # assign gamma
        self.gamma = self.alpha / p_sum

        # get all possible quantization levels

        # tensor size: 2**b x 2**b
        tensor_size = 2**self.b
        levels = torch.zeros(size=(tensor_size, tensor_size))
        for level0 in range(tensor_size):
            for level1 in range(tensor_size):
                levels[level0][level1] = gamma * (p0[level0] + p1[level1])

        # -------------------------------------------------------------------------------------------

        # if not check_min_max_valid(min_val, max_val):
        #     return torch.tensor([1.0], device=min_val.device.type), torch.tensor([0], device=min_val.device.type)

        # quant_min, quant_max = self.quant_min, self.quant_max
        # min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        # max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        # device = min_val_neg.device

class APoTObserver(NonUniformQuantizationObserverBase):
    # !!!!! fix this !!!!!
    alpha = 0
    gamma = 0
    level_indices = torch.Tensor()
    min_val = torch.Tensor()
    max_val = torch.Tensor()

    def __init__(
        self,
        min_val = torch.Tensor(),
        max_val = torch.Tensor(),
        b = 0,
        k = 0) -> None:
        super(APoTObserver, self).__init__(min_val, max_val, b, k)

    def calculate_qparams(self):
        return self._calculate_qparams(self.min_val, self.max_val)

    def _calculate_qparams(self, min_val, max_val):
        return super(APoTObserver, self)._calculate_qparams(min_val, max_val)

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
