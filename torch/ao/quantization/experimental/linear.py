import torch
import numpy as np
import math

from torch.nn.quantized.modules.utils import WeightedQuantizedModule
from torch.ao.quantization.experimental.observer import APoTObserver
from torch.ao.quantization.experimental.quantizer import quantize_APoT, dequantize_APoT

class LinearAPoT(WeightedQuantizedModule):
    r"""
    A quantized linear module with quantized tensor as inputs and outputs
    to support APoT quantization.
    We adopt the same interface as `torch.nn.Linear`, see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear for documentation.

    Similar to :class:`~torch.nn.Linear`, attributes will be randomly
    initialized at module creation time and will be overwritten later

    Attributes:
        alpha: `alpha` qparam of output Quantized Tensor, type: Tensor
        gamma: `gamma` qparam of output Quantized Tensor, type: Tensor
        quantization_levels: `quantization_levels` qparam of output Quantized Tensor, type: Tensor
        level_indices: `level_indices` qparam of output Quantized Tensor, type: Tensor
    """

    def __init__(self, weight2quantize: torch.Tensor, signed=False):
        # self.in_features = in_features
        # self.out_features = out_features

        super().__init__()

        # hard code to APoT paper example inputs: b=4, k=2
        observer = APoTObserver(b=4, k=2)

        observer(weight2quantize)

        self.alpha, self.gamma, self.quantization_levels, self.level_indices = observer.calculate_qparams(signed=signed)

        quantized = quantize_APoT(weight2quantize, self.alpha, self.gamma, self.quantization_levels, self.level_indices)
        self.weight = dequantize_APoT(quantized)
        self.weight = self.weight.reshape(weight2quantize.shape)

    def decompose_APoT(self):
        r"""
        Helper function to decompose sums of PoT from APoT quantization
        into corresponding tuples of PoT terms.
        """
        size = int(math.sqrt(torch.numel(self.weight)))

        # decompose APoT weight
        p_all = []

        # create the "levels"
        p0 = torch.tensor([0, 2**0, 2**-2, 2**-4])
        p1 = torch.tensor([0, 2**-1, 2**-3, 2**-5])

        # make pair of tensors that associate tuple (decomp1, decomp2) with value
        all_levels_decomposed = []
        vals = []
        for level0 in range(size):
            for level1 in range(size):
                all_levels_decomposed.append((p0[level0], p1[level1]))
                vals.append(self.gamma * (p0[level0] + p1[level1]))

        # decompose levels in weight
        levels_decomposed = []
        for ele in self.weight.flatten():
            idx = vals.index(ele)
            levels_decomposed.append(all_levels_decomposed[idx])

        # reshape decomposed levels array into 2d matrix size
        np_levels_decomposed = np.empty(len(levels_decomposed), dtype=object)
        np_levels_decomposed[:] = levels_decomposed
        np_levels_decomposed = np_levels_decomposed.reshape(size, size)

        return np_levels_decomposed

    def linear_APoT_fn(self, activation: torch.Tensor) -> torch.Tensor:
        r"""
        Multiply APoT quantized weight and uniformly quantized activation
        with bitshifting instead of matrix multiplication.
        Args:
            activation (Tensor): uniformly quantized activation tensor
        """
        # assert activation is 2D tensor
        size = int(math.sqrt(torch.numel(self.weight)))

        levels_decomposed = self.decompose_APoT()

        rows1 = self.weight.shape[0]
        cols1 = self.weight.shape[1]

        rows2 = activation.shape[0]
        cols2 = activation.shape[1]

        result = torch.zeros(rows1, cols2)

        # compute matrix multiplication with bitshifts
        for i in range(rows1):
            for j in range(cols1):
                for k in range(rows2):
                    ele1 = levels_decomposed[i][k]
                    r = int(activation[k][j])

                    for x in ele1:
                        # curr_result = x * r
                        # print("curr result", curr_result)
                        if x == 0:
                            curr_result = 0.0
                        else:
                            x = int(math.log2(x))

                            if x == 0:
                                curr_result = r
                            elif x > 0:
                                curr_result = float(r << x)
                            else:
                                x = - x
                                r_bin = bin(r)
                                r_bin = r_bin[2:]

                                # pad r_bin with 0s
                                while len(r_bin) < x + 1:
                                    r_bin = "0" + r_bin

                                # perform shifts, store decimal portion
                                dec = ""
                                for m in range(x):
                                    dec = r_bin[-1] + dec
                                    r_bin = "0" + r_bin[:-1]

                                # convert dec portion from binary -> decimal
                                dec_portion = 0.0
                                for exp in range(len(dec)):
                                    if int(dec[exp]):
                                        dec_portion = dec_portion + (2 ** - (exp + 1))

                                # convert whole portion from binary -> decimal
                                r_bin = r_bin[::-1]
                                whole_portion = 0.0
                                for exp in range(len(r_bin)):
                                    if int(r_bin[exp]):
                                        whole_portion = whole_portion + (2 ** exp)

                                curr_result = float(whole_portion + dec_portion)

                        result[i][j] += curr_result

        result = result * self.gamma

        return result

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        r"""
        Call linear_APoT_fn to multiply activation with an APoT quantized weight.
        Args:
            activation (Tensor): uniformly quantized activation tensor
        """
        return self.linear_APoT_fn(activation)


    def from_reference(self,  # type: ignore[override]
                       ref_qlinear,
                       alpha: torch.Tensor,
                       gamma: torch.Tensor,
                       quantization_levels: torch.Tensor,
                       level_indices: torch.Tensor):
        r"""Create a (fbgemm/qnnpack) quantized module from a reference quantized module

        Args:
            ref_qlinear (Module): a reference quantized linear module, either
                                  produced by torch.ao.quantization.experimental
                                  utilities or provided by the user
        """
        qlinear = ref_qlinear
        qlinear.alpha = alpha
        qlinear.gamma = gamma
        qlinear.quantization_levels = quantization_levels
        qlinear.level_indices = level_indices
        return qlinear
