import torch
import itertools
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

    def __init__(self, weight2quantize, signed=False):
        # self.in_features = in_features
        # self.out_features = out_features

        super().__init__()

        observer = APoTObserver(b=4, k=2)

        self.b = 4
        self.k = 2
        self.n = 2

        observer(weight2quantize)

        self.alpha, self.gamma, self.quantization_levels, self.level_indices = observer.calculate_qparams(signed=signed)

        quantized = quantize_APoT(weight2quantize, self.alpha, self.gamma, self.quantization_levels, self.level_indices)
        self.weight = dequantize_APoT(quantized)
        print("weight", self.weight)

    """
    Instead of matrix multiplication, we use bitshifting to multiply weight and bias.
    """
    def linear_APoT_fn(self, activation: torch.Tensor, signed=False) -> torch.Tensor:
        # assert activation is 2D tensor

        size = int(math.sqrt(torch.numel(self.weight)))
        self.weight = self.weight.reshape(size, size)

        # decompose APoT weight
        p_all = []

        # create the "levels"
        p0 = torch.tensor([0, 2**0, 2**-2, 2**-4])
        p1 = torch.tensor([0, 2**-1, 2**-3, 2**-5])

        # self.gamma = 2 / 3

        # make pair of tensors that associate tuple (decomp1, decomp2) with value

        levels = []
        vals = []
        for level0 in range(4):
            for level1 in range(4):
                levels.append((p0[level0], p1[level1]))
                vals.append(self.gamma * (p0[level0] + p1[level1]))

        level_tuples = np.empty(shape=(2, 2), dtype=object)
        for ele in self.weight:
            print(ele)
            idx = vals.index(ele)
            level_tuples.append(levels[idx])

        print(level_tuples)

        # level_tuples = level_tuples.reshape(2, 2)

        # print(level_tuples)

        activation = torch.tensor([[5, 8, 1, 2], [6, 7, 3, 0], [4, 5, 9, 1], [2, 5, 3, 1]])

        rows1 = level_tuples.shape[0]
        cols1 = level_tuples.shape[1]

        rows2 = activation.shape[0]
        cols2 = activation.shape[1]

        result = torch.zeros(rows1, cols2)

        for i in range(rows1):
            for j in range(cols2):
                for k in range(rows2):
                    ele1 = levels[i][k]
                    r = int(activation[k][j])

                    for x in ele1:
                        curr_result = x * r
                        print("curr result", curr_result)
                        # if x == 0:
                        #     curr_result = 0
                        # else:
                        #     x = int(math.log2(x))

                        #     if x == 0:
                        #         curr_result = r
                        #     elif x > 0:
                        #         print("hi")
                        #         curr_result = float(r << x)
                        #     else:
                        #         x = - x

                        #         r_bin = bin(r)

                        #         r_bin = r_bin[2:]

                        #         while len(r_bin) < x + 1:
                        #             r_bin = "0" + r_bin

                        #         # perform shifts, store decimal portion
                        #         dec = ""
                        #         for m in range(x):
                        #             dec = r_bin[-1] + dec
                        #             r_bin = r_bin[:-1]

                        #         # convert dec portion from binary -> decimal
                        #         dec_portion = 0.0
                        #         for exp in range(len(dec)):
                        #             if int(dec[exp]):
                        #                 dec_portion = dec_portion + (2**-(exp+1))

                        #         # convert whole portion from binary -> decimal
                        #         whole_portion = 0.0
                        #         for exp in range(len(r_bin)):
                        #             if int(r_bin[exp]):
                        #                 whole_portion = whole_portion + (2**exp)

                        #         curr_result = whole_portion + dec_portion

                        #         print("dec result", dec_portion)
                        #         print("curr result", curr_result)

                        result[i][j] += float(curr_result)

        result = result * self.gamma

        print("result", result)

        return result


    r"""
    activation is a uniformly quantized activation tensor,
    passed in to be multiplied by an APoT quantized weight.
    """
    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        return self.linear_APoT_fn(activation)


    def from_reference(cls, ref_qlinear, output_scale, output_zero_point):
        r"""Create a (fbgemm/qnnpack) quantized module from a reference quantized module

        Args:
            ref_qlinear (Module): a reference quantized linear module, either produced by torch.ao.quantization
                          utilities or provided by the user
            output_scale (float): scale for output Tensor
            zero_point (int): zero point for output Tensor
        """
        # qlinear = cls(
        #     ref_qlinear.in_features,
        #     ref_qlinear.out_features)
        # qweight = ref_qlinear.get_quantized_weight()
        # qlinear.set_weight_bias(qweight, ref_qlinear.bias)

        # qlinear.scale = float(output_scale)
        # qlinear.zero_point = int(output_zero_point)
        # return qlinear
        return None
