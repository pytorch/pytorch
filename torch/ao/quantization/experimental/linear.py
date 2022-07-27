import torch
import numpy as np

from torch.nn.quantized.modules.utils import WeightedQuantizedModule
from torch.ao.quantization.experimental.observer import APoTObserver
from torch.ao.quantization.experimental.quantizer import quantize_APoT

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
        weight: APoT quantized tensor from weight2quantize
        weight_transposed: transposed weight tensor, used in linear transformation calculation (y = x * A^T + b)
    """

    def __init__(self, weight2quantize: torch.Tensor):
        assert weight2quantize.dim() == 2

        super().__init__()

        # hard code b, k to match uniform quantization: b=4, k=1
        self.b = 8
        self.k = 1
        self.n = 8

        observer = APoTObserver(b=self.b, k=self.k)

        observer(weight2quantize)

        self.alpha, self.gamma, self.quantization_levels, self.level_indices = observer.calculate_qparams(signed=False)

        quantized = quantize_APoT(weight2quantize, self.alpha, self.gamma, self.quantization_levels, self.level_indices)
        self.weight = quantized.data
        self.weight_transposed = torch.transpose(self.weight, 0, 1)

        self.weight_transposed = self.weight_transposed.reshape(weight2quantize.shape)
        self.weight_transposed = self.weight_transposed.reshape(weight2quantize.shape[1], weight2quantize.shape[0])

    def decompose_APoT(self, x):
        r"""
        Decompose APoT quantized terms into binary digits
        Args:
            x (Tensor): binary representation of APoT quantized tensor
        """
        # remove "0b" prefix from binary representation
        x = x[2:]

        # initialize list of blocks
        blocks = []

        while x:
            blocks.append(x[0:self.k])
            x = x[self.k:]

        return blocks

    def bitshift_multiplication(self, decomposed_weight, activation):
        r"""
        Compute matrix multiplication result of input weight and activation using bitshifting
        Args:
            decomposed_weight (Tensor): APoT quantized weight decomposed into binary
            activation (Tensor): uniformly quantized activation
        """
        rows2 = activation.size(dim=0)
        cols2 = activation.size(dim=1)

        rows1 = decomposed_weight.shape[0]
        cols1 = decomposed_weight.shape[1]

        result = torch.zeros(rows1, cols2)

        # compute matrix multiplication with bitshifts
        for i in range(rows2):
            for j in range(cols2):
                for k in range(rows1):
                    weight_val = decomposed_weight[k][j]
                    r = int(activation[i][k])

                    for idx in range(len(weight_val)):
                        ele = int(weight_val[idx])

                        x = len(weight_val) - 1 - idx

                        if ele:
                            curr_result = r << x
                        else:
                            curr_result = 0
                        result[i][j] += curr_result

        return result

    def linear_APoT_fn(self, activation: torch.Tensor) -> torch.FloatTensor:
        r"""
        Multiply APoT quantized weight and uniformly quantized activation (dtype: quint8)
        with bitshifting instead of matrix multiplication.
        Result has dtype torch.float32
        Args:
            activation (Tensor): uniformly quantized activation tensor
        """
        assert activation.dim() == 2

        weight_rows = self.weight_transposed.size()[0]
        weight_cols = self.weight_transposed.size()[1]

        decomposed_weight = np.empty(shape=(weight_rows, weight_cols), dtype=object)
        for row in range(weight_rows):
            for col in range(weight_cols):
                decomposed_weight[row][col] = self.decompose_APoT(bin(self.weight_transposed[row][col]))

        rows1 = self.weight_transposed.size(dim=0)
        cols1 = self.weight_transposed.size(dim=1)

        rows2 = activation.size(dim=0)
        cols2 = activation.size(dim=1)

        result = self.bitshift_multiplication(decomposed_weight, activation).type(torch.FloatTensor)

        return result

    def forward(self, activation: torch.Tensor) -> torch.FloatTensor:
        r"""
        Call linear_APoT_fn to multiply activation with an APoT quantized weight.
        Args:
            activation (Tensor): uniformly quantized activation tensor
        """
        return self.linear_APoT_fn(activation)

    @classmethod
    def from_reference(cls,  # type: ignore[override]
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
