import torch

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
    """

    def __init__(self, in_features, out_features, weight2quantize):
        self.in_features = in_features
        self.out_features = out_features

        super().__init__()

        observer = APoTObserver(b=8, k=1)

        self.alpha, self.gamma, self.quantization_levels, self.level_indices = observer.calculate_qparams(signed=False)

        self.weight = quantize_APoT(weight2quantize, self.alpha, self.gamma, self.quantization_levels, self.level_indices)

    """
    Instead of matrix multiplication, we use bitshifting to multiply weight and bias.
    """
    def linear_APoT_fn(self, activation: torch.Tensor) -> torch.Tensor:
        # assert activation is 2D tensor

        # decompose APoT weight
        tensor




    """
    x is a uniformly quantized activation tensor,
    passed in to be multiplied by an APoT quantized weight.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_APoT_fn(x)
