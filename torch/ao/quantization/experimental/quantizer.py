import torch
from torch import Tensor
from typing import Tuple
from torch.ao.quantization.experimental.apot_utils import float_to_apot, apot_to_float

# class to store APoT quantizer and
# implement quantize and dequantize
class APoTQuantizer():
    alpha: torch.Tensor
    gamma: torch.Tensor
    quantization_levels: torch.Tensor
    level_indices: torch.Tensor

    def __init__(
        self,
        alpha: torch.Tensor,
        gamma: torch.Tensor,
        quantization_levels: torch.Tensor,
            level_indices: torch.Tensor) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.quantization_levels = quantization_levels
        self.level_indices = level_indices

    r""" Quantizes fp Tensor to integer APoT representation.
    Conversion is based on the qparams from a specified APoT non-uniform observer.
    The approach follows the method outlined in the APoT paper: https://arxiv.org/pdf/1909.13144.pdf.
    Args:
        tensor2quantize: fp Tensor
    Returns:
        result: integer APoT representation of tensor2quantize
    """
    def quantize_APoT(self, tensor2quantize: Tensor):
        result = torch.tensor([])
        # map float_to_apot over tensor2quantize elements
        result_data = tensor2quantize.apply_(lambda x: float_to_apot(x, self.quantization_levels, self.level_indices))

        from torch.ao.quantization.experimental.APoT_tensor import TensorAPoT

        result = TensorAPoT(self, result_data)

        return result

    r""" Dequantizes integer Tensor to floating point (fp) representation
    based on the calculated quantization levels from a specified APoT non-uniform observer.
    The approach follows the method outlined in the APoT paper: https://arxiv.org/pdf/1909.13144.pdf.
    Args:
        float2apot: quantized APoT Tensor to dequantize
    Returns:
        result: fp representation of input Tensor
    """
    def dequantize_APoT(self, float2apot):
        apot_tensor = float2apot.data

        # map apot_to_float over tensor2quantize elements
        result = apot_tensor.apply_(lambda x: float(apot_to_float(x, self.quantization_levels, self.level_indices)))

        return result

    def q_apot_alpha(self) -> float:
        raise NotImplementedError

r""" Global method to create quantizer and call quantizer quantize_APoT
"""
def quantize_APoT(tensor2quantize: Tensor, qparams: Tuple):
    quantizer = APoTQuantizer(alpha=qparams[0], gamma=qparams[1], quantization_levels=qparams[2], level_indices=qparams[3])
    return quantizer.quantize_APoT(tensor2quantize)

r""" Global method to create quantizer and call quantizer dequantize_APoT
"""
def dequantize_APoT(float2apot, qparams: Tuple):
    quantizer = APoTQuantizer(alpha=qparams[0], gamma=qparams[1], quantization_levels=qparams[2], level_indices=qparams[3])
    return quantizer.dequantize_APoT(float2apot)
