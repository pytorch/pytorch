import torch
from torch import Tensor
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
        quantization_levels: Tensor from APoT calculated qparams
        level_indices: Tensor from APoT calculated qparams
    Returns:
        result: integer APoT representation of tensor2quantize
    """
    @staticmethod
    def quantize_APoT(tensor2quantize: Tensor, quantization_levels: Tensor, level_indices: Tensor):
        result = torch.tensor([])
        # map float_to_apot over tensor2quantize elements
        result = tensor2quantize.apply_(lambda x: float_to_apot(x, quantization_levels, level_indices))

        return result

    r""" Dequantizes integer Tensor to floating point (fp) representation
    based on the calculated quantization levels from a specified APoT non-uniform observer.
    The approach follows the method outlined in the APoT paper: https://arxiv.org/pdf/1909.13144.pdf.
    Args:
        float2apot: quantized APoT Tensor to dequantize
    Returns:
        result: fp representation of input Tensor
    """
    @staticmethod
    def dequantize_APoT(float2apot):
        quantization_levels = float2apot.quantizer.quantization_levels
        level_indices = float2apot.quantizer.level_indices
        apot_tensor = float2apot.data

        # map apot_to_float over tensor2quantize elements
        result = apot_tensor.apply_(lambda x: float(apot_to_float(x, quantization_levels, level_indices)))

        return result

    def q_apot_alpha(self) -> float:
        raise NotImplementedError

def quantize_APoT(tensor2quantize: Tensor, quantization_levels: Tensor, level_indices: Tensor):
    return APoTQuantizer.quantize_APoT(tensor2quantize, quantization_levels, level_indices)

def dequantize_APoT(float2apot):
    return APoTQuantizer.dequantize_APoT(float2apot)
