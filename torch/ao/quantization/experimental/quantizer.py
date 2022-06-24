import torch
from torch import Tensor
from torch.ao.quantization.experimental.apot_utils import float_to_apot, apot_to_float, clip

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
    def quantize(self, tensor2quantize: Tensor):
        result = torch.tensor([])

        # clip tensor2quantize values based on alpha qparam
        tensor2quantize = tensor2quantize.apply_(lambda x: clip(x, self.alpha))
        # map float_to_apot over tensor2quantize elements
        result_data = tensor2quantize.apply_(lambda x: float_to_apot(x, self.quantization_levels, self.level_indices))

        from torch.ao.quantization.experimental.APoT_tensor import TensorAPoT

        result = TensorAPoT(self, result_data)

        return result

    r""" Dequantizes integer Tensor to floating point (fp) representation
    based on the calculated quantization levels from a specified APoT non-uniform observer.
    The approach follows the method outlined in the APoT paper: https://arxiv.org/pdf/1909.13144.pdf.
    Args:
        apot_tensor: quantized APoT Tensor to dequantize
    Returns:
        result: fp representation of input Tensor
    """
    def dequantize(self, apot_tensor) -> Tensor:
        apot_tensor_data = apot_tensor.data

        # map apot_to_float over tensor2quantize elements
        result = apot_tensor_data.apply_(lambda x: float(apot_to_float(x, self.quantization_levels, self.level_indices)))

        return result

    def q_apot_alpha(self) -> float:
        raise NotImplementedError

r""" Global method to create quantizer and call quantizer quantize_APoT
"""
def quantize_APoT(tensor2quantize: Tensor, alpha: Tensor, gamma: Tensor, quantization_levels: Tensor, level_indices: Tensor):
    quantizer = APoTQuantizer(alpha=alpha, gamma=gamma, quantization_levels=quantization_levels, level_indices=level_indices)
    return quantizer.quantize(tensor2quantize)

r""" Global method to create quantizer and call quantizer dequantize_APoT
"""
def dequantize_APoT(apot_tensor, alpha: Tensor, gamma: Tensor, quantization_levels: Tensor, level_indices: Tensor) -> Tensor:
    quantizer = APoTQuantizer(alpha=alpha, gamma=gamma, quantization_levels=quantization_levels, level_indices=level_indices)
    return apot_tensor.quantizer.dequantize(apot_tensor)
