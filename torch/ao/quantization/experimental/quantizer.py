import torch
from torch import Tensor
from torch.ao.quantization.experimental.apot_utils import float_to_apot, apot_to_float

# class to store APoT quantizer
# implements quantize and dequantize
# and stores APoT observer
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
    Conversion is based on the calculated quantization levels from a specified APoT non-uniform observer.
    The approach follows the method outlined in the APoT paper: https://arxiv.org/pdf/1909.13144.pdf.
    Args:
        tensor2quantize: fp Tensor
        signed: bool to indicate whether to include signed quantization levels
        min_val: optional arg to override min value in tensor2quantize
        max_val: optional arg to override max value in tensor2quantize
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
        signed: bool to indicate whether to include signed quantization levels
        min_val: minimum value of fp dequantized Tensor
        max_val: maximum value of fp dequantized Tensor
    Returns:
        result: fp representation of input Tensor
    """
    def dequantize(self, float2apot: Tensor):  # type: ignore[override]
        float2apot = float2apot.float()

        # map apot_to_float over tensor2quantize elements
        result = float2apot.apply_(lambda x: float(apot_to_float(x, self.quantization_levels, self.level_indices)))

        return result

    def q_apot_alpha(self) -> float:
        raise NotImplementedError
