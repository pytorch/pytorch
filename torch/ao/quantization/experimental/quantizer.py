import torch
from torch import Tensor
from torch.ao.quantization.experimental.observer import APoTObserver
from torch.ao.quantization.experimental.apot_utils import float_to_apot, apot_to_float

# class to store APoT quantizer
# implements quantize and dequantize
# and stores all quantization parameters
class APoTQuantizer():
    observer: APoTObserver

    def __init__(
        self,
            observer: APoTObserver) -> None:
        self.observer = observer

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
    def quantize_APoT(self, tensor2quantize: Tensor, signed: bool, min_val=None, max_val=None):
        if min_val is None:
            min_val = torch.min(tensor2quantize)
        if max_val is None:
            max_val = torch.max(tensor2quantize)

        # get qparams
        qparams = self.observer.calculate_qparams(signed=signed, min_val=min_val, max_val=max_val)
        quantization_levels = qparams[1]
        level_indices = qparams[2]

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
    def dequantize(self, float2apot: Tensor, signed: bool, min_val: Tensor, max_val: Tensor):  # type: ignore[override]
        float2apot = float2apot.float()

        # get qparams
        qparams = self.observer.calculate_qparams(signed=signed, min_val=min_val, max_val=max_val)
        quantization_levels = qparams[1]
        level_indices = qparams[2]

        # map apot_to_float over tensor2quantize elements
        result = float2apot.apply_(lambda x: float(apot_to_float(x, quantization_levels, level_indices)))

        return result

    def q_apot_alpha(self) -> float:
        raise NotImplementedError
