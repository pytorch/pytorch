import torch
from torch import Tensor
from torch.ao.quantization.experimental.observer import APoTObserver, apot_to_float

# class to store APoT quantized tensor
class TensorAPoT(torch.Tensor):
    @staticmethod
    def quantize_APoT(tensor2quantize: Tensor) -> Tensor:
        raise NotImplementedError

    """ Dequantizes integer Tensor to floating point representation
    based on the calculated quantization levels from a specified APoT non-uniform observer.
    The approach follows the method outlined in the APoT paper: https://arxiv.org/pdf/1909.13144.pdf.
    Args:
        tensor2dequantize: integer Tensor
        b: total number of bits across all terms in non-uniform observer
        k: base bitwidth, i.e. bitwidth of every term, in non-uniform observer
    Returns:
        result: floating point representation of input Tensor
    """
    @staticmethod
    def dequantize(tensor2dequantize: Tensor, b: int, k: int) -> Tensor:  # type: ignore[override]
        tensor2dequantize = tensor2dequantize.float()

        max_val = torch.max(tensor2dequantize)

        # make observer
        obs = APoTObserver(max_val=max_val, b=b, k=k)
        obs_result = obs.calculate_qparams(signed=False)

        quantized_levels = obs_result[1]
        level_indices = obs_result[2]

        # map apot_to_float over tensor2quantize elements
        result = tensor2dequantize.apply_(lambda x: float(apot_to_float(x, quantized_levels, level_indices)))

        return result

    def q_apot_alpha(self) -> float:
        raise NotImplementedError
