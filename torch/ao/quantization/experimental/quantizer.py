import torch
from torch import Tensor
from torch.ao.quantization.experimental.observer import APoTObserver
from torch.ao.quantization.experimental.apot_utils import float_to_apot, apot_to_float

# class to store APoT quantizer
# implements quantize and dequantize
# and stores all quantization parameters
class APoTQuantizer():
    b: int
    k: int
    n: int
    signed: bool
    quantization_levels: torch.Tensor
    level_indices: torch.Tensor
    data: torch.Tensor

    def __init__(
        self,
        b,
        k,
        max_val,
        signed,
            dtype=torch.quint8) -> None:
        self.signed = signed
        self.use_int_repr = True

        # check for valid inputs of b, k
        assert(k and k != 0)
        assert(b % k == 0)
        self.b = b
        self.k = k
        self.n = b // k

        # make observer, get quantizion levels and level indices
        obs = APoTObserver(max_val=max_val, b=b, k=k)
        obs_result = obs.calculate_qparams(signed=signed)
        self.quantization_levels = obs_result[1]
        self.level_indices = obs_result[2]

    r""" Quantizes fp Tensor to integer APoT representatio.
    Conversion is based on the calculated quantization levels from a specified APoT non-uniform observer.
    The approach follows the method outlined in the APoT paper: https://arxiv.org/pdf/1909.13144.pdf.
    Args:
        tensor2quantize: fp Tensor
    Returns:
        result: integer APoT representation of tensor2quantize
    """
    def quantize_APoT(self, tensor2quantize: Tensor):
        result = torch.tensor([])
        # map float_to_apot over tensor2quantize elements
        result = tensor2quantize.apply_(lambda x: float_to_apot(x, self.quantization_levels, self.level_indices))

        self.data = result

        return result

    r""" Dequantizes integer Tensor to floating point representation
    based on the calculated quantization levels from a specified APoT non-uniform observer.
    The approach follows the method outlined in the APoT paper: https://arxiv.org/pdf/1909.13144.pdf.
    Args:
        self: APoTQuantizer with attr data to dequantize
    Returns:
        result: floating point representation of input Tensor
    """
    def dequantize(self):  # type: ignore[override]
        tensor2dequantize = self.data.float()

        quantization_levels = self.quantization_levels
        level_indices = self.level_indices

        # map apot_to_float over tensor2quantize elements
        result = tensor2dequantize.apply_(lambda x: float(apot_to_float(x, quantization_levels, level_indices)))

        return result

    def q_apot_alpha(self) -> float:
        raise NotImplementedError
