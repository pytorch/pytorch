import torch
from torch import Tensor
from torch.ao.quantization.experimental.observer import APoTObserver
from torch.ao.quantization.experimental.apot_utils import float_to_apot, float_to_reduced_precision, apot_to_float

# class to store APoT quantizer
# implements quantize and dequantize
# and stores all quantization parameters
class APoTQuantizer(torch.Tensor):
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
        obs = APoTObserver(max_val=1.0, b=b, k=k)
        obs_result = obs.calculate_qparams(signed=signed)
        self.quantization_levels = obs_result[1]
        self.level_indices = obs_result[2]

    r""" Quantizes fp Tensor to integer or reduced precision fp representation, depending on user input.
    Conversion is based on the calculated quantization levels from a specified APoT non-uniform observer.
    The approach follows the method outlined in the APoT paper: https://arxiv.org/pdf/1909.13144.pdf.
    Args:
        tensor2quantize: fp Tensor
        use_int_repr: bool flag to specify int of reduced precision fp representation of APoT tensor
    Returns:
        result: APoT representation of tensor2quantize (integer or reduced precision fp)
    """
    def quantize_APoT(self, tensor2quantize: Tensor, use_int_repr: bool):
        self.use_int_repr = use_int_repr
        if use_int_repr:
            # map float_to_apot over tensor2quantize elements
            self.data = tensor2quantize.apply_(lambda x: float_to_apot(x, self.quantization_levels, self.level_indices))
        else:
            self.data = tensor2quantize.apply_(lambda x:
                                               float_to_reduced_precision(x, self.quantization_levels, self.level_indices))

        return self

    r""" Dequantizes integer Tensor to floating point representation
    based on the calculated quantization levels from a specified APoT non-uniform observer.
    The approach follows the method outlined in the APoT paper: https://arxiv.org/pdf/1909.13144.pdf.
    Args:
        self: APoT tensor to dequantize
    Returns:
        result: floating point representation of input Tensor
    """
    def dequantize(self):  # type: ignore[override]
        if self.use_int_repr:
            tensor2dequantize = self.data.float()

            max_val = 1.0

            quantization_levels = self.quantization_levels
            level_indices = self.level_indices

            # map apot_to_float over tensor2quantize elements
            result = tensor2dequantize.apply_(lambda x: float(apot_to_float(x, quantization_levels, level_indices)))

            return result
        else:
            return self.data

    def q_apot_alpha(self) -> float:
        raise NotImplementedError
