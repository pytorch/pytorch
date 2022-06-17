import torch
import enum
from torch import Tensor
from torch.ao.quantization.experimental.observer import APoTObserver, float_to_apot, float_to_reduced_precision

# enum to represent APoT representation
class APoTRepr(enum.Enum):
    level_indices = 1
    reduced_precision_fp = 2

# class to store APoT quantized tensor
class TensorAPoT(torch.Tensor):
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
        apot_repr: enum flag to specify whether to return an integer or reduced fp APoT representation of tensor
    Returns:
        result: APoT representation of tensor2quantize (integer or reduced precision fp)
    """
    def quantize_APoT(self, tensor2quantize: Tensor, apot_repr: APoTRepr):
        if apot_repr == APoTRepr.level_indices:
            # map float_to_apot over tensor2quantize elements
            self.data = tensor2quantize.apply_(lambda x: float_to_apot(x, self.quantization_levels, self.level_indices))
        elif apot_repr == APoTRepr.reduced_precision_fp:
            self.data = tensor2quantize.apply_(lambda x:
                                               float_to_reduced_precision(x, self.quantization_levels, self.level_indices))
        return self

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

        # by paper defn, max value of floating point tensor will be 1.0
        max_val = 1.0

        # make observer
        obs = APoTObserver(max_val=max_val, b=b, k=k)
        obs_result = obs.calculate_qparams(signed=False)

        quantized_levels = obs_result[1]
        level_indices = obs_result[2]

        print("quantized levels", quantized_levels)
        print("level indices", level_indices)

        # map apot_to_float over tensor2quantize elements
        result = tensor2dequantize.apply_(lambda x: float(apot_to_float(x, quantized_levels, level_indices)))

        return result

    def q_apot_alpha(self) -> float:
        raise NotImplementedError
