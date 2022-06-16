import torch
from torch import Tensor
from torch.ao.quantization.experimental.observer import APoTObserver, float_to_apot

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

    """ Quantizes fp Tensor to integer or reduced precision fp representation, depending on user input.
    Conversion is based on the calculated quantization levels from a specified APoT non-uniform observer.
    The approach follows the method outlined in the APoT paper: https://arxiv.org/pdf/1909.13144.pdf.
    Args:
        tensor2quantize: fp Tensor
        b: total number of bits across all terms in non-uniform observer
        k: base bitwidth, i.e. bitwidth of every term, in non-uniform observer
        signed: boolean value indicating whether quantization levels include signed (negative) values
    Returns:
        result: APoT representation of tensor2quantize (integer or reduced precision fp)
    """
    def quantize_APoT(self, tensor2quantize: Tensor):
        # map float_to_apot over tensor2quantize elements
        self.data = tensor2quantize.apply_(lambda x: float_to_apot(x, self.quantization_levels, self.level_indices))

        return self

    @staticmethod
    def dequantize(self) -> Tensor:  # type: ignore[override]
        raise NotImplementedError

    def q_apot_alpha(self) -> float:
        raise NotImplementedError
