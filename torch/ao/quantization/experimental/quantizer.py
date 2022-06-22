from torch import Tensor

# class to store APoT quantizer
# implements quantize and dequantize
# and stores all quantization parameters
class APoTQuantizer():
    @staticmethod
    def quantize_APoT(tensor2quantize: Tensor) -> Tensor:
        raise NotImplementedError

    def dequantize(self) -> Tensor:
        raise NotImplementedError

    def q_apot_alpha(self) -> float:
        raise NotImplementedError
