import torch
from torch import Tensor
from torch.ao.quantization.experimental.observer import APoTObserver

# class to store APoT quantized tensor
class TensorAPoT(torch.Tensor):
    @staticmethod
    def quantize_APoT(tensor2quantize: Tensor) -> Tensor:
        # traverse tensor2quantize, quantize each element
        for x in tensor2quantize:


    def dequantize(self) -> Tensor:
        raise NotImplementedError

    def q_apot_alpha(self) -> float:
        raise NotImplementedError
