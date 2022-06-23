import torch
from torch.ao.quantization.experimental.quantizer import APoTQuantizer

# class to store APoT quantized tensor
class TensorAPoT():
    quantizer: APoTQuantizer
    data: torch.Tensor

    def __init__(self, quantizer: APoTQuantizer, tensor2quantize: torch.Tensor, signed=False):
        self.quantizer = quantizer
        self.data = quantizer.quantize_APoT(tensor2quantize=tensor2quantize, signed=signed)

    def int_repr(self):
        return self.data
