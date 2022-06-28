import torch
from torch.ao.quantization.experimental.quantizer import APoTQuantizer

# class to store APoT quantized tensor
class TensorAPoT():
    quantizer: APoTQuantizer
    data: torch.Tensor

    def __init__(self, quantizer: APoTQuantizer, tensor2quantize: torch.Tensor):
        self.quantizer = quantizer
        self.data = quantizer.quantize_APoT(tensor2quantize)

    def int_repr(self):
        return self.data
