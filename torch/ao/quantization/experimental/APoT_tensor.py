import torch
from torch.ao.quantization.experimental.quantizer import APoTQuantizer

# class to store APoT quantized tensor
class TensorAPoT():
    quantizer: APoTQuantizer
    data: torch.Tensor

    def __init__(self, quantizer):
        self.quantizer = quantizer
        self.data = quantizer.data

    def int_repr(self):
        return self.quantizer.data
