import torch
from torch.ao.quantization.experimental.quantizer import APoTQuantizer
from torch.ao.quantization.experimental.apot_utils import float_to_apot

# class to store APoT quantized tensor
class TensorAPoT(torch.Tensor):
    quantizer: APoTQuantizer
    data: torch.Tensor

    def __init__(self, quantizer: APoTQuantizer):
        self.quantizer = quantizer
        self.data = quantizer.data

    def int_repr(self):
        return self.quantizer.data
