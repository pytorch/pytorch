import torch
from torch.ao.quantization.experimental.quantizer import APoTQuantizer

# class to store APoT quantized tensor
class TensorAPoT(torch.Tensor):
    quantizer: APoTQuantizer
    data: torch.Tensor
    data_int_repr: torch.Tensor
    dtype: torch.dtype

    def __init__(self, quantizer):
        self.quantizer = quantizer
        self.data = quantizer.data
        self.dtype = torch.quint8

    def int_repr(self):
