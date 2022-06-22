import torch
from torch.ao.quantization.experimental.quantizer import APoTQuantizer

# class to store APoT quantized tensor
class TensorAPoT():
    quantizer: APoTQuantizer

    def __init__(self, quantizer):
        self.quantizer = quantizer

    def int_repr(self, tensor: torch.Tensor):
        return self.quantizer.quantize_APoT(tensor)
