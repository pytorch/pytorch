import torch
from torch.ao.quantization.experimental.quantizer import APoTQuantizer

# class to store APoT quantized tensor
class TensorAPoT():
    quantizer: APoTQuantizer
    data: torch.Tensor

    def __init__(self, quantizer: APoTQuantizer, tensor2quantize: torch.Tensor):
        self.quantizer = quantizer
        self.data = APoTQuantizer.quantize_APoT(tensor2quantize=tensor2quantize, quantization_levels=quantizer.quantization_levels, level_indices=quantizer.level_indices)

    def int_repr(self):
        return self.data
