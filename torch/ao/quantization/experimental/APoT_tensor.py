import torch
from torch.ao.quantization.experimental.quantizer import APoTQuantizer

# class to store APoT quantized tensor
class TensorAPoT(torch.Tensor):
    quantizer: APoTQuantizer

    def __init__(self, quantizer):
        raise NotImplementedError

    def int_repr(self):
        raise NotImplementedError
