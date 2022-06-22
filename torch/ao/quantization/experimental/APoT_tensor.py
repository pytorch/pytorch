import torch
from torch.ao.quantization.experimental.quantizer import APoTQuantizer

# test1.3

# class to store APoT quantized tensor
class TensorAPoT(torch.Tensor):
    quantizer: APoTQuantizer

    def __init__(self, quantizer):
        raise NotImplementedError

    def int_repr(self):
        raise NotImplementedError

# test1.1 test1.1
