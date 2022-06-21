import torch
from torch.ao.quantization.experimental.quantizer import APoTQuantizer

# class to store APoT quantized tensor
class TensorAPoT(torch.Tensor):
    quantizer: APoTQuantizer
    data: torch.Tensor
    dtype: torch.dtype

    def __init__(self):
        raise NotImplementedError

    def int_repr(self):
        raise NotImplementedError
