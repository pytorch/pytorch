import torch
from torch.ao.quantization.experimental.quantizer import APoTQuantizer

# class to store APoT quantized tensor
class TensorAPoT(torch.Tensor):
    quantizer: APoTQuantizer
    data_reduced_precision_repr: torch.Tensor
    data_int_repr: torch.Tensor

    def __init__(self, quantizer):
        raise NotImplementedError

    def int_repr(self):
        raise NotImplementedError
