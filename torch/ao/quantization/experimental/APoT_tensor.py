import torch
from torch.ao.quantization.experimental.quantizer import APoTQuantizer
from torch.ao.quantization.experimental.apot_utils import float_to_apot

# class to store APoT quantized tensor
class TensorAPoT(torch.Tensor):
    quantizer: APoTQuantizer
    data: torch.Tensor
    use_int_repr: bool

    def __init__(self, quantizer: APoTQuantizer):
        self.quantizer = quantizer
        self.data = quantizer.data
        self.use_int_repr = quantizer.use_int_repr

    def int_repr(self):
        if self.use_int_repr:
            return self.quantizer.data
        else:
            data_copy = torch.clone(self.data)
            # convert to int representation
            quantizer_int_repr = data_copy.apply_(lambda x:
                                                  float_to_apot(x,
                                                                self.quantizer.quantization_levels,
                                                                self.quantizer.level_indices))

            return quantizer_int_repr
