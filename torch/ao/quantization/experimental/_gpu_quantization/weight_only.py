import torch
from torch.ao.quantization.experimental._gpu_quantization.quant_primitives import (
    dynamically_quantize_per_channel,
)

__all__ = ["WeightOnlyInt8QuantLinear"]


class WeightOnlyInt8QuantLinear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        w_int8 = kwargs.pop("w_int8")
        scales = kwargs.pop("scales")
        super().__init__(*args, **kwargs)
        self.w_int8 = w_int8
        self.scales = scales

    def forward(self, x):
        # if len(x.shape)<=2:
        #     y = torch.mm(x, self.w_int8.to(x.dtype)) * self.scales
        # else: # turn x into 2d tensor, then undo it for y
        x_view = x.view(-1, x.shape[-1])
        y = torch.mm(x_view, self.w_int8.to(x.dtype)) * self.scales
        y = y.reshape(*x.shape[:-1], -1)
        if self.bias is not None:
            y += self.bias
        return y

    @classmethod
    def from_float(cls, mod):
        w_fp32 = mod.weight
        w_int8, scales, _zp = dynamically_quantize_per_channel(
            w_fp32, -128, 127, torch.int8
        )
        # create the new module with a toy size to ensure initialization is fast
        fake_in_features, fake_out_features = 8, 8
        new_mod = cls(
            fake_in_features,
            fake_out_features,
            bias=mod.bias is not None,
            w_int8=w_int8.t().contiguous(),
            scales=scales,
        )
        new_mod.in_features = mod.in_features
        new_mod.out_features = mod.out_features
        del new_mod.weight
        new_mod.bias = mod.bias
        device_to_use = next(mod.parameters()).device
        new_mod.to(device_to_use)
        return new_mod
