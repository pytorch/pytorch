import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.intrinsic import LinearReLU
from torch.nn.utils.parametrize import (
    is_parametrized,
    type_before_parametrizations,
    transfer_parametrizations_and_params,
)

class Linear(nn.Linear):
    r"""
    A linear module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.Linear`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
    for documentation.

    Similar to `torch.nn.Linear`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight: fake quant module for weight
    """
    _FLOAT_MODULE = nn.Linear

    def __init__(self, in_features, out_features, bias=True,
                 qconfig=None, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(in_features, out_features, bias, **factory_kwargs)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)

    def forward(self, input):
        return F.linear(input, self.weight_fake_quant(self.weight), self.bias)

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict
            Args: `mod` a float module, either produced by torch.ao.quantization utilities
            or directly from user
        """
        assert type_before_parametrizations(mod) == cls._FLOAT_MODULE, (
            " qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"
        if type_before_parametrizations(mod) == LinearReLU:
            mod = mod[0]

        qconfig = mod.qconfig
        qat_linear = cls(mod.in_features, mod.out_features, bias=mod.bias is not None, qconfig=qconfig)

        if is_parametrized(mod, "weight"):
            transfer_parametrizations_and_params(mod, qat_linear, "weight")
        else:
            qat_linear.weight = mod.weight

        if is_parametrized(mod, "bias"):
            transfer_parametrizations_and_params(mod, qat_linear, "bias")
        else:
            qat_linear.bias = mod.bias

        return qat_linear

    def to_float(self):
        linear = torch.nn.Linear(self.in_features, self.out_features, self.bias is not None)
        linear.weight = torch.nn.Parameter(self.weight.detach())
        if self.bias is not None:
            linear.bias = torch.nn.Parameter(self.bias.detach())
        linear.train(self.training)
        return linear
