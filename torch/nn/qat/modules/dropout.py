import torch
import torch.nn as nn

class Dropout(nn.Dropout):
    r"""
    A dropout module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.Dropout`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
    for documentation.

    Similar to `torch.nn.dropout`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight: fake quant module for weight
    """
    _FLOAT_MODULE = nn.Dropout

    def __init__(self, p=0.5, inplace=False,
                 qconfig=None) -> None:
        super().__init__(p, inplace)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig

    def forward(self, input):
        return input

    def _get_name(self):
        return 'QuantizedDropout'

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.ao.quantization utilities
            or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, ' qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'

        qconfig = mod.qconfig
        qat_dropout = cls(mod.p, inplace=mod.inplace, qconfig=qconfig)
        return qat_dropout

    def to_float(self):
        dropout = torch.nn.dropout(p=self.p, inplace=self.inplace)
        return dropout
