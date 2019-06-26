from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .. import functional as F
from ...modules.module import Module
from ...._jit_internal import weak_module, weak_script_method

@weak_module
class ReLU(Module):
    r"""
    We used same interface as `torch.nn.ReLU` please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.ReLU for documentation

    Only difference is we are calling quantized ReLU implementation and right
    now we do not support inplace.
    """


    def __init__(self, inplace=False):
        super(ReLU, self).__init__(inplace)
        assert not inplace, 'torch.nn.quantized.ReLU does not support inplace'


    @weak_script_method
    def forward(self, input):
        return F.relu(input)

    @staticmethod
    def from_float(mod):
        return ReLU(mod.inplace)
