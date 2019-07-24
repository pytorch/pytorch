from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch

from .quantize import QuantStub
from .quantize import DeQuantStub

# Map for swapping float module to quantized ones.
_DEFAULT_MODULE_MAPPING = {
    torch.nn.Linear: torch.nn.quantized.Linear,
    torch.nn.ReLU: torch.nn.quantized.ReLU,
    torch.nn.Conv2d: torch.nn.quantized.Conv2d,
    QuantStub: torch.nn.quantized.Quantize,
    DeQuantStub: nntorch.nn.quantizedq.DeQuantize,
    # QAT modules:
    torch.nn.qat.Linear: torch.nn.quantized.Linear,
    torch.nn.qat.Conv2d: torch.nn.quantized.Conv2d,
}

# Map for swapping float module to qat modules.
_DEFAULT_QAT_MODULE_MAPPING = {
    torch.nn.Linear: torch.nn.qat.Linear,
    torch.nn.Conv2d: torch.nn.qat.Conv2d,
}
